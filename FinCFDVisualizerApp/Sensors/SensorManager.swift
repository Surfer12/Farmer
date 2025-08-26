import CoreMotion
import CoreBluetooth
import CoreLocation
import Combine
import Foundation

// MARK: - Sensor Manager

class SensorManager: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    // Motion Manager
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    
    // Bluetooth Manager
    private var centralManager: CBCentralManager!
    private var pressureSensorPeripheral: CBPeripheral?
    private var pressureSensorCharacteristic: CBCharacteristic?
    
    // Location Manager
    private let locationManager = CLLocationManager()
    
    // Published Properties
    @Published var imuData: IMUData = IMUData.zero
    @Published var pressureData: [Double] = []
    @Published var environmentalData: EnvironmentalData = EnvironmentalData.default
    @Published var turnAngle: Double = 0.0
    @Published var isIMUActive = false
    @Published var isPressureSensorConnected = false
    @Published var sensorStatus: SensorStatus = .disconnected
    
    // Data Collection
    private var sensorDataBuffer: [SensorData] = []
    private let maxBufferSize = 1000
    private var cancellables = Set<AnyCancellable>()
    
    // Calibration
    private var imuCalibrationOffset: IMUData = IMUData.zero
    private var isCalibrated = false
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupMotionManager()
        setupBluetoothManager()
        setupLocationManager()
        startDataCollection()
    }
    
    // MARK: - Motion Management
    
    private func setupMotionManager() {
        motionManager.deviceMotionUpdateInterval = 0.1 // 10 Hz
        motionManager.accelerometerUpdateInterval = 0.05 // 20 Hz
        motionManager.gyroUpdateInterval = 0.05 // 20 Hz
        motionManager.magnetometerUpdateInterval = 0.1 // 10 Hz
    }
    
    func startIMUMonitoring() {
        guard motionManager.isDeviceMotionAvailable else {
            print("❌ Device motion not available")
            return
        }
        
        // Start device motion updates
        motionManager.startDeviceMotionUpdates(
            using: .xMagneticNorthZVertical,
            to: .main
        ) { [weak self] motion, error in
            if let error = error {
                print("❌ Motion update error: \(error)")
                return
            }
            
            guard let motion = motion else { return }
            self?.processMotionData(motion)
        }
        
        // Start accelerometer updates
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
            if let error = error {
                print("❌ Accelerometer error: \(error)")
                return
            }
            
            guard let data = data else { return }
            self?.processAccelerometerData(data)
        }
        
        isIMUActive = true
        print("✅ IMU monitoring started")
    }
    
    func stopIMUMonitoring() {
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        motionManager.stopMagnetometerUpdates()
        
        isIMUActive = false
        print("🛑 IMU monitoring stopped")
    }
    
    private func processMotionData(_ motion: CMDeviceMotion) {
        let attitude = motion.attitude
        let rotationRate = motion.rotationRate
        let userAcceleration = motion.userAcceleration
        let gravity = motion.gravity
        
        // Calculate turn angle from yaw
        let turnAngleRad = attitude.yaw
        let turnAngleDeg = turnAngleRad * 180.0 / .pi
        
        // Update IMU data
        let newIMUData = IMUData(
            acceleration: Vector3D(
                x: userAcceleration.x + gravity.x,
                y: userAcceleration.y + gravity.y,
                z: userAcceleration.z + gravity.z
            ),
            rotation: Vector3D(
                x: rotationRate.x,
                y: rotationRate.y,
                z: rotationRate.z
            ),
            attitude: Attitude(
                pitch: attitude.pitch,
                roll: attitude.roll,
                yaw: attitude.yaw
            ),
            turnAngle: turnAngleDeg
        )
        
        // Apply calibration offset if available
        let calibratedData = isCalibrated ? applyCalibration(to: newIMUData) : newIMUData
        
        DispatchQueue.main.async { [weak self] in
            self?.imuData = calibratedData
            self?.turnAngle = max(0, min(20, abs(calibratedData.turnAngle))) // Clamp to 0-20°
        }
    }
    
    private func processAccelerometerData(_ data: CMAccelerometerData) {
        // Additional accelerometer processing if needed
        // This can be used for more precise motion detection
    }
    
    // MARK: - Bluetooth Management
    
    private func setupBluetoothManager() {
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    func startPressureSensorScan() {
        guard centralManager.state == .poweredOn else {
            print("❌ Bluetooth not powered on")
            return
        }
        
        // Scan for pressure sensor peripherals
        // In a real implementation, you would scan for specific service UUIDs
        centralManager.scanForPeripherals(withServices: nil, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])
        
        print("🔍 Scanning for pressure sensors...")
    }
    
    func stopPressureSensorScan() {
        centralManager.stopScan()
        print("🛑 Stopped pressure sensor scan")
    }
    
    // MARK: - Location Management
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.requestWhenInUseAuthorization()
    }
    
    func startLocationUpdates() {
        guard CLLocationManager.locationServicesEnabled() else {
            print("❌ Location services not enabled")
            return
        }
        
        locationManager.startUpdatingLocation()
        print("📍 Location updates started")
    }
    
    func stopLocationUpdates() {
        locationManager.stopUpdatingLocation()
        print("🛑 Location updates stopped")
    }
    
    // MARK: - Data Collection and Processing
    
    private func startDataCollection() {
        // Combine IMU and pressure data into sensor data buffer
        Publishers.CombineLatest3(
            $imuData,
            $pressureData,
            $environmentalData
        )
        .throttle(for: .seconds(0.1), scheduler: DispatchQueue.main, latest: true)
        .sink { [weak self] imu, pressure, environmental in
            let sensorData = SensorData(
                timestamp: Date(),
                imuData: imu,
                pressureData: pressure,
                environmentalData: environmental
            )
            
            self?.addToBuffer(sensorData)
        }
        .store(in: &cancellables)
    }
    
    private func addToBuffer(_ data: SensorData) {
        sensorDataBuffer.append(data)
        
        // Maintain buffer size
        if sensorDataBuffer.count > maxBufferSize {
            sensorDataBuffer.removeFirst()
        }
    }
    
    // MARK: - Calibration
    
    func calibrateIMU() {
        guard isIMUActive else {
            print("❌ IMU not active, cannot calibrate")
            return
        }
        
        // Take current IMU reading as offset
        imuCalibrationOffset = imuData
        isCalibrated = true
        
        print("✅ IMU calibrated with offset: \(imuCalibrationOffset)")
    }
    
    private func applyCalibration(to data: IMUData) -> IMUData {
        return IMUData(
            acceleration: Vector3D(
                x: data.acceleration.x - imuCalibrationOffset.acceleration.x,
                y: data.acceleration.y - imuCalibrationOffset.acceleration.y,
                z: data.acceleration.z - imuCalibrationOffset.acceleration.z
            ),
            rotation: Vector3D(
                x: data.rotation.x - imuCalibrationOffset.rotation.x,
                y: data.rotation.y - imuCalibrationOffset.rotation.y,
                z: data.rotation.z - imuCalibrationOffset.rotation.z
            ),
            attitude: Attitude(
                pitch: data.attitude.pitch - imuCalibrationOffset.attitude.pitch,
                roll: data.attitude.roll - imuCalibrationOffset.attitude.roll,
                yaw: data.attitude.yaw - imuCalibrationOffset.attitude.yaw
            ),
            turnAngle: data.turnAngle - imuCalibrationOffset.turnAngle
        )
    }
    
    // MARK: - Mock Data Generation (for testing)
    
    func startMockDataGeneration() {
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.generateMockPressureData()
            self?.generateMockEnvironmentalData()
        }
        
        print("🔄 Mock data generation started")
    }
    
    private func generateMockPressureData() {
        // Generate realistic pressure data based on CFD patterns
        let baseTime = Date().timeIntervalSince1970
        var mockPressure: [Double] = []
        
        for i in 0..<20 {
            let x = Double(i) / 19.0 // Normalized position (0-1)
            let timeVariation = sin(baseTime * 0.5) * 0.1
            
            // Pressure distribution pattern (higher at leading edge, lower at trailing)
            var pressure = 1.0 - (x * x) + timeVariation
            
            // Add some noise
            pressure += Double.random(in: -0.1...0.1)
            
            // Normalize to 0-1 range
            pressure = max(0, min(1, pressure))
            mockPressure.append(pressure)
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.pressureData = mockPressure
        }
    }
    
    private func generateMockEnvironmentalData() {
        let newEnvironmentalData = EnvironmentalData(
            waterTemperature: 20.0 + Double.random(in: -2...2),
            pressure: 101.3 + Double.random(in: -0.5...0.5),
            salinity: 35.0 + Double.random(in: -1...1)
        )
        
        DispatchQueue.main.async { [weak self] in
            self?.environmentalData = newEnvironmentalData
        }
    }
    
    // MARK: - Data Export and Analysis
    
    func exportSensorData() -> Data? {
        do {
            return try JSONEncoder().encode(sensorDataBuffer)
        } catch {
            print("❌ Failed to export sensor data: \(error)")
            return nil
        }
    }
    
    func getSensorStatistics() -> SensorStatistics {
        guard !sensorDataBuffer.isEmpty else {
            return SensorStatistics.empty
        }
        
        let imuReadings = sensorDataBuffer.map { $0.imuData }
        let pressureReadings = sensorDataBuffer.flatMap { $0.pressureData }
        
        return SensorStatistics(
            totalReadings: sensorDataBuffer.count,
            averageTurnAngle: imuReadings.map { $0.turnAngle }.reduce(0, +) / Double(imuReadings.count),
            maxTurnAngle: imuReadings.map { $0.turnAngle }.max() ?? 0,
            averagePressure: pressureReadings.reduce(0, +) / Double(max(pressureReadings.count, 1)),
            dataCollectionDuration: sensorDataBuffer.last?.timestamp.timeIntervalSince(sensorDataBuffer.first?.timestamp ?? Date()) ?? 0
        )
    }
    
    func clearBuffer() {
        sensorDataBuffer.removeAll()
        print("🗑️ Sensor data buffer cleared")
    }
    
    // MARK: - Utility Methods
    
    func startAllSensors() {
        startIMUMonitoring()
        startPressureSensorScan()
        startLocationUpdates()
        startMockDataGeneration() // For testing
        
        sensorStatus = .connected
    }
    
    func stopAllSensors() {
        stopIMUMonitoring()
        stopPressureSensorScan()
        stopLocationUpdates()
        
        sensorStatus = .disconnected
    }
}

// MARK: - CBCentralManagerDelegate

extension SensorManager: CBCentralManagerDelegate {
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("✅ Bluetooth powered on")
            sensorStatus = .scanning
            
        case .poweredOff:
            print("❌ Bluetooth powered off")
            sensorStatus = .disconnected
            
        case .resetting:
            print("🔄 Bluetooth resetting")
            sensorStatus = .disconnected
            
        case .unauthorized:
            print("❌ Bluetooth unauthorized")
            sensorStatus = .error
            
        case .unsupported:
            print("❌ Bluetooth unsupported")
            sensorStatus = .error
            
        case .unknown:
            print("❓ Bluetooth state unknown")
            sensorStatus = .disconnected
            
        @unknown default:
            print("❓ Unknown bluetooth state")
            sensorStatus = .disconnected
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
        
        // Look for pressure sensor characteristics in advertisement data
        if let localName = advertisementData[CBAdvertisementDataLocalNameKey] as? String,
           localName.lowercased().contains("pressure") || localName.lowercased().contains("fin") {
            
            print("🔍 Found potential pressure sensor: \(localName)")
            
            // Connect to the peripheral
            pressureSensorPeripheral = peripheral
            peripheral.delegate = self
            central.connect(peripheral, options: nil)
            central.stopScan()
        }
        
        // For demo purposes, simulate finding a pressure sensor after a few seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
            if self?.pressureSensorPeripheral == nil {
                print("🔄 Simulating pressure sensor connection")
                self?.isPressureSensorConnected = true
                self?.sensorStatus = .connected
                central.stopScan()
            }
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("✅ Connected to pressure sensor: \(peripheral.name ?? "Unknown")")
        
        isPressureSensorConnected = true
        sensorStatus = .connected
        
        // Discover services
        peripheral.discoverServices(nil)
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("❌ Failed to connect to pressure sensor: \(error?.localizedDescription ?? "Unknown error")")
        
        isPressureSensorConnected = false
        sensorStatus = .error
    }
    
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("🔌 Disconnected from pressure sensor")
        
        isPressureSensorConnected = false
        sensorStatus = .disconnected
        
        // Attempt to reconnect
        if let sensor = pressureSensorPeripheral {
            central.connect(sensor, options: nil)
        }
    }
}

// MARK: - CBPeripheralDelegate

extension SensorManager: CBPeripheralDelegate {
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        if let error = error {
            print("❌ Service discovery error: \(error)")
            return
        }
        
        guard let services = peripheral.services else { return }
        
        for service in services {
            print("🔍 Discovered service: \(service.uuid)")
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        if let error = error {
            print("❌ Characteristic discovery error: \(error)")
            return
        }
        
        guard let characteristics = service.characteristics else { return }
        
        for characteristic in characteristics {
            print("🔍 Discovered characteristic: \(characteristic.uuid)")
            
            // Subscribe to pressure data notifications
            if characteristic.properties.contains(.notify) {
                pressureSensorCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
                print("✅ Subscribed to pressure sensor notifications")
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("❌ Characteristic update error: \(error)")
            return
        }
        
        guard let data = characteristic.value else { return }
        
        // Parse pressure data (format depends on sensor specification)
        let pressureValues = parsePressureData(data)
        
        DispatchQueue.main.async { [weak self] in
            self?.pressureData = pressureValues
        }
    }
    
    private func parsePressureData(_ data: Data) -> [Double] {
        // Mock pressure data parsing
        // In a real implementation, this would parse the actual sensor data format
        var pressureValues: [Double] = []
        
        let byteArray = [UInt8](data)
        for i in stride(from: 0, to: byteArray.count, by: 2) {
            if i + 1 < byteArray.count {
                let value = Double(UInt16(byteArray[i]) | (UInt16(byteArray[i + 1]) << 8)) / 65535.0
                pressureValues.append(value)
            }
        }
        
        return pressureValues
    }
}

// MARK: - CLLocationManagerDelegate

extension SensorManager: CLLocationManagerDelegate {
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        
        // Update environmental data with location information
        let newEnvironmentalData = EnvironmentalData(
            waterTemperature: environmentalData.waterTemperature,
            pressure: location.altitude < 0 ? 101.3 + abs(location.altitude) * 0.01 : 101.3, // Approximate underwater pressure
            salinity: environmentalData.salinity
        )
        
        DispatchQueue.main.async { [weak self] in
            self?.environmentalData = newEnvironmentalData
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("❌ Location error: \(error)")
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            startLocationUpdates()
        case .denied, .restricted:
            print("❌ Location access denied")
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        @unknown default:
            break
        }
    }
}

// MARK: - Supporting Types

enum SensorStatus: String, CaseIterable {
    case disconnected = "Disconnected"
    case scanning = "Scanning"
    case connected = "Connected"
    case error = "Error"
    
    var color: UIColor {
        switch self {
        case .disconnected: return .systemRed
        case .scanning: return .systemYellow
        case .connected: return .systemGreen
        case .error: return .systemRed
        }
    }
}

struct SensorStatistics {
    let totalReadings: Int
    let averageTurnAngle: Double
    let maxTurnAngle: Double
    let averagePressure: Double
    let dataCollectionDuration: TimeInterval
    
    static let empty = SensorStatistics(
        totalReadings: 0,
        averageTurnAngle: 0,
        maxTurnAngle: 0,
        averagePressure: 0,
        dataCollectionDuration: 0
    )
}

// MARK: - Extensions for Default Values

extension IMUData {
    static let zero = IMUData(
        acceleration: Vector3D(x: 0, y: 0, z: 0),
        rotation: Vector3D(x: 0, y: 0, z: 0),
        attitude: Attitude(pitch: 0, roll: 0, yaw: 0),
        turnAngle: 0
    )
}

extension EnvironmentalData {
    static let `default` = EnvironmentalData(
        waterTemperature: 20.0,
        pressure: 101.3,
        salinity: 35.0
    )
}