import CoreMotion
import CoreBluetooth
import Combine
import Foundation

class SensorManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    
    // MARK: - Published Properties
    @Published var turnAngle: Float = 0.0
    @Published var pitchAngle: Float = 0.0
    @Published var rollAngle: Float = 0.0
    @Published var accelerationData: CMAcceleration = CMAcceleration()
    @Published var rotationRate: CMRotationRate = CMRotationRate()
    @Published var pressureData: [Float] = []
    @Published var isConnected: Bool = false
    @Published var sensorStatus: SensorStatus = .disconnected
    
    // MARK: - Core Motion
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private var deviceMotionQueue = OperationQueue()
    
    // MARK: - Bluetooth
    private var centralManager: CBCentralManager!
    private var pressureSensorPeripheral: CBPeripheral?
    private var pressureCharacteristic: CBCharacteristic?
    
    // MARK: - Sensor Configuration
    private let updateInterval: TimeInterval = 0.05 // 20Hz
    private let pressureSensorServiceUUID = CBUUID(string: "180F") // Custom pressure sensor service
    private let pressureCharacteristicUUID = CBUUID(string: "2A19") // Custom pressure characteristic
    
    // MARK: - Data Processing
    private var motionFilter = MotionFilter()
    private var pressureFilter = PressureFilter()
    private var calibrationData = CalibrationData()
    
    enum SensorStatus {
        case disconnected
        case connecting
        case connected
        case calibrating
        case ready
        case error(String)
    }
    
    override init() {
        super.init()
        setupCoreMotion()
        setupBluetooth()
        deviceMotionQueue.maxConcurrentOperationCount = 1
    }
    
    // MARK: - Core Motion Setup
    private func setupCoreMotion() {
        guard motionManager.isDeviceMotionAvailable else {
            print("❌ Device motion not available")
            return
        }
        
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.accelerometerUpdateInterval = updateInterval
        motionManager.gyroscopeUpdateInterval = updateInterval
        
        // Configure motion reference frame
        motionManager.showsDeviceMovementDisplay = true
    }
    
    // MARK: - Bluetooth Setup
    private func setupBluetooth() {
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    // MARK: - Public Interface
    func startMonitoring() {
        startCoreMotionUpdates()
        startBluetoothScanning()
        sensorStatus = .connecting
    }
    
    func stopMonitoring() {
        stopCoreMotionUpdates()
        stopBluetoothScanning()
        sensorStatus = .disconnected
    }
    
    func calibrateSensors() {
        sensorStatus = .calibrating
        calibrationData.reset()
        
        // Collect calibration data for 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            self.calibrationData.finalize()
            self.sensorStatus = .ready
            print("✅ Sensor calibration completed")
        }
    }
    
    // MARK: - Core Motion Updates
    private func startCoreMotionUpdates() {
        // Device Motion Updates (attitude, rotation, acceleration)
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(
                using: .xMagneticNorthZVertical,
                to: deviceMotionQueue
            ) { [weak self] motion, error in
                guard let self = self, let motion = motion else { return }
                
                DispatchQueue.main.async {
                    self.processDeviceMotion(motion)
                }
            }
        }
        
        // Raw Accelerometer
        if motionManager.isAccelerometerAvailable {
            motionManager.startAccelerometerUpdates(to: deviceMotionQueue) { [weak self] data, error in
                guard let self = self, let data = data else { return }
                
                DispatchQueue.main.async {
                    self.accelerationData = data.acceleration
                }
            }
        }
        
        // Raw Gyroscope
        if motionManager.isGyroAvailable {
            motionManager.startGyroUpdates(to: deviceMotionQueue) { [weak self] data, error in
                guard let self = self, let data = data else { return }
                
                DispatchQueue.main.async {
                    self.rotationRate = data.rotationRate
                }
            }
        }
        
        // Barometric Pressure (for altitude/depth sensing)
        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: deviceMotionQueue) { [weak self] data, error in
                guard let self = self, let data = data else { return }
                
                DispatchQueue.main.async {
                    // Convert pressure to depth approximation for surfing
                    let pressureKPa = data.pressure.floatValue
                    self.processBarometricPressure(pressureKPa)
                }
            }
        }
    }
    
    private func stopCoreMotionUpdates() {
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        altimeter.stopRelativeAltitudeUpdates()
    }
    
    private func processDeviceMotion(_ motion: CMDeviceMotion) {
        let attitude = motion.attitude
        
        // Apply calibration offsets
        let rawYaw = Float(attitude.yaw * 180 / .pi)
        let rawPitch = Float(attitude.pitch * 180 / .pi)
        let rawRoll = Float(attitude.roll * 180 / .pi)
        
        // Filter and calibrate angles
        turnAngle = motionFilter.filterYaw(rawYaw - calibrationData.yawOffset)
        pitchAngle = motionFilter.filterPitch(rawPitch - calibrationData.pitchOffset)
        rollAngle = motionFilter.filterRoll(rawRoll - calibrationData.rollOffset)
        
        // Update calibration if in calibration mode
        if case .calibrating = sensorStatus {
            calibrationData.addSample(yaw: rawYaw, pitch: rawPitch, roll: rawRoll)
        }
        
        // Detect surfing maneuvers
        detectManeuvers(motion: motion)
    }
    
    private func processBarometricPressure(_ pressureKPa: Float) {
        // Convert to water depth approximation (very rough)
        // 1 kPa ≈ 0.1m water depth
        let estimatedDepth = max(0, (pressureKPa - 101.325) * 0.1)
        
        // Add to pressure data array (simplified)
        if pressureData.count >= 100 {
            pressureData.removeFirst()
        }
        pressureData.append(estimatedDepth)
    }
    
    // MARK: - Maneuver Detection
    private func detectManeuvers(motion: CMDeviceMotion) {
        let rotationRate = motion.rotationRate
        let userAcceleration = motion.userAcceleration
        
        // Detect turns (yaw rate threshold)
        let yawRate = abs(Float(rotationRate.z))
        if yawRate > 1.5 { // rad/s threshold for significant turn
            NotificationCenter.default.post(name: .turnDetected, object: yawRate)
        }
        
        // Detect bottom turns (combined pitch and roll)
        let pitchRate = abs(Float(rotationRate.x))
        let rollRate = abs(Float(rotationRate.y))
        if pitchRate > 1.0 && rollRate > 0.8 {
            NotificationCenter.default.post(name: .bottomTurnDetected, object: nil)
        }
        
        // Detect aerials (vertical acceleration spike)
        let verticalAccel = Float(userAcceleration.z)
        if verticalAccel > 2.0 {
            NotificationCenter.default.post(name: .aerialDetected, object: verticalAccel)
        }
    }
    
    // MARK: - Bluetooth Delegate Methods
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("✅ Bluetooth powered on")
            startBluetoothScanning()
        case .poweredOff:
            print("❌ Bluetooth powered off")
            sensorStatus = .error("Bluetooth disabled")
        case .unauthorized:
            sensorStatus = .error("Bluetooth unauthorized")
        case .unsupported:
            sensorStatus = .error("Bluetooth unsupported")
        default:
            sensorStatus = .error("Bluetooth unavailable")
        }
    }
    
    private func startBluetoothScanning() {
        guard centralManager.state == .poweredOn else { return }
        
        let services = [pressureSensorServiceUUID]
        centralManager.scanForPeripherals(withServices: services, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])
        
        print("🔍 Scanning for pressure sensors...")
    }
    
    private func stopBluetoothScanning() {
        centralManager.stopScan()
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        
        print("📡 Discovered peripheral: \(peripheral.name ?? "Unknown")")
        
        // Check if this is our pressure sensor
        if let name = peripheral.name, name.contains("PressureSensor") {
            pressureSensorPeripheral = peripheral
            peripheral.delegate = self
            centralManager.connect(peripheral, options: nil)
            centralManager.stopScan()
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("✅ Connected to pressure sensor")
        isConnected = true
        peripheral.discoverServices([pressureSensorServiceUUID])
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("❌ Failed to connect: \(error?.localizedDescription ?? "Unknown error")")
        sensorStatus = .error("Connection failed")
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        
        for service in services {
            if service.uuid == pressureSensorServiceUUID {
                peripheral.discoverCharacteristics([pressureCharacteristicUUID], for: service)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        
        for characteristic in characteristics {
            if characteristic.uuid == pressureCharacteristicUUID {
                pressureCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
                sensorStatus = .ready
                print("✅ Pressure sensor ready")
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == pressureCharacteristicUUID,
              let data = characteristic.value else { return }
        
        processPressureSensorData(data)
    }
    
    // MARK: - Pressure Sensor Data Processing
    private func processPressureSensorData(_ data: Data) {
        // Parse pressure sensor data (format depends on sensor)
        // Assuming 4-byte float array of pressure values
        let pressureCount = data.count / 4
        var newPressureData: [Float] = []
        
        for i in 0..<pressureCount {
            let offset = i * 4
            let pressureBytes = data.subdata(in: offset..<(offset + 4))
            let pressure = pressureBytes.withUnsafeBytes { bytes in
                bytes.load(as: Float.self)
            }
            newPressureData.append(pressure)
        }
        
        // Apply filtering and update
        let filteredPressures = pressureFilter.filter(newPressureData)
        
        DispatchQueue.main.async {
            self.pressureData = filteredPressures
        }
    }
}

// MARK: - Supporting Classes
class MotionFilter {
    private var yawHistory: [Float] = []
    private var pitchHistory: [Float] = []
    private var rollHistory: [Float] = []
    private let historySize = 5
    
    func filterYaw(_ value: Float) -> Float {
        return applyMovingAverage(&yawHistory, value: value)
    }
    
    func filterPitch(_ value: Float) -> Float {
        return applyMovingAverage(&pitchHistory, value: value)
    }
    
    func filterRoll(_ value: Float) -> Float {
        return applyMovingAverage(&rollHistory, value: value)
    }
    
    private func applyMovingAverage(_ history: inout [Float], value: Float) -> Float {
        history.append(value)
        if history.count > historySize {
            history.removeFirst()
        }
        return history.reduce(0, +) / Float(history.count)
    }
}

class PressureFilter {
    private let smoothingFactor: Float = 0.8
    private var previousValues: [Float] = []
    
    func filter(_ values: [Float]) -> [Float] {
        var filtered: [Float] = []
        
        for (index, value) in values.enumerated() {
            if index < previousValues.count {
                let smoothed = smoothingFactor * previousValues[index] + (1 - smoothingFactor) * value
                filtered.append(smoothed)
            } else {
                filtered.append(value)
            }
        }
        
        previousValues = filtered
        return filtered
    }
}

class CalibrationData {
    var yawOffset: Float = 0
    var pitchOffset: Float = 0
    var rollOffset: Float = 0
    
    private var yawSamples: [Float] = []
    private var pitchSamples: [Float] = []
    private var rollSamples: [Float] = []
    
    func reset() {
        yawSamples.removeAll()
        pitchSamples.removeAll()
        rollSamples.removeAll()
    }
    
    func addSample(yaw: Float, pitch: Float, roll: Float) {
        yawSamples.append(yaw)
        pitchSamples.append(pitch)
        rollSamples.append(roll)
    }
    
    func finalize() {
        yawOffset = yawSamples.reduce(0, +) / Float(max(yawSamples.count, 1))
        pitchOffset = pitchSamples.reduce(0, +) / Float(max(pitchSamples.count, 1))
        rollOffset = rollSamples.reduce(0, +) / Float(max(rollSamples.count, 1))
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let turnDetected = Notification.Name("turnDetected")
    static let bottomTurnDetected = Notification.Name("bottomTurnDetected")
    static let aerialDetected = Notification.Name("aerialDetected")
}