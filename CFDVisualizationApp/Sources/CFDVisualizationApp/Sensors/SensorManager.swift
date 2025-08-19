import Foundation
import CoreMotion
import CoreBluetooth
import HealthKit
import Combine

/// Comprehensive sensor management for CFD visualization and cognitive metrics
class SensorManager: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var turnAngle: Float = 0.0 // Yaw angle from IMU
    @Published var pitch: Float = 0.0 // Pitch angle
    @Published var roll: Float = 0.0 // Roll angle
    @Published var acceleration: (x: Float, y: Float, z: Float) = (0, 0, 0)
    @Published var pressureData: [Float] = [] // From Bluetooth pressure sensors
    @Published var heartRate: Double = 0.0 // From HealthKit
    @Published var heartRateVariability: Double = 0.0 // HRV for flow state
    @Published var isConnected: Bool = false
    @Published var sensorStatus: SensorStatus = .disconnected
    
    // MARK: - Private Properties
    
    private let motionManager = CMMotionManager()
    private var centralManager: CBCentralManager?
    private var pressureSensorPeripheral: CBPeripheral?
    private let healthStore = HKHealthStore()
    
    // Combine cancellables
    private var cancellables = Set<AnyCancellable>()
    
    // Data processing
    private let kalmanFilter = KalmanFilter()
    private var dataBuffer: CircularBuffer<SensorReading> = CircularBuffer(capacity: 100)
    
    // Timing
    private let updateInterval: TimeInterval = 0.1 // 10 Hz
    private var lastUpdateTime: Date = Date()
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupSensors()
        setupHealthKit()
        setupDataProcessing()
    }
    
    // MARK: - Setup Methods
    
    private func setupSensors() {
        // Configure motion manager
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.accelerometerUpdateInterval = updateInterval
        
        // Setup Bluetooth central manager
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    private func setupHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit not available")
            return
        }
        
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        
        let typesToRead: Set<HKObjectType> = [heartRateType, hrvType]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            if success {
                self?.startHealthKitCollection()
            } else {
                print("HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    private func setupDataProcessing() {
        // Combine sensor streams for real-time processing
        Publishers.CombineLatest3($turnAngle, $pitch, $roll)
            .debounce(for: .milliseconds(50), scheduler: RunLoop.main)
            .sink { [weak self] yaw, pitch, roll in
                self?.processSensorData(yaw: yaw, pitch: pitch, roll: roll)
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public Methods
    
    /// Starts all sensor monitoring
    func startMonitoring() {
        startIMUMonitoring()
        startBluetoothScanning()
        startHealthKitCollection()
        sensorStatus = .connecting
    }
    
    /// Stops all sensor monitoring
    func stopMonitoring() {
        stopIMUMonitoring()
        stopBluetoothScanning()
        stopHealthKitCollection()
        sensorStatus = .disconnected
    }
    
    /// Gets current angle of attack based on board orientation
    func getCurrentAngleOfAttack() -> Float {
        // Convert IMU data to surfboard angle of attack
        // This is a simplified calculation - in practice would need calibration
        let aoa = pitch * 180.0 / Float.pi // Convert radians to degrees
        return max(-20.0, min(20.0, aoa)) // Clamp to realistic range
    }
    
    /// Gets current flow state metrics
    func getFlowStateMetrics() -> FlowStateMetrics {
        return FlowStateMetrics(
            heartRate: heartRate,
            hrv: heartRateVariability,
            motionStability: calculateMotionStability(),
            focusLevel: calculateFocusLevel(),
            timestamp: Date()
        )
    }
    
    // MARK: - IMU Monitoring
    
    private func startIMUMonitoring() {
        guard motionManager.isDeviceMotionAvailable else {
            print("Device motion not available")
            return
        }
        
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self = self, let motion = motion else {
                if let error = error {
                    print("Motion update error: \(error)")
                }
                return
            }
            
            // Update published properties
            self.turnAngle = Float(motion.attitude.yaw)
            self.pitch = Float(motion.attitude.pitch)
            self.roll = Float(motion.attitude.roll)
            
            // Update acceleration
            let accel = motion.userAcceleration
            self.acceleration = (
                x: Float(accel.x),
                y: Float(accel.y),
                z: Float(accel.z)
            )
            
            // Apply Kalman filtering for smoothing
            let filteredData = self.kalmanFilter.update(
                yaw: self.turnAngle,
                pitch: self.pitch,
                roll: self.roll
            )
            
            self.turnAngle = filteredData.yaw
            self.pitch = filteredData.pitch
            self.roll = filteredData.roll
            
            // Store in buffer for analysis
            let reading = SensorReading(
                timestamp: Date(),
                yaw: self.turnAngle,
                pitch: self.pitch,
                roll: self.roll,
                acceleration: self.acceleration
            )
            self.dataBuffer.append(reading)
        }
    }
    
    private func stopIMUMonitoring() {
        motionManager.stopDeviceMotionUpdates()
    }
    
    // MARK: - Bluetooth Pressure Sensors
    
    private func startBluetoothScanning() {
        guard let centralManager = centralManager else { return }
        
        if centralManager.state == .poweredOn {
            centralManager.scanForPeripherals(withServices: nil, options: nil)
        }
    }
    
    private func stopBluetoothScanning() {
        centralManager?.stopScan()
        centralManager?.cancelPeripheralConnection(pressureSensorPeripheral!)
    }
    
    // MARK: - HealthKit Integration
    
    private func startHealthKitCollection() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        startHeartRateCollection()
        startHRVCollection()
    }
    
    private func startHeartRateCollection() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        
        let query = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("Heart rate observer error: \(error)")
                return
            }
            
            self?.fetchLatestHeartRate()
        }
        
        healthStore.execute(query)
        fetchLatestHeartRate() // Initial fetch
    }
    
    private func fetchLatestHeartRate() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        let query = HKSampleQuery(sampleType: heartRateType, predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else { return }
            
            DispatchQueue.main.async {
                self?.heartRate = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }
        }
        
        healthStore.execute(query)
    }
    
    private func startHRVCollection() {
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        
        let query = HKObserverQuery(sampleType: hrvType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("HRV observer error: \(error)")
                return
            }
            
            self?.fetchLatestHRV()
        }
        
        healthStore.execute(query)
        fetchLatestHRV() // Initial fetch
    }
    
    private func fetchLatestHRV() {
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrvType, predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else { return }
            
            DispatchQueue.main.async {
                self?.heartRateVariability = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            }
        }
        
        healthStore.execute(query)
    }
    
    private func stopHealthKitCollection() {
        // HealthKit queries are automatically managed
    }
    
    // MARK: - Data Processing
    
    private func processSensorData(yaw: Float, pitch: Float, roll: Float) {
        // Update connection status
        isConnected = true
        sensorStatus = .connected
        lastUpdateTime = Date()
        
        // Process data for cognitive metrics
        updateCognitiveMetrics()
    }
    
    private func updateCognitiveMetrics() {
        // This would implement more sophisticated cognitive load analysis
        // For now, we'll use simplified metrics based on motion stability and HRV
    }
    
    private func calculateMotionStability() -> Float {
        guard dataBuffer.count > 10 else { return 0.5 }
        
        // Calculate variance in motion over recent samples
        let recentSamples = dataBuffer.recent(10)
        let yawVariance = variance(recentSamples.map { $0.yaw })
        let pitchVariance = variance(recentSamples.map { $0.pitch })
        let rollVariance = variance(recentSamples.map { $0.roll })
        
        let totalVariance = yawVariance + pitchVariance + rollVariance
        
        // Convert to stability metric (lower variance = higher stability)
        return max(0.0, min(1.0, 1.0 - totalVariance / 10.0))
    }
    
    private func calculateFocusLevel() -> Float {
        // Combine HRV and motion stability for focus estimation
        let hrvComponent = min(1.0, Float(heartRateVariability / 50.0)) // Normalize HRV
        let motionComponent = calculateMotionStability()
        
        return (hrvComponent + motionComponent) / 2.0
    }
    
    private func variance(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        
        let mean = values.reduce(0, +) / Float(values.count)
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        return squaredDiffs.reduce(0, +) / Float(values.count)
    }
}

// MARK: - CBCentralManagerDelegate

extension SensorManager: CBCentralManagerDelegate {
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            startBluetoothScanning()
        case .poweredOff, .unauthorized, .unsupported:
            sensorStatus = .error("Bluetooth not available")
        default:
            break
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        
        // Look for pressure sensor peripherals
        if let name = peripheral.name, name.contains("PressureSensor") {
            pressureSensorPeripheral = peripheral
            central.connect(peripheral, options: nil)
            central.stopScan()
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.delegate = self
        peripheral.discoverServices(nil)
        sensorStatus = .connected
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        sensorStatus = .error("Failed to connect to pressure sensor")
    }
}

// MARK: - CBPeripheralDelegate

extension SensorManager: CBPeripheralDelegate {
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        
        for service in services {
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        
        for characteristic in characteristics {
            if characteristic.properties.contains(.notify) {
                peripheral.setNotifyValue(true, for: characteristic)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value else { return }
        
        // Parse pressure sensor data
        let pressureValues = parsePressureData(data)
        
        DispatchQueue.main.async {
            self.pressureData = pressureValues
        }
    }
    
    private func parsePressureData(_ data: Data) -> [Float] {
        // Parse binary pressure data from sensor
        // This is a simplified implementation
        var values: [Float] = []
        
        for i in stride(from: 0, to: data.count, by: 4) {
            if i + 4 <= data.count {
                let floatValue = data.subdata(in: i..<i+4).withUnsafeBytes { $0.load(as: Float.self) }
                values.append(floatValue)
            }
        }
        
        return values
    }
}

// MARK: - Supporting Types

enum SensorStatus {
    case disconnected
    case connecting
    case connected
    case error(String)
    
    var description: String {
        switch self {
        case .disconnected:
            return "Disconnected"
        case .connecting:
            return "Connecting..."
        case .connected:
            return "Connected"
        case .error(let message):
            return "Error: \(message)"
        }
    }
}

struct SensorReading {
    let timestamp: Date
    let yaw: Float
    let pitch: Float
    let roll: Float
    let acceleration: (x: Float, y: Float, z: Float)
}

struct FlowStateMetrics {
    let heartRate: Double
    let hrv: Double // Heart rate variability
    let motionStability: Float
    let focusLevel: Float
    let timestamp: Date
    
    /// Overall flow state score (0.0 to 1.0)
    var flowScore: Float {
        let hrvNormalized = min(1.0, Float(hrv / 50.0)) // Normalize HRV
        let hrNormalized = 1.0 - abs(Float(heartRate - 150.0) / 50.0) // Optimal around 150 BPM
        
        return (hrvNormalized + max(0, hrNormalized) + motionStability + focusLevel) / 4.0
    }
    
    var flowState: FlowState {
        let score = flowScore
        if score > 0.8 {
            return .optimal
        } else if score > 0.6 {
            return .good
        } else if score > 0.4 {
            return .moderate
        } else {
            return .poor
        }
    }
}

enum FlowState {
    case optimal, good, moderate, poor
    
    var description: String {
        switch self {
        case .optimal: return "Optimal Flow"
        case .good: return "Good Flow"
        case .moderate: return "Moderate Flow"
        case .poor: return "Poor Flow"
        }
    }
    
    var color: (r: Float, g: Float, b: Float) {
        switch self {
        case .optimal: return (0.0, 1.0, 0.0) // Green
        case .good: return (0.5, 1.0, 0.0) // Yellow-green
        case .moderate: return (1.0, 1.0, 0.0) // Yellow
        case .poor: return (1.0, 0.0, 0.0) // Red
        }
    }
}

/// Simple Kalman filter for sensor data smoothing
class KalmanFilter {
    private var processNoise: Float = 0.01
    private var measurementNoise: Float = 0.1
    private var estimation: (yaw: Float, pitch: Float, roll: Float) = (0, 0, 0)
    private var errorCovariance: Float = 1.0
    
    func update(yaw: Float, pitch: Float, roll: Float) -> (yaw: Float, pitch: Float, roll: Float) {
        // Simplified Kalman filter implementation
        let kalmanGain = errorCovariance / (errorCovariance + measurementNoise)
        
        estimation.yaw += kalmanGain * (yaw - estimation.yaw)
        estimation.pitch += kalmanGain * (pitch - estimation.pitch)
        estimation.roll += kalmanGain * (roll - estimation.roll)
        
        errorCovariance = (1 - kalmanGain) * errorCovariance + processNoise
        
        return estimation
    }
}

/// Circular buffer for sensor data storage
class CircularBuffer<T> {
    private var buffer: [T]
    private var head = 0
    private var tail = 0
    private var _count = 0
    private let capacity: Int
    
    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array<T?>(repeating: nil, count: capacity) as! [T]
    }
    
    func append(_ element: T) {
        buffer[tail] = element
        tail = (tail + 1) % capacity
        
        if _count < capacity {
            _count += 1
        } else {
            head = (head + 1) % capacity
        }
    }
    
    var count: Int { return _count }
    
    func recent(_ n: Int) -> [T] {
        let actualN = min(n, _count)
        var result: [T] = []
        
        for i in 0..<actualN {
            let index = (tail - 1 - i + capacity) % capacity
            result.append(buffer[index])
        }
        
        return result.reversed()
    }
}