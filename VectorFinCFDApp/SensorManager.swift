import CoreMotion
import CoreBluetooth
import Combine
import Foundation

class SensorManager: NSObject, ObservableObject, CBCentralManagerDelegate {
    private let motionManager = CMMotionManager()
    private var centralManager: CBCentralManager!
    
    @Published var turnAngle: Float = 0.0
    @Published var pressureData: [Float] = []
    @Published var isBluetoothEnabled = false
    @Published var sensorStatus = "Initializing..."
    
    // Sensor configuration
    private let updateInterval: TimeInterval = 0.1 // 10 Hz
    private let maxTurnAngle: Float = 20.0
    private let pressureSensorCount = 10
    
    // Mock pressure data for development
    private var mockPressureTimer: Timer?
    
    override init() {
        super.init()
        setupBluetooth()
        setupMotionManager()
        startMockPressureData()
    }
    
    deinit {
        stopMonitoring()
        mockPressureTimer?.invalidate()
    }
    
    // MARK: - Bluetooth Setup
    
    private func setupBluetooth() {
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        DispatchQueue.main.async { [weak self] in
            switch central.state {
            case .poweredOn:
                self?.isBluetoothEnabled = true
                self?.sensorStatus = "Bluetooth enabled, scanning for sensors..."
                self?.startScanning()
            case .poweredOff:
                self?.isBluetoothEnabled = false
                self?.sensorStatus = "Bluetooth is powered off"
            case .unauthorized:
                self?.sensorStatus = "Bluetooth access denied"
            case .unsupported:
                self?.sensorStatus = "Bluetooth not supported"
            case .resetting:
                self?.sensorStatus = "Bluetooth is resetting..."
            case .unknown:
                self?.sensorStatus = "Bluetooth state unknown"
            @unknown default:
                self?.sensorStatus = "Bluetooth state unknown"
            }
        }
    }
    
    private func startScanning() {
        guard centralManager.state == .poweredOn else { return }
        
        // Scan for pressure sensor peripherals
        // In a real implementation, you would specify the service UUID
        centralManager.scanForPeripherals(withServices: nil, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])
        
        // Stop scanning after 10 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) { [weak self] in
            self?.centralManager.stopScan()
            self?.sensorStatus = "Scan complete, using mock data"
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
        // Check if this is a pressure sensor peripheral
        if let name = peripheral.name, name.contains("Pressure") || name.contains("Fin") {
            sensorStatus = "Found pressure sensor: \(name)"
            
            // In a real implementation, you would connect to the peripheral
            // For now, we'll simulate connection and data reception
            simulatePressureSensorConnection(peripheral: peripheral)
        }
    }
    
    private func simulatePressureSensorConnection(peripheral: CBPeripheral) {
        sensorStatus = "Connected to \(peripheral.name ?? "Pressure Sensor")"
        
        // Simulate real-time pressure data updates
        DispatchQueue.main.async { [weak self] in
            self?.generateRealisticPressureData()
        }
    }
    
    // MARK: - Motion Manager Setup
    
    private func setupMotionManager() {
        guard motionManager.isDeviceMotionAvailable else {
            sensorStatus = "Device motion not available"
            return
        }
        
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.showsDeviceMovementDisplay = true
    }
    
    func startMonitoring() {
        startMotionUpdates()
        startMockPressureData()
    }
    
    func stopMonitoring() {
        motionManager.stopDeviceMotionUpdates()
        mockPressureTimer?.invalidate()
        mockPressureTimer = nil
    }
    
    private func startMotionUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            sensorStatus = "Device motion not available"
            return
        }
        
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            if let error = error {
                self?.sensorStatus = "Motion error: \(error.localizedDescription)"
                return
            }
            
            guard let motion = motion else { return }
            
            // Calculate turn angle from device attitude
            let yaw = motion.attitude.yaw
            let yawDegrees = Float(yaw * 180 / .pi)
            
            // Normalize to 0-20° range for fin analysis
            let normalizedAngle = abs(yawDegrees).truncatingRemainder(dividingBy: 360)
            let clampedAngle = min(normalizedAngle, self?.maxTurnAngle ?? 20.0)
            
            DispatchQueue.main.async {
                self?.turnAngle = clampedAngle
                self?.sensorStatus = "Monitoring motion - Turn: \(Int(clampedAngle))°"
            }
        }
    }
    
    // MARK: - Pressure Data Generation
    
    private func startMockPressureData() {
        mockPressureTimer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            self?.generateRealisticPressureData()
        }
    }
    
    private func generateRealisticPressureData() {
        // Generate realistic pressure data based on current turn angle
        // Simulate CFD pressure distribution on Vector 3/2 fins
        
        var newPressureData: [Float] = []
        
        // Side fin 1 pressure (affected by turn angle)
        let sideFin1Pressure = calculateFinPressure(
            basePressure: 0.3,
            turnEffect: turnAngle,
            finType: .side,
            position: .left
        )
        newPressureData.append(sideFin1Pressure)
        
        // Side fin 2 pressure (opposite effect)
        let sideFin2Pressure = calculateFinPressure(
            basePressure: 0.3,
            turnEffect: turnAngle,
            finType: .side,
            position: .right
        )
        newPressureData.append(sideFin2Pressure)
        
        // Center fin pressure (minimal turn effect)
        let centerFinPressure = calculateFinPressure(
            basePressure: 0.4,
            turnEffect: turnAngle,
            finType: .center,
            position: .center
        )
        newPressureData.append(centerFinPressure)
        
        // Additional pressure points for detailed mapping
        for i in 3..<pressureSensorCount {
            let basePressure = Float.random(in: 0.2...0.6)
            let turnEffect = turnAngle * Float.random(in: 0.1...0.3)
            let pressure = basePressure + (turnEffect / 20.0)
            newPressureData.append(max(0, min(1, pressure)))
        }
        
        DispatchQueue.main.async {
            self.pressureData = newPressureData
        }
    }
    
    private func calculateFinPressure(basePressure: Float, turnEffect: Float, finType: FinType, position: FinPosition) -> Float {
        let normalizedTurn = turnEffect / maxTurnAngle
        
        var pressure = basePressure
        
        switch finType {
        case .side:
            switch position {
            case .left:
                // Left fin: pressure increases with right turn
                pressure += normalizedTurn * 0.4
            case .right:
                // Right fin: pressure decreases with right turn
                pressure -= normalizedTurn * 0.4
            case .center:
                pressure += normalizedTurn * 0.1
            }
        case .center:
            // Center fin: minimal turn effect
            pressure += normalizedTurn * 0.05
        }
        
        // Apply Vector 3/2 foil characteristics
        pressure *= 1.12 // 12% pressure increase for raked fins
        
        // Add some realistic noise
        let noise = Float.random(in: -0.05...0.05)
        pressure += noise
        
        return max(0, min(1, pressure))
    }
    
    // MARK: - Data Export
    
    func exportSensorData() -> String {
        let timestamp = Date().ISO8601String()
        let data = """
        Timestamp: \(timestamp)
        Turn Angle: \(turnAngle)°
        Pressure Data: \(pressureData.map { String(format: "%.3f", $0) }.joined(separator: ", "))
        Sensor Status: \(sensorStatus)
        Bluetooth Enabled: \(isBluetoothEnabled)
        """
        return data
    }
}

// MARK: - Supporting Types

enum FinType {
    case side
    case center
}

enum FinPosition {
    case left
    case right
    case center
}

// MARK: - Extensions

extension Date {
    func ISO8601String() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}