import Foundation
import Combine
import CoreMotion
import CoreBluetooth

final class SensorManager: NSObject, ObservableObject {
    @Published var turnAoA: Float = 0
    @Published var pressureValues: [Float] = []

    private let motion = CMMotionManager()
    private var cancellables: Set<AnyCancellable> = []

    // Bluetooth placeholders
    private var central: CBCentralManager?

    override init() {
        super.init()
        #if USE_MOCK_SENSORS
        startMocking()
        #else
        central = CBCentralManager(delegate: self, queue: .main)
        #endif
    }

    func start() {
        #if USE_MOCK_SENSORS
        return
        #else
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 0.08
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
            guard let self = self, let attitude = data?.attitude else { return }
            let yawDeg = Float(attitude.yaw * 180 / .pi)
            self.turnAoA = max(0, min(20, abs(yawDeg)))
        }
        #endif
    }

    private func startMocking() {
        // Simulate IMU AoA and pressure values
        Timer.publish(every: 0.12, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self = self else { return }
                let t = Date().timeIntervalSinceReferenceDate
                let aoa = Float(10 + 8 * sin(t * 0.7))
                self.turnAoA = max(0, min(20, aoa))
                let base = (sin(t) + 1) * 0.5
                self.pressureValues = (0..<32).map { _ in Float(min(1.0, max(0.0, base + Double.random(in: -0.1...0.1)))) }
            }
            .store(in: &cancellables)
    }
}

extension SensorManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        // For a real device, start scanning when powered on
        if central.state == .poweredOn {
            // central.scanForPeripherals(withServices: [...], options: nil)
        }
    }
}