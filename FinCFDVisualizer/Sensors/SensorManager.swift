import Foundation
import Combine

#if os(iOS)
import CoreMotion
import CoreBluetooth
#endif

final class SensorManager: NSObject {
    private let turnAngleSubject = PassthroughSubject<Int, Never>()
    var turnAnglePublisher: AnyPublisher<Int, Never> { turnAngleSubject.eraseToAnyPublisher() }

    private let pressureMapSubject = PassthroughSubject<[Float], Never>()
    var pressureMapPublisher: AnyPublisher<[Float], Never> { pressureMapSubject.eraseToAnyPublisher() }

    #if os(iOS)
    private let motionManager = CMMotionManager()
    private var centralManager: CBCentralManager?
    #endif

    func start() {
        #if os(iOS)
        startMotion()
        centralManager = CBCentralManager(delegate: self, queue: .main)
        #else
        // macOS: emit a slow, synthetic AoA sweep for demo purposes
        var degrees = 0
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] timer in
            guard let self else { return }
            self.turnAngleSubject.send(degrees)
            degrees = (degrees + 1) % 21
        }
        #endif
    }

    #if os(iOS)
    private func startMotion() {
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 0.08
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            // Use yaw as a proxy for rail-to-rail rotation AoA
            let yawDegrees = Int((motion.attitude.yaw * 180.0 / .pi).truncatingRemainder(dividingBy: 360))
            let clamped = max(0, min(20, abs(yawDegrees)))
            self.turnAngleSubject.send(clamped)
        }
    }
    #endif
}

#if os(iOS)
extension SensorManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            // TODO: supply your pressure sensor service UUIDs
            central.scanForPeripherals(withServices: nil, options: nil)
        default:
            break
        }
    }
}
#endif