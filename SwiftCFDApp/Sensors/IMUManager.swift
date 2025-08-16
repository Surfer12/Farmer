import Foundation
import Combine
import CoreMotion

final class IMUManager: ObservableObject {
    @Published var aoaDegrees: Float = 0
    private let motion = CMMotionManager()

    func start() -> AppError? {
        guard motion.isDeviceMotionAvailable else { return .motionUnavailable }
        motion.deviceMotionUpdateInterval = 0.05
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, error in
            if let _ = error {
                return
            }
            guard let data = data else { return }
            // Approx AoA from roll magnitude (board banking). Clamp to 0–20° range.
            let degrees = min(max(Float(abs(data.attitude.roll)) * 180 / .pi, 0), Defaults.aoaDegreesRange.upperBound)
            self?.aoaDegrees = degrees
        }
        return nil
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
    }
}