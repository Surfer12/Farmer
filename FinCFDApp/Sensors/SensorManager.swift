import Foundation
import Combine

#if os(iOS)
import CoreMotion
import CoreBluetooth
final class SensorManager: NSObject, ObservableObject, CBCentralManagerDelegate {
	let motionManager = CMMotionManager()
	private var centralManager: CBCentralManager!
	@Published var turnAngle: Float = 0.0
	@Published var pressureData: [Float] = []

	override init() {
		super.init()
		centralManager = CBCentralManager(delegate: self, queue: .main)
	}

	func startMonitoring() {
		if motionManager.isDeviceMotionAvailable {
			motionManager.deviceMotionUpdateInterval = 0.1
			motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
				guard let attitude = motion?.attitude else { return }
				self?.turnAngle = Float(attitude.yaw * 180.0 / .pi)
			}
		}
	}

	func centralManagerDidUpdateState(_ central: CBCentralManager) {
		if central.state == .poweredOn {
			central.scanForPeripherals(withServices: nil)
		}
	}

	func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
		// Implement BLE discovery/connection/parse pressure data
	}
}
#else
final class SensorManager: ObservableObject {
	@Published var turnAngle: Float = 0.0
	@Published var pressureData: [Float] = []

	func startMonitoring() {
		// macOS: Provide stubs or integrate external sensors as needed
	}
}
#endif