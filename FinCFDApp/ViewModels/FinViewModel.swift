import Foundation
import Combine

final class FinViewModel: ObservableObject {
	@Published var turnAngle: Float = 0.0 {
		didSet {
			updatePredictions(aoa: turnAngle)
			visualizer.updateAngleOfAttack(aoaDegrees: turnAngle)
		}
	}
	@Published var liftDrag: (lift: Float, drag: Float)?
	@Published var pressureData: [Float] = []
	@Published var hrv: Double?

	let visualizer = FinVisualizer()
	private var cancellables: Set<AnyCancellable> = []
	private let sensorManager = SensorManager()
	private let predictor: FinPredictor?
	private let cognitiveTracker = CognitiveTracker()

	init() {
		predictor = FinPredictor()

		sensorManager.$turnAngle
			.receive(on: DispatchQueue.main)
			.sink { [weak self] angle in
				self?.turnAngle = angle
			}
			.store(in: &cancellables)

		sensorManager.$pressureData
			.receive(on: DispatchQueue.main)
			.sink { [weak self] data in
				self?.pressureData = data
				self?.visualizer.updatePressureMap(pressureData: data)
			}
			.store(in: &cancellables)
	}

	func startMonitoring() {
		sensorManager.startMonitoring()
	}

	func updatePredictions(aoa: Float) {
		guard let result = predictor?.predictLiftDrag(aoa: aoa, rake: 6.5, re: 1e6) else { return }
		liftDrag = result
	}

	func fetchHRV() {
		cognitiveTracker.fetchHRV { [weak self] value in
			DispatchQueue.main.async { self?.hrv = value }
		}
	}
}