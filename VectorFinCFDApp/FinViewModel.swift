// SPDX-License-Identifier: GPL-3.0-only
import Foundation
import Combine

final class FinViewModel: ObservableObject {
	@Published var turnAngle: Float = 0.0 {
		didSet {
			updatePredictions(aoa: turnAngle)
			visualizer.updateFlowVisualization(angleOfAttack: turnAngle)
			visualizer.animateFinRotation(to: turnAngle)
		}
	}
	@Published var liftDrag: (lift: Float, drag: Float)?
	@Published var pressureData: [Float] = []
	@Published var hrv: Double?
	
	let visualizer = FinVisualizer()
	
	private var cancellables: Set<AnyCancellable> = []
	private let sensorManager = SensorManager()
	private let predictor = FinPredictor()
	private let cognitiveTracker = CognitiveTracker()
	
	init() {
		bindPipelines()
	}
	
	func startMonitoring() {
		sensorManager.startMonitoring()
	}
	
	private func bindPipelines() {
		// Turn angle from sensors -> UI + predictions + animations
		sensorManager.$turnAngle
			.receive(on: DispatchQueue.main)
			.removeDuplicates()
			.debounce(for: .milliseconds(80), scheduler: DispatchQueue.main)
			.sink { [weak self] angle in
				self?.turnAngle = angle
			}
			.store(in: &cancellables)
		
		// Pressure data -> visualization
		sensorManager.$pressureData
			.receive(on: DispatchQueue.main)
			.sink { [weak self] data in
				guard let self else { return }
				self.pressureData = data
				self.visualizer.updatePressureMap(pressureData: data)
			}
			.store(in: &cancellables)
	}
	
	func updatePredictions(aoa: Float) {
		do {
			let result = try predictor.predictLiftDrag(aoa: aoa, rake: 6.5, re: 1_000_000)
			DispatchQueue.main.async { [weak self] in
				self?.liftDrag = result
			}
		} catch {
			print("Prediction error: \(error)")
		}
	}
	
	func fetchHRV() {
		cognitiveTracker.fetchHRV { [weak self] outcome in
			DispatchQueue.main.async {
				switch outcome {
				case .success(let value):
					self?.hrv = value
				case .failure(let error):
					print("HRV fetch error: \(error)")
				}
			}
		}
	}
}