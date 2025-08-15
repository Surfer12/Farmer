import Foundation
import Combine

final class FinViewModel: ObservableObject {
    // Inputs / State
    @Published var aoa: Float = 10
    @Published var flowMode: FlowMode = .laminar {
        didSet { visualizer.setFlowMode(flowMode) }
    }

    // Outputs
    @Published var liftDrag: LiftDrag = LiftDrag(lift: 0, drag: 0)
    @Published var imuTurnAoA: Float = 0
    @Published var avgPressure: Float = 0
    @Published var hrvText: String = "--"

    let visualizer = FinVisualizer()

    private let predictor = FinPredictor()
    private let sensors = SensorManager()
    private let cognitive = CognitiveTracker()
    private var bag: Set<AnyCancellable> = []

    func start() {
        visualizer.setupFinModel()
        sensors.start()

        // IMU → AoA display and visualization tilt
        sensors.$turnAoA
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.imuTurnAoA = value
                self?.visualizer.tiltFins(forAoA: value)
            }
            .store(in: &bag)

        // Pressure → color map
        sensors.$pressureValues
            .receive(on: DispatchQueue.main)
            .sink { [weak self] values in
                self?.avgPressure = values.isEmpty ? 0 : values.reduce(0, +) / Float(values.count)
                self?.visualizer.updatePressureMap(values: values)
            }
            .store(in: &bag)

        // Kick initial prediction
        userChangedAoA()
    }

    func userChangedAoA() {
        Task { @MainActor in
            do {
                let result = try predictor.predict(aoa: aoa, rake: 6.5, reynolds: 1_000_000)
                self.liftDrag = result
                self.visualizer.tiltFins(forAoA: aoa)
                self.flowMode = aoa < 12 ? .laminar : .turbulent
            } catch {
                // Keep UI responsive; show zeros on error
                self.liftDrag = LiftDrag(lift: 0, drag: 0)
            }
        }
    }

    func fetchHRV() {
        cognitive.fetchHRV { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let ms): self?.hrvText = String(format: "%.1f ms", ms)
                case .failure: self?.hrvText = "--"
                }
            }
        }
    }
}