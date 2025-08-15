import Foundation
import Combine

final class FinViewModel: ObservableObject {
    // Inputs
    @Published var isManualAoA: Bool = false
    @Published var manualAoADegrees: Double = 5.0

    // Live values
    @Published private(set) var liveAoADegrees: Int = 0
    @Published private(set) var latestHRVms: Double?

    // Outputs
    @Published private(set) var liftCoefficient: Double = 0
    @Published private(set) var dragCoefficient: Double = 0

    // Config
    @Published var reynoldsNumber: Double = 1_000_000

    var reynoldsDisplayString: String {
        if reynoldsNumber >= 1000 {
            return String(format: "%.2e", reynoldsNumber)
        } else {
            return String(format: "%.0f", reynoldsNumber)
        }
    }

    let visualizer = FinVisualizer()

    private let sensorManager = SensorManager()
    private let predictor = FinPredictor()

    private var cancellables: Set<AnyCancellable> = []

    func startup() {
        bind()
        sensorManager.start()
    }

    func fetchHRV() {
        CognitiveTracker.shared.fetchHRV { [weak self] value in
            DispatchQueue.main.async {
                self?.latestHRVms = value
            }
        }
    }

    private func bind() {
        sensorManager.turnAnglePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] angleDegrees in
                guard let self else { return }
                self.liveAoADegrees = Int(angleDegrees)
                self.updateAoA()
                self.predict()
            }
            .store(in: &cancellables)

        sensorManager.pressureMapPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] normalized in
                self?.visualizer.updatePressureMap(pressureDataNormalized: normalized)
            }
            .store(in: &cancellables)

        $manualAoADegrees
            .combineLatest($isManualAoA)
            .debounce(for: .milliseconds(60), scheduler: DispatchQueue.main)
            .sink { [weak self] _, _ in
                self?.updateAoA()
                self?.predict()
            }
            .store(in: &cancellables)

        $reynoldsNumber
            .removeDuplicates()
            .debounce(for: .milliseconds(120), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.predict() }
            .store(in: &cancellables)
    }

    private func currentAoA() -> Double {
        if isManualAoA { return manualAoADegrees }
        return Double(liveAoADegrees)
    }

    private func updateAoA() {
        visualizer.updateAoA(degrees: Float(currentAoA()))
    }

    private func predict() {
        let aoa = currentAoA()
        let rakeDegrees: Double = 6.5
        let re = reynoldsNumber
        let result = predictor.predictLiftDrag(aoaDegrees: aoa, rakeDegrees: rakeDegrees, reynolds: re)
        liftCoefficient = result.lift
        dragCoefficient = result.drag
    }
}