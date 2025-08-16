// SPDX-License-Identifier: GPL-3.0-only
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
    @Published private(set) var psiScore: Double = 0

    // Config
    @Published var reynoldsNumber: Double = 1_000_000
    @Published var psiParameters = PsiParameters()

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
                self?.updatePsi()
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
                self.updatePsi()
            }
            .store(in: &cancellables)

        sensorManager.pressureMapPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] normalized in
                self?.visualizer.updatePressureMap(pressureDataNormalized: normalized)
                self?.updatePsi()
            }
            .store(in: &cancellables)

        $manualAoADegrees
            .combineLatest($isManualAoA)
            .debounce(for: .milliseconds(60), scheduler: DispatchQueue.main)
            .sink { [weak self] _, _ in
                self?.updateAoA()
                self?.predict()
                self?.updatePsi()
            }
            .store(in: &cancellables)

        $reynoldsNumber
            .removeDuplicates()
            .debounce(for: .milliseconds(120), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.predict()
                self?.updatePsi()
            }
            .store(in: &cancellables)

        $psiParameters
            .debounce(for: .milliseconds(120), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.updatePsi() }
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

    private func updatePsi() {
        // Map to S and N proxies
        let aoa = currentAoA()
        let S = min(1.0, max(0.0, 0.6 + 0.4 * (20.0 - abs(aoa - 10.0)) / 20.0))
        let modelConf = 1.0 - min(1.0, max(0.0, (abs(aoa - 15.0) / 20.0))) * 0.2
        let N = min(1.0, max(0.0, 0.8 * modelConf))
        var p = psiParameters
        if let hrv = latestHRVms {
            // Map HRV to cognitive risk: lower HRV -> higher risk, approx 0.06..0.16
            let clamped = min(120.0, max(20.0, hrv))
            let t = (120.0 - clamped) / 100.0 // 0 at 120ms, 1 at 20ms
            p.riskCognitive = 0.06 + 0.10 * t
        }
        psiScore = PsiModel.compute(S: S, N: N, parameters: p)
    }

    // Quick actions
    func setLaminar() { isManualAoA = true; manualAoADegrees = 10.0 }
    func setTurbulent() { isManualAoA = true; manualAoADegrees = 20.0 }
}