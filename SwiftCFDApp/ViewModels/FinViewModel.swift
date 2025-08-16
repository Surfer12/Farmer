import Foundation
import Combine

final class FinViewModel: ObservableObject {
    // Inputs
    @Published var aoaDegrees: Float = 0
    @Published var flowMode: FlowMode = .laminar

    // Outputs
    @Published var liftDrag: LiftDrag = LiftDrag(lift: 0, drag: 0)
    @Published var hrvSDNNms: Double?
    @Published var appError: AppError?

    // Services
    let visualizer = FinVisualizer()
    private let imu = IMUManager()
    private let pressure = PressureSensorManager()
    private let predictor: FinPredicting = FinPredictor()
    private let cognitive = CognitiveTracker()

    private var bag = Set<AnyCancellable>()

    init() {
        bind()
    }

    func start() {
        if let err = imu.start() { appError = err }
        pressure.startScanning()
        cognitive.requestAuthorization { [weak self] err in
            if let err = err { self?.appError = err }
        }
        cognitive.fetchLatest()
    }

    private func bind() {
        imu.$aoaDegrees
            .receive(on: DispatchQueue.main)
            .removeDuplicates()
            .map { min(max($0, Defaults.aoaDegreesRange.lowerBound), Defaults.aoaDegreesRange.upperBound) }
            .assign(to: &$aoaDegrees)

        $aoaDegrees
            .combineLatest(Just(Defaults.reynolds), Just(Defaults.rakeDegrees), Just(FinFoil.vector32))
            .debounce(for: .milliseconds(90), scheduler: DispatchQueue.main)
            .sink { [weak self] aoa, re, rake, foil in
                guard let self = self else { return }
                do {
                    let result = try self.predictor.predict(aoaDegrees: aoa, rakeDegrees: rake, reynolds: re, foil: foil)
                    self.liftDrag = result
                } catch {
                    self.appError = .predictionFailed
                }
            }
            .store(in: &bag)

        pressure.$pressures
            .receive(on: DispatchQueue.main)
            .sink { [weak self] values in
                self?.visualizer.updatePressureTexture(values)
            }
            .store(in: &bag)

        $flowMode
            .sink { [weak self] mode in
                self?.visualizer.setFlowMode(mode)
            }
            .store(in: &bag)

        cognitive.$hrvSDNNms
            .receive(on: DispatchQueue.main)
            .assign(to: &$hrvSDNNms)
    }
}