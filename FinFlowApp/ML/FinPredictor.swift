import Foundation
import CoreML

final class FinPredictor {
    private let model: FinCFDModel?

    init() {
        model = try? FinCFDModel(configuration: MLModelConfiguration())
    }

    func predict(aoa: Float, rake: Float, reynolds: Float) throws -> LiftDrag {
        if let model = model {
            do {
                let input = try MLMultiArray(shape: [3], dataType: .float32)
                input[0] = NSNumber(value: aoa)
                input[1] = NSNumber(value: rake)
                input[2] = NSNumber(value: reynolds)
                let out = try model.prediction(input: FinCFDModelInput(input: input))
                let lift = Float(truncating: out.output[0])
                let drag = Float(truncating: out.output[1])
                return LiftDrag(lift: lift, drag: drag)
            } catch {
                throw AppError.predictionFailed(error.localizedDescription)
            }
        }
        // Fallback surrogate: simple aerodynamic proxy using Cl ~ a*alpha, Cd ~ Cd0 + k*Cl^2
        let alphaRad = Double(max(0, min(20, aoa))) * .pi / 180.0
        let a = 2.2 // slope tuned for foil at Re ~ 1e5-1e6
        let cl = a * alphaRad
        let cd0 = 0.012
        let k = 0.95
        let cd = cd0 + k * pow(cl, 2)
        // Reynolds scaling (weak)
        let reScale = min(1.2, max(0.8, Double(reynolds) / 1_000_000.0))
        return LiftDrag(lift: Float(cl * reScale), drag: Float(cd))
    }
}