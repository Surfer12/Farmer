import Foundation
import CoreML

protocol FinPredicting {
    func predict(aoaDegrees: Float, rakeDegrees: Float, reynolds: Float, foil: FinFoil) throws -> LiftDrag
}

final class FinPredictor: FinPredicting {
    private let model: MLModel?

    init() {
        if let url = Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodelc") {
            do {
                model = try MLModel(contentsOf: url)
            } catch {
                model = nil
            }
        } else {
            model = nil
        }
    }

    func predict(aoaDegrees: Float, rakeDegrees: Float, reynolds: Float, foil: FinFoil) throws -> LiftDrag {
        if let model = model {
            do {
                let aoa = NSNumber(value: aoaDegrees)
                let rake = NSNumber(value: rakeDegrees)
                let re = NSNumber(value: reynolds)
                let array = try MLMultiArray(shape: [3], dataType: .float32)
                array[0] = aoa
                array[1] = rake
                array[2] = re

                let input = MLDictionaryFeatureProvider(dictionary: ["input": array])
                let out = try model.prediction(from: input)
                if let vec = out.featureValue(for: "output")?.multiArrayValue, vec.count >= 2 {
                    let lift = vec[0].floatValue
                    let drag = vec[1].floatValue
                    return LiftDrag(lift: lift, drag: drag)
                }
            } catch {
                // fall through to heuristic
            }
        }
        // Heuristic fallback (thin foil + simple drag polar)
        let aoaRad = max(0, min(Defaults.aoaDegreesRange.upperBound, aoaDegrees)) * .pi / 180
        let rakeFactor: Float = 1.0 + Float(rakeDegrees / 90.0) * 0.05
        let foilGain: Float = (foil == .vector32) ? 1.12 : 1.0 // ~12% lift gain
        let reFactor: Float = min(1.2, max(0.8, pow(reynolds / 1_000_000, 0.06)))
        let cl = Float(2.0 * .pi) * aoaRad * rakeFactor * foilGain * reFactor
        let cd0: Float = 0.010
        let k: Float = 0.090
        let cd = cd0 + k * cl * cl
        // Scaled outputs to convenient units
        return LiftDrag(lift: cl, drag: cd)
    }
}