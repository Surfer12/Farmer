import Foundation
import CoreML

struct FinPredictionResult {
    let lift: Double
    let drag: Double
}

final class FinPredictor {
    private let model: MLModel?

    init() {
        // Try to load a compiled model first
        if let compiledURL = Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: compiledURL)
        } else if let sourceURL = Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodel"),
                  let compiledURL = try? MLModel.compileModel(at: sourceURL) {
            model = try? MLModel(contentsOf: compiledURL)
        } else {
            model = nil
        }
    }

    func predictLiftDrag(aoaDegrees: Double, rakeDegrees: Double, reynolds: Double) -> FinPredictionResult {
        if let model,
           let inputArray = try? MLMultiArray(shape: [3], dataType: .double) {
            inputArray[0] = NSNumber(value: aoaDegrees)
            inputArray[1] = NSNumber(value: rakeDegrees)
            inputArray[2] = NSNumber(value: reynolds)
            let provider = try? MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: inputArray)])
            if let provider,
               let prediction = try? model.prediction(from: provider),
               let outputArray = prediction.featureValue(for: "output")?.multiArrayValue,
               outputArray.count >= 2 {
                let lift = outputArray[0].doubleValue
                let drag = outputArray[1].doubleValue
                return FinPredictionResult(lift: lift, drag: drag)
            }
        }
        return physicsFallback(aoaDegrees: aoaDegrees, rakeDegrees: rakeDegrees, reynolds: reynolds)
    }

    private func physicsFallback(aoaDegrees: Double, rakeDegrees: Double, reynolds: Double) -> FinPredictionResult {
        let aoaRad = aoaDegrees * .pi / 180.0
        let rakeRad = rakeDegrees * .pi / 180.0
        let baseLift = 2.0 * .pi * aoaRad
        let rakeFactor = max(0.82, 1.0 - 0.08 * (rakeRad * 180.0 / .pi) / 10.0)
        let stallFactor: Double
        if aoaDegrees <= 15.0 {
            stallFactor = 1.0
        } else if aoaDegrees >= 20.0 {
            stallFactor = 0.6
        } else {
            stallFactor = 1.0 - 0.4 * ((aoaDegrees - 15.0) / 5.0)
        }
        let cl = baseLift * rakeFactor * stallFactor
        let kDrag = 0.01
        let cdProfile = 0.008 + 0.02 * pow(1e6 / max(1e5, reynolds), 0.25)
        let cdInduced = kDrag * cl * cl
        let cd = cdProfile + cdInduced
        return FinPredictionResult(lift: cl, drag: cd)
    }
}