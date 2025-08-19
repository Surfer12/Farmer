import Foundation
import CoreML

final class FinPredictor {
	private let model: MLModel

	init?() {
		guard let url = Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodelc") ?? Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodel") else {
			return nil
		}
		do {
			if url.pathExtension == "mlmodel" {
				let compiled = try MLModel.compileModel(at: url)
				model = try MLModel(contentsOf: compiled)
			} else {
				model = try MLModel(contentsOf: url)
			}
		} catch {
			return nil
		}
	}

	func predictLiftDrag(aoa: Float, rake: Float, re: Float) -> (lift: Float, drag: Float)? {
		let values: [Float] = [aoa, rake, re]
		guard let inputArray = try? MLMultiArray(shape: [3], dataType: .float32) else { return nil }
		for (i, v) in values.enumerated() { inputArray[i] = NSNumber(value: v) }

		let input: MLFeatureProvider
		if let desc = model.modelDescription.inputDescriptionsByName.first {
			let name = desc.key
			input = try! MLDictionaryFeatureProvider(dictionary: [name: inputArray])
		} else {
			return nil
		}

		guard let out = try? model.prediction(from: input) else { return nil }
		if let first = model.modelDescription.outputDescriptionsByName.first?.key, let arr = out.featureValue(for: first)?.multiArrayValue, arr.count >= 2 {
			let lift = Float(truncating: arr[0])
			let drag = Float(truncating: arr[1])
			return (lift, drag)
		}
		return nil
	}
}