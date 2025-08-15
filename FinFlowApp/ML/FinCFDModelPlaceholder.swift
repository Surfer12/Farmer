import Foundation

// Placeholder types to allow compilation WITHOUT the real Core ML model.
// Enable with the Swift flag: -DUSE_PLACEHOLDER_MLMODEL
// Remove or disable this file once `FinCFDModel.mlmodel` is added to the project.

#if USE_PLACEHOLDER_MLMODEL
import CoreML
public struct FinCFDModelInput : MLFeatureProvider {
	public var input: MLMultiArray
	public var featureNames: Set<String> { ["input"] }
	public func featureValue(for featureName: String) -> MLFeatureValue? {
		featureName == "input" ? MLFeatureValue(multiArray: input) : nil
	}
}

public class FinCFDModel {
	public init(configuration: MLModelConfiguration) throws {}
	public func prediction(input: FinCFDModelInput) throws -> (output: [NSNumber]) {
		// Dummy; real model will replace this.
		let alpha = Double(truncating: input.input[0])
		let cl = 2.2 * alpha * .pi / 180
		let cd = 0.012 + 0.95 * cl * cl
		return ([NSNumber(value: cl), NSNumber(value: cd)])
	}
}
#endif