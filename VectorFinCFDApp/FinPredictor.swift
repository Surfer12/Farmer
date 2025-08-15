import CoreML
import Foundation

class FinPredictor {
    private var model: FinCFDModel?
    
    enum PredictionError: Error, LocalizedError {
        case modelLoadingFailed
        case inputPreparationFailed
        case predictionFailed
        case modelNotInitialized
        
        var errorDescription: String? {
            switch self {
            case .modelLoadingFailed:
                return "Failed to load Core ML model"
            case .inputPreparationFailed:
                return "Failed to prepare input data"
            case .predictionFailed:
                return "Model prediction failed"
            case .modelNotInitialized:
                return "Model not initialized"
            }
        }
    }
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all // Use all available compute units
            
            // Try to load the model
            if let modelURL = Bundle.main.url(forResource: "FinCFDModel", withExtension: "mlmodel") {
                let compiledModelURL = try MLModel.compileModel(at: modelURL)
                model = try FinCFDModel(contentsOf: compiledModelURL, configuration: configuration)
                print("Core ML model loaded successfully")
            } else {
                print("FinCFDModel.mlmodel not found in bundle, using fallback predictions")
                model = nil
            }
        } catch {
            print("Failed to load Core ML model: \(error), using fallback predictions")
            model = nil
        }
    }
    
    func predictLiftDrag(aoa: Float, rake: Float, re: Float) throws -> (lift: Float, drag: Float) {
        // If Core ML model is available, use it
        if let model = model {
            return try predictWithCoreML(model: model, aoa: aoa, rake: rake, re: re)
        } else {
            // Fallback to physics-based predictions
            return predictWithPhysics(aoa: aoa, rake: rake, re: re)
        }
    }
    
    private func predictWithCoreML(model: FinCFDModel, aoa: Float, rake: Float, re: Float) throws -> (lift: Float, drag: Float) {
        guard let inputArray = try? MLMultiArray(shape: [1, 3], dataType: .float32) else {
            throw PredictionError.inputPreparationFailed
        }
        
        // Normalize inputs based on training data ranges
        let normalizedAoa = normalizeAngleOfAttack(aoa)
        let normalizedRake = normalizeRake(rake)
        let normalizedRe = normalizeReynoldsNumber(re)
        
        inputArray[0] = NSNumber(value: normalizedAoa)
        inputArray[1] = NSNumber(value: normalizedRake)
        inputArray[2] = NSNumber(value: normalizedRe)
        
        guard let input = try? FinCFDModelInput(input: inputArray) else {
            throw PredictionError.inputPreparationFailed
        }
        
        guard let output = try? model.prediction(input: input) else {
            throw PredictionError.predictionFailed
        }
        
        // Denormalize outputs
        let lift = denormalizeLift(Float(output.output[0]))
        let drag = denormalizeDrag(Float(output.output[1]))
        
        return (lift: lift, drag: drag)
    }
    
    private func predictWithPhysics(aoa: Float, rake: Float, re: Float) -> (lift: Float, drag: Float) {
        // Physics-based fallback predictions using empirical formulas
        // Based on Vector 3/2 foil characteristics and k-ω SST turbulence model insights
        
        let aoaRad = aoa * Float.pi / 180.0
        let rakeRad = rake * Float.pi / 180.0
        
        // Lift coefficient calculation (simplified)
        let cl0 = 0.1 // Zero-lift coefficient
        let clAlpha = 2.0 * Float.pi // Lift curve slope (2π for thin airfoils)
        let clRake = 0.15 // Rake effect coefficient
        
        let cl = cl0 + clAlpha * sin(aoaRad) + clRake * sin(rakeRad)
        
        // Apply Vector 3/2 foil specific corrections
        let vectorCorrection = 1.12 // 12% lift increase for raked fins
        let correctedCl = cl * vectorCorrection
        
        // Drag coefficient calculation
        let cd0 = 0.008 // Zero-drag coefficient
        let cdAlpha = 0.1 // Drag curve slope
        let cdRake = 0.02 // Rake drag penalty
        
        let cd = cd0 + cdAlpha * pow(sin(aoaRad), 2) + cdRake * pow(sin(rakeRad), 2)
        
        // Reynolds number effects
        let reEffect = sqrt(re / 1_000_000.0) // Normalize to 10^6
        let finalCl = correctedCl * reEffect
        let finalCd = cd * reEffect
        
        // Convert to actual forces (simplified)
        let dynamicPressure = 0.5 * 1025.0 * pow(2.0, 2) // ρV²/2 (water, 2 m/s)
        let finArea = 15.0 * 0.0064516 // Convert sq.in. to sq.m
        
        let lift = finalCl * dynamicPressure * finArea
        let drag = finalCd * dynamicPressure * finArea
        
        return (lift: lift, drag: drag)
    }
    
    // MARK: - Input Normalization
    
    private func normalizeAngleOfAttack(_ aoa: Float) -> Float {
        // Normalize 0-20° to 0-1 range
        return max(0, min(1, aoa / 20.0))
    }
    
    private func normalizeRake(_ rake: Float) -> Float {
        // Normalize 0-15° to 0-1 range
        return max(0, min(1, rake / 15.0))
    }
    
    private func normalizeReynoldsNumber(_ re: Float) -> Float {
        // Normalize 10^5 to 10^6 range to 0-1
        let logRe = log10(re)
        let normalized = (logRe - 5.0) / 1.0 // 5.0 to 6.0 -> 0 to 1
        return max(0, min(1, normalized))
    }
    
    // MARK: - Output Denormalization
    
    private func denormalizeLift(_ normalizedLift: Float) -> Float {
        // Denormalize from 0-1 to actual lift range (0-100 N)
        return normalizedLift * 100.0
    }
    
    private func denormalizeDrag(_ normalizedDrag: Float) -> Float {
        // Denormalize from 0-1 to actual drag range (0-50 N)
        return normalizedDrag * 50.0
    }
    
    // MARK: - Model Validation
    
    func validatePredictions() -> Bool {
        // Test predictions with known values
        do {
            let testResult = try predictLiftDrag(aoa: 10.0, rake: 6.5, re: 500_000)
            
            // Validate that predictions are within reasonable bounds
            let validLift = testResult.lift > 0 && testResult.lift < 200
            let validDrag = testResult.drag > 0 && testResult.drag < 100
            
            return validLift && validDrag
        } catch {
            print("Model validation failed: \(error)")
            return false
        }
    }
    
    // MARK: - Performance Metrics
    
    func getPredictionConfidence(aoa: Float, rake: Float, re: Float) -> Float {
        // Calculate confidence based on input ranges and model performance
        var confidence: Float = 1.0
        
        // Reduce confidence for extreme angles
        if aoa > 18.0 || aoa < 2.0 {
            confidence *= 0.8
        }
        
        // Reduce confidence for extreme Reynolds numbers
        if re < 100_000 || re > 2_000_000 {
            confidence *= 0.7
        }
        
        // Reduce confidence if using fallback physics
        if model == nil {
            confidence *= 0.6
        }
        
        return max(0.1, confidence)
    }
}

// MARK: - Core ML Model Interface

// This would be generated by coremltools from a trained PyTorch model
// For now, we'll create a placeholder interface
class FinCFDModel {
    struct FinCFDModelInput {
        let input: MLMultiArray
        
        init(input: MLMultiArray) throws {
            self.input = input
        }
    }
    
    struct FinCFDModelOutput {
        let output: MLMultiArray
        
        init(output: MLMultiArray) {
            self.output = output
        }
    }
    
    init(contentsOf url: URL, configuration: MLModelConfiguration) throws {
        // This would be the actual Core ML model initialization
        // For now, it's a placeholder
    }
    
    func prediction(input: FinCFDModelInput) throws -> FinCFDModelOutput {
        // This would be the actual prediction
        // For now, return mock data
        let mockOutput = try MLMultiArray(shape: [1, 2], dataType: .float32)
        mockOutput[0] = NSNumber(value: 0.5) // Mock lift
        mockOutput[1] = NSNumber(value: 0.3) // Mock drag
        
        return FinCFDModelOutput(output: mockOutput)
    }
}