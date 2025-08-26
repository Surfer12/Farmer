import CoreML
import Foundation

// MARK: - Core ML Predictor

class FinPredictor: ObservableObject {
    
    // MARK: - Properties
    
    private var model: FinCFDModel?
    private let modelConfiguration: MLModelConfiguration
    
    @Published var isModelLoaded = false
    @Published var lastPrediction: LiftDragPrediction?
    @Published var predictionHistory: [LiftDragPrediction] = []
    
    // Performance metrics
    private var predictionCount = 0
    private var averageInferenceTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    init() {
        modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .all // Use GPU when available
        loadModel()
    }
    
    // MARK: - Model Loading
    
    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                // Try to load the Core ML model
                self?.model = try FinCFDModel(configuration: self?.modelConfiguration ?? MLModelConfiguration())
                
                DispatchQueue.main.async {
                    self?.isModelLoaded = true
                    print("✅ FinCFDModel loaded successfully")
                }
            } catch {
                print("❌ Failed to load FinCFDModel: \(error)")
                
                // Fall back to mathematical model if Core ML model is not available
                DispatchQueue.main.async {
                    self?.isModelLoaded = true // Still mark as loaded to use fallback
                    print("🔄 Using mathematical fallback model")
                }
            }
        }
    }
    
    // MARK: - Prediction Methods
    
    func predictLiftDrag(
        angleOfAttack: Double,
        rake: Double,
        reynoldsNumber: Double,
        finSpec: FinSpecification
    ) async throws -> LiftDragPrediction {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let prediction: LiftDragPrediction
        
        if let model = model {
            // Use Core ML model
            prediction = try await performCoreMLPrediction(
                aoa: angleOfAttack,
                rake: rake,
                re: reynoldsNumber,
                finSpec: finSpec,
                model: model
            )
        } else {
            // Use mathematical fallback model
            prediction = performMathematicalPrediction(
                aoa: angleOfAttack,
                rake: rake,
                re: reynoldsNumber,
                finSpec: finSpec
            )
        }
        
        let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
        updatePerformanceMetrics(inferenceTime: inferenceTime)
        
        DispatchQueue.main.async { [weak self] in
            self?.lastPrediction = prediction
            self?.predictionHistory.append(prediction)
            
            // Keep only last 100 predictions for performance
            if self?.predictionHistory.count ?? 0 > 100 {
                self?.predictionHistory.removeFirst()
            }
        }
        
        return prediction
    }
    
    private func performCoreMLPrediction(
        aoa: Double,
        rake: Double,
        re: Double,
        finSpec: FinSpecification,
        model: FinCFDModel
    ) async throws -> LiftDragPrediction {
        
        // Prepare input data
        guard let inputArray = try? MLMultiArray(shape: [1, 5], dataType: .float32) else {
            throw PredictionError.inputPreparationFailed
        }
        
        // Normalize inputs for neural network
        inputArray[0] = NSNumber(value: aoa / 20.0) // Normalize AoA (0-20°)
        inputArray[1] = NSNumber(value: rake / 10.0) // Normalize rake
        inputArray[2] = NSNumber(value: (re - 100000) / 900000) // Normalize Re
        inputArray[3] = NSNumber(value: finSpec.area / 20.0) // Normalize area
        inputArray[4] = NSNumber(value: finSpec.foilType == .vector32 ? 1.0 : 0.0) // Foil type
        
        guard let input = try? FinCFDModelInput(input: inputArray) else {
            throw PredictionError.inputPreparationFailed
        }
        
        guard let output = try? model.prediction(input: input) else {
            throw PredictionError.predictionFailed
        }
        
        // Extract and denormalize outputs
        let lift = Double(output.output[0]) * 100.0 // Denormalize lift
        let drag = Double(output.output[1]) * 50.0  // Denormalize drag
        let pressureCoeff = Double(output.output[2])
        
        return LiftDragPrediction(
            timestamp: Date(),
            angleOfAttack: aoa,
            rake: rake,
            reynoldsNumber: re,
            finSpecification: finSpec,
            lift: lift,
            drag: drag,
            pressureCoefficient: pressureCoeff,
            liftToDragRatio: drag != 0 ? lift / drag : 0,
            flowRegime: determineFlowRegime(aoa: aoa),
            confidence: 0.95, // Core ML model confidence
            modelType: .coreML
        )
    }
    
    private func performMathematicalPrediction(
        aoa: Double,
        rake: Double,
        re: Double,
        finSpec: FinSpecification
    ) -> LiftDragPrediction {
        
        // Mathematical model based on CFD equations and empirical data
        let aoaRad = aoa * .pi / 180
        
        // Lift calculation using thin airfoil theory with corrections
        let liftSlope = calculateLiftSlope(finSpec: finSpec, re: re)
        var lift = liftSlope * aoaRad * finSpec.area
        
        // Apply rake correction (12% increase for Vector 3/2)
        if finSpec.rake > 0 && finSpec.foilType == .vector32 {
            lift *= 1.12
        }
        
        // Drag calculation using quadratic model
        let inducedDragCoeff = calculateInducedDragCoeff(lift: lift, finSpec: finSpec)
        let profileDragCoeff = calculateProfileDragCoeff(finSpec: finSpec, re: re, aoa: aoa)
        let totalDragCoeff = inducedDragCoeff + profileDragCoeff
        
        let drag = totalDragCoeff * finSpec.area
        
        // Pressure coefficient estimation
        let pressureCoeff = calculatePressureCoefficient(aoa: aoa, finSpec: finSpec)
        
        return LiftDragPrediction(
            timestamp: Date(),
            angleOfAttack: aoa,
            rake: rake,
            reynoldsNumber: re,
            finSpecification: finSpec,
            lift: lift,
            drag: drag,
            pressureCoefficient: pressureCoeff,
            liftToDragRatio: drag != 0 ? lift / drag : 0,
            flowRegime: determineFlowRegime(aoa: aoa),
            confidence: 0.85, // Mathematical model confidence
            modelType: .mathematical
        )
    }
    
    // MARK: - Mathematical Model Helpers
    
    private func calculateLiftSlope(finSpec: FinSpecification, re: Double) -> Double {
        // Lift slope based on foil type and Reynolds number
        var baseLiftSlope: Double
        
        switch finSpec.foilType {
        case .vector32:
            baseLiftSlope = 6.2 // rad^-1
        case .symmetric:
            baseLiftSlope = 5.8
        case .scimitarTip:
            baseLiftSlope = 6.0
        }
        
        // Reynolds number correction
        let reCorrection = min(1.0, re / 1_000_000)
        return baseLiftSlope * reCorrection
    }
    
    private func calculateInducedDragCoeff(lift: Double, finSpec: FinSpecification) -> Double {
        // Induced drag coefficient calculation
        let aspectRatio = calculateAspectRatio(finSpec: finSpec)
        let liftCoeff = lift / finSpec.area
        
        return (liftCoeff * liftCoeff) / (.pi * aspectRatio * 0.85) // 0.85 is efficiency factor
    }
    
    private func calculateProfileDragCoeff(finSpec: FinSpecification, re: Double, aoa: Double) -> Double {
        // Profile drag coefficient based on foil type and conditions
        var baseDragCoeff: Double
        
        switch finSpec.foilType {
        case .vector32:
            baseDragCoeff = 0.008
        case .symmetric:
            baseDragCoeff = 0.010
        case .scimitarTip:
            baseDragCoeff = 0.007
        }
        
        // Reynolds number effect
        let reEffect = 1.0 - (log10(re) - 5.0) * 0.1
        
        // Angle of attack effect (drag increases with AoA squared)
        let aoaEffect = 1.0 + (aoa / 20.0) * (aoa / 20.0) * 0.5
        
        return baseDragCoeff * reEffect * aoaEffect
    }
    
    private func calculateAspectRatio(finSpec: FinSpecification) -> Double {
        // Estimate aspect ratio based on fin area
        let chord = sqrt(finSpec.area * 0.8) // Approximate chord length
        let span = finSpec.area / chord
        return span * span / finSpec.area
    }
    
    private func calculatePressureCoefficient(aoa: Double, finSpec: FinSpecification) -> Double {
        // Pressure coefficient calculation based on angle of attack
        let aoaRad = aoa * .pi / 180
        
        // Base pressure coefficient
        var baseCp = -2.0 * sin(aoaRad)
        
        // Foil type correction
        switch finSpec.foilType {
        case .vector32:
            baseCp *= 1.1 // Enhanced pressure differential
        case .symmetric:
            baseCp *= 1.0
        case .scimitarTip:
            baseCp *= 0.9
        }
        
        return baseCp
    }
    
    private func determineFlowRegime(aoa: Double) -> FlowRegime {
        switch aoa {
        case 0...10:
            return .laminar
        case 10...15:
            return .transitional
        default:
            return .turbulent
        }
    }
    
    // MARK: - Batch Predictions
    
    func predictLiftDragSweep(
        aoaRange: ClosedRange<Double>,
        steps: Int,
        rake: Double,
        reynoldsNumber: Double,
        finSpec: FinSpecification
    ) async throws -> [LiftDragPrediction] {
        
        let aoaStep = (aoaRange.upperBound - aoaRange.lowerBound) / Double(steps - 1)
        var predictions: [LiftDragPrediction] = []
        
        for i in 0..<steps {
            let aoa = aoaRange.lowerBound + Double(i) * aoaStep
            
            do {
                let prediction = try await predictLiftDrag(
                    angleOfAttack: aoa,
                    rake: rake,
                    reynoldsNumber: reynoldsNumber,
                    finSpec: finSpec
                )
                predictions.append(prediction)
            } catch {
                print("Failed to predict for AoA \(aoa): \(error)")
                // Continue with next angle
            }
        }
        
        return predictions
    }
    
    // MARK: - Performance Monitoring
    
    private func updatePerformanceMetrics(inferenceTime: TimeInterval) {
        predictionCount += 1
        averageInferenceTime = ((averageInferenceTime * Double(predictionCount - 1)) + inferenceTime) / Double(predictionCount)
    }
    
    func getPerformanceMetrics() -> PredictionPerformanceMetrics {
        return PredictionPerformanceMetrics(
            totalPredictions: predictionCount,
            averageInferenceTime: averageInferenceTime,
            modelType: model != nil ? .coreML : .mathematical,
            isModelLoaded: isModelLoaded
        )
    }
    
    // MARK: - Utility Methods
    
    func clearHistory() {
        predictionHistory.removeAll()
        predictionCount = 0
        averageInferenceTime = 0
    }
    
    func exportPredictions() -> Data? {
        do {
            return try JSONEncoder().encode(predictionHistory)
        } catch {
            print("Failed to export predictions: \(error)")
            return nil
        }
    }
}

// MARK: - Supporting Types

struct LiftDragPrediction: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let angleOfAttack: Double
    let rake: Double
    let reynoldsNumber: Double
    let finSpecification: FinSpecification
    let lift: Double
    let drag: Double
    let pressureCoefficient: Double
    let liftToDragRatio: Double
    let flowRegime: FlowRegime
    let confidence: Double
    let modelType: ModelType
    
    var efficiency: Double {
        return liftToDragRatio
    }
    
    var isOptimal: Bool {
        return liftToDragRatio > 10.0 && flowRegime == .laminar
    }
}

enum ModelType: String, Codable {
    case coreML = "Core ML"
    case mathematical = "Mathematical"
}

struct PredictionPerformanceMetrics {
    let totalPredictions: Int
    let averageInferenceTime: TimeInterval
    let modelType: ModelType
    let isModelLoaded: Bool
}

enum PredictionError: Error, LocalizedError {
    case modelLoadingFailed
    case inputPreparationFailed
    case predictionFailed
    case invalidParameters
    
    var errorDescription: String? {
        switch self {
        case .modelLoadingFailed:
            return "Failed to load the Core ML model"
        case .inputPreparationFailed:
            return "Failed to prepare input data for prediction"
        case .predictionFailed:
            return "Prediction failed during inference"
        case .invalidParameters:
            return "Invalid input parameters provided"
        }
    }
}

// MARK: - Mock Core ML Model Interface

// This would normally be generated by Xcode when you add the .mlmodel file
// For now, we'll create a mock interface

class FinCFDModel {
    init(configuration: MLModelConfiguration) throws {
        // Mock initialization
        // In a real implementation, this would load the actual .mlmodel file
    }
    
    func prediction(input: FinCFDModelInput) throws -> FinCFDModelOutput {
        // Mock prediction - replace with actual Core ML inference
        let lift = Float.random(in: 10...80)
        let drag = Float.random(in: 2...15)
        let pressure = Float.random(in: -2...0)
        
        return FinCFDModelOutput(output: MLMultiArray(arrayLiteral: lift, drag, pressure))
    }
}

class FinCFDModelInput {
    let input: MLMultiArray
    
    init(input: MLMultiArray) {
        self.input = input
    }
}

class FinCFDModelOutput {
    let output: MLMultiArray
    
    init(output: MLMultiArray) {
        self.output = output
    }
}

// MARK: - MLMultiArray Extension for Mock Data

extension MLMultiArray {
    convenience init(arrayLiteral elements: Float...) {
        try! self.init(shape: [NSNumber(value: elements.count)], dataType: .float32)
        for (index, element) in elements.enumerated() {
            self[index] = NSNumber(value: element)
        }
    }
}