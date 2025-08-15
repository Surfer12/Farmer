import CoreML
import Foundation

// MARK: - Core ML Model Interface
class FinPredictor: ObservableObject {
    private var model: MLModel?
    private let modelName = "FinCFDModel"
    
    // CFD Parameters
    struct CFDInput {
        let angleOfAttack: Float    // 0-20 degrees
        let rake: Float            // 6.5 degrees for Vector 3/2
        let reynoldsNumber: Float  // 1e5 to 1e6
        let velocity: Float        // m/s
        let finArea: Float         // sq.in.
    }
    
    struct CFDOutput {
        let lift: Float           // Lift force (N)
        let drag: Float           // Drag force (N)
        let liftCoefficient: Float
        let dragCoefficient: Float
        let pressureDistribution: [Float]
        let velocityField: [Float]
    }
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            print("❌ Core ML model not found. Using analytical approximation.")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("✅ Core ML CFD model loaded successfully")
        } catch {
            print("❌ Failed to load Core ML model: \(error)")
        }
    }
    
    // MARK: - Primary Prediction Interface
    func predictLiftDrag(input: CFDInput) -> CFDOutput? {
        if let mlModel = model {
            return predictWithCoreML(input: input)
        } else {
            return predictWithAnalyticalModel(input: input)
        }
    }
    
    // MARK: - Core ML Prediction
    private func predictWithCoreML(input: CFDInput) -> CFDOutput? {
        do {
            // Prepare input for Core ML model
            let inputArray = try MLMultiArray(shape: [1, 5], dataType: .float32)
            inputArray[0] = NSNumber(value: input.angleOfAttack)
            inputArray[1] = NSNumber(value: input.rake)
            inputArray[2] = NSNumber(value: input.reynoldsNumber)
            inputArray[3] = NSNumber(value: input.velocity)
            inputArray[4] = NSNumber(value: input.finArea)
            
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
            let prediction = try model!.prediction(from: inputFeatures)
            
            // Extract outputs
            if let outputArray = prediction.featureValue(for: "output")?.multiArrayValue {
                let lift = Float(truncating: outputArray[0])
                let drag = Float(truncating: outputArray[1])
                let cl = Float(truncating: outputArray[2])
                let cd = Float(truncating: outputArray[3])
                
                // Generate pressure and velocity fields
                let pressureField = generatePressureField(from: prediction)
                let velocityField = generateVelocityField(from: prediction)
                
                return CFDOutput(
                    lift: lift,
                    drag: drag,
                    liftCoefficient: cl,
                    dragCoefficient: cd,
                    pressureDistribution: pressureField,
                    velocityField: velocityField
                )
            }
        } catch {
            print("❌ Core ML prediction failed: \(error)")
        }
        
        return nil
    }
    
    // MARK: - Analytical Backup Model
    private func predictWithAnalyticalModel(input: CFDInput) -> CFDOutput {
        // Vector 3/2 foil characteristics
        let aspectRatio: Float = 4.0  // Typical for surfboard fins
        let efficiency: Float = 0.85  // Foil efficiency factor
        
        // Convert angle to radians
        let aoaRad = input.angleOfAttack * .pi / 180
        
        // Lift coefficient calculation (thin airfoil theory + corrections)
        let clAlpha: Float = 2 * .pi / (1 + 2 / aspectRatio) // Lift curve slope
        let cl = clAlpha * aoaRad * efficiency
        
        // Drag coefficient (induced + profile drag)
        let inducedDrag = cl * cl / (.pi * aspectRatio * efficiency)
        let profileDrag: Float = 0.008 + 0.003 * aoaRad * aoaRad // Empirical
        let cd = inducedDrag + profileDrag
        
        // Dynamic pressure
        let dynamicPressure = 0.5 * 1025 * input.velocity * input.velocity // Seawater density
        
        // Forces calculation
        let areaM2 = input.finArea * 0.00064516 // Convert sq.in. to m²
        let lift = cl * dynamicPressure * areaM2
        let drag = cd * dynamicPressure * areaM2
        
        // Generate synthetic pressure and velocity fields
        let pressureField = generateAnalyticalPressureField(input: input, cl: cl)
        let velocityField = generateAnalyticalVelocityField(input: input)
        
        return CFDOutput(
            lift: lift,
            drag: drag,
            liftCoefficient: cl,
            dragCoefficient: cd,
            pressureDistribution: pressureField,
            velocityField: velocityField
        )
    }
    
    // MARK: - Field Generation
    private func generatePressureField(from prediction: MLFeatureProvider) -> [Float] {
        // Extract pressure field from Core ML output if available
        if let pressureArray = prediction.featureValue(for: "pressure_field")?.multiArrayValue {
            var field: [Float] = []
            for i in 0..<pressureArray.count {
                field.append(Float(truncating: pressureArray[i]))
            }
            return field
        }
        
        // Fallback to analytical generation
        return Array(repeating: 0.0, count: 300)
    }
    
    private func generateVelocityField(from prediction: MLFeatureProvider) -> [Float] {
        // Extract velocity field from Core ML output if available
        if let velocityArray = prediction.featureValue(for: "velocity_field")?.multiArrayValue {
            var field: [Float] = []
            for i in 0..<velocityArray.count {
                field.append(Float(truncating: velocityArray[i]))
            }
            return field
        }
        
        // Fallback to analytical generation
        return Array(repeating: 1.0, count: 300)
    }
    
    private func generateAnalyticalPressureField(input: CFDInput, cl: Float) -> [Float] {
        var pressureField: [Float] = []
        let gridSize = 20
        let gridHeight = 15
        
        for j in 0..<gridHeight {
            for i in 0..<gridSize {
                let x = Float(i - gridSize/2) * 0.5  // Grid coordinates
                let y = Float(j - gridHeight/2) * 0.4
                
                // Distance from fin centerline
                let distance = sqrt(x * x + y * y)
                
                // Pressure calculation based on potential flow + corrections
                let baseP = -0.5 * 1025 * input.velocity * input.velocity
                
                // Circulation effect (lift generation)
                let circulation = cl * input.velocity * sqrt(input.finArea * 0.00064516)
                let circulationP = circulation / (2 * .pi * max(distance, 0.1))
                
                // Angle of attack influence
                let aoaEffect = sin(input.angleOfAttack * .pi / 180) * x * 0.15
                
                // Distance decay
                let distanceDecay = exp(-distance / 2.0)
                
                let pressure = baseP * distanceDecay + circulationP + aoaEffect
                pressureField.append(pressure)
            }
        }
        
        return pressureField
    }
    
    private func generateAnalyticalVelocityField(input: CFDInput) -> [Float] {
        var velocityField: [Float] = []
        let gridSize = 20
        let gridHeight = 15
        
        for j in 0..<gridHeight {
            for i in 0..<gridSize {
                let x = Float(i - gridSize/2) * 0.5
                let y = Float(j - gridHeight/2) * 0.4
                
                // Distance from fin
                let distance = sqrt(x * x + y * y)
                
                // Velocity magnitude calculation
                let freestream = input.velocity
                let perturbation = freestream * 0.3 * exp(-distance / 1.5)
                let aoaInfluence = sin(input.angleOfAttack * .pi / 180) * 0.2
                
                let velocity = freestream + perturbation + aoaInfluence
                velocityField.append(max(0.1, velocity))
            }
        }
        
        return velocityField
    }
    
    // MARK: - Advanced CFD Analysis
    func analyzeBoundaryLayer(input: CFDInput) -> BoundaryLayerData {
        let re = input.reynoldsNumber
        
        // Boundary layer thickness estimation (Blasius solution)
        let deltaX = 5.0 * sqrt(input.finArea * 0.00064516) / sqrt(re) // Characteristic length
        
        // Transition point estimation
        let transitionRe: Float = 500000 // Critical Reynolds number
        let isTransitional = re > transitionRe
        
        // Skin friction coefficient
        let cf = isTransitional ? 
            0.074 / pow(re, 0.2) :  // Turbulent
            1.328 / sqrt(re)        // Laminar
        
        return BoundaryLayerData(
            thickness: deltaX,
            isTransitional: isTransitional,
            skinFrictionCoeff: cf,
            separationPoint: estimateSeparationPoint(aoa: input.angleOfAttack)
        )
    }
    
    private func estimateSeparationPoint(aoa: Float) -> Float {
        // Simplified separation estimation for Vector 3/2 foil
        let criticalAOA: Float = 12.0 // Degrees
        if aoa > criticalAOA {
            return max(0.3, 0.8 - (aoa - criticalAOA) * 0.05)
        }
        return 0.95 // No separation
    }
    
    // MARK: - k-ω SST Turbulence Model Approximation
    func estimateTurbulenceParameters(input: CFDInput) -> TurbulenceData {
        let re = input.reynoldsNumber
        let velocity = input.velocity
        
        // Turbulent kinetic energy estimation
        let turbulenceIntensity: Float = re > 500000 ? 0.05 : 0.02
        let k = 1.5 * pow(velocity * turbulenceIntensity, 2)
        
        // Specific dissipation rate
        let cmu: Float = 0.09
        let lt = 0.1 * sqrt(input.finArea * 0.00064516) // Turbulent length scale
        let omega = sqrt(k) / (pow(cmu, 0.25) * lt)
        
        return TurbulenceData(
            kineticEnergy: k,
            dissipationRate: omega,
            turbulenceIntensity: turbulenceIntensity,
            viscosityRatio: min(100, k / (omega * 1e-6)) // Turbulent/molecular viscosity
        )
    }
}

// MARK: - Supporting Data Structures
struct BoundaryLayerData {
    let thickness: Float
    let isTransitional: Bool
    let skinFrictionCoeff: Float
    let separationPoint: Float
}

struct TurbulenceData {
    let kineticEnergy: Float
    let dissipationRate: Float
    let turbulenceIntensity: Float
    let viscosityRatio: Float
}

// MARK: - Utility Extensions
extension FinPredictor {
    func calculateLiftToDragRatio(lift: Float, drag: Float) -> Float {
        guard drag > 0.001 else { return 0 }
        return lift / drag
    }
    
    func estimateStallAngle(finArea: Float, aspectRatio: Float) -> Float {
        // Empirical stall angle estimation for Vector 3/2 foil
        let baseStallAngle: Float = 14.0
        let aspectRatioCorrection = min(2.0, max(0.5, aspectRatio / 4.0))
        return baseStallAngle * aspectRatioCorrection
    }
    
    func calculateEfficiency(lift: Float, drag: Float, velocity: Float, power: Float) -> Float {
        let thrust = lift * sin(atan2(drag, lift))
        return (thrust * velocity) / max(power, 0.1)
    }
}