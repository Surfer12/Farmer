import Foundation

/// Represents CFD simulation data for fin performance analysis
struct CFDData {
    let timestamp: Date
    let reynoldsNumber: Float // Re ≈ 10^5–10^6
    let angleOfAttack: Float // 0°–20° AoA range
    let rakeAngle: Float
    
    // Performance coefficients
    let liftCoefficient: Float
    let dragCoefficient: Float
    let pressureDifferential: Float // 30% pressure differential
    
    // Pressure distribution data
    let pressureMap: [Float] // Normalized pressure values (0.0 to 1.0)
    let flowVectors: [FlowVector]
    
    // Flow characteristics
    let flowRegime: FlowRegime
    let turbulenceIntensity: Float
    
    enum FlowRegime {
        case laminar(quality: Float)
        case transitional(quality: Float)
        case turbulent(quality: Float)
        case separated(quality: Float)
        
        var description: String {
            switch self {
            case .laminar(let quality):
                return "Laminar (Q: \(String(format: "%.2f", quality)))"
            case .transitional(let quality):
                return "Transitional (Q: \(String(format: "%.2f", quality)))"
            case .turbulent(let quality):
                return "Turbulent (Q: \(String(format: "%.2f", quality)))"
            case .separated(let quality):
                return "Separated (Q: \(String(format: "%.2f", quality)))"
            }
        }
    }
    
    init(reynoldsNumber: Float, angleOfAttack: Float, rakeAngle: Float) {
        self.timestamp = Date()
        self.reynoldsNumber = reynoldsNumber
        self.angleOfAttack = angleOfAttack
        self.rakeAngle = rakeAngle
        
        // Calculate performance coefficients based on input parameters
        let geometry = rakeAngle > 0 ? FinGeometry.sideFin : FinGeometry.centerFin
        self.liftCoefficient = geometry.calculateLiftCoefficient(angleOfAttack: angleOfAttack)
        self.dragCoefficient = geometry.calculateDragCoefficient(angleOfAttack: angleOfAttack)
        
        // Calculate pressure differential (30% for Vector 3/2)
        self.pressureDifferential = 0.30 * abs(sin(angleOfAttack * Float.pi / 180.0))
        
        // Generate synthetic pressure map data
        self.pressureMap = CFDData.generatePressureMap(aoa: angleOfAttack, rake: rakeAngle)
        
        // Generate flow vectors
        self.flowVectors = CFDData.generateFlowVectors(aoa: angleOfAttack)
        
        // Determine flow regime
        self.flowRegime = CFDData.determineFlowRegime(aoa: angleOfAttack, re: reynoldsNumber)
        
        // Calculate turbulence intensity
        self.turbulenceIntensity = CFDData.calculateTurbulenceIntensity(aoa: angleOfAttack, re: reynoldsNumber)
    }
    
    /// Generates synthetic pressure map data for visualization
    private static func generatePressureMap(aoa: Float, rake: Float) -> [Float] {
        let gridSize = 64 // 64x64 pressure map
        var pressureMap: [Float] = []
        
        for i in 0..<gridSize {
            for j in 0..<gridSize {
                let x = Float(i) / Float(gridSize - 1) // 0 to 1
                let y = Float(j) / Float(gridSize - 1) // 0 to 1
                
                // Create pressure distribution based on airfoil theory
                let chordPosition = x
                let spanPosition = y
                
                // Higher pressure on lower surface, lower on upper surface
                let basePressure = 0.5 + 0.3 * sin(aoa * Float.pi / 180.0) * (0.5 - spanPosition)
                
                // Add chord-wise variation
                let chordEffect = sin(chordPosition * Float.pi) * 0.2
                
                // Add rake angle effect
                let rakeEffect = rake * 0.01 * (spanPosition - 0.5)
                
                let pressure = max(0.0, min(1.0, basePressure + chordEffect + rakeEffect))
                pressureMap.append(pressure)
            }
        }
        
        return pressureMap
    }
    
    /// Generates flow vectors for visualization
    private static func generateFlowVectors(aoa: Float) -> [FlowVector] {
        var vectors: [FlowVector] = []
        let numVectors = 100
        
        for i in 0..<numVectors {
            let x = Float.random(in: -2...6) // Around fin area
            let y = Float.random(in: -2...2)
            let z = Float.random(in: -1...1)
            
            // Flow direction influenced by angle of attack
            let flowAngle = aoa * Float.pi / 180.0
            let vx = cos(flowAngle) + Float.random(in: -0.1...0.1)
            let vy = sin(flowAngle) + Float.random(in: -0.1...0.1)
            let vz = Float.random(in: -0.05...0.05)
            
            let magnitude = sqrt(vx*vx + vy*vy + vz*vz)
            
            vectors.append(FlowVector(
                position: (x, y, z),
                velocity: (vx, vy, vz),
                magnitude: magnitude
            ))
        }
        
        return vectors
    }
    
    /// Determines flow regime based on AoA and Reynolds number
    private static func determineFlowRegime(aoa: Float, re: Float) -> FlowRegime {
        let absAoa = abs(aoa)
        
        if absAoa < 5.0 && re > 1e5 {
            return .laminar(quality: 0.9)
        } else if absAoa < 10.0 {
            return .transitional(quality: 0.7)
        } else if absAoa < 15.0 {
            return .turbulent(quality: 0.6)
        } else {
            return .separated(quality: 0.3)
        }
    }
    
    /// Calculates turbulence intensity
    private static func calculateTurbulenceIntensity(aoa: Float, re: Float) -> Float {
        let baseIntensity: Float = 0.05 // 5% base turbulence
        let aoaEffect = abs(aoa) * 0.01 // Increase with AoA
        let reEffect = max(0.0, (1e6 - re) / 1e6 * 0.02) // Higher at lower Re
        
        return min(0.3, baseIntensity + aoaEffect + reEffect)
    }
    
    /// Calculates lift and drag forces for a given rider weight
    func calculateForces(riderWeight: Float) -> (lift: Float, drag: Float) {
        // Convert coefficients to forces (simplified)
        let dynamicPressure = 0.5 * 1025.0 * 10.0 * 10.0 // ρ * V² / 2 (seawater, ~10 m/s)
        let referenceArea = FinGeometry.sideFinArea / 144.0 // Convert sq.in to sq.ft
        
        let lift = liftCoefficient * dynamicPressure * referenceArea
        let drag = dragCoefficient * dynamicPressure * referenceArea
        
        return (lift, drag)
    }
}

/// Represents a flow vector in 3D space
struct FlowVector {
    let position: (x: Float, y: Float, z: Float)
    let velocity: (vx: Float, vy: Float, vz: Float)
    let magnitude: Float
    
    /// Color coding based on velocity magnitude
    var color: (r: Float, g: Float, b: Float) {
        let normalizedMag = min(1.0, magnitude / 2.0) // Normalize to 0-1
        return (normalizedMag, 1.0 - normalizedMag, 0.5) // Red to green gradient
    }
}