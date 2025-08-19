import Foundation

// MARK: - Fin Configuration Models

struct FinSpecification: Codable, Identifiable {
    let id = UUID()
    let name: String
    let area: Double // square inches
    let angle: Double // degrees
    let foilType: FoilType
    let position: FinPosition
    let rake: Double // degrees
    
    // Vector 3/2 Blackstix+ specifications
    static let vector32SideFin = FinSpecification(
        name: "Vector 3/2 Side Fin",
        area: 15.00,
        angle: 6.5,
        foilType: .vector32,
        position: .side,
        rake: 6.5
    )
    
    static let vector32CenterFin = FinSpecification(
        name: "Vector 3/2 Center Fin",
        area: 14.50,
        angle: 0.0,
        foilType: .symmetric,
        position: .center,
        rake: 0.0
    )
}

enum FoilType: String, CaseIterable, Codable {
    case vector32 = "Vector 3/2"
    case symmetric = "Symmetric"
    case scimitarTip = "Scimitar Tip"
    
    var description: String {
        switch self {
        case .vector32:
            return "Vector 3/2 foil with optimized lift-to-drag ratio"
        case .symmetric:
            return "Symmetric foil for stability"
        case .scimitarTip:
            return "Scimitar tip for reduced drag"
        }
    }
}

enum FinPosition: String, CaseIterable, Codable {
    case side = "Side"
    case center = "Center"
}

// MARK: - CFD Data Models

struct CFDResult: Codable {
    let timestamp: Date
    let angleOfAttack: Double // degrees (0-20°)
    let reynoldsNumber: Double // Re ≈ 10^5–10^6
    let lift: Double
    let drag: Double
    let pressureCoefficient: Double
    let flowRegime: FlowRegime
    let pressureDistribution: [PressurePoint]
    
    var liftToDragRatio: Double {
        guard drag != 0 else { return 0 }
        return lift / drag
    }
}

struct PressurePoint: Codable {
    let x: Double // Normalized chord position (0-1)
    let y: Double // Normalized span position (0-1)
    let z: Double // Normalized thickness position (0-1)
    let pressure: Double // Pressure coefficient
    let velocity: Vector3D
}

struct Vector3D: Codable {
    let x: Double
    let y: Double
    let z: Double
    
    var magnitude: Double {
        return sqrt(x*x + y*y + z*z)
    }
}

enum FlowRegime: String, CaseIterable, Codable {
    case laminar = "Laminar"
    case transitional = "Transitional"
    case turbulent = "Turbulent"
    
    var criticalAngle: Double {
        switch self {
        case .laminar: return 10.0
        case .transitional: return 15.0
        case .turbulent: return 20.0
        }
    }
}

// MARK: - Hydrodynamic Performance Model

struct HydrodynamicPerformance: Codable {
    let finSpec: FinSpecification
    let cfdResult: CFDResult
    let waveConditions: WaveConditions
    let riderWeight: Double // lbs (125-175)
    
    // Performance metrics
    var liftIncrease: Double {
        // 12% lift increase for Raked fins vs Pivot
        return finSpec.rake > 0 ? 1.12 : 1.0
    }
    
    var pressureDifferential: Double {
        // 30% pressure differential across fin
        return 0.30
    }
}

struct WaveConditions: Codable {
    let height: Double // feet (2-6 ft)
    let period: Double // seconds
    let direction: Double // degrees
    let temperature: Double // Celsius
    
    static let typical = WaveConditions(
        height: 4.0,
        period: 8.0,
        direction: 0.0,
        temperature: 20.0
    )
}

// MARK: - Cognitive Integration Models

struct CognitiveMetrics: Codable {
    let timestamp: Date
    let heartRateVariability: Double // ms (HRV)
    let reactionTime: Double // ms
    let flowStateIndex: Double // 0-1
    let cognitiveLoad: Double // 0-1
    
    var isInFlowState: Bool {
        return flowStateIndex > 0.7 && cognitiveLoad < 0.3
    }
}

struct SensorData: Codable {
    let timestamp: Date
    let imuData: IMUData
    let pressureData: [Double]
    let environmentalData: EnvironmentalData
}

struct IMUData: Codable {
    let acceleration: Vector3D
    let rotation: Vector3D
    let attitude: Attitude
    let turnAngle: Double // Derived from attitude
}

struct Attitude: Codable {
    let pitch: Double // radians
    let roll: Double // radians
    let yaw: Double // radians
}

struct EnvironmentalData: Codable {
    let waterTemperature: Double // Celsius
    let pressure: Double // kPa
    let salinity: Double // ppt
}