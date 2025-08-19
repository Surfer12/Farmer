import Foundation
import SceneKit

/// Represents the geometric properties of the Vector 3/2 Blackstix+ fin system
struct FinGeometry {
    // Vector 3/2 Blackstix+ specifications
    static let sideFinArea: Float = 15.00 // sq.in.
    static let centerFinArea: Float = 14.50 // sq.in.
    static let sideFinAngle: Float = 6.5 // degrees
    static let finHeight: Float = 4.48 // inches
    static let finBase: Float = 4.63 // inches
    static let finThickness: Float = 0.1 // inches
    
    // Foil types
    enum FoilType {
        case vector32 // 3/2 foil for side fins
        case symmetric // symmetric foil for center fin
        case scimitarTip // Scimitar tip variant
    }
    
    let area: Float
    let height: Float
    let base: Float
    let angle: Float // in degrees
    let foilType: FoilType
    let thickness: Float
    
    init(area: Float, height: Float, base: Float, angle: Float = 0.0, foilType: FoilType, thickness: Float = 0.1) {
        self.area = area
        self.height = height
        self.base = base
        self.angle = angle
        self.foilType = foilType
        self.thickness = thickness
    }
    
    // Predefined fin configurations
    static let sideFin = FinGeometry(
        area: sideFinArea,
        height: finHeight,
        base: finBase,
        angle: sideFinAngle,
        foilType: .vector32,
        thickness: finThickness
    )
    
    static let centerFin = FinGeometry(
        area: centerFinArea,
        height: finHeight,
        base: finBase,
        angle: 0.0,
        foilType: .symmetric,
        thickness: finThickness
    )
    
    /// Creates a SceneKit geometry for the fin
    func createSCNGeometry() -> SCNGeometry {
        let geometry = SCNBox(
            width: CGFloat(base),
            height: CGFloat(height),
            length: CGFloat(thickness),
            chamferRadius: 0.05
        )
        
        // Apply foil-specific modifications
        switch foilType {
        case .vector32:
            // Add concave surface characteristics for 3/2 foil
            geometry.name = "Vector32Foil"
        case .symmetric:
            geometry.name = "SymmetricFoil"
        case .scimitarTip:
            geometry.name = "ScimitarTipFoil"
        }
        
        return geometry
    }
    
    /// Calculates theoretical lift coefficient based on angle of attack
    func calculateLiftCoefficient(angleOfAttack: Float) -> Float {
        // Simplified lift coefficient calculation
        // In practice, this would use CFD data or more complex aerodynamic models
        let aoaRadians = angleOfAttack * Float.pi / 180.0
        let baseCl = sin(2.0 * aoaRadians) // Basic thin airfoil theory
        
        // Apply foil-specific corrections
        switch foilType {
        case .vector32:
            return baseCl * 1.12 // 12% increase for raked fins
        case .symmetric:
            return baseCl
        case .scimitarTip:
            return baseCl * 1.08 // 8% increase for scimitar tip
        }
    }
    
    /// Calculates drag coefficient based on angle of attack
    func calculateDragCoefficient(angleOfAttack: Float) -> Float {
        let aoaRadians = angleOfAttack * Float.pi / 180.0
        let baseCd = 0.01 + 0.1 * sin(aoaRadians) * sin(aoaRadians) // Simplified drag model
        
        switch foilType {
        case .vector32:
            return baseCd * 0.95 // Reduced drag for efficient foil
        case .symmetric:
            return baseCd
        case .scimitarTip:
            return baseCd * 0.92 // Lower drag for scimitar tip
        }
    }
}