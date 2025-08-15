import Foundation
import SwiftUI

public struct LiftDrag: Equatable {
    public let lift: Float
    public let drag: Float
}

public struct FinSpec: Equatable {
    public let sideAreaSqIn: Float
    public let centerAreaSqIn: Float
    public let cantDegrees: Float
    public let rakeDegrees: Float
    public let foil: String
}

public enum AppError: Error, LocalizedError, Equatable {
    case coreMLUnavailable
    case predictionFailed(String)
    case healthDataUnavailable
    case bluetoothUnavailable
    case motionUnavailable
    case authorizationDenied

    public var errorDescription: String? {
        switch self {
        case .coreMLUnavailable: return "Core ML model unavailable."
        case .predictionFailed(let msg): return "Prediction failed: \(msg)"
        case .healthDataUnavailable: return "Health data unavailable."
        case .bluetoothUnavailable: return "Bluetooth unavailable."
        case .motionUnavailable: return "Device motion unavailable."
        case .authorizationDenied: return "Authorization denied."
        }
    }
}

public enum FlowMode: String, CaseIterable, Identifiable {
    case laminar
    case turbulent
    public var id: String { rawValue }
}

public enum ColorMap {
    /// Maps [0, 1] pressure to blue->red gradient
    public static func color(forNormalized value: Float) -> NSColorOrUIColor {
        let v = max(0.0, min(1.0, value))
        let r = CGFloat(v)
        let g = CGFloat(0.1 * (1.0 - v))
        let b = CGFloat(1.0 - v)
        return NSColorOrUIColor(red: r, green: g, blue: b, alpha: 1.0)
    }
}

#if os(iOS)
import UIKit
public typealias NSColorOrUIColor = UIColor
#elseif os(macOS)
import AppKit
public typealias NSColorOrUIColor = NSColor
#endif