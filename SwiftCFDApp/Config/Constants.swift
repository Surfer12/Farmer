import Foundation
import CoreBluetooth

enum FinFoil: String, CaseIterable, Identifiable {
    case vector32 = "Vector 3/2"
    case symmetric = "Symmetric"

    var id: String { rawValue }
}

struct FinSpec: Equatable {
    var sideAreaSqIn: Double = 15.00
    var centerAreaSqIn: Double = 14.50
    var sideCantDegrees: Double = 6.5
    var foil: FinFoil = .vector32
}

enum FlowMode: String, CaseIterable, Identifiable {
    case laminar
    case turbulent

    var id: String { rawValue }
}

enum BluetoothConfig {
    // Replace with real values if available
    static let pressureService = CBUUID(string: "0000A0A0-0000-1000-8000-00805F9B34FB")
    static let pressureCharacteristic = CBUUID(string: "0000A0A1-0000-1000-8000-00805F9B34FB")
}

struct Defaults {
    static let reynolds: Float = 1_000_000
    static let rakeDegrees: Float = 6.5
    static let aoaDegreesRange: ClosedRange<Float> = 0...20
}