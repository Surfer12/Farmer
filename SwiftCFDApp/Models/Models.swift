import Foundation

struct LiftDrag: Equatable {
    var lift: Float
    var drag: Float
}

struct SensorSnapshot: Equatable {
    var aoaDegrees: Float
    var pressure: [Float]
    var timestamp: Date
}