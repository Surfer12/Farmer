// PINN Test Example - Standalone demonstration
// This file shows the key functionality of the PINN implementation

import Foundation

// MARK: - Simplified PINN for Testing
class SimplePINN {
    private var weights: [[Double]]
    private var biases: [[Double]]
    
    init() {
        // Simple 2→5→1 architecture for testing
        self.weights = [
            Array(repeating: 0.1, count: 2),  // Input to hidden
            Array(repeating: 0.1, count: 5)   // Hidden to output
        ]
        self.biases = [
            Array(repeating: 0.0, count: 5),  // Hidden bias
            Array(repeating: 0.0, count: 1)   // Output bias
        ]
    }
    
    func forward(_ input: [Double]) -> [Double] {
        var current = input
        
        // First layer
        var hidden = Array(repeating: 0.0, count: 5)
        for i in 0..<5 {
            for j in 0..<2 {
                hidden[i] += weights[0][j] * input[j]
            }
            hidden[i] = tanh(hidden[i] + biases[0][i])
        }
        
        // Output layer
        var output = 0.0
        for i in 0..<5 {
            output += weights[1][i] * hidden[i]
        }
        output += biases[1][0]
        
        return [output]
    }
    
    func predict(x: Double, t: Double) -> Double {
        return forward([x, t])[0]
    }
}

// MARK: - Ψ Framework Analysis
struct SimpleAnalysis {
    let S_x: Double      // State inference accuracy
    let N_x: Double      // ML gradient descent convergence
    let alpha_t: Double  // Real-time validation balance
    let O_hybrid: Double // Hybrid output
    let Psi_x: Double    // Final Ψ value
    
    init(S_x: Double, N_x: Double, alpha_t: Double) {
        self.S_x = S_x
        self.N_x = N_x
        self.alpha_t = alpha_t
        
        // Compute hybrid output
        self.O_hybrid = (1.0 - alpha_t) * S_x + alpha_t * N_x
        
        // Simplified Ψ calculation (without penalties for demo)
        self.Psi_x = O_hybrid
    }
}

// MARK: - Test Functions
func testPINN() {
    print("=== PINN Test Example ===\n")
    
    // Create PINN
    let model = SimplePINN()
    
    // Test predictions
    let testPoints = [
        (x: -1.0, t: 0.0),
        (x: -0.5, t: 0.5),
        (x: 0.0, t: 1.0),
        (x: 0.5, t: 0.5),
        (x: 1.0, t: 0.0)
    ]
    
    print("PINN Predictions:")
    for (x, t) in testPoints {
        let prediction = model.predict(x: x, t: t)
        print("  u(\(String(format: "%.1f", x)), \(String(format: "%.1f", t))) = \(String(format: "%.4f", prediction))")
    }
    print()
    
    // Test Ψ framework
    let analysis = SimpleAnalysis(
        S_x: 0.72,
        N_x: 0.85,
        alpha_t: 0.5
    )
    
    print("Ψ Framework Analysis:")
    print("  S(x) = \(String(format: "%.2f", analysis.S_x)) (state inference accuracy)")
    print("  N(x) = \(String(format: "%.2f", analysis.N_x)) (ML gradient descent convergence)")
    print("  α(t) = \(String(format: "%.2f", analysis.alpha_t)) (real-time validation balance)")
    print("  O_hybrid = (1-α)*S(x) + α*N(x) = \(String(format: "%.3f", analysis.O_hybrid))")
    print("  Ψ(x) = \(String(format: "%.3f", analysis.Psi_x))")
    print()
    
    // Test different α values
    print("Ψ Values for Different α Values:")
    for alpha in stride(from: 0.0, through: 1.0, by: 0.2) {
        let testAnalysis = SimpleAnalysis(S_x: 0.72, N_x: 0.85, alpha_t: alpha)
        print("  α = \(String(format: "%.1f", alpha)): Ψ = \(String(format: "%.3f", testAnalysis.Psi_x))")
    }
    print()
    
    // Demonstrate solution comparison
    let x = Array(stride(from: -1.0, through: 1.0, by: 0.5))
    let t = 1.0
    
    print("Solution Comparison at t = \(t):")
    print("  x\t\tPINN u\t\tAnalytical u\tDifference")
    print("  " + String(repeating: "-", count: 50))
    
    for xVal in x {
        let pinnVal = model.predict(x: xVal, t: t)
        let analyticalVal = sin(Double.pi * xVal) * exp(-0.01 * Double.pi * Double.pi * t)
        let diff = abs(pinnVal - analyticalVal)
        
        print(String(format: "  %.1f\t\t%.3f\t\t%.3f\t\t%.3f", xVal, pinnVal, analyticalVal, diff))
    }
    print()
    
    // Generate sample chart data
    print("Sample Chart Data (JSON format):")
    let chartData = [
        "type": "line",
        "data": [
            "labels": x.map { String(format: "%.1f", $0) },
            "datasets": [
                [
                    "label": "PINN u",
                    "data": x.map { model.predict(x: $0, t: t) },
                    "borderColor": "#1E90FF",
                    "backgroundColor": "#1E90FF",
                    "fill": false,
                    "tension": 0.4
                ],
                [
                    "label": "Analytical u",
                    "data": x.map { sin(Double.pi * $0) * exp(-0.01 * Double.pi * Double.pi * t) },
                    "borderColor": "#FF4500",
                    "backgroundColor": "#FF4500",
                    "fill": false,
                    "tension": 0.4
                ]
            ]
        ]
    ]
    
    if let jsonData = try? JSONSerialization.data(withJSONObject: chartData, options: .prettyPrinted),
       let jsonString = String(data: jsonData, encoding: .utf8) {
        print(jsonString)
    }
}

// MARK: - Main Execution
print("PINN Test Example - Swift Implementation")
print("========================================\n")

// Run the test
testPINN()

print("Test completed successfully!")
print("\nTo run the full implementation:")
print("1. Use Swift Package Manager: swift run uoif-cli pinn")
print("2. Use Xcode: Open Farmer.xcodeproj and run UOIFCLI target")
print("3. Check PINN_README.md for complete documentation")