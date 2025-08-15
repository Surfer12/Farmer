// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

// MARK: - PINN Training Example
public class PINNExample {
    
    public static func runCompleteExample() {
        print("=== Physics-Informed Neural Network (PINN) Example ===\n")
        
        // Step 1: Generate training data
        let (x, t) = generateTrainingData()
        print("Generated training data:")
        print("  Spatial points (x): \(x.count) points from -1.0 to 1.0")
        print("  Temporal points (t): \(t.count) points from 0.0 to 1.0\n")
        
        // Step 2: Initialize and train PINN
        let model = PINN(inputSize: 2, hiddenSize: 20, outputSize: 1)
        print("Initialized PINN with Xavier initialization")
        print("  Architecture: 2 → 20 → 20 → 1")
        print("  Learning rate: 0.005")
        print("  Perturbation: 1e-5\n")
        
        // Step 3: Run training
        train(model: model, epochs: 1000, x: x, t: t, printEvery: 100)
        
        // Step 4: Analyze results with Ψ framework
        let analysis = performAnalysis(model: model, x: x, t: t)
        printAnalysis(analysis)
        
        // Step 5: Compare solutions
        let solutions = compareSolutions(pinn: model, x: x, t: 1.0)
        printSolutions(solutions, x: x)
        
        // Step 6: Generate visualization data
        let chartData = generateChartData(solutions: solutions, x: x)
        printChartData(chartData)
    }
    
    // MARK: - Ψ Framework Analysis
    private static func performAnalysis(model: PINN, x: [Double], t: [Double]) -> PINNAnalysis {
        // Compute state inference accuracy (S_x)
        let S_x = computeStateAccuracy(model: model, x: x, t: t)
        
        // Compute ML gradient descent convergence (N_x)
        let N_x = computeConvergenceRate(model: model, x: x, t: t)
        
        // Set real-time validation balance (alpha_t)
        let alpha_t = 0.5
        
        // Compute PDE residual error (R_cognitive)
        let R_cognitive = model.computeTotalLoss(x: x, t: t)
        
        // Estimate computational overhead (R_efficiency)
        let R_efficiency = 0.10
        
        return PINNAnalysis(
            S_x: S_x,
            N_x: N_x,
            alpha_t: alpha_t,
            R_cognitive: R_cognitive,
            R_efficiency: R_efficiency
        )
    }
    
    private static func computeStateAccuracy(model: PINN, x: [Double], t: [Double]) -> Double {
        // Compute accuracy based on how well the PINN satisfies the PDE
        let pdeLoss = model.computeTotalLoss(x: x, t: t)
        // Convert loss to accuracy (0 = perfect, 1 = poor)
        return max(0.0, min(1.0, 1.0 - pdeLoss))
    }
    
    private static func computeConvergenceRate(model: PINN, x: [Double], t: [Double]) -> Double {
        // Estimate convergence rate based on loss reduction
        // This is a simplified metric - in practice, you'd track loss over epochs
        let currentLoss = model.computeTotalLoss(x: x, t: t)
        let baseLoss = 1.0 // Assume starting loss
        return max(0.0, min(1.0, 1.0 - currentLoss / baseLoss))
    }
    
    // MARK: - Results Display
    private static func printAnalysis(_ analysis: PINNAnalysis) {
        print("=== Ψ Framework Analysis Results ===")
        print("Step 1: Outputs")
        print("  S(x) = \(String(format: "%.2f", analysis.S_x)) (state inference accuracy)")
        print("  N(x) = \(String(format: "%.2f", analysis.N_x)) (ML gradient descent convergence)")
        
        print("\nStep 2: Hybrid")
        print("  α(t) = \(String(format: "%.2f", analysis.alpha_t)) (real-time validation balance)")
        print("  O_hybrid = (1-α)*S(x) + α*N(x) = \(String(format: "%.3f", analysis.O_hybrid))")
        
        print("\nStep 3: Penalties")
        print("  R_cognitive = \(String(format: "%.2f", analysis.R_cognitive)) (PDE residual error)")
        print("  R_efficiency = \(String(format: "%.2f", analysis.R_efficiency)) (computational overhead)")
        print("  λ1 = 0.6, λ2 = 0.4 (regularization weights)")
        print("  P_total = λ1*R_cognitive + λ2*R_efficiency = \(String(format: "%.3f", analysis.P_total))")
        print("  exp(-P_total) ≈ \(String(format: "%.3f", exp(-analysis.P_total)))")
        
        print("\nStep 4: Probability")
        print("  P(H|E,β) = 0.80 (base probability)")
        print("  β = 1.2 (responsiveness factor)")
        print("  P_adj = P*β = \(String(format: "%.2f", analysis.P_adj))")
        
        print("\nStep 5: Ψ(x)")
        print("  Ψ(x) = O_hybrid * exp(-P_total) * P_adj ≈ \(String(format: "%.3f", analysis.Psi_x))")
        
        print("\nStep 6: Interpretation")
        print("  Ψ(x) ≈ \(String(format: "%.2f", analysis.Psi_x)) indicates \(interpretPsiValue(analysis.Psi_x))")
        print()
    }
    
    private static func interpretPsiValue(_ psi: Double) -> String {
        switch psi {
        case 0.8...:
            return "excellent model performance with high accuracy and efficiency"
        case 0.6..<0.8:
            return "solid model performance with balanced accuracy and efficiency"
        case 0.4..<0.6:
            return "moderate model performance with room for improvement"
        case 0.2..<0.4:
            return "suboptimal performance requiring optimization"
        default:
            return "poor performance needing significant improvements"
        }
    }
    
    private static func printSolutions(_ solutions: (pinn: [Double], rk4: [Double]), x: [Double]) {
        print("=== Solution Comparison at t=1.0 ===")
        print("  x\t\tPINN u\t\tRK4 u\t\tDifference")
        print("  " + String(repeating: "-", count: 60))
        
        for i in stride(from: 0, to: min(21, x.count), by: 2) {
            let xVal = x[i]
            let pinnVal = solutions.pinn[i]
            let rk4Val = solutions.rk4[i]
            let diff = abs(pinnVal - rk4Val)
            
            print(String(format: "  %.1f\t\t%.3f\t\t%.3f\t\t%.3f", xVal, pinnVal, rk4Val, diff))
        }
        print()
    }
    
    // MARK: - Chart Data Generation
    private static func generateChartData(solutions: (pinn: [Double], rk4: [Double]), x: [Double]) -> ChartData {
        let labels = x.map { String(format: "%.1f", $0) }
        let pinnData = solutions.pinn
        let rk4Data = solutions.rk4
        
        return ChartData(
            labels: labels,
            datasets: [
                Dataset(
                    label: "PINN u",
                    data: pinnData,
                    borderColor: "#1E90FF",
                    backgroundColor: "#1E90FF",
                    fill: false,
                    tension: 0.4
                ),
                Dataset(
                    label: "RK4 u",
                    data: rk4Data,
                    borderColor: "#FF4500",
                    backgroundColor: "#FF4500",
                    fill: false,
                    tension: 0.4
                )
            ]
        )
    }
    
    private static func printChartData(_ chartData: ChartData) {
        print("=== Chart.js Configuration ===")
        print("Use this configuration to visualize the PINN vs RK4 solutions:")
        print()
        
        let jsonData = try! JSONEncoder().encode(chartData)
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
        }
        print()
        
        print("Chart Features:")
        print("  • Blue line: PINN solution")
        print("  • Red line: RK4 reference solution")
        print("  • Smooth curves (tension: 0.4) for better visualization")
        print("  • x-axis: Spatial coordinate from -1.0 to 1.0")
        print("  • y-axis: Solution values u(x, t=1)")
        print()
    }
}

// MARK: - Chart Data Structures
public struct ChartData: Codable {
    public let type: String = "line"
    public let data: ChartDataContent
    public let options: ChartOptions
    
    public init(labels: [String], datasets: [Dataset]) {
        self.data = ChartDataContent(labels: labels, datasets: datasets)
        self.options = ChartOptions()
    }
}

public struct ChartDataContent: Codable {
    public let labels: [String]
    public let datasets: [Dataset]
}

public struct Dataset: Codable {
    public let label: String
    public let data: [Double]
    public let borderColor: String
    public let backgroundColor: String
    public let fill: Bool
    public let tension: Double
}

public struct ChartOptions: Codable {
    public let scales: ChartScales
    public let plugins: ChartPlugins
    
    public init() {
        self.scales = ChartScales()
        self.plugins = ChartPlugins()
    }
}

public struct ChartScales: Codable {
    public let x: ChartAxis
    public let y: ChartAxis
    
    public init() {
        self.x = ChartAxis(title: ChartTitle(display: true, text: "x"))
        self.y = ChartAxis(title: ChartTitle(display: true, text: "u(x, t=1)"), beginAtZero: true)
    }
}

public struct ChartAxis: Codable {
    public let title: ChartTitle
    public let beginAtZero: Bool?
    
    public init(title: ChartTitle, beginAtZero: Bool? = nil) {
        self.title = title
        self.beginAtZero = beginAtZero
    }
}

public struct ChartTitle: Codable {
    public let display: Bool
    public let text: String
}

public struct ChartPlugins: Codable {
    public let legend: ChartLegend
    
    public init() {
        self.legend = ChartLegend(display: true)
    }
}

public struct ChartLegend: Codable {
    public let display: Bool
}