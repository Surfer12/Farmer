import Foundation
import SwiftUI
import Charts
import Accelerate

// MARK: - Mathematical Components

/// Represents the hybrid PINN-RK4 system with cognitive regularization
class HybridPINNRK4System {
    
    // MARK: - Core Components
    
    private var pinn: PINN
    private var rk4Solver: RK4Solver
    
    // Hyperparameters
    var alpha: Double = 0.5  // α(t) for real-time validation flows
    var lambda1: Double = 0.6  // Weight for cognitive regularization
    var lambda2: Double = 0.4  // Weight for efficiency regularization
    var beta: Double = 1.2     // β for model responsiveness
    
    // Performance metrics
    private var trainingHistory: [TrainingMetrics] = []
    
    init() {
        self.pinn = PINN()
        self.rk4Solver = RK4Solver()
    }
    
    // MARK: - Hybrid Output Computation
    
    /// Computes Ψ(x) = O_hybrid × exp(-P_total) × P(H|E,β)
    func computeHybridOutput(x: Double, t: Double, epoch: Int) -> HybridResult {
        // Step 1: Compute individual outputs
        let S_x = pinn.forward(x: x, t: t)  // State inference
        let N_x = computeMLGradientAnalysis(x: x, t: t)  // ML gradient descent analysis
        
        // Step 2: Hybrid combination
        let O_hybrid = alpha * S_x + (1 - alpha) * N_x
        
        // Step 3: Regularization penalties
        let R_cognitive = computeCognitiveRegularization(x: x, t: t)
        let R_efficiency = computeEfficiencyRegularization()
        let P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        let expTerm = exp(-P_total)
        
        // Step 4: Probability computation
        let P_base = computeBaseProbability(x: x, t: t, epoch: epoch)
        let P_adjusted = adjustProbabilityWithBeta(P_base, beta: beta)
        
        // Step 5: Final Ψ(x) computation
        let psi = O_hybrid * expTerm * P_adjusted
        
        return HybridResult(
            S_x: S_x,
            N_x: N_x,
            O_hybrid: O_hybrid,
            R_cognitive: R_cognitive,
            R_efficiency: R_efficiency,
            P_total: P_total,
            expTerm: expTerm,
            P_base: P_base,
            P_adjusted: P_adjusted,
            psi: psi
        )
    }
    
    // MARK: - Component Implementations
    
    private func computeMLGradientAnalysis(x: Double, t: Double) -> Double {
        // Simulate ML gradient descent analysis using RK4 solution
        let rk4Solution = rk4Solver.solve(x: x, t: t)
        return tanh(rk4Solution)  // Normalized output
    }
    
    private func computeCognitiveRegularization(x: Double, t: Double) -> Double {
        // R_cognitive for PDE residual accuracy
        let pdeResidual = computePDEResidual(x: x, t: t)
        return abs(pdeResidual) / (1.0 + abs(pdeResidual))  // Normalized penalty
    }
    
    private func computeEfficiencyRegularization() -> Double {
        // R_efficiency for training loop efficiency
        let computationalCost = Double(pinn.getTotalParameters()) / 1000.0
        return computationalCost / (1.0 + computationalCost)
    }
    
    private func computeBaseProbability(x: Double, t: Double, epoch: Int) -> Double {
        // P(H|E,β) base computation
        let evidence = computeEvidence(x: x, t: t)
        let hypothesis = computeHypothesis(epoch: epoch)
        return sigmoid(hypothesis - evidence)
    }
    
    private func adjustProbabilityWithBeta(_ probability: Double, beta: Double) -> Double {
        // Adjust probability with responsiveness parameter β
        return 1.0 / (1.0 + exp(-beta * (probability - 0.5)))
    }
    
    private func computeEvidence(x: Double, t: Double) -> Double {
        let pinnSolution = pinn.forward(x: x, t: t)
        let rk4Solution = rk4Solver.solve(x: x, t: t)
        return abs(pinnSolution - rk4Solution)
    }
    
    private func computeHypothesis(epoch: Int) -> Double {
        return 1.0 - exp(-Double(epoch) / 100.0)  // Improves with training
    }
    
    private func computePDEResidual(x: Double, t: Double) -> Double {
        // For heat equation: ∂u/∂t - ∂²u/∂x²
        let dx = 1e-6
        let dt = 1e-6
        
        let u = pinn.forward(x: x, t: t)
        let u_t = (pinn.forward(x: x, t: t + dt) - pinn.forward(x: x, t: t - dt)) / (2 * dt)
        let u_x = (pinn.forward(x: x + dx, t: t) - pinn.forward(x: x - dx, t: t)) / (2 * dx)
        let u_xx = (pinn.forward(x: x + dx, t: t) - 2 * u + pinn.forward(x: x - dx, t: t)) / (dx * dx)
        
        return u_t - u_xx  // Heat equation residual
    }
    
    // MARK: - Training and Validation
    
    func trainEpoch(trainingData: [(x: Double, t: Double)], learningRate: Double = 0.005) -> TrainingMetrics {
        var totalLoss = 0.0
        var hybridResults: [HybridResult] = []
        
        // Batch processing for efficiency
        let batchSize = min(20, trainingData.count)
        
        for batchStart in stride(from: 0, to: trainingData.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, trainingData.count)
            let batch = Array(trainingData[batchStart..<batchEnd])
            
            var batchLoss = 0.0
            
            for (i, data) in batch.enumerated() {
                let epoch = trainingHistory.count
                let result = computeHybridOutput(x: data.x, t: data.t, epoch: epoch)
                hybridResults.append(result)
                
                // Compute loss based on hybrid output
                let targetSolution = analyticalSolution(x: data.x, t: data.t)
                let loss = pow(result.psi - targetSolution, 2)
                batchLoss += loss
            }
            
            // Update PINN parameters using gradient approximation
            updatePINNParameters(batch: batch, learningRate: learningRate)
            totalLoss += batchLoss
        }
        
        let avgLoss = totalLoss / Double(trainingData.count)
        let avgPsi = hybridResults.map { $0.psi }.reduce(0, +) / Double(hybridResults.count)
        
        let metrics = TrainingMetrics(
            epoch: trainingHistory.count,
            loss: avgLoss,
            averagePsi: avgPsi,
            averageHybridOutput: hybridResults.map { $0.O_hybrid }.reduce(0, +) / Double(hybridResults.count),
            averageCognitiveReg: hybridResults.map { $0.R_cognitive }.reduce(0, +) / Double(hybridResults.count),
            averageEfficiencyReg: hybridResults.map { $0.R_efficiency }.reduce(0, +) / Double(hybridResults.count)
        )
        
        trainingHistory.append(metrics)
        return metrics
    }
    
    private func updatePINNParameters(batch: [(x: Double, t: Double)], learningRate: Double) {
        let perturbation = 1e-5
        
        for layer in pinn.layers {
            // Update weights using finite difference approximation
            for i in 0..<layer.weights.count {
                for j in 0..<layer.weights[i].count {
                    let originalWeight = layer.weights[i][j]
                    
                    // Forward perturbation
                    layer.weights[i][j] = originalWeight + perturbation
                    let lossPlus = computeBatchLoss(batch: batch)
                    
                    // Backward perturbation
                    layer.weights[i][j] = originalWeight - perturbation
                    let lossMinus = computeBatchLoss(batch: batch)
                    
                    // Gradient approximation
                    let gradient = (lossPlus - lossMinus) / (2 * perturbation)
                    
                    // Update weight
                    layer.weights[i][j] = originalWeight - learningRate * gradient
                }
            }
            
            // Update biases
            for i in 0..<layer.biases.count {
                let originalBias = layer.biases[i]
                
                layer.biases[i] = originalBias + perturbation
                let lossPlus = computeBatchLoss(batch: batch)
                
                layer.biases[i] = originalBias - perturbation
                let lossMinus = computeBatchLoss(batch: batch)
                
                let gradient = (lossPlus - lossMinus) / (2 * perturbation)
                layer.biases[i] = originalBias - learningRate * gradient
            }
        }
    }
    
    private func computeBatchLoss(batch: [(x: Double, t: Double)]) -> Double {
        var totalLoss = 0.0
        
        for data in batch {
            let result = computeHybridOutput(x: data.x, t: data.t, epoch: trainingHistory.count)
            let target = analyticalSolution(x: data.x, t: data.t)
            totalLoss += pow(result.psi - target, 2)
        }
        
        return totalLoss / Double(batch.count)
    }
    
    private func analyticalSolution(x: Double, t: Double) -> Double {
        // Analytical solution for heat equation with initial condition u(x,0) = -sin(πx)
        return -sin(.pi * x) * exp(-.pi * .pi * t)
    }
    
    // MARK: - Utility Functions
    
    private func sigmoid(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }
    
    func getTrainingHistory() -> [TrainingMetrics] {
        return trainingHistory
    }
    
    func getCurrentAlpha() -> Double {
        return alpha
    }
    
    func adaptiveUpdateAlpha(basedOnPerformance performance: Double) {
        // Real-time validation flow adjustment
        alpha = 0.3 + 0.4 * sigmoid(performance - 0.5)
        alpha = max(0.1, min(0.9, alpha))  // Clamp between 0.1 and 0.9
    }
}

// MARK: - Supporting Classes

class PINN {
    var layers: [DenseLayer]
    
    init() {
        layers = [
            DenseLayer(inputSize: 2, outputSize: 32),
            DenseLayer(inputSize: 32, outputSize: 32),
            DenseLayer(inputSize: 32, outputSize: 32),
            DenseLayer(inputSize: 32, outputSize: 1)
        ]
    }
    
    func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        
        for (index, layer) in layers.enumerated() {
            input = layer.forward(input)
            // Apply activation function (tanh) except for output layer
            if index < layers.count - 1 {
                input = input.map { tanh($0) }
            }
        }
        
        return input[0]
    }
    
    func getTotalParameters() -> Int {
        return layers.reduce(0) { total, layer in
            total + layer.weights.flatMap { $0 }.count + layer.biases.count
        }
    }
}

class DenseLayer {
    var weights: [[Double]]
    var biases: [Double]
    
    init(inputSize: Int, outputSize: Int) {
        // Xavier/Glorot initialization
        let bound = sqrt(6.0 / Double(inputSize + outputSize))
        
        weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in
                Double.random(in: -bound...bound)
            }
        }
        
        biases = (0..<outputSize).map { _ in
            Double.random(in: -0.1...0.1)
        }
    }
    
    func forward(_ input: [Double]) -> [Double] {
        return (0..<weights.count).map { i in
            let weightedSum = zip(weights[i], input).map(*).reduce(0, +)
            return weightedSum + biases[i]
        }
    }
}

class RK4Solver {
    func solve(x: Double, t: Double) -> Double {
        // RK4 solution for heat equation (simplified for demonstration)
        // In practice, this would solve the full PDE numerically
        let dt = 0.01
        let steps = Int(t / dt)
        
        var u = -sin(.pi * x)  // Initial condition
        
        for _ in 0..<steps {
            let k1 = heatEquationRHS(u: u, x: x)
            let k2 = heatEquationRHS(u: u + 0.5 * dt * k1, x: x)
            let k3 = heatEquationRHS(u: u + 0.5 * dt * k2, x: x)
            let k4 = heatEquationRHS(u: u + dt * k3, x: x)
            
            u += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        }
        
        return u
    }
    
    private func heatEquationRHS(u: Double, x: Double) -> Double {
        // Simplified: du/dt = -π²u (for the analytical solution)
        return -.pi * .pi * u
    }
}

// MARK: - Data Structures

struct HybridResult {
    let S_x: Double          // State inference
    let N_x: Double          // ML gradient descent analysis
    let O_hybrid: Double     // Hybrid output
    let R_cognitive: Double  // Cognitive regularization
    let R_efficiency: Double // Efficiency regularization
    let P_total: Double      // Total penalty
    let expTerm: Double      // exp(-P_total)
    let P_base: Double       // Base probability
    let P_adjusted: Double   // Adjusted probability with β
    let psi: Double          // Final Ψ(x) value
    
    var interpretation: String {
        switch psi {
        case 0.8...:
            return "Excellent model performance"
        case 0.6..<0.8:
            return "Good model performance"
        case 0.4..<0.6:
            return "Moderate model performance"
        case 0.2..<0.4:
            return "Poor model performance"
        default:
            return "Very poor model performance"
        }
    }
}

struct TrainingMetrics {
    let epoch: Int
    let loss: Double
    let averagePsi: Double
    let averageHybridOutput: Double
    let averageCognitiveReg: Double
    let averageEfficiencyReg: Double
}

// MARK: - SwiftUI Visualization

struct HybridPINNVisualizationView: View {
    @StateObject private var system = HybridPINNRK4System()
    @State private var isTraining = false
    @State private var currentEpoch = 0
    @State private var trainingData: [(x: Double, t: Double)] = []
    @State private var solutionData: [SolutionPoint] = []
    @State private var selectedTime: Double = 0.5
    
    var body: some View {
        VStack(spacing: 20) {
            // Title
            Text("Hybrid PINN-RK4 System")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // Control Panel
            HStack(spacing: 20) {
                VStack {
                    Text("Time: \(selectedTime, specifier: "%.2f")")
                    Slider(value: $selectedTime, in: 0...1, step: 0.1)
                        .frame(width: 150)
                }
                
                Button(isTraining ? "Stop Training" : "Start Training") {
                    if isTraining {
                        stopTraining()
                    } else {
                        startTraining()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(trainingData.isEmpty)
                
                Button("Generate Data") {
                    generateTrainingData()
                }
                .buttonStyle(.bordered)
            }
            
            // Metrics Display
            if let lastMetrics = system.getTrainingHistory().last {
                MetricsView(metrics: lastMetrics, alpha: system.getCurrentAlpha())
            }
            
            // Solution Comparison Chart
            if !solutionData.isEmpty {
                SolutionComparisonChart(data: solutionData, time: selectedTime)
                    .frame(height: 300)
            }
            
            // Training History Chart
            if !system.getTrainingHistory().isEmpty {
                TrainingHistoryChart(history: system.getTrainingHistory())
                    .frame(height: 200)
            }
            
            Spacer()
        }
        .padding()
        .onAppear {
            generateTrainingData()
        }
        .onChange(of: selectedTime) { _ in
            updateSolutionData()
        }
    }
    
    private func generateTrainingData() {
        trainingData = []
        
        // Generate spatial points
        for x in stride(from: -1.0, through: 1.0, by: 0.1) {
            for t in stride(from: 0.0, through: 1.0, by: 0.2) {
                trainingData.append((x: x, t: t))
            }
        }
        
        updateSolutionData()
    }
    
    private func updateSolutionData() {
        solutionData = []
        
        for x in stride(from: -1.0, through: 1.0, by: 0.05) {
            let result = system.computeHybridOutput(x: x, t: selectedTime, epoch: currentEpoch)
            let rk4Solution = system.rk4Solver.solve(x: x, t: selectedTime)
            let analytical = -sin(.pi * x) * exp(-.pi * .pi * selectedTime)
            
            solutionData.append(SolutionPoint(
                x: x,
                pinn: result.S_x,
                rk4: rk4Solution,
                hybrid: result.psi,
                analytical: analytical
            ))
        }
    }
    
    private func startTraining() {
        isTraining = true
        currentEpoch = 0
        
        Task {
            while isTraining && currentEpoch < 1000 {
                let metrics = system.trainEpoch(trainingData: trainingData)
                
                // Adaptive alpha adjustment
                system.adaptiveUpdateAlpha(basedOnPerformance: metrics.averagePsi)
                
                await MainActor.run {
                    currentEpoch += 1
                    if currentEpoch % 10 == 0 {
                        updateSolutionData()
                    }
                }
                
                // Small delay to allow UI updates
                try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            
            await MainActor.run {
                isTraining = false
                updateSolutionData()
            }
        }
    }
    
    private func stopTraining() {
        isTraining = false
    }
}

struct MetricsView: View {
    let metrics: TrainingMetrics
    let alpha: Double
    
    var body: some View {
        HStack(spacing: 30) {
            VStack {
                Text("Epoch")
                Text("\(metrics.epoch)")
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            
            VStack {
                Text("Loss")
                Text(String(format: "%.4f", metrics.loss))
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(lossColor(metrics.loss))
            }
            
            VStack {
                Text("Avg Ψ(x)")
                Text(String(format: "%.3f", metrics.averagePsi))
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(psiColor(metrics.averagePsi))
            }
            
            VStack {
                Text("α(t)")
                Text(String(format: "%.3f", alpha))
                    .font(.title2)
                    .fontWeight(.semibold)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(10)
    }
    
    private func lossColor(_ loss: Double) -> Color {
        switch loss {
        case 0..<0.01: return .green
        case 0.01..<0.1: return .orange
        default: return .red
        }
    }
    
    private func psiColor(_ psi: Double) -> Color {
        switch psi {
        case 0.8...: return .green
        case 0.6..<0.8: return .blue
        case 0.4..<0.6: return .orange
        default: return .red
        }
    }
}

struct SolutionComparisonChart: View {
    let data: [SolutionPoint]
    let time: Double
    
    var body: some View {
        VStack {
            Text("Solution Comparison at t = \(time, specifier: "%.1f")")
                .font(.headline)
            
            Chart(data) { point in
                LineMark(
                    x: .value("x", point.x),
                    y: .value("PINN", point.pinn)
                )
                .foregroundStyle(.blue)
                .symbol(.circle)
                
                LineMark(
                    x: .value("x", point.x),
                    y: .value("RK4", point.rk4)
                )
                .foregroundStyle(.red)
                .symbol(.square)
                
                LineMark(
                    x: .value("x", point.x),
                    y: .value("Hybrid Ψ(x)", point.hybrid)
                )
                .foregroundStyle(.purple)
                .symbol(.diamond)
                .lineStyle(StrokeStyle(lineWidth: 2))
                
                LineMark(
                    x: .value("x", point.x),
                    y: .value("Analytical", point.analytical)
                )
                .foregroundStyle(.green)
                .lineStyle(StrokeStyle(dash: [5, 5]))
            }
            .chartXAxis {
                AxisMarks(position: .bottom)
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            .chartLegend(position: .bottom)
        }
    }
}

struct TrainingHistoryChart: View {
    let history: [TrainingMetrics]
    
    var body: some View {
        VStack {
            Text("Training Progress")
                .font(.headline)
            
            Chart(history) { metrics in
                LineMark(
                    x: .value("Epoch", metrics.epoch),
                    y: .value("Loss", metrics.loss)
                )
                .foregroundStyle(.red)
                
                LineMark(
                    x: .value("Epoch", metrics.epoch),
                    y: .value("Avg Ψ(x)", metrics.averagePsi)
                )
                .foregroundStyle(.blue)
            }
            .chartYScale(domain: 0...1)
        }
    }
}

struct SolutionPoint: Identifiable {
    let id = UUID()
    let x: Double
    let pinn: Double
    let rk4: Double
    let hybrid: Double
    let analytical: Double
}

// MARK: - Example Usage and Testing

struct ContentView: View {
    var body: some View {
        HybridPINNVisualizationView()
    }
}

// Example of running a single training step (matching your numerical example)
func demonstrateNumericalExample() {
    let system = HybridPINNRK4System()
    
    // Set parameters to match your example
    system.alpha = 0.5
    system.lambda1 = 0.6
    system.lambda2 = 0.4
    system.beta = 1.2
    
    let result = system.computeHybridOutput(x: 0.0, t: 0.1, epoch: 1)
    
    print("=== Numerical Example Demonstration ===")
    print("Step 1 - Outputs:")
    print("  S(x) = \(String(format: "%.3f", result.S_x))")
    print("  N(x) = \(String(format: "%.3f", result.N_x))")
    
    print("Step 2 - Hybrid:")
    print("  α = \(system.alpha)")
    print("  O_hybrid = \(String(format: "%.3f", result.O_hybrid))")
    
    print("Step 3 - Penalties:")
    print("  R_cognitive = \(String(format: "%.3f", result.R_cognitive))")
    print("  R_efficiency = \(String(format: "%.3f", result.R_efficiency))")
    print("  P_total = \(String(format: "%.3f", result.P_total))")
    print("  exp(-P_total) = \(String(format: "%.3f", result.expTerm))")
    
    print("Step 4 - Probability:")
    print("  P_base = \(String(format: "%.3f", result.P_base))")
    print("  β = \(system.beta)")
    print("  P_adjusted = \(String(format: "%.3f", result.P_adjusted))")
    
    print("Step 5 - Final Ψ(x):")
    print("  Ψ(x) = \(String(format: "%.3f", result.psi))")
    
    print("Step 6 - Interpretation:")
    print("  \(result.interpretation)")
}

#if DEBUG
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
#endif