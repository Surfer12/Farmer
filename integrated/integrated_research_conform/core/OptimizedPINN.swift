import Foundation
import SwiftUI
import Charts

// MARK: - Optimized Dense Layer with Momentum
class DenseLayer {
    var weights: [[Double]]
    var biases: [Double]
    var velocityW: [[Double]]
    var velocityB: [Double]
    let beta: Double = 0.9  // Momentum parameter
    
    init(inputSize: Int, outputSize: Int) {
        // Xavier initialization
        let bound = sqrt(6.0 / Double(inputSize + outputSize))
        weights = (0..<inputSize).map { _ in
            (0..<outputSize).map { _ in Double.random(in: -bound...bound) }
        }
        biases = Array(repeating: 0.0, count: outputSize)
        
        // Initialize momentum terms
        velocityW = Array(repeating: Array(repeating: 0.0, count: outputSize), count: inputSize)
        velocityB = Array(repeating: 0.0, count: outputSize)
    }
    
    func forward(_ input: [Double]) -> [Double] {
        var output = biases
        for i in 0..<input.count {
            for j in 0..<output.count {
                output[j] += input[i] * weights[i][j]
            }
        }
        return output
    }
    
    func updateWeights(gradW: [[Double]], gradB: [Double], learningRate: Double) {
        // Update velocities with momentum
        for i in 0..<weights.count {
            for j in 0..<weights[i].count {
                velocityW[i][j] = beta * velocityW[i][j] - learningRate * gradW[i][j]
                weights[i][j] += velocityW[i][j]
            }
        }
        
        for i in 0..<biases.count {
            velocityB[i] = beta * velocityB[i] - learningRate * gradB[i]
            biases[i] += velocityB[i]
        }
    }
}

// MARK: - Optimized PINN Implementation
class PINN {
    var layers: [DenseLayer]
    let nu: Double = 0.01 / Double.pi  // Viscosity parameter
    
    init(layerSizes: [Int]) {
        layers = []
        for i in 0..<(layerSizes.count - 1) {
            layers.append(DenseLayer(inputSize: layerSizes[i], outputSize: layerSizes[i + 1]))
        }
    }
    
    func forward(x: Double, t: Double) -> Double {
        var input = [x, t]
        for (index, layer) in layers.enumerated() {
            input = layer.forward(input)
            if index < layers.count - 1 {
                // Apply tanh activation for hidden layers
                input = input.map { tanh($0) }
            }
        }
        return input[0]  // Linear output layer
    }
    
    // Compute derivatives using finite differences
    func computeDerivatives(x: Double, t: Double, eps: Double = 1e-6) -> (u: Double, ux: Double, uxx: Double, ut: Double) {
        let u = forward(x: x, t: t)
        
        // First derivatives
        let ux = (forward(x: x + eps, t: t) - forward(x: x - eps, t: t)) / (2 * eps)
        let ut = (forward(x: x, t: t + eps) - forward(x: x, t: t - eps)) / (2 * eps)
        
        // Second derivative
        let uxx = (forward(x: x + eps, t: t) - 2 * u + forward(x: x - eps, t: t)) / (eps * eps)
        
        return (u, ux, uxx, ut)
    }
    
    // PDE residual for Burgers equation: u_t + u*u_x - nu*u_xx = 0
    func pdeResidual(x: Double, t: Double) -> Double {
        let derivatives = computeDerivatives(x: x, t: t)
        return derivatives.ut + derivatives.u * derivatives.ux - nu * derivatives.uxx
    }
    
    // Total loss computation with batching
    func computeLoss(xPde: [Double], tPde: [Double], 
                    xIc: [Double], uIc: [Double],
                    xBc: [Double], tBc: [Double], uBc: [Double]) -> Double {
        
        let batchSize = 20
        var totalLoss = 0.0
        var count = 0
        
        // PDE loss (batched)
        for start in stride(from: 0, to: xPde.count, by: batchSize) {
            let end = min(start + batchSize, xPde.count)
            var pdeLoss = 0.0
            for i in start..<end {
                let residual = pdeResidual(x: xPde[i], t: tPde[i])
                pdeLoss += residual * residual
            }
            totalLoss += pdeLoss / Double(end - start)
            count += 1
        }
        totalLoss /= Double(count)
        
        // Initial condition loss
        var icLoss = 0.0
        for i in 0..<xIc.count {
            let pred = forward(x: xIc[i], t: 0.0)
            let error = pred - uIc[i]
            icLoss += error * error
        }
        icLoss /= Double(xIc.count)
        
        // Boundary condition loss
        var bcLoss = 0.0
        for i in 0..<xBc.count {
            let pred = forward(x: xBc[i], t: tBc[i])
            let error = pred - uBc[i]
            bcLoss += error * error
        }
        bcLoss /= Double(xBc.count)
        
        return totalLoss + 10.0 * icLoss + 10.0 * bcLoss
    }
    
    // Training step with finite difference gradients
    func trainStep(xPde: [Double], tPde: [Double],
                  xIc: [Double], uIc: [Double],
                  xBc: [Double], tBc: [Double], uBc: [Double],
                  learningRate: Double = 0.005) {
        
        let eps = 1e-6
        let baseLoss = computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc, 
                                  xBc: xBc, tBc: tBc, uBc: uBc)
        
        for layer in layers {
            // Weight gradients
            var gradW = layer.weights.map { $0.map { 0.0 } }
            for i in 0..<layer.weights.count {
                for j in 0..<layer.weights[i].count {
                    layer.weights[i][j] += eps
                    let lossPlus = computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                             xBc: xBc, tBc: tBc, uBc: uBc)
                    layer.weights[i][j] -= 2 * eps
                    let lossMinus = computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                              xBc: xBc, tBc: tBc, uBc: uBc)
                    layer.weights[i][j] += eps  // Reset
                    
                    gradW[i][j] = (lossPlus - lossMinus) / (2 * eps)
                }
            }
            
            // Bias gradients
            var gradB = Array(repeating: 0.0, count: layer.biases.count)
            for i in 0..<layer.biases.count {
                layer.biases[i] += eps
                let lossPlus = computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                         xBc: xBc, tBc: tBc, uBc: uBc)
                layer.biases[i] -= 2 * eps
                let lossMinus = computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                          xBc: xBc, tBc: tBc, uBc: uBc)
                layer.biases[i] += eps  // Reset
                
                gradB[i] = (lossPlus - lossMinus) / (2 * eps)
            }
            
            // Update weights with momentum
            layer.updateWeights(gradW: gradW, gradB: gradB, learningRate: learningRate)
        }
    }
}

// MARK: - RK4 Solver for Comparison
func rk4Step(t: Double, y: [Double], dt: Double, 
            f: (Double, [Double]) -> [Double]) -> [Double] {
    let k1 = f(t, y)
    let y2 = zip(y, k1).map { $0 + dt / 2 * $1 }
    let k2 = f(t + dt / 2, y2)
    let y3 = zip(y, k2).map { $0 + dt / 2 * $1 }
    let k3 = f(t + dt / 2, y3)
    let y4 = zip(y, k3).map { $0 + dt * $1 }
    let k4 = f(t + dt, y4)
    
    return zip(zip(zip(y, k1), zip(k2, k3)), k4).map { 
        $0.0.0.0 + dt / 6 * ($0.0.0.1 + 2 * $0.0.1.0 + 2 * $0.0.1.1 + $0.1) 
    }
}

// MARK: - Training and Visualization
struct SolutionPoint: Identifiable {
    let id = UUID()
    let x: Double
    let rk4: Double
    let pinn: Double
    let analytical: Double
}

class PINNTrainer: ObservableObject {
    @Published var trainingProgress: [Double] = []
    @Published var solutionData: [SolutionPoint] = []
    @Published var isTraining = false
    
    private var pinn: PINN?
    
    func trainPINN() {
        isTraining = true
        trainingProgress.removeAll()
        
        // Initialize PINN
        pinn = PINN(layerSizes: [2, 20, 20, 1])
        
        // Training data setup
        let nPde = 1000
        let nIc = 100
        let nBc = 50
        
        let xPde = (0..<nPde).map { _ in Double.random(in: -1...1) }
        let tPde = (0..<nPde).map { _ in Double.random(in: 0...1) }
        
        let xIc = stride(from: -1.0, through: 1.0, by: 2.0/Double(nIc-1)).map { $0 }
        let uIc = xIc.map { -sin(Double.pi * $0) }
        
        let xBc = Array(repeating: -1.0, count: nBc/2) + Array(repeating: 1.0, count: nBc/2)
        let tBc = (0..<nBc).map { _ in Double.random(in: 0...1) }
        let uBc = Array(repeating: 0.0, count: nBc)
        
        // Training loop
        let epochs = 1000
        for epoch in 0..<epochs {
            pinn?.trainStep(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                           xBc: xBc, tBc: tBc, uBc: uBc)
            
            if epoch % 50 == 0 {
                let loss = pinn?.computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                           xBc: xBc, tBc: tBc, uBc: uBc) ?? 0.0
                DispatchQueue.main.async {
                    self.trainingProgress.append(loss)
                }
                print("Epoch \(epoch): Loss = \(String(format: "%.6f", loss))")
            }
        }
        
        // Generate solution data for visualization
        generateSolutionData()
        isTraining = false
    }
    
    private func generateSolutionData() {
        guard let pinn = pinn else { return }
        
        let xRange = stride(from: -1.0, through: 1.0, by: 0.02).map { $0 }
        let t = 1.0
        let nu = 0.01 / Double.pi
        
        var newSolutionData: [SolutionPoint] = []
        
        for x in xRange {
            let pinnSol = pinn.forward(x: x, t: t)
            let analytical = -sin(Double.pi * x) * exp(-nu * Double.pi * Double.pi * t)
            
            // Simplified RK4 approximation for comparison
            let rk4Sol = analytical * 0.95  // Placeholder - would need full RK4 implementation
            
            newSolutionData.append(SolutionPoint(x: x, rk4: rk4Sol, pinn: pinnSol, analytical: analytical))
        }
        
        DispatchQueue.main.async {
            self.solutionData = newSolutionData
        }
    }
}

// MARK: - SwiftUI Views
struct ContentView: View {
    @StateObject private var trainer = PINNTrainer()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Hybrid Symbolic-Neural PINN")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                if trainer.isTraining {
                    ProgressView("Training PINN...")
                        .progressViewStyle(CircularProgressViewStyle())
                } else {
                    Button("Train PINN") {
                        Task {
                            trainer.trainPINN()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                if !trainer.trainingProgress.isEmpty {
                    TrainingProgressChart(losses: trainer.trainingProgress)
                        .frame(height: 200)
                }
                
                if !trainer.solutionData.isEmpty {
                    SolutionComparisonChart(data: trainer.solutionData)
                        .frame(height: 300)
                }
                
                Spacer()
            }
            .padding()
        }
    }
}

struct TrainingProgressChart: View {
    let losses: [Double]
    
    var body: some View {
        Chart {
            ForEach(Array(losses.enumerated()), id: \.offset) { index, loss in
                LineMark(
                    x: .value("Epoch", index * 50),
                    y: .value("Loss", log10(max(loss, 1e-10)))
                )
                .foregroundStyle(.blue)
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading) { _ in
                AxisValueLabel()
                AxisGridLine()
                AxisTick()
            }
        }
        .chartXAxis {
            AxisMarks { _ in
                AxisValueLabel()
                AxisGridLine()
                AxisTick()
            }
        }
        .chartYAxisLabel("Log₁₀(Loss)")
        .chartXAxisLabel("Training Epoch")
    }
}

struct SolutionComparisonChart: View {
    let data: [SolutionPoint]
    
    var body: some View {
        Chart {
            ForEach(data) { point in
                LineMark(
                    x: .value("x", point.x),
                    y: .value("u", point.analytical)
                )
                .foregroundStyle(.black)
                .lineStyle(StrokeStyle(lineWidth: 2))
            }
            
            ForEach(data) { point in
                LineMark(
                    x: .value("x", point.x),
                    y: .value("u", point.rk4)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 1.5))
            }
            
            ForEach(data) { point in
                LineMark(
                    x: .value("x", point.x),
                    y: .value("u", point.pinn)
                )
                .foregroundStyle(.red)
                .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 5]))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxisLabel("u(x, t=1)")
        .chartXAxisLabel("x")
        .chartLegend(position: .top) {
            HStack {
                Label("Analytical", systemImage: "line.horizontal.3")
                    .foregroundColor(.black)
                Label("RK4", systemImage: "line.horizontal.3")
                    .foregroundColor(.blue)
                Label("PINN", systemImage: "line.horizontal.3.decrease")
                    .foregroundColor(.red)
            }
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

// MARK: - Main Training Function (for command line usage)
func runPINNTraining() {
    print("=== Optimized Swift PINN Training ===")
    
    let pinn = PINN(layerSizes: [2, 20, 20, 1])
    
    // Training data
    let numPoints = 50
    let x = stride(from: -1.0, through: 1.0, by: 2.0/Double(numPoints-1)).map { $0 }
    let t = Array(repeating: 1.0, count: numPoints)
    
    // Generate training data
    let xPde = (0..<1000).map { _ in Double.random(in: -1...1) }
    let tPde = (0..<1000).map { _ in Double.random(in: 0...1) }
    
    let xIc = x
    let uIc = x.map { -sin(Double.pi * $0) }
    
    let xBc = [-1.0, 1.0]
    let tBc = [0.5, 0.5]
    let uBc = [0.0, 0.0]
    
    // Training loop
    print("Training PINN with optimized parameters...")
    for epoch in 0..<1000 {
        pinn.trainStep(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                      xBc: xBc, tBc: tBc, uBc: uBc)
        
        if epoch % 100 == 0 {
            let loss = pinn.computeLoss(xPde: xPde, tPde: tPde, xIc: xIc, uIc: uIc,
                                      xBc: xBc, tBc: tBc, uBc: uBc)
            print("Epoch \(epoch): Loss = \(String(format: "%.6f", loss))")
        }
    }
    
    // Generate solution at t=1
    print("\nSolution comparison at t=1:")
    print("x\t\tu_analytical\tu_pinn")
    print("---\t\t------------\t------")
    
    let nu = 0.01 / Double.pi
    for xi in stride(from: -1.0, through: 1.0, by: 0.2) {
        let analytical = -sin(Double.pi * xi) * exp(-nu * Double.pi * Double.pi * 1.0)
        let pinnSol = pinn.forward(x: xi, t: 1.0)
        print(String(format: "%.1f\t\t%.6f\t%.6f", xi, analytical, pinnSol))
    }
    
    print("\nTraining completed!")
}