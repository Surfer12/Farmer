import Foundation

// MARK: - Math Helpers

@inline(__always)
func tanhArray(_ x: [Double]) -> [Double] {
    x.map { tanh($0) }
}

@inline(__always)
func dot(_ a: [Double], _ b: [Double]) -> Double {
    precondition(a.count == b.count)
    var s = 0.0
    for i in 0..<a.count { s += a[i] * b[i] }
    return s
}

@inline(__always)
func clamp(_ value: Double, _ lo: Double, _ hi: Double) -> Double {
    return min(max(value, lo), hi)
}

// Central finite difference derivative
func finiteDiff1(_ f: (Double) -> Double, at x: Double, dx: Double = 1e-6) -> Double {
    let f1 = f(x + dx)
    let f2 = f(x - dx)
    return (f1 - f2) / (2.0 * dx)
}

// Second derivative via central difference
func finiteDiff2(_ f: (Double) -> Double, at x: Double, dx: Double = 1e-4) -> Double {
    let f1 = f(x + dx)
    let f0 = f(x)
    let f_1 = f(x - dx)
    return (f1 - 2.0 * f0 + f_1) / (dx * dx)
}

// MARK: - Dense Layer

final class DenseLayer {
    var weights: [[Double]] // shape: [outputSize][inputSize]
    var biases: [Double]    // shape: [outputSize]

    init(inputSize: Int, outputSize: Int) {
        let bound = sqrt(6.0 / Double(inputSize + outputSize)) // Xavier uniform
        self.weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Double.random(in: -bound...bound) }
        }
        self.biases = Array(repeating: 0.0, count: outputSize)
    }

    func forward(_ input: [Double]) -> [Double] {
        precondition(input.count == weights.first?.count ?? 0)
        var output = [Double](repeating: 0.0, count: weights.count)
        for o in 0..<weights.count {
            output[o] = dot(weights[o], input) + biases[o]
        }
        return output
    }
}

// MARK: - PINN Model

final class PINN {
    var layers: [DenseLayer]

    init() {
        self.layers = [
            DenseLayer(inputSize: 2, outputSize: 20),
            DenseLayer(inputSize: 20, outputSize: 20),
            DenseLayer(inputSize: 20, outputSize: 1)
        ]
    }

    // Forward pass: tanh on hidden layers, linear output
    func forward(x: Double, t: Double) -> Double {
        var activations: [Double] = [x, t]
        for (idx, layer) in layers.enumerated() {
            let z = layer.forward(activations)
            if idx < layers.count - 1 {
                activations = tanhArray(z)
            } else {
                activations = z // linear output
            }
        }
        return activations[0]
    }
}

// MARK: - PDE Setup (1D Heat Equation: u_t = u_xx)

struct PDESetup {
    // Analytical solution for initial condition u(x,0) = -sin(pi x) under homogeneous boundaries on [-1, 1]
    // For demonstration we use u(x,t) = -exp(-pi^2 t) * sin(pi x)
    static func initialCondition(_ x: Double) -> Double {
        return -sin(.pi * x)
    }

    static func analytical(_ x: Double, _ t: Double) -> Double {
        return -exp(-Double.pi * Double.pi * t) * sin(.pi * x)
    }
}

// MARK: - Losses

// PDE residual loss: mean((u_t - u_xx)^2)
func pdeLoss(model: PINN, xs: [Double], ts: [Double], dx: Double = 1e-3, dt: Double = 1e-3) -> Double {
    precondition(xs.count == ts.count)
    if xs.isEmpty { return 0.0 }

    var lossSum = 0.0
    for i in 0..<xs.count {
        let x = xs[i]
        let t = ts[i]
        // avoid boundaries for t +/- dt
        let tMinus = clamp(t - dt, 0.0 + dt, 1.0 - dt)
        let tPlus  = clamp(t + dt, 0.0 + dt, 1.0 - dt)

        // u_t via central diff in t
        let ut = (model.forward(x: x, t: tPlus) - model.forward(x: x, t: tMinus)) / (2.0 * dt)

        // u_xx via second diff in x
        let uxx = finiteDiff2({ xx in model.forward(x: xx, t: t) }, at: x, dx: dx)

        let r = ut - uxx
        lossSum += r * r
    }
    return lossSum / Double(xs.count)
}

// Initial condition MSE at t = 0
func icLoss(model: PINN, xs: [Double]) -> Double {
    if xs.isEmpty { return 0.0 }
    var sum = 0.0
    for x in xs {
        let u0 = model.forward(x: x, t: 0.0)
        let gt = PDESetup.initialCondition(x)
        let e = u0 - gt
        sum += e * e
    }
    return sum / Double(xs.count)
}

// L2 weight decay as an efficiency/regularization proxy
func l2Weights(model: PINN) -> Double {
    var s = 0.0
    for layer in model.layers {
        for row in layer.weights { for w in row { s += w * w } }
        for b in layer.biases { s += b * b }
    }
    return s
}

// MARK: - Training (Numerical Gradients)

struct TrainConfig {
    var learningRate: Double = 0.005
    var gradEpsilon: Double = 1e-5
    var batchSize: Int = 32
    var lambdaIC: Double = 1.0
    var lambdaEff: Double = 1e-4 // weight decay strength
}

func totalLoss(model: PINN, xs: [Double], ts: [Double], icXs: [Double], cfg: TrainConfig) -> Double {
    let pde = pdeLoss(model: model, xs: xs, ts: ts)
    let ic  = icLoss(model: model, xs: icXs)
    let eff = l2Weights(model: model)
    return pde + cfg.lambdaIC * ic + cfg.lambdaEff * eff
}

// Mini-batch sampler
func sampleBatch(xs: [Double], ts: [Double], batchSize: Int) -> ([Double], [Double]) {
    precondition(xs.count == ts.count)
    if xs.isEmpty { return ([], []) }
    let n = xs.count
    var idxs = Array(0..<n)
    idxs.shuffle()
    let k = min(batchSize, n)
    let batchIdxs = Array(idxs.prefix(k))
    let bx = batchIdxs.map { xs[$0] }
    let bt = batchIdxs.map { ts[$0] }
    return (bx, bt)
}

func trainStep(model: PINN, fullXs: [Double], fullTs: [Double], icXs: [Double], cfg: TrainConfig) {
    let (xs, ts) = sampleBatch(xs: fullXs, ts: fullTs, batchSize: cfg.batchSize)
    let baseLoss = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
    let eps = cfg.gradEpsilon

    for layer in model.layers {
        // weights
        for o in 0..<layer.weights.count {
            for i in 0..<layer.weights[o].count {
                layer.weights[o][i] += eps
                let lp = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
                layer.weights[o][i] -= 2.0 * eps
                let lm = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
                // restore
                layer.weights[o][i] += eps
                let grad = (lp - lm) / (2.0 * eps)
                layer.weights[o][i] -= cfg.learningRate * grad
            }
        }
        // biases
        for o in 0..<layer.biases.count {
            layer.biases[o] += eps
            let lp = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
            layer.biases[o] -= 2.0 * eps
            let lm = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
            // restore
            layer.biases[o] += eps
            let grad = (lp - lm) / (2.0 * eps)
            layer.biases[o] -= cfg.learningRate * grad
        }
    }

    // Optional: sanity check base vs new loss; not strictly required
    _ = baseLoss
}

func train(model: PINN,
           epochs: Int,
           xs: [Double], ts: [Double],
           icXs: [Double],
           printEvery: Int = 50,
           cfg: TrainConfig = TrainConfig()) {
    for epoch in 1...epochs {
        trainStep(model: model, fullXs: xs, fullTs: ts, icXs: icXs, cfg: cfg)
        if epoch % printEvery == 0 || epoch == 1 {
            let l = totalLoss(model: model, xs: xs, ts: ts, icXs: icXs, cfg: cfg)
            print("Epoch \(epoch) | Loss: \(String(format: "%.6f", l))")
        }
    }
}

// MARK: - Hybrid Metric

// Computes Ψ = O_hybrid * exp(-P_total) * P_adj
// where O_hybrid = α N + (1-α) S; P_total = λ1 R_cognitive + λ2 R_efficiency; P_adj = min(1, P^β)
func hybridMetric(S: Double, N: Double, alpha: Double,
                  lambda1: Double, lambda2: Double,
                  R_cognitive: Double, R_efficiency: Double,
                  P: Double, beta: Double) -> Double {
    let O = alpha * N + (1.0 - alpha) * S
    let Ptotal = lambda1 * R_cognitive + lambda2 * R_efficiency
    let expFactor = exp(-Ptotal)
    let Padj = min(1.0, pow(P, beta))
    return O * expFactor * Padj
}

// MARK: - Data Generation

func linspace(_ start: Double, _ end: Double, _ num: Int) -> [Double] {
    precondition(num >= 2)
    if num == 2 { return [start, end] }
    let step = (end - start) / Double(num - 1)
    return (0..<num).map { i in start + Double(i) * step }
}

func gridXT(xMin: Double, xMax: Double, nx: Int,
            tMin: Double, tMax: Double, nt: Int,
            avoidTEdgesBy: Double = 1e-3) -> ([Double], [Double]) {
    let xs = linspace(xMin, xMax, nx)
    var ts = linspace(tMin, tMax, nt)
    // avoid exact edges for finite diffs
    ts = ts.map { clamp($0, tMin + avoidTEdgesBy, tMax - avoidTEdgesBy) }
    var gx = [Double]()
    var gt = [Double]()
    gx.reserveCapacity(nx * nt)
    gt.reserveCapacity(nx * nt)
    for t in ts {
        for x in xs {
            gx.append(x)
            gt.append(t)
        }
    }
    return (gx, gt)
}

// MARK: - Demo main (CLI)

#if !canImport(SwiftUI)
// On Linux or without SwiftUI, run a CLI training demo
let model = PINN()
let (gx, gt) = gridXT(xMin: -1.0, xMax: 1.0, nx: 25, tMin: 0.0, tMax: 1.0, nt: 25)
let icXs = linspace(-1.0, 1.0, 50)

let cfg = TrainConfig(learningRate: 0.005, gradEpsilon: 1e-5, batchSize: 32, lambdaIC: 1.0, lambdaEff: 1e-4)
print("Starting training (CLI)...")
train(model: model, epochs: 1000, xs: gx, ts: gt, icXs: icXs, printEvery: 50, cfg: cfg)

// Evaluate hybrid metric example provided by the user
let S_ex = 0.72
let N_ex = 0.85
let alpha_ex = 0.5
let Rc_ex = 0.15
let Re_ex = 0.10
let l1_ex = 0.6
let l2_ex = 0.4
let P_ex = 0.80
let beta_ex = 1.2
let psi = hybridMetric(S: S_ex, N: N_ex, alpha: alpha_ex, lambda1: l1_ex, lambda2: l2_ex, R_cognitive: Rc_ex, R_efficiency: Re_ex, P: P_ex, beta: beta_ex)
print(String(format: "Hybrid Ψ ≈ %.3f", psi))

// Print sample predictions at t=1.0 for quick inspection
let sampleX = linspace(-1.0, 1.0, 21)
var pred: [Double] = []
var exact: [Double] = []
for x in sampleX {
    pred.append(model.forward(x: x, t: 1.0))
    exact.append(PDESetup.analytical(x, 1.0))
}
print("x, u_PINN(t=1), u_exact(t=1)")
for i in 0..<sampleX.count {
    print(String(format: "% .3f, % .3f, % .3f", sampleX[i], pred[i], exact[i]))
}
#endif

// MARK: - SwiftUI Visualization (Xcode)

#if canImport(SwiftUI)
import SwiftUI
import Charts

struct SeriesPoint: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
    let label: String
}

struct SolutionChart: View {
    var body: some View {
        let xVals: [Double] = stride(from: -1.0, through: 1.0, by: 0.1).map { $0 }
        // Example hard-coded arrays similar to the description. Replace with real outputs in-app if desired.
        let pinnU: [Double] = [0.0, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.4, 0.1, -0.3, -0.6, -0.8, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.0]
        let rk4U:  [Double] = [0.0, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.3, 0.0, -0.4, -0.7, -0.8, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.0]

        let pinnSeries = zip(xVals, pinnU).map { SeriesPoint(x: $0.0, y: $0.1, label: "PINN") }
        let rk4Series  = zip(xVals, rk4U).map { SeriesPoint(x: $0.0, y: $0.1, label: "RK4") }

        VStack(alignment: .leading) {
            Text("PINN vs RK4 at t = 1").font(.headline)
            Chart {
                ForEach(pinnSeries) { p in
                    LineMark(x: .value("x", p.x), y: .value("u", p.y))
                        .foregroundStyle(.blue)
                        .interpolationMethod(.catmullRom)
                }
                ForEach(rk4Series) { p in
                    LineMark(x: .value("x", p.x), y: .value("u", p.y))
                        .foregroundStyle(.red)
                        .interpolationMethod(.catmullRom)
                }
            }
            .frame(height: 280)
            .padding(.vertical)
            Text("Blue: PINN (hard-coded sample). Red: RK4 (hard-coded sample). Replace with trained outputs.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

struct SolutionChart_Previews: PreviewProvider {
    static var previews: some View {
        SolutionChart()
    }
}
#endif