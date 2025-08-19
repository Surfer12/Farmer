import Foundation

// MARK: - Utilities

@inline(__always)
func randUniform01() -> Double {
    return Double.random(in: 0.0...1.0)
}

@inline(__always)
func randUniform(_ low: Double, _ high: Double) -> Double {
    return low + (high - low) * randUniform01()
}

@inline(__always)
func randSign() -> Double { return Bool.random() ? 1.0 : -1.0 }

// MARK: - Dense Layer

final class DenseLayer {
    var weights: [[Double]] // [out][in]
    var biases: [Double]
    let inputSize: Int
    let outputSize: Int
    let useTanh: Bool

    init(inputSize: Int, outputSize: Int, useTanh: Bool) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.useTanh = useTanh
        self.weights = Array(repeating: Array(repeating: 0.0, count: inputSize), count: outputSize)
        self.biases = Array(repeating: 0.0, count: outputSize)
        // Xavier/Glorot uniform init for tanh layers; simple scaled init for linear
        let limit: Double
        if useTanh {
            limit = sqrt(6.0 / Double(inputSize + outputSize))
        } else {
            limit = sqrt(2.0 / Double(inputSize))
        }
        for o in 0..<outputSize {
            for i in 0..<inputSize {
                weights[o][i] = randUniform(-limit, limit)
            }
            biases[o] = 0.0
        }
    }

    func forward(_ input: [Double]) -> [Double] {
        precondition(input.count == inputSize)
        var out = Array(repeating: 0.0, count: outputSize)
        for o in 0..<outputSize {
            var s = biases[o]
            let row = weights[o]
            for i in 0..<inputSize {
                s += row[i] * input[i]
            }
            out[o] = useTanh ? tanh(s) : s
        }
        return out
    }
}

// MARK: - PINN Model

final class PINN {
    var layers: [DenseLayer] = []

    init(sizes: [Int]) {
        precondition(sizes.count >= 2)
        for i in 0..<(sizes.count - 1) {
            let isLast = i == sizes.count - 2
            let layer = DenseLayer(inputSize: sizes[i], outputSize: sizes[i + 1], useTanh: !isLast)
            layers.append(layer)
        }
    }

    func forward(x: Double, t: Double) -> Double {
        var v: [Double] = [x, t]
        for layer in layers {
            v = layer.forward(v)
        }
        // Output is scalar
        return v[0]
    }

    // Flatten parameters for SPSA
    func flattenedParameters() -> [Double] {
        var theta: [Double] = []
        for layer in layers {
            for o in 0..<layer.outputSize {
                for i in 0..<layer.inputSize {
                    theta.append(layer.weights[o][i])
                }
            }
            for b in 0..<layer.outputSize {
                theta.append(layer.biases[b])
            }
        }
        return theta
    }

    func assignFlattenedParameters(_ theta: [Double]) {
        var idx = 0
        for layer in layers {
            for o in 0..<layer.outputSize {
                for i in 0..<layer.inputSize {
                    layer.weights[o][i] = theta[idx]
                    idx += 1
                }
            }
            for b in 0..<layer.outputSize {
                layer.biases[b] = theta[idx]
                idx += 1
            }
        }
        precondition(idx == theta.count)
    }

    func parameterCount() -> Int {
        var total = 0
        for layer in layers {
            total += layer.outputSize * layer.inputSize
            total += layer.outputSize
        }
        return total
    }
}

// MARK: - Finite Differences for PDE Residuals

@inline(__always)
func partialDerivative_x(of model: PINN, x: Double, t: Double, dx: Double) -> Double {
    let uPlus = model.forward(x: x + dx, t: t)
    let uMinus = model.forward(x: x - dx, t: t)
    return (uPlus - uMinus) / (2.0 * dx)
}

@inline(__always)
func partialDerivative_t(of model: PINN, x: Double, t: Double, dt: Double) -> Double {
    let uPlus = model.forward(x: x, t: t + dt)
    let uMinus = model.forward(x: x, t: t - dt)
    return (uPlus - uMinus) / (2.0 * dt)
}

// MARK: - Losses

struct LossConfig {
    let pdeWeight: Double
    let icWeight: Double
    let dx: Double
    let dt: Double
    let numCollocation: Int
    let numIC: Int
    let xMin: Double
    let xMax: Double
    let tMin: Double
    let tMax: Double
}

@inline(__always)
func initialConditionU(x: Double) -> Double {
    // u(x, 0) = -sin(pi x)
    return -sin(Double.pi * x)
}

func pdeResidualMSE(model: PINN, cfg: LossConfig) -> Double {
    var mse = 0.0
    let n = max(1, cfg.numCollocation)
    for _ in 0..<n {
        let x = randUniform(cfg.xMin, cfg.xMax)
        let t = randUniform(cfg.tMin, cfg.tMax)
        let u = model.forward(x: x, t: t)
        let ux = partialDerivative_x(of: model, x: x, t: t, dx: cfg.dx)
        let ut = partialDerivative_t(of: model, x: x, t: t, dt: cfg.dt)
        let r = ut + u * ux // inviscid Burgers: u_t + u u_x = 0
        mse += r * r
    }
    return mse / Double(n)
}

func initialConditionMSE(model: PINN, cfg: LossConfig) -> Double {
    var mse = 0.0
    let n = max(1, cfg.numIC)
    for i in 0..<n {
        let x = cfg.xMin + (Double(i) + 0.5) * (cfg.xMax - cfg.xMin) / Double(n)
        let u = model.forward(x: x, t: 0.0)
        let target = initialConditionU(x: x)
        let e = u - target
        mse += e * e
    }
    return mse / Double(n)
}

func totalLoss(model: PINN, cfg: LossConfig) -> (total: Double, pde: Double, ic: Double) {
    let pde = pdeResidualMSE(model: model, cfg: cfg)
    let ic = initialConditionMSE(model: model, cfg: cfg)
    let total = cfg.pdeWeight * pde + cfg.icWeight * ic
    return (total, pde, ic)
}

// MARK: - SPSA Optimizer (No Autograd)

struct SPSAConfig {
    var a: Double // learning rate
    var c: Double // perturbation magnitude
}

func spsaStep(model: PINN, lossCfg: LossConfig, spsaCfg: SPSAConfig) -> (newLoss: Double, oldLoss: Double) {
    let theta0 = model.flattenedParameters()
    let dim = theta0.count
    var delta: [Double] = Array(repeating: 0.0, count: dim)
    for i in 0..<dim { delta[i] = randSign() }

    func evalAt(_ theta: [Double]) -> Double {
        model.assignFlattenedParameters(theta)
        return totalLoss(model: model, cfg: lossCfg).total
    }

    let c = spsaCfg.c
    var thetaPlus = theta0
    var thetaMinus = theta0
    for i in 0..<dim {
        thetaPlus[i] = theta0[i] + c * delta[i]
        thetaMinus[i] = theta0[i] - c * delta[i]
    }

    let lossPlus = evalAt(thetaPlus)
    let lossMinus = evalAt(thetaMinus)

    // Gradient estimate: g_i ≈ (L+ - L-) / (2c) * (1/delta_i)
    let scale = (lossPlus - lossMinus) / (2.0 * c)
    var grad: [Double] = Array(repeating: 0.0, count: dim)
    for i in 0..<dim {
        grad[i] = scale * (1.0 / delta[i])
    }

    let a = spsaCfg.a
    var thetaNew = theta0
    for i in 0..<dim {
        thetaNew[i] = theta0[i] - a * grad[i]
    }

    // Evaluate new loss and assign
    let oldLoss = evalAt(theta0)
    let newLoss = evalAt(thetaNew)
    model.assignFlattenedParameters(thetaNew)

    return (newLoss, oldLoss)
}

// MARK: - RK4 Method-of-Lines Validator

func centralDiffPeriodic(_ u: [Double], dx: Double) -> [Double] {
    let n = u.count
    var ux = Array(repeating: 0.0, count: n)
    for i in 0..<n {
        let ip = (i + 1) % n
        let im = (i - 1 + n) % n
        ux[i] = (u[ip] - u[im]) / (2.0 * dx)
    }
    return ux
}

func rhsBurgersInviscid(u: [Double], dx: Double) -> [Double] {
    let ux = centralDiffPeriodic(u, dx: dx)
    var du = Array(repeating: 0.0, count: u.count)
    for i in 0..<u.count {
        du[i] = -u[i] * ux[i]
    }
    return du
}

func rk4StepVector(u: [Double], dt: Double, dx: Double) -> [Double] {
    let k1 = rhsBurgersInviscid(u: u, dx: dx)
    var u2 = u
    for i in 0..<u.count { u2[i] = u[i] + 0.5 * dt * k1[i] }
    let k2 = rhsBurgersInviscid(u: u2, dx: dx)
    var u3 = u
    for i in 0..<u.count { u3[i] = u[i] + 0.5 * dt * k2[i] }
    let k3 = rhsBurgersInviscid(u: u3, dx: dx)
    var u4 = u
    for i in 0..<u.count { u4[i] = u[i] + dt * k3[i] }
    let k4 = rhsBurgersInviscid(u: u4, dx: dx)

    var out = u
    for i in 0..<u.count {
        out[i] = u[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    }
    return out
}

func rk4Integrate(xMin: Double, xMax: Double, n: Int, tFinal: Double, dt: Double) -> (x: [Double], u: [Double]) {
    let dx = (xMax - xMin) / Double(n)
    var x: [Double] = Array(repeating: 0.0, count: n)
    var u: [Double] = Array(repeating: 0.0, count: n)
    for i in 0..<n {
        x[i] = xMin + Double(i) * dx
        u[i] = initialConditionU(x: x[i])
    }
    var t = 0.0
    while t < tFinal - 1e-12 {
        u = rk4StepVector(u: u, dt: dt, dx: dx)
        t += dt
    }
    return (x, u)
}

// MARK: - Hybrid Metric Ψ(x)

struct HybridInputs {
    let S: Double
    let N: Double
    let alpha: Double
    let R_cognitive: Double
    let R_efficiency: Double
    let lambda1: Double
    let lambda2: Double
    let P: Double
    let beta: Double
}

struct HybridOutput {
    let O_hybrid: Double
    let P_total: Double
    let expTerm: Double
    let P_adj: Double
    let psi: Double
}

func computeHybrid(_ h: HybridInputs) -> HybridOutput {
    let O_hybrid = h.alpha * h.S + (1.0 - h.alpha) * h.N
    let P_total = h.lambda1 * h.R_cognitive + h.lambda2 * h.R_efficiency
    let expTerm = exp(-P_total)
    let P_adj = min(1.0, pow(h.P, h.beta))
    let psi = O_hybrid * expTerm * P_adj
    return HybridOutput(O_hybrid: O_hybrid, P_total: P_total, expTerm: expTerm, P_adj: P_adj, psi: psi)
}

// MARK: - Main Demo

func main() {
    // 1) Build model
    let model = PINN(sizes: [2, 32, 32, 1])

    // 2) Loss configuration
    let lossCfg = LossConfig(
        pdeWeight: 1.0,
        icWeight: 1.0,
        dx: 1e-3,
        dt: 1e-3,
        numCollocation: 256,
        numIC: 128,
        xMin: -1.0,
        xMax: 1.0,
        tMin: 0.0,
        tMax: 1.0
    )

    // 3) Evaluate initial loss and run a few SPSA steps
    let initial = totalLoss(model: model, cfg: lossCfg)
    print("Initial Loss -> total: \(String(format: "%.6f", initial.total)), pde: \(String(format: "%.6f", initial.pde)), ic: \(String(format: "%.6f", initial.ic)))")

    var spsaCfg = SPSAConfig(a: 1e-2, c: 1e-3)
    let steps = 5 // keep small for demo
    for step in 1...steps {
        let (newL, oldL) = spsaStep(model: model, lossCfg: lossCfg, spsaCfg: spsaCfg)
        print("SPSA step \(step): old=\(String(format: "%.6f", oldL)) new=\(String(format: "%.6f", newL))")
        // simple anneal
        spsaCfg.a *= 0.9
        spsaCfg.c *= 0.95
    }

    let final = totalLoss(model: model, cfg: lossCfg)
    print("Final Loss -> total: \(String(format: "%.6f", final.total)), pde: \(String(format: "%.6f", final.pde)), ic: \(String(format: "%.6f", final.ic)))")

    // 4) RK4 validator at t=0.5
    let tFinal = 0.5
    let gridN = 101
    let dt = 1e-3
    let (xRef, uRef) = rk4Integrate(xMin: -1.0, xMax: 1.0, n: gridN, tFinal: tFinal, dt: dt)

    // Model predictions at tFinal
    var uPred: [Double] = Array(repeating: 0.0, count: gridN)
    for i in 0..<gridN {
        uPred[i] = model.forward(x: xRef[i], t: tFinal)
    }

    var mseRef = 0.0
    for i in 0..<gridN {
        let e = uPred[i] - uRef[i]
        mseRef += e * e
    }
    mseRef /= Double(gridN)
    print("RK4 vs PINN MSE at t=\(tFinal): \(String(format: "%.6f", mseRef)))")

    // 5) Save CSV for plotting
    let csvPath = "rk4_vs_pinn.csv"
    var csv = "x,u_ref,u_pinn\n"
    for i in 0..<gridN {
        csv += "\(xRef[i]),\(uRef[i]),\(uPred[i])\n"
    }
    do {
        try csv.write(to: URL(fileURLWithPath: csvPath), atomically: true, encoding: .utf8)
        print("Saved CSV: \(csvPath)")
    } catch {
        fputs("Failed to write CSV: \(error)\n", stderr)
    }

    // 6) Hybrid metric demonstration with your numerical example
    let h = HybridInputs(
        S: 0.72,
        N: 0.85,
        alpha: 0.5,
        R_cognitive: 0.15,
        R_efficiency: 0.10,
        lambda1: 0.6,
        lambda2: 0.4,
        P: 0.80,
        beta: 1.2
    )
    let out = computeHybrid(h)
    print("Hybrid components:")
    print("  O_hybrid = \(String(format: "%.3f", out.O_hybrid)) (expected ~0.785)")
    print("  P_total  = \(String(format: "%.3f", out.P_total)) (expected ~0.130)")
    print("  exp(-P)  = \(String(format: "%.3f", out.expTerm)) (expected ~0.878)")
    print("  P_adj    = \(String(format: "%.3f", out.P_adj)) (expected ~0.960)")
    print("  Psi      = \(String(format: "%.3f", out.psi)) (expected ~0.662)")
}

main()