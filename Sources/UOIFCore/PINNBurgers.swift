// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public struct PINNHybridMetrics {
  public let S_symbolic: Double
  public let N_external: Double
  public let alpha: Double
  public let hybrid: Double
  public let penalty: Double
  public let posterior: Double
  public let psi: Double
  public let baseLoss: Double
  public let newLoss: Double
}

private enum Activation {
  case tanh
  case identity

  func apply(_ x: Double) -> Double {
    switch self {
    case .tanh: return tanh(x)
    case .identity: return x
    }
  }
}

private struct DenseLayer {
  var weights: [[Double]] // [output][input]
  var biases: [Double]    // [output]
  let activation: Activation

  init(inputSize: Int, outputSize: Int, activation: Activation, rng: inout RandomNumberGenerator) {
    // Xavier uniform init
    let limit = sqrt(6.0 / Double(inputSize + outputSize))
    var w: [[Double]] = []
    for _ in 0..<outputSize {
      var row: [Double] = []
      for _ in 0..<inputSize {
        let r = Double.random(in: -limit...limit, using: &rng)
        row.append(r)
      }
      w.append(row)
    }
    let b = (0..<outputSize).map { _ in 0.0 }
    self.weights = w
    self.biases = b
    self.activation = activation
  }

  func forward(_ input: [Double]) -> [Double] {
    let outputSize = biases.count
    let inputSize = input.count
    var out = Array(repeating: 0.0, count: outputSize)
    for o in 0..<outputSize {
      var s = biases[o]
      for i in 0..<inputSize {
        s += weights[o][i] * input[i]
      }
      out[o] = activation.apply(s)
    }
    return out
  }
}

public final class PINNBurgers {
  private var layers: [DenseLayer]

  public init(hiddenWidth: Int = 16, hiddenDepth: Int = 2, rngSeed: UInt64 = 42) {
    var rng = SeededGenerator(seed: rngSeed)
    var built: [DenseLayer] = []
    // Input: (x, t)
    var last = 2
    for _ in 0..<hiddenDepth {
      built.append(DenseLayer(inputSize: last, outputSize: hiddenWidth, activation: .tanh, rng: &rng))
      last = hiddenWidth
    }
    built.append(DenseLayer(inputSize: last, outputSize: 1, activation: .identity, rng: &rng))
    self.layers = built
  }

  public func forward(x: Double, t: Double) -> Double {
    var h = [x, t]
    for layer in layers {
      h = layer.forward(h)
    }
    return h[0]
  }

  private func finiteDiff(_ f: (Double) -> Double, at x: Double, dx: Double) -> Double {
    let f1 = f(x + dx)
    let f2 = f(x - dx)
    return (f1 - f2) / (2.0 * dx)
  }

  private func pdeResidual(x: Double, t: Double, h: Double = 1e-4) -> Double {
    let u = forward(x: x, t: t)
    let ut = finiteDiff({ self.forward(x: x, t: $0) }, at: t, dx: h)
    let ux = finiteDiff({ self.forward(x: $0, t: t) }, at: x, dx: h)
    return ut + u * ux
  }

  private func initialCondition(_ x: Double) -> Double {
    return -sin(Double.pi * x)
  }

  private func loss(pdeXs: [Double], pdeTs: [Double], icXs: [Double]) -> (pde: Double, ic: Double, total: Double) {
    var pdeAccum = 0.0
    for t in pdeTs {
      for x in pdeXs {
        let r = pdeResidual(x: x, t: t)
        pdeAccum += r * r
      }
    }
    let pdeCount = Double(pdeTs.count * pdeXs.count)
    let pdeLoss = pdeAccum / max(1.0, pdeCount)

    var icAccum = 0.0
    for x in icXs {
      let u0 = initialCondition(x)
      let u = forward(x: x, t: 0.0)
      let d = u - u0
      icAccum += d * d
    }
    let icLoss = icAccum / max(1.0, Double(icXs.count))

    let total = pdeLoss + icLoss
    return (pdeLoss, icLoss, total)
  }

  public func trainStepFiniteDiff(learningRate: Double, epsilon: Double, pdeXs: [Double], pdeTs: [Double], icXs: [Double]) -> (baseLoss: Double, newLoss: Double, pdeLoss: Double) {
    // Compute base loss
    let base = loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs)
    var currentLoss = base.total

    // Update each parameter using central difference gradient approximation
    for li in 0..<layers.count {
      // Weights
      for o in 0..<layers[li].weights.count {
        for i in 0..<layers[li].weights[o].count {
          var plusModel = self
          plusModel.layers[li].weights[o][i] += epsilon
          let Lplus = plusModel.loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs).total

          var minusModel = self
          minusModel.layers[li].weights[o][i] -= epsilon
          let Lminus = minusModel.loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs).total

          let grad = (Lplus - Lminus) / (2.0 * epsilon)
          layers[li].weights[o][i] -= learningRate * grad
        }
      }
      // Biases
      for o in 0..<layers[li].biases.count {
        var plusModel = self
        plusModel.layers[li].biases[o] += epsilon
        let Lplus = plusModel.loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs).total

        var minusModel = self
        minusModel.layers[li].biases[o] -= epsilon
        let Lminus = minusModel.loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs).total

        let grad = (Lplus - Lminus) / (2.0 * epsilon)
        layers[li].biases[o] -= learningRate * grad
      }
    }

    // New loss after update
    let after = loss(pdeXs: pdeXs, pdeTs: pdeTs, icXs: icXs)
    currentLoss = after.total
    return (base.total, currentLoss, base.pde)
  }

  // Semi-discrete baseline using RK4 on u_t + (0.5 u^2)_x = 0 with central differences (periodic)
  public static func rk4Baseline(u0: [Double], dx: Double, dt: Double, steps: Int) -> [Double] {
    func dudt(_ u: [Double]) -> [Double] {
      let n = u.count
      var du = Array(repeating: 0.0, count: n)
      // central difference for flux derivative
      for i in 0..<n {
        let ip = (i + 1) % n
        let im = (i - 1 + n) % n
        let fip = 0.5 * u[ip] * u[ip]
        let fim = 0.5 * u[im] * u[im]
        du[i] = -(fip - fim) / (2.0 * dx)
      }
      return du
    }

    var u = u0
    for _ in 0..<steps {
      let k1 = dudt(u)
      let u2 = zip(u, k1).map { $0 + 0.5 * dt * $1 }
      let k2 = dudt(u2)
      let u3 = zip(u, k2).map { $0 + 0.5 * dt * $1 }
      let k3 = dudt(u3)
      let u4 = zip(u, k3).map { $0 + dt * $1 }
      let k4 = dudt(u4)
      for i in 0..<u.count {
        u[i] = u[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
      }
    }
    return u
  }

  public static func runSingleStepDemo(alpha: Double = 0.5) -> PINNHybridMetrics {
    // Domain and sampling
    let numX = 24
    let numTX = 12
    let xs = (0..<numX).map { Double($0) / Double(numX - 1) } // [0,1]
    let ts = (1..<numTX).map { Double($0) / Double(numTX) }    // (0,1]

    // Build model
    let model = PINNBurgers(hiddenWidth: 8, hiddenDepth: 2, rngSeed: 7)
    var mutableModel = model

    // One training step
    let step = mutableModel.trainStepFiniteDiff(learningRate: 1e-2, epsilon: 1e-4, pdeXs: xs, pdeTs: ts, icXs: xs)

    // Derive S and N heuristics
    let pdeScore = exp(-step.pdeLoss) // in (0,1]
    let improvement = max(0.0, step.baseLoss - step.newLoss)
    let N = min(1.0, 0.5 + 0.5 * tanh(5.0 * improvement))
    let S = max(0.0, min(1.0, pdeScore))

    // Hybrid and Psi using example penalty/posterior settings
    let lambda1 = 0.6
    let lambda2 = 0.4
    let R_cognitive = 0.15
    let R_efficiency = 0.10
    let penalty = PsiModel.computePenalty(
      lambdaAuthority: lambda1,
      lambdaVerifiability: lambda2,
      riskAuthority: R_cognitive,
      riskVerifiability: R_efficiency
    )

    let posterior = PsiModel.computePosteriorCapped(basePosterior: 0.80, beta: 1.2)
    let hybrid = PsiModel.computeHybrid(alpha: alpha, S: S, N: N)
    let psi = hybrid * penalty * posterior

    return PINNHybridMetrics(
      S_symbolic: S,
      N_external: N,
      alpha: alpha,
      hybrid: hybrid,
      penalty: penalty,
      posterior: posterior,
      psi: psi,
      baseLoss: step.baseLoss,
      newLoss: step.newLoss
    )
  }
}

private struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64
  init(seed: UInt64) { self.state = seed == 0 ? 0x123456789ABCDEF : seed }
  mutating func next() -> UInt64 {
    // Xorshift64*
    var x = state
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state = x
    return x &* 2685821657736338717
  }
}
