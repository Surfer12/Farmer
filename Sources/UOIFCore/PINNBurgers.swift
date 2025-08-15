// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

// Minimal feedforward network with tanh activations for u(x, t)
struct DenseLayer {
  var weights: [[Double]] // shape: outDim x inDim
  var biases: [Double]    // shape: outDim

  init(inputSize: Int, outputSize: Int, rng: inout RandomNumberGenerator) {
    weights = Array(repeating: Array(repeating: 0.0, count: inputSize), count: outputSize)
    biases = Array(repeating: 0.0, count: outputSize)
    let scale = 1.0 / sqrt(Double(inputSize))
    for o in 0..<outputSize {
      for i in 0..<inputSize {
        weights[o][i] = Double.random(in: -scale...scale, using: &rng)
      }
      biases[o] = Double.random(in: -scale...scale, using: &rng)
    }
  }

  func forward(_ input: [Double]) -> [Double] {
    let outDim = weights.count
    let inDim = input.count
    precondition(outDim == biases.count)
    precondition(inDim == (weights.first?.count ?? 0))
    var output = Array(repeating: 0.0, count: outDim)
    for o in 0..<outDim {
      var sum = biases[o]
      for i in 0..<inDim {
        sum += weights[o][i] * input[i]
      }
      output[o] = sum
    }
    return output
  }
}

struct FeedForwardNet {
  var layers: [DenseLayer]

  init(sizes: [Int], seed: UInt64 = 42) {
    precondition(sizes.count >= 2)
    var rng = SeededGenerator(seed: seed)
    layers = []
    for i in 0..<(sizes.count - 1) {
      layers.append(DenseLayer(inputSize: sizes[i], outputSize: sizes[i + 1], rng: &rng))
    }
  }

  func forward(_ input: [Double]) -> [Double] {
    precondition(input.count == 2) // [x, t]
    var a = input
    for idx in 0..<(layers.count - 1) {
      a = tanhVec(layers[idx].forward(a))
    }
    // last layer linear
    a = layers.last!.forward(a)
    return a
  }
}

// Utilities
private func tanhVec(_ v: [Double]) -> [Double] {
  return v.map { tanh($0) }
}

private struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64
  init(seed: UInt64) { self.state = seed &+ 0x9E3779B97F4A7C15 }
  mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
  }
}

// Central finite differences for partials by perturbing x or t
private func centralDiff(_ f: (Double) -> Double, at x: Double, h: Double) -> Double {
  return (f(x + h) - f(x - h)) / (2.0 * h)
}

public struct PINNBurgers1D {
  public let net: FeedForwardNet
  public let eps: Double

  public init(hiddenWidth: Int = 20, hiddenLayers: Int = 2, seed: UInt64 = 42, eps: Double = 1e-5) {
    precondition(hiddenLayers >= 1)
    var sizes: [Int] = [2]
    for _ in 0..<hiddenLayers { sizes.append(hiddenWidth) }
    sizes.append(1)
    self.net = FeedForwardNet(sizes: sizes, seed: seed)
    self.eps = eps
  }

  public func u(x: Double, t: Double) -> Double {
    return net.forward([x, t])[0]
  }

  public func u_x(x: Double, t: Double) -> Double {
    let h = eps
    return centralDiff({ xx in u(x: xx, t: t) }, at: x, h: h)
  }

  public func u_t(x: Double, t: Double) -> Double {
    let h = eps
    return centralDiff({ tt in u(x: x, t: tt) }, at: t, h: h)
  }

  // Inviscid Burgers residual: r = u_t + u * u_x
  public func residual(x: Double, t: Double) -> Double {
    let value = u(x: x, t: t)
    return u_t(x: x, t: t) + value * u_x(x: x, t: t)
  }

  public static func initialCondition(x: Double) -> Double {
    return -sin(.pi * x)
  }

  public func initialLoss(xs: [Double]) -> Double {
    var loss = 0.0
    for x in xs {
      let pred = u(x: x, t: 0.0)
      let truth = Self.initialCondition(x: x)
      let diff = pred - truth
      loss += diff * diff
    }
    return loss / Double(max(xs.count, 1))
  }

  public func pdeLoss(xs: [Double], ts: [Double]) -> Double {
    var loss = 0.0
    var count = 0
    for x in xs {
      for t in ts {
        let r = residual(x: x, t: t)
        loss += r * r
        count += 1
      }
    }
    return count > 0 ? loss / Double(count) : 0.0
  }
}

public enum RK4 {
  // One RK4 step for ODE y' = f(t, y)
  public static func step(f: (Double, [Double]) -> [Double], y: [Double], t: Double, dt: Double) -> [Double] {
    let k1 = f(t, y)
    let y2 = zip(y, k1).map { $0 + 0.5 * dt * $1 }
    let k2 = f(t + 0.5 * dt, y2)
    let y3 = zip(y, k2).map { $0 + 0.5 * dt * $1 }
    let k3 = f(t + 0.5 * dt, y3)
    let y4 = zip(y, k3).map { $0 + dt * $1 }
    let k4 = f(t + dt, y4)
    var out = y
    for i in 0..<y.count {
      out[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
    }
    return out
  }
}

public enum PINNDemo {
  public static func run() {
    // Sample a tiny collocation grid
    let xs = stride(from: -1.0, through: 1.0, by: 0.5).map { $0 }
    let ts = stride(from: 0.0, through: 1.0, by: 0.5).map { $0 }

    let model = PINNBurgers1D(hiddenWidth: 16, hiddenLayers: 2, seed: 123)
    let lossIC = model.initialLoss(xs: xs)
    let lossPDE = model.pdeLoss(xs: xs, ts: ts)

    print("PINN demo — Burgers (inviscid, residual via finite differences)")
    print(String(format: "initialLoss=%.4f, pdeLoss=%.4f", lossIC, lossPDE))

    // Simple RK4 utility demo on a toy ODE dy/dt = -y
    let y0 = [1.0]
    let f: (Double, [Double]) -> [Double] = { _, y in [-y[0]] }
    let y1 = RK4.step(f: f, y: y0, t: 0.0, dt: 0.1)
    print(String(format: "RK4 demo: y(0)=%.3f -> y(0.1)=%.3f (ref≈%.3f)", y0[0], y1[0], exp(-0.1)))
  }
}