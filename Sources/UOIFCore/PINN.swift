// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public struct DenseLayer {
  public var weights: [[Double]]
  public var biases: [Double]

  public init(inputSize: Int, outputSize: Int, rng: inout RandomNumberGenerator) {
    self.weights = (0..<outputSize).map { _ in
      (0..<inputSize).map { _ in Double.random(in: -0.5...0.5, using: &rng) * 0.1 }
    }
    self.biases = (0..<outputSize).map { _ in Double.random(in: -0.5...0.5, using: &rng) * 0.1 }
  }

  public func forward(_ input: [Double], activation: (Double) -> Double) -> [Double] {
    precondition(weights.first?.count == input.count)
    var out = [Double](repeating: 0.0, count: biases.count)
    for o in 0..<biases.count {
      var acc = biases[o]
      for i in 0..<input.count {
        acc += weights[o][i] * input[i]
      }
      out[o] = activation(acc)
    }
    return out
  }
}

public final class PINNModel {
  public var layers: [DenseLayer]

  public init(hiddenSizes: [Int], rng: inout RandomNumberGenerator) {
    var sizes = [2]
    sizes.append(contentsOf: hiddenSizes)
    sizes.append(1)

    self.layers = []
    for i in 0..<(sizes.count - 1) {
      layers.append(DenseLayer(inputSize: sizes[i], outputSize: sizes[i + 1], rng: &rng))
    }
  }

  public func forward(x: Double, t: Double) -> Double {
    var input: [Double] = [x, t]
    for idx in 0..<(layers.count - 1) {
      input = layers[idx].forward(input) { tanh($0) }
    }
    let finalLayer = layers[layers.count - 1]
    let out = finalLayer.forward(input) { $0 }
    return out[0]
  }
}

public enum FiniteDiff {
  public static func derivative(_ f: (Double) -> Double, at x: Double, dx: Double = 1e-4) -> Double {
    let f1 = f(x + dx)
    let f0 = f(x - dx)
    return (f1 - f0) / (2.0 * dx)
  }
}

public enum Burgers1D {
  // Inviscid Burgers: u_t + u * u_x = 0
  public static func pdeResidualSquared(model: PINNModel, x: Double, t: Double) -> Double {
    let u = model.forward(x: x, t: t)
    let ut = FiniteDiff.derivative({ tt in model.forward(x: x, t: tt) }, at: t)
    let ux = FiniteDiff.derivative({ xx in model.forward(x: xx, t: t) }, at: x)
    let residual = ut + u * ux
    return residual * residual
  }

  public static func initialCondition(x: Double) -> Double {
    return -sin(Double.pi * x)
  }

  public static func icLoss(model: PINNModel, xs: [Double]) -> Double {
    let total = xs.reduce(0.0) { acc, x in
      let u = model.forward(x: x, t: 0.0)
      let uTrue = initialCondition(x: x)
      return acc + pow(u - uTrue, 2)
    }
    return total / Double(max(xs.count, 1))
  }

  public static func pdeLoss(model: PINNModel, xs: [Double], ts: [Double]) -> Double {
    var total = 0.0
    var count = 0
    for x in xs {
      for t in ts {
        total += pdeResidualSquared(model: model, x: x, t: t)
        count += 1
      }
    }
    return total / Double(max(count, 1))
  }

  public static func totalLoss(model: PINNModel, xs: [Double], ts: [Double], icWeight: Double = 1.0, pdeWeight: Double = 1.0) -> Double {
    let lIC = icLoss(model: model, xs: xs)
    let lPDE = pdeLoss(model: model, xs: xs, ts: ts)
    return icWeight * lIC + pdeWeight * lPDE
  }
}

public enum FiniteDiffSGD {
  public static func trainOneStep(model: PINNModel, xs: [Double], ts: [Double], learningRate: Double = 1e-2, epsilon: Double = 1e-4, icWeight: Double = 1.0, pdeWeight: Double = 1.0) -> (lossBefore: Double, lossAfter: Double) {
    let baseLoss = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)

    for li in 0..<model.layers.count {
      // Update weights
      for o in 0..<model.layers[li].weights.count {
        for i in 0..<model.layers[li].weights[o].count {
          let original = model.layers[li].weights[o][i]

          model.layers[li].weights[o][i] = original + epsilon
          let lossPlus = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)

          model.layers[li].weights[o][i] = original - epsilon
          let lossMinus = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)

          let grad = (lossPlus - lossMinus) / (2.0 * epsilon)
          model.layers[li].weights[o][i] = original - learningRate * grad
        }
      }

      // Update biases
      for o in 0..<model.layers[li].biases.count {
        let original = model.layers[li].biases[o]

        model.layers[li].biases[o] = original + epsilon
        let lossPlus = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)

        model.layers[li].biases[o] = original - epsilon
        let lossMinus = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)

        let grad = (lossPlus - lossMinus) / (2.0 * epsilon)
        model.layers[li].biases[o] = original - learningRate * grad
      }
    }

    let newLoss = Burgers1D.totalLoss(model: model, xs: xs, ts: ts, icWeight: icWeight, pdeWeight: pdeWeight)
    return (baseLoss, newLoss)
  }
}

public enum RK4 {
  public static func step(f: (Double, [Double]) -> [Double], t: Double, y: [Double], dt: Double) -> [Double] {
    let k1 = f(t, y)
    let y2 = zip(y, k1).map { $0 + dt * 0.5 * $1 }
    let k2 = f(t + 0.5 * dt, y2)
    let y3 = zip(y, k2).map { $0 + dt * 0.5 * $1 }
    let k3 = f(t + 0.5 * dt, y3)
    let y4 = zip(y, k3).map { $0 + dt * $1 }
    let k4 = f(t + dt, y4)
    return zip(y, zip(k1, zip(k2, zip(k3, k4)))).map { y0, ks in
      let (k1v, rest1) = ks
      let (k2v, rest2) = rest1
      let (k3v, k4v) = rest2
      return y0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    }
  }
}

public enum PINNDemo {
  public static func singleStepDemo(hidden: [Int] = [8, 8], samplePoints: Int = 12, seed: UInt64 = 42) -> (lossBefore: Double, lossAfter: Double, sampleU: [(x: Double, t: Double, u: Double)]) {
    var rng = SeededGenerator(seed: seed)
    let model = PINNModel(hiddenSizes: hidden, rng: &rng)

    let xs = (0..<samplePoints).map { i in Double(i) / Double(max(samplePoints - 1, 1)) }
    let ts = (0..<samplePoints).map { i in Double(i) / Double(max(samplePoints - 1, 1)) * 0.5 }

    let result = FiniteDiffSGD.trainOneStep(model: model, xs: xs, ts: ts, learningRate: 5e-2, epsilon: 1e-4, icWeight: 1.0, pdeWeight: 1.0)

    let probes: [(Double, Double)] = [ (0.25, 0.0), (0.5, 0.25), (0.75, 0.5) ]
    let sampleU = probes.map { (x, t) in (x: x, t: t, u: model.forward(x: x, t: t)) }
    return (result.lossBefore, result.lossAfter, sampleU)
  }
}

public struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64
  public init(seed: UInt64) { self.state = seed != 0 ? seed : 0xdead_beef_cafe_babe }
  public mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
  }
}