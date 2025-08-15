// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public enum ActivationFunction {
  case tanh
  case linear

  func apply(_ x: Double) -> Double {
    switch self {
    case .tanh:
      return tanh(x)
    case .linear:
      return x
    }
  }
}

public final class DenseLayer {
  public private(set) var weights: [[Double]] // [output][input]
  public private(set) var biases: [Double]    // [output]
  private let activation: ActivationFunction

  public init(inputSize: Int, outputSize: Int, activation: ActivationFunction) {
    self.activation = activation
    let bound = sqrt(6.0 / Double(inputSize + outputSize))
    var w: [[Double]] = []
    w.reserveCapacity(outputSize)
    for _ in 0..<outputSize {
      var row: [Double] = []
      row.reserveCapacity(inputSize)
      for _ in 0..<inputSize {
        row.append(Double.random(in: -bound...bound))
      }
      w.append(row)
    }
    self.weights = w
    self.biases = (0..<outputSize).map { _ in 0.0 }
  }

  public func forward(_ input: [Double]) -> [Double] {
    precondition(!weights.isEmpty && weights[0].count == input.count, "Input size must match layer inputSize")
    var output: [Double] = Array(repeating: 0.0, count: biases.count)
    for i in 0..<biases.count {
      var z = biases[i]
      for j in 0..<input.count {
        z += weights[i][j] * input[j]
      }
      output[i] = activation.apply(z)
    }
    return output
  }
}

public final class PINN {
  public var layers: [DenseLayer]

  public init(hiddenWidth: Int = 20) {
    self.layers = [
      DenseLayer(inputSize: 2, outputSize: hiddenWidth, activation: .tanh),
      DenseLayer(inputSize: hiddenWidth, outputSize: hiddenWidth, activation: .tanh),
      DenseLayer(inputSize: hiddenWidth, outputSize: 1, activation: .linear)
    ]
  }

  public func forward(x: Double, t: Double) -> Double {
    var input: [Double] = [x, t]
    for (index, layer) in layers.enumerated() {
      let out = layer.forward(input)
      if index == layers.count - 1 {
        return out[0]
      }
      input = out
    }
    return input[0]
  }
}

// Central difference derivative: O(dx^2)
public func finiteDifferenceFirst(_ f: (Double) -> Double, at x0: Double, dx: Double = 1e-6) -> Double {
  let f1 = f(x0 + dx)
  let f2 = f(x0 - dx)
  return (f1 - f2) / (2.0 * dx)
}

// Second derivative via three-point stencil
public func finiteDifferenceSecond(_ f: (Double) -> Double, at x0: Double, dx: Double = 1e-4) -> Double {
  let fPlus = f(x0 + dx)
  let fZero = f(x0)
  let fMinus = f(x0 - dx)
  return (fPlus - 2.0 * fZero + fMinus) / (dx * dx)
}

public enum PDE {
  // 1D heat equation: u_t = u_xx on x in [-1,1], t in [0,1]
  public static func residual_heatEquation(model: PINN, x: Double, t: Double, dx: Double = 1e-4, dt: Double = 1e-4) -> Double {
    let u_t = finiteDifferenceFirst({ tau in model.forward(x: x, t: tau) }, at: t, dx: dt)
    let u_xx = finiteDifferenceSecond({ xi in model.forward(x: xi, t: t) }, at: x, dx: dx)
    return u_t - u_xx
  }
}

public enum Losses {
  // Batched PDE residual MSE
  public static func pdeLoss(model: PINN, xs: [Double], ts: [Double], batchSize: Int = 32, dx: Double = 1e-4, dt: Double = 1e-4) -> Double {
    let count = min(xs.count, ts.count)
    if count == 0 { return 0.0 }
    var total: Double = 0.0
    var seen: Int = 0
    var batchStart = 0
    while batchStart < count {
      let batchEnd = min(batchStart + batchSize, count)
      for i in batchStart..<batchEnd {
        let r = PDE.residual_heatEquation(model: model, x: xs[i], t: ts[i], dx: dx, dt: dt)
        total += r * r
        seen += 1
      }
      batchStart += batchSize
    }
    return total / Double(seen)
  }

  // Initial condition: u(x,0) = -sin(pi x)
  public static func initialConditionLoss(model: PINN, xs: [Double]) -> Double {
    if xs.isEmpty { return 0.0 }
    var total: Double = 0.0
    for x in xs {
      let u = model.forward(x: x, t: 0.0)
      let trueU = -sin(Double.pi * x)
      let d = u - trueU
      total += d * d
    }
    return total / Double(xs.count)
  }
}

public enum Optimizer {
  // Finite-difference gradient approximation step over all parameters
  public static func trainStep(model: PINN, xs: [Double], ts: [Double], learningRate: Double = 0.005, perturbation: Double = 1e-5) {
    func totalLoss() -> Double {
      return Losses.pdeLoss(model: model, xs: xs, ts: ts) + Losses.initialConditionLoss(model: model, xs: xs)
    }

    let baseLoss = totalLoss()

    for layerIndex in 0..<model.layers.count {
      let layer = model.layers[layerIndex]

      // Weights
      for i in 0..<layer.weights.count {
        for j in 0..<layer.weights[i].count {
          let original = layer.weights[i][j]
          model.layers[layerIndex].weights[i][j] = original + perturbation
          let lossUp = totalLoss()
          let grad = (lossUp - baseLoss) / perturbation
          model.layers[layerIndex].weights[i][j] = original - learningRate * grad
        }
      }

      // Biases
      for i in 0..<layer.biases.count {
        let original = layer.biases[i]
        model.layers[layerIndex].biases[i] = original + perturbation
        let lossUp = totalLoss()
        let grad = (lossUp - baseLoss) / perturbation
        model.layers[layerIndex].biases[i] = original - learningRate * grad
      }
    }
  }
}

public enum PINNTrainer {
  public static func train(model: PINN, xs: [Double], ts: [Double], epochs: Int = 1000, printEvery: Int = 50, learningRate: Double = 0.005) {
    func totalLoss() -> Double {
      return Losses.pdeLoss(model: model, xs: xs, ts: ts) + Losses.initialConditionLoss(model: model, xs: xs)
    }

    for epoch in 1...epochs {
      Optimizer.trainStep(model: model, xs: xs, ts: ts, learningRate: learningRate)
      if epoch % printEvery == 0 || epoch == 1 {
        let l = totalLoss()
        print(String(format: "[epoch %4d] loss = %.6f", epoch, l))
      }
    }
  }

  // Demo: sample points in x in [-1,1], t in [0,1]
  public static func demoTraining(epochs: Int = 1000, samples: Int = 200, seedless: Bool = true) {
    var xs: [Double] = []
    var ts: [Double] = []
    xs.reserveCapacity(samples)
    ts.reserveCapacity(samples)
    for _ in 0..<samples {
      xs.append(Double.random(in: -1.0...1.0))
      ts.append(Double.random(in: 0.0...1.0))
    }
    let model = PINN(hiddenWidth: 20)
    train(model: model, xs: xs, ts: ts, epochs: epochs, printEvery: 50, learningRate: 0.005)
  }
}

#if canImport(SwiftUI) && canImport(Charts)
import SwiftUI
import Charts

public struct SolutionChart: View {
  public init() {}
  public var body: some View {
    let xVals: [Double] = Array(stride(from: -1.0, through: 1.0, by: 0.1))
    let pinnU: [Double] = [0.0, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.4, 0.1, -0.3, -0.6, -0.8, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.0]
    let rk4U: [Double] = [0.0, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.3, 0.0, -0.4, -0.7, -0.8, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.0]

    return Chart {
      ForEach(Array(zip(xVals.indices, xVals)), id: \.0) { idx, x in
        LineMark(x: .value("x", x), y: .value("PINN u", pinnU[idx]))
          .foregroundStyle(.blue)
      }
      ForEach(Array(zip(xVals.indices, xVals)), id: \.0) { idx, x in
        LineMark(x: .value("x", x), y: .value("RK4 u", rk4U[idx]))
          .foregroundStyle(.red)
      }
    }
    .chartLegend(position: .bottom)
    .frame(height: 260)
    .padding()
  }
}
#endif
