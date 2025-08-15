import Foundation

// MARK: - Math helpers
@inline(__always) func clamp(_ value: Double, min minValue: Double, max maxValue: Double) -> Double {
	return max(minValue, min(maxValue, value))
}

@inline(__always) func uniformRandom(in range: ClosedRange<Double>) -> Double {
	Double.random(in: range)
}

// MARK: - Dense Layer with Xavier initialization
final class DenseLayer {
	var weights: [[Double]] // [output][input]
	var biases: [Double]    // [output]
	let inputSize: Int
	let outputSize: Int

	init(inputSize: Int, outputSize: Int) {
		self.inputSize = inputSize
		self.outputSize = outputSize
		let bound = sqrt(6.0 / Double(inputSize + outputSize))
		self.weights = (0..<outputSize).map { _ in
			(0..<inputSize).map { _ in uniformRandom(in: -bound...bound) }
		}
		self.biases = Array(repeating: 0.0, count: outputSize)
	}

	func forward(_ input: [Double], activation: Bool) -> [Double] {
		precondition(input.count == inputSize, "Input size mismatch")
		var output = Array(repeating: 0.0, count: outputSize)
		for o in 0..<outputSize {
			var sum = biases[o]
			for i in 0..<inputSize {
				sum += weights[o][i] * input[i]
			}
			output[o] = activation ? tanh(sum) : sum
		}
		return output
	}
}

// MARK: - PINN Model
final class PINN {
	var layers: [DenseLayer]

	init() {
		layers = [
			DenseLayer(inputSize: 2, outputSize: 20),
			DenseLayer(inputSize: 20, outputSize: 20),
			DenseLayer(inputSize: 20, outputSize: 1)
		]
	}

	func forward(x: Double, t: Double) -> Double {
		var input: [Double] = [x, t]
		for idx in 0..<layers.count {
			let isLast = (idx == layers.count - 1)
			input = layers[idx].forward(input, activation: !isLast)
		}
		return input[0]
	}
}

// MARK: - Finite differences
@inline(__always) func finiteDiff(_ f: (Double) -> Double, at point: Double, dx: Double = 1e-6) -> Double {
	return (f(point + dx) - f(point - dx)) / (2.0 * dx)
}

@inline(__always) func secondFiniteDiff(_ f: (Double) -> Double, at point: Double, dx: Double = 1e-4) -> Double {
	let fPlus = f(point + dx)
	let f0 = f(point)
	let fMinus = f(point - dx)
	return (fPlus - 2.0 * f0 + fMinus) / (dx * dx)
}

// MARK: - PDE and losses (1D heat equation: u_t = u_xx)
struct LossBreakdown {
	var pde: Double
	var ic: Double
	var bc: Double
	var l2: Double
	var total: Double { pde + ic + bc + l2 }
}

func heatEquationResidualSquared(model: PINN, x: Double, t: Double, dt: Double = 1e-4, dx: Double = 1e-3) -> Double {
	let u: (Double, Double) -> Double = { xVal, tVal in model.forward(x: xVal, t: tVal) }
	let ut = (u(x, t + dt) - u(x, t - dt)) / (2.0 * dt)
	let ux2 = secondFiniteDiff({ xx in u(xx, t) }, at: x, dx: dx)
	let residual = ut - ux2
	return residual * residual
}

func pdeLoss(model: PINN, xs: [Double], ts: [Double], batchSize: Int = 32) -> Double {
	let count = min(xs.count, ts.count)
	if count == 0 { return 0.0 }
	var sumSq = 0.0
	var processed = 0
	var idx = 0
	while idx < count {
		let end = min(idx + batchSize, count)
		for i in idx..<end {
			sumSq += heatEquationResidualSquared(model: model, x: xs[i], t: ts[i])
			processed += 1
		}
		idx = end
	}
	return sumSq / Double(processed)
}

func icLoss(model: PINN, xs: [Double]) -> Double {
	if xs.isEmpty { return 0.0 }
	var sumSq = 0.0
	for x in xs {
		let u0 = model.forward(x: x, t: 0.0)
		let target = -sin(.pi * x)
		sumSq += (u0 - target) * (u0 - target)
	}
	return sumSq / Double(xs.count)
}

func bcLoss(model: PINN, ts: [Double]) -> Double {
	if ts.isEmpty { return 0.0 }
	var sumSq = 0.0
	for t in ts {
		let left = model.forward(x: -1.0, t: t)
		let right = model.forward(x: 1.0, t: t)
		sumSq += left * left + right * right
	}
	return sumSq / Double(ts.count)
}

func l2Penalty(model: PINN, weight: Double = 1e-4) -> Double {
	var sum = 0.0
	for layer in model.layers {
		for row in layer.weights { for w in row { sum += w * w } }
		for b in layer.biases { sum += b * b }
	}
	return weight * sum
}

// MARK: - Gradient-free parameter update (central difference)
func trainStep(model: PINN,
               xsPDE: [Double], tsPDE: [Double],
               xsIC: [Double], tsBC: [Double],
               learningRate: Double = 0.005,
               perturbation: Double = 1e-5,
               lambdaPDE: Double = 1.0,
               lambdaIC: Double = 1.0,
               lambdaBC: Double = 1.0,
               lambdaL2: Double = 1.0) -> LossBreakdown {

	func totalLoss() -> LossBreakdown {
		let pde = lambdaPDE * pdeLoss(model: model, xs: xsPDE, ts: tsPDE)
		let ic = lambdaIC * icLoss(model: model, xs: xsIC)
		let bc = lambdaBC * bcLoss(model: model, ts: tsBC)
		let l2 = lambdaL2 * l2Penalty(model: model)
		return LossBreakdown(pde: pde, ic: ic, bc: bc, l2: l2)
	}

	func totalLossScalar() -> Double { totalLoss().total }

	// Compute and apply gradients for weights and biases
	for layer in model.layers {
		// Weights
		for o in 0..<layer.outputSize {
			for i in 0..<layer.inputSize {
				let oldVal = layer.weights[o][i]
				layer.weights[o][i] = oldVal + perturbation
				let lossPlus = totalLossScalar()
				layer.weights[o][i] = oldVal - perturbation
				let lossMinus = totalLossScalar()
				let grad = (lossPlus - lossMinus) / (2.0 * perturbation)
				layer.weights[o][i] = oldVal - learningRate * grad
			}
		}
		// Biases
		for o in 0..<layer.outputSize {
			let oldVal = layer.biases[o]
			layer.biases[o] = oldVal + perturbation
			let lossPlus = totalLossScalar()
			layer.biases[o] = oldVal - perturbation
			let lossMinus = totalLossScalar()
			let grad = (lossPlus - lossMinus) / (2.0 * perturbation)
			layer.biases[o] = oldVal - learningRate * grad
		}
	}

	return totalLoss()
}

// MARK: - Analytic solution (for evaluation / hybrid metric)
// Heat equation on [-1, 1] with u(x,0) = -sin(pi x) and Dirichlet 0 boundaries has solution:
// u(x,t) = -exp(-pi^2 t) * sin(pi x)
func analyticHeat(_ x: Double, _ t: Double) -> Double {
	return -exp(-(Double.pi * Double.pi) * t) * sin(.pi * x)
}

// MARK: - Hybrid metrics (S, N, alpha, penalties, probability)
struct HybridMetrics {
	var S: Double
	var N: Double
	var alpha: Double
	var O_hybrid: Double
	var R_cognitive: Double
	var R_efficiency: Double
	var P_total: Double
	var expPenalty: Double
	var P: Double
	var beta: Double
	var P_adjusted: Double
	var psi: Double
}

func computeHybridMetrics(model: PINN,
                          x: Double,
                          t: Double,
                          lambda1: Double = 0.6,
                          lambda2: Double = 0.4,
                          beta: Double = 1.2) -> HybridMetrics {
	let S = model.forward(x: x, t: t)
	let N = analyticHeat(x, t)
	let alpha = 0.5
	let O = alpha * S + (1.0 - alpha) * N
	// Penalties: use local PDE residual and a tiny L2 as proxies
	let R_cognitive = heatEquationResidualSquared(model: model, x: x, t: t)
	var eff = 0.0
	for layer in model.layers { for row in layer.weights { for w in row { eff += w * w } } }
	let R_efficiency = 1e-6 * eff
	let P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
	let expPenalty = exp(-P_total)
	// Map total loss proxy to a probability in [0,1]
	let baseP = exp(-P_total)
	let P_adjusted = pow(clamp(baseP, min: 0.0, max: 1.0), beta)
	let psi = O * expPenalty * P_adjusted
	return HybridMetrics(S: S, N: N, alpha: alpha, O_hybrid: O, R_cognitive: R_cognitive, R_efficiency: R_efficiency, P_total: P_total, expPenalty: expPenalty, P: baseP, beta: beta, P_adjusted: P_adjusted, psi: psi)
}

// MARK: - Data generation
func linspace(_ start: Double, _ end: Double, _ count: Int) -> [Double] {
	if count <= 1 { return [start] }
	let step = (end - start) / Double(count - 1)
	return (0..<count).map { start + Double($0) * step }
}

func randomUniform(_ start: Double, _ end: Double, _ count: Int) -> [Double] {
	(0..<count).map { _ in Double.random(in: start...end) }
}

// MARK: - Training
func train(model: PINN,
           epochs: Int = 1000,
           printEvery: Int = 50,
           pdePoints: Int = 128,
           icPoints: Int = 64,
           bcPoints: Int = 64,
           learningRate: Double = 0.003) {

	let xsIC = linspace(-1.0, 1.0, icPoints)
	let tsBC = linspace(0.0, 1.0, bcPoints)

	for epoch in 1...epochs {
		let xsPDE = randomUniform(-1.0, 1.0, pdePoints)
		let tsPDE = randomUniform(0.0, 1.0, pdePoints)
		let loss = trainStep(model: model,
							xsPDE: xsPDE, tsPDE: tsPDE,
							xsIC: xsIC, tsBC: tsBC,
							learningRate: learningRate)
		if epoch % printEvery == 0 || epoch == 1 || epoch == epochs {
			print(String(format: "epoch %4d | pde: %.6f ic: %.6f bc: %.6f l2: %.6f total: %.6f", epoch, loss.pde, loss.ic, loss.bc, loss.l2, loss.total))
		}
	}
}

// MARK: - Entry
func main() {
	let model = PINN()
	// Quick training run; increase epochs for better results
	let epochsEnv = ProcessInfo.processInfo.environment["EPOCHS"].flatMap { Int($0) } ?? 1000
	train(model: model, epochs: epochsEnv, printEvery: 50)

	// Demonstrate hybrid metrics at a sample point
	let x: Double = 0.2
	let t: Double = 1.0
	let metrics = computeHybridMetrics(model: model, x: x, t: t)
	print(String(format: "S(x)=%.3f, N(x)=%.3f, alpha=%.2f, O_hybrid=%.3f", metrics.S, metrics.N, metrics.alpha, metrics.O_hybrid))
	print(String(format: "R_cognitive=%.3f, R_efficiency=%.3f, P_total=%.3f, exp=%.3f", metrics.R_cognitive, metrics.R_efficiency, metrics.P_total, metrics.expPenalty))
	print(String(format: "P=%.3f, beta=%.2f, P_adj=%.3f", metrics.P, metrics.beta, metrics.P_adjusted))
	print(String(format: "Psi(x)=%.3f", metrics.psi))
}

main()