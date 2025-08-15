import XCTest
@testable import UOIFCore

final class PINNTests: XCTestCase {
    
    var pinn: PINN!
    
    override func setUp() {
        super.setUp()
        pinn = PINN()
    }
    
    override func tearDown() {
        pinn = nil
        super.tearDown()
    }
    
    // MARK: - DenseLayer Tests
    
    func testDenseLayerInitialization() {
        let layer = DenseLayer(inputSize: 3, outputSize: 4)
        
        XCTAssertEqual(layer.weights.count, 4)
        XCTAssertEqual(layer.weights[0].count, 3)
        XCTAssertEqual(layer.biases.count, 4)
        
        // Check that weights are within reasonable bounds for Xavier initialization
        let expectedBound = sqrt(6.0 / Double(3 + 4))
        for weights in layer.weights {
            for weight in weights {
                XCTAssertGreaterThanOrEqual(weight, -expectedBound)
                XCTAssertLessThanOrEqual(weight, expectedBound)
            }
        }
    }
    
    func testDenseLayerForward() {
        let layer = DenseLayer(inputSize: 2, outputSize: 3)
        let input = [1.0, 2.0]
        let output = layer.forward(input)
        
        XCTAssertEqual(output.count, 3)
        
        // Check that output is reasonable (tanh activation bounds)
        for value in output {
            XCTAssertGreaterThanOrEqual(value, -1.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
    }
    
    // MARK: - PINN Tests
    
    func testPINNInitialization() {
        XCTAssertEqual(pinn.layers.count, 3)
        XCTAssertEqual(pinn.layers[0].weights.count, 20)
        XCTAssertEqual(pinn.layers[0].weights[0].count, 2)
        XCTAssertEqual(pinn.layers[1].weights.count, 20)
        XCTAssertEqual(pinn.layers[1].weights[0].count, 20)
        XCTAssertEqual(pinn.layers[2].weights.count, 1)
        XCTAssertEqual(pinn.layers[2].weights[0].count, 20)
    }
    
    func testPINNForward() {
        let x = 0.5
        let t = 1.0
        let output = pinn.forward(x: x, t: t)
        
        // Output should be a single Double value
        XCTAssertTrue(output.isFinite)
        XCTAssertGreaterThanOrEqual(output, -1.0)
        XCTAssertLessThanOrEqual(output, 1.0)
    }
    
    // MARK: - Hybrid Output Tests
    
    func testHybridOutput() {
        let hybrid = HybridOutput(stateInference: 0.72, mlGradient: 0.85, alpha: 0.5)
        
        XCTAssertEqual(hybrid.stateInference, 0.72)
        XCTAssertEqual(hybrid.mlGradient, 0.85)
        XCTAssertEqual(hybrid.alpha, 0.5)
        XCTAssertEqual(hybrid.hybridValue, 0.785, accuracy: 1e-6)
    }
    
    func testHybridOutputDefaultAlpha() {
        let hybrid = HybridOutput(stateInference: 0.6, mlGradient: 0.8)
        
        XCTAssertEqual(hybrid.alpha, 0.5)
        XCTAssertEqual(hybrid.hybridValue, 0.7, accuracy: 1e-6)
    }
    
    // MARK: - Regularization Tests
    
    func testCognitiveRegularization() {
        let reg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
        
        XCTAssertEqual(reg.pdeResidual, 0.15)
        XCTAssertEqual(reg.weight, 0.6)
        XCTAssertEqual(reg.value, 0.09, accuracy: 1e-6)
    }
    
    func testCognitiveRegularizationDefaultWeight() {
        let reg = CognitiveRegularization(pdeResidual: 0.2)
        
        XCTAssertEqual(reg.weight, 0.6)
        XCTAssertEqual(reg.value, 0.12, accuracy: 1e-6)
    }
    
    func testEfficiencyRegularization() {
        let reg = EfficiencyRegularization(trainingEfficiency: 0.1, weight: 0.4)
        
        XCTAssertEqual(reg.trainingEfficiency, 0.1)
        XCTAssertEqual(reg.weight, 0.4)
        XCTAssertEqual(reg.value, 0.04, accuracy: 1e-6)
    }
    
    func testEfficiencyRegularizationDefaultWeight() {
        let reg = EfficiencyRegularization(trainingEfficiency: 0.15)
        
        XCTAssertEqual(reg.weight, 0.4)
        XCTAssertEqual(reg.value, 0.06, accuracy: 1e-6)
    }
    
    // MARK: - Probability Model Tests
    
    func testProbabilityModel() {
        let prob = ProbabilityModel(hypothesis: 0.8, beta: 1.2)
        
        XCTAssertEqual(prob.hypothesis, 0.8)
        XCTAssertEqual(prob.beta, 1.2)
        XCTAssertEqual(prob.adjustedProbability, 0.96, accuracy: 1e-6)
    }
    
    func testProbabilityModelDefaultBeta() {
        let prob = ProbabilityModel(hypothesis: 0.7)
        
        XCTAssertEqual(prob.beta, 1.2)
        XCTAssertEqual(prob.adjustedProbability, 0.84, accuracy: 1e-6)
    }
    
    func testProbabilityModelCapped() {
        let prob = ProbabilityModel(hypothesis: 0.9, beta: 1.5)
        
        // Should be capped at 1.0
        XCTAssertEqual(prob.adjustedProbability, 1.0, accuracy: 1e-6)
    }
    
    // MARK: - Performance Metric Tests
    
    func testPerformanceMetric() {
        let hybrid = HybridOutput(stateInference: 0.72, mlGradient: 0.85, alpha: 0.5)
        let cognitiveReg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
        let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.1, weight: 0.4)
        let probability = ProbabilityModel(hypothesis: 0.8, beta: 1.2)
        
        let metric = PerformanceMetric(
            hybridOutput: hybrid,
            cognitiveReg: cognitiveReg,
            efficiencyReg: efficiencyReg,
            probability: probability
        )
        
        // Calculate expected value manually
        let expectedValue = 0.785 * exp(-(0.09 + 0.04)) * 0.96
        XCTAssertEqual(metric.value, expectedValue, accuracy: 1e-6)
    }
    
    func testPerformanceMetricInterpretation() {
        let hybrid = HybridOutput(stateInference: 0.8, mlGradient: 0.9, alpha: 0.5)
        let cognitiveReg = CognitiveRegularization(pdeResidual: 0.05, weight: 0.6)
        let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.03, weight: 0.4)
        let probability = ProbabilityModel(hypothesis: 0.9, beta: 1.1)
        
        let metric = PerformanceMetric(
            hybridOutput: hybrid,
            cognitiveReg: cognitiveReg,
            efficiencyReg: efficiencyReg,
            probability: probability
        )
        
        // Should be excellent performance
        XCTAssertTrue(metric.interpretation.contains("Excellent"))
    }
    
    // MARK: - Numerical Utilities Tests
    
    func testFiniteDiff() {
        let f: (Double) -> Double = { x in x * x } // f(x) = x²
        let derivative = finiteDiff(f: f, at: 2.0)
        
        // f'(x) = 2x, so f'(2) = 4
        XCTAssertEqual(derivative, 4.0, accuracy: 1e-3)
    }
    
    func testFiniteDiffCustomStep() {
        let f: (Double) -> Double = { x in x * x * x } // f(x) = x³
        let derivative = finiteDiff(f: f, at: 1.0, dx: 1e-4)
        
        // f'(x) = 3x², so f'(1) = 3
        XCTAssertEqual(derivative, 3.0, accuracy: 1e-3)
    }
    
    // MARK: - Loss Function Tests
    
    func testICLoss() {
        let x = [-1.0, 0.0, 1.0]
        let loss = icLoss(model: pinn, x: x)
        
        XCTAssertTrue(loss.isFinite)
        XCTAssertGreaterThanOrEqual(loss, 0.0)
    }
    
    func testPDELoss() {
        let x = Array(stride(from: -1.0, to: 1.0, by: 0.2))
        let t = Array(repeating: 1.0, count: x.count)
        let loss = pdeLoss(model: pinn, x: x, t: t)
        
        XCTAssertTrue(loss.isFinite)
        XCTAssertGreaterThanOrEqual(loss, 0.0)
    }
    
    // MARK: - Training Tests
    
    func testTrainStep() {
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        // Store original weights
        let originalWeights = pinn.layers[0].weights[0][0]
        
        trainStep(model: pinn, x: x, t: t, learningRate: 0.01)
        
        // Weights should have changed
        let newWeights = pinn.layers[0].weights[0][0]
        XCTAssertNotEqual(originalWeights, newWeights)
    }
    
    func testTrainingLoop() {
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        // This should complete without errors
        train(model: pinn, epochs: 10, x: x, t: t, printEvery: 5)
        
        // Model should still be valid
        let output = pinn.forward(x: 0.0, t: 0.0)
        XCTAssertTrue(output.isFinite)
    }
    
    // MARK: - Numerical Example Tests
    
    func testNumericalExample() {
        let metric = runNumericalExample()
        
        // Verify the expected values from the requirements
        XCTAssertEqual(metric.hybridOutput.stateInference, 0.72)
        XCTAssertEqual(metric.hybridOutput.mlGradient, 0.85)
        XCTAssertEqual(metric.hybridOutput.alpha, 0.5)
        XCTAssertEqual(metric.hybridOutput.hybridValue, 0.785, accuracy: 1e-6)
        
        XCTAssertEqual(metric.cognitiveReg.pdeResidual, 0.15)
        XCTAssertEqual(metric.cognitiveReg.weight, 0.6)
        XCTAssertEqual(metric.efficiencyReg.trainingEfficiency, 0.10)
        XCTAssertEqual(metric.efficiencyReg.weight, 0.4)
        
        XCTAssertEqual(metric.probability.hypothesis, 0.80)
        XCTAssertEqual(metric.probability.beta, 1.2)
        XCTAssertEqual(metric.probability.adjustedProbability, 0.96, accuracy: 1e-6)
        
        // Final Ψ(x) should be approximately 0.662
        XCTAssertEqual(metric.value, 0.662, accuracy: 1e-3)
        
        // Should be "Good model performance with solid accuracy"
        XCTAssertTrue(metric.interpretation.contains("Good model performance"))
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceMetricCalculation() {
        let hybrid = HybridOutput(stateInference: 0.72, mlGradient: 0.85, alpha: 0.5)
        let cognitiveReg = CognitiveRegularization(pdeResidual: 0.15, weight: 0.6)
        let efficiencyReg = EfficiencyRegularization(trainingEfficiency: 0.10, weight: 0.4)
        let probability = ProbabilityModel(hypothesis: 0.80, beta: 1.2)
        
        let metric = PerformanceMetric(
            hybridOutput: hybrid,
            cognitiveReg: cognitiveReg,
            efficiencyReg: efficiencyReg,
            probability: probability
        )
        
        // Measure performance
        measure {
            _ = metric.value
        }
    }
    
    func testTrainingPerformance() {
        let x = Array(stride(from: -1.0, to: 1.0, by: 0.1))
        let t = Array(repeating: 1.0, count: x.count)
        
        measure {
            trainStep(model: pinn, x: x, t: t)
        }
    }
}