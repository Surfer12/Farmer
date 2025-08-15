// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import XCTest
@testable import UOIFCore

final class PINNTests: XCTestCase {
    
    func testDenseLayerInitialization() {
        let layer = DenseLayer(inputSize: 3, outputSize: 2)
        
        XCTAssertEqual(layer.weights.count, 2)
        XCTAssertEqual(layer.weights[0].count, 3)
        XCTAssertEqual(layer.biases.count, 2)
        
        // Check that weights are initialized with reasonable values
        for row in layer.weights {
            for weight in row {
                XCTAssertTrue(weight >= -1.0 && weight <= 1.0)
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
            XCTAssertTrue(value >= -1.0 && value <= 1.0)
        }
    }
    
    func testPINNInitialization() {
        let layerSizes = [2, 10, 5, 1]
        let pinn = PINN(layerSizes: layerSizes)
        
        XCTAssertEqual(pinn.layers.count, 3) // 2->10, 10->5, 5->1
        XCTAssertEqual(pinn.layers[0].weights.count, 10)
        XCTAssertEqual(pinn.layers[0].weights[0].count, 2)
        XCTAssertEqual(pinn.layers[1].weights.count, 5)
        XCTAssertEqual(pinn.layers[1].weights[0].count, 10)
        XCTAssertEqual(pinn.layers[2].weights.count, 1)
        XCTAssertEqual(pinn.layers[2].weights[0].count, 5)
    }
    
    func testPINNForward() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        let x = 0.5
        let t = 0.3
        let output = pinn.forward(x: x, t: t)
        
        // Output should be a single value
        XCTAssertTrue(output.isFinite)
        XCTAssertTrue(output >= -1.0 && output <= 1.0) // tanh activation bounds
    }
    
    func testPINNDerivatives() {
        let pinn = PINN(layerSizes: [2, 10, 1])
        let x = 0.0
        let t = 0.0
        
        let dx = pinn.dx(x, t)
        let dt = pinn.dt(x, t)
        let dxx = pinn.dxx(x, t)
        
        // Derivatives should be finite
        XCTAssertTrue(dx.isFinite)
        XCTAssertTrue(dt.isFinite)
        XCTAssertTrue(dxx.isFinite)
    }
    
    func testPDEBurgersLoss() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        let loss = pdeLoss(model: pinn, x: x, t: t)
        
        XCTAssertTrue(loss >= 0.0) // Loss should be non-negative
        XCTAssertTrue(loss.isFinite)
    }
    
    func testInitialConditionLoss() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        let x = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        let loss = icLoss(model: pinn, x: x)
        
        XCTAssertTrue(loss >= 0.0) // Loss should be non-negative
        XCTAssertTrue(loss.isFinite)
    }
    
    func testBoundaryConditionLoss() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        let t = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        let loss = bcLoss(model: pinn, t: t)
        
        XCTAssertTrue(loss >= 0.0) // Loss should be non-negative
        XCTAssertTrue(loss.isFinite)
    }
    
    func testRK4Integration() {
        // Test RK4 with simple linear ODE: dy/dt = y
        func linearODE(_ t: Double, _ y: [Double]) -> [Double] {
            return y
        }
        
        let y0 = [1.0]
        let t0 = 0.0
        let dt = 0.1
        let y1 = rk4(f: linearODE, y: y0, t: t0, dt: dt)
        
        XCTAssertEqual(y1.count, 1)
        XCTAssertTrue(y1[0] > 1.0) // Solution should grow exponentially
        XCTAssertTrue(y1[0].isFinite)
    }
    
    func testHybridTrainer() {
        let model = PINN(layerSizes: [2, 5, 1])
        let trainer = HybridTrainer(model: model, learningRate: 0.01, epochs: 10)
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        let step = trainer.trainingStep(x: x, t: t)
        
        // Verify all components are present
        XCTAssertEqual(step.S_x, 0.72)
        XCTAssertEqual(step.N_x, 0.85)
        XCTAssertEqual(step.alpha_t, 0.5)
        XCTAssertEqual(step.O_hybrid, 0.785) // 0.5 * 0.72 + 0.5 * 0.85
        XCTAssertEqual(step.R_cognitive, 0.15)
        XCTAssertEqual(step.R_efficiency, 0.10)
        XCTAssertEqual(step.P_total, 0.13) // 0.6 * 0.15 + 0.4 * 0.10
        XCTAssertEqual(step.P_adj, 0.96) // min(1.2 * 0.80, 1.0)
        
        // Verify final Ψ(x) calculation
        let expectedPsi = 0.785 * exp(-0.13) * 0.96
        XCTAssertEqual(step.Psi_x, expectedPsi, accuracy: 1e-10)
    }
    
    func testTrainingStepMathematicalFramework() {
        let model = PINN(layerSizes: [2, 5, 1])
        let trainer = HybridTrainer(model: model, learningRate: 0.01, epochs: 10)
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        let step = trainer.trainingStep(x: x, t: t)
        
        // Step 1: Outputs verification
        XCTAssertEqual(step.S_x, 0.72)
        XCTAssertEqual(step.N_x, 0.85)
        
        // Step 2: Hybrid verification
        XCTAssertEqual(step.alpha_t, 0.5)
        XCTAssertEqual(step.O_hybrid, 0.785)
        
        // Step 3: Penalties verification
        XCTAssertEqual(step.R_cognitive, 0.15)
        XCTAssertEqual(step.R_efficiency, 0.10)
        XCTAssertEqual(step.P_total, 0.13)
        
        // Step 4: Probability verification
        XCTAssertEqual(step.P_adj, 0.96)
        
        // Step 5: Ψ(x) verification
        let penalty_exp = exp(-0.13)
        let expectedPsi = 0.785 * penalty_exp * 0.96
        XCTAssertEqual(step.Psi_x, expectedPsi, accuracy: 1e-10)
    }
    
    func testFiniteDifference() {
        // Test finite difference on simple function: f(x) = x^2
        func square(_ x: Double) -> Double {
            return x * x
        }
        
        let x = 2.0
        let dx = 0.001
        let derivative = finiteDiff(f: square, at: x, dx: dx)
        let expected = 2.0 * x // d/dx(x^2) = 2x
        
        XCTAssertEqual(derivative, expected, accuracy: 0.01)
    }
    
    func testPINNTrainingStep() {
        let model = PINN(layerSizes: [2, 5, 1])
        let x = [-0.5, 0.0, 0.5]
        let t = [0.0, 0.5, 1.0]
        
        // Test that training step updates model parameters
        let initialWeights = model.layers[0].weights[0][0]
        trainStep(model: model, x: x, t: t, learningRate: 0.01)
        let finalWeights = model.layers[0].weights[0][0]
        
        // Weights should change after training step
        XCTAssertNotEqual(initialWeights, finalWeights)
    }
    
    func testPerformance() {
        let model = PINN(layerSizes: [2, 20, 20, 1])
        let x = Array(-1.0...1.0).stride(by: 0.1).map { $0 }
        let t = Array(0.0...1.0).stride(by: 0.1).map { $0 }
        
        measure {
            let _ = pdeLoss(model: model, x: x, t: t)
        }
    }
    
    static var allTests = [
        ("testDenseLayerInitialization", testDenseLayerInitialization),
        ("testDenseLayerForward", testDenseLayerForward),
        ("testPINNInitialization", testPINNInitialization),
        ("testPINNForward", testPINNForward),
        ("testPINNDerivatives", testPINNDerivatives),
        ("testPDEBurgersLoss", testPDEBurgersLoss),
        ("testInitialConditionLoss", testInitialConditionLoss),
        ("testBoundaryConditionLoss", testBoundaryConditionLoss),
        ("testRK4Integration", testRK4Integration),
        ("testHybridTrainer", testHybridTrainer),
        ("testTrainingStepMathematicalFramework", testTrainingStepMathematicalFramework),
        ("testFiniteDifference", testFiniteDifference),
        ("testPINNTrainingStep", testPINNTrainingStep),
        ("testPerformance", testPerformance)
    ]
}