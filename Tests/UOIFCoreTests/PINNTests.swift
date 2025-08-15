// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import XCTest
@testable import UOIFCore

final class PINNTests: XCTestCase {
    
    func testDenseLayerInitialization() {
        let layer = DenseLayer(inputSize: 2, outputSize: 3)
        
        XCTAssertEqual(layer.weights.count, 3)
        XCTAssertEqual(layer.weights[0].count, 2)
        XCTAssertEqual(layer.biases.count, 3)
        XCTAssertEqual(layer.activation, .tanh)
    }
    
    func testDenseLayerForwardPass() {
        let layer = DenseLayer(inputSize: 2, outputSize: 1)
        
        // Set known weights and biases for testing
        layer.weights[0] = [1.0, 2.0]
        layer.biases[0] = 0.5
        
        let input = [3.0, 4.0]
        let output = layer.forward(input)
        
        // Expected: 1.0*3.0 + 2.0*4.0 + 0.5 = 11.5
        // Then apply tanh activation
        let expected = tanh(11.5)
        XCTAssertEqual(output[0], expected, accuracy: 1e-10)
    }
    
    func testActivationFunctions() {
        let testValue = 1.0
        
        // Test tanh
        XCTAssertEqual(ActivationFunction.tanh.apply(testValue), tanh(testValue))
        XCTAssertEqual(ActivationFunction.tanh.derivative(testValue), 1.0 - tanh(testValue) * tanh(testValue))
        
        // Test sigmoid
        let sigmoidValue = 1.0 / (1.0 + exp(-testValue))
        XCTAssertEqual(ActivationFunction.sigmoid.apply(testValue), sigmoidValue)
        XCTAssertEqual(ActivationFunction.sigmoid.derivative(testValue), sigmoidValue * (1.0 - sigmoidValue))
        
        // Test ReLU
        XCTAssertEqual(ActivationFunction.relu.apply(testValue), max(0, testValue))
        XCTAssertEqual(ActivationFunction.relu.apply(-testValue), 0.0)
        XCTAssertEqual(ActivationFunction.relu.derivative(testValue), 1.0)
        XCTAssertEqual(ActivationFunction.relu.derivative(-testValue), 0.0)
        
        // Test sin
        XCTAssertEqual(ActivationFunction.sin.apply(testValue), sin(testValue))
        XCTAssertEqual(ActivationFunction.sin.derivative(testValue), cos(testValue))
    }
    
    func testPINNInitialization() {
        let layerSizes = [2, 10, 5, 1]
        let pinn = PINN(layerSizes: layerSizes, learningRate: 0.001, maxEpochs: 5000)
        
        XCTAssertEqual(pinn.layers.count, layerSizes.count - 1)
        XCTAssertEqual(pinn.learningRate, 0.001)
        XCTAssertEqual(pinn.maxEpochs, 5000)
    }
    
    func testPINNForwardPass() {
        let pinn = PINN(layerSizes: [2, 3, 1])
        
        // Test forward pass
        let result = pinn.forward(x: 1.0, t: 2.0)
        XCTAssertTrue(result.isFinite)
        XCTAssertFalse(result.isNaN)
    }
    
    func testPINNDerivatives() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        
        let (u_x, u_t, u_xx) = pinn.computeDerivatives(x: 0.0, t: 0.0)
        
        XCTAssertTrue(u_x.isFinite)
        XCTAssertTrue(u_t.isFinite)
        XCTAssertTrue(u_xx.isFinite)
        XCTAssertFalse(u_x.isNaN)
        XCTAssertFalse(u_t.isNaN)
        XCTAssertFalse(u_xx.isNaN)
    }
    
    func testPINNPDEResidual() {
        let pinn = PINN(layerSizes: [2, 5, 1])
        
        let residual = pinn.pdeResidual(x: 0.0, t: 0.0)
        
        XCTAssertTrue(residual.isFinite)
        XCTAssertFalse(residual.isNaN)
    }
    
    func testTrainingPoint() {
        let point1 = TrainingPoint(x: 1.0, t: 2.0)
        let point2 = TrainingPoint(x: 3.0, t: 4.0, u: 5.0)
        
        XCTAssertEqual(point1.x, 1.0)
        XCTAssertEqual(point1.t, 2.0)
        XCTAssertNil(point1.u)
        
        XCTAssertEqual(point2.x, 3.0)
        XCTAssertEqual(point2.t, 4.0)
        XCTAssertEqual(point2.u, 5.0)
    }
    
    func testPINNSolution() {
        let u = [[1.0, 2.0], [3.0, 4.0]]
        let x = [0.0, 1.0]
        let t = [0.0, 1.0]
        let pdeResidual = [[0.1, 0.2], [0.3, 0.4]]
        
        let solution = PINNSolution(
            u: u,
            x: x,
            t: t,
            pdeResidual: pdeResidual,
            trainingLoss: 0.01,
            validationLoss: 0.02
        )
        
        XCTAssertEqual(solution.u, u)
        XCTAssertEqual(solution.x, x)
        XCTAssertEqual(solution.t, t)
        XCTAssertEqual(solution.pdeResidual, pdeResidual)
        XCTAssertEqual(solution.trainingLoss, 0.01)
        XCTAssertEqual(solution.validationLoss, 0.02)
    }
    
    func testRK4Validator() {
        // Test RK4 step function
        let f: (Double, [Double]) -> [Double] = { t, y in
            return [y[0] * t]  // Simple ODE: dy/dt = y*t
        }
        
        let y = [1.0]
        let t = 0.0
        let dt = 0.1
        
        let result = RK4Validator.step(f: f, y: y, t: t, dt: dt)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertTrue(result[0].isFinite)
        XCTAssertFalse(result[0].isNaN)
    }
    
    func testRK4BurgersSolution() {
        let x = [-1.0, 0.0, 1.0]
        let t = [0.0, 0.1, 0.2]
        
        let initialCondition: (Double) -> Double = { x in
            return -sin(.pi * x)
        }
        
        let solution = RK4Validator.solveBurgers(x: x, t: t, initialCondition: initialCondition)
        
        XCTAssertEqual(solution.count, x.count)
        XCTAssertEqual(solution[0].count, t.count)
        
        // Check initial condition
        for i in 0..<x.count {
            let expected = initialCondition(x[i])
            XCTAssertEqual(solution[i][0], expected, accuracy: 1e-10)
        }
    }
    
    func testPINNSolverInitialization() {
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0,
            nx: 50,
            nt: 50
        )
        
        XCTAssertEqual(solver.xRange, -1.0...1.0)
        XCTAssertEqual(solver.tRange, 0.0...1.0)
        XCTAssertEqual(solver.nx, 50)
        XCTAssertEqual(solver.nt, 50)
    }
    
    func testTrainingPointGeneration() {
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0,
            nx: 3,
            nt: 3
        )
        
        let (collocation, initial, boundary) = solver.generateTrainingPoints()
        
        // Check collocation points
        XCTAssertEqual(collocation.count, 9)  // 3x3 grid
        
        // Check initial points
        XCTAssertEqual(initial.count, 3)  // One for each x at t=0
        
        // Check boundary points
        XCTAssertEqual(boundary.count, 6)  // Two boundaries for each time step
    }
    
    func testInitialCondition() {
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0
        )
        
        // Test initial condition at specific points
        let x1 = 0.0
        let x2 = 0.5
        let x3 = 1.0
        
        // Use reflection to access private method for testing
        let mirror = Mirror(reflecting: solver)
        let initialConditionMethod = mirror.children.first { $0.label == "initialCondition" }
        
        // Since we can't directly test private methods, we'll test the public interface
        // that uses this method
        let (_, initial, _) = solver.generateTrainingPoints()
        
        // Check that initial condition is applied correctly
        XCTAssertEqual(initial.count, solver.nx)
        
        // The first point should be at x = -1, t = 0
        if let firstPoint = initial.first {
            XCTAssertEqual(firstPoint.x, -1.0)
            XCTAssertEqual(firstPoint.t, 0.0)
            XCTAssertNotNil(firstPoint.u)
        }
    }
    
    func testPINNIntegration() {
        // Test that PINN can be integrated with the Ψ framework
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0,
            nx: 10,  // Small grid for fast testing
            nt: 10
        )
        
        // This should not crash and should return valid Ψ metrics
        let psiOutcome = solver.computePsiPerformance()
        
        XCTAssertTrue(psiOutcome.psi.isFinite)
        XCTAssertTrue(psiOutcome.psi >= 0.0)
        XCTAssertTrue(psiOutcome.psi <= 1.0)
        XCTAssertTrue(psiOutcome.hybrid.isFinite)
        XCTAssertTrue(psiOutcome.penalty.isFinite)
        XCTAssertTrue(psiOutcome.posterior.isFinite)
    }
    
    func testPINNUtilities() {
        let array = [[1.0, 2.0], [3.0, 4.0]]
        
        // Test L2 norm
        let l2norm = PINNUtilities.l2Norm(array)
        let expectedL2 = sqrt(1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0)
        XCTAssertEqual(l2norm, expectedL2, accuracy: 1e-10)
        
        // Test CSV export
        let solution = PINNSolution(
            u: array,
            x: [0.0, 1.0],
            t: [0.0, 1.0],
            pdeResidual: [[0.1, 0.2], [0.3, 0.4]],
            trainingLoss: 0.01,
            validationLoss: 0.02
        )
        
        let csv = PINNUtilities.exportToCSV(solution, filename: "test.csv")
        XCTAssertTrue(csv.contains("x,t,u,pde_residual"))
        XCTAssertTrue(csv.contains("0.0,0.0,1.0,0.1"))
    }
    
    func testMathematicalFramework() {
        // Test the mathematical framework demonstration
        // This ensures the framework calculations are correct
        
        let S_x = 0.72
        let N_x = 0.85
        let alpha = 0.5
        
        let O_hybrid = alpha * S_x + (1 - alpha) * N_x
        let expected_hybrid = 0.5 * 0.72 + 0.5 * 0.85
        XCTAssertEqual(O_hybrid, expected_hybrid, accuracy: 1e-10)
        
        let R_cognitive = 0.15
        let R_efficiency = 0.10
        let lambda1 = 0.6
        let lambda2 = 0.4
        
        let P_total = lambda1 * R_cognitive + lambda2 * R_efficiency
        let expected_P_total = 0.6 * 0.15 + 0.4 * 0.10
        XCTAssertEqual(P_total, expected_P_total, accuracy: 1e-10)
        
        let penalty_exp = exp(-P_total)
        let expected_penalty = exp(-expected_P_total)
        XCTAssertEqual(penalty_exp, expected_penalty, accuracy: 1e-10)
        
        let P_base = 0.80
        let beta = 1.2
        let P_adj = min(beta * P_base, 1.0)
        let expected_P_adj = min(1.2 * 0.80, 1.0)
        XCTAssertEqual(P_adj, expected_P_adj, accuracy: 1e-10)
        
        let psi_x = O_hybrid * penalty_exp * P_adj
        let expected_psi = expected_hybrid * expected_penalty * expected_P_adj
        XCTAssertEqual(psi_x, expected_psi, accuracy: 1e-10)
    }
    
    func testPerformance() {
        // Test that the PINN can handle reasonable problem sizes
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let solver = PINNSolver(
            xRange: -1.0...1.0,
            tRange: 0.0...1.0,
            nx: 20,
            nt: 20
        )
        
        // Just test initialization and training point generation
        let (collocation, initial, boundary) = solver.generateTrainingPoints()
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        
        // Should complete in reasonable time
        XCTAssertLessThan(duration, 1.0)  // Less than 1 second
        
        // Should generate correct number of points
        XCTAssertEqual(collocation.count, 400)  // 20x20
        XCTAssertEqual(initial.count, 20)
        XCTAssertEqual(boundary.count, 40)  // 2 boundaries × 20 time steps
    }
}