import Foundation

// MARK: - PINN Test Runner
class PINNTestRunner {
    
    static func runAllTests() {
        print("🧪 PINN Test Suite")
        print("=" * 50)
        
        testPINNInitialization()
        testHybridFramework()
        testRK4Solver()
        testComparison()
        testChartJSExport()
        
        print("\n✅ All tests completed!")
    }
    
    // Test 1: PINN Initialization and Forward Pass
    static func testPINNInitialization() {
        print("\n1️⃣ Testing PINN Initialization...")
        
        let pinn = PINN()
        let result = pinn.forward(0.0, 0.0)
        
        print("   ✓ PINN initialized successfully")
        print("   ✓ Forward pass result: \(String(format: "%.6f", result))")
        
        // Test derivatives
        let derivatives = pinn.computeDerivatives(0.0, 0.0)
        print("   ✓ Derivatives computed: u_x=\(String(format: "%.6f", derivatives.u_x)), u_t=\(String(format: "%.6f", derivatives.u_t)), u_xx=\(String(format: "%.6f", derivatives.u_xx))")
        
        // Test PDE residual
        let residual = pinn.pdeResidual(0.0, 0.0)
        print("   ✓ PDE residual: \(String(format: "%.6f", residual))")
    }
    
    // Test 2: Hybrid Framework
    static func testHybridFramework() {
        print("\n2️⃣ Testing Hybrid Framework...")
        
        let pinn = PINN()
        
        // Test individual components
        let sX = HybridFramework.stateInference(0.0, 0.5, model: pinn)
        let nX = HybridFramework.gradientDescentAnalysis(loss: 0.1, previousLoss: 0.2)
        let alphaT = HybridFramework.realTimeValidation(0.5)
        
        print("   ✓ S(x) = \(String(format: "%.3f", sX))")
        print("   ✓ N(x) = \(String(format: "%.3f", nX))")
        print("   ✓ α(t) = \(String(format: "%.3f", alphaT))")
        
        // Test hybrid output computation
        let (hybrid, psi) = HybridFramework.computeHybridOutput(
            sX: sX, nX: nX, alphaT: alphaT,
            rCognitive: 0.15, rEfficiency: 0.10
        )
        
        print("   ✓ O_hybrid = \(String(format: "%.3f", hybrid))")
        print("   ✓ Ψ(x) = \(String(format: "%.3f", psi))")
        
        // Validate ranges
        assert(sX >= 0.0 && sX <= 1.0, "S(x) should be in [0,1]")
        assert(nX >= 0.0 && nX <= 1.0, "N(x) should be in [0,1]")
        assert(alphaT >= 0.0 && alphaT <= 1.0, "α(t) should be in [0,1]")
        assert(psi >= 0.0 && psi <= 1.0, "Ψ(x) should be in [0,1]")
        
        print("   ✓ All values within expected ranges")
    }
    
    // Test 3: RK4 Solver
    static func testRK4Solver() {
        print("\n3️⃣ Testing RK4 Solver...")
        
        let rk4 = RK4Solver()
        let xPoints = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        // Test initial condition
        let ic = rk4.initialCondition(0.0)
        print("   ✓ Initial condition at x=0: \(String(format: "%.6f", ic))")
        
        // Test solution at t=0.1
        let solution = rk4.solve(xPoints: xPoints, finalTime: 0.1)
        print("   ✓ RK4 solution computed for \(xPoints.count) points")
        print("   ✓ Solution values: \(solution.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
        
        // Validate boundary conditions
        assert(abs(solution.first ?? 1.0) < 1e-10, "Left boundary should be ~0")
        assert(abs(solution.last ?? 1.0) < 1e-10, "Right boundary should be ~0")
        
        print("   ✓ Boundary conditions satisfied")
    }
    
    // Test 4: PINN vs RK4 Comparison
    static func testComparison() {
        print("\n4️⃣ Testing PINN vs RK4 Comparison...")
        
        let pinn = PINN()
        let rk4 = RK4Solver()
        let xPoints = Array(stride(from: -1.0, through: 1.0, by: 0.2))
        
        let comparison = PINNRKComparison.compareAtTime(
            pinnModel: pinn,
            rk4Solver: rk4,
            xPoints: xPoints,
            time: 0.5
        )
        
        print("   ✓ Comparison completed for \(xPoints.count) points")
        print("   ✓ MSE: \(String(format: "%.8f", comparison.mse))")
        print("   ✓ Max Error: \(String(format: "%.8f", comparison.maxError))")
        
        // Test BNSL analysis
        let mockHistory = [(epoch: 50, loss: 0.1, psi: 0.6), 
                          (epoch: 100, loss: 0.05, psi: 0.7),
                          (epoch: 150, loss: 0.02, psi: 0.8)]
        
        let (inflectionPoints, scaling) = PINNRKComparison.analyzeBNSL(trainingHistory: mockHistory)
        print("   ✓ BNSL Analysis: \(scaling)")
        print("   ✓ Inflection points: \(inflectionPoints)")
    }
    
    // Test 5: Chart.js Export
    static func testChartJSExport() {
        print("\n5️⃣ Testing Chart.js Export...")
        
        let xPoints = [-1.0, 0.0, 1.0]
        let pinnData = [0.0, 0.5, 0.0]
        let rk4Data = [0.0, 0.6, 0.0]
        
        let chartConfig = PINNExample.generateChartJSConfig(
            xPoints: xPoints,
            pinnData: pinnData,
            rk4Data: rk4Data
        )
        
        print("   ✓ Chart.js configuration generated")
        print("   ✓ Config length: \(chartConfig.count) characters")
        
        // Validate JSON structure
        assert(chartConfig.contains("\"type\": \"line\""), "Should contain line chart type")
        assert(chartConfig.contains("PINN u"), "Should contain PINN label")
        assert(chartConfig.contains("RK4 u"), "Should contain RK4 label")
        assert(chartConfig.contains("#1E90FF"), "Should contain PINN color")
        assert(chartConfig.contains("#FF4500"), "Should contain RK4 color")
        
        print("   ✓ Chart.js structure validated")
    }
    
    // Comprehensive Integration Test
    static func runIntegrationTest() {
        print("\n🔄 Running Integration Test...")
        
        // Create models
        let pinn = PINN()
        let trainer = PINNTrainer(model: pinn)
        let rk4 = RK4Solver()
        
        // Generate small dataset
        let x = [-1.0, -0.5, 0.0, 0.5, 1.0]
        let t = [0.0, 0.25, 0.5]
        
        // Train for a few epochs
        trainer.train(epochs: 10, x: x, t: t, printEvery: 5)
        
        // Compare solutions
        let comparison = PINNRKComparison.compareAtTime(
            pinnModel: pinn,
            rk4Solver: rk4,
            xPoints: x,
            time: 0.5
        )
        
        print("   ✓ Integration test completed")
        print("   ✓ Final MSE: \(String(format: "%.6f", comparison.mse))")
        
        // Generate visualization data
        let vizData = PINNRKComparison.generateVisualizationData(
            pinnModel: pinn,
            rk4Solver: rk4,
            time: 0.5
        )
        
        print("   ✓ Visualization data generated: \(vizData.xPoints.count) points")
        
        let trainingHistory = trainer.getTrainingHistory()
        if !trainingHistory.isEmpty {
            let finalPsi = trainingHistory.last?.psi ?? 0.0
            print("   ✓ Final Ψ(x): \(String(format: "%.3f", finalPsi))")
        }
    }
}

// MARK: - Test Execution Extension
extension PINNExample {
    static func runTests() {
        PINNTestRunner.runAllTests()
        PINNTestRunner.runIntegrationTest()
        
        print("\n🎯 Running Complete Example...")
        runCompleteExample()
    }
}