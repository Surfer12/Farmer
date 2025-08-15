import XCTest
@testable import CFDVisualizationApp

/// Comprehensive test suite for CFD Visualization App
final class CFDVisualizationAppTests: XCTestCase {
    
    // MARK: - Test Properties
    
    var finGeometry: FinGeometry!
    var cfdPredictor: FinCFDPredictor!
    var sensorManager: SensorManager!
    var viewModel: CFDViewModel!
    
    // MARK: - Setup and Teardown
    
    override func setUpWithError() throws {
        finGeometry = FinGeometry.sideFin
        cfdPredictor = FinCFDPredictor()
        sensorManager = SensorManager()
        viewModel = CFDViewModel()
    }
    
    override func tearDownWithError() throws {
        finGeometry = nil
        cfdPredictor = nil
        sensorManager = nil
        viewModel = nil
    }
    
    // MARK: - Fin Geometry Tests
    
    func testFinGeometrySpecifications() throws {
        // Test Vector 3/2 Blackstix+ specifications
        XCTAssertEqual(FinGeometry.sideFinArea, 15.00, accuracy: 0.01, "Side fin area should be 15.00 sq.in")
        XCTAssertEqual(FinGeometry.centerFinArea, 14.50, accuracy: 0.01, "Center fin area should be 14.50 sq.in")
        XCTAssertEqual(FinGeometry.sideFinAngle, 6.5, accuracy: 0.1, "Side fin angle should be 6.5°")
        XCTAssertEqual(FinGeometry.finHeight, 4.48, accuracy: 0.01, "Fin height should be 4.48 inches")
        XCTAssertEqual(FinGeometry.finBase, 4.63, accuracy: 0.01, "Fin base should be 4.63 inches")
    }
    
    func testFinGeometryCreation() throws {
        let sideFin = FinGeometry.sideFin
        let centerFin = FinGeometry.centerFin
        
        XCTAssertEqual(sideFin.foilType, .vector32, "Side fin should use Vector 3/2 foil")
        XCTAssertEqual(centerFin.foilType, .symmetric, "Center fin should use symmetric foil")
        XCTAssertEqual(sideFin.angle, 6.5, accuracy: 0.1, "Side fin should have 6.5° angle")
        XCTAssertEqual(centerFin.angle, 0.0, accuracy: 0.1, "Center fin should have 0° angle")
    }
    
    func testLiftCoefficientCalculation() throws {
        let sideFin = FinGeometry.sideFin
        let centerFin = FinGeometry.centerFin
        
        // Test at 10° angle of attack
        let sideFinCl = sideFin.calculateLiftCoefficient(angleOfAttack: 10.0)
        let centerFinCl = centerFin.calculateLiftCoefficient(angleOfAttack: 10.0)
        
        // Side fin should have 12% higher lift coefficient
        XCTAssertGreaterThan(sideFinCl, centerFinCl, "Side fin should have higher lift coefficient")
        XCTAssertEqual(sideFinCl / centerFinCl, 1.12, accuracy: 0.01, "Side fin should have 12% higher Cl")
    }
    
    func testDragCoefficientCalculation() throws {
        let sideFin = FinGeometry.sideFin
        let centerFin = FinGeometry.centerFin
        
        // Test at 10° angle of attack
        let sideFinCd = sideFin.calculateDragCoefficient(angleOfAttack: 10.0)
        let centerFinCd = centerFin.calculateDragCoefficient(angleOfAttack: 10.0)
        
        // Side fin should have lower drag coefficient
        XCTAssertLessThan(sideFinCd, centerFinCd, "Side fin should have lower drag coefficient")
    }
    
    // MARK: - CFD Data Tests
    
    func testCFDDataGeneration() throws {
        let cfdData = CFDData(reynoldsNumber: 5e5, angleOfAttack: 10.0, rakeAngle: 6.5)
        
        XCTAssertEqual(cfdData.angleOfAttack, 10.0, accuracy: 0.1, "Angle of attack should be preserved")
        XCTAssertEqual(cfdData.rakeAngle, 6.5, accuracy: 0.1, "Rake angle should be preserved")
        XCTAssertEqual(cfdData.reynoldsNumber, 5e5, accuracy: 1e4, "Reynolds number should be preserved")
        
        // Test pressure differential (30% for Vector 3/2)
        XCTAssertGreaterThan(cfdData.pressureDifferential, 0.0, "Pressure differential should be positive")
        XCTAssertLessThanOrEqual(cfdData.pressureDifferential, 0.30, "Pressure differential should not exceed 30%")
        
        // Test pressure map generation
        XCTAssertEqual(cfdData.pressureMap.count, 64 * 64, "Pressure map should have 64x64 elements")
        XCTAssertTrue(cfdData.pressureMap.allSatisfy { $0 >= 0.0 && $0 <= 1.0 }, "Pressure values should be normalized")
        
        // Test flow vectors
        XCTAssertEqual(cfdData.flowVectors.count, 100, "Should generate 100 flow vectors")
        XCTAssertTrue(cfdData.flowVectors.allSatisfy { $0.magnitude > 0 }, "All flow vectors should have positive magnitude")
    }
    
    func testCFDFlowRegimeDetection() throws {
        // Test laminar flow (low AoA, high Re)
        let laminarData = CFDData(reynoldsNumber: 1e6, angleOfAttack: 2.0, rakeAngle: 0.0)
        if case .laminar = laminarData.flowRegime {
            // Test passes
        } else {
            XCTFail("Should detect laminar flow for low AoA and high Re")
        }
        
        // Test separated flow (high AoA)
        let separatedData = CFDData(reynoldsNumber: 5e5, angleOfAttack: 18.0, rakeAngle: 0.0)
        if case .separated = separatedData.flowRegime {
            // Test passes
        } else {
            XCTFail("Should detect separated flow for high AoA")
        }
    }
    
    func testForceCalculation() throws {
        let cfdData = CFDData(reynoldsNumber: 5e5, angleOfAttack: 10.0, rakeAngle: 6.5)
        let forces = cfdData.calculateForces(riderWeight: 150.0)
        
        XCTAssertGreaterThan(forces.lift, 0.0, "Lift force should be positive at positive AoA")
        XCTAssertGreaterThan(forces.drag, 0.0, "Drag force should always be positive")
    }
    
    // MARK: - CFD Predictor Tests
    
    func testCFDPredictorInitialization() throws {
        XCTAssertNotNil(cfdPredictor, "CFD predictor should initialize successfully")
    }
    
    func testCoefficientPrediction() throws {
        let prediction = cfdPredictor.predictCoefficients(
            angleOfAttack: 10.0,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        
        XCTAssertNotNil(prediction, "Prediction should not be nil")
        XCTAssertGreaterThan(prediction!.lift, 0.0, "Lift coefficient should be positive")
        XCTAssertGreaterThan(prediction!.drag, 0.0, "Drag coefficient should be positive")
    }
    
    func testVector32Performance() throws {
        let prediction = cfdPredictor.predictVector32Performance(
            angleOfAttack: 10.0,
            reynoldsNumber: 5e5
        )
        
        XCTAssertNotNil(prediction, "Vector 3/2 prediction should not be nil")
        XCTAssertEqual(prediction!.performanceGain, 0.12, accuracy: 0.01, "Should show 12% performance gain")
        XCTAssertGreaterThan(prediction!.liftToDragRatio, 0.0, "L/D ratio should be positive")
        XCTAssertGreaterThan(prediction!.confidence, 0.0, "Confidence should be positive")
        XCTAssertLessThanOrEqual(prediction!.confidence, 1.0, "Confidence should not exceed 1.0")
    }
    
    func testPerformanceCurveGeneration() throws {
        let curve = cfdPredictor.predictPerformanceCurve(
            aoaRange: 0...20,
            steps: 21,
            reynoldsNumber: 5e5
        )
        
        XCTAssertEqual(curve.count, 21, "Should generate 21 prediction points")
        XCTAssertEqual(curve.first?.angleOfAttack, 0.0, accuracy: 0.1, "First point should be at 0° AoA")
        XCTAssertEqual(curve.last?.angleOfAttack, 20.0, accuracy: 0.1, "Last point should be at 20° AoA")
        
        // Test that curve shows realistic behavior
        let maxLDRatio = curve.max { $0.liftToDragRatio < $1.liftToDragRatio }
        XCTAssertNotNil(maxLDRatio, "Should find maximum L/D ratio")
        XCTAssertGreaterThan(maxLDRatio!.angleOfAttack, 5.0, "Optimal AoA should be greater than 5°")
        XCTAssertLessThan(maxLDRatio!.angleOfAttack, 15.0, "Optimal AoA should be less than 15°")
    }
    
    func testInputValidation() throws {
        // Test valid inputs
        let validResult = cfdPredictor.validateInputs(
            angleOfAttack: 10.0,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        XCTAssertTrue(validResult.isValid, "Valid inputs should pass validation")
        XCTAssertTrue(validResult.warnings.isEmpty, "Valid inputs should have no warnings")
        
        // Test invalid angle of attack
        let invalidAoAResult = cfdPredictor.validateInputs(
            angleOfAttack: 30.0,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        XCTAssertFalse(invalidAoAResult.isValid, "Invalid AoA should fail validation")
        
        // Test invalid Reynolds number
        let invalidReResult = cfdPredictor.validateInputs(
            angleOfAttack: 10.0,
            rakeAngle: 6.5,
            reynoldsNumber: 1e2
        )
        XCTAssertFalse(invalidReResult.isValid, "Invalid Re should fail validation")
    }
    
    // MARK: - Sensor Manager Tests
    
    func testSensorManagerInitialization() throws {
        XCTAssertNotNil(sensorManager, "Sensor manager should initialize successfully")
        XCTAssertEqual(sensorManager.sensorStatus, .disconnected, "Initial status should be disconnected")
        XCTAssertFalse(sensorManager.isConnected, "Should not be connected initially")
    }
    
    func testAngleOfAttackCalculation() throws {
        // Set a known pitch angle
        sensorManager.pitch = Float(10.0 * .pi / 180.0) // 10 degrees in radians
        
        let aoa = sensorManager.getCurrentAngleOfAttack()
        XCTAssertEqual(aoa, 10.0, accuracy: 0.1, "Should convert pitch to AoA correctly")
        
        // Test clamping
        sensorManager.pitch = Float(30.0 * .pi / 180.0) // 30 degrees
        let clampedAoa = sensorManager.getCurrentAngleOfAttack()
        XCTAssertEqual(clampedAoa, 20.0, accuracy: 0.1, "Should clamp AoA to maximum 20°")
    }
    
    func testFlowStateMetrics() throws {
        // Set up test data
        sensorManager.heartRate = 150.0
        sensorManager.heartRateVariability = 40.0
        
        let metrics = sensorManager.getFlowStateMetrics()
        
        XCTAssertEqual(metrics.heartRate, 150.0, accuracy: 0.1, "Heart rate should be preserved")
        XCTAssertEqual(metrics.hrv, 40.0, accuracy: 0.1, "HRV should be preserved")
        XCTAssertGreaterThanOrEqual(metrics.flowScore, 0.0, "Flow score should be non-negative")
        XCTAssertLessThanOrEqual(metrics.flowScore, 1.0, "Flow score should not exceed 1.0")
    }
    
    // MARK: - View Model Tests
    
    func testViewModelInitialization() throws {
        XCTAssertNotNil(viewModel, "View model should initialize successfully")
        XCTAssertEqual(viewModel.currentAngleOfAttack, 0.0, accuracy: 0.1, "Initial AoA should be 0°")
        XCTAssertEqual(viewModel.currentRakeAngle, 6.5, accuracy: 0.1, "Initial rake should be 6.5°")
        XCTAssertEqual(viewModel.reynoldsNumber, 5e5, accuracy: 1e4, "Initial Re should be 5e5")
        XCTAssertFalse(viewModel.isRealTimeMode, "Should not be in real-time mode initially")
    }
    
    func testAngleOfAttackSetting() throws {
        viewModel.setAngleOfAttack(15.0)
        XCTAssertEqual(viewModel.currentAngleOfAttack, 15.0, accuracy: 0.1, "Should set AoA correctly")
        
        // Test clamping
        viewModel.setAngleOfAttack(25.0)
        XCTAssertEqual(viewModel.currentAngleOfAttack, 20.0, accuracy: 0.1, "Should clamp AoA to maximum")
        
        viewModel.setAngleOfAttack(-25.0)
        XCTAssertEqual(viewModel.currentAngleOfAttack, -20.0, accuracy: 0.1, "Should clamp AoA to minimum")
    }
    
    func testRakeAngleSetting() throws {
        viewModel.setRakeAngle(8.0)
        XCTAssertEqual(viewModel.currentRakeAngle, 8.0, accuracy: 0.1, "Should set rake angle correctly")
        
        // Test clamping
        viewModel.setRakeAngle(15.0)
        XCTAssertEqual(viewModel.currentRakeAngle, 10.0, accuracy: 0.1, "Should clamp rake to maximum")
        
        viewModel.setRakeAngle(-5.0)
        XCTAssertEqual(viewModel.currentRakeAngle, 0.0, accuracy: 0.1, "Should clamp rake to minimum")
    }
    
    func testReynoldsNumberSetting() throws {
        viewModel.setReynoldsNumber(1e6)
        XCTAssertEqual(viewModel.reynoldsNumber, 1e6, accuracy: 1e4, "Should set Re correctly")
        
        // Test clamping
        viewModel.setReynoldsNumber(1e8)
        XCTAssertEqual(viewModel.reynoldsNumber, 1e7, accuracy: 1e5, "Should clamp Re to maximum")
        
        viewModel.setReynoldsNumber(1e3)
        XCTAssertEqual(viewModel.reynoldsNumber, 1e4, accuracy: 1e2, "Should clamp Re to minimum")
    }
    
    func testPerformanceComparison() throws {
        // Set up test conditions
        viewModel.setAngleOfAttack(10.0)
        
        // Wait for predictions to complete (in real app, this would be async)
        let comparison = viewModel.getPerformanceComparison()
        
        XCTAssertGreaterThan(comparison.improvement, 0.0, "Should show performance improvement")
        XCTAssertEqual(comparison.improvement, 0.12, accuracy: 0.01, "Should show 12% improvement")
    }
    
    // MARK: - Performance Tests
    
    func testCFDPredictionPerformance() throws {
        measure {
            _ = cfdPredictor.predictCoefficients(
                angleOfAttack: 10.0,
                rakeAngle: 6.5,
                reynoldsNumber: 5e5
            )
        }
    }
    
    func testPerformanceCurveGeneration() throws {
        measure {
            _ = cfdPredictor.predictPerformanceCurve(
                aoaRange: 0...20,
                steps: 21,
                reynoldsNumber: 5e5
            )
        }
    }
    
    func testCFDDataGeneration() throws {
        measure {
            _ = CFDData(reynoldsNumber: 5e5, angleOfAttack: 10.0, rakeAngle: 6.5)
        }
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndCFDPipeline() throws {
        // Test complete pipeline from sensor data to visualization
        
        // 1. Set up sensor data
        viewModel.setAngleOfAttack(10.0)
        viewModel.setRakeAngle(6.5)
        viewModel.setReynoldsNumber(5e5)
        
        // 2. Verify CFD predictions are generated
        XCTAssertNotNil(viewModel.cfdData, "CFD data should be generated")
        
        // 3. Verify performance metrics are calculated
        XCTAssertGreaterThan(viewModel.liftCoefficient, 0.0, "Lift coefficient should be positive")
        XCTAssertGreaterThan(viewModel.dragCoefficient, 0.0, "Drag coefficient should be positive")
        XCTAssertGreaterThan(viewModel.liftToDragRatio, 0.0, "L/D ratio should be positive")
        
        // 4. Verify forces are calculated
        let forces = viewModel.getCurrentForces()
        XCTAssertGreaterThan(forces.lift, 0.0, "Lift force should be positive")
        XCTAssertGreaterThan(forces.drag, 0.0, "Drag force should be positive")
        
        // 5. Verify performance curve is generated
        XCTAssertFalse(viewModel.performanceCurve.isEmpty, "Performance curve should be generated")
    }
    
    func testFlowStateIntegration() throws {
        // Test integration between sensor data and flow state calculation
        
        // Set up realistic sensor data
        sensorManager.heartRate = 150.0
        sensorManager.heartRateVariability = 45.0
        sensorManager.pitch = Float(10.0 * .pi / 180.0)
        
        let metrics = sensorManager.getFlowStateMetrics()
        
        // Verify flow state calculation
        XCTAssertGreaterThan(metrics.flowScore, 0.0, "Flow score should be positive")
        XCTAssertNotEqual(metrics.flowState, .poor, "Should not be in poor flow state with good metrics")
        
        // Verify angle of attack conversion
        let aoa = sensorManager.getCurrentAngleOfAttack()
        XCTAssertEqual(aoa, 10.0, accuracy: 0.1, "Should convert sensor data to AoA")
    }
    
    // MARK: - Error Handling Tests
    
    func testErrorHandling() throws {
        // Test prediction with invalid inputs
        let invalidPrediction = cfdPredictor.predictCoefficients(
            angleOfAttack: Float.nan,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        
        XCTAssertNil(invalidPrediction, "Should return nil for invalid inputs")
    }
    
    func testEdgeCases() throws {
        // Test zero angle of attack
        let zeroPrediction = cfdPredictor.predictCoefficients(
            angleOfAttack: 0.0,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        
        XCTAssertNotNil(zeroPrediction, "Should handle zero AoA")
        XCTAssertEqual(zeroPrediction!.lift, 0.0, accuracy: 0.01, "Lift should be zero at zero AoA")
        
        // Test maximum angle of attack
        let maxPrediction = cfdPredictor.predictCoefficients(
            angleOfAttack: 20.0,
            rakeAngle: 6.5,
            reynoldsNumber: 5e5
        )
        
        XCTAssertNotNil(maxPrediction, "Should handle maximum AoA")
        XCTAssertGreaterThan(maxPrediction!.drag, maxPrediction!.lift, "Drag should exceed lift at high AoA")
    }
}

// MARK: - Mock Classes for Testing

class MockSensorManager: SensorManager {
    var mockHeartRate: Double = 150.0
    var mockHRV: Double = 40.0
    var mockPitch: Float = 0.0
    
    override var heartRate: Double {
        get { mockHeartRate }
        set { mockHeartRate = newValue }
    }
    
    override var heartRateVariability: Double {
        get { mockHRV }
        set { mockHRV = newValue }
    }
    
    override var pitch: Float {
        get { mockPitch }
        set { mockPitch = newValue }
    }
    
    override func startMonitoring() {
        // Mock implementation - don't actually start sensors
        isConnected = true
        sensorStatus = .connected
    }
    
    override func stopMonitoring() {
        // Mock implementation
        isConnected = false
        sensorStatus = .disconnected
    }
}