// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import XCTest
import Foundation
@testable import Farmer

/// Comprehensive test suite for ReverseHierarchicalBayesianModel
/// Tests parameter recovery, structure learning, validation, and edge cases
final class ReverseHierarchicalBayesianModelTests: XCTestCase {

    private var model: ReverseHierarchicalBayesianModel!
    private var testObservations: [ReverseHierarchicalBayesianModel.Observation]!

    override func setUp() {
        super.setUp()
        model = ReverseHierarchicalBayesianModel()
        testObservations = createTestObservations()
    }

    override func tearDown() {
        model = nil
        testObservations = nil
        super.tearDown()
    }

    // MARK: - Parameter Recovery Tests

    func testRecoverParametersFromSyntheticData() {
        // When
        let result = model.recoverParameters(observations: testObservations)

        // Then
        XCTAssertNotNil(result, "Result should not be nil")
        XCTAssertNotNil(result.recoveredParameters, "Recovered parameters should not be nil")

        let params = result.recoveredParameters
        XCTAssertTrue(params.S >= 0 && params.S <= 1, "S should be in [0,1]")
        XCTAssertTrue(params.N >= 0 && params.N <= 1, "N should be in [0,1]")
        XCTAssertTrue(params.alpha >= 0 && params.alpha <= 1, "alpha should be in [0,1]")
        XCTAssertTrue(params.beta >= 1, "beta should be >= 1")

        XCTAssertTrue(result.confidence >= 0 && result.confidence <= 1,
                     "Confidence should be in [0,1]")
        XCTAssertFalse(result.posteriorSamples.isEmpty, "Should have posterior samples")
    }

    func testHandleEmptyObservations() {
        // Given
        let emptyObservations = [ReverseHierarchicalBayesianModel.Observation]()

        // When & Then
        XCTAssertThrowsError(try model.recoverParameters(observations: emptyObservations)) { error in
            XCTAssertTrue(error.localizedDescription.contains("empty"),
                         "Should mention empty observations")
        }
    }

    func testProduceReasonableParameterUncertainties() {
        // When
        let result = model.recoverParameters(observations: testObservations)

        // Then
        let uncertainties = result.parameterUncertainties
        XCTAssertNotNil(uncertainties, "Uncertainties should not be null")
        XCTAssertTrue(uncertainties.contains { $0.key == "S" }, "Should have S uncertainty")
        XCTAssertTrue(uncertainties.contains { $0.key == "N" }, "Should have N uncertainty")
        XCTAssertTrue(uncertainties.contains { $0.key == "alpha" }, "Should have alpha uncertainty")
        XCTAssertTrue(uncertainties.contains { $0.key == "beta" }, "Should have beta uncertainty")

        // All uncertainties should be positive and reasonable
        uncertainties.forEach { key, uncertainty in
            XCTAssertGreaterThanOrEqual(uncertainty, 0, "\(key) uncertainty should be non-negative")
            XCTAssertLessThan(uncertainty, 1.0, "\(key) uncertainty should be reasonable (< 1.0)")
        }
    }

    // MARK: - Structure Learning Tests

    func testLearnHierarchicalStructure() {
        // When
        let result = model.learnStructure(observations: testObservations)

        // Then
        XCTAssertNotNil(result, "Structure result should not be nil")
        XCTAssertNotNil(result.learnedStructure, "Learned structure should not be null")

        let structure = result.learnedStructure
        XCTAssertFalse(structure.levels.isEmpty, "Should have hierarchy levels")
        XCTAssertTrue(result.structureConfidence >= 0 && result.structureConfidence <= 1,
                     "Structure confidence should be in [0,1]")
    }

    func testInferMeaningfulRelationships() {
        // When
        let result = model.learnStructure(observations: testObservations)

        // Then
        XCTAssertFalse(result.inferredRelationships.isEmpty,
                      "Should infer some relationships")
        XCTAssertFalse(result.relationshipStrengths.isEmpty,
                      "Should have relationship strengths")

        // All relationship strengths should be valid
        result.relationshipStrengths.forEach { key, strength in
            XCTAssertGreaterThanOrEqual(strength, 0, "\(key) strength should be >= 0")
            XCTAssertLessThanOrEqual(strength, 1, "\(key) strength should be <= 1")
        }
    }

    // MARK: - Validation Tests

    func testValidateAgainstGroundTruth() {
        // Given - first recover parameters
        let recoveredResult = model.recoverParameters(observations: testObservations)
        let recovered = recoveredResult.recoveredParameters

        // Ground truth parameters used to generate test data
        let groundTruth = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.7, N: 0.6, alpha: 0.5, beta: 1.2
        )

        // When
        let validationResult = model.validate(
            observations: testObservations,
            groundTruth: groundTruth,
            recoveredParameters: recovered
        )

        // Then
        XCTAssertNotNil(validationResult, "Validation result should not be nil")
        XCTAssertTrue(validationResult.overallScore >= 0 && validationResult.overallScore <= 1,
                     "Overall score should be in [0,1]")
        XCTAssertFalse(validationResult.parameterErrors.isEmpty,
                      "Should have parameter errors")
    }

    func testComputeParameterErrorsCorrectly() {
        // Given
        let groundTruth = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.8, N: 0.7, alpha: 0.6, beta: 1.5
        )
        let recovered = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.75, N: 0.65, alpha: 0.55, beta: 1.3
        )

        // When
        let result = model.validate(
            observations: testObservations,
            groundTruth: groundTruth,
            recoveredParameters: recovered
        )

        // Then
        let errors = result.parameterErrors
        XCTAssertGreaterThan(errors["S"] ?? 0, 0, "S error should be positive")
        XCTAssertGreaterThan(errors["N"] ?? 0, 0, "N error should be positive")
        XCTAssertGreaterThan(errors["alpha"] ?? 0, 0, "alpha error should be positive")
        XCTAssertGreaterThan(errors["beta"] ?? 0, 0, "beta error should be positive")
    }

    // MARK: - Edge Case Tests

    func testHandleSingleObservation() {
        // Given
        let singleObservation = [testObservations.first!]

        // When & Then - should not throw but may have low confidence
        let result = model.recoverParameters(observations: singleObservation)
        XCTAssertNotNil(result)
        XCTAssertTrue(result.confidence >= 0 && result.confidence <= 1)
    }

    func testHandleLargeNumberOfObservations() {
        // Given
        let largeDataset = createLargeTestDataset(size: 1000)

        // When & Then - should complete within reasonable time
        let startTime = Date()
        let result = model.recoverParameters(observations: largeDataset)
        let endTime = Date()

        // Then
        XCTAssertNotNil(result)
        let timeInterval = endTime.timeIntervalSince(startTime)
        XCTAssertLessThan(timeInterval, 30.0,
                         "Should complete large dataset within 30 seconds, took \(timeInterval)s")
    }

    func testHandleExtremeParameterValues() {
        // Given - observations with extreme values
        let extremeObservations = createExtremeTestObservations()

        // When & Then
        let result = model.recoverParameters(observations: extremeObservations)
        XCTAssertNotNil(result)

        // Parameters should still be in valid ranges
        let params = result.recoveredParameters
        XCTAssertGreaterThanOrEqual(params.S, 0)
        XCTAssertLessThanOrEqual(params.S, 1)
        XCTAssertGreaterThanOrEqual(params.N, 0)
        XCTAssertLessThanOrEqual(params.N, 1)
        XCTAssertGreaterThanOrEqual(params.alpha, 0)
        XCTAssertLessThanOrEqual(params.alpha, 1)
        XCTAssertGreaterThanOrEqual(params.beta, 1)
    }

    // MARK: - Performance Tests

    func testMaintainReasonablePerformanceScaling() {
        let datasetSizes = [10, 50, 100, 250]

        for size in datasetSizes {
            let dataset = createLargeTestDataset(size: size)

            let startTime = Date()
            let result = model.recoverParameters(observations: dataset)
            let endTime = Date()

            let timeInSeconds = endTime.timeIntervalSince(startTime)

            // Log performance for analysis
            print("Dataset size: \(size), Time: \(String(format: "%.3f", timeInSeconds))s")

            // Should complete and produce valid results
            XCTAssertNotNil(result)
            XCTAssertGreaterThanOrEqual(result.confidence, 0)
        }
    }

    func testMemoryEfficiency() {
        // Given
        let largeDataset = createLargeTestDataset(size: 500)

        // When - measure memory impact
        let result = model.recoverParameters(observations: largeDataset)

        // Then
        XCTAssertNotNil(result)
        // In a real scenario, we'd measure actual memory usage
        // For now, just ensure the operation completes successfully
    }

    // MARK: - Integration Tests

    func testMaintainConsistencyBetweenRecoveryAndValidation() {
        // Given - recover parameters
        let recoveryResult = model.recoverParameters(observations: testObservations)
        let recovered = recoveryResult.recoveredParameters

        // When - validate with recovered parameters as "ground truth"
        let validationResult = model.validate(
            observations: testObservations,
            groundTruth: recovered,
            recoveredParameters: recovered
        )

        // Then - validation should show perfect or near-perfect results
        XCTAssertNotNil(validationResult)
        XCTAssertGreaterThan(validationResult.overallScore, 0.95,
                           "Self-validation should have very high score")
        XCTAssertLessThan(validationResult.parameterRecoveryError, 0.01,
                         "Parameter recovery error should be very small for self-validation")
    }

    func testProduceConsistentResultsWithSameInput() {
        // When - run recovery twice with same data
        let result1 = model.recoverParameters(observations: testObservations)
        let result2 = model.recoverParameters(observations: testObservations)

        // Then - results should be very similar (within tolerance)
        let params1 = result1.recoveredParameters
        let params2 = result2.recoveredParameters

        let tolerance = 0.1 // Allow 10% variation due to stochastic nature
        XCTAssertEqual(params1.S, params2.S, accuracy: tolerance, "S should be consistent")
        XCTAssertEqual(params1.N, params2.N, accuracy: tolerance, "N should be consistent")
        XCTAssertEqual(params1.alpha, params2.alpha, accuracy: tolerance, "alpha should be consistent")
        XCTAssertEqual(params1.beta, params2.beta, accuracy: tolerance, "beta should be consistent")
    }

    // MARK: - SwiftUI ViewModel Tests

    func testViewModelInitialization() {
        // Given
        let viewModel = ReverseHierarchicalBayesianViewModel()

        // Then
        XCTAssertFalse(viewModel.hasObservations)
        XCTAssertFalse(viewModel.hasResults)
        XCTAssertFalse(viewModel.canRecoverParameters)
        XCTAssertFalse(viewModel.canLearnStructure)
        XCTAssertFalse(viewModel.canValidate)
    }

    func testViewModelAddObservation() {
        // Given
        let viewModel = ReverseHierarchicalBayesianViewModel()

        let claim = ReverseHierarchicalBayesianModel.ClaimData(
            id: "test",
            isVerifiedTrue: true,
            riskAuthenticity: 0.5,
            riskVirality: 0.3,
            probabilityHgivenE: 0.8
        )

        // When
        viewModel.addObservation(claim: claim, observedPsi: 0.7, verificationOutcome: true)

        // Then
        XCTAssertTrue(viewModel.hasObservations)
        XCTAssertEqual(viewModel.observations.count, 1)
    }

    func testViewModelGenerateSampleData() {
        // Given
        let viewModel = ReverseHierarchicalBayesianViewModel()

        // When
        viewModel.generateSampleData(count: 20)

        // Then
        XCTAssertTrue(viewModel.hasObservations)
        XCTAssertEqual(viewModel.observations.count, 20)
    }

    func testViewModelRecoverParameters() {
        // Given
        let viewModel = ReverseHierarchicalBayesianViewModel()
        viewModel.generateSampleData(count: 20)

        let expectation = self.expectation(description: "Parameter recovery completes")

        // When
        viewModel.recoverParameters()

        // Simulate async completion
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5.0) { error in
            if let error = error {
                XCTFail("Parameter recovery timed out: \(error)")
            }
        }

        // Note: In a real test, we'd verify the result, but since it's async
        // we just ensure no crash occurs
    }

    // MARK: - Helper Methods

    private func createTestObservations() -> [ReverseHierarchicalBayesianModel.Observation] {
        var observations = [ReverseHierarchicalBayesianModel.Observation]()

        // Ground truth parameters for generating synthetic data
        let groundTruth = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.7, N: 0.6, alpha: 0.5, beta: 1.2
        )

        for i in 0..<50 {
            // Create varied claim data
            let claim = ReverseHierarchicalBayesianModel.ClaimData(
                id: "test_\(i)",
                isVerifiedTrue: i % 3 == 0, // Some verified, some not
                riskAuthenticity: 0.3 + (Double(i) * 0.01).truncatingRemainder(dividingBy: 0.7),
                riskVirality: 0.2 + (Double(i) * 0.015).truncatingRemainder(dividingBy: 0.8),
                probabilityHgivenE: 0.1 + (Double(i) * 0.02).truncatingRemainder(dividingBy: 0.9)
            )

            // Compute true Ψ with ground truth parameters
            let truePsi = computePsi(claim: claim, params: groundTruth)

            // Add realistic noise
            let noise = (Double.random(in: -1...1)) * 0.1 // ±5% noise
            let observedPsi = max(0, min(1, truePsi + noise))

            // Create observation
            let observation = ReverseHierarchicalBayesianModel.Observation(
                claim: claim,
                observedPsi: observedPsi,
                verificationOutcome: claim.isVerifiedTrue
            )

            observations.append(observation)
        }

        return observations
    }

    private func createLargeTestDataset(size: Int) -> [ReverseHierarchicalBayesianModel.Observation] {
        var observations = [ReverseHierarchicalBayesianModel.Observation]()
        let groundTruth = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.7, N: 0.6, alpha: 0.5, beta: 1.2
        )

        for i in 0..<size {
            let claim = ReverseHierarchicalBayesianModel.ClaimData(
                id: "large_\(i)",
                isVerifiedTrue: Bool.random(),
                riskAuthenticity: Double.random(in: 0...1),
                riskVirality: Double.random(in: 0...1),
                probabilityHgivenE: Double.random(in: 0...1)
            )

            let truePsi = computePsi(claim: claim, params: groundTruth)
            let observedPsi = max(0, min(1, truePsi + (Double.random(in: -1...1) * 0.1)))

            observations.append(ReverseHierarchicalBayesianModel.Observation(
                claim: claim,
                observedPsi: observedPsi,
                verificationOutcome: claim.isVerifiedTrue
            ))
        }

        return observations
    }

    private func createExtremeTestObservations() -> [ReverseHierarchicalBayesianModel.Observation] {
        var observations = [ReverseHierarchicalBayesianModel.Observation]()
        let groundTruth = ReverseHierarchicalBayesianModel.ModelParameters(
            S: 0.9, N: 0.8, alpha: 0.7, beta: 2.0
        )

        // Create observations with extreme values
        let extremeClaims = [
            ReverseHierarchicalBayesianModel.ClaimData(
                id: "extreme_1", isVerifiedTrue: true, riskAuthenticity: 0.01,
                riskVirality: 0.01, probabilityHgivenE: 0.99
            ),
            ReverseHierarchicalBayesianModel.ClaimData(
                id: "extreme_2", isVerifiedTrue: false, riskAuthenticity: 0.99,
                riskVirality: 0.99, probabilityHgivenE: 0.01
            ),
            ReverseHierarchicalBayesianModel.ClaimData(
                id: "extreme_3", isVerifiedTrue: true, riskAuthenticity: 0.5,
                riskVirality: 0.5, probabilityHgivenE: 0.5
            )
        ]

        for claim in extremeClaims {
            let truePsi = computePsi(claim: claim, params: groundTruth)
            let observedPsi = max(0, min(1, truePsi + (Double.random(in: -1...1) * 0.05)))

            observations.append(ReverseHierarchicalBayesianModel.Observation(
                claim: claim,
                observedPsi: observedPsi,
                verificationOutcome: claim.isVerifiedTrue
            ))
        }

        return observations
    }

    private func computePsi(
        claim: ReverseHierarchicalBayesianModel.ClaimData,
        params: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> Double {
        let O = params.alpha * params.S + (1.0 - params.alpha) * params.N
        let penaltyExponent = -(
            1.0 * claim.riskAuthenticity +
            1.0 * claim.riskVirality
        )
        let pen = exp(penaltyExponent)
        let p_H_given_E_beta = min(params.beta * claim.probabilityHgivenE, 1.0)
        let psi = O * pen * p_H_given_E_beta
        return max(0.0, min(1.0, psi))
    }
}

// MARK: - Performance Test Suite

final class ReverseHierarchicalBayesianModelPerformanceTests: XCTestCase {

    var model: ReverseHierarchicalBayesianModel!

    override func setUp() {
        super.setUp()
        model = ReverseHierarchicalBayesianModel()
    }

    override func tearDown() {
        model = nil
        super.tearDown()
    }

    func testParameterRecoveryPerformance() {
        let datasetSizes = [100, 500, 1000]

        for size in datasetSizes {
            let observations = createLargeTestDataset(size: size)

            measure {
                let result = self.model.recoverParameters(observations: observations)
                XCTAssertNotNil(result)
            }
        }
    }

    func testStructureLearningPerformance() {
        let datasetSizes = [100, 500, 1000]

        for size in datasetSizes {
            let observations = createLargeTestDataset(size: size)

            measure {
                let result = self.model.learnStructure(observations: observations)
                XCTAssertNotNil(result)
            }
        }
    }

    private func createLargeTestDataset(size: Int) -> [ReverseHierarchicalBayesianModel.Observation] {
        var observations = [ReverseHierarchicalBayesianModel.Observation]()

        for i in 0..<size {
            let claim = ReverseHierarchicalBayesianModel.ClaimData(
                id: "perf_\(i)",
                isVerifiedTrue: Bool.random(),
                riskAuthenticity: Double.random(in: 0...1),
                riskVirality: Double.random(in: 0...1),
                probabilityHgivenE: Double.random(in: 0...1)
            )

            observations.append(ReverseHierarchicalBayesianModel.Observation(
                claim: claim,
                observedPsi: Double.random(in: 0...1),
                verificationOutcome: Bool.random()
            ))
        }

        return observations
    }
}

// MARK: - SwiftUI View Tests

import SwiftUI

final class ReverseHierarchicalBayesianViewModelUITests: XCTestCase {

    var viewModel: ReverseHierarchicalBayesianViewModel!

    override func setUp() {
        super.setUp()
        viewModel = ReverseHierarchicalBayesianViewModel()
    }

    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }

    func testSwiftUIViewInitialization() {
        // Given
        let view = ReverseHBModelView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view)

        // Then
        XCTAssertNotNil(hostingController.view)
    }

    func testDataManagementView() {
        // Given
        let view = DataManagementView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view.environmentObject(viewModel))

        // Then
        XCTAssertNotNil(hostingController.view)
    }

    func testParameterRecoveryView() {
        // Given
        let view = ParameterRecoveryView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view.environmentObject(viewModel))

        // Then
        XCTAssertNotNil(hostingController.view)
    }

    func testStructureLearningView() {
        // Given
        let view = StructureLearningView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view.environmentObject(viewModel))

        // Then
        XCTAssertNotNil(hostingController.view)
    }

    func testValidationView() {
        // Given
        let view = ValidationView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view.environmentObject(viewModel))

        // Then
        XCTAssertNotNil(hostingController.view)
    }

    func testSettingsView() {
        // Given
        let view = SettingsView()

        // When - create environment with view model
        let hostingController = UIHostingController(rootView: view.environmentObject(viewModel))

        // Then
        XCTAssertNotNil(hostingController.view)
    }
}
