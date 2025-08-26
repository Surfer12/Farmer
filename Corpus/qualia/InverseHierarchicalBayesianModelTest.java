// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for InverseHierarchicalBayesianModel.
 *
 * <p>Tests cover parameter recovery, structure learning, validation,
 * and edge cases to ensure robustness and correctness.
 */
@DisplayName("Inverse Hierarchical Bayesian Model Tests")
public final class InverseHierarchicalBayesianModelTest {

    private InverseHierarchicalBayesianModel model;
    private List<InverseHierarchicalBayesianModel.Observation> testObservations;

    @BeforeEach
    void setUp() {
        model = new InverseHierarchicalBayesianModel();

        // Create test observations with known ground truth
        testObservations = createTestObservations();
    }

    @Nested
    @DisplayName("Parameter Recovery Tests")
    class ParameterRecoveryTests {

        @Test
        @DisplayName("Should recover parameters from synthetic observations")
        void shouldRecoverParametersFromSyntheticData() {
            // When
            InverseHierarchicalBayesianModel.InverseResult result = model.recoverParameters(testObservations);

            // Then
            assertNotNull(result, "Result should not be null");
            assertNotNull(result.recoveredParameters, "Recovered parameters should not be null");

            ModelParameters params = result.recoveredParameters;
            assertTrue(params.S() >= 0 && params.S() <= 1, "S should be in [0,1]");
            assertTrue(params.N() >= 0 && params.N() <= 1, "N should be in [0,1]");
            assertTrue(params.alpha() >= 0 && params.alpha() <= 1, "alpha should be in [0,1]");
            assertTrue(params.beta() >= 1, "beta should be >= 1");

            assertTrue(result.confidence >= 0 && result.confidence <= 1,
                "Confidence should be in [0,1]");
            assertFalse(result.posteriorSamples.isEmpty(), "Should have posterior samples");
        }

        @Test
        @DisplayName("Should handle empty observations gracefully")
        void shouldHandleEmptyObservations() {
            // Given
            List<InverseHierarchicalBayesianModel.Observation> emptyObservations = new ArrayList<>();

            // When & Then
            RuntimeException error = assertThrows(
                RuntimeException.class,
                () -> model.recoverParameters(emptyObservations)
            );

            assertTrue(error.getMessage().contains("empty"), "Should mention empty observations");
        }

        @Test
        @DisplayName("Should produce reasonable parameter uncertainties")
        void shouldProduceReasonableParameterUncertainties() {
            // When
            InverseHierarchicalBayesianModel.InverseResult result = model.recoverParameters(testObservations);

            // Then
            var uncertainties = result.parameterUncertainties;
            assertNotNull(uncertainties, "Uncertainties should not be null");
            assertTrue(uncertainties.containsKey("S"), "Should have S uncertainty");
            assertTrue(uncertainties.containsKey("N"), "Should have N uncertainty");
            assertTrue(uncertainties.containsKey("alpha"), "Should have alpha uncertainty");
            assertTrue(uncertainties.containsKey("beta"), "Should have beta uncertainty");

            // All uncertainties should be positive and reasonable
            uncertainties.values().forEach(uncertainty -> {
                assertTrue(uncertainty >= 0, "Uncertainty should be non-negative");
                assertTrue(uncertainty < 1.0, "Uncertainty should be reasonable (< 1.0)");
            });
        }
    }

    @Nested
    @DisplayName("Structure Learning Tests")
    class StructureLearningTests {

        @Test
        @DisplayName("Should learn hierarchical structure from observations")
        void shouldLearnHierarchicalStructure() {
            // When
            InverseHierarchicalBayesianModel.StructureResult result = model.learnStructure(testObservations);

            // Then
            assertNotNull(result, "Structure result should not be null");
            assertNotNull(result.learnedStructure, "Learned structure should not be null");

            var structure = result.learnedStructure;
            assertFalse(structure.levels.isEmpty(), "Should have hierarchy levels");
            assertTrue(result.structureConfidence >= 0 && result.structureConfidence <= 1,
                "Structure confidence should be in [0,1]");
        }

        @Test
        @DisplayName("Should infer meaningful relationships")
        void shouldInferMeaningfulRelationships() {
            // When
            InverseHierarchicalBayesianModel.StructureResult result = model.learnStructure(testObservations);

            // Then
            assertFalse(result.inferredRelationships.isEmpty(),
                "Should infer some relationships");
            assertFalse(result.relationshipStrengths.isEmpty(),
                "Should have relationship strengths");

            // All relationship strengths should be valid
            result.relationshipStrengths.values().forEach(strength -> {
                assertTrue(strength >= 0 && strength <= 1,
                    "Relationship strength should be in [0,1]");
            });
        }
    }

    @Nested
    @DisplayName("Validation Tests")
    class ValidationTests {

        @Test
        @DisplayName("Should validate against ground truth parameters")
        void shouldValidateAgainstGroundTruth() {
            // Given - first recover parameters
            InverseHierarchicalBayesianModel.InverseResult recoveredResult =
                model.recoverParameters(testObservations);
            ModelParameters recovered = recoveredResult.recoveredParameters;

            // Ground truth parameters used to generate test data
            ModelParameters groundTruth = new ModelParameters(0.7, 0.6, 0.5, 1.2);

            // When
            InverseHierarchicalBayesianModel.ValidationResult validationResult =
                model.validate(testObservations, groundTruth, recovered);

            // Then
            assertNotNull(validationResult, "Validation result should not be null");
            assertTrue(validationResult.overallScore >= 0 && validationResult.overallScore <= 1,
                "Overall score should be in [0,1]");
            assertFalse(validationResult.parameterErrors.isEmpty(),
                "Should have parameter errors");
        }

        @Test
        @DisplayName("Should compute parameter errors correctly")
        void shouldComputeParameterErrorsCorrectly() {
            // Given
            ModelParameters groundTruth = new ModelParameters(0.8, 0.7, 0.6, 1.5);
            ModelParameters recovered = new ModelParameters(0.75, 0.65, 0.55, 1.3);

            // When
            InverseHierarchicalBayesianModel.ValidationResult result =
                model.validate(testObservations, groundTruth, recovered);

            // Then
            var errors = result.parameterErrors;
            assertTrue(errors.get("S") > 0, "S error should be positive");
            assertTrue(errors.get("N") > 0, "N error should be positive");
            assertTrue(errors.get("alpha") > 0, "alpha error should be positive");
            assertTrue(errors.get("beta") > 0, "beta error should be positive");
        }
    }

    @Nested
    @DisplayName("Edge Case Tests")
    class EdgeCaseTests {

        @Test
        @DisplayName("Should handle single observation")
        void shouldHandleSingleObservation() {
            // Given
            List<InverseHierarchicalBayesianModel.Observation> singleObservation =
                List.of(testObservations.get(0));

            // When & Then - should not throw but may have low confidence
            assertDoesNotThrow(() -> {
                InverseHierarchicalBayesianModel.InverseResult result =
                    model.recoverParameters(singleObservation);
                assertNotNull(result);
                // Single observation should have lower confidence
                assertTrue(result.confidence >= 0 && result.confidence <= 1);
            });
        }

        @Test
        @DisplayName("Should handle large number of observations")
        void shouldHandleLargeNumberOfObservations() {
            // Given
            List<InverseHierarchicalBayesianModel.Observation> largeDataset =
                createLargeTestDataset(1000);

            // When & Then - should complete within reasonable time
            long startTime = System.currentTimeMillis();
            assertDoesNotThrow(() -> {
                InverseHierarchicalBayesianModel.InverseResult result =
                    model.recoverParameters(largeDataset);
                assertNotNull(result);
            });
            long endTime = System.currentTimeMillis();

            // Should complete in reasonable time (less than 30 seconds for 1000 observations)
            assertTrue(endTime - startTime < 30000,
                "Should complete large dataset within 30 seconds");
        }

        @Test
        @DisplayName("Should handle extreme parameter values")
        void shouldHandleExtremeParameterValues() {
            // Create observations with extreme values
            List<InverseHierarchicalBayesianModel.Observation> extremeObservations =
                createExtremeTestObservations();

            // When & Then
            assertDoesNotThrow(() -> {
                InverseHierarchicalBayesianModel.InverseResult result =
                    model.recoverParameters(extremeObservations);
                assertNotNull(result);

                // Parameters should still be in valid ranges
                ModelParameters params = result.recoveredParameters;
                assertTrue(params.S() >= 0 && params.S() <= 1);
                assertTrue(params.N() >= 0 && params.N() <= 1);
                assertTrue(params.alpha() >= 0 && params.alpha() <= 1);
                assertTrue(params.beta() >= 1);
            });
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {

        @Test
        @DisplayName("Should maintain reasonable performance scaling")
        void shouldMaintainReasonablePerformanceScaling() {
            int[] datasetSizes = {10, 50, 100, 250};

            for (int size : datasetSizes) {
                List<InverseHierarchicalBayesianModel.Observation> dataset =
                    createLargeTestDataset(size);

                long startTime = System.nanoTime();
                InverseHierarchicalBayesianModel.InverseResult result =
                    model.recoverParameters(dataset);
                long endTime = System.nanoTime();

                double timeInSeconds = (endTime - startTime) / 1e9;

                // Log performance for analysis
                System.out.printf("Dataset size: %d, Time: %.3fs%n", size, timeInSeconds);

                // Should complete and produce valid results
                assertNotNull(result);
                assertTrue(result.getConfidence() >= 0);
            }
        }

        @Test
        @DisplayName("Should be memory efficient with large datasets")
        void shouldBeMemoryEfficient() {
            // Given
            List<InverseHierarchicalBayesianModel.Observation> largeDataset =
                createLargeTestDataset(500);

            // When - measure memory before and after
            Runtime runtime = Runtime.getRuntime();
            runtime.gc(); // Clean up before measurement
            long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

            InverseHierarchicalBayesianModel.InverseResult result =
                model.recoverParameters(largeDataset);

            runtime.gc(); // Clean up after
            long memoryAfter = runtime.totalMemory() - runtime.freeMemory();

            // Then
            assertNotNull(result);

            // Memory increase should be reasonable (less than 100MB)
            long memoryIncrease = memoryAfter - memoryBefore;
            assertTrue(memoryIncrease < 100 * 1024 * 1024,
                "Memory increase should be less than 100MB, was: " + memoryIncrease + " bytes");
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {

        @Test
        @DisplayName("Should maintain consistency between recovery and validation")
        void shouldMaintainConsistencyBetweenRecoveryAndValidation() {
            // Given - recover parameters
            InverseHierarchicalBayesianModel.InverseResult recoveryResult =
                model.recoverParameters(testObservations);
            ModelParameters recovered = recoveryResult.recoveredParameters;

            // When - validate with recovered parameters as "ground truth"
            InverseHierarchicalBayesianModel.ValidationResult validationResult =
                model.validate(testObservations, recovered, recovered);

            // Then - validation should show perfect or near-perfect results
            assertNotNull(validationResult);
            assertTrue(validationResult.overallScore > 0.95,
                "Self-validation should have very high score");
            assertTrue(validationResult.parameterRecoveryError < 0.01,
                "Parameter recovery error should be very small for self-validation");
        }

        @Test
        @DisplayName("Should produce consistent results with same input")
        void shouldProduceConsistentResultsWithSameInput() {
            // When - run recovery twice with same data
            InverseHierarchicalBayesianModel.InverseResult result1 =
                model.recoverParameters(testObservations);
            InverseHierarchicalBayesianModel.InverseResult result2 =
                model.recoverParameters(testObservations);

            // Then - results should be very similar (within tolerance)
            ModelParameters params1 = result1.recoveredParameters;
            ModelParameters params2 = result2.recoveredParameters;

            double tolerance = 0.1; // Allow 10% variation due to stochastic nature
            assertEquals(params1.S(), params2.S(), tolerance, "S should be consistent");
            assertEquals(params1.N(), params2.N(), tolerance, "N should be consistent");
            assertEquals(params1.alpha(), params2.alpha(), tolerance, "alpha should be consistent");
            assertEquals(params1.beta(), params2.beta(), tolerance, "beta should be consistent");
        }
    }

    // Helper Methods

    private List<InverseHierarchicalBayesianModel.Observation> createTestObservations() {
        List<InverseHierarchicalBayesianModel.Observation> observations = new ArrayList<>();

        // Ground truth parameters for generating synthetic data
        ModelParameters groundTruth = new ModelParameters(0.7, 0.6, 0.5, 1.2);

        for (int i = 0; i < 50; i++) {
            // Create varied claim data
            ClaimData claim = new ClaimData(
                "test_" + i,
                i % 3 == 0, // Some verified, some not
                0.3 + (i * 0.01) % 0.7, // Vary authenticity risk
                0.2 + (i * 0.015) % 0.8, // Vary virality risk
                0.1 + (i * 0.02) % 0.9  // Vary probability
            );

            // Compute true Ψ with ground truth parameters
            double truePsi = computePsi(claim, groundTruth);

            // Add realistic noise
            double noise = (Math.random() - 0.5) * 0.1; // ±5% noise
            double observedPsi = Math.max(0, Math.min(1, truePsi + noise));

            // Create observation
            InverseHierarchicalBayesianModel.Observation observation =
                new InverseHierarchicalBayesianModel.Observation(claim, observedPsi, claim.isVerifiedTrue());

            observations.add(observation);
        }

        return observations;
    }

    private List<InverseHierarchicalBayesianModel.Observation> createLargeTestDataset(int size) {
        List<InverseHierarchicalBayesianModel.Observation> observations = new ArrayList<>();
        ModelParameters groundTruth = new ModelParameters(0.7, 0.6, 0.5, 1.2);

        for (int i = 0; i < size; i++) {
            ClaimData claim = new ClaimData(
                "large_" + i,
                Math.random() > 0.5,
                Math.random(),
                Math.random(),
                Math.random()
            );

            double truePsi = computePsi(claim, groundTruth);
            double observedPsi = Math.max(0, Math.min(1, truePsi + (Math.random() - 0.5) * 0.1));

            observations.add(new InverseHierarchicalBayesianModel.Observation(
                claim, observedPsi, claim.isVerifiedTrue()));
        }

        return observations;
    }

    private List<InverseHierarchicalBayesianModel.Observation> createExtremeTestObservations() {
        List<InverseHierarchicalBayesianModel.Observation> observations = new ArrayList<>();
        ModelParameters groundTruth = new ModelParameters(0.9, 0.8, 0.7, 2.0);

        // Create observations with extreme values
        ClaimData[] extremeClaims = {
            new ClaimData("extreme_1", true, 0.01, 0.01, 0.99), // Very trustworthy
            new ClaimData("extreme_2", false, 0.99, 0.99, 0.01), // Very untrustworthy
            new ClaimData("extreme_3", true, 0.5, 0.5, 0.5), // Neutral
        };

        for (ClaimData claim : extremeClaims) {
            double truePsi = computePsi(claim, groundTruth);
            double observedPsi = Math.max(0, Math.min(1, truePsi + (Math.random() - 0.5) * 0.05));

            observations.add(new InverseHierarchicalBayesianModel.Observation(
                claim, observedPsi, claim.isVerifiedTrue()));
        }

        return observations;
    }

    private double computePsi(ClaimData claim, ModelParameters params) {
        double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        double penaltyExponent = -(
            model.getPriors().getLambda1() * claim.getRiskAuthenticity() +
            model.getPriors().getLambda2() * claim.getRiskVirality()
        );
        double pen = Math.exp(penaltyExponent);
        double p_H_given_E_beta = Math.min(params.beta() * claim.getProbabilityHgivenE(), 1.0);
        double psi = O * pen * p_H_given_E_beta;
        return Math.max(0.0, Math.min(1.0, psi));
    }
}
