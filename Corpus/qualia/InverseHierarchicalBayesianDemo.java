// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Demonstration and testing of the Inverse Hierarchical Bayesian Model.
 *
 * <p>This class shows how to use the inverse HB model to recover parameters
 * from observed Ψ scores and perform structure learning.
 */
public final class InverseHierarchicalBayesianDemo {

    public static void main(String[] args) {
        System.out.println("=== Inverse Hierarchical Bayesian Model Demo ===\n");

        // Create the inverse model
        InverseHierarchicalBayesianModel inverseModel = new InverseHierarchicalBayesianModel();

        // Generate synthetic observations for demonstration
        List<InverseHierarchicalBayesianModel.Observation> observations = generateSyntheticObservations();

        System.out.println("Generated " + observations.size() + " synthetic observations");
        System.out.println("Sample observations:");
        for (int i = 0; i < Math.min(5, observations.size()); i++) {
            var obs = observations.get(i);
            System.out.printf("  Claim %s: Ψ=%.3f, Verified=%s%n",
                obs.claim.id(), obs.observedPsi, obs.verificationOutcome);
        }
        System.out.println();

        // Perform parameter recovery
        System.out.println("=== Parameter Recovery ===");
        long startTime = System.nanoTime();

        InverseHierarchicalBayesianModel.InverseResult result = inverseModel.recoverParameters(observations);

        long endTime = System.nanoTime();
        double duration = (endTime - startTime) / 1e9;

        System.out.println("Recovery completed in " + String.format("%.2f", duration) + " seconds");
        System.out.println("Recovered parameters:");
        System.out.printf("  S = %.4f%n", result.recoveredParameters.S());
        System.out.printf("  N = %.4f%n", result.recoveredParameters.N());
        System.out.printf("  α = %.4f%n", result.recoveredParameters.alpha());
        System.out.printf("  β = %.4f%n", result.recoveredParameters.beta());
        System.out.printf("Recovery confidence: %.3f%n", result.confidence);
        System.out.printf("Log evidence: %.3f%n", result.logEvidence);

        System.out.println("\nParameter uncertainties:");
        for (Map.Entry<String, Double> entry : result.parameterUncertainties.entrySet()) {
            System.out.printf("  %s: ±%.4f%n", entry.getKey(), entry.getValue());
        }
        System.out.println();

        // Perform structure learning
        System.out.println("=== Structure Learning ===");
        InverseHierarchicalBayesianModel.StructureResult structureResult = inverseModel.learnStructure(observations);

        System.out.println("Learned hierarchical levels:");
        for (String level : structureResult.learnedStructure.levels) {
            System.out.println("  " + level);
        }

        System.out.println("\nInferred relationships:");
        for (Map.Entry<String, List<String>> entry : structureResult.learnedStructure.relationships.entrySet()) {
            System.out.println("  " + entry.getKey() + " → " + entry.getValue());
        }

        System.out.println("\nLevel weights:");
        for (Map.Entry<String, Double> entry : structureResult.learnedStructure.levelWeights.entrySet()) {
            System.out.printf("  %s: %.3f%n", entry.getKey(), entry.getValue());
        }

        System.out.printf("Structure confidence: %.3f%n", structureResult.structureConfidence);
        System.out.println();

        // Demonstrate validation (using synthetic ground truth)
        System.out.println("=== Validation ===");
        ModelParameters groundTruth = new ModelParameters(0.8, 0.6, 0.7, 1.5);
        InverseHierarchicalBayesianModel.ValidationResult validation =
            inverseModel.validate(observations, groundTruth, result.recoveredParameters);

        System.out.println("Validation against ground truth:");
        System.out.printf("  Overall score: %.3f%n", validation.overallScore);
        System.out.printf("  Parameter recovery error: %.4f%n", validation.parameterRecoveryError);
        System.out.printf("  Confidence accuracy: %.3f%n", validation.confidenceAccuracy);

        System.out.println("\nParameter-specific errors:");
        for (Map.Entry<String, Double> entry : validation.parameterErrors.entrySet()) {
            System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
        }

        System.out.println("\n=== Demo Complete ===");
    }

    /**
     * Generate synthetic observations for demonstration purposes.
     * In practice, these would come from real data or forward model outputs.
     */
    private static List<InverseHierarchicalBayesianModel.Observation> generateSyntheticObservations() {
        List<InverseHierarchicalBayesianModel.Observation> observations = new ArrayList<>();

        // Ground truth parameters for generating synthetic data
        ModelParameters trueParams = new ModelParameters(0.8, 0.6, 0.7, 1.5);

        // Generate 100 synthetic observations
        for (int i = 0; i < 100; i++) {
            // Create synthetic claim data
            String id = "claim_" + i;
            double riskAuth = Math.random() * 2.0; // 0-2 range
            double riskVirality = Math.random() * 1.5; // 0-1.5 range
            double probHgivenE = Math.random(); // 0-1 range

            ClaimData claim = new ClaimData(id, Math.random() < 0.7, riskAuth, riskVirality, probHgivenE);

            // Compute true Ψ using forward model
            double truePsi = computePsi(claim, trueParams);

            // Add observation noise
            double noise = (Math.random() - 0.5) * 0.1; // ±5% noise
            double observedPsi = Math.max(0.0, Math.min(1.0, truePsi + noise));

            // Generate verification outcome based on true Ψ
            boolean verificationOutcome = Math.random() < truePsi;

            observations.add(new InverseHierarchicalBayesianModel.Observation(
                claim, observedPsi, verificationOutcome));
        }

        return observations;
    }

    /**
     * Compute Ψ using the forward model (copied from HierarchicalBayesianModel for demo)
     */
    private static double computePsi(ClaimData claim, ModelParameters params) {
        double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        double penaltyExponent = -(
                1.0 * claim.riskAuthenticity() +  // lambda1 = 1.0
                1.0 * claim.riskVirality()        // lambda2 = 1.0
        );
        double pen = Math.exp(penaltyExponent);
        double p_H_given_E_beta = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);
        double psi = O * pen * p_H_given_E_beta;
        return Math.max(0.0, Math.min(1.0, psi));
    }
}
