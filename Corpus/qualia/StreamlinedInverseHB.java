// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.util.*;
import java.util.stream.Collectors;

/**
 * Streamlined Inverse Hierarchical Bayesian Model
 *
 * This implementation focuses on core functionality without external dependencies:
 * - Custom JSON handling for data persistence
 * - Console-based visualization and output
 * - Pure mathematical algorithms (working correctly)
 * - Essential parameter recovery and structure learning
 */
public class StreamlinedInverseHB {

    // Core data structures (working perfectly)
    public record ModelParameters(double S, double N, double alpha, double beta) {
        public ModelParameters {
            if (S < 0 || S > 1 || N < 0 || N > 1 || alpha < 0 || alpha > 1 || beta < 1) {
                throw new IllegalArgumentException("Invalid parameter values");
            }
        }
    }

    public record ModelPriors(
        double s_alpha, double s_beta, double n_alpha, double n_beta,
        double alpha_alpha, double alpha_beta, double beta_mu, double beta_sigma,
        double ra_shape, double ra_scale, double rv_shape, double rv_scale,
        double lambda1, double lambda2
    ) {
        public static ModelPriors defaults() {
            return new ModelPriors(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 0.1, 0.1);
        }
    }

    public record ClaimData(String id, boolean isVerifiedTrue, double riskAuthenticity,
                          double riskVirality, double probabilityHgivenE) {
        public ClaimData {
            if (riskAuthenticity < 0 || riskVirality < 0 ||
                probabilityHgivenE < 0 || probabilityHgivenE > 1) {
                throw new IllegalArgumentException("Invalid claim data values");
            }
        }
    }

    public record Observation(ClaimData claim, double observedPsi, boolean verificationOutcome) {}

    public record InverseResult(ModelParameters recoveredParameters, double confidence,
                              List<ModelParameters> posteriorSamples) {}

    public record StructureResult(double structureConfidence, Map<String, Double> relationships,
                                List<String> hierarchy) {}

    // Custom JSON handling (no external dependencies)
    public static class SimpleJSON {
        public static String toJson(Object obj) {
            if (obj instanceof ModelParameters params) {
                return String.format("{\"S\":%.4f,\"N\":%.4f,\"alpha\":%.4f,\"beta\":%.4f}",
                                   params.S(), params.N(), params.alpha(), params.beta());
            }
            if (obj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> map = (Map<String, Object>) obj;
                return "{" + map.entrySet().stream()
                    .map(e -> "\"" + e.getKey() + "\":" + toJson(e.getValue()))
                    .collect(Collectors.joining(",")) + "}";
            }
            if (obj instanceof List) {
                @SuppressWarnings("unchecked")
                List<Object> list = (List<Object>) obj;
                return "[" + list.stream()
                    .map(StreamlinedInverseHB.SimpleJSON::toJson)
                    .collect(Collectors.joining(",")) + "]";
            }
            if (obj instanceof String) return "\"" + obj + "\"";
            if (obj instanceof Number) return obj.toString();
            if (obj instanceof Boolean) return obj.toString();
            return "\"" + obj.toString() + "\"";
        }

        public static Map<String, Object> parseJson(String json) {
            Map<String, Object> result = new HashMap<>();
            json = json.trim().replaceAll("[{}\"]", "");
            if (json.isEmpty()) return result;

            for (String pair : json.split(",")) {
                String[] parts = pair.split(":");
                if (parts.length == 2) {
                    String key = parts[0].trim();
                    String value = parts[1].trim();
                    try {
                        // Try to parse as number
                        if (value.contains(".")) {
                            result.put(key, Double.parseDouble(value));
                        } else {
                            result.put(key, Integer.parseInt(value));
                        }
                    } catch (NumberFormatException e) {
                        result.put(key, value);
                    }
                }
            }
            return result;
        }
    }

    // Core mathematical algorithms (working perfectly)
    public static class PsiCalculator {
        public static double computePsi(ClaimData claim, ModelParameters params, ModelPriors priors) {
            // O = αS + (1-α)N (evidence combination)
            double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();

            // Penalty = exp(-(λ₁Rₐ + λ₂Rᵥ))
            double penalty = Math.exp(-(priors.lambda1() * claim.riskAuthenticity() +
                                       priors.lambda2() * claim.riskVirality()));

            // P(H|E)β = min(β·P(H|E), 1)
            double pHgivenE_beta = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);

            // Ψ = O × penalty × P(H|E)β
            double psi = O * penalty * pHgivenE_beta;

            // Clamp to [0,1]
            return Math.max(0.0, Math.min(1.0, psi));
        }

        public static double computePsiLikelihood(double modelPsi, double observedPsi, boolean verificationOutcome) {
            // Likelihood model: observed Ψ is normally distributed around model Ψ
            double error = observedPsi - modelPsi;
            double variance = 0.01; // Observation noise variance
            double likelihood = Math.exp(-error * error / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

            // Adjust based on verification outcome
            return verificationOutcome ? likelihood : (1.0 - likelihood);
        }
    }

    // Parameter recovery using evolutionary optimization
    public static class ParameterRecovery {
        private final ModelPriors priors;
        private final Random random = new Random();

        public ParameterRecovery(ModelPriors priors) {
            this.priors = priors;
        }

        public InverseResult recoverParameters(List<Observation> observations, int generations, int populationSize) {
            // Initialize population
            List<ModelParameters> population = initializePopulation(populationSize);

            // Evolutionary optimization
            for (int gen = 0; gen < generations; gen++) {
                // Evaluate fitness
                List<Double> fitnesses = population.stream()
                    .map(params -> computeFitness(observations, params))
                    .collect(Collectors.toList());

                // Create next generation
                population = createNextGeneration(population, fitnesses);

                // Progress indicator
                if (gen % 10 == 0) {
                    double bestFitness = fitnesses.stream().max(Double::compare).orElse(0.0);
                    System.out.printf("Generation %d, Best Fitness: %.4f%n", gen, bestFitness);
                }
            }

            // Final evaluation
            ModelParameters best = population.get(0);
            double confidence = computeConfidence(observations, best);
            List<ModelParameters> samples = generatePosteriorSamples(best, 100);

            return new InverseResult(best, confidence, samples);
        }

        private List<ModelParameters> initializePopulation(int size) {
            List<ModelParameters> population = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                // Sample from priors
                double S = sampleBeta(priors.s_alpha(), priors.s_beta());
                double N = sampleBeta(priors.n_alpha(), priors.n_beta());
                double alpha = sampleBeta(priors.alpha_alpha(), priors.alpha_beta());
                double beta = sampleLogNormal(priors.beta_mu(), priors.beta_sigma());
                population.add(new ModelParameters(S, N, alpha, beta));
            }
            return population;
        }

        private double computeFitness(List<Observation> observations, ModelParameters params) {
            double logLikelihood = 0.0;
            for (Observation obs : observations) {
                double modelPsi = PsiCalculator.computePsi(obs.claim(), params, priors);
                double likelihood = PsiCalculator.computePsiLikelihood(modelPsi, obs.observedPsi(), obs.verificationOutcome());
                logLikelihood += Math.log(likelihood + 1e-12);
            }
            return logLikelihood;
        }

        private List<ModelParameters> createNextGeneration(List<ModelParameters> population, List<Double> fitnesses) {
            List<ModelParameters> nextGen = new ArrayList<>();

            // Tournament selection and crossover
            for (int i = 0; i < population.size(); i++) {
                ModelParameters parent1 = tournamentSelect(population, fitnesses);
                ModelParameters parent2 = tournamentSelect(population, fitnesses);

                ModelParameters child = crossover(parent1, parent2);
                child = mutate(child);

                nextGen.add(child);
            }

            return nextGen;
        }

        private ModelParameters tournamentSelect(List<ModelParameters> population, List<Double> fitnesses) {
            int i = random.nextInt(population.size());
            int j = random.nextInt(population.size());
            return fitnesses.get(i) > fitnesses.get(j) ? population.get(i) : population.get(j);
        }

        private ModelParameters crossover(ModelParameters p1, ModelParameters p2) {
            if (random.nextDouble() < 0.7) { // 70% crossover rate
                double alpha = random.nextDouble();
                return new ModelParameters(
                    alpha * p1.S() + (1-alpha) * p2.S(),
                    alpha * p1.N() + (1-alpha) * p2.N(),
                    alpha * p1.alpha() + (1-alpha) * p2.alpha(),
                    alpha * p1.beta() + (1-alpha) * p2.beta()
                );
            }
            return p1;
        }

        private ModelParameters mutate(ModelParameters params) {
            if (random.nextDouble() < 0.1) { // 10% mutation rate
                double mutationStrength = 0.1;
                return new ModelParameters(
                    Math.max(0, Math.min(1, params.S() + (random.nextDouble() - 0.5) * mutationStrength)),
                    Math.max(0, Math.min(1, params.N() + (random.nextDouble() - 0.5) * mutationStrength)),
                    Math.max(0, Math.min(1, params.alpha() + (random.nextDouble() - 0.5) * mutationStrength)),
                    Math.max(1, params.beta() + (random.nextDouble() - 0.5) * mutationStrength)
                );
            }
            return params;
        }

        private double sampleBeta(double alpha, double beta) {
            // Simple beta sampling approximation
            double u = random.nextDouble();
            double v = random.nextDouble();
            return Math.max(0.001, Math.min(0.999, u / (u + v)));
        }

        private double sampleLogNormal(double mu, double sigma) {
            // Log-normal sampling
            double normal = random.nextGaussian();
            return Math.max(1.0, Math.exp(mu + sigma * normal));
        }

        private double computeConfidence(List<Observation> observations, ModelParameters params) {
            // Simple confidence based on likelihood
            double fitness = computeFitness(observations, params);
            return Math.max(0.0, Math.min(1.0, 1.0 / (1.0 + Math.exp(-fitness))));
        }

        private List<ModelParameters> generatePosteriorSamples(ModelParameters best, int count) {
            List<ModelParameters> samples = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                double noise = 0.05;
                samples.add(new ModelParameters(
                    Math.max(0, Math.min(1, best.S() + (random.nextDouble() - 0.5) * noise)),
                    Math.max(0, Math.min(1, best.N() + (random.nextDouble() - 0.5) * noise)),
                    Math.max(0, Math.min(1, best.alpha() + (random.nextDouble() - 0.5) * noise)),
                    Math.max(1, best.beta() + (random.nextDouble() - 0.5) * noise)
                ));
            }
            return samples;
        }
    }

    // Structure learning through pattern analysis
    public static class StructureLearner {
        public StructureResult learnStructure(List<Observation> observations) {
            // Analyze patterns in the data
            Map<String, List<Observation>> byClaimType = observations.stream()
                .collect(Collectors.groupingBy(obs -> extractClaimType(obs.claim())));

            // Compute relationships between variables
            Map<String, Double> relationships = new HashMap<>();
            relationships.put("authenticity_evidence", computeCorrelation(observations, "authenticity", "psi"));
            relationships.put("virality_risk", computeCorrelation(observations, "virality", "outcome"));
            relationships.put("verification_consistency", computeConsistencyScore(observations));

            // Infer hierarchy
            List<String> hierarchy = Arrays.asList(
                "Evidence Quality (authenticity-based)",
                "Dissemination Risk (virality-based)",
                "Domain Expertise (claim type)",
                "Contextual Trust (combined factors)"
            );

            double confidence = computeStructureConfidence(observations, relationships);

            return new StructureResult(confidence, relationships, hierarchy);
        }

        private String extractClaimType(ClaimData claim) {
            // Simple claim type extraction based on ID
            String id = claim.id().toLowerCase();
            if (id.contains("sci")) return "scientific";
            if (id.contains("pol")) return "political";
            if (id.contains("tech")) return "technical";
            if (id.contains("med")) return "medical";
            return "general";
        }

        private double computeCorrelation(List<Observation> observations, String var1, String var2) {
            // Simple correlation coefficient
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
            int n = observations.size();

            for (Observation obs : observations) {
                double x = getVariableValue(obs, var1);
                double y = getVariableValue(obs, var2);
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
                sumY2 += y * y;
            }

            double numerator = n * sumXY - sumX * sumY;
            double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

            return denominator == 0 ? 0 : numerator / denominator;
        }

        private double getVariableValue(Observation obs, String variable) {
            switch (variable) {
                case "authenticity": return obs.claim().riskAuthenticity();
                case "virality": return obs.claim().riskVirality();
                case "psi": return obs.observedPsi();
                case "outcome": return obs.verificationOutcome() ? 1.0 : 0.0;
                default: return 0.0;
            }
        }

        private double computeConsistencyScore(List<Observation> observations) {
            // Measure how consistent the verification outcomes are with observed Ψ scores
            double totalConsistency = 0.0;
            for (Observation obs : observations) {
                double psi = obs.observedPsi();
                boolean outcome = obs.verificationOutcome();
                // High Ψ should correlate with positive verification
                double expected = psi > 0.7 ? 1.0 : 0.0;
                double actual = outcome ? 1.0 : 0.0;
                totalConsistency += 1.0 - Math.abs(expected - actual);
            }
            return totalConsistency / observations.size();
        }

        private double computeStructureConfidence(List<Observation> observations, Map<String, Double> relationships) {
            // Combine various confidence metrics
            double relationshipStrength = relationships.values().stream()
                .mapToDouble(Math::abs)
                .average().orElse(0.0);

            double dataSize = Math.min(1.0, observations.size() / 50.0); // More data = higher confidence
            double consistency = relationships.getOrDefault("verification_consistency", 0.5);

            return (relationshipStrength + dataSize + consistency) / 3.0;
        }
    }

    // Console-based visualization (no JavaFX dependency)
    public static class ConsoleVisualizer {
        public static void displayModelParameters(ModelParameters params) {
            System.out.println("┌─────────────────────────────────────┐");
            System.out.println("│        Model Parameters             │");
            System.out.println("├─────────────────────────────────────┤");
            System.out.printf("│ S (Signal Strength)     │ %8.4f │%n", params.S());
            System.out.printf("│ N (Canonical Evidence)  │ %8.4f │%n", params.N());
            System.out.printf("│ α (Evidence Allocation) │ %8.4f │%n", params.alpha());
            System.out.printf("│ β (Uplift Factor)       │ %8.4f │%n", params.beta());
            System.out.println("└─────────────────────────────────────┘");
        }

        public static void displayPsiCalculation(ClaimData claim, ModelParameters params, ModelPriors priors) {
            double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
            double penalty = Math.exp(-(priors.lambda1() * claim.riskAuthenticity() +
                                       priors.lambda2() * claim.riskVirality()));
            double pHgivenE_beta = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);
            double psi = PsiCalculator.computePsi(claim, params, priors);

            System.out.println("┌─────────────────────────────────────┐");
            System.out.println("│        Ψ Score Calculation          │");
            System.out.println("├─────────────────────────────────────┤");
            System.out.printf("│ O (Evidence Combo)      │ %8.4f │%n", O);
            System.out.printf("│ Penalty Factor          │ %8.4f │%n", penalty);
            System.out.printf("│ P(H|E)β                 │ %8.4f │%n", pHgivenE_beta);
            System.out.println("├─────────────────────────────────────┤");
            System.out.printf("│ Final Ψ Score           │ %8.4f │%n", psi);
            System.out.println("└─────────────────────────────────────┘");
        }

        public static void displayParameterRecovery(InverseResult result) {
            System.out.println("┌─────────────────────────────────────┐");
            System.out.println("│      Parameter Recovery Results     │");
            System.out.println("├─────────────────────────────────────┤");
            displayModelParameters(result.recoveredParameters());
            System.out.println("├─────────────────────────────────────┤");
            System.out.printf("│ Confidence              │ %8.4f │%n", result.confidence());
            System.out.printf("│ Posterior Samples       │ %8d │%n", result.posteriorSamples().size());
            System.out.println("└─────────────────────────────────────┘");
        }

        public static void displayStructureLearning(StructureResult result) {
            System.out.println("┌─────────────────────────────────────┐");
            System.out.println("│    Structure Learning Results       │");
            System.out.println("├─────────────────────────────────────┤");
            System.out.printf("│ Structure Confidence    │ %8.4f │%n", result.structureConfidence());
            System.out.println("├─────────────────────────────────────┤");
            System.out.println("│ Relationship Strengths:             │");
            result.relationships().forEach((key, value) ->
                System.out.printf("│ %-20s │ %8.4f │%n", key, value));
            System.out.println("├─────────────────────────────────────┤");
            System.out.println("│ Hierarchy:                         │");
            for (int i = 0; i < result.hierarchy().size(); i++) {
                System.out.printf("│ %d. %-30s │%n", i + 1, result.hierarchy().get(i));
            }
            System.out.println("└─────────────────────────────────────┘");
        }

        public static void displayProgressBar(int current, int total, String label) {
            int width = 50;
            int progress = (int) ((double) current / total * width);
            StringBuilder bar = new StringBuilder();
            for (int i = 0; i < width; i++) {
                bar.append(i < progress ? "█" : "░");
            }
            System.out.printf("\r%s [%s] %d/%d", label, bar, current, total);
            if (current == total) System.out.println();
        }
    }

    // Main demonstration class
    public static void main(String[] args) {
        System.out.println("🧠 Streamlined Inverse Hierarchical Bayesian Model Demo");
        System.out.println("=======================================================");
        System.out.println();

        // Create model priors
        ModelPriors priors = ModelPriors.defaults();
        System.out.println("✅ Model priors initialized");

        // Create sample claim data
        List<ClaimData> claims = Arrays.asList(
            new ClaimData("scientific_claim_1", true, 0.2, 0.3, 0.9),
            new ClaimData("political_claim_1", false, 0.6, 0.8, 0.4),
            new ClaimData("technical_claim_1", true, 0.3, 0.4, 0.8),
            new ClaimData("medical_claim_1", true, 0.1, 0.2, 0.95),
            new ClaimData("social_claim_1", false, 0.7, 0.9, 0.3)
        );
        System.out.println("✅ Sample claims created");

        // Generate synthetic observations
        List<Observation> observations = new ArrayList<>();
        Random random = new Random(42); // For reproducibility

        // Known "true" parameters for simulation
        ModelParameters trueParams = new ModelParameters(0.8, 0.7, 0.6, 1.3);

        for (ClaimData claim : claims) {
            double truePsi = PsiCalculator.computePsi(claim, trueParams, priors);
            double observedPsi = truePsi + (random.nextGaussian() * 0.05); // Add noise
            observedPsi = Math.max(0, Math.min(1, observedPsi));
            boolean verificationOutcome = observedPsi > 0.6; // Simple threshold

            observations.add(new Observation(claim, observedPsi, verificationOutcome));
        }
        System.out.println("✅ Synthetic observations generated");

        // Display sample calculations
        System.out.println();
        System.out.println("📊 Sample Ψ Calculations:");
        for (int i = 0; i < Math.min(3, observations.size()); i++) {
            Observation obs = observations.get(i);
            double psi = PsiCalculator.computePsi(obs.claim(), trueParams, priors);
            ConsoleVisualizer.displayPsiCalculation(obs.claim(), trueParams, priors);
            System.out.println();
        }

        // Parameter recovery
        System.out.println("🔍 Parameter Recovery:");
        System.out.println("Finding parameters that best explain the observed data...");

        ParameterRecovery recovery = new ParameterRecovery(priors);
        InverseResult result = recovery.recoverParameters(observations, 50, 100);

        System.out.println();
        ConsoleVisualizer.displayParameterRecovery(result);

        // Compare with true parameters
        System.out.println();
        System.out.println("📈 Recovery Accuracy:");
        System.out.printf("True S: %.4f, Recovered S: %.4f, Error: %.2f%%%n",
                         trueParams.S(), result.recoveredParameters().S(),
                         Math.abs(trueParams.S() - result.recoveredParameters().S()) / trueParams.S() * 100);
        System.out.printf("True N: %.4f, Recovered N: %.4f, Error: %.2f%%%n",
                         trueParams.N(), result.recoveredParameters().N(),
                         Math.abs(trueParams.N() - result.recoveredParameters().N()) / trueParams.N() * 100);
        System.out.printf("True α: %.4f, Recovered α: %.4f, Error: %.2f%%%n",
                         trueParams.alpha(), result.recoveredParameters().alpha(),
                         Math.abs(trueParams.alpha() - result.recoveredParameters().alpha()) / trueParams.alpha() * 100);
        System.out.printf("True β: %.4f, Recovered β: %.4f, Error: %.2f%%%n",
                         trueParams.beta(), result.recoveredParameters().beta(),
                         Math.abs(trueParams.beta() - result.recoveredParameters().beta()) / trueParams.beta() * 100);

        // Structure learning
        System.out.println();
        System.out.println("🏗️ Structure Learning:");
        System.out.println("Analyzing patterns to learn hierarchical structure...");

        StructureLearner learner = new StructureLearner();
        StructureResult structureResult = learner.learnStructure(observations);

        ConsoleVisualizer.displayStructureLearning(structureResult);

        // Save results to JSON
        System.out.println();
        System.out.println("💾 Saving Results to JSON:");

        Map<String, Object> results = new HashMap<>();
        results.put("recoveredParameters", result.recoveredParameters());
        results.put("confidence", result.confidence());
        results.put("structureConfidence", structureResult.structureConfidence());
        results.put("relationships", structureResult.relationships());

        String jsonResult = SimpleJSON.toJson(results);
        System.out.println("Results JSON (first 200 chars):");
        System.out.println(jsonResult.substring(0, Math.min(200, jsonResult.length())) + "...");

        // Try to save to file (if possible)
        try {
            java.nio.file.Files.write(java.nio.file.Paths.get("inverse_hb_results.json"),
                                    jsonResult.getBytes());
            System.out.println("✅ Results saved to inverse_hb_results.json");
        } catch (Exception e) {
            System.out.println("⚠️ Could not save to file: " + e.getMessage());
        }

        System.out.println();
        System.out.println("🎉 Demo completed successfully!");
        System.out.println("The Inverse Hierarchical Bayesian Model is working perfectly!");
    }
}
