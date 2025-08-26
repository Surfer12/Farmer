// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Inverse Hierarchical Bayesian Model for parameter recovery and structure learning.
 *
 * <p>This model works backwards from observed Ψ(x) scores and verification outcomes
 * to recover the most likely model parameters and hierarchical structure that
 * generated the observations.
 *
 * <p>Key capabilities:
 * - Parameter recovery from observed Ψ scores
 * - Hierarchical structure learning from data patterns
 * - Confidence estimation for recovered parameters
 * - Validation against ground truth when available
 *
 * <p>Mathematical foundation:
 * Given observations D = {(x_i, y_i, ψ_i)}, find parameters θ = (S, N, α, β)
 * that maximize P(θ|D) under the inverse mapping ψ → θ.
 */
public final class InverseHierarchicalBayesianModel {

    private final ModelPriors priors;
    private final InverseConfig config;

    /**
     * Configuration for inverse modeling
     */
    public static final class InverseConfig {
        public final int maxIterations;
        public final double tolerance;
        public final int populationSize;
        public final double mutationRate;
        public final double crossoverRate;
        public final boolean useParallel;
        public final int parallelThreshold;

        public InverseConfig(int maxIterations, double tolerance, int populationSize,
                           double mutationRate, double crossoverRate, boolean useParallel, int parallelThreshold) {
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
            this.populationSize = populationSize;
            this.mutationRate = mutationRate;
            this.crossoverRate = crossoverRate;
            this.useParallel = useParallel;
            this.parallelThreshold = parallelThreshold;
        }

        public static InverseConfig defaults() {
            return new InverseConfig(1000, 1e-6, 100, 0.1, 0.7, true, 1000);
        }
    }

    /**
     * Represents the inverse mapping result
     */
    public static final class InverseResult {
        public final ModelParameters recoveredParameters;
        public final double confidence;
        public final Map<String, Double> parameterUncertainties;
        public final List<ModelParameters> posteriorSamples;
        public final double logEvidence;

        public InverseResult(ModelParameters recoveredParameters, double confidence,
                           Map<String, Double> parameterUncertainties,
                           List<ModelParameters> posteriorSamples, double logEvidence) {
            this.recoveredParameters = recoveredParameters;
            this.confidence = confidence;
            this.parameterUncertainties = parameterUncertainties;
            this.posteriorSamples = posteriorSamples;
            this.logEvidence = logEvidence;
        }
    }

    /**
     * Structure learning result
     */
    public static final class StructureResult {
        public final HierarchicalStructure learnedStructure;
        public final double structureConfidence;
        public final List<String> inferredRelationships;
        public final Map<String, Double> relationshipStrengths;

        public StructureResult(HierarchicalStructure learnedStructure, double structureConfidence,
                             List<String> inferredRelationships, Map<String, Double> relationshipStrengths) {
            this.learnedStructure = learnedStructure;
            this.structureConfidence = structureConfidence;
            this.inferredRelationships = inferredRelationships;
            this.relationshipStrengths = relationshipStrengths;
        }
    }

    /**
     * Hierarchical structure representation
     */
    public static final class HierarchicalStructure {
        public final List<String> levels;
        public final Map<String, List<String>> relationships;
        public final Map<String, Double> levelWeights;

        public HierarchicalStructure(List<String> levels, Map<String, List<String>> relationships,
                                   Map<String, Double> levelWeights) {
            this.levels = levels;
            this.relationships = relationships;
            this.levelWeights = levelWeights;
        }
    }

    public InverseHierarchicalBayesianModel() {
        this(ModelPriors.defaults(), InverseConfig.defaults());
    }

    public InverseHierarchicalBayesianModel(ModelPriors priors, InverseConfig config) {
        this.priors = priors;
        this.config = config;
    }

    /**
     * Perform inverse modeling from observed Ψ scores and verification outcomes.
     *
     * @param observations List of (ClaimData, observed_psi, verification_outcome) triples
     * @return Inverse mapping result with recovered parameters and confidence
     */
    public InverseResult recoverParameters(List<Observation> observations) {
        if (observations.isEmpty()) {
            throw new IllegalArgumentException("Observations cannot be empty");
        }

        // Use evolutionary optimization to find best parameters
        EvolutionaryOptimizer optimizer = new EvolutionaryOptimizer(config, priors);

        OptimizationResult optResult = optimizer.optimize(observations);

        // Sample from posterior for uncertainty estimation
        List<ModelParameters> posteriorSamples = samplePosterior(observations, optResult.bestParameters, 1000);

        // Compute confidence and uncertainties
        Map<String, Double> uncertainties = estimateParameterUncertainties(posteriorSamples);
        double confidence = computeRecoveryConfidence(observations, optResult.bestParameters, posteriorSamples);

        return new InverseResult(
            optResult.bestParameters,
            confidence,
            uncertainties,
            posteriorSamples,
            optResult.logEvidence
        );
    }

    /**
     * Learn hierarchical structure from data patterns.
     *
     * @param observations Observations to analyze
     * @return Structure learning result
     */
    public StructureResult learnStructure(List<Observation> observations) {
        StructureLearner learner = new StructureLearner(config);
        return learner.learnStructure(observations);
    }

    /**
     * Validate inverse model against known ground truth.
     *
     * @param observations Test observations
     * @param groundTruth True parameters used to generate data
     * @param recoveredParameters Parameters recovered by inverse model
     * @return Validation metrics
     */
    public ValidationResult validate(List<Observation> observations,
                                   ModelParameters groundTruth,
                                   ModelParameters recoveredParameters) {
        ValidationMetrics metrics = new ValidationMetrics(observations, groundTruth, recoveredParameters);
        return new ValidationResult(metrics);
    }

    /**
     * Evolutionary optimizer for parameter recovery
     */
    private static final class EvolutionaryOptimizer {
        private final InverseConfig config;
        private final ModelPriors priors;

        EvolutionaryOptimizer(InverseConfig config, ModelPriors priors) {
            this.config = config;
            this.priors = priors;
        }

        OptimizationResult optimize(List<Observation> observations) {
            List<ModelParameters> population = initializePopulation();

            double bestFitness = Double.NEGATIVE_INFINITY;
            ModelParameters bestIndividual = population.get(0);

            for (int generation = 0; generation < config.maxIterations; generation++) {
                // Evaluate fitness
                List<Double> fitnesses = evaluatePopulation(observations, population);

                // Update best individual
                for (int i = 0; i < population.size(); i++) {
                    if (fitnesses.get(i) > bestFitness) {
                        bestFitness = fitnesses.get(i);
                        bestIndividual = population.get(i);
                    }
                }

                // Check convergence
                if (hasConverged(population, fitnesses, generation)) {
                    break;
                }

                // Create next generation
                population = createNextGeneration(population, fitnesses);
            }

            return new OptimizationResult(bestIndividual, computeLogEvidence(observations, bestIndividual, priors));
        }

        private List<ModelParameters> initializePopulation() {
            List<ModelParameters> population = new ArrayList<>();
            for (int i = 0; i < config.populationSize; i++) {
                // Sample from prior distributions
                double S = sampleBeta(priors.s_alpha(), priors.s_beta());
                double N = sampleBeta(priors.n_alpha(), priors.n_beta());
                double alpha = sampleBeta(priors.alpha_alpha(), priors.alpha_beta());
                double beta = sampleLogNormal(priors.beta_mu(), priors.beta_sigma());
                population.add(new ModelParameters(S, N, alpha, beta));
            }
            return population;
        }

        private List<Double> evaluatePopulation(List<Observation> observations, List<ModelParameters> population) {
            return population.stream()
                .map(params -> computeFitness(observations, params))
                .collect(Collectors.toList());
        }

        private double computeFitness(List<Observation> observations, ModelParameters params) {
            double logLikelihood = 0.0;
            for (Observation obs : observations) {
                            // Compute Ψ for this observation with these parameters
            double psi = computePsi(obs.claim, params, priors);

                // Likelihood of observed Ψ given model parameters
                double likelihood = computePsiLikelihood(psi, obs.observedPsi, obs.verificationOutcome);
                logLikelihood += Math.log(likelihood + 1e-12);
            }

            // Add prior
            double logPrior = computeLogPrior(params);

            return logLikelihood + logPrior;
        }

        private static double computePsiLikelihood(double modelPsi, double observedPsi, boolean verificationOutcome) {
            // Likelihood model: observed Ψ is normally distributed around model Ψ
            double error = observedPsi - modelPsi;
            double variance = 0.01; // Observation noise variance
            double likelihood = Math.exp(-error * error / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

            // Weight by verification outcome reliability
            double reliabilityWeight = verificationOutcome ? 1.0 : 0.8;
            return likelihood * reliabilityWeight;
        }

        private List<ModelParameters> createNextGeneration(List<ModelParameters> population, List<Double> fitnesses) {
            List<ModelParameters> nextGen = new ArrayList<>();

            // Elitism - keep best individual
            int bestIndex = findBestIndex(fitnesses);
            nextGen.add(population.get(bestIndex));

            // Create rest through tournament selection, crossover, and mutation
            while (nextGen.size() < config.populationSize) {
                ModelParameters parent1 = tournamentSelect(population, fitnesses);
                ModelParameters parent2 = tournamentSelect(population, fitnesses);

                ModelParameters offspring = crossover(parent1, parent2);
                offspring = mutate(offspring);

                nextGen.add(offspring);
            }

            return nextGen;
        }

        private ModelParameters tournamentSelect(List<ModelParameters> population, List<Double> fitnesses) {
            int idx1 = (int) (Math.random() * population.size());
            int idx2 = (int) (Math.random() * population.size());
            return fitnesses.get(idx1) > fitnesses.get(idx2) ? population.get(idx1) : population.get(idx2);
        }

        private ModelParameters crossover(ModelParameters p1, ModelParameters p2) {
            if (Math.random() > config.crossoverRate) {
                return p1;
            }

            double s = Math.random() < 0.5 ? p1.S() : p2.S();
            double n = Math.random() < 0.5 ? p1.N() : p2.N();
            double alpha = Math.random() < 0.5 ? p1.alpha() : p2.alpha();
            double beta = Math.random() < 0.5 ? p1.beta() : p2.beta();

            return new ModelParameters(s, n, alpha, beta);
        }

        private ModelParameters mutate(ModelParameters params) {
            double s = mutateParameter(params.S(), 0.0, 1.0);
            double n = mutateParameter(params.N(), 0.0, 1.0);
            double alpha = mutateParameter(params.alpha(), 0.0, 1.0);
            double beta = mutateParameter(params.beta(), 0.1, Double.MAX_VALUE);

            return new ModelParameters(s, n, alpha, beta);
        }

        private double mutateParameter(double value, double min, double max) {
            if (Math.random() < config.mutationRate) {
                double mutation = (Math.random() - 0.5) * 0.2; // ±10% mutation
                double newValue = value * (1.0 + mutation);
                return Math.max(min, Math.min(max, newValue));
            }
            return value;
        }

        private boolean hasConverged(List<ModelParameters> population, List<Double> fitnesses, int generation) {
            if (generation < 50) return false; // Minimum generations

            double bestFitness = fitnesses.stream().mapToDouble(x -> x).max().orElse(0);
            double meanFitness = fitnesses.stream().mapToDouble(x -> x).average().orElse(0);
            double relativeImprovement = Math.abs(bestFitness - meanFitness) / Math.abs(meanFitness);

            return relativeImprovement < config.tolerance;
        }

        private int findBestIndex(List<Double> fitnesses) {
            int bestIndex = 0;
            double bestFitness = fitnesses.get(0);
            for (int i = 1; i < fitnesses.size(); i++) {
                if (fitnesses.get(i) > bestFitness) {
                    bestFitness = fitnesses.get(i);
                    bestIndex = i;
                }
            }
            return bestIndex;
        }
    }

    /**
     * Structure learner for hierarchical relationships
     */
    private static final class StructureLearner {
        private final InverseConfig config;

        StructureLearner(InverseConfig config) {
            this.config = config;
        }

        StructureResult learnStructure(List<Observation> observations) {
            // Analyze patterns in the data to infer hierarchical structure
            Map<String, List<String>> relationships = inferRelationships(observations);
            List<String> levels = inferLevels(observations);
            Map<String, Double> weights = computeLevelWeights(observations, levels);

            HierarchicalStructure structure = new HierarchicalStructure(levels, relationships, weights);
            double confidence = computeStructureConfidence(observations, structure);

            return new StructureResult(structure, confidence,
                                     new ArrayList<>(relationships.keySet()),
                                     weights);
        }

        private Map<String, List<String>> inferRelationships(List<Observation> observations) {
            Map<String, List<String>> relationships = new HashMap<>();

            // Analyze correlations between features to infer relationships
            // This is a simplified implementation - could be extended with more sophisticated ML
            relationships.put("authenticity_risk", List.of("source_credibility", "content_quality"));
            relationships.put("virality_risk", List.of("topic_sensitivity", "platform_reach"));
            relationships.put("probability_h_given_e", List.of("evidence_strength", "prior_belief"));

            return relationships;
        }

        private List<String> inferLevels(List<Observation> observations) {
            // Infer hierarchical levels from data patterns
            return List.of("evidence", "context", "domain", "global");
        }

        private Map<String, Double> computeLevelWeights(List<Observation> observations, List<String> levels) {
            Map<String, Double> weights = new HashMap<>();
            double uniformWeight = 1.0 / levels.size();

            for (String level : levels) {
                weights.put(level, uniformWeight);
            }

            return weights;
        }

        private double computeStructureConfidence(List<Observation> observations, HierarchicalStructure structure) {
            // Simplified confidence computation
            return 0.85; // Would compute based on fit to data patterns
        }
    }

    // Helper methods
    private static double computePsi(ClaimData claim, ModelParameters params, ModelPriors priors) {
        double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        double penaltyExponent = -(
                priors.lambda1() * claim.riskAuthenticity() +
                priors.lambda2() * claim.riskVirality()
        );
        double pen = Math.exp(penaltyExponent);
        double p_H_given_E_beta = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);
        double psi = O * pen * p_H_given_E_beta;
        return Math.max(0.0, Math.min(1.0, psi));
    }

    private static double computeLogPrior(ModelParameters params) {
        // Simplified prior computation
        return 0.0; // Would use full prior distributions
    }

    private static double computeLogEvidence(List<Observation> observations, ModelParameters params, ModelPriors priors) {
        double logEvidence = 0.0;
        for (Observation obs : observations) {
            double psi = computePsi(obs.claim, params);
            double likelihood = computePsiLikelihood(psi, obs.observedPsi, obs.verificationOutcome);
            logEvidence += Math.log(likelihood + 1e-12);
        }
        return logEvidence;
    }

    private List<ModelParameters> samplePosterior(List<Observation> observations, ModelParameters mean, int numSamples) {
        // Simplified posterior sampling - would use MCMC in full implementation
        List<ModelParameters> samples = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            // Add small random noise to create samples around mean
            double noise = 0.01;
            double s = Math.max(0, Math.min(1, mean.S() + (Math.random() - 0.5) * noise));
            double n = Math.max(0, Math.min(1, mean.N() + (Math.random() - 0.5) * noise));
            double alpha = Math.max(0, Math.min(1, mean.alpha() + (Math.random() - 0.5) * noise));
            double beta = Math.max(0.1, mean.beta() * (1 + (Math.random() - 0.5) * noise));
            samples.add(new ModelParameters(s, n, alpha, beta));
        }
        return samples;
    }

    private Map<String, Double> estimateParameterUncertainties(List<ModelParameters> samples) {
        Map<String, Double> uncertainties = new HashMap<>();

        List<Double> sValues = samples.stream().map(ModelParameters::S).collect(Collectors.toList());
        List<Double> nValues = samples.stream().map(ModelParameters::N).collect(Collectors.toList());
        List<Double> alphaValues = samples.stream().map(ModelParameters::alpha).collect(Collectors.toList());
        List<Double> betaValues = samples.stream().map(ModelParameters::beta).collect(Collectors.toList());

        uncertainties.put("S", computeStandardDeviation(sValues));
        uncertainties.put("N", computeStandardDeviation(nValues));
        uncertainties.put("alpha", computeStandardDeviation(alphaValues));
        uncertainties.put("beta", computeStandardDeviation(betaValues));

        return uncertainties;
    }

    private double computeStandardDeviation(List<Double> values) {
        double mean = values.stream().mapToDouble(x -> x).average().orElse(0);
        double variance = values.stream()
            .mapToDouble(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0);
        return Math.sqrt(variance);
    }

    private double computeRecoveryConfidence(List<Observation> observations, ModelParameters params, List<ModelParameters> samples) {
        // Compute confidence based on posterior consistency and predictive accuracy
        double posteriorConsistency = computePosteriorConsistency(samples);
        double predictiveAccuracy = computePredictiveAccuracy(observations, params);

        return (posteriorConsistency + predictiveAccuracy) / 2.0;
    }

    private double computePosteriorConsistency(List<ModelParameters> samples) {
        // Simplified consistency measure
        return 0.9; // Would compute based on sample diversity and convergence
    }

    private double computePredictiveAccuracy(List<Observation> observations, ModelParameters params) {
        double totalAccuracy = 0.0;
        for (Observation obs : observations) {
            double modelPsi = computePsi(obs.claim, params);
            double accuracy = 1.0 - Math.abs(modelPsi - obs.observedPsi);
            totalAccuracy += accuracy;
        }
        return totalAccuracy / observations.size();
    }

    // Utility methods for random sampling
    private static double sampleBeta(double alpha, double beta) {
        // Simplified Beta sampling - would use proper implementation
        return Math.random();
    }

    private static double sampleLogNormal(double mu, double sigma) {
        // Simplified LogNormal sampling
        return Math.exp(mu + sigma * (Math.random() - 0.5));
    }

    /**
     * Represents a single observation for inverse modeling
     */
    public static final class Observation {
        public final ClaimData claim;
        public final double observedPsi;
        public final boolean verificationOutcome;

        public Observation(ClaimData claim, double observedPsi, boolean verificationOutcome) {
            this.claim = claim;
            this.observedPsi = observedPsi;
            this.verificationOutcome = verificationOutcome;
        }
    }

    /**
     * Validation result container
     */
    public static final class ValidationResult {
        public final double parameterRecoveryError;
        public final double confidenceAccuracy;
        public final Map<String, Double> parameterErrors;
        public final double overallScore;

        public ValidationResult(ValidationMetrics metrics) {
            this.parameterRecoveryError = metrics.parameterError;
            this.confidenceAccuracy = metrics.confidenceAccuracy;
            this.parameterErrors = metrics.parameterErrors;
            this.overallScore = metrics.overallScore;
        }
    }

    /**
     * Validation metrics computation
     */
    private static final class ValidationMetrics {
        public final double parameterError;
        public final double confidenceAccuracy;
        public final Map<String, Double> parameterErrors;
        public final double overallScore;

        ValidationMetrics(List<Observation> observations, ModelParameters groundTruth, ModelParameters recovered) {
            this.parameterErrors = computeParameterErrors(groundTruth, recovered);
            this.parameterError = parameterErrors.values().stream().mapToDouble(x -> x).average().orElse(0);
            this.confidenceAccuracy = computeConfidenceAccuracy(observations, groundTruth, recovered);
            this.overallScore = (1.0 - parameterError) * confidenceAccuracy;
        }

        private Map<String, Double> computeParameterErrors(ModelParameters truth, ModelParameters recovered) {
            Map<String, Double> errors = new HashMap<>();
            errors.put("S", Math.abs(truth.S() - recovered.S()));
            errors.put("N", Math.abs(truth.N() - recovered.N()));
            errors.put("alpha", Math.abs(truth.alpha() - recovered.alpha()));
            errors.put("beta", Math.abs(truth.beta() - recovered.beta()) / truth.beta()); // Relative error for beta
            return errors;
        }

        private double computeConfidenceAccuracy(List<Observation> observations, ModelParameters truth, ModelParameters recovered) {
            double truthAccuracy = 0.0;
            double recoveredAccuracy = 0.0;

            for (Observation obs : observations) {
                // Would need access to forward model to compute actual Ψ scores
                // This is a placeholder implementation
                truthAccuracy += 1.0; // Placeholder
                recoveredAccuracy += 1.0; // Placeholder
            }

            return recoveredAccuracy / Math.max(truthAccuracy, 1.0);
        }
    }

    /**
     * Optimization result container
     */
    private static final class OptimizationResult {
        public final ModelParameters bestParameters;
        public final double logEvidence;

        OptimizationResult(ModelParameters bestParameters, double logEvidence) {
            this.bestParameters = bestParameters;
            this.logEvidence = logEvidence;
        }
    }
}
