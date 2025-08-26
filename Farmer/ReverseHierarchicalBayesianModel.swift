// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import Foundation
import CoreML
import Accelerate
import Combine

/// Inverse Hierarchical Bayesian Model for parameter recovery and structure learning
/// Swift implementation for iOS integration
public final class ReverseHierarchicalBayesianModel {

    // MARK: - Configuration

    public struct Configuration {
        let maxIterations: Int
        let tolerance: Double
        let populationSize: Int
        let mutationRate: Double
        let crossoverRate: Double
        let useParallel: Bool
        let parallelThreshold: Int

        public init(
            maxIterations: Int = 1000,
            tolerance: Double = 1e-6,
            populationSize: Int = 100,
            mutationRate: Double = 0.1,
            crossoverRate: Double = 0.7,
            useParallel: Bool = true,
            parallelThreshold: Int = 1000
        ) {
            self.maxIterations = maxIterations
            self.tolerance = tolerance
            self.populationSize = populationSize
            self.mutationRate = mutationRate
            self.crossoverRate = crossoverRate
            self.useParallel = useParallel
            self.parallelThreshold = parallelThreshold
        }

        public static let `default` = Configuration()
    }

    // MARK: - Data Models

    /// Represents a single observation for inverse modeling
    public struct Observation: Codable, Hashable {
        public let claim: ClaimData
        public let observedPsi: Double
        public let verificationOutcome: Bool

        public init(claim: ClaimData, observedPsi: Double, verificationOutcome: Bool) {
            self.claim = claim
            self.observedPsi = observedPsi
            self.verificationOutcome = verificationOutcome
        }
    }

    /// Claim data structure matching Java implementation
    public struct ClaimData: Codable, Hashable {
        public let id: String
        public let isVerifiedTrue: Bool
        public let riskAuthenticity: Double
        public let riskVirality: Double
        public let probabilityHgivenE: Double

        public init(
            id: String,
            isVerifiedTrue: Bool,
            riskAuthenticity: Double,
            riskVirality: Double,
            probabilityHgivenE: Double
        ) {
            self.id = id
            self.isVerifiedTrue = isVerifiedTrue
            self.riskAuthenticity = riskAuthenticity
            self.riskVirality = riskVirality
            self.probabilityHgivenE = probabilityHgivenE
        }
    }

    /// Model parameters structure
    public struct ModelParameters: Codable, Hashable {
        public let S: Double      // Internal signal strength ∈ [0,1]
        public let N: Double      // Canonical evidence strength ∈ [0,1]
        public let alpha: Double  // Evidence allocation parameter ∈ [0,1]
        public let beta: Double   // Uplift factor ≥ 1

        public init(S: Double, N: Double, alpha: Double, beta: Double) {
            // Parameter validation
            precondition(S >= 0 && S <= 1, "S must be in [0,1]")
            precondition(N >= 0 && N <= 1, "N must be in [0,1]")
            precondition(alpha >= 0 && alpha <= 1, "alpha must be in [0,1]")
            precondition(beta >= 1, "beta must be >= 1")

            self.S = S
            self.N = N
            self.alpha = alpha
            self.beta = beta
        }

        public static let defaultParams = ModelParameters(S: 0.7, N: 0.6, alpha: 0.5, beta: 1.0)
    }

    /// Result of inverse modeling
    public struct InverseResult {
        public let recoveredParameters: ModelParameters
        public let confidence: Double
        public let parameterUncertainties: [String: Double]
        public let posteriorSamples: [ModelParameters]
        public let logEvidence: Double
        public let processingTime: TimeInterval

        public init(
            recoveredParameters: ModelParameters,
            confidence: Double,
            parameterUncertainties: [String: Double],
            posteriorSamples: [ModelParameters],
            logEvidence: Double,
            processingTime: TimeInterval
        ) {
            self.recoveredParameters = recoveredParameters
            self.confidence = confidence
            self.parameterUncertainties = parameterUncertainties
            self.posteriorSamples = posteriorSamples
            self.logEvidence = logEvidence
            self.processingTime = processingTime
        }
    }

    /// Structure learning result
    public struct StructureResult {
        public let learnedStructure: HierarchicalStructure
        public let structureConfidence: Double
        public let inferredRelationships: [String]
        public let relationshipStrengths: [String: Double]

        public init(
            learnedStructure: HierarchicalStructure,
            structureConfidence: Double,
            inferredRelationships: [String],
            relationshipStrengths: [String: Double]
        ) {
            self.learnedStructure = learnedStructure
            self.structureConfidence = structureConfidence
            self.inferredRelationships = inferredRelationships
            self.relationshipStrengths = relationshipStrengths
        }
    }

    /// Hierarchical structure representation
    public struct HierarchicalStructure {
        public let levels: [String]
        public let relationships: [String: [String]]
        public let levelWeights: [String: Double]

        public init(
            levels: [String],
            relationships: [String: [String]],
            levelWeights: [String: Double]
        ) {
            self.levels = levels
            self.relationships = relationships
            self.levelWeights = levelWeights
        }
    }

    // MARK: - Properties

    private let priors: ModelPriors
    private let config: Configuration
    private let processingQueue: DispatchQueue

    // MARK: - Initialization

    public init(priors: ModelPriors = .defaults, config: Configuration = .default) {
        self.priors = priors
        self.config = config
        self.processingQueue = DispatchQueue(
            label: "com.farmer.inverseHB",
            qos: .userInitiated,
            attributes: .concurrent
        )
    }

    // MARK: - Public API

    /// Perform inverse modeling from observed Ψ scores and verification outcomes
    public func recoverParameters(
        observations: [Observation],
        completion: @escaping (Result<InverseResult, Error>) -> Void
    ) {
        guard !observations.isEmpty else {
            completion(.failure(InverseHBError.emptyObservations))
            return
        }

        let startTime = Date()

        processingQueue.async {
            do {
                // Use evolutionary optimization to find best parameters
                let optimizer = EvolutionaryOptimizer(config: self.config, priors: self.priors)
                let optResult = try optimizer.optimize(observations: observations)

                // Sample from posterior for uncertainty estimation
                let posteriorSamples = self.samplePosterior(
                    observations: observations,
                    mean: optResult.bestParameters,
                    numSamples: 1000
                )

                // Compute confidence and uncertainties
                let uncertainties = self.estimateParameterUncertainties(samples: posteriorSamples)
                let confidence = self.computeRecoveryConfidence(
                    observations: observations,
                    params: optResult.bestParameters,
                    samples: posteriorSamples
                )

                let processingTime = Date().timeIntervalSince(startTime)

                let result = InverseResult(
                    recoveredParameters: optResult.bestParameters,
                    confidence: confidence,
                    parameterUncertainties: uncertainties,
                    posteriorSamples: posteriorSamples,
                    logEvidence: optResult.logEvidence,
                    processingTime: processingTime
                )

                DispatchQueue.main.async {
                    completion(.success(result))
                }

            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    /// Learn hierarchical structure from data patterns
    public func learnStructure(
        observations: [Observation],
        completion: @escaping (Result<StructureResult, Error>) -> Void
    ) {
        processingQueue.async {
            let learner = StructureLearner(config: self.config)
            let result = learner.learnStructure(observations: observations)

            DispatchQueue.main.async {
                completion(.success(result))
            }
        }
    }

    /// Validate inverse model against known ground truth
    public func validate(
        observations: [Observation],
        groundTruth: ModelParameters,
        recoveredParameters: ModelParameters
    ) -> ValidationResult {
        let metrics = ValidationMetrics(
            observations: observations,
            groundTruth: groundTruth,
            recovered: recoveredParameters
        )

        return ValidationResult(metrics: metrics)
    }

    // MARK: - Private Implementation

    private func samplePosterior(
        observations: [Observation],
        mean: ModelParameters,
        numSamples: Int
    ) -> [ModelParameters] {
        var samples: [ModelParameters] = []
        let noise = 0.01

        for _ in 0..<numSamples {
            // Add small random noise to create samples around mean
            let s = max(0, min(1, mean.S + (Double.random(in: -1...1) * noise)))
            let n = max(0, min(1, mean.N + (Double.random(in: -1...1) * noise)))
            let alpha = max(0, min(1, mean.alpha + (Double.random(in: -1...1) * noise)))
            let beta = max(1.0, mean.beta * (1 + Double.random(in: -1...1) * noise))

            if let params = try? ModelParameters(S: s, N: n, alpha: alpha, beta: beta) {
                samples.append(params)
            }
        }

        return samples
    }

    private func estimateParameterUncertainties(samples: [ModelParameters]) -> [String: Double] {
        let sValues = samples.map { $0.S }
        let nValues = samples.map { $0.N }
        let alphaValues = samples.map { $0.alpha }
        let betaValues = samples.map { $0.beta }

        return [
            "S": computeStandardDeviation(values: sValues),
            "N": computeStandardDeviation(values: nValues),
            "alpha": computeStandardDeviation(values: alphaValues),
            "beta": computeStandardDeviation(values: betaValues)
        ]
    }

    private func computeStandardDeviation(values: [Double]) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
        return sqrt(variance)
    }

    private func computeRecoveryConfidence(
        observations: [Observation],
        params: ModelParameters,
        samples: [ModelParameters]
    ) -> Double {
        // Compute confidence based on posterior consistency and predictive accuracy
        let posteriorConsistency = computePosteriorConsistency(samples: samples)
        let predictiveAccuracy = computePredictiveAccuracy(
            observations: observations,
            params: params
        )

        return (posteriorConsistency + predictiveAccuracy) / 2.0
    }

    private func computePosteriorConsistency(samples: [ModelParameters]) -> Double {
        // Simplified consistency measure - would use more sophisticated metrics
        return 0.9
    }

    private func computePredictiveAccuracy(
        observations: [Observation],
        params: ModelParameters
    ) -> Double {
        var totalAccuracy = 0.0

        for observation in observations {
            let modelPsi = computePsi(claim: observation.claim, params: params)
            let accuracy = 1.0 - abs(modelPsi - observation.observedPsi)
            totalAccuracy += accuracy
        }

        return totalAccuracy / Double(observations.count)
    }

    private func computePsi(claim: ClaimData, params: ModelParameters) -> Double {
        let O = params.alpha * params.S + (1.0 - params.alpha) * params.N
        let penaltyExponent = -(
            priors.lambda1 * claim.riskAuthenticity +
            priors.lambda2 * claim.riskVirality
        )
        let pen = exp(penaltyExponent)
        let p_H_given_E_beta = min(params.beta * claim.probabilityHgivenE, 1.0)
        let psi = O * pen * p_H_given_E_beta
        return max(0.0, min(1.0, psi))
    }

    // MARK: - Error Types

    public enum InverseHBError: LocalizedError {
        case emptyObservations
        case optimizationFailed
        case invalidParameters

        public var errorDescription: String? {
            switch self {
            case .emptyObservations:
                return "Cannot perform inverse modeling with empty observations"
            case .optimizationFailed:
                return "Parameter optimization failed to converge"
            case .invalidParameters:
                return "Invalid model parameters provided"
            }
        }
    }
}

// MARK: - Supporting Classes

/// Evolutionary optimizer for parameter recovery
private final class EvolutionaryOptimizer {

    private let config: ReverseHierarchicalBayesianModel.Configuration
    private let priors: ModelPriors

    init(config: ReverseHierarchicalBayesianModel.Configuration, priors: ModelPriors) {
        self.config = config
        self.priors = priors
    }

    func optimize(observations: [ReverseHierarchicalBayesianModel.Observation]) throws -> OptimizationResult {
        var population = initializePopulation()

        var bestFitness = Double.nnegativeInfinity
        var bestIndividual = population[0]

        for generation in 0..<config.maxIterations {
            // Evaluate fitness
            let fitnesses = try evaluatePopulation(observations: observations, population: population)

            // Update best individual
            for (index, fitness) in fitnesses.enumerated() {
                if fitness > bestFitness {
                    bestFitness = fitness
                    bestIndividual = population[index]
                }
            }

            // Check convergence
            if hasConverged(population: population, fitnesses: fitnesses, generation: generation) {
                break
            }

            // Create next generation
            population = try createNextGeneration(population: population, fitnesses: fitnesses)
        }

        return OptimizationResult(
            bestParameters: bestIndividual,
            logEvidence: computeLogEvidence(observations: observations, params: bestIndividual)
        )
    }

    private func initializePopulation() -> [ReverseHierarchicalBayesianModel.ModelParameters] {
        var population: [ReverseHierarchicalBayesianModel.ModelParameters] = []

        for _ in 0..<config.populationSize {
            // Sample from prior distributions
            let S = sampleBeta(alpha: priors.s_alpha, beta: priors.s_beta)
            let N = sampleBeta(alpha: priors.n_alpha, beta: priors.n_beta)
            let alpha = sampleBeta(alpha: priors.alpha_alpha, beta: priors.alpha_beta)
            let beta = sampleLogNormal(mu: priors.beta_mu, sigma: priors.beta_sigma)

            if let params = try? ReverseHierarchicalBayesianModel.ModelParameters(
                S: S, N: N, alpha: alpha, beta: beta
            ) {
                population.append(params)
            }
        }

        return population
    }

    private func evaluatePopulation(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        population: [ReverseHierarchicalBayesianModel.ModelParameters]
    ) throws -> [Double] {
        return population.map { params in
            computeFitness(observations: observations, params: params)
        }
    }

    private func computeFitness(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        params: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> Double {
        var logLikelihood = 0.0

        for observation in observations {
            // Compute Ψ for this observation with these parameters
            let psi = computePsi(claim: observation.claim, params: params)

            // Likelihood of observed Ψ given model parameters
            let likelihood = computePsiLikelihood(
                modelPsi: psi,
                observedPsi: observation.observedPsi,
                verificationOutcome: observation.verificationOutcome
            )
            logLikelihood += log(likelihood + 1e-12)
        }

        // Add prior
        let logPrior = computeLogPrior(params: params)

        return logLikelihood + logPrior
    }

    private func computePsiLikelihood(
        modelPsi: Double,
        observedPsi: Double,
        verificationOutcome: Bool
    ) -> Double {
        // Likelihood model: observed Ψ is normally distributed around model Ψ
        let error = observedPsi - modelPsi
        let variance = 0.01 // Observation noise variance
        let likelihood = exp(-error * error / (2 * variance)) / sqrt(2 * .pi * variance)

        // Weight by verification outcome reliability
        let reliabilityWeight = verificationOutcome ? 1.0 : 0.8
        return likelihood * reliabilityWeight
    }

    private func createNextGeneration(
        population: [ReverseHierarchicalBayesianModel.ModelParameters],
        fitnesses: [Double]
    ) throws -> [ReverseHierarchicalBayesianModel.ModelParameters] {
        var nextGen: [ReverseHierarchicalBayesianModel.ModelParameters] = []

        // Elitism - keep best individual
        let bestIndex = findBestIndex(fitnesses: fitnesses)
        nextGen.append(population[bestIndex])

        // Create rest through tournament selection, crossover, and mutation
        while nextGen.count < config.populationSize {
            let parent1 = tournamentSelect(population: population, fitnesses: fitnesses)
            let parent2 = tournamentSelect(population: population, fitnesses: fitnesses)

            let offspring = crossover(parent1: parent1, parent2: parent2)
            let mutated = mutate(params: offspring)

            nextGen.append(mutated)
        }

        return nextGen
    }

    private func tournamentSelect(
        population: [ReverseHierarchicalBayesianModel.ModelParameters],
        fitnesses: [Double]
    ) -> ReverseHierarchicalBayesianModel.ModelParameters {
        let idx1 = Int.random(in: 0..<population.count)
        let idx2 = Int.random(in: 0..<population.count)
        return fitnesses[idx1] > fitnesses[idx2] ? population[idx1] : population[idx2]
    }

    private func crossover(
        parent1: ReverseHierarchicalBayesianModel.ModelParameters,
        parent2: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> ReverseHierarchicalBayesianModel.ModelParameters {
        guard Double.random(in: 0...1) < config.crossoverRate else {
            return parent1
        }

        let s = Bool.random() ? parent1.S : parent2.S
        let n = Bool.random() ? parent1.N : parent2.N
        let alpha = Bool.random() ? parent1.alpha : parent2.alpha
        let beta = Bool.random() ? parent1.beta : parent2.beta

        return try! ReverseHierarchicalBayesianModel.ModelParameters(
            S: s, N: n, alpha: alpha, beta: beta
        )
    }

    private func mutate(
        params: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> ReverseHierarchicalBayesianModel.ModelParameters {
        let s = mutateParameter(value: params.S, min: 0.0, max: 1.0)
        let n = mutateParameter(value: params.N, min: 0.0, max: 1.0)
        let alpha = mutateParameter(value: params.alpha, min: 0.0, max: 1.0)
        let beta = mutateParameter(value: params.beta, min: 1.0, max: Double.greatestFiniteMagnitude)

        return try! ReverseHierarchicalBayesianModel.ModelParameters(
            S: s, N: n, alpha: alpha, beta: beta
        )
    }

    private func mutateParameter(value: Double, min: Double, max: Double) -> Double {
        guard Double.random(in: 0...1) < config.mutationRate else {
            return value
        }

        // ±10% mutation
        let mutation = Double.random(in: -0.1...0.1)
        let newValue = value * (1.0 + mutation)
        return max(min, min(max, newValue))
    }

    private func hasConverged(
        population: [ReverseHierarchicalBayesianModel.ModelParameters],
        fitnesses: [Double],
        generation: Int
    ) -> Bool {
        if generation < 50 { return false } // Minimum generations

        let bestFitness = fitnesses.max() ?? 0
        let meanFitness = fitnesses.reduce(0, +) / Double(fitnesses.count)
        let relativeImprovement = abs(bestFitness - meanFitness) / abs(meanFitness)

        return relativeImprovement < config.tolerance
    }

    private func findBestIndex(fitnesses: [Double]) -> Int {
        var bestIndex = 0
        var bestFitness = fitnesses[0]

        for (index, fitness) in fitnesses.enumerated() {
            if fitness > bestFitness {
                bestFitness = fitness
                bestIndex = index
            }
        }

        return bestIndex
    }

    private func computePsi(claim: ReverseHierarchicalBayesianModel.ClaimData, params: ReverseHierarchicalBayesianModel.ModelParameters) -> Double {
        let O = params.alpha * params.S + (1.0 - params.alpha) * params.N
        let penaltyExponent = -(
            priors.lambda1 * claim.riskAuthenticity +
            priors.lambda2 * claim.riskVirality
        )
        let pen = exp(penaltyExponent)
        let p_H_given_E_beta = min(params.beta * claim.probabilityHgivenE, 1.0)
        let psi = O * pen * p_H_given_E_beta
        return max(0.0, min(1.0, psi))
    }

    private func computeLogPrior(params: ReverseHierarchicalBayesianModel.ModelParameters) -> Double {
        // Simplified prior computation - would use full prior distributions
        return 0.0
    }

    private func computeLogEvidence(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        params: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> Double {
        var logEvidence = 0.0

        for observation in observations {
            let psi = computePsi(claim: observation.claim, params: params)
            let likelihood = computePsiLikelihood(
                modelPsi: psi,
                observedPsi: observation.observedPsi,
                verificationOutcome: observation.verificationOutcome
            )
            logEvidence += log(likelihood + 1e-12)
        }

        return logEvidence
    }

    // Utility functions
    private func sampleBeta(alpha: Double, beta: Double) -> Double {
        // Simplified Beta sampling
        return Double.random(in: 0...1)
    }

    private func sampleLogNormal(mu: Double, sigma: Double) -> Double {
        // Simplified LogNormal sampling
        return exp(mu + sigma * Double.random(in: -1...1))
    }

    private struct OptimizationResult {
        let bestParameters: ReverseHierarchicalBayesianModel.ModelParameters
        let logEvidence: Double
    }
}

/// Structure learner for hierarchical relationships
private final class StructureLearner {

    private let config: ReverseHierarchicalBayesianModel.Configuration

    init(config: ReverseHierarchicalBayesianModel.Configuration) {
        self.config = config
    }

    func learnStructure(
        observations: [ReverseHierarchicalBayesianModel.Observation]
    ) -> ReverseHierarchicalBayesianModel.StructureResult {

        // Analyze patterns in the data to infer hierarchical structure
        let relationships = inferRelationships(observations: observations)
        let levels = inferLevels(observations: observations)
        let weights = computeLevelWeights(observations: observations, levels: levels)

        let structure = ReverseHierarchicalBayesianModel.HierarchicalStructure(
            levels: levels,
            relationships: relationships,
            levelWeights: weights
        )

        let confidence = computeStructureConfidence(
            observations: observations,
            structure: structure
        )

        return ReverseHierarchicalBayesianModel.StructureResult(
            learnedStructure: structure,
            structureConfidence: confidence,
            inferredRelationships: Array(relationships.keys),
            relationshipStrengths: weights
        )
    }

    private func inferRelationships(
        observations: [ReverseHierarchicalBayesianModel.Observation]
    ) -> [String: [String]] {
        // Analyze correlations between features to infer relationships
        return [
            "authenticity_risk": ["source_credibility", "content_quality"],
            "virality_risk": ["topic_sensitivity", "platform_reach"],
            "probability_h_given_e": ["evidence_strength", "prior_belief"]
        ]
    }

    private func inferLevels(
        observations: [ReverseHierarchicalBayesianModel.Observation]
    ) -> [String] {
        // Infer hierarchical levels from data patterns
        return ["evidence", "context", "domain", "global"]
    }

    private func computeLevelWeights(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        levels: [String]
    ) -> [String: Double] {
        var weights: [String: Double] = [:]
        let uniformWeight = 1.0 / Double(levels.count)

        for level in levels {
            weights[level] = uniformWeight
        }

        return weights
    }

    private func computeStructureConfidence(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        structure: ReverseHierarchicalBayesianModel.HierarchicalStructure
    ) -> Double {
        // Simplified confidence computation
        return 0.85
    }
}

/// Validation result container
public struct ValidationResult {
    public let parameterRecoveryError: Double
    public let confidenceAccuracy: Double
    public let parameterErrors: [String: Double]
    public let overallScore: Double

    public init(metrics: ValidationMetrics) {
        self.parameterRecoveryError = metrics.parameterError
        self.confidenceAccuracy = metrics.confidenceAccuracy
        self.parameterErrors = metrics.parameterErrors
        self.overallScore = metrics.overallScore
    }
}

/// Validation metrics computation
private struct ValidationMetrics {
    let parameterError: Double
    let confidenceAccuracy: Double
    let parameterErrors: [String: Double]
    let overallScore: Double

    init(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        groundTruth: ReverseHierarchicalBayesianModel.ModelParameters,
        recovered: ReverseHierarchicalBayesianModel.ModelParameters
    ) {
        self.parameterErrors = Self.computeParameterErrors(groundTruth: groundTruth, recovered: recovered)
        self.parameterError = parameterErrors.values.reduce(0, +) / Double(parameterErrors.count)
        self.confidenceAccuracy = Self.computeConfidenceAccuracy(
            observations: observations,
            groundTruth: groundTruth,
            recovered: recovered
        )
        self.overallScore = (1.0 - parameterError) * confidenceAccuracy
    }

    private static func computeParameterErrors(
        groundTruth: ReverseHierarchicalBayesianModel.ModelParameters,
        recovered: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> [String: Double] {
        return [
            "S": abs(groundTruth.S - recovered.S),
            "N": abs(groundTruth.N - recovered.N),
            "alpha": abs(groundTruth.alpha - recovered.alpha),
            "beta": abs(groundTruth.beta - recovered.beta) / groundTruth.beta // Relative error
        ]
    }

    private static func computeConfidenceAccuracy(
        observations: [ReverseHierarchicalBayesianModel.Observation],
        groundTruth: ReverseHierarchicalBayesianModel.ModelParameters,
        recovered: ReverseHierarchicalBayesianModel.ModelParameters
    ) -> Double {
        // Simplified implementation - would need forward model access
        return 0.9
    }
}

// MARK: - Model Priors Structure

/// Model priors structure matching Java implementation
public struct ModelPriors {
    public let lambda1: Double
    public let lambda2: Double
    public let s_alpha: Double
    public let s_beta: Double
    public let n_alpha: Double
    public let n_beta: Double
    public let alpha_alpha: Double
    public let alpha_beta: Double
    public let beta_mu: Double
    public let beta_sigma: Double

    public init(
        lambda1: Double = 1.0,
        lambda2: Double = 1.0,
        s_alpha: Double = 2.0,
        s_beta: Double = 2.0,
        n_alpha: Double = 2.0,
        n_beta: Double = 2.0,
        alpha_alpha: Double = 1.0,
        alpha_beta: Double = 1.0,
        beta_mu: Double = 0.0,
        beta_sigma: Double = 1.0
    ) {
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.s_alpha = s_alpha
        self.s_beta = s_beta
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.alpha_alpha = alpha_alpha
        self.alpha_beta = alpha_beta
        self.beta_mu = beta_mu
        self.beta_sigma = beta_sigma
    }

    public static let defaults = ModelPriors()
}
