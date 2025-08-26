// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import Foundation
import SwiftUI
import Combine

/// ViewModel for managing inverse HB model state and operations
/// Provides SwiftUI-compatible reactive interface
@MainActor
public final class ReverseHierarchicalBayesianViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published public var observations: [ReverseHierarchicalBayesianModel.Observation] = []
    @Published public var isProcessing = false
    @Published public var processingProgress: Double = 0.0
    @Published public var currentOperation: String = ""

    // Results
    @Published public var inverseResult: ReverseHierarchicalBayesianModel.InverseResult?
    @Published public var structureResult: ReverseHierarchicalBayesianModel.StructureResult?
    @Published public var validationResult: ValidationResult?

    // Configuration
    @Published public var config = ReverseHierarchicalBayesianModel.Configuration.default
    @Published public var priors = ModelPriors.defaults

    // MARK: - Private Properties

    private let inverseModel: ReverseHierarchicalBayesianModel
    private var cancellables = Set<AnyCancellable>()
    private let progressTimer = Timer.publish(every: 0.1, on: .main, in: .common)

    // MARK: - Initialization

    public init() {
        self.inverseModel = ReverseHierarchicalBayesianModel()

        setupBindings()
    }

    public init(config: ReverseHierarchicalBayesianModel.Configuration, priors: ModelPriors) {
        self.config = config
        self.priors = priors
        self.inverseModel = ReverseHierarchicalBayesianModel(priors: priors, config: config)

        setupBindings()
    }

    private func setupBindings() {
        // Update model when configuration changes
        Publishers.CombineLatest($config, $priors)
            .debounce(for: .milliseconds(500), scheduler: DispatchQueue.main)
            .sink { [weak self] newConfig, newPriors in
                self?.updateModel(config: newConfig, priors: newPriors)
            }
            .store(in: &cancellables)
    }

    private func updateModel(config: ReverseHierarchicalBayesianModel.Configuration, priors: ModelPriors) {
        // Reinitialize model with new configuration
        // In a real implementation, you might want to update the existing model
        objectWillChange.send()
    }

    // MARK: - Public Methods

    /// Add a new observation to the dataset
    public func addObservation(
        claim: ReverseHierarchicalBayesianModel.ClaimData,
        observedPsi: Double,
        verificationOutcome: Bool
    ) {
        let observation = ReverseHierarchicalBayesianModel.Observation(
            claim: claim,
            observedPsi: observedPsi,
            verificationOutcome: verificationOutcome
        )
        observations.append(observation)
    }

    /// Remove observation at index
    public func removeObservation(at index: Int) {
        guard index >= 0 && index < observations.count else { return }
        observations.remove(at: index)
        clearResults()
    }

    /// Clear all observations and results
    public func clearAll() {
        observations.removeAll()
        clearResults()
    }

    /// Perform parameter recovery
    public func recoverParameters() {
        guard !observations.isEmpty else {
            print("Cannot recover parameters: no observations available")
            return
        }

        startProcessing(operation: "Recovering Parameters...")

        inverseModel.recoverParameters(observations: observations) { [weak self] result in
            guard let self = self else { return }

            self.stopProcessing()

            switch result {
            case .success(let inverseResult):
                self.inverseResult = inverseResult
                print("Parameter recovery completed successfully")
                print("Recovered parameters: S=\(inverseResult.recoveredParameters.S), " +
                      "N=\(inverseResult.recoveredParameters.N), " +
                      "alpha=\(inverseResult.recoveredParameters.alpha), " +
                      "beta=\(inverseResult.recoveredParameters.beta)")
                print("Confidence: \(String(format: "%.2f", inverseResult.confidence * 100))%")

            case .failure(let error):
                print("Parameter recovery failed: \(error.localizedDescription)")
                self.inverseResult = nil
            }
        }
    }

    /// Learn hierarchical structure
    public func learnStructure() {
        guard !observations.isEmpty else {
            print("Cannot learn structure: no observations available")
            return
        }

        startProcessing(operation: "Learning Structure...")

        inverseModel.learnStructure(observations: observations) { [weak self] result in
            guard let self = self else { return }

            self.stopProcessing()

            switch result {
            case .success(let structureResult):
                self.structureResult = structureResult
                print("Structure learning completed successfully")
                print("Inferred relationships: \(structureResult.inferredRelationships.joined(separator: ", "))")
                print("Structure confidence: \(String(format: "%.2f", structureResult.structureConfidence * 100))%")

            case .failure(let error):
                print("Structure learning failed: \(error.localizedDescription)")
                self.structureResult = nil
            }
        }
    }

    /// Validate against ground truth parameters
    public func validateAgainst(groundTruth: ReverseHierarchicalBayesianModel.ModelParameters) {
        guard let recovered = inverseResult?.recoveredParameters else {
            print("No recovered parameters available for validation")
            return
        }

        let result = inverseModel.validate(
            observations: observations,
            groundTruth: groundTruth,
            recoveredParameters: recovered
        )

        validationResult = result

        print("Validation completed:")
        print("Parameter recovery error: \(String(format: "%.4f", result.parameterRecoveryError))")
        print("Confidence accuracy: \(String(format: "%.2f", result.confidenceAccuracy * 100))%")
        print("Overall score: \(String(format: "%.2f", result.overallScore * 100))%")
    }

    /// Generate sample data for testing
    public func generateSampleData(count: Int = 20) {
        clearAll()

        for i in 0..<count {
            let claim = ReverseHierarchicalBayesianModel.ClaimData(
                id: "sample_\(i)",
                isVerifiedTrue: Bool.random(),
                riskAuthenticity: Double.random(in: 0...1),
                riskVirality: Double.random(in: 0...1),
                probabilityHgivenE: Double.random(in: 0.1...0.9)
            )

            // Generate realistic observed Ψ based on claim properties
            let truePsi = computeGroundTruthPsi(claim: claim)
            let noise = Double.random(in: -0.1...0.1)
            let observedPsi = max(0, min(1, truePsi + noise))

            let verificationOutcome = Bool.random() // Simulated verification

            addObservation(
                claim: claim,
                observedPsi: observedPsi,
                verificationOutcome: verificationOutcome
            )
        }

        print("Generated \(count) sample observations")
    }

    /// Export results to JSON
    public func exportResults() -> Data? {
        let exportData = ExportData(
            observations: observations,
            inverseResult: inverseResult,
            structureResult: structureResult,
            validationResult: validationResult,
            timestamp: Date(),
            config: config,
            priors: priors
        )

        return try? JSONEncoder().encode(exportData)
    }

    // MARK: - Private Methods

    private func startProcessing(operation: String) {
        isProcessing = true
        currentOperation = operation
        processingProgress = 0.0

        // Simulate progress updates
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
            guard let self = self, self.isProcessing else {
                timer.invalidate()
                return
            }

            self.processingProgress = min(self.processingProgress + 0.05, 0.95)
        }
    }

    private func stopProcessing() {
        isProcessing = false
        currentOperation = ""
        processingProgress = 1.0

        // Reset progress after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.processingProgress = 0.0
        }
    }

    private func clearResults() {
        inverseResult = nil
        structureResult = nil
        validationResult = nil
    }

    private func computeGroundTruthPsi(claim: ReverseHierarchicalBayesianModel.ClaimData) -> Double {
        // Simplified ground truth computation for sample data generation
        let baseEvidence = claim.probabilityHgivenE
        let authenticityPenalty = exp(-claim.riskAuthenticity)
        let viralityPenalty = exp(-claim.riskVirality * 0.5)

        let psi = baseEvidence * authenticityPenalty * viralityPenalty
        return max(0, min(1, psi))
    }

    // MARK: - Computed Properties

    public var hasObservations: Bool {
        !observations.isEmpty
    }

    public var hasResults: Bool {
        inverseResult != nil || structureResult != nil
    }

    public var canRecoverParameters: Bool {
        hasObservations && !isProcessing
    }

    public var canLearnStructure: Bool {
        hasObservations && !isProcessing
    }

    public var canValidate: Bool {
        inverseResult != nil && !isProcessing
    }
}

// MARK: - Supporting Types

/// Data structure for exporting results
private struct ExportData: Codable {
    let observations: [ReverseHierarchicalBayesianModel.Observation]
    let inverseResult: ReverseHierarchicalBayesianModel.InverseResult?
    let structureResult: ReverseHierarchicalBayesianModel.StructureResult?
    let validationResult: ValidationResult?
    let timestamp: Date
    let config: ReverseHierarchicalBayesianModel.Configuration
    let priors: ModelPriors
}

/// Extension to make Configuration Codable for export
extension ReverseHierarchicalBayesianModel.Configuration: Codable {
    enum CodingKeys: String, CodingKey {
        case maxIterations, tolerance, populationSize, mutationRate, crossoverRate, useParallel, parallelThreshold
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let maxIterations = try container.decode(Int.self, forKey: .maxIterations)
        let tolerance = try container.decode(Double.self, forKey: .tolerance)
        let populationSize = try container.decode(Int.self, forKey: .populationSize)
        let mutationRate = try container.decode(Double.self, forKey: .mutationRate)
        let crossoverRate = try container.decode(Double.self, forKey: .crossoverRate)
        let useParallel = try container.decode(Bool.self, forKey: .useParallel)
        let parallelThreshold = try container.decode(Int.self, forKey: .parallelThreshold)

        self.init(
            maxIterations: maxIterations,
            tolerance: tolerance,
            populationSize: populationSize,
            mutationRate: mutationRate,
            crossoverRate: crossoverRate,
            useParallel: useParallel,
            parallelThreshold: parallelThreshold
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(maxIterations, forKey: .maxIterations)
        try container.encode(tolerance, forKey: .tolerance)
        try container.encode(populationSize, forKey: .populationSize)
        try container.encode(mutationRate, forKey: .mutationRate)
        try container.encode(crossoverRate, forKey: .crossoverRate)
        try container.encode(useParallel, forKey: .useParallel)
        try container.encode(parallelThreshold, forKey: .parallelThreshold)
    }
}

/// Extension to make ModelPriors Codable for export
extension ModelPriors: Codable {
    enum CodingKeys: String, CodingKey {
        case lambda1, lambda2, s_alpha, s_beta, n_alpha, n_beta, alpha_alpha, alpha_beta, beta_mu, beta_sigma
    }
}
