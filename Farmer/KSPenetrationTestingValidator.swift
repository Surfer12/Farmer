// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  KSPenetrationTestingValidator.swift
//  Farmer
//
//  Created by Ryan David Oates on 8/26/25.
//  K-S validation framework for penetration testing quality assurance

import Foundation
import CoreML
import Accelerate

/// K-S validation result structure
struct KSValidationResult: Codable {
    let ksStatistic: Double
    let pValue: Double
    let isValid: Bool
    let confidenceLevel: Double
    let recommendation: String
    let distributionSimilarity: Double
    let timestamp: Date

    init(
        ksStatistic: Double,
        pValue: Double,
        isValid: Bool,
        confidenceLevel: Double,
        recommendation: String,
        distributionSimilarity: Double
    ) {
        self.ksStatistic = ksStatistic
        self.pValue = pValue
        self.isValid = isValid
        self.confidenceLevel = confidenceLevel
        self.recommendation = recommendation
        self.distributionSimilarity = distributionSimilarity
        self.timestamp = Date()
    }
}

/// Progressive validation stage
enum ValidationStage: String, Codable {
    case reconnaissance = "Reconnaissance"
    case scanning = "Scanning"
    case gainingAccess = "Gaining Access"
    case maintainingAccess = "Maintaining Access"
    case coveringTracks = "Covering Tracks"
}

/// K-S Penetration Testing Validator
class KSPenetrationTestingValidator {

    private let alpha: Double
    private let confidenceThreshold: Double
    private let maxSyntheticRatio: Double
    private var validationHistory: [KSValidationResult] = []
    private var stageThresholds: [ValidationStage: [String: Double]]

    /// Initialize K-S validator for penetration testing
    /// - Parameters:
    ///   - alpha: Significance level for K-S tests
    ///   - confidenceThreshold: Minimum confidence for validation
    ///   - maxSyntheticRatio: Maximum allowed synthetic data proportion
    init(alpha: Double = 0.05, confidenceThreshold: Double = 0.7, maxSyntheticRatio: Double = 0.5) {
        self.alpha = alpha
        self.confidenceThreshold = confidenceThreshold
        self.maxSyntheticRatio = maxSyntheticRatio

        // Define stage-specific validation thresholds
        self.stageThresholds = [
            .reconnaissance: ["min_similarity": 0.6, "max_synthetic": 0.8],
            .scanning: ["min_similarity": 0.7, "max_synthetic": 0.6],
            .gainingAccess: ["min_similarity": 0.8, "max_synthetic": 0.4],
            .maintainingAccess: ["min_similarity": 0.85, "max_synthetic": 0.3],
            .coveringTracks: ["min_similarity": 0.9, "max_synthetic": 0.2]
        ]

        print("K-S Penetration Testing Validator initialized:")
        print("  Alpha: \(alpha)")
        print("  Confidence threshold: \(confidenceThreshold)")
        print("  Max synthetic ratio: \(maxSyntheticRatio)")
    }

    /// Perform one-sample K-S test against reference distribution
    /// - Parameters:
    ///   - data: Sample data to test
    ///   - referenceType: Type of reference distribution
    /// - Returns: Validation result
    func oneSampleKSTest(data: [Double], referenceType: String = "normal") -> KSValidationResult {
        // Simplified implementation - would integrate with scipy.stats.kstest in Python
        // For now, using basic statistical measures

        let mean = data.reduce(0, +) / Double(data.count)
        let variance = data.map { pow($0 - mean, 2) }.reduce(0, +) / Double(data.count - 1)
        let stdDev = sqrt(variance)

        // Basic normality test using skewness and kurtosis approximation
        let skewness = data.map { pow(($0 - mean) / stdDev, 3) }.reduce(0, +) / Double(data.count)
        let kurtosis = data.map { pow(($0 - mean) / stdDev, 4) }.reduce(0, +) / Double(data.count) - 3.0

        // Simple K-S-like statistic based on deviation from normal distribution
        let ksStatistic = max(abs(skewness), abs(kurtosis) / 2.0) / 2.0

        // Approximate p-value using chi-square distribution
        let pValue = exp(-pow(ksStatistic * sqrt(Double(data.count)), 2) / 2.0)

        let isValid = pValue > alpha
        let confidenceLevel = 1.0 - ksStatistic
        let distributionSimilarity = 1.0 - ksStatistic

        var recommendation = ""
        if isValid && confidenceLevel > confidenceThreshold {
            recommendation = "ACCEPT: Data follows expected \(referenceType) distribution"
        } else if isValid {
            recommendation = "CAUTION: Statistically valid but low confidence"
        } else {
            recommendation = "REJECT: Significant deviation from expected distribution"
        }

        let result = KSValidationResult(
            ksStatistic: ksStatistic,
            pValue: pValue,
            isValid: isValid,
            confidenceLevel: confidenceLevel,
            recommendation: recommendation,
            distributionSimilarity: distributionSimilarity
        )

        validationHistory.append(result)
        return result
    }

    /// Perform two-sample K-S test between synthetic and real data
    /// - Parameters:
    ///   - syntheticData: Synthetic penetration testing data
    ///   - realData: Real-world reference data
    /// - Returns: Validation result
    func twoSampleKSTest(syntheticData: [Double], realData: [Double]) -> KSValidationResult {
        // Simplified implementation - would use scipy.stats.ks_2samp

        // Sort both datasets
        let sortedSynthetic = syntheticData.sorted()
        let sortedReal = realData.sorted()

        // Compute empirical CDFs
        let n1 = Double(syntheticData.count)
        let n2 = Double(realData.count)

        // Simple K-S statistic computation
        var maxDifference = 0.0

        for (i, value) in sortedSynthetic.enumerated() {
            let cdf1 = Double(i + 1) / n1

            // Find corresponding value in real data CDF
            let realIndex = sortedReal.firstIndex { $0 >= value } ?? sortedReal.count
            let cdf2 = Double(realIndex) / n2

            let difference = abs(cdf1 - cdf2)
            maxDifference = max(maxDifference, difference)
        }

        // Approximate p-value
        let ksStatistic = maxDifference
        let pValue = exp(-2.0 * pow(maxDifference, 2) * (n1 * n2) / (n1 + n2))

        let isValid = pValue > alpha
        let confidenceLevel = 1.0 - ksStatistic
        let distributionSimilarity = 1.0 - ksStatistic

        var recommendation = ""
        if isValid && confidenceLevel > confidenceThreshold {
            recommendation = "ACCEPT: Synthetic data maintains real-world fidelity"
        } else if isValid && confidenceLevel > 0.5 {
            recommendation = "CONDITIONAL: Acceptable with monitoring"
        } else if ksStatistic < 0.2 {
            recommendation = "CAUTION: Statistically different but practically similar"
        } else {
            recommendation = "REJECT: Significant distribution mismatch - risk of invalid results"
        }

        let result = KSValidationResult(
            ksStatistic: ksStatistic,
            pValue: pValue,
            isValid: isValid,
            confidenceLevel: confidenceLevel,
            recommendation: recommendation,
            distributionSimilarity: distributionSimilarity
        )

        validationHistory.append(result)
        return result
    }

    /// Progressive validation gate for penetration testing stages
    /// - Parameters:
    ///   - syntheticData: Synthetic penetration testing data
    ///   - realData: Real-world reference data
    ///   - stage: Current penetration testing stage
    /// - Returns: Validation gate result
    func progressiveValidationGate(
        syntheticData: [Double],
        realData: [Double],
        stage: ValidationStage
    ) -> [String: Any] {

        let ksResult = twoSampleKSTest(syntheticData: syntheticData, realData: realData)

        guard let thresholds = stageThresholds[stage] else {
            return ["error": "Unknown validation stage"]
        }

        let minSimilarity = thresholds["min_similarity"] ?? 0.7
        let maxSynthetic = thresholds["max_synthetic"] ?? 0.5

        // Determine optimal synthetic ratio
        var syntheticRatio = 0.0
        if ksResult.distributionSimilarity >= minSimilarity {
            syntheticRatio = min(maxSynthetic, maxSyntheticRatio * ksResult.distributionSimilarity)
        } else {
            syntheticRatio = max(0.1, maxSynthetic * ksResult.distributionSimilarity)
        }

        let gateStatus = ksResult.distributionSimilarity >= minSimilarity ? "PASS" : "CONDITIONAL"
        let thresholdMet = ksResult.distributionSimilarity >= minSimilarity

        return [
            "ks_result": ksResult,
            "gate_status": gateStatus,
            "recommended_synthetic_ratio": syntheticRatio,
            "stage": stage.rawValue,
            "threshold_met": thresholdMet,
            "quality_score": ksResult.distributionSimilarity,
            "stage_thresholds": thresholds
        ]
    }

    /// Validate penetration testing attack patterns
    /// - Parameters:
    ///   - attackPatterns: Observed attack patterns
    ///   - referencePatterns: Known attack pattern distributions
    /// - Returns: Validation results for each pattern type
    func validateAttackPatterns(
        attackPatterns: [String: [Double]],
        referencePatterns: [String: [Double]]
    ) -> [String: KSValidationResult] {

        var results: [String: KSValidationResult] = [:]

        for (patternType, patterns) in attackPatterns {
            if let reference = referencePatterns[patternType] {
                let result = twoSampleKSTest(syntheticData: patterns, realData: reference)
                results[patternType] = result

                print("Pattern '\(patternType)' validation:")
                print("  K-S Statistic: \(String(format: "%.4f", result.ksStatistic))")
                print("  Distribution Similarity: \(String(format: "%.3f", result.distributionSimilarity))")
                print("  Recommendation: \(result.recommendation)")
            }
        }

        return results
    }

    /// Validate security finding confidence using K-S statistics
    /// - Parameters:
    ///   - securityFindings: Array of security findings
    ///   - referenceFindings: Reference security finding distributions
    /// - Returns: Findings with K-S validated confidence
    func validateSecurityFindingConfidence(
        securityFindings: [SecurityFinding],
        referenceFindings: [String: [Double]]
    ) -> [SecurityFinding] {

        var validatedFindings: [SecurityFinding] = []

        for finding in securityFindings {
            // Extract relevant metrics for validation
            let severityScore = severityToNumeric(finding.severity)
            let cvssScore = finding.cvssScore

            // Validate against reference distributions
            if let severityRef = referenceFindings["severity_scores"] {
                let severityValidation = twoSampleKSTest(
                    syntheticData: [severityScore],
                    realData: severityRef
                )

                // Update finding confidence based on K-S validation
                var updatedFinding = finding
                let adjustedConfidence = finding.koopmanStability * severityValidation.distributionSimilarity

                // Create new finding with validated confidence
                let validatedFinding = SecurityFinding(
                    vulnerabilityType: finding.vulnerabilityType,
                    severity: finding.severity,
                    title: finding.title,
                    description: finding.description,
                    location: finding.location,
                    recommendation: finding.recommendation + "\n[K-S Validation: \(severityValidation.recommendation)]",
                    koopmanStability: adjustedConfidence,
                    spectralRadius: finding.spectralRadius,
                    conditionNumber: finding.conditionNumber,
                    reconstructionError: finding.reconstructionError,
                    dominantModes: finding.dominantModes,
                    exploitVector: finding.exploitVector,
                    impactAssessment: finding.impactAssessment + "\nStatistical Confidence: \(String(format: "%.3f", severityValidation.confidenceLevel))",
                    cvssScore: cvssScore
                )

                validatedFindings.append(validatedFinding)
            } else {
                validatedFindings.append(finding)
            }
        }

        return validatedFindings
    }

    /// Perform adversarial validation using bootstrap sampling
    /// - Parameters:
    ///   - syntheticData: Synthetic dataset
    ///   - realData: Real dataset
    ///   - nBootstrap: Number of bootstrap samples
    /// - Returns: Comprehensive validation results
    func adversarialValidation(
        syntheticData: [Double],
        realData: [Double],
        nBootstrap: Int = 100
    ) -> [String: Any] {

        var bootstrapStats: [Double] = []
        var bootstrapPValues: [Double] = []

        // Bootstrap sampling
        for _ in 0..<nBootstrap {
            let synthSample = sampleWithReplacement(data: syntheticData, size: syntheticData.count / 2)
            let realSample = sampleWithReplacement(data: realData, size: realData.count / 2)

            let result = twoSampleKSTest(syntheticData: synthSample, realData: realSample)
            bootstrapStats.append(result.ksStatistic)
            bootstrapPValues.append(result.pValue)
        }

        // Calculate stability metrics
        let meanKS = bootstrapStats.reduce(0, +) / Double(bootstrapStats.count)
        let stdKS = calculateStandardDeviation(data: bootstrapStats, mean: meanKS)
        let stabilityScore = 1.0 - stdKS

        let validProportion = Double(bootstrapPValues.filter { $0 > alpha }.count) / Double(bootstrapPValues.count)
        let overallScore = (stabilityScore + validProportion) / 2.0

        // Risk assessment
        let riskLevel = overallScore > 0.8 ? "LOW" :
                       overallScore > 0.6 ? "MEDIUM" : "HIGH"

        let recommendation = generateAdversarialRecommendation(overallScore, riskLevel)

        return [
            "mean_ks_statistic": meanKS,
            "ks_statistic_std": stdKS,
            "validation_stability": stabilityScore,
            "proportion_valid": validProportion,
            "overall_score": overallScore,
            "risk_level": riskLevel,
            "recommendation": recommendation,
            "bootstrap_samples": nBootstrap
        ]
    }

    // MARK: - Helper Methods

    private func severityToNumeric(_ severity: SeverityLevel) -> Double {
        switch severity {
        case .critical: return 5.0
        case .high: return 4.0
        case .medium: return 3.0
        case .low: return 2.0
        case .info: return 1.0
        }
    }

    private func sampleWithReplacement(data: [Double], size: Int) -> [Double] {
        var sample: [Double] = []
        for _ in 0..<size {
            let randomIndex = Int.random(in: 0..<data.count)
            sample.append(data[randomIndex])
        }
        return sample
    }

    private func calculateStandardDeviation(data: [Double], mean: Double) -> Double {
        let variance = data.map { pow($0 - mean, 2) }.reduce(0, +) / Double(data.count - 1)
        return sqrt(variance)
    }

    private func generateAdversarialRecommendation(_ score: Double, _ risk: String) -> String {
        if score > 0.8 && risk == "LOW" {
            return "OPTIMAL: Validation results maintain excellent fidelity"
        } else if score > 0.6 && risk == "MEDIUM" {
            return "ACCEPTABLE: Monitor for distribution drift"
        } else if risk == "MEDIUM" {
            return "CAUTION: Increase real data validation and implement grounding"
        } else {
            return "CRITICAL: High validation risk - prioritize real-world testing"
        }
    }

    /// Generate comprehensive validation report
    /// - Returns: Formatted validation report
    func generateValidationReport() -> String {
        var report = """
        K-S Penetration Testing Validation Report
        =======================================

        Generated: \(Date())
        Total Validations: \(validationHistory.count)

        Validation Summary:
        """

        let validCount = validationHistory.filter { $0.isValid }.count
        let avgConfidence = validationHistory.map { $0.confidenceLevel }.reduce(0, +) / Double(validationHistory.count)

        report += "\nValid Tests: \(validCount)/\(validationHistory.count)"
        report += "\nAverage Confidence: \(String(format: "%.3f", avgConfidence))"

        if !validationHistory.isEmpty {
            report += "\n\nRecent Validations:"
            for (index, result) in validationHistory.suffix(5).enumerated() {
                report += "\n\(index + 1). K-S: \(String(format: "%.4f", result.ksStatistic))"
                report += " | Confidence: \(String(format: "%.3f", result.confidenceLevel))"
                report += " | \(result.recommendation)"
            }
        }

        return report
    }
}
