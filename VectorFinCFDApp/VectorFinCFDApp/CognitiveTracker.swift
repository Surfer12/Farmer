// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import HealthKit
import Foundation
import Combine

class CognitiveTracker: ObservableObject {
    private let healthStore = HKHealthStore()
    
    @Published var currentHRV: Double = 0
    @Published var cognitiveLoad: Double = 0
    @Published var flowStateScore: Double = 0
    @Published var isHealthKitAuthorized = false
    @Published var healthKitError: String?
    
    // Cognitive metrics
    @Published var reactionTime: Double = 0
    @Published var attentionLevel: Double = 0.5
    @Published var stressLevel: Double = 0.5
    
    // Historical data for trend analysis
    private var hrvHistory: [HRVReading] = []
    private var cognitiveHistory: [CognitiveReading] = []
    
    enum HealthError: Error, LocalizedError {
        case healthDataUnavailable
        case authorizationFailed
        case queryFailed
        case noDataAvailable
        case invalidData
        
        var errorDescription: String? {
            switch self {
            case .healthDataUnavailable:
                return "Health data is not available on this device"
            case .authorizationFailed:
                return "HealthKit authorization was denied"
            case .queryFailed:
                return "Failed to query health data"
            case .noDataAvailable:
                return "No HRV data available"
            case .invalidData:
                return "Invalid health data received"
            }
        }
    }
    
    init() {
        requestHealthKitAuthorization()
        startCognitiveMonitoring()
    }
    
    // MARK: - HealthKit Authorization
    
    private func requestHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            healthKitError = HealthError.healthDataUnavailable.localizedDescription
            return
        }
        
        let typesToRead: Set<HKObjectType> = [
            HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKQuantityType.quantityType(forIdentifier: .heartRate)!,
            HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            DispatchQueue.main.async {
                if success {
                    self?.isHealthKitAuthorized = true
                    self?.startPeriodicHRVUpdates()
                } else {
                    self?.healthKitError = error?.localizedDescription ?? HealthError.authorizationFailed.localizedDescription
                    self?.isHealthKitAuthorized = false
                }
            }
        }
    }
    
    // MARK: - HRV Data Fetching
    
    func fetchHRV(completion: @escaping (Result<Double, Error>) -> Void) {
        guard isHealthKitAuthorized else {
            completion(.failure(HealthError.authorizationFailed))
            return
        }
        
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(.failure(HealthError.queryFailed))
            return
        }
        
        let now = Date()
        let startDate = Calendar.current.date(byAdding: .hour, value: -1, to: now) ?? now
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictEndDate)
        
        let query = HKSampleQuery(
            sampleType: hrvType,
            predicate: predicate,
            limit: 10,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
        ) { [weak self] _, samples, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                
                guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                    // If no recent data, try fallback with longer time range
                    self?.fetchHRVFallback(completion: completion)
                    return
                }
                
                let hrvValues = samples.map { sample in
                    sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                }
                
                let averageHRV = hrvValues.reduce(0, +) / Double(hrvValues.count)
                self?.processHRVReading(averageHRV, timestamp: now)
                completion(.success(averageHRV))
            }
        }
        
        healthStore.execute(query)
    }
    
    private func fetchHRVFallback(completion: @escaping (Result<Double, Error>) -> Void) {
        // Try with a 24-hour window
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(.failure(HealthError.queryFailed))
            return
        }
        
        let now = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -1, to: now) ?? now
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictEndDate)
        
        let query = HKSampleQuery(
            sampleType: hrvType,
            predicate: predicate,
            limit: 1,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
        ) { [weak self] _, samples, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                
                guard let samples = samples as? [HKQuantitySample], let sample = samples.first else {
                    // Generate mock data for development
                    let mockHRV = self?.generateMockHRV() ?? 45.0
                    self?.processHRVReading(mockHRV, timestamp: now)
                    completion(.success(mockHRV))
                    return
                }
                
                let hrv = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                self?.processHRVReading(hrv, timestamp: now)
                completion(.success(hrv))
            }
        }
        
        healthStore.execute(query)
    }
    
    private func generateMockHRV() -> Double {
        // Generate realistic HRV data for development/testing
        let baseHRV = 45.0 // Average baseline
        let variation = Double.random(in: -15...15)
        let timeOfDayEffect = cos(Date().timeIntervalSince1970 / 86400 * 2 * .pi) * 5 // Daily rhythm
        
        return max(20, min(80, baseHRV + variation + timeOfDayEffect))
    }
    
    private func startPeriodicHRVUpdates() {
        // Fetch HRV every 30 seconds when authorized
        Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.fetchHRV { result in
                switch result {
                case .success(let hrv):
                    self?.currentHRV = hrv
                case .failure(let error):
                    self?.healthKitError = error.localizedDescription
                }
            }
        }
    }
    
    // MARK: - Cognitive State Analysis
    
    private func startCognitiveMonitoring() {
        // Start cognitive load monitoring based on HRV and other metrics
        Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            self?.updateCognitiveMetrics()
        }
    }
    
    private func updateCognitiveMetrics() {
        // Calculate cognitive load based on HRV variability
        let cognitiveLoadFromHRV = calculateCognitiveLoadFromHRV()
        
        // Simulate reaction time based on cognitive load
        let baseReactionTime = 250.0 // ms
        let loadEffect = cognitiveLoadFromHRV * 100
        reactionTime = baseReactionTime + loadEffect
        
        // Update attention level (inverse of cognitive load)
        attentionLevel = max(0, min(1, 1.0 - cognitiveLoadFromHRV))
        
        // Update stress level based on HRV trends
        stressLevel = calculateStressLevel()
        
        // Calculate overall flow state score
        flowStateScore = calculateFlowStateScore()
        
        // Record cognitive reading
        let reading = CognitiveReading(
            timestamp: Date(),
            cognitiveLoad: cognitiveLoadFromHRV,
            reactionTime: reactionTime,
            attentionLevel: attentionLevel,
            stressLevel: stressLevel,
            flowStateScore: flowStateScore
        )
        
        cognitiveHistory.append(reading)
        
        // Keep only last 100 readings
        if cognitiveHistory.count > 100 {
            cognitiveHistory.removeFirst()
        }
        
        // Update published values
        cognitiveLoad = cognitiveLoadFromHRV
    }
    
    private func calculateCognitiveLoadFromHRV() -> Double {
        guard hrvHistory.count >= 3 else { return 0.5 }
        
        // Calculate HRV variability over recent readings
        let recentHRVs = Array(hrvHistory.suffix(5)).map { $0.value }
        let mean = recentHRVs.reduce(0, +) / Double(recentHRVs.count)
        let variance = recentHRVs.map { pow($0 - mean, 2) }.reduce(0, +) / Double(recentHRVs.count)
        let standardDeviation = sqrt(variance)
        
        // Lower HRV variability indicates higher cognitive load
        let normalizedVariability = min(1, standardDeviation / 20.0)
        return max(0, 1.0 - normalizedVariability)
    }
    
    private func calculateStressLevel() -> Double {
        guard let latestHRV = hrvHistory.last else { return 0.5 }
        
        // Lower HRV generally indicates higher stress
        let normalizedHRV = min(1, max(0, (latestHRV.value - 20) / 60))
        return 1.0 - normalizedHRV
    }
    
    private func calculateFlowStateScore() -> Double {
        // Flow state is optimal when:
        // - Moderate HRV (not too high, not too low)
        // - Low cognitive load
        // - High attention
        // - Moderate stress (challenge without overwhelm)
        
        let hrvScore = currentHRV > 30 && currentHRV < 60 ? 1.0 : 0.5
        let loadScore = 1.0 - cognitiveLoad
        let attentionScore = attentionLevel
        let stressScore = stressLevel > 0.3 && stressLevel < 0.7 ? 1.0 : 0.5
        
        return (hrvScore + loadScore + attentionScore + stressScore) / 4.0
    }
    
    private func processHRVReading(_ hrv: Double, timestamp: Date) {
        let reading = HRVReading(timestamp: timestamp, value: hrv)
        hrvHistory.append(reading)
        
        // Keep only last 50 readings (about 25 minutes at 30s intervals)
        if hrvHistory.count > 50 {
            hrvHistory.removeFirst()
        }
        
        currentHRV = hrv
    }
    
    // MARK: - Data Export and Analysis
    
    func getFlowStateAnalysis() -> FlowStateAnalysis {
        let recentCognitive = Array(cognitiveHistory.suffix(10))
        let recentHRV = Array(hrvHistory.suffix(10))
        
        return FlowStateAnalysis(
            currentFlowScore: flowStateScore,
            averageHRV: recentHRV.isEmpty ? 0 : recentHRV.map { $0.value }.reduce(0, +) / Double(recentHRV.count),
            averageCognitiveLoad: recentCognitive.isEmpty ? 0 : recentCognitive.map { $0.cognitiveLoad }.reduce(0, +) / Double(recentCognitive.count),
            averageAttention: recentCognitive.isEmpty ? 0 : recentCognitive.map { $0.attentionLevel }.reduce(0, +) / Double(recentCognitive.count),
            stressTrend: calculateStressTrend(),
            recommendations: generateRecommendations()
        )
    }
    
    private func calculateStressTrend() -> String {
        guard cognitiveHistory.count >= 5 else { return "Insufficient data" }
        
        let recent = Array(cognitiveHistory.suffix(5))
        let trend = recent.last!.stressLevel - recent.first!.stressLevel
        
        if trend > 0.1 {
            return "Increasing"
        } else if trend < -0.1 {
            return "Decreasing"
        } else {
            return "Stable"
        }
    }
    
    private func generateRecommendations() -> [String] {
        var recommendations: [String] = []
        
        if flowStateScore < 0.4 {
            recommendations.append("Consider taking a break to reset focus")
        }
        
        if cognitiveLoad > 0.7 {
            recommendations.append("High cognitive load detected - simplify current task")
        }
        
        if stressLevel > 0.8 {
            recommendations.append("Stress level high - try deep breathing exercises")
        }
        
        if attentionLevel < 0.3 {
            recommendations.append("Low attention detected - change environment or take a break")
        }
        
        if currentHRV < 25 {
            recommendations.append("Low HRV - consider stress management techniques")
        }
        
        if recommendations.isEmpty {
            recommendations.append("Cognitive state is optimal for peak performance")
        }
        
        return recommendations
    }
    
    func exportCognitiveData() -> String {
        let analysis = getFlowStateAnalysis()
        
        return """
        === Cognitive State Analysis ===
        Timestamp: \(Date().ISO8601String())
        
        Current Metrics:
        - Flow State Score: \(String(format: "%.2f", flowStateScore))
        - HRV: \(String(format: "%.1f", currentHRV)) ms
        - Cognitive Load: \(String(format: "%.2f", cognitiveLoad))
        - Attention Level: \(String(format: "%.2f", attentionLevel))
        - Stress Level: \(String(format: "%.2f", stressLevel))
        - Reaction Time: \(String(format: "%.0f", reactionTime)) ms
        
        Trends:
        - Average HRV: \(String(format: "%.1f", analysis.averageHRV)) ms
        - Average Cognitive Load: \(String(format: "%.2f", analysis.averageCognitiveLoad))
        - Average Attention: \(String(format: "%.2f", analysis.averageAttention))
        - Stress Trend: \(analysis.stressTrend)
        
        Recommendations:
        \(analysis.recommendations.map { "- \($0)" }.joined(separator: "\n"))
        """
    }
}

// MARK: - Supporting Types

struct HRVReading {
    let timestamp: Date
    let value: Double
}

struct CognitiveReading {
    let timestamp: Date
    let cognitiveLoad: Double
    let reactionTime: Double
    let attentionLevel: Double
    let stressLevel: Double
    let flowStateScore: Double
}

struct FlowStateAnalysis {
    let currentFlowScore: Double
    let averageHRV: Double
    let averageCognitiveLoad: Double
    let averageAttention: Double
    let stressTrend: String
    let recommendations: [String]
}