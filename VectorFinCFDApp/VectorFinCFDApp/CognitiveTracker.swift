import HealthKit
import Foundation
import Combine

class CognitiveTracker: ObservableObject {
    private let healthStore = HKHealthStore()
    
    @Published var isHealthKitAvailable = false
    @Published var lastHRVReading: Double?
    @Published var hrvHistory: [HRVReading] = []
    @Published var cognitiveLoadScore: Double = 0.0
    
    // Cognitive load thresholds based on research
    private let optimalHRVThreshold: Double = 50.0
    private let moderateHRVThreshold: Double = 30.0
    private let maxHRVHistory = 100
    
    enum HealthError: Error, LocalizedError {
        case healthDataUnavailable
        case queryFailed
        case noDataAvailable
        case permissionDenied
        
        var errorDescription: String? {
            switch self {
            case .healthDataUnavailable:
                return "HealthKit is not available on this device"
            case .queryFailed:
                return "Failed to query health data"
            case .noDataAvailable:
                return "No HRV data available"
            case .permissionDenied:
                return "HealthKit permission denied"
            }
        }
    }
    
    init() {
        checkHealthKitAvailability()
    }
    
    // MARK: - HealthKit Setup
    
    private func checkHealthKitAvailability() {
        isHealthKitAvailable = HKHealthStore.isHealthDataAvailable()
        
        if isHealthKitAvailable {
            requestHealthKitPermissions()
        }
    }
    
    private func requestHealthKitPermissions() {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            print("HRV type not available")
            return
        }
        
        let typesToRead: Set<HKSampleType> = [hrvType]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            DispatchQueue.main.async {
                if success {
                    print("HealthKit permissions granted")
                } else {
                    print("HealthKit permissions denied: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
    }
    
    // MARK: - HRV Data Fetching
    
    func fetchHRV(completion: @escaping (Result<Double, Error>) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(.failure(HealthError.healthDataUnavailable))
            return
        }
        
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(.failure(HealthError.queryFailed))
            return
        }
        
        // Check authorization status
        let authStatus = healthStore.authorizationStatus(for: hrvType)
        guard authStatus == .sharingAuthorized else {
            completion(.failure(HealthError.permissionDenied))
            return
        }
        
        // Create query for most recent HRV reading
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrvType, predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(.failure(HealthError.noDataAvailable))
                return
            }
            
            let hrv = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            
            DispatchQueue.main.async {
                self?.processHRVReading(hrv, timestamp: sample.endDate)
            }
            
            completion(.success(hrv))
        }
        
        healthStore.execute(query)
    }
    
    func fetchHRVHistory(completion: @escaping (Result<[HRVReading], Error>) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(.failure(HealthError.healthDataUnavailable))
            return
        }
        
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(.failure(HealthError.queryFailed))
            return
        }
        
        // Query last 24 hours of HRV data
        let calendar = Calendar.current
        let now = Date()
        let startDate = calendar.date(byAdding: .day, value: -1, to: now) ?? now
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictStartDate)
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: true)
        
        let query = HKSampleQuery(sampleType: hrvType, predicate: predicate, limit: maxHRVHistory, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            let readings = samples?.compactMap { sample -> HRVReading? in
                guard let quantitySample = sample as? HKQuantitySample else { return nil }
                let hrv = quantitySample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                return HRVReading(value: hrv, timestamp: quantitySample.endDate)
            } ?? []
            
            DispatchQueue.main.async {
                self?.hrvHistory = readings
            }
            
            completion(.success(readings))
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Cognitive Load Analysis
    
    private func processHRVReading(_ hrv: Double, timestamp: Date) {
        lastHRVReading = hrv
        
        let reading = HRVReading(value: hrv, timestamp: timestamp)
        hrvHistory.append(reading)
        
        // Keep only recent readings
        if hrvHistory.count > maxHRVHistory {
            hrvHistory.removeFirst(hrvHistory.count - maxHRVHistory)
        }
        
        // Update cognitive load score
        updateCognitiveLoadScore()
    }
    
    private func updateCognitiveLoadScore() {
        guard !hrvHistory.isEmpty else {
            cognitiveLoadScore = 0.0
            return
        }
        
        // Calculate cognitive load based on HRV patterns
        let recentReadings = Array(hrvHistory.suffix(10)) // Last 10 readings
        let avgHRV = recentReadings.map { $0.value }.reduce(0, +) / Double(recentReadings.count)
        
        // HRV variability indicates cognitive load
        let hrvVariance = calculateHRVVariance(recentReadings)
        let normalizedVariance = min(hrvVariance / 100.0, 1.0) // Normalize to 0-1
        
        // Lower HRV and higher variance indicate higher cognitive load
        let hrvScore = max(0, (avgHRV - moderateHRVThreshold) / (optimalHRVThreshold - moderateHRVThreshold))
        let varianceScore = 1.0 - normalizedVariance
        
        // Combined cognitive load score (0 = low load, 1 = high load)
        cognitiveLoadScore = (hrvScore + varianceScore) / 2.0
    }
    
    private func calculateHRVVariance(_ readings: [HRVReading]) -> Double {
        guard readings.count > 1 else { return 0.0 }
        
        let values = readings.map { $0.value }
        let mean = values.reduce(0, +) / Double(values.count)
        let squaredDifferences = values.map { pow($0 - mean, 2) }
        let variance = squaredDifferences.reduce(0, +) / Double(values.count)
        
        return variance
    }
    
    // MARK: - Flow State Assessment
    
    func assessFlowState() -> FlowStateAssessment {
        guard let lastHRV = lastHRVReading else {
            return FlowStateAssessment(
                state: .unknown,
                confidence: 0.0,
                recommendations: ["No HRV data available"]
            )
        }
        
        var state: FlowState
        var confidence: Double
        var recommendations: [String] = []
        
        // Assess flow state based on HRV and cognitive load
        if lastHRV > optimalHRVThreshold && cognitiveLoadScore < 0.3 {
            state = .optimal
            confidence = 0.9
            recommendations = [
                "Excellent flow state maintained",
                "Continue current technique",
                "Consider pushing performance boundaries"
            ]
        } else if lastHRV > moderateHRVThreshold && cognitiveLoadScore < 0.5 {
            state = .good
            confidence = 0.7
            recommendations = [
                "Good flow state",
                "Focus on breathing rhythm",
                "Maintain smooth movements"
            ]
        } else if lastHRV > 20.0 && cognitiveLoadScore < 0.7 {
            state = .moderate
            confidence = 0.5
            recommendations = [
                "Moderate flow state",
                "Take deep breaths",
                "Simplify technique",
                "Focus on fundamentals"
            ]
        } else {
            state = .challenging
            confidence = 0.3
            recommendations = [
                "Flow state compromised",
                "Take a moment to reset",
                "Focus on breathing",
                "Consider technique adjustments"
            ]
        }
        
        return FlowStateAssessment(
            state: state,
            confidence: confidence,
            recommendations: recommendations
        )
    }
    
    // MARK: - Data Export
    
    func exportCognitiveData() -> String {
        let timestamp = Date().ISO8601String()
        let hrvStr = lastHRVReading?.description ?? "N/A"
        let avgHRV = hrvHistory.isEmpty ? 0.0 : hrvHistory.map { $0.value }.reduce(0, +) / Double(hrvHistory.count)
        
        return """
        Cognitive Performance Data
        ==========================
        Timestamp: \(timestamp)
        Current HRV: \(hrvStr) ms
        Average HRV: \(String(format: "%.1f", avgHRV)) ms
        Cognitive Load Score: \(String(format: "%.2f", cognitiveLoadScore))
        HRV History Count: \(hrvHistory.count)
        Flow State Assessment: \(assessFlowState().state.rawValue)
        """
    }
}

// MARK: - Supporting Types

struct HRVReading: Identifiable {
    let id = UUID()
    let value: Double
    let timestamp: Date
    
    var formattedValue: String {
        return String(format: "%.1f ms", value)
    }
    
    var timeAgo: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: timestamp, relativeTo: Date())
    }
}

enum FlowState: String, CaseIterable {
    case optimal = "Optimal Flow"
    case good = "Good Flow"
    case moderate = "Moderate Flow"
    case challenging = "Challenging"
    case unknown = "Unknown"
    
    var description: String {
        switch self {
        case .optimal:
            return "Peak performance state with excellent cognitive efficiency"
        case .good:
            return "Strong performance with good cognitive control"
        case .moderate:
            return "Adequate performance with room for improvement"
        case .challenging:
            return "Performance may be compromised by cognitive load"
        case .unknown:
            return "Insufficient data for assessment"
        }
    }
    
    var color: String {
        switch self {
        case .optimal: return "green"
        case .good: return "blue"
        case .moderate: return "yellow"
        case .challenging: return "red"
        case .unknown: return "gray"
        }
    }
}

struct FlowStateAssessment {
    let state: FlowState
    let confidence: Double
    let recommendations: [String]
    
    var summary: String {
        return "\(state.rawValue) (Confidence: \(String(format: "%.0f", confidence * 100))%)"
    }
}

// MARK: - Extensions

extension Date {
    func ISO8601String() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}