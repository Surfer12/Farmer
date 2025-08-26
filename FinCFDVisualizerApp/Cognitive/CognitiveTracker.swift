import HealthKit
import Foundation
import Combine

// MARK: - Cognitive Tracker

class CognitiveTracker: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    private let healthStore = HKHealthStore()
    private var cancellables = Set<AnyCancellable>()
    
    // Published Properties
    @Published var currentHRV: Double = 0.0
    @Published var heartRate: Double = 0.0
    @Published var cognitiveMetrics: CognitiveMetrics?
    @Published var flowStateIndex: Double = 0.0
    @Published var isHealthKitAuthorized = false
    @Published var reactionTime: Double = 0.0
    
    // Data Collection
    private var hrvHistory: [HRVReading] = []
    private var heartRateHistory: [HeartRateReading] = []
    private var cognitiveHistory: [CognitiveMetrics] = []
    
    // Flow State Analysis
    private let flowStateAnalyzer = FlowStateAnalyzer()
    private var reactionTimeTracker = ReactionTimeTracker()
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        requestHealthKitAuthorization()
        setupDataStreams()
    }
    
    // MARK: - HealthKit Authorization
    
    private func requestHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("❌ HealthKit not available on this device")
            return
        }
        
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            DispatchQueue.main.async {
                if success {
                    self?.isHealthKitAuthorized = true
                    print("✅ HealthKit authorization granted")
                    self?.startHealthKitMonitoring()
                } else {
                    print("❌ HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
    }
    
    // MARK: - Health Data Monitoring
    
    private func startHealthKitMonitoring() {
        startHRVMonitoring()
        startHeartRateMonitoring()
        startCognitiveAnalysis()
    }
    
    private func startHRVMonitoring() {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            print("❌ HRV type not available")
            return
        }
        
        let query = HKObserverQuery(sampleType: hrvType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("❌ HRV observer error: \(error)")
                return
            }
            
            self?.fetchLatestHRV()
        }
        
        healthStore.execute(query)
        
        // Also fetch initial HRV data
        fetchLatestHRV()
    }
    
    private func fetchLatestHRV() {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return }
        
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: hrvType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sortDescriptor]
        ) { [weak self] _, samples, error in
            
            if let error = error {
                print("❌ HRV fetch error: \(error)")
                return
            }
            
            guard let sample = samples?.first as? HKQuantitySample else {
                // Generate mock HRV data for testing
                self?.generateMockHRVData()
                return
            }
            
            let hrvValue = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            let reading = HRVReading(timestamp: sample.endDate, value: hrvValue)
            
            DispatchQueue.main.async {
                self?.processHRVReading(reading)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func startHeartRateMonitoring() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            print("❌ Heart rate type not available")
            return
        }
        
        let query = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("❌ Heart rate observer error: \(error)")
                return
            }
            
            self?.fetchLatestHeartRate()
        }
        
        healthStore.execute(query)
        fetchLatestHeartRate()
    }
    
    private func fetchLatestHeartRate() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sortDescriptor]
        ) { [weak self] _, samples, error in
            
            if let error = error {
                print("❌ Heart rate fetch error: \(error)")
                return
            }
            
            guard let sample = samples?.first as? HKQuantitySample else {
                // Generate mock heart rate data
                self?.generateMockHeartRateData()
                return
            }
            
            let heartRateValue = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            let reading = HeartRateReading(timestamp: sample.endDate, value: heartRateValue)
            
            DispatchQueue.main.async {
                self?.processHeartRateReading(reading)
            }
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Data Processing
    
    private func processHRVReading(_ reading: HRVReading) {
        currentHRV = reading.value
        hrvHistory.append(reading)
        
        // Keep only last 100 readings
        if hrvHistory.count > 100 {
            hrvHistory.removeFirst()
        }
        
        updateCognitiveMetrics()
    }
    
    private func processHeartRateReading(_ reading: HeartRateReading) {
        heartRate = reading.value
        heartRateHistory.append(reading)
        
        // Keep only last 100 readings
        if heartRateHistory.count > 100 {
            heartRateHistory.removeFirst()
        }
        
        updateCognitiveMetrics()
    }
    
    private func updateCognitiveMetrics() {
        let newMetrics = CognitiveMetrics(
            timestamp: Date(),
            heartRateVariability: currentHRV,
            reactionTime: reactionTime,
            flowStateIndex: flowStateAnalyzer.calculateFlowState(hrv: currentHRV, heartRate: heartRate),
            cognitiveLoad: calculateCognitiveLoad()
        )
        
        cognitiveMetrics = newMetrics
        flowStateIndex = newMetrics.flowStateIndex
        cognitiveHistory.append(newMetrics)
        
        // Keep only last 50 cognitive metrics
        if cognitiveHistory.count > 50 {
            cognitiveHistory.removeFirst()
        }
    }
    
    private func calculateCognitiveLoad() -> Double {
        // Calculate cognitive load based on HRV and heart rate variability
        guard !hrvHistory.isEmpty, !heartRateHistory.isEmpty else { return 0.5 }
        
        let recentHRV = Array(hrvHistory.suffix(10))
        let recentHR = Array(heartRateHistory.suffix(10))
        
        // Calculate HRV coefficient of variation
        let hrvMean = recentHRV.map { $0.value }.reduce(0, +) / Double(recentHRV.count)
        let hrvStdDev = sqrt(recentHRV.map { pow($0.value - hrvMean, 2) }.reduce(0, +) / Double(recentHRV.count))
        let hrvCV = hrvStdDev / hrvMean
        
        // Calculate heart rate variability
        let hrMean = recentHR.map { $0.value }.reduce(0, +) / Double(recentHR.count)
        let hrStdDev = sqrt(recentHR.map { pow($0.value - hrMean, 2) }.reduce(0, +) / Double(recentHR.count))
        let hrCV = hrStdDev / hrMean
        
        // Cognitive load inversely related to HRV and directly related to HR variability
        let cognitiveLoad = max(0, min(1, (1.0 - hrvCV) * 0.7 + hrCV * 0.3))
        
        return cognitiveLoad
    }
    
    // MARK: - Cognitive Analysis
    
    private func startCognitiveAnalysis() {
        // Start reaction time monitoring
        reactionTimeTracker.startMonitoring { [weak self] reactionTime in
            DispatchQueue.main.async {
                self?.reactionTime = reactionTime
            }
        }
        
        // Periodic cognitive analysis
        Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.performCognitiveAnalysis()
        }
    }
    
    private func performCognitiveAnalysis() {
        guard let currentMetrics = cognitiveMetrics else { return }
        
        // Analyze flow state trends
        let flowTrend = analyzeFlowStateTrend()
        
        // Generate recommendations
        let recommendations = generateCognitiveRecommendations(
            metrics: currentMetrics,
            trend: flowTrend
        )
        
        // Post notifications if needed
        if currentMetrics.flowStateIndex > 0.8 {
            postFlowStateNotification(type: .optimal)
        } else if currentMetrics.cognitiveLoad > 0.8 {
            postFlowStateNotification(type: .overload)
        }
    }
    
    private func analyzeFlowStateTrend() -> FlowStateTrend {
        guard cognitiveHistory.count >= 5 else { return .stable }
        
        let recentFlow = Array(cognitiveHistory.suffix(5)).map { $0.flowStateIndex }
        let trend = (recentFlow.last! - recentFlow.first!) / 4.0
        
        if trend > 0.1 {
            return .improving
        } else if trend < -0.1 {
            return .declining
        } else {
            return .stable
        }
    }
    
    private func generateCognitiveRecommendations(
        metrics: CognitiveMetrics,
        trend: FlowStateTrend
    ) -> [CognitiveRecommendation] {
        
        var recommendations: [CognitiveRecommendation] = []
        
        // HRV-based recommendations
        if metrics.heartRateVariability < 20 {
            recommendations.append(.breathingExercise)
        }
        
        // Flow state recommendations
        if metrics.flowStateIndex < 0.3 {
            recommendations.append(.relaxation)
        } else if metrics.flowStateIndex > 0.9 {
            recommendations.append(.maintainFocus)
        }
        
        // Cognitive load recommendations
        if metrics.cognitiveLoad > 0.7 {
            recommendations.append(.reduceComplexity)
        }
        
        // Trend-based recommendations
        switch trend {
        case .declining:
            recommendations.append(.takeBreak)
        case .improving:
            recommendations.append(.maintainFocus)
        case .stable:
            break
        }
        
        return recommendations
    }
    
    // MARK: - Mock Data Generation (for testing)
    
    private func generateMockHRVData() {
        // Generate realistic HRV data for testing
        let baseHRV = 35.0 + Double.random(in: -10...10)
        let reading = HRVReading(timestamp: Date(), value: baseHRV)
        
        DispatchQueue.main.async { [weak self] in
            self?.processHRVReading(reading)
        }
    }
    
    private func generateMockHeartRateData() {
        // Generate realistic heart rate data
        let baseHR = 75.0 + Double.random(in: -15...15)
        let reading = HeartRateReading(timestamp: Date(), value: baseHR)
        
        DispatchQueue.main.async { [weak self] in
            self?.processHeartRateReading(reading)
        }
    }
    
    func startMockDataGeneration() {
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.generateMockHRVData()
            self?.generateMockHeartRateData()
        }
        
        print("🔄 Mock cognitive data generation started")
    }
    
    // MARK: - Data Access Methods
    
    func getFlowStateHistory() -> [CognitiveMetrics] {
        return cognitiveHistory
    }
    
    func getHRVStatistics() -> HRVStatistics {
        guard !hrvHistory.isEmpty else { return HRVStatistics.empty }
        
        let values = hrvHistory.map { $0.value }
        return HRVStatistics(
            average: values.reduce(0, +) / Double(values.count),
            minimum: values.min() ?? 0,
            maximum: values.max() ?? 0,
            standardDeviation: calculateStandardDeviation(values),
            readingCount: values.count
        )
    }
    
    private func calculateStandardDeviation(_ values: [Double]) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
        return sqrt(variance)
    }
    
    // MARK: - Notifications
    
    private func postFlowStateNotification(type: FlowStateNotificationType) {
        let notification = Notification.Name("FlowStateUpdate")
        NotificationCenter.default.post(name: notification, object: type)
    }
    
    // MARK: - Data Export
    
    func exportCognitiveData() -> Data? {
        let exportData = CognitiveExportData(
            cognitiveHistory: cognitiveHistory,
            hrvHistory: hrvHistory,
            heartRateHistory: heartRateHistory,
            exportTimestamp: Date()
        )
        
        do {
            return try JSONEncoder().encode(exportData)
        } catch {
            print("❌ Failed to export cognitive data: \(error)")
            return nil
        }
    }
}

// MARK: - Flow State Analyzer

class FlowStateAnalyzer {
    
    func calculateFlowState(hrv: Double, heartRate: Double) -> Double {
        // Flow state calculation based on HRV and heart rate
        // Optimal flow state typically occurs with:
        // - Higher HRV (better autonomic balance)
        // - Moderate heart rate (not too high or low)
        
        let normalizedHRV = min(1.0, max(0.0, (hrv - 10.0) / 50.0)) // Normalize HRV (10-60ms range)
        let normalizedHR = 1.0 - abs(heartRate - 75.0) / 50.0 // Optimal around 75 bpm
        let normalizedHRClamped = min(1.0, max(0.0, normalizedHR))
        
        // Weighted combination (HRV is more important for flow state)
        let flowState = (normalizedHRV * 0.7) + (normalizedHRClamped * 0.3)
        
        return min(1.0, max(0.0, flowState))
    }
}

// MARK: - Reaction Time Tracker

class ReactionTimeTracker {
    
    private var startTime: Date?
    private var reactionTimes: [Double] = []
    
    func startMonitoring(callback: @escaping (Double) -> Void) {
        // Simulate reaction time measurement
        Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in
            let reactionTime = self.simulateReactionTime()
            callback(reactionTime)
        }
    }
    
    private func simulateReactionTime() -> Double {
        // Simulate reaction time between 200-800ms
        let baseTime = 400.0 // Base reaction time in ms
        let variation = Double.random(in: -200...400)
        let reactionTime = max(200, baseTime + variation)
        
        reactionTimes.append(reactionTime)
        
        // Keep only last 20 measurements
        if reactionTimes.count > 20 {
            reactionTimes.removeFirst()
        }
        
        return reactionTime
    }
    
    func getAverageReactionTime() -> Double {
        guard !reactionTimes.isEmpty else { return 0 }
        return reactionTimes.reduce(0, +) / Double(reactionTimes.count)
    }
}

// MARK: - Supporting Types

struct HRVReading: Codable {
    let timestamp: Date
    let value: Double // milliseconds
}

struct HeartRateReading: Codable {
    let timestamp: Date
    let value: Double // beats per minute
}

struct HRVStatistics {
    let average: Double
    let minimum: Double
    let maximum: Double
    let standardDeviation: Double
    let readingCount: Int
    
    static let empty = HRVStatistics(
        average: 0, minimum: 0, maximum: 0, standardDeviation: 0, readingCount: 0
    )
}

enum FlowStateTrend {
    case improving
    case declining
    case stable
}

enum CognitiveRecommendation: String, CaseIterable {
    case breathingExercise = "Practice deep breathing"
    case relaxation = "Take a moment to relax"
    case maintainFocus = "Great focus - keep it up!"
    case reduceComplexity = "Simplify your approach"
    case takeBreak = "Consider taking a short break"
    
    var description: String {
        return self.rawValue
    }
    
    var icon: String {
        switch self {
        case .breathingExercise: return "lungs"
        case .relaxation: return "leaf"
        case .maintainFocus: return "target"
        case .reduceComplexity: return "minus.circle"
        case .takeBreak: return "pause.circle"
        }
    }
}

enum FlowStateNotificationType {
    case optimal
    case overload
    case improving
    case declining
}

struct CognitiveExportData: Codable {
    let cognitiveHistory: [CognitiveMetrics]
    let hrvHistory: [HRVReading]
    let heartRateHistory: [HeartRateReading]
    let exportTimestamp: Date
}