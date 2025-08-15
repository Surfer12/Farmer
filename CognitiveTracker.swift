import HealthKit
import Combine
import Foundation

class CognitiveTracker: ObservableObject {
    
    // MARK: - Published Properties
    @Published var hrv: Double? = nil
    @Published var heartRate: Double? = nil
    @Published var cognitiveLoad: Float = 0.0
    @Published var flowState: FlowState = .baseline
    @Published var stressLevel: Float = 0.0
    @Published var isAuthorized: Bool = false
    
    // MARK: - HealthKit
    private let healthStore = HKHealthStore()
    private var heartRateQuery: HKQuery?
    private var hrvQuery: HKQuery?
    
    // MARK: - Data Processing
    private var hrvHistory: [Double] = []
    private var heartRateHistory: [Double] = []
    private var cognitiveAnalyzer = CognitiveAnalyzer()
    
    // MARK: - Flow State Definitions
    enum FlowState: String, CaseIterable {
        case baseline = "Baseline"
        case focused = "Focused"
        case flow = "Flow State"
        case stressed = "Stressed"
        case fatigued = "Fatigued"
        
        var color: String {
            switch self {
            case .baseline: return "gray"
            case .focused: return "blue"
            case .flow: return "green"
            case .stressed: return "red"
            case .fatigued: return "orange"
            }
        }
        
        var description: String {
            switch self {
            case .baseline: return "Normal state"
            case .focused: return "High attention, ready for action"
            case .flow: return "Optimal performance zone"
            case .stressed: return "High cognitive load"
            case .fatigued: return "Reduced capacity"
            }
        }
    }
    
    // MARK: - Initialization
    init() {
        requestHealthKitAuthorization()
    }
    
    // MARK: - HealthKit Authorization
    private func requestHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("❌ HealthKit not available on this device")
            return
        }
        
        let typesToRead: Set<HKObjectType> = [
            HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKQuantityType.quantityType(forIdentifier: .heartRate)!,
            HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!,
            HKQuantityType.quantityType(forIdentifier: .walkingHeartRateAverage)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            DispatchQueue.main.async {
                self?.isAuthorized = success
                if success {
                    print("✅ HealthKit authorized")
                    self?.startContinuousMonitoring()
                } else {
                    print("❌ HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
    }
    
    // MARK: - Continuous Monitoring
    func startContinuousMonitoring() {
        guard isAuthorized else { return }
        
        startHRVMonitoring()
        startHeartRateMonitoring()
        
        // Start cognitive analysis timer
        Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.analyzeCognitiveState()
        }
    }
    
    func stopMonitoring() {
        if let query = heartRateQuery {
            healthStore.stop(query)
        }
        if let query = hrvQuery {
            healthStore.stop(query)
        }
    }
    
    // MARK: - HRV Monitoring
    private func startHRVMonitoring() {
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        
        let query = HKObserverQuery(sampleType: hrvType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("❌ HRV observer error: \(error)")
                return
            }
            self?.fetchLatestHRV()
        }
        
        hrvQuery = query
        healthStore.execute(query)
        
        // Initial fetch
        fetchLatestHRV()
    }
    
    func fetchHRV() {
        fetchLatestHRV()
    }
    
    private func fetchLatestHRV() {
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        
        let query = HKSampleQuery(
            sampleType: hrvType,
            predicate: nil,
            limit: 10,
            sortDescriptors: [sortDescriptor]
        ) { [weak self] _, samples, error in
            
            if let error = error {
                print("❌ HRV fetch error: \(error)")
                return
            }
            
            guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                print("⚠️ No HRV data available")
                return
            }
            
            DispatchQueue.main.async {
                // Get most recent HRV value
                let latestSample = samples[0]
                let hrvValue = latestSample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                self?.hrv = hrvValue
                
                // Add to history for analysis
                self?.hrvHistory.append(hrvValue)
                if self?.hrvHistory.count ?? 0 > 50 {
                    self?.hrvHistory.removeFirst()
                }
                
                print("📊 HRV: \(hrvValue) ms")
            }
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Heart Rate Monitoring
    private func startHeartRateMonitoring() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        
        let query = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                print("❌ Heart rate observer error: \(error)")
                return
            }
            self?.fetchLatestHeartRate()
        }
        
        heartRateQuery = query
        healthStore.execute(query)
        
        // Initial fetch
        fetchLatestHeartRate()
    }
    
    private func fetchLatestHeartRate() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 10,
            sortDescriptors: [sortDescriptor]
        ) { [weak self] _, samples, error in
            
            if let error = error {
                print("❌ Heart rate fetch error: \(error)")
                return
            }
            
            guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else { return }
            
            DispatchQueue.main.async {
                let latestSample = samples[0]
                let heartRateValue = latestSample.quantity.doubleValue(for: HKUnit(from: "count/min"))
                self?.heartRate = heartRateValue
                
                // Add to history
                self?.heartRateHistory.append(heartRateValue)
                if self?.heartRateHistory.count ?? 0 > 50 {
                    self?.heartRateHistory.removeFirst()
                }
            }
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Cognitive State Analysis
    private func analyzeCognitiveState() {
        guard let currentHRV = hrv, let currentHR = heartRate else { return }
        
        let analysis = cognitiveAnalyzer.analyze(
            hrv: currentHRV,
            heartRate: currentHR,
            hrvHistory: hrvHistory,
            heartRateHistory: heartRateHistory
        )
        
        DispatchQueue.main.async {
            self.cognitiveLoad = analysis.cognitiveLoad
            self.flowState = analysis.flowState
            self.stressLevel = analysis.stressLevel
        }
    }
    
    // MARK: - Flow State Optimization
    func getFlowStateRecommendations() -> [String] {
        switch flowState {
        case .baseline:
            return [
                "Ready for action - consider warming up",
                "Good time to review technique",
                "Prepare for optimal conditions"
            ]
        case .focused:
            return [
                "Excellent focus - perfect for technical maneuvers",
                "High attention state detected",
                "Consider challenging conditions"
            ]
        case .flow:
            return [
                "🎯 FLOW STATE ACHIEVED!",
                "Optimal performance zone",
                "Trust your instincts and ride the wave"
            ]
        case .stressed:
            return [
                "High stress detected - consider easier conditions",
                "Focus on breathing and relaxation",
                "Take breaks between sessions"
            ]
        case .fatigued:
            return [
                "Fatigue detected - consider rest",
                "Hydrate and recover",
                "Shorter session recommended"
            ]
        }
    }
    
    func getCognitiveMetrics() -> CognitiveMetrics {
        return CognitiveMetrics(
            hrv: hrv ?? 0,
            heartRate: heartRate ?? 0,
            cognitiveLoad: cognitiveLoad,
            stressLevel: stressLevel,
            flowState: flowState,
            recommendations: getFlowStateRecommendations()
        )
    }
}

// MARK: - Cognitive Analyzer
class CognitiveAnalyzer {
    
    struct CognitiveAnalysis {
        let cognitiveLoad: Float
        let flowState: CognitiveTracker.FlowState
        let stressLevel: Float
    }
    
    // Baseline values for comparison (would be personalized over time)
    private var baselineHRV: Double = 45.0 // ms
    private var baselineHR: Double = 70.0 // bpm
    
    func analyze(hrv: Double, heartRate: Double, hrvHistory: [Double], heartRateHistory: [Double]) -> CognitiveAnalysis {
        
        // Calculate relative metrics
        let hrvRatio = Float(hrv / baselineHRV)
        let hrRatio = Float(heartRate / baselineHR)
        
        // HRV trend analysis
        let hrvTrend = calculateTrend(data: hrvHistory)
        let hrTrend = calculateTrend(data: heartRateHistory)
        
        // Cognitive load calculation (inverse relationship with HRV)
        let cognitiveLoad = calculateCognitiveLoad(hrvRatio: hrvRatio, hrRatio: hrRatio)
        
        // Stress level (combination of HR elevation and HRV reduction)
        let stressLevel = calculateStressLevel(hrvRatio: hrvRatio, hrRatio: hrRatio)
        
        // Flow state determination
        let flowState = determineFlowState(
            cognitiveLoad: cognitiveLoad,
            stressLevel: stressLevel,
            hrvRatio: hrvRatio,
            hrRatio: hrRatio,
            hrvTrend: hrvTrend
        )
        
        return CognitiveAnalysis(
            cognitiveLoad: cognitiveLoad,
            flowState: flowState,
            stressLevel: stressLevel
        )
    }
    
    private func calculateCognitiveLoad(hrvRatio: Float, hrRatio: Float) -> Float {
        // Higher cognitive load = lower HRV + higher HR
        let hrvComponent = max(0, 1.0 - hrvRatio) * 0.6
        let hrComponent = max(0, hrRatio - 1.0) * 0.4
        
        return min(1.0, hrvComponent + hrComponent)
    }
    
    private func calculateStressLevel(hrvRatio: Float, hrRatio: Float) -> Float {
        // Stress indicators: low HRV + elevated HR
        let stressFromHRV = max(0, 1.0 - hrvRatio)
        let stressFromHR = max(0, hrRatio - 1.1) // 10% above baseline
        
        return min(1.0, (stressFromHRV * 0.7) + (stressFromHR * 0.3))
    }
    
    private func determineFlowState(
        cognitiveLoad: Float,
        stressLevel: Float,
        hrvRatio: Float,
        hrRatio: Float,
        hrvTrend: Float
    ) -> CognitiveTracker.FlowState {
        
        // Flow state criteria (based on research)
        if hrvRatio > 1.1 && cognitiveLoad < 0.3 && stressLevel < 0.2 && hrvTrend > 0 {
            return .flow
        }
        
        // Focused state
        if hrvRatio > 0.9 && cognitiveLoad < 0.5 && stressLevel < 0.4 {
            return .focused
        }
        
        // Stressed state
        if stressLevel > 0.6 || cognitiveLoad > 0.7 {
            return .stressed
        }
        
        // Fatigued state
        if hrvRatio < 0.7 && hrRatio < 0.9 && hrvTrend < -0.1 {
            return .fatigued
        }
        
        // Default baseline
        return .baseline
    }
    
    private func calculateTrend(data: [Double]) -> Float {
        guard data.count >= 3 else { return 0 }
        
        let recent = Array(data.suffix(5))
        let older = Array(data.prefix(max(1, data.count - 5)))
        
        let recentAvg = recent.reduce(0, +) / Double(recent.count)
        let olderAvg = older.reduce(0, +) / Double(older.count)
        
        return Float((recentAvg - olderAvg) / olderAvg)
    }
}

// MARK: - Supporting Data Structures
struct CognitiveMetrics {
    let hrv: Double
    let heartRate: Double
    let cognitiveLoad: Float
    let stressLevel: Float
    let flowState: CognitiveTracker.FlowState
    let recommendations: [String]
    
    var flowStateScore: Float {
        switch flowState {
        case .flow: return 1.0
        case .focused: return 0.8
        case .baseline: return 0.5
        case .stressed: return 0.2
        case .fatigued: return 0.1
        }
    }
}

// MARK: - Extensions for Integration
extension CognitiveTracker {
    
    func correlateWithSurfingMetrics(liftDrag: (lift: Float, drag: Float)?, angleOfAttack: Float) -> SurfingCorrelation {
        let metrics = getCognitiveMetrics()
        
        // Calculate performance correlation
        let performanceScore = calculatePerformanceScore(liftDrag: liftDrag, aoa: angleOfAttack)
        let cognitiveScore = metrics.flowStateScore
        
        let correlation = abs(performanceScore - cognitiveScore)
        let isOptimal = correlation < 0.2 && cognitiveScore > 0.7
        
        return SurfingCorrelation(
            performanceScore: performanceScore,
            cognitiveScore: cognitiveScore,
            correlation: 1.0 - correlation, // Invert for positive correlation
            isOptimalZone: isOptimal,
            recommendations: generatePerformanceRecommendations(
                performance: performanceScore,
                cognitive: cognitiveScore
            )
        )
    }
    
    private func calculatePerformanceScore(liftDrag: (lift: Float, drag: Float)?, angleOfAttack: Float) -> Float {
        guard let liftDrag = liftDrag else { return 0.5 }
        
        let liftToDragRatio = liftDrag.drag > 0.001 ? liftDrag.lift / liftDrag.drag : 0
        let optimalAOA: Float = 8.0 // Optimal angle for Vector 3/2
        let aoaScore = 1.0 - abs(angleOfAttack - optimalAOA) / 20.0
        
        // Normalize L/D ratio (typical range 5-15 for fins)
        let ldScore = min(1.0, max(0.0, (liftToDragRatio - 5.0) / 10.0))
        
        return max(0, min(1.0, (ldScore * 0.7) + (aoaScore * 0.3)))
    }
    
    private func generatePerformanceRecommendations(performance: Float, cognitive: Float) -> [String] {
        var recommendations: [String] = []
        
        if cognitive > 0.8 && performance < 0.6 {
            recommendations.append("Great focus! Try adjusting fin angle for better performance")
        } else if cognitive < 0.4 && performance > 0.7 {
            recommendations.append("Good technique, but consider relaxation for flow state")
        } else if cognitive > 0.7 && performance > 0.7 {
            recommendations.append("🔥 Perfect sync! You're in the optimal zone")
        } else {
            recommendations.append("Focus on both technique and mindset")
        }
        
        return recommendations
    }
}

struct SurfingCorrelation {
    let performanceScore: Float
    let cognitiveScore: Float
    let correlation: Float
    let isOptimalZone: Bool
    let recommendations: [String]
}