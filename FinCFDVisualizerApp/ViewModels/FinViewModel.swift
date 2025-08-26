import SwiftUI
import Combine
import Foundation

// MARK: - Main ViewModel

class FinViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var turnAngle: Double = 0.0
    @Published var liftDrag: LiftDragPrediction?
    @Published var pressureData: [Double] = []
    @Published var flowStateIndex: Double = 0.0
    @Published var currentHRV: Double = 0.0
    @Published var heartRate: Double = 0.0
    @Published var reactionTime: Double = 0.0
    @Published var isProcessing = false
    
    // MARK: - Components
    
    let visualizer = FinVisualizer()
    let predictor = FinPredictor()
    let sensorManager = SensorManager()
    let cognitiveTracker = CognitiveTracker()
    
    // MARK: - Private Properties
    
    private var cancellables = Set<AnyCancellable>()
    private let finSpecification = FinSpecification.vector32SideFin
    private var predictionTimer: Timer?
    
    // Data pipeline settings
    private let updateInterval: TimeInterval = 0.1
    private let predictionInterval: TimeInterval = 1.0
    
    // MARK: - Initialization
    
    init() {
        setupDataPipeline()
        setupPredictionEngine()
        setupVisualization()
    }
    
    // MARK: - Setup Methods
    
    private func setupDataPipeline() {
        // Combine sensor data streams
        setupSensorDataPipeline()
        setupCognitiveDataPipeline()
        setupVisualizationPipeline()
    }
    
    private func setupSensorDataPipeline() {
        // Turn angle from sensor manager
        sensorManager.$turnAngle
            .debounce(for: .seconds(updateInterval), scheduler: DispatchQueue.main)
            .sink { [weak self] angle in
                self?.turnAngle = angle
                self?.updateVisualizationAngle(angle)
                self?.triggerPrediction()
            }
            .store(in: &cancellables)
        
        // Pressure data from sensor manager
        sensorManager.$pressureData
            .debounce(for: .seconds(updateInterval), scheduler: DispatchQueue.main)
            .sink { [weak self] data in
                self?.pressureData = data
                self?.updatePressureVisualization(data)
            }
            .store(in: &cancellables)
    }
    
    private func setupCognitiveDataPipeline() {
        // HRV data from cognitive tracker
        cognitiveTracker.$currentHRV
            .debounce(for: .seconds(0.5), scheduler: DispatchQueue.main)
            .sink { [weak self] hrv in
                self?.currentHRV = hrv
            }
            .store(in: &cancellables)
        
        // Heart rate data
        cognitiveTracker.$heartRate
            .debounce(for: .seconds(0.5), scheduler: DispatchQueue.main)
            .sink { [weak self] hr in
                self?.heartRate = hr
            }
            .store(in: &cancellables)
        
        // Flow state index
        cognitiveTracker.$flowStateIndex
            .debounce(for: .seconds(0.5), scheduler: DispatchQueue.main)
            .sink { [weak self] flowState in
                self?.flowStateIndex = flowState
            }
            .store(in: &cancellables)
        
        // Reaction time
        cognitiveTracker.$reactionTime
            .debounce(for: .seconds(1.0), scheduler: DispatchQueue.main)
            .sink { [weak self] rt in
                self?.reactionTime = rt
            }
            .store(in: &cancellables)
    }
    
    private func setupVisualizationPipeline() {
        // Update visualization based on combined sensor and prediction data
        Publishers.CombineLatest3(
            $turnAngle,
            $pressureData,
            $liftDrag
        )
        .throttle(for: .seconds(updateInterval), scheduler: DispatchQueue.main, latest: true)
        .sink { [weak self] angle, pressure, prediction in
            self?.updateVisualization(angle: angle, pressure: pressure, prediction: prediction)
        }
        .store(in: &cancellables)
    }
    
    private func setupPredictionEngine() {
        // Monitor prediction results
        predictor.$lastPrediction
            .compactMap { $0 }
            .sink { [weak self] prediction in
                self?.liftDrag = prediction
            }
            .store(in: &cancellables)
    }
    
    private func setupVisualization() {
        // Setup the 3D fin model
        visualizer.setupFinModel()
    }
    
    // MARK: - Public Methods
    
    func startMonitoring() {
        // Start all monitoring systems
        sensorManager.startAllSensors()
        cognitiveTracker.startMockDataGeneration() // Use mock data for demo
        
        // Start prediction timer
        startPredictionTimer()
        
        print("✅ All monitoring systems started")
    }
    
    func stopMonitoring() {
        // Stop all monitoring systems
        sensorManager.stopAllSensors()
        predictionTimer?.invalidate()
        predictionTimer = nil
        
        print("🛑 All monitoring systems stopped")
    }
    
    func calibrateSensors() {
        sensorManager.calibrateIMU()
        print("🔧 Sensors calibrated")
    }
    
    func resetVisualization() {
        visualizer.resetVisualization()
        turnAngle = 0.0
        print("🔄 Visualization reset")
    }
    
    func exportData() {
        // Export all data
        exportSensorData()
        exportCognitiveData()
        exportPredictionData()
    }
    
    func clearDataBuffer() {
        sensorManager.clearBuffer()
        predictor.clearHistory()
        print("🗑️ Data buffers cleared")
    }
    
    func startMockDataGeneration() {
        sensorManager.startMockDataGeneration()
        cognitiveTracker.startMockDataGeneration()
        print("🔄 Mock data generation started")
    }
    
    // MARK: - Private Methods
    
    private func startPredictionTimer() {
        predictionTimer = Timer.scheduledTimer(withTimeInterval: predictionInterval, repeats: true) { [weak self] _ in
            self?.triggerPrediction()
        }
    }
    
    private func triggerPrediction() {
        guard !isProcessing else { return }
        
        Task { [weak self] in
            await self?.performPrediction()
        }
    }
    
    private func performPrediction() async {
        guard let self = self else { return }
        
        DispatchQueue.main.async {
            self.isProcessing = true
        }
        
        do {
            let prediction = try await predictor.predictLiftDrag(
                angleOfAttack: turnAngle,
                rake: finSpecification.rake,
                reynoldsNumber: 500_000, // Typical surfing Re
                finSpec: finSpecification
            )
            
            DispatchQueue.main.async {
                self.liftDrag = prediction
                self.isProcessing = false
            }
            
        } catch {
            print("❌ Prediction error: \(error)")
            
            DispatchQueue.main.async {
                self.isProcessing = false
            }
        }
    }
    
    private func updateVisualizationAngle(_ angle: Double) {
        visualizer.updateAngleOfAttack(Float(angle))
    }
    
    private func updatePressureVisualization(_ data: [Double]) {
        visualizer.updatePressureMap(pressureData: data)
    }
    
    private func updateVisualization(angle: Double, pressure: [Double], prediction: LiftDragPrediction?) {
        // Update 3D visualization with combined data
        visualizer.updateAngleOfAttack(Float(angle))
        
        if !pressure.isEmpty {
            visualizer.updatePressureMap(pressureData: pressure)
        }
        
        // Additional visualization updates based on prediction
        if let prediction = prediction {
            updateVisualizationWithPrediction(prediction)
        }
    }
    
    private func updateVisualizationWithPrediction(_ prediction: LiftDragPrediction) {
        // Update visualization colors/effects based on prediction results
        // This could include changing fin colors based on lift/drag ratios
        
        // Example: Change particle system based on flow regime
        switch prediction.flowRegime {
        case .laminar:
            // Smooth, organized flow visualization
            break
        case .transitional:
            // Mixed flow visualization
            break
        case .turbulent:
            // Chaotic flow visualization
            break
        }
    }
    
    // MARK: - Data Export Methods
    
    private func exportSensorData() {
        guard let data = sensorManager.exportSensorData() else {
            print("❌ Failed to export sensor data")
            return
        }
        
        // In a real app, you would save this to Documents directory or share
        print("📤 Sensor data exported: \(data.count) bytes")
    }
    
    private func exportCognitiveData() {
        guard let data = cognitiveTracker.exportCognitiveData() else {
            print("❌ Failed to export cognitive data")
            return
        }
        
        print("📤 Cognitive data exported: \(data.count) bytes")
    }
    
    private func exportPredictionData() {
        guard let data = predictor.exportPredictions() else {
            print("❌ Failed to export prediction data")
            return
        }
        
        print("📤 Prediction data exported: \(data.count) bytes")
    }
    
    // MARK: - Analytics Methods
    
    func getPerformanceMetrics() -> PerformanceMetrics {
        let sensorStats = sensorManager.getSensorStatistics()
        let hrvStats = cognitiveTracker.getHRVStatistics()
        let predictionMetrics = predictor.getPerformanceMetrics()
        
        return PerformanceMetrics(
            sensorStatistics: sensorStats,
            hrvStatistics: hrvStats,
            predictionMetrics: predictionMetrics,
            currentFlowState: flowStateIndex,
            averageLiftDragRatio: calculateAverageLiftDragRatio(),
            sessionDuration: calculateSessionDuration()
        )
    }
    
    private func calculateAverageLiftDragRatio() -> Double {
        let predictions = predictor.predictionHistory
        guard !predictions.isEmpty else { return 0.0 }
        
        let totalRatio = predictions.map { $0.liftToDragRatio }.reduce(0, +)
        return totalRatio / Double(predictions.count)
    }
    
    private func calculateSessionDuration() -> TimeInterval {
        let stats = sensorManager.getSensorStatistics()
        return stats.dataCollectionDuration
    }
    
    // MARK: - Cognitive Integration
    
    func getCognitiveRecommendations() -> [CognitiveRecommendation] {
        var recommendations: [CognitiveRecommendation] = []
        
        // HRV-based recommendations
        if currentHRV < 20 {
            recommendations.append(.breathingExercise)
        }
        
        // Flow state recommendations
        if flowStateIndex < 0.3 {
            recommendations.append(.relaxation)
        } else if flowStateIndex > 0.8 {
            recommendations.append(.maintainFocus)
        }
        
        // Performance-based recommendations
        if let prediction = liftDrag {
            if prediction.liftToDragRatio < 5.0 {
                recommendations.append(.reduceComplexity)
            }
        }
        
        // Reaction time recommendations
        if reactionTime > 600 {
            recommendations.append(.takeBreak)
        }
        
        return recommendations
    }
    
    // MARK: - Real-time Analysis
    
    func getFlowStateAnalysis() -> FlowStateAnalysis {
        let cognitiveMetrics = cognitiveTracker.cognitiveMetrics
        let recentHistory = cognitiveTracker.getFlowStateHistory().suffix(10)
        
        var trend: FlowStateTrend = .stable
        if recentHistory.count >= 2 {
            let recent = Array(recentHistory.suffix(5))
            let older = Array(recentHistory.prefix(5))
            
            let recentAvg = recent.map { $0.flowStateIndex }.reduce(0, +) / Double(recent.count)
            let olderAvg = older.map { $0.flowStateIndex }.reduce(0, +) / Double(older.count)
            
            let difference = recentAvg - olderAvg
            if difference > 0.1 {
                trend = .improving
            } else if difference < -0.1 {
                trend = .declining
            }
        }
        
        return FlowStateAnalysis(
            currentIndex: flowStateIndex,
            trend: trend,
            recommendations: getCognitiveRecommendations(),
            isOptimal: flowStateIndex > 0.8 && currentHRV > 30,
            cognitiveLoad: cognitiveMetrics?.cognitiveLoad ?? 0.5
        )
    }
}

// MARK: - Supporting Types

struct PerformanceMetrics {
    let sensorStatistics: SensorStatistics
    let hrvStatistics: HRVStatistics
    let predictionMetrics: PredictionPerformanceMetrics
    let currentFlowState: Double
    let averageLiftDragRatio: Double
    let sessionDuration: TimeInterval
}

struct FlowStateAnalysis {
    let currentIndex: Double
    let trend: FlowStateTrend
    let recommendations: [CognitiveRecommendation]
    let isOptimal: Bool
    let cognitiveLoad: Double
    
    var description: String {
        switch currentIndex {
        case 0.8...1.0:
            return "Optimal flow state - peak performance"
        case 0.6..<0.8:
            return "Good flow state - focused and engaged"
        case 0.4..<0.6:
            return "Moderate flow state - room for improvement"
        default:
            return "Low flow state - consider relaxation techniques"
        }
    }
    
    var color: Color {
        switch currentIndex {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .yellow
        default: return .red
        }
    }
}

// MARK: - Extensions

extension FinViewModel {
    
    // Convenience methods for SwiftUI bindings
    var turnAngleBinding: Binding<Double> {
        Binding(
            get: { self.turnAngle },
            set: { self.turnAngle = $0 }
        )
    }
    
    var isConnected: Bool {
        return sensorManager.sensorStatus == .connected
    }
    
    var statusColor: Color {
        switch sensorManager.sensorStatus {
        case .connected: return .green
        case .scanning: return .yellow
        case .disconnected: return .red
        case .error: return .red
        }
    }
    
    // Real-time performance indicators
    var performanceGrade: String {
        guard let prediction = liftDrag else { return "N/A" }
        
        let ratio = prediction.liftToDragRatio
        switch ratio {
        case 15...: return "A+"
        case 12..<15: return "A"
        case 10..<12: return "B+"
        case 8..<10: return "B"
        case 6..<8: return "C+"
        case 4..<6: return "C"
        default: return "D"
        }
    }
    
    var optimalAngleRange: ClosedRange<Double> {
        // Based on Vector 3/2 foil characteristics
        return 8.0...12.0
    }
    
    var isInOptimalRange: Bool {
        return optimalAngleRange.contains(turnAngle)
    }
}

// MARK: - Deinitializer

extension FinViewModel {
    deinit {
        stopMonitoring()
        cancellables.removeAll()
    }
}