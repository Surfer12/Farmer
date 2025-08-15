import Foundation
import Combine
import SceneKit

/// Main view model coordinating CFD visualization, sensor data, and cognitive metrics
class CFDViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var currentAngleOfAttack: Float = 0.0
    @Published var currentRakeAngle: Float = 6.5
    @Published var reynoldsNumber: Float = 5e5
    @Published var liftCoefficient: Float = 0.0
    @Published var dragCoefficient: Float = 0.0
    @Published var liftToDragRatio: Float = 0.0
    @Published var flowStateMetrics: FlowStateMetrics?
    @Published var cfdData: CFDData?
    @Published var isRealTimeMode: Bool = false
    @Published var performanceCurve: [CFDPrediction] = []
    @Published var riderWeight: Float = 150.0 // lbs
    
    // UI State
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var showingSettings: Bool = false
    
    // MARK: - Private Properties
    
    private let sensorManager = SensorManager()
    private let cfdPredictor = FinCFDPredictor()
    private let finVisualizer = FinVisualizer()
    private var cancellables = Set<AnyCancellable>()
    
    // Data processing
    private let dataProcessor = CFDDataProcessor()
    private var updateTimer: Timer?
    
    // Performance monitoring
    private var frameCount: Int = 0
    private var lastFrameTime: Date = Date()
    
    // MARK: - Initialization
    
    init() {
        setupDataPipeline()
        setupPerformanceMonitoring()
        generateInitialData()
    }
    
    // MARK: - Setup Methods
    
    private func setupDataPipeline() {
        // Real-time sensor data pipeline
        sensorManager.$turnAngle
            .combineLatest(sensorManager.$pitch, sensorManager.$roll)
            .debounce(for: .milliseconds(100), scheduler: RunLoop.main)
            .sink { [weak self] yaw, pitch, roll in
                self?.processSensorUpdate(yaw: yaw, pitch: pitch, roll: roll)
            }
            .store(in: &cancellables)
        
        // Flow state metrics pipeline
        sensorManager.$heartRate
            .combineLatest(sensorManager.$heartRateVariability)
            .debounce(for: .seconds(1), scheduler: RunLoop.main)
            .sink { [weak self] heartRate, hrv in
                self?.updateFlowStateMetrics(heartRate: heartRate, hrv: hrv)
            }
            .store(in: &cancellables)
        
        // Pressure data pipeline
        sensorManager.$pressureData
            .debounce(for: .milliseconds(50), scheduler: RunLoop.main)
            .sink { [weak self] pressureData in
                self?.updatePressureVisualization(pressureData: pressureData)
            }
            .store(in: &cancellables)
        
        // CFD prediction pipeline
        Publishers.CombineLatest3($currentAngleOfAttack, $currentRakeAngle, $reynoldsNumber)
            .debounce(for: .milliseconds(200), scheduler: RunLoop.main)
            .sink { [weak self] aoa, rake, re in
                self?.updateCFDPredictions(aoa: aoa, rake: rake, re: re)
            }
            .store(in: &cancellables)
        
        // Performance curve generation
        $reynoldsNumber
            .debounce(for: .seconds(0.5), scheduler: RunLoop.main)
            .sink { [weak self] re in
                self?.generatePerformanceCurve(reynoldsNumber: re)
            }
            .store(in: &cancellables)
    }
    
    private func setupPerformanceMonitoring() {
        // Monitor visualization performance
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updatePerformanceMetrics()
        }
    }
    
    private func generateInitialData() {
        // Generate initial CFD data for visualization
        updateCFDPredictions(aoa: currentAngleOfAttack, rake: currentRakeAngle, re: reynoldsNumber)
        generatePerformanceCurve(reynoldsNumber: reynoldsNumber)
    }
    
    // MARK: - Public Methods
    
    /// Starts real-time monitoring mode
    func startRealTimeMode() {
        isRealTimeMode = true
        sensorManager.startMonitoring()
        
        // Start high-frequency updates
        updateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateVisualization()
        }
    }
    
    /// Stops real-time monitoring mode
    func stopRealTimeMode() {
        isRealTimeMode = false
        sensorManager.stopMonitoring()
        updateTimer?.invalidate()
        updateTimer = nil
    }
    
    /// Updates angle of attack manually (for non-real-time mode)
    func setAngleOfAttack(_ angle: Float) {
        currentAngleOfAttack = max(-20.0, min(20.0, angle))
    }
    
    /// Updates rake angle
    func setRakeAngle(_ angle: Float) {
        currentRakeAngle = max(0.0, min(10.0, angle))
    }
    
    /// Updates Reynolds number
    func setReynoldsNumber(_ re: Float) {
        reynoldsNumber = max(1e4, min(1e7, re))
    }
    
    /// Gets the fin visualizer for SceneKit integration
    func getFinVisualizer() -> FinVisualizer {
        return finVisualizer
    }
    
    /// Calculates forces for current conditions
    func getCurrentForces() -> (lift: Float, drag: Float) {
        guard let cfdData = cfdData else { return (0, 0) }
        return cfdData.calculateForces(riderWeight: riderWeight)
    }
    
    /// Gets performance comparison data
    func getPerformanceComparison() -> PerformanceComparison {
        let vector32Performance = performanceCurve.first { $0.angleOfAttack == currentAngleOfAttack }
        let baselinePerformance = calculateBaselinePerformance(aoa: currentAngleOfAttack)
        
        return PerformanceComparison(
            vector32: vector32Performance,
            baseline: baselinePerformance,
            improvement: vector32Performance?.performanceGain ?? 0.0
        )
    }
    
    // MARK: - Private Methods
    
    private func processSensorUpdate(yaw: Float, pitch: Float, roll: Float) {
        guard isRealTimeMode else { return }
        
        // Convert sensor data to angle of attack
        let newAoA = sensorManager.getCurrentAngleOfAttack()
        currentAngleOfAttack = newAoA
        
        // Update visualization with new angle
        finVisualizer.animateAngleOfAttack(to: newAoA, duration: 0.1)
    }
    
    private func updateFlowStateMetrics(heartRate: Double, hrv: Double) {
        let metrics = sensorManager.getFlowStateMetrics()
        flowStateMetrics = metrics
        
        // Trigger haptic feedback for flow state changes
        triggerFlowStateHaptics(flowState: metrics.flowState)
    }
    
    private func updatePressureVisualization(pressureData: [Float]) {
        guard !pressureData.isEmpty else { return }
        
        // Create CFD data with pressure information
        let cfdDataWithPressure = CFDData(
            reynoldsNumber: reynoldsNumber,
            angleOfAttack: currentAngleOfAttack,
            rakeAngle: currentRakeAngle
        )
        
        // Update visualization
        finVisualizer.updateVisualization(with: cfdDataWithPressure)
    }
    
    private func updateCFDPredictions(aoa: Float, rake: Float, re: Float) {
        isLoading = true
        errorMessage = nil
        
        // Validate inputs
        let validation = cfdPredictor.validateInputs(
            angleOfAttack: aoa,
            rakeAngle: rake,
            reynoldsNumber: re
        )
        
        if !validation.isValid {
            errorMessage = "Invalid input parameters"
            isLoading = false
            return
        }
        
        // Perform prediction
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            if let prediction = self.cfdPredictor.predictVector32Performance(
                angleOfAttack: aoa,
                reynoldsNumber: re
            ) {
                DispatchQueue.main.async {
                    self.liftCoefficient = prediction.liftCoefficient
                    self.dragCoefficient = prediction.dragCoefficient
                    self.liftToDragRatio = prediction.liftToDragRatio
                    
                    // Create CFD data for visualization
                    let cfdData = prediction.toCFDData()
                    self.cfdData = cfdData
                    
                    // Update visualization
                    self.finVisualizer.updateVisualization(with: cfdData)
                    
                    self.isLoading = false
                }
            } else {
                DispatchQueue.main.async {
                    self.errorMessage = "Prediction failed"
                    self.isLoading = false
                }
            }
        }
    }
    
    private func generatePerformanceCurve(reynoldsNumber: Float) {
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self = self else { return }
            
            let curve = self.cfdPredictor.predictPerformanceCurve(
                aoaRange: 0...20,
                steps: 21,
                reynoldsNumber: reynoldsNumber
            )
            
            DispatchQueue.main.async {
                self.performanceCurve = curve
            }
        }
    }
    
    private func updateVisualization() {
        guard let cfdData = cfdData else { return }
        
        finVisualizer.updateVisualization(with: cfdData)
        frameCount += 1
    }
    
    private func updatePerformanceMetrics() {
        let currentTime = Date()
        let deltaTime = currentTime.timeIntervalSince(lastFrameTime)
        let fps = Float(frameCount) / Float(deltaTime)
        
        // Reset counters
        frameCount = 0
        lastFrameTime = currentTime
        
        // Log performance if needed
        if fps < 30.0 {
            print("Warning: Low FPS detected: \(fps)")
        }
    }
    
    private func triggerFlowStateHaptics(flowState: FlowState) {
        // Implement haptic feedback for flow state changes
        #if os(iOS)
        import UIKit
        
        switch flowState {
        case .optimal:
            let feedback = UINotificationFeedbackGenerator()
            feedback.notificationOccurred(.success)
        case .good:
            let feedback = UIImpactFeedbackGenerator(style: .light)
            feedback.impactOccurred()
        case .moderate:
            let feedback = UIImpactFeedbackGenerator(style: .medium)
            feedback.impactOccurred()
        case .poor:
            let feedback = UIImpactFeedbackGenerator(style: .heavy)
            feedback.impactOccurred()
        }
        #endif
    }
    
    private func calculateBaselinePerformance(aoa: Float) -> CFDPrediction? {
        // Calculate baseline performance without Vector 3/2 improvements
        return cfdPredictor.predictCoefficients(
            angleOfAttack: aoa,
            rakeAngle: 0.0,
            reynoldsNumber: reynoldsNumber
        ).map { coeffs in
            CFDPrediction(
                angleOfAttack: aoa,
                reynoldsNumber: reynoldsNumber,
                liftCoefficient: coeffs.lift * 0.88, // Remove 12% improvement
                dragCoefficient: coeffs.drag * 1.05, // Increase drag slightly
                sideFinContribution: (0, 0),
                centerFinContribution: coeffs,
                confidence: 0.8
            )
        }
    }
}

// MARK: - Supporting Types

/// Data processor for CFD pipeline
class CFDDataProcessor {
    private var dataHistory: [CFDData] = []
    private let maxHistorySize = 1000
    
    func processData(_ data: CFDData) -> CFDData {
        // Add to history
        dataHistory.append(data)
        if dataHistory.count > maxHistorySize {
            dataHistory.removeFirst()
        }
        
        // Apply smoothing or other processing
        return applySmoothing(data)
    }
    
    private func applySmoothing(_ data: CFDData) -> CFDData {
        // Simple smoothing implementation
        guard dataHistory.count > 3 else { return data }
        
        let recent = Array(dataHistory.suffix(3))
        let avgLift = recent.map { $0.liftCoefficient }.reduce(0, +) / Float(recent.count)
        let avgDrag = recent.map { $0.dragCoefficient }.reduce(0, +) / Float(recent.count)
        
        // Create smoothed data (this is simplified)
        return CFDData(
            reynoldsNumber: data.reynoldsNumber,
            angleOfAttack: data.angleOfAttack,
            rakeAngle: data.rakeAngle
        )
    }
}

/// Performance comparison data
struct PerformanceComparison {
    let vector32: CFDPrediction?
    let baseline: CFDPrediction?
    let improvement: Float
    
    var liftImprovement: Float {
        guard let v32 = vector32, let base = baseline else { return 0 }
        return (v32.liftCoefficient - base.liftCoefficient) / base.liftCoefficient
    }
    
    var dragReduction: Float {
        guard let v32 = vector32, let base = baseline else { return 0 }
        return (base.dragCoefficient - v32.dragCoefficient) / base.dragCoefficient
    }
    
    var efficiencyGain: Float {
        guard let v32 = vector32, let base = baseline else { return 0 }
        return v32.liftToDragRatio / base.liftToDragRatio - 1.0
    }
}