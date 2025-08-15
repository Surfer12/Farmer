import Combine
import Foundation
import SwiftUI

class FinViewModel: ObservableObject {
    
    // MARK: - Published Properties
    @Published var angleOfAttack: Float = 5.0 {
        didSet { updatePredictions() }
    }
    
    @Published var reynoldsNumber: Float = 1000000 {
        didSet { updatePredictions() }
    }
    
    @Published var turnAngle: Float = 0.0
    @Published var liftDrag: (lift: Float, drag: Float)?
    @Published var pressureData: [Float] = []
    @Published var hrv: Double?
    @Published var cognitiveMetrics: CognitiveMetrics?
    @Published var surfingCorrelation: SurfingCorrelation?
    
    // MARK: - Computed Properties
    var liftToDragRatio: Float {
        guard let liftDrag = liftDrag, liftDrag.drag > 0.001 else { return 0 }
        return liftDrag.lift / liftDrag.drag
    }
    
    var velocity: Float {
        // Estimate velocity from Reynolds number and fin characteristics
        let finChord: Float = 0.15 // Approximate chord length in meters
        let kinematicViscosity: Float = 1.05e-6 // Seawater at 20°C
        return reynoldsNumber * kinematicViscosity / finChord
    }
    
    // MARK: - Components
    let visualizer = FinVisualizer()
    private let predictor: FinPredictor
    private let sensorManager = SensorManager()
    private let cognitiveTracker = CognitiveTracker()
    
    // MARK: - Combine Subscriptions
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Configuration
    private let finConfiguration = FinConfiguration(
        sideFinArea: 15.00,      // sq.in.
        centerFinArea: 14.50,    // sq.in.
        rakeAngle: 6.5,          // degrees
        foilType: .vector32
    )
    
    // MARK: - Initialization
    init() {
        predictor = FinPredictor()
        setupBindings()
        setupNotifications()
    }
    
    // MARK: - Setup Methods
    private func setupBindings() {
        // Sensor data bindings
        sensorManager.$turnAngle
            .receive(on: DispatchQueue.main)
            .sink { [weak self] angle in
                self?.turnAngle = angle
                self?.updateAngleOfAttackFromSensor(angle)
            }
            .store(in: &cancellables)
        
        sensorManager.$pressureData
            .receive(on: DispatchQueue.main)
            .sink { [weak self] data in
                self?.pressureData = data
                self?.visualizer.updatePressureMap(pressureData: data)
            }
            .store(in: &cancellables)
        
        // Cognitive tracking bindings
        cognitiveTracker.$hrv
            .receive(on: DispatchQueue.main)
            .sink { [weak self] hrv in
                self?.hrv = hrv
                self?.updateCognitiveAnalysis()
            }
            .store(in: &cancellables)
        
        // Combined analysis updates
        Publishers.CombineLatest3(
            $liftDrag.compactMap { $0 },
            $angleOfAttack,
            cognitiveTracker.$flowState
        )
        .debounce(for: .milliseconds(500), scheduler: DispatchQueue.main)
        .sink { [weak self] liftDrag, aoa, flowState in
            self?.updateSurfingCorrelation(liftDrag: liftDrag, aoa: aoa)
        }
        .store(in: &cancellables)
        
        // Real-time visualization updates
        Publishers.CombineLatest($angleOfAttack, $pressureData)
            .debounce(for: .milliseconds(100), scheduler: DispatchQueue.main)
            .sink { [weak self] aoa, pressureData in
                self?.updateVisualization(aoa: aoa, pressureData: pressureData)
            }
            .store(in: &cancellables)
    }
    
    private func setupNotifications() {
        // Surfing maneuver detection
        NotificationCenter.default.publisher(for: .turnDetected)
            .sink { [weak self] notification in
                if let yawRate = notification.object as? Float {
                    self?.handleTurnDetected(yawRate: yawRate)
                }
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .bottomTurnDetected)
            .sink { [weak self] _ in
                self?.handleBottomTurnDetected()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .aerialDetected)
            .sink { [weak self] notification in
                if let acceleration = notification.object as? Float {
                    self?.handleAerialDetected(acceleration: acceleration)
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public Interface
    func startMonitoring() {
        sensorManager.startMonitoring()
        cognitiveTracker.startContinuousMonitoring()
        
        // Initial predictions
        updatePredictions()
        
        print("✅ Started monitoring sensors and cognitive state")
    }
    
    func stopMonitoring() {
        sensorManager.stopMonitoring()
        cognitiveTracker.stopMonitoring()
        
        print("⏹️ Stopped monitoring")
    }
    
    func calibrateSensors() {
        sensorManager.calibrateSensors()
    }
    
    func fetchHRV() {
        cognitiveTracker.fetchHRV()
    }
    
    // MARK: - Prediction Updates
    private func updatePredictions() {
        let input = FinPredictor.CFDInput(
            angleOfAttack: angleOfAttack,
            rake: finConfiguration.rakeAngle,
            reynoldsNumber: reynoldsNumber,
            velocity: velocity,
            finArea: finConfiguration.sideFinArea
        )
        
        if let output = predictor.predictLiftDrag(input: input) {
            DispatchQueue.main.async {
                self.liftDrag = (lift: output.lift, drag: output.drag)
                
                // Update pressure visualization with predicted data
                if !output.pressureDistribution.isEmpty {
                    self.pressureData = output.pressureDistribution
                }
            }
        }
    }
    
    private func updateAngleOfAttackFromSensor(_ sensorAngle: Float) {
        // Convert sensor turn angle to fin angle of attack
        // This would need calibration based on board orientation
        let convertedAOA = abs(sensorAngle) * 0.3 // Simple conversion factor
        let clampedAOA = min(20.0, max(0.0, convertedAOA))
        
        if abs(angleOfAttack - clampedAOA) > 0.5 {
            angleOfAttack = clampedAOA
        }
    }
    
    // MARK: - Cognitive Analysis
    private func updateCognitiveAnalysis() {
        cognitiveMetrics = cognitiveTracker.getCognitiveMetrics()
    }
    
    private func updateSurfingCorrelation(liftDrag: (lift: Float, drag: Float), aoa: Float) {
        surfingCorrelation = cognitiveTracker.correlateWithSurfingMetrics(
            liftDrag: liftDrag,
            angleOfAttack: aoa
        )
    }
    
    // MARK: - Visualization Updates
    private func updateVisualization(aoa: Float, pressureData: [Float]) {
        // Update 3D visualization
        visualizer.updateFinOrientation(angleOfAttack: aoa)
        visualizer.updateFlowVisualization(angleOfAttack: aoa, velocity: velocity)
        
        if !pressureData.isEmpty {
            visualizer.updatePressureMap(pressureData: pressureData)
        } else {
            // Generate synthetic pressure field if no sensor data
            let syntheticPressure = visualizer.generatePressureField(aoa: aoa, velocity: velocity)
            visualizer.updatePressureMap(pressureData: syntheticPressure)
        }
    }
    
    // MARK: - Maneuver Handling
    private func handleTurnDetected(yawRate: Float) {
        print("🌊 Turn detected: \(yawRate) rad/s")
        
        // Analyze turn performance
        let turnAnalysis = analyzeTurnPerformance(yawRate: yawRate)
        
        // Update UI with turn feedback
        DispatchQueue.main.async {
            // Could trigger haptic feedback or UI animations
            self.provideTurnFeedback(analysis: turnAnalysis)
        }
    }
    
    private func handleBottomTurnDetected() {
        print("🏄‍♂️ Bottom turn detected")
        
        // Analyze bottom turn mechanics
        if let liftDrag = liftDrag {
            let efficiency = calculateTurnEfficiency(liftDrag: liftDrag)
            print("Turn efficiency: \(efficiency)")
        }
    }
    
    private func handleAerialDetected(acceleration: Float) {
        print("🚀 Aerial detected: \(acceleration) g")
        
        // Track aerial performance
        let aerialMetrics = AerialMetrics(
            acceleration: acceleration,
            angleOfAttack: angleOfAttack,
            cognitiveState: cognitiveTracker.flowState
        )
        
        // Store for session analysis
        storeAerialMetrics(aerialMetrics)
    }
    
    // MARK: - Performance Analysis
    private func analyzeTurnPerformance(yawRate: Float) -> TurnAnalysis {
        let optimalYawRate: Float = 2.0 // rad/s
        let efficiency = 1.0 - abs(yawRate - optimalYawRate) / optimalYawRate
        
        let analysis = TurnAnalysis(
            yawRate: yawRate,
            efficiency: max(0, min(1, efficiency)),
            angleOfAttack: angleOfAttack,
            liftToDragRatio: liftToDragRatio,
            cognitiveState: cognitiveTracker.flowState
        )
        
        return analysis
    }
    
    private func calculateTurnEfficiency(liftDrag: (lift: Float, drag: Float)) -> Float {
        let ldRatio = liftDrag.drag > 0 ? liftDrag.lift / liftDrag.drag : 0
        let optimalLD: Float = 10.0
        
        return max(0, min(1, ldRatio / optimalLD))
    }
    
    private func provideTurnFeedback(analysis: TurnAnalysis) {
        // Could implement haptic feedback, sound, or visual cues
        if analysis.efficiency > 0.8 {
            print("✅ Excellent turn!")
        } else if analysis.efficiency > 0.6 {
            print("👍 Good turn")
        } else {
            print("💡 Turn could be optimized")
        }
    }
    
    private func storeAerialMetrics(_ metrics: AerialMetrics) {
        // Store for session analysis and improvement tracking
        // Could be saved to Core Data or CloudKit
        print("📊 Aerial metrics stored: \(metrics)")
    }
    
    // MARK: - Session Management
    func startSession() -> SurfingSession {
        let session = SurfingSession(
            startTime: Date(),
            finConfiguration: finConfiguration,
            initialCognitiveState: cognitiveTracker.flowState
        )
        
        print("🏄‍♂️ Started surfing session")
        return session
    }
    
    func endSession(_ session: SurfingSession) -> SessionSummary {
        let endTime = Date()
        let duration = endTime.timeIntervalSince(session.startTime)
        
        let summary = SessionSummary(
            duration: duration,
            averageAngleOfAttack: angleOfAttack, // Would calculate actual average
            averageLiftToDragRatio: liftToDragRatio,
            finalCognitiveState: cognitiveTracker.flowState,
            totalTurns: 0, // Would track from maneuver detection
            totalAerials: 0,
            flowStatePercentage: calculateFlowStatePercentage()
        )
        
        print("📈 Session completed: \(summary)")
        return summary
    }
    
    private func calculateFlowStatePercentage() -> Float {
        // Would calculate based on session data
        return cognitiveMetrics?.flowStateScore ?? 0.5
    }
}

// MARK: - Supporting Data Structures
struct FinConfiguration {
    let sideFinArea: Float
    let centerFinArea: Float
    let rakeAngle: Float
    let foilType: FoilType
    
    enum FoilType {
        case vector32
        case symmetric
        case custom(String)
    }
}

struct TurnAnalysis {
    let yawRate: Float
    let efficiency: Float
    let angleOfAttack: Float
    let liftToDragRatio: Float
    let cognitiveState: CognitiveTracker.FlowState
}

struct AerialMetrics {
    let acceleration: Float
    let angleOfAttack: Float
    let cognitiveState: CognitiveTracker.FlowState
    let timestamp: Date = Date()
}

struct SurfingSession {
    let id = UUID()
    let startTime: Date
    let finConfiguration: FinConfiguration
    let initialCognitiveState: CognitiveTracker.FlowState
}

struct SessionSummary {
    let duration: TimeInterval
    let averageAngleOfAttack: Float
    let averageLiftToDragRatio: Float
    let finalCognitiveState: CognitiveTracker.FlowState
    let totalTurns: Int
    let totalAerials: Int
    let flowStatePercentage: Float
    
    var formattedDuration: String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: duration) ?? "Unknown"
    }
}

// MARK: - Extensions
extension FinViewModel {
    
    func getOptimizationSuggestions() -> [OptimizationSuggestion] {
        var suggestions: [OptimizationSuggestion] = []
        
        // Angle of attack suggestions
        if angleOfAttack > 15 {
            suggestions.append(OptimizationSuggestion(
                category: .technique,
                title: "Reduce Angle of Attack",
                description: "Current angle (\(angleOfAttack)°) may cause flow separation",
                priority: .high
            ))
        }
        
        // L/D ratio suggestions
        if liftToDragRatio < 5 {
            suggestions.append(OptimizationSuggestion(
                category: .performance,
                title: "Improve Lift-to-Drag Ratio",
                description: "Current L/D ratio (\(liftToDragRatio)) is below optimal",
                priority: .medium
            ))
        }
        
        // Cognitive state suggestions
        if let metrics = cognitiveMetrics {
            if metrics.stressLevel > 0.6 {
                suggestions.append(OptimizationSuggestion(
                    category: .cognitive,
                    title: "Reduce Stress Level",
                    description: "High stress may impact performance",
                    priority: .high
                ))
            }
        }
        
        return suggestions
    }
}

struct OptimizationSuggestion {
    let category: Category
    let title: String
    let description: String
    let priority: Priority
    
    enum Category {
        case technique
        case performance
        case cognitive
        case equipment
    }
    
    enum Priority {
        case low
        case medium
        case high
    }
}