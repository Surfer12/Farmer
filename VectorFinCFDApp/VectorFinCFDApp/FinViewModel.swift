import Combine
import Foundation
import SwiftUI

class FinViewModel: ObservableObject {
    @Published var turnAngle: Float = 0.0
    @Published var liftDrag: (lift: Float, drag: Float)?
    @Published var pressureData: [Float] = []
    @Published var hrv: Double?
    @Published var cognitiveLoad: Double = 0.0
    @Published var flowState: FlowState = .neutral
    
    let visualizer = FinVisualizer()
    private var cancellables = Set<AnyCancellable>()
    private let sensorManager = SensorManager()
    private let predictor: FinPredictor
    private let cognitiveTracker = CognitiveTracker()
    
    // Performance tracking
    @Published var predictionConfidence: Float = 0.0
    @Published var lastUpdateTime = Date()
    @Published var isMonitoring = false
    
    init() {
        do {
            predictor = try FinPredictor()
        } catch {
            fatalError("Failed to initialize FinPredictor: \(error)")
        }
        
        setupCombinePipelines()
    }
    
    deinit {
        cancellables.removeAll()
    }
    
    // MARK: - Combine Pipeline Setup
    
    private func setupCombinePipelines() {
        // Turn angle pipeline with debouncing for UI responsiveness
        sensorManager.$turnAngle
            .debounce(for: .seconds(0.1), scheduler: DispatchQueue.main)
            .sink { [weak self] angle in
                self?.turnAngle = angle
                self?.updatePredictions(aoa: angle)
                self?.updateFlowState(angle: angle)
            }
            .store(in: &cancellables)
        
        // Pressure data pipeline with real-time updates
        sensorManager.$pressureData
            .sink { [weak self] data in
                self?.pressureData = data
                self?.visualizer.updatePressureMap(pressureData: data)
                self?.updateCognitiveLoad(pressureData: data)
            }
            .store(in: &cancellables)
        
        // Sensor status monitoring
        sensorManager.$sensorStatus
            .sink { [weak self] status in
                if status.contains("error") || status.contains("failed") {
                    self?.flowState = .error
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public Methods
    
    func startMonitoring() {
        isMonitoring = true
        sensorManager.startMonitoring()
        lastUpdateTime = Date()
    }
    
    func stopMonitoring() {
        isMonitoring = false
        sensorManager.stopMonitoring()
    }
    
    func updatePredictions(aoa: Float) {
        do {
            let result = try predictor.predictLiftDrag(aoa: aoa, rake: 6.5, re: 1_000_000)
            liftDrag = result
            
            // Update prediction confidence
            predictionConfidence = predictor.getPredictionConfidence(aoa: aoa, rake: 6.5, re: 1_000_000)
            
            // Update visualizer with new angle
            visualizer.animateFinRotation(to: aoa)
            
        } catch {
            print("Prediction error: \(error)")
            predictionConfidence = 0.0
        }
    }
    
    func fetchHRV() {
        cognitiveTracker.fetchHRV { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let value):
                    self?.hrv = value
                    self?.updateFlowStateFromHRV(value)
                case .failure(let error):
                    print("HRV fetch error: \(error)")
                    self?.hrv = nil
                }
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func updateFlowState(angle: Float) {
        // Determine flow state based on turn angle and performance
        if let liftDrag = liftDrag {
            let liftToDragRatio = liftDrag.lift / max(liftDrag.drag, 0.1)
            
            if angle < 5.0 && liftToDragRatio > 8.0 {
                flowState = .optimal
            } else if angle < 10.0 && liftToDragRatio > 5.0 {
                flowState = .good
            } else if angle < 15.0 {
                flowState = .moderate
            } else {
                flowState = .challenging
            }
        }
    }
    
    private func updateFlowStateFromHRV(_ hrv: Double) {
        // HRV-based flow state assessment
        // Higher HRV typically indicates better flow state
        if hrv > 50.0 {
            if flowState == .optimal || flowState == .good {
                // Maintain or improve flow state
            } else {
                flowState = .good
            }
        } else if hrv < 30.0 {
            flowState = .challenging
        }
    }
    
    private func updateCognitiveLoad(pressureData: [Float]) {
        // Calculate cognitive load based on pressure complexity
        let pressureVariance = calculateVariance(pressureData)
        let normalizedVariance = min(pressureVariance * 10, 1.0)
        
        cognitiveLoad = Double(normalizedVariance)
    }
    
    private func calculateVariance(_ data: [Float]) -> Float {
        guard data.count > 1 else { return 0.0 }
        
        let mean = data.reduce(0, +) / Float(data.count)
        let squaredDifferences = data.map { pow($0 - mean, 2) }
        let variance = squaredDifferences.reduce(0, +) / Float(data.count)
        
        return variance
    }
    
    // MARK: - Data Export
    
    func exportSessionData() -> String {
        let timestamp = Date().ISO8601String()
        let liftDragStr = liftDrag.map { "Lift: \($0.lift), Drag: \($0.drag)" } ?? "N/A"
        
        return """
        Vector 3/2 Fin CFD Session Data
        ================================
        Timestamp: \(timestamp)
        Turn Angle: \(turnAngle)°
        Performance: \(liftDragStr)
        HRV: \(hrv?.description ?? "N/A") ms
        Flow State: \(flowState.rawValue)
        Cognitive Load: \(String(format: "%.2f", cognitiveLoad))
        Prediction Confidence: \(String(format: "%.1f", predictionConfidence * 100))%
        Pressure Data: \(pressureData.map { String(format: "%.3f", $0) }.joined(separator: ", "))
        """
    }
    
    // MARK: - Performance Analysis
    
    func getPerformanceMetrics() -> PerformanceMetrics {
        let avgPressure = pressureData.isEmpty ? 0.0 : Double(pressureData.reduce(0, +)) / Double(pressureData.count)
        let pressureRange = pressureData.isEmpty ? 0.0 : Double(pressureData.max()! - pressureData.min()!)
        
        return PerformanceMetrics(
            turnAngle: Double(turnAngle),
            lift: Double(liftDrag?.lift ?? 0),
            drag: Double(liftDrag?.drag ?? 0),
            liftToDragRatio: liftDrag.map { Double($0.lift / max($0.drag, 0.1)) } ?? 0.0,
            averagePressure: avgPressure,
            pressureRange: pressureRange,
            hrv: hrv ?? 0.0,
            cognitiveLoad: cognitiveLoad,
            flowState: flowState,
            confidence: Double(predictionConfidence)
        )
    }
}

// MARK: - Supporting Types

enum FlowState: String, CaseIterable {
    case optimal = "Optimal Flow"
    case good = "Good Flow"
    case moderate = "Moderate Flow"
    case challenging = "Challenging"
    case neutral = "Neutral"
    case error = "Error"
    
    var color: Color {
        switch self {
        case .optimal: return .green
        case .good: return .blue
        case .moderate: return .yellow
        case .challenging: return .orange
        case .neutral: return .gray
        case .error: return .red
        }
    }
    
    var description: String {
        switch self {
        case .optimal:
            return "Perfect balance of lift and control. Optimal for high-performance surfing."
        case .good:
            return "Strong performance with good lift-to-drag ratio. Excellent flow state."
        case .moderate:
            return "Adequate performance. Consider angle adjustments for optimization."
        case .challenging:
            return "Performance may be compromised. Focus on technique and positioning."
        case .neutral:
            return "Baseline performance. Ready for optimization."
        case .error:
            return "System error detected. Check sensor connections and app status."
        }
    }
}

struct PerformanceMetrics {
    let turnAngle: Double
    let lift: Double
    let drag: Double
    let liftToDragRatio: Double
    let averagePressure: Double
    let pressureRange: Double
    let hrv: Double
    let cognitiveLoad: Double
    let flowState: FlowState
    let confidence: Double
    
    var efficiency: Double {
        // Calculate efficiency based on lift-to-drag ratio and pressure distribution
        let pressureEfficiency = 1.0 - (pressureRange / max(averagePressure, 0.1))
        let aerodynamicEfficiency = min(liftToDragRatio / 10.0, 1.0)
        return (pressureEfficiency + aerodynamicEfficiency) / 2.0
    }
}

// MARK: - Extensions

extension Date {
    func ISO8601String() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}
