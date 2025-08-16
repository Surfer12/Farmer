<<<<<<< Current (Your changes)
import Combine
=======
// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import Combine
import Foundation

class FinViewModel: ObservableObject {
    @Published var turnAngle: Float = 0.0
    @Published var liftDrag: (lift: Float, drag: Float)?
    @Published var pressureData: [Float] = []
    @Published var hrv: Double?
    @Published var predictionConfidence: Float = 1.0
    @Published var flowState: FlowState = .neutral
    @Published var isMonitoring = false
    
    let visualizer = FinVisualizer()
    private var cancellables = Set<AnyCancellable>()
    private let sensorManager = SensorManager()
    private let predictor: FinPredictor
    private let cognitiveTracker = CognitiveTracker()
    
    // Performance metrics
    @Published var performanceMetrics = PerformanceMetrics()
    
    // Error handling
    @Published var errorMessage: String?
    @Published var isShowingError = false
    
    init() {
        predictor = FinPredictor()
        setupDataPipeline()
        setupErrorHandling()
    }
    
    private func setupDataPipeline() {
        // Combine pipeline for turn angle updates
        sensorManager.$turnAngle
            .debounce(for: .seconds(0.1), scheduler: DispatchQueue.main)
            .removeDuplicates()
            .sink { [weak self] angle in
                self?.turnAngle = angle
                self?.updatePredictions(aoa: angle)
                self?.updateFlowState(angle: angle)
            }
            .store(in: &cancellables)
        
        // Combine pipeline for pressure data updates
        sensorManager.$pressureData
            .debounce(for: .seconds(0.05), scheduler: DispatchQueue.main)
            .sink { [weak self] data in
                self?.pressureData = data
                self?.visualizer.updatePressureMap(pressureData: data)
                self?.updatePerformanceMetrics(pressureData: data)
            }
            .store(in: &cancellables)
        
        // Monitor sensor status
        sensorManager.$sensorStatus
            .sink { [weak self] status in
                if status.contains("error") || status.contains("failed") {
                    self?.showError("Sensor Error: \(status)")
                }
            }
            .store(in: &cancellables)
    }
    
    private func setupErrorHandling() {
        // Monitor for prediction errors
        NotificationCenter.default.publisher(for: .predictionError)
            .sink { [weak self] notification in
                if let error = notification.object as? Error {
                    self?.showError("Prediction Error: \(error.localizedDescription)")
                }
            }
            .store(in: &cancellables)
    }
    
    func startMonitoring() {
        isMonitoring = true
        sensorManager.startMonitoring()
        
        // Start periodic HRV updates
        Timer.publish(every: 30, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.fetchHRV()
            }
            .store(in: &cancellables)
    }
    
    func stopMonitoring() {
        isMonitoring = false
        sensorManager.stopMonitoring()
        cancellables.removeAll()
    }
    
    private func updatePredictions(aoa: Float) {
        do {
            let result = try predictor.predictLiftDrag(aoa: aoa, rake: 6.5, re: 500_000)
            liftDrag = result
            predictionConfidence = predictor.getPredictionConfidence(aoa: aoa, rake: 6.5, re: 500_000)
            
            // Update performance metrics
            performanceMetrics.updateLiftDrag(lift: result.lift, drag: result.drag)
            
        } catch {
            showError("Prediction failed: \(error.localizedDescription)")
            NotificationCenter.default.post(name: .predictionError, object: error)
        }
    }
    
    private func updateFlowState(angle: Float) {
        // Calculate flow state based on angle and HRV
        let normalizedAngle = angle / 20.0
        let hrvFactor = (hrv ?? 50.0) / 100.0 // Normalize HRV
        
        let flowScore = (1.0 - normalizedAngle) * 0.7 + hrvFactor * 0.3
        
        if flowScore > 0.8 {
            flowState = .optimal
        } else if flowScore > 0.6 {
            flowState = .good
        } else if flowScore > 0.4 {
            flowState = .moderate
        } else {
            flowState = .poor
        }
    }
    
    private func updatePerformanceMetrics(pressureData: [Float]) {
        performanceMetrics.updatePressure(pressureData)
        
        // Calculate efficiency metrics
        if let liftDrag = liftDrag {
            let liftToDrag = liftDrag.drag > 0 ? liftDrag.lift / liftDrag.drag : 0
            performanceMetrics.liftToDragRatio = liftToDrag
        }
    }
    
    func fetchHRV() {
        cognitiveTracker.fetchHRV { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let value):
                    self?.hrv = value
                    self?.performanceMetrics.updateHRV(value)
                case .failure(let error):
                    self?.showError("HRV fetch failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    func resetMetrics() {
        performanceMetrics = PerformanceMetrics()
        errorMessage = nil
        isShowingError = false
    }
    
    func exportData() -> String {
        let timestamp = Date().ISO8601String()
        let sensorData = sensorManager.exportSensorData()
        let metricsData = performanceMetrics.exportData()
        
        return """
        === Vector 3/2 Fin CFD Analysis Export ===
        Export Time: \(timestamp)
        
        Current State:
        - Turn Angle: \(turnAngle)°
        - Lift: \(liftDrag?.lift ?? 0) N
        - Drag: \(liftDrag?.drag ?? 0) N
        - HRV: \(hrv ?? 0) ms
        - Flow State: \(flowState.rawValue)
        - Prediction Confidence: \(predictionConfidence)
        
        \(sensorData)
        
        \(metricsData)
        """
    }
    
    private func showError(_ message: String) {
        errorMessage = message
        isShowingError = true
        
        // Auto-dismiss after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.isShowingError = false
        }
    }
}

// MARK: - Supporting Types

enum FlowState: String, CaseIterable {
    case optimal = "Optimal"
    case good = "Good"
    case moderate = "Moderate"
    case poor = "Poor"
    case neutral = "Neutral"
    
    var color: String {
        switch self {
        case .optimal: return "green"
        case .good: return "blue"
        case .moderate: return "orange"
        case .poor: return "red"
        case .neutral: return "gray"
        }
    }
}

struct PerformanceMetrics {
    var averageLift: Float = 0
    var averageDrag: Float = 0
    var liftToDragRatio: Float = 0
    var averageHRV: Double = 0
    var maxPressure: Float = 0
    var minPressure: Float = 1
    var pressureVariance: Float = 0
    
    private var liftHistory: [Float] = []
    private var dragHistory: [Float] = []
    private var hrvHistory: [Double] = []
    private var pressureHistory: [[Float]] = []
    
    mutating func updateLiftDrag(lift: Float, drag: Float) {
        liftHistory.append(lift)
        dragHistory.append(drag)
        
        // Keep only last 100 readings
        if liftHistory.count > 100 {
            liftHistory.removeFirst()
            dragHistory.removeFirst()
        }
        
        averageLift = liftHistory.reduce(0, +) / Float(liftHistory.count)
        averageDrag = dragHistory.reduce(0, +) / Float(dragHistory.count)
    }
    
    mutating func updateHRV(_ hrv: Double) {
        hrvHistory.append(hrv)
        
        if hrvHistory.count > 50 {
            hrvHistory.removeFirst()
        }
        
        averageHRV = hrvHistory.reduce(0, +) / Double(hrvHistory.count)
    }
    
    mutating func updatePressure(_ pressureData: [Float]) {
        pressureHistory.append(pressureData)
        
        if pressureHistory.count > 100 {
            pressureHistory.removeFirst()
        }
        
        let flatPressures = pressureHistory.flatMap { $0 }
        if !flatPressures.isEmpty {
            maxPressure = flatPressures.max() ?? 0
            minPressure = flatPressures.min() ?? 1
            
            let average = flatPressures.reduce(0, +) / Float(flatPressures.count)
            let variance = flatPressures.map { pow($0 - average, 2) }.reduce(0, +) / Float(flatPressures.count)
            pressureVariance = sqrt(variance)
        }
    }
    
    func exportData() -> String {
        return """
        Performance Metrics:
        - Average Lift: \(String(format: "%.2f", averageLift)) N
        - Average Drag: \(String(format: "%.2f", averageDrag)) N
        - L/D Ratio: \(String(format: "%.2f", liftToDragRatio))
        - Average HRV: \(String(format: "%.1f", averageHRV)) ms
        - Pressure Range: \(String(format: "%.3f", minPressure)) - \(String(format: "%.3f", maxPressure))
        - Pressure Variance: \(String(format: "%.3f", pressureVariance))
        """
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let predictionError = Notification.Name("predictionError")
}

// MARK: - Date Extensions

extension Date {
    func ISO8601String() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}
>>>>>>> Incoming (Background Agent changes)
