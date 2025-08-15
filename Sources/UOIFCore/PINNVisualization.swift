import SwiftUI
import Charts

// MARK: - PINN Solution Data Models

/// Data point for PINN solution visualization
public struct PINNSolutionPoint: Identifiable {
    public let id = UUID()
    public let x: Double
    public let pinnValue: Double
    public let rk4Value: Double
    public let time: Double
    
    public init(x: Double, pinnValue: Double, rk4Value: Double, time: Double) {
        self.x = x
        self.pinnValue = pinnValue
        self.rk4Value = rk4Value
        self.time = time
    }
}

/// Performance metrics visualization data
public struct PerformanceMetricsData: Identifiable {
    public let id = UUID()
    public let epoch: Int
    public let pdeLoss: Double
    public let icLoss: Double
    public let totalLoss: Double
    public let performanceMetric: Double
    
    public init(epoch: Int, pdeLoss: Double, icLoss: Double, totalLoss: Double, performanceMetric: Double) {
        self.epoch = epoch
        self.pdeLoss = pdeLoss
        self.icLoss = icLoss
        self.totalLoss = totalLoss
        self.performanceMetric = performanceMetric
    }
}

// MARK: - PINN Solution Chart

/// SwiftUI Chart component for visualizing PINN vs RK4 solutions
public struct PINNSolutionChart: View {
    public let solutionData: [PINNSolutionPoint]
    public let title: String
    
    public init(solutionData: [PINNSolutionPoint], title: String = "PINN vs RK4 Solution Comparison") {
        self.solutionData = solutionData
        self.title = title
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)
            
            Chart {
                ForEach(solutionData) { point in
                    LineMark(
                        x: .value("Position", point.x),
                        y: .value("PINN", point.pinnValue)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .symbol(Circle().strokeBorder(lineWidth: 2))
                    .symbolSize(30)
                }
                
                ForEach(solutionData) { point in
                    LineMark(
                        x: .value("Position", point.x),
                        y: .value("RK4", point.rk4Value)
                    )
                    .foregroundStyle(.red)
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                    .symbol(Diamond().strokeBorder(lineWidth: 2))
                    .symbolSize(30)
                }
            }
            .chartXAxis {
                AxisMarks(position: .bottom) {
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) {
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .chartLegend(position: .top, alignment: .center) {
                HStack {
                    HStack {
                        Circle()
                            .fill(.blue)
                            .frame(width: 12, height: 12)
                        Text("PINN Solution")
                            .font(.caption)
                    }
                    
                    HStack {
                        Diamond()
                            .fill(.red)
                            .frame(width: 12, height: 12)
                        Text("RK4 Solution")
                            .font(.caption)
                    }
                }
            }
            .frame(height: 300)
            .padding()
        }
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Training Progress Chart

/// SwiftUI Chart component for visualizing training progress
public struct TrainingProgressChart: View {
    public let trainingData: [PerformanceMetricsData]
    public let title: String
    
    public init(trainingData: [PerformanceMetricsData], title: String = "Training Progress") {
        self.trainingData = trainingData
        self.title = title
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)
            
            Chart {
                ForEach(trainingData) { data in
                    LineMark(
                        x: .value("Epoch", data.epoch),
                        y: .value("PDE Loss", data.pdeLoss)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .symbol(Circle().strokeBorder(lineWidth: 2))
                    .symbolSize(20)
                }
                
                ForEach(trainingData) { data in
                    LineMark(
                        x: .value("Epoch", data.epoch),
                        y: .value("IC Loss", data.icLoss)
                    )
                    .foregroundStyle(.red)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .symbol(Diamond().strokeBorder(lineWidth: 2))
                    .symbolSize(20)
                }
                
                ForEach(trainingData) { data in
                    LineMark(
                        x: .value("Epoch", data.epoch),
                        y: .value("Total Loss", data.totalLoss)
                    )
                    .foregroundStyle(.green)
                    .lineStyle(StrokeStyle(lineWidth: 3))
                    .symbol(Square().strokeBorder(lineWidth: 2))
                    .symbolSize(20)
                }
            }
            .chartXAxis {
                AxisMarks(position: .bottom) {
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) {
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
                }
            }
            .chartLegend(position: .top, alignment: .center) {
                HStack(spacing: 20) {
                    HStack {
                        Circle()
                            .fill(.blue)
                            .frame(width: 10, height: 10)
                        Text("PDE Loss")
                            .font(.caption)
                    }
                    
                    HStack {
                        Diamond()
                            .fill(.red)
                            .frame(width: 10, height: 10)
                        Text("IC Loss")
                            .font(.caption)
                    }
                    
                    HStack {
                        Square()
                            .fill(.green)
                            .frame(width: 10, height: 10)
                        Text("Total Loss")
                            .font(.caption)
                    }
                }
            }
            .frame(height: 300)
            .padding()
        }
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Performance Metrics Display

/// SwiftUI component for displaying performance metrics
public struct PerformanceMetricsDisplay: View {
    public let hybridOutput: HybridOutput
    public let cognitiveReg: CognitiveRegularization
    public let efficiencyReg: EfficiencyRegularization
    public let probability: ProbabilityModel
    public let performanceMetric: PerformanceMetric
    
    public init(performanceMetric: PerformanceMetric) {
        self.hybridOutput = performanceMetric.hybridOutput
        self.cognitiveReg = performanceMetric.cognitiveReg
        self.efficiencyReg = performanceMetric.efficiencyReg
        self.probability = performanceMetric.probability
        self.performanceMetric = performanceMetric
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Performance Metrics")
                .font(.title2)
                .fontWeight(.bold)
                .padding(.horizontal)
            
            VStack(spacing: 16) {
                // Hybrid Output Section
                VStack(alignment: .leading, spacing: 8) {
                    Text("Hybrid Output")
                        .font(.headline)
                        .foregroundColor(.blue)
                    
                    HStack {
                        VStack(alignment: .leading) {
                            Text("S(x) - State Inference")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.3f", hybridOutput.stateInference))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading) {
                            Text("N(x) - ML Gradient")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.3f", hybridOutput.mlGradient))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading) {
                            Text("α - Weight")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.2f", hybridOutput.alpha))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading) {
                            Text("O_hybrid")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.3f", hybridOutput.hybridValue))
                                .font(.title3)
                                .fontWeight(.semibold)
                                .foregroundColor(.blue)
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
                
                // Regularization Section
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Cognitive Regularization")
                            .font(.headline)
                            .foregroundColor(.orange)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("PDE Residual: \(String(format: "%.3f", cognitiveReg.pdeResidual))")
                                .font(.caption)
                            Text("Weight λ1: \(String(format: "%.2f", cognitiveReg.weight))")
                                .font(.caption)
                            Text("Value: \(String(format: "%.3f", cognitiveReg.value))")
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Efficiency Regularization")
                            .font(.headline)
                            .foregroundColor(.purple)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Training Efficiency: \(String(format: "%.3f", efficiencyReg.trainingEfficiency))")
                                .font(.caption)
                            Text("Weight λ2: \(String(format: "%.2f", efficiencyReg.weight))")
                                .font(.caption)
                            Text("Value: \(String(format: "%.3f", efficiencyReg.value))")
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }
                
                // Probability Section
                VStack(alignment: .leading, spacing: 8) {
                    Text("Probability Model")
                        .font(.headline)
                        .foregroundColor(.green)
                    
                    HStack {
                        VStack(alignment: .leading) {
                            Text("P(H|E)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.3f", probability.hypothesis))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading) {
                            Text("β - Responsiveness")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.2f", probability.beta))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading) {
                            Text("P_adj")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.3f", probability.adjustedProbability))
                                .font(.title3)
                                .fontWeight(.semibold)
                                .foregroundColor(.green)
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
                
                // Final Performance Metric
                VStack(alignment: .leading, spacing: 8) {
                    Text("Integrated Performance Metric Ψ(x)")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    HStack {
                        Text(String(format: "%.3f", performanceMetric.value))
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.blue)
                        
                        Spacer()
                        
                        VStack(alignment: .trailing) {
                            Text(performanceMetric.interpretation)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.trailing)
                        }
                    }
                }
                .padding()
                .background(Color(.systemBlue).opacity(0.1))
                .cornerRadius(8)
            }
            .padding(.horizontal)
        }
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Main PINN Dashboard

/// Main dashboard combining all PINN visualization components
public struct PINNDashboard: View {
    public let solutionData: [PINNSolutionPoint]
    public let trainingData: [PerformanceMetricsData]
    public let performanceMetric: PerformanceMetric
    
    public init(solutionData: [PINNSolutionPoint], 
                trainingData: [PerformanceMetricsData], 
                performanceMetric: PerformanceMetric) {
        self.solutionData = solutionData
        self.trainingData = trainingData
        self.performanceMetric = performanceMetric
    }
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                PINNSolutionChart(solutionData: solutionData)
                
                TrainingProgressChart(trainingData: trainingData)
                
                PerformanceMetricsDisplay(performanceMetric: performanceMetric)
            }
            .padding()
        }
        .navigationTitle("PINN Analysis Dashboard")
        .background(Color(.systemGroupedBackground))
    }
}

// MARK: - Sample Data Generator

/// Utility for generating sample data for visualization
public struct SampleDataGenerator {
    
    /// Generate sample PINN solution data
    public static func generateSolutionData() -> [PINNSolutionPoint] {
        let xValues = Array(stride(from: -1.0, to: 1.0, by: 0.1))
        let pinnValues: [Double] = [0.0, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.4, 0.1, -0.3, -0.6, -0.8, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.0]
        let rk4Values: [Double] = [0.0, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.3, 0.0, -0.4, -0.7, -0.8, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.0]
        
        return zip(xValues, zip(pinnValues, rk4Values)).map { x, values in
            PINNSolutionPoint(
                x: x,
                pinnValue: values.0,
                rk4Value: values.1,
                time: 1.0
            )
        }
    }
    
    /// Generate sample training progress data
    public static func generateTrainingData() -> [PerformanceMetricsData] {
        return (0..<20).map { epoch in
            let pdeLoss = 0.1 * exp(-Double(epoch) * 0.1) + 0.01 * Double.random(in: 0...1)
            let icLoss = 0.05 * exp(-Double(epoch) * 0.15) + 0.005 * Double.random(in: 0...1)
            let totalLoss = pdeLoss + icLoss
            let performanceMetric = 0.8 * exp(-totalLoss) + 0.2 * Double.random(in: 0...1)
            
            return PerformanceMetricsData(
                epoch: epoch * 50,
                pdeLoss: pdeLoss,
                icLoss: icLoss,
                totalLoss: totalLoss,
                performanceMetric: performanceMetric
            )
        }
    }
}