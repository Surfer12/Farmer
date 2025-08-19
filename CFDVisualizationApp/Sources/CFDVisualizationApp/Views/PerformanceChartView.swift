import SwiftUI
import Charts

/// Performance chart view displaying lift/drag curves across angle of attack range
struct PerformanceChartView: View {
    let performanceCurve: [CFDPrediction]
    @Environment(\.dismiss) private var dismiss
    @State private var selectedMetric: PerformanceMetric = .liftCoefficient
    @State private var showingComparison = false
    
    enum PerformanceMetric: String, CaseIterable {
        case liftCoefficient = "Lift Coefficient"
        case dragCoefficient = "Drag Coefficient"
        case liftToDragRatio = "L/D Ratio"
        case efficiency = "Efficiency"
        
        var color: Color {
            switch self {
            case .liftCoefficient: return .blue
            case .dragCoefficient: return .red
            case .liftToDragRatio: return .green
            case .efficiency: return .orange
            }
        }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Metric Selector
                metricSelector
                
                // Performance Chart
                performanceChart
                    .frame(height: 300)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                
                // Performance Summary
                performanceSummary
                
                // Comparison Toggle
                Toggle("Show Baseline Comparison", isOn: $showingComparison)
                    .padding(.horizontal)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Performance Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
    
    private var metricSelector: some View {
        Picker("Performance Metric", selection: $selectedMetric) {
            ForEach(PerformanceMetric.allCases, id: \.self) { metric in
                Text(metric.rawValue).tag(metric)
            }
        }
        .pickerStyle(SegmentedPickerStyle())
        .padding(.horizontal)
    }
    
    @ViewBuilder
    private var performanceChart: some View {
        if #available(iOS 16.0, macOS 13.0, *) {
            Chart {
                ForEach(performanceCurve, id: \.timestamp) { prediction in
                    LineMark(
                        x: .value("Angle of Attack", prediction.angleOfAttack),
                        y: .value(selectedMetric.rawValue, getMetricValue(prediction: prediction))
                    )
                    .foregroundStyle(selectedMetric.color)
                    .lineStyle(StrokeStyle(lineWidth: 3))
                    
                    // Add points
                    PointMark(
                        x: .value("Angle of Attack", prediction.angleOfAttack),
                        y: .value(selectedMetric.rawValue, getMetricValue(prediction: prediction))
                    )
                    .foregroundStyle(selectedMetric.color)
                    .symbolSize(30)
                }
                
                // Baseline comparison if enabled
                if showingComparison {
                    ForEach(generateBaselineData(), id: \.timestamp) { prediction in
                        LineMark(
                            x: .value("Angle of Attack", prediction.angleOfAttack),
                            y: .value("Baseline", getMetricValue(prediction: prediction))
                        )
                        .foregroundStyle(.gray)
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                    }
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: 5)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let angle = value.as(Double.self) {
                            Text("\(angle, specifier: "%.0f")°")
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartXAxisLabel("Angle of Attack (degrees)")
            .chartYAxisLabel(selectedMetric.rawValue)
        } else {
            // Fallback for older iOS versions
            Text("Performance charts require iOS 16.0 or later")
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
    
    private var performanceSummary: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Performance Summary")
                .font(.headline)
            
            if let optimalPrediction = findOptimalPerformance() {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Optimal AoA:")
                            .fontWeight(.semibold)
                        Spacer()
                        Text("\(optimalPrediction.angleOfAttack, specifier: "%.1f")°")
                            .foregroundColor(.blue)
                    }
                    
                    HStack {
                        Text("Max L/D Ratio:")
                            .fontWeight(.semibold)
                        Spacer()
                        Text("\(optimalPrediction.liftToDragRatio, specifier: "%.2f")")
                            .foregroundColor(.green)
                    }
                    
                    HStack {
                        Text("Performance Gain:")
                            .fontWeight(.semibold)
                        Spacer()
                        Text("+\(optimalPrediction.performanceGain * 100, specifier: "%.1f")%")
                            .foregroundColor(.orange)
                    }
                    
                    HStack {
                        Text("Confidence:")
                            .fontWeight(.semibold)
                        Spacer()
                        Text("\(optimalPrediction.confidence * 100, specifier: "%.0f")%")
                            .foregroundColor(optimalPrediction.confidence > 0.8 ? .green : .orange)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Helper Methods
    
    private func getMetricValue(prediction: CFDPrediction) -> Double {
        switch selectedMetric {
        case .liftCoefficient:
            return Double(prediction.liftCoefficient)
        case .dragCoefficient:
            return Double(prediction.dragCoefficient)
        case .liftToDragRatio:
            return Double(prediction.liftToDragRatio)
        case .efficiency:
            return Double(prediction.liftCoefficient / max(0.001, prediction.dragCoefficient))
        }
    }
    
    private func findOptimalPerformance() -> CFDPrediction? {
        return performanceCurve.max { a, b in
            a.liftToDragRatio < b.liftToDragRatio
        }
    }
    
    private func generateBaselineData() -> [CFDPrediction] {
        // Generate baseline performance data (without Vector 3/2 improvements)
        return performanceCurve.map { prediction in
            CFDPrediction(
                angleOfAttack: prediction.angleOfAttack,
                reynoldsNumber: prediction.reynoldsNumber,
                liftCoefficient: prediction.liftCoefficient * 0.88, // Remove 12% improvement
                dragCoefficient: prediction.dragCoefficient * 1.05, // Increase drag
                sideFinContribution: (0, 0),
                centerFinContribution: (0, 0),
                confidence: prediction.confidence * 0.9
            )
        }
    }
}

/// Detailed flow metrics view
struct FlowMetricsDetailView: View {
    let metrics: FlowStateMetrics?
    @Environment(\.dismiss) private var dismiss
    @State private var showingHistory = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    if let metrics = metrics {
                        // Current Flow State
                        currentFlowStateCard(metrics: metrics)
                        
                        // Detailed Metrics
                        detailedMetricsCard(metrics: metrics)
                        
                        // Recommendations
                        recommendationsCard(metrics: metrics)
                        
                        // History Button
                        Button(action: { showingHistory = true }) {
                            HStack {
                                Image(systemName: "clock.arrow.circlepath")
                                Text("View History")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                    } else {
                        Text("No flow metrics available")
                            .font(.headline)
                            .foregroundColor(.secondary)
                            .padding()
                    }
                }
                .padding()
            }
            .navigationTitle("Flow Metrics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .sheet(isPresented: $showingHistory) {
            FlowHistoryView()
        }
    }
    
    private func currentFlowStateCard(metrics: FlowStateMetrics) -> some View {
        VStack(spacing: 16) {
            Text("Current Flow State")
                .font(.headline)
            
            // Flow state indicator
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 15)
                    .frame(width: 150, height: 150)
                
                Circle()
                    .trim(from: 0, to: CGFloat(metrics.flowScore))
                    .stroke(
                        Color(
                            red: Double(metrics.flowState.color.r),
                            green: Double(metrics.flowState.color.g),
                            blue: Double(metrics.flowState.color.b)
                        ),
                        style: StrokeStyle(lineWidth: 15, lineCap: .round)
                    )
                    .frame(width: 150, height: 150)
                    .rotationEffect(.degrees(-90))
                
                VStack {
                    Text("\(metrics.flowScore * 100, specifier: "%.0f")%")
                        .font(.title)
                        .fontWeight(.bold)
                    Text(metrics.flowState.description)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                }
            }
            
            Text("Last updated: \(metrics.timestamp, style: .time)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func detailedMetricsCard(metrics: FlowStateMetrics) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Detailed Metrics")
                .font(.headline)
            
            VStack(spacing: 12) {
                metricRow(title: "Heart Rate", value: "\(metrics.heartRate, specifier: "%.0f") BPM", color: .red)
                metricRow(title: "Heart Rate Variability", value: "\(metrics.hrv, specifier: "%.1f") ms", color: .orange)
                metricRow(title: "Motion Stability", value: "\(metrics.motionStability * 100, specifier: "%.0f")%", color: .blue)
                metricRow(title: "Focus Level", value: "\(metrics.focusLevel * 100, specifier: "%.0f")%", color: .green)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func metricRow(title: String, value: String, color: Color) -> some View {
        HStack {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(title)
                .fontWeight(.medium)
            Spacer()
            Text(value)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
    }
    
    private func recommendationsCard(metrics: FlowStateMetrics) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recommendations")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                ForEach(generateRecommendations(metrics: metrics), id: \.self) { recommendation in
                    HStack(alignment: .top) {
                        Image(systemName: "lightbulb.fill")
                            .foregroundColor(.yellow)
                            .frame(width: 20)
                        Text(recommendation)
                            .font(.callout)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func generateRecommendations(metrics: FlowStateMetrics) -> [String] {
        var recommendations: [String] = []
        
        switch metrics.flowState {
        case .optimal:
            recommendations.append("Maintain current rhythm and breathing pattern")
            recommendations.append("Focus on smooth, controlled movements")
        case .good:
            recommendations.append("Slightly reduce effort to enter optimal flow")
            recommendations.append("Pay attention to breath control")
        case .moderate:
            recommendations.append("Take deeper breaths to reduce stress")
            recommendations.append("Focus on technique over power")
        case .poor:
            recommendations.append("Consider taking a short break")
            recommendations.append("Focus on relaxation and breath control")
            recommendations.append("Reduce intensity until flow improves")
        }
        
        if metrics.heartRate > 170 {
            recommendations.append("Heart rate is high - consider reducing intensity")
        }
        
        if metrics.motionStability < 0.5 {
            recommendations.append("Focus on stability and balance")
        }
        
        return recommendations
    }
}

/// Flow history view (placeholder)
struct FlowHistoryView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                Text("Flow State History")
                    .font(.title)
                
                Text("Historical flow state data would be displayed here")
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding()
                
                Spacer()
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#if DEBUG
struct PerformanceChartView_Previews: PreviewProvider {
    static var previews: some View {
        PerformanceChartView(performanceCurve: [
            CFDPrediction(
                angleOfAttack: 5.0,
                reynoldsNumber: 5e5,
                liftCoefficient: 0.8,
                dragCoefficient: 0.05,
                sideFinContribution: (0.8, 0.05),
                centerFinContribution: (0.7, 0.04),
                confidence: 0.9
            )
        ])
    }
}
#endif