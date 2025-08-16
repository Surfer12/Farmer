import SwiftUI
import Charts

struct CognitiveDetailView: View {
    @ObservedObject var viewModel: FinViewModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Cognitive Overview
                    cognitiveOverviewSection
                    
                    // HRV Analysis
                    hrvAnalysisSection
                    
                    // Flow State Assessment
                    flowStateSection
                    
                    // Cognitive Load Analysis
                    cognitiveLoadSection
                    
                    // Recommendations
                    recommendationsSection
                    
                    Spacer(minLength: 20)
                }
                .padding(.horizontal)
            }
            .navigationTitle("Cognitive Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    // MARK: - Cognitive Overview Section
    
    private var cognitiveOverviewSection: some View {
        VStack(spacing: 16) {
            Text("Cognitive Performance Overview")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            let metrics = viewModel.getPerformanceMetrics()
            
            VStack(spacing: 12) {
                HStack {
                    Text("Overall Cognitive Score")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(String(format: "%.1f", (1.0 - metrics.cognitiveLoad) * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
                
                ProgressView(value: 1.0 - metrics.cognitiveLoad)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                
                HStack {
                    Text("Current Flow State")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(metrics.flowState.rawValue)
                        .font(.subheadline)
                        .foregroundColor(metrics.flowState.color)
                }
                
                HStack {
                    Text("HRV Status")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    if let hrv = viewModel.hrv {
                        Text(hrv > 50.0 ? "Optimal" : hrv > 30.0 ? "Good" : "Attention")
                            .font(.subheadline)
                            .foregroundColor(hrv > 50.0 ? .green : hrv > 30.0 ? .blue : .orange)
                    } else {
                        Text("No Data")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - HRV Analysis Section
    
    private var hrvAnalysisSection: some View {
        VStack(spacing: 16) {
            Text("Heart Rate Variability Analysis")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                if let hrv = viewModel.hrv {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Current HRV")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("\(hrv, specifier: "%.1f") ms")
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.purple)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 4) {
                            Text("Status")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text(hrvStatusText(hrv))
                                .font(.subheadline)
                                .foregroundColor(hrvStatusColor(hrv))
                        }
                    }
                    
                    // HRV interpretation
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Interpretation:")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Text(hrvInterpretation(hrv))
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.leading)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "heart.slash")
                            .font(.title)
                            .foregroundColor(.secondary)
                        
                        Text("No HRV Data Available")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Text("Connect your Apple Watch or compatible heart rate monitor to view HRV data")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - Flow State Section
    
    private var flowStateSection: some View {
        VStack(spacing: 16) {
            Text("Flow State Assessment")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                let metrics = viewModel.getPerformanceMetrics()
                
                FlowStateCard(
                    state: metrics.flowState,
                    confidence: metrics.confidence,
                    description: metrics.flowState.description
                )
                
                // Flow state indicators
                VStack(alignment: .leading, spacing: 8) {
                    Text("Flow State Indicators:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    VStack(alignment: .leading, spacing: 6) {
                        FlowIndicator(
                            text: "Optimal: HRV > 50ms, Low cognitive load",
                            isActive: metrics.flowState == .optimal,
                            color: .green
                        )
                        FlowIndicator(
                            text: "Good: HRV > 30ms, Moderate cognitive load",
                            isActive: metrics.flowState == .good,
                            color: .blue
                        )
                        FlowIndicator(
                            text: "Moderate: HRV > 20ms, Higher cognitive load",
                            isActive: metrics.flowState == .moderate,
                            color: .yellow
                        )
                        FlowIndicator(
                            text: "Challenging: HRV < 20ms, High cognitive load",
                            isActive: metrics.flowState == .challenging,
                            color: .orange
                        )
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - Cognitive Load Section
    
    private var cognitiveLoadSection: some View {
        VStack(spacing: 16) {
            Text("Cognitive Load Analysis")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                let metrics = viewModel.getPerformanceMetrics()
                
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Current Load")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Text("\(metrics.cognitiveLoad, specifier: "%.2f")")
                            .font(.title)
                            .fontWeight(.bold)
                            .foregroundColor(.orange)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Status")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Text(cognitiveLoadStatus(metrics.cognitiveLoad))
                            .font(.subheadline)
                            .foregroundColor(cognitiveLoadColor(metrics.cognitiveLoad))
                    }
                }
                
                // Cognitive load breakdown
                VStack(alignment: .leading, spacing: 8) {
                    Text("Load Factors:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    VStack(alignment: .leading, spacing: 6) {
                        LoadFactor(
                            text: "Pressure complexity: \(String(format: "%.1f", metrics.pressureRange * 100))%",
                            value: metrics.pressureRange,
                            color: .blue
                        )
                        LoadFactor(
                            text: "Performance variance: \(String(format: "%.1f", (1.0 - metrics.confidence) * 100))%",
                            value: 1.0 - metrics.confidence,
                            color: .red
                        )
                        LoadFactor(
                            text: "Flow state stress: \(String(format: "%.1f", metrics.flowState == .challenging ? 0.8 : 0.2))",
                            value: metrics.flowState == .challenging ? 0.8 : 0.2,
                            color: .purple
                        )
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - Recommendations Section
    
    private var recommendationsSection: some View {
        VStack(spacing: 16) {
            Text("Performance Recommendations")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                let metrics = viewModel.getPerformanceMetrics()
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Based on current metrics:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    VStack(alignment: .leading, spacing: 6) {
                        if metrics.flowState == .optimal {
                            RecommendationBullet(
                                text: "Maintain current technique and breathing rhythm",
                                color: .green
                            )
                            RecommendationBullet(
                                text: "Consider pushing performance boundaries",
                                color: .green
                            )
                        } else if metrics.flowState == .good {
                            RecommendationBullet(
                                text: "Focus on smooth, controlled movements",
                                color: .blue
                            )
                            RecommendationBullet(
                                text: "Maintain consistent breathing pattern",
                                color: .blue
                            )
                        } else if metrics.flowState == .moderate {
                            RecommendationBullet(
                                text: "Take deep breaths to reduce cognitive load",
                                color: .yellow
                            )
                            RecommendationBullet(
                                text: "Simplify technique and focus on fundamentals",
                                color: .yellow
                            )
                        } else {
                            RecommendationBullet(
                                text: "Take a moment to reset and breathe",
                                color: .orange
                            )
                            RecommendationBullet(
                                text: "Consider technique adjustments for current conditions",
                                color: .orange
                            )
                        }
                        
                        // General recommendations
                        RecommendationBullet(
                            text: "Monitor HRV trends for optimal performance timing",
                            color: .purple
                        )
                        RecommendationBullet(
                            text: "Adjust turn angles based on pressure feedback",
                            color: .blue
                        )
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - Helper Methods
    
    private func hrvStatusText(_ hrv: Double) -> String {
        if hrv > 50.0 { return "Optimal" }
        else if hrv > 30.0 { return "Good" }
        else if hrv > 20.0 { return "Moderate" }
        else { return "Attention" }
    }
    
    private func hrvStatusColor(_ hrv: Double) -> Color {
        if hrv > 50.0 { return .green }
        else if hrv > 30.0 { return .blue }
        else if hrv > 20.0 { return .yellow }
        else { return .orange }
    }
    
    private func hrvInterpretation(_ hrv: Double) -> String {
        if hrv > 50.0 {
            return "Excellent autonomic nervous system balance. You're in an optimal state for peak performance with minimal cognitive load."
        } else if hrv > 30.0 {
            return "Good autonomic balance. You're maintaining good flow state with moderate cognitive efficiency."
        } else if hrv > 20.0 {
            return "Moderate autonomic balance. Consider focusing on breathing and technique to improve flow state."
        } else {
            return "Reduced autonomic balance. Focus on relaxation and breathing to optimize performance."
        }
    }
    
    private func cognitiveLoadStatus(_ load: Double) -> String {
        if load < 0.3 { return "Low" }
        else if load < 0.6 { return "Moderate" }
        else { return "High" }
    }
    
    private func cognitiveLoadColor(_ load: Double) -> Color {
        if load < 0.3 { return .green }
        else if load < 0.6 { return .yellow }
        else { return .orange }
    }
}

// MARK: - Supporting Views

struct FlowStateCard: View {
    let state: FlowState
    let confidence: Double
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(state.rawValue)
                    .font(.headline)
                    .foregroundColor(state.color)
                Spacer()
                Text("\(String(format: "%.0f", confidence * 100))% confidence")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.leading)
        }
        .padding()
        .background(state.color.opacity(0.1))
        .cornerRadius(8)
    }
}

struct FlowIndicator: View {
    let text: String
    let isActive: Bool
    let color: Color
    
    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(isActive ? color : Color.gray.opacity(0.3))
                .frame(width: 8, height: 8)
            
            Text(text)
                .font(.caption)
                .foregroundColor(isActive ? .primary : .secondary)
        }
    }
}

struct LoadFactor: View {
    let text: String
    let value: Double
    let color: Color
    
    var body: some View {
        HStack {
            Text(text)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            ProgressView(value: value)
                .progressViewStyle(LinearProgressViewStyle(tint: color))
                .frame(width: 60)
        }
    }
}

struct RecommendationBullet: View {
    let text: String
    let color: Color
    
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
                .padding(.top, 6)
            
            Text(text)
                .font(.caption)
                .foregroundColor(.primary)
                .multilineTextAlignment(.leading)
        }
    }
}

struct CognitiveDetailView_Previews: PreviewProvider {
    static var previews: some View {
        CognitiveDetailView(viewModel: FinViewModel())
    }
}
