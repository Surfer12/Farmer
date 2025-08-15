import SwiftUI
import SceneKit

/// Main SwiftUI view for CFD visualization and cognitive integration
struct CFDVisualizationView: View {
    @StateObject private var viewModel = CFDViewModel()
    @State private var selectedTab = 0
    @State private var showingPerformanceChart = false
    @State private var showingFlowMetrics = false
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Main 3D Visualization Tab
                mainVisualizationView
                    .tabItem {
                        Image(systemName: "cube.transparent")
                        Text("3D View")
                    }
                    .tag(0)
                
                // Performance Analysis Tab
                performanceAnalysisView
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("Performance")
                    }
                    .tag(1)
                
                // Flow State Tab
                flowStateView
                    .tabItem {
                        Image(systemName: "heart.fill")
                        Text("Flow State")
                    }
                    .tag(2)
                
                // Settings Tab
                settingsView
                    .tabItem {
                        Image(systemName: "gearshape.fill")
                        Text("Settings")
                    }
                    .tag(3)
            }
            .navigationTitle("CFD Visualization")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    realTimeModeToggle
                }
            }
        }
        .sheet(isPresented: $showingPerformanceChart) {
            PerformanceChartView(performanceCurve: viewModel.performanceCurve)
        }
        .sheet(isPresented: $showingFlowMetrics) {
            FlowMetricsDetailView(metrics: viewModel.flowStateMetrics)
        }
        .alert("Error", isPresented: .constant(viewModel.errorMessage != nil)) {
            Button("OK") { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }
    
    // MARK: - Main Visualization View
    
    private var mainVisualizationView: some View {
        VStack(spacing: 0) {
            // 3D SceneKit View
            sceneKitView
                .frame(height: 400)
                .background(Color.black)
                .cornerRadius(12)
                .shadow(radius: 5)
            
            // Control Panel
            controlPanel
                .padding(.horizontal)
            
            // Real-time Data Display
            realTimeDataPanel
                .padding(.horizontal)
            
            Spacer()
        }
        .padding()
    }
    
    private var sceneKitView: some View {
        SceneView(
            scene: createScene(),
            options: [.autoenablesDefaultLighting, .allowsCameraControl]
        )
        .overlay(alignment: .topLeading) {
            // Overlay information
            VStack(alignment: .leading, spacing: 4) {
                Text("Vector 3/2 Blackstix+")
                    .font(.caption)
                    .foregroundColor(.white)
                Text("AoA: \(viewModel.currentAngleOfAttack, specifier: "%.1f")°")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.8))
                if let flowMetrics = viewModel.flowStateMetrics {
                    Text("Flow: \(flowMetrics.flowState.description)")
                        .font(.caption2)
                        .foregroundColor(Color(
                            red: Double(flowMetrics.flowState.color.r),
                            green: Double(flowMetrics.flowState.color.g),
                            blue: Double(flowMetrics.flowState.color.b)
                        ))
                }
            }
            .padding(8)
            .background(.black.opacity(0.5))
            .cornerRadius(8)
            .padding()
        }
    }
    
    private var controlPanel: some View {
        VStack(spacing: 16) {
            // Angle of Attack Control
            VStack(alignment: .leading) {
                HStack {
                    Text("Angle of Attack")
                        .font(.headline)
                    Spacer()
                    Text("\(viewModel.currentAngleOfAttack, specifier: "%.1f")°")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.blue)
                }
                
                Slider(
                    value: Binding(
                        get: { viewModel.currentAngleOfAttack },
                        set: { viewModel.setAngleOfAttack($0) }
                    ),
                    in: -20...20,
                    step: 0.5
                ) {
                    Text("AoA")
                } minimumValueLabel: {
                    Text("-20°")
                        .font(.caption)
                } maximumValueLabel: {
                    Text("20°")
                        .font(.caption)
                }
                .disabled(viewModel.isRealTimeMode)
            }
            
            // Rake Angle Control
            VStack(alignment: .leading) {
                HStack {
                    Text("Rake Angle")
                        .font(.headline)
                    Spacer()
                    Text("\(viewModel.currentRakeAngle, specifier: "%.1f")°")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                }
                
                Slider(
                    value: Binding(
                        get: { viewModel.currentRakeAngle },
                        set: { viewModel.setRakeAngle($0) }
                    ),
                    in: 0...10,
                    step: 0.5
                ) {
                    Text("Rake")
                } minimumValueLabel: {
                    Text("0°")
                        .font(.caption)
                } maximumValueLabel: {
                    Text("10°")
                        .font(.caption)
                }
            }
            
            // Reynolds Number Control
            VStack(alignment: .leading) {
                HStack {
                    Text("Reynolds Number")
                        .font(.headline)
                    Spacer()
                    Text("\(viewModel.reynoldsNumber, specifier: "%.0e")")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                }
                
                Slider(
                    value: Binding(
                        get: { log10(viewModel.reynoldsNumber) },
                        set: { viewModel.setReynoldsNumber(pow(10, $0)) }
                    ),
                    in: 4...7,
                    step: 0.1
                ) {
                    Text("Re")
                } minimumValueLabel: {
                    Text("10⁴")
                        .font(.caption)
                } maximumValueLabel: {
                    Text("10⁷")
                        .font(.caption)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var realTimeDataPanel: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Performance Metrics")
                    .font(.headline)
                Spacer()
                if viewModel.isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            HStack(spacing: 20) {
                // Lift Coefficient
                VStack {
                    Text("Lift Coeff.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.liftCoefficient, specifier: "%.3f")")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                
                Divider()
                
                // Drag Coefficient
                VStack {
                    Text("Drag Coeff.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.dragCoefficient, specifier: "%.3f")")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.red)
                }
                
                Divider()
                
                // L/D Ratio
                VStack {
                    Text("L/D Ratio")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.liftToDragRatio, specifier: "%.1f")")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
            }
            
            // Force Display
            let forces = viewModel.getCurrentForces()
            HStack(spacing: 20) {
                VStack {
                    Text("Lift Force")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(forces.lift, specifier: "%.1f") N")
                        .font(.callout)
                        .fontWeight(.semibold)
                }
                
                VStack {
                    Text("Drag Force")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(forces.drag, specifier: "%.1f") N")
                        .font(.callout)
                        .fontWeight(.semibold)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Performance Analysis View
    
    private var performanceAnalysisView: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Performance Chart Button
                Button(action: { showingPerformanceChart = true }) {
                    HStack {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("View Performance Curves")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                
                // Performance Comparison
                performanceComparisonView
                
                // Fin Configuration Details
                finConfigurationView
            }
            .padding()
        }
        .navigationTitle("Performance Analysis")
    }
    
    private var performanceComparisonView: some View {
        let comparison = viewModel.getPerformanceComparison()
        
        return VStack(alignment: .leading, spacing: 12) {
            Text("Vector 3/2 vs Baseline")
                .font(.headline)
            
            HStack(spacing: 20) {
                VStack(alignment: .leading) {
                    Text("Lift Improvement")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("+\(comparison.liftImprovement * 100, specifier: "%.1f")%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
                
                VStack(alignment: .leading) {
                    Text("Drag Reduction")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("-\(comparison.dragReduction * 100, specifier: "%.1f")%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                
                VStack(alignment: .leading) {
                    Text("Efficiency Gain")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("+\(comparison.efficiencyGain * 100, specifier: "%.1f")%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var finConfigurationView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Fin Configuration")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Side Fins:")
                        .fontWeight(.semibold)
                    Spacer()
                    Text("15.00 sq.in, 6.5° rake, Vector 3/2 foil")
                        .font(.callout)
                }
                
                HStack {
                    Text("Center Fin:")
                        .fontWeight(.semibold)
                    Spacer()
                    Text("14.50 sq.in, symmetric foil")
                        .font(.callout)
                }
                
                HStack {
                    Text("Rider Weight:")
                        .fontWeight(.semibold)
                    Spacer()
                    Text("\(viewModel.riderWeight, specifier: "%.0f") lbs")
                        .font(.callout)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Flow State View
    
    private var flowStateView: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Flow State Indicator
                if let metrics = viewModel.flowStateMetrics {
                    flowStateIndicator(metrics: metrics)
                } else {
                    Text("Connect sensors to view flow state")
                        .font(.headline)
                        .foregroundColor(.secondary)
                        .padding()
                }
                
                // Detailed Metrics Button
                Button(action: { showingFlowMetrics = true }) {
                    HStack {
                        Image(systemName: "heart.text.square")
                        Text("Detailed Metrics")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                
                // Cognitive Load Explanation
                cognitiveLoadExplanation
            }
            .padding()
        }
        .navigationTitle("Flow State")
    }
    
    private func flowStateIndicator(metrics: FlowStateMetrics) -> some View {
        VStack(spacing: 16) {
            // Flow State Circle
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 20)
                    .frame(width: 200, height: 200)
                
                Circle()
                    .trim(from: 0, to: CGFloat(metrics.flowScore))
                    .stroke(
                        Color(
                            red: Double(metrics.flowState.color.r),
                            green: Double(metrics.flowState.color.g),
                            blue: Double(metrics.flowState.color.b)
                        ),
                        style: StrokeStyle(lineWidth: 20, lineCap: .round)
                    )
                    .frame(width: 200, height: 200)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 1.0), value: metrics.flowScore)
                
                VStack {
                    Text("\(metrics.flowScore * 100, specifier: "%.0f")%")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    Text(metrics.flowState.description)
                        .font(.headline)
                        .multilineTextAlignment(.center)
                }
            }
            
            // Metrics Breakdown
            HStack(spacing: 20) {
                VStack {
                    Text("Heart Rate")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(metrics.heartRate, specifier: "%.0f") BPM")
                        .font(.callout)
                        .fontWeight(.semibold)
                }
                
                VStack {
                    Text("HRV")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(metrics.hrv, specifier: "%.1f") ms")
                        .font(.callout)
                        .fontWeight(.semibold)
                }
                
                VStack {
                    Text("Stability")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(metrics.motionStability * 100, specifier: "%.0f")%")
                        .font(.callout)
                        .fontWeight(.semibold)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var cognitiveLoadExplanation: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Understanding Flow State")
                .font(.headline)
            
            Text("Flow state represents the optimal balance between challenge and skill, where you perform at your peak with minimal cognitive load.")
                .font(.body)
                .foregroundColor(.secondary)
            
            VStack(alignment: .leading, spacing: 8) {
                flowStateExplanationRow(color: .green, state: "Optimal", description: "Peak performance, minimal effort")
                flowStateExplanationRow(color: .yellow, state: "Good", description: "High performance, moderate effort")
                flowStateExplanationRow(color: .orange, state: "Moderate", description: "Average performance, increased effort")
                flowStateExplanationRow(color: .red, state: "Poor", description: "Struggling, high cognitive load")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func flowStateExplanationRow(color: Color, state: String, description: String) -> some View {
        HStack {
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
            Text(state)
                .fontWeight(.semibold)
            Text("-")
                .foregroundColor(.secondary)
            Text(description)
                .foregroundColor(.secondary)
        }
        .font(.callout)
    }
    
    // MARK: - Settings View
    
    private var settingsView: some View {
        Form {
            Section("Rider Configuration") {
                HStack {
                    Text("Weight")
                    Spacer()
                    TextField("Weight", value: Binding(
                        get: { viewModel.riderWeight },
                        set: { viewModel.riderWeight = $0 }
                    ), format: .number)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    Text("lbs")
                        .foregroundColor(.secondary)
                }
            }
            
            Section("Sensor Settings") {
                Toggle("Real-time Mode", isOn: Binding(
                    get: { viewModel.isRealTimeMode },
                    set: { _ in toggleRealTimeMode() }
                ))
                
                HStack {
                    Text("Sensor Status")
                    Spacer()
                    // This would show actual sensor status
                    Text("Connected")
                        .foregroundColor(.green)
                }
            }
            
            Section("Visualization") {
                Toggle("Show Flow Vectors", isOn: .constant(true))
                Toggle("Show Pressure Maps", isOn: .constant(true))
                Toggle("Auto-rotate View", isOn: .constant(false))
            }
            
            Section("About") {
                HStack {
                    Text("App Version")
                    Spacer()
                    Text("1.0.0")
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text("CFD Model")
                    Spacer()
                    Text("Vector 3/2 Blackstix+")
                        .foregroundColor(.secondary)
                }
            }
        }
        .navigationTitle("Settings")
    }
    
    // MARK: - Helper Views
    
    private var realTimeModeToggle: some View {
        Button(action: toggleRealTimeMode) {
            HStack {
                Image(systemName: viewModel.isRealTimeMode ? "pause.fill" : "play.fill")
                Text(viewModel.isRealTimeMode ? "Pause" : "Live")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(viewModel.isRealTimeMode ? Color.red : Color.green)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
    }
    
    // MARK: - Helper Methods
    
    private func createScene() -> SCNScene {
        let scene = SCNScene()
        let finVisualizer = viewModel.getFinVisualizer()
        scene.rootNode.addChildNode(finVisualizer)
        
        // Set up camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 5, z: 10)
        cameraNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(cameraNode)
        
        return scene
    }
    
    private func toggleRealTimeMode() {
        if viewModel.isRealTimeMode {
            viewModel.stopRealTimeMode()
        } else {
            viewModel.startRealTimeMode()
        }
    }
}

// MARK: - Preview

#if DEBUG
struct CFDVisualizationView_Previews: PreviewProvider {
    static var previews: some View {
        CFDVisualizationView()
    }
}
#endif