import SwiftUI
import SceneKit
import Combine

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Main 3D Visualization Tab
            MainVisualizationView(viewModel: viewModel)
                .tabItem {
                    Image(systemName: "cube.transparent")
                    Text("3D View")
                }
                .tag(0)
            
            // Data Analytics Tab
            AnalyticsView(viewModel: viewModel)
                .tabItem {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                    Text("Analytics")
                }
                .tag(1)
            
            // Cognitive Metrics Tab
            CognitiveView(viewModel: viewModel)
                .tabItem {
                    Image(systemName: "brain.head.profile")
                    Text("Flow State")
                }
                .tag(2)
            
            // Settings Tab
            SettingsView(viewModel: viewModel)
                .tabItem {
                    Image(systemName: "gearshape")
                    Text("Settings")
                }
                .tag(3)
        }
        .accentColor(.blue)
        .onAppear {
            viewModel.startMonitoring()
        }
    }
}

// MARK: - Main Visualization View

struct MainVisualizationView: View {
    @ObservedObject var viewModel: FinViewModel
    @State private var isControlsExpanded = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.cyan.opacity(0.05)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Header
                    headerView
                    
                    // 3D Visualization
                    Fin3DView(visualizer: viewModel.visualizer)
                        .frame(height: geometry.size.height * 0.5)
                        .background(Color.black.opacity(0.05))
                        .cornerRadius(15)
                        .shadow(radius: 10)
                        .padding(.horizontal)
                    
                    // Controls and Metrics
                    controlsAndMetricsView
                        .padding(.top)
                    
                    Spacer()
                }
            }
        }
        .navigationBarHidden(true)
    }
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Vector 3/2 CFD Visualizer")
                    .font(.title2)
                    .fontWeight(.bold)
                
                HStack {
                    Circle()
                        .fill(viewModel.sensorManager.sensorStatus.color)
                        .frame(width: 8, height: 8)
                    
                    Text(viewModel.sensorManager.sensorStatus.rawValue)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // Real-time indicators
            HStack(spacing: 16) {
                VStack {
                    Text("\(Int(viewModel.turnAngle))")
                        .font(.title3)
                        .fontWeight(.semibold)
                    Text("AoA°")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                VStack {
                    Text("\(viewModel.flowStateIndex, specifier: "%.2f")")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(flowStateColor)
                    Text("Flow")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal)
        .padding(.top)
    }
    
    private var controlsAndMetricsView: some View {
        VStack(spacing: 20) {
            // Angle of Attack Control
            VStack {
                HStack {
                    Text("Angle of Attack")
                        .font(.headline)
                    Spacer()
                    Text("\(Int(viewModel.turnAngle))°")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                
                Slider(value: $viewModel.turnAngle, in: 0...20, step: 1) {
                    Text("AoA")
                } minimumValueLabel: {
                    Text("0°")
                        .font(.caption)
                } maximumValueLabel: {
                    Text("20°")
                        .font(.caption)
                }
                .accentColor(.blue)
            }
            .padding()
            .background(Color.white.opacity(0.8))
            .cornerRadius(12)
            .shadow(radius: 5)
            
            // Performance Metrics
            performanceMetricsView
            
            // Quick Actions
            quickActionsView
        }
        .padding(.horizontal)
    }
    
    private var performanceMetricsView: some View {
        HStack(spacing: 20) {
            MetricCard(
                title: "Lift",
                value: viewModel.liftDrag?.lift ?? 0,
                unit: "N",
                color: .green,
                icon: "arrow.up"
            )
            
            MetricCard(
                title: "Drag",
                value: viewModel.liftDrag?.drag ?? 0,
                unit: "N",
                color: .orange,
                icon: "arrow.left"
            )
            
            MetricCard(
                title: "L/D",
                value: viewModel.liftDrag?.liftToDragRatio ?? 0,
                unit: "",
                color: .purple,
                icon: "divide"
            )
        }
    }
    
    private var quickActionsView: some View {
        HStack(spacing: 16) {
            ActionButton(
                title: "Calibrate",
                icon: "gyroscope",
                action: {
                    viewModel.calibrateSensors()
                }
            )
            
            ActionButton(
                title: "Reset View",
                icon: "arrow.counterclockwise",
                action: {
                    viewModel.resetVisualization()
                }
            )
            
            ActionButton(
                title: "Export Data",
                icon: "square.and.arrow.up",
                action: {
                    viewModel.exportData()
                }
            )
        }
    }
    
    private var flowStateColor: Color {
        switch viewModel.flowStateIndex {
        case 0.8...1.0:
            return .green
        case 0.5..<0.8:
            return .yellow
        default:
            return .red
        }
    }
}

// MARK: - 3D View Wrapper

struct Fin3DView: UIViewRepresentable {
    let visualizer: FinVisualizer
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = visualizer.scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = false // We have custom lighting
        scnView.backgroundColor = UIColor.clear
        scnView.antialiasingMode = .multisampling4X
        
        // Add gesture recognizers
        let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap))
        scnView.addGestureRecognizer(tapGesture)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Updates handled by the visualizer
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject {
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            // Handle tap interactions with 3D scene
            print("3D scene tapped")
        }
    }
}

// MARK: - Analytics View

struct AnalyticsView: View {
    @ObservedObject var viewModel: FinViewModel
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Performance Chart
                    performanceChartView
                    
                    // Sensor Statistics
                    sensorStatisticsView
                    
                    // Prediction History
                    predictionHistoryView
                    
                    // Flow State Timeline
                    flowStateTimelineView
                }
                .padding()
            }
            .navigationTitle("Analytics")
        }
    }
    
    private var performanceChartView: some View {
        VStack(alignment: .leading) {
            Text("Performance Metrics")
                .font(.headline)
                .padding(.bottom)
            
            // Placeholder for chart - would use Charts framework in iOS 16+
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.blue.opacity(0.1))
                .frame(height: 200)
                .overlay(
                    VStack {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.system(size: 40))
                            .foregroundColor(.blue)
                        Text("Lift/Drag vs Angle of Attack")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                )
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var sensorStatisticsView: some View {
        VStack(alignment: .leading) {
            Text("Sensor Statistics")
                .font(.headline)
                .padding(.bottom)
            
            let stats = viewModel.sensorManager.getSensorStatistics()
            
            HStack(spacing: 20) {
                StatCard(title: "Total Readings", value: "\(stats.totalReadings)")
                StatCard(title: "Avg Turn Angle", value: "\(stats.averageTurnAngle, specifier: "%.1f")°")
            }
            
            HStack(spacing: 20) {
                StatCard(title: "Max Turn Angle", value: "\(stats.maxTurnAngle, specifier: "%.1f")°")
                StatCard(title: "Avg Pressure", value: "\(stats.averagePressure, specifier: "%.2f")")
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var predictionHistoryView: some View {
        VStack(alignment: .leading) {
            Text("Recent Predictions")
                .font(.headline)
                .padding(.bottom)
            
            ForEach(viewModel.predictor.predictionHistory.suffix(5), id: \.id) { prediction in
                PredictionRow(prediction: prediction)
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var flowStateTimelineView: some View {
        VStack(alignment: .leading) {
            Text("Flow State Timeline")
                .font(.headline)
                .padding(.bottom)
            
            // Placeholder for flow state timeline
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.green.opacity(0.1))
                .frame(height: 150)
                .overlay(
                    VStack {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 30))
                            .foregroundColor(.green)
                        Text("Flow State: \(viewModel.flowStateIndex, specifier: "%.2f")")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                )
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
}

// MARK: - Cognitive View

struct CognitiveView: View {
    @ObservedObject var viewModel: FinViewModel
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Current State Card
                    currentStateCard
                    
                    // HRV Metrics
                    hrvMetricsView
                    
                    // Flow State Indicator
                    flowStateIndicatorView
                    
                    // Recommendations
                    recommendationsView
                }
                .padding()
            }
            .navigationTitle("Flow State")
        }
    }
    
    private var currentStateCard: some View {
        VStack(spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Current Flow State")
                        .font(.headline)
                    Text(flowStateDescription)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                ZStack {
                    Circle()
                        .stroke(Color.gray.opacity(0.3), lineWidth: 8)
                        .frame(width: 80, height: 80)
                    
                    Circle()
                        .trim(from: 0, to: viewModel.flowStateIndex)
                        .stroke(flowStateColor, lineWidth: 8)
                        .frame(width: 80, height: 80)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut, value: viewModel.flowStateIndex)
                    
                    Text("\(Int(viewModel.flowStateIndex * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(flowStateColor)
                }
            }
            
            HStack(spacing: 20) {
                VStack {
                    Text("\(viewModel.currentHRV, specifier: "%.1f")")
                        .font(.title3)
                        .fontWeight(.semibold)
                    Text("HRV (ms)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                VStack {
                    Text("\(viewModel.heartRate, specifier: "%.0f")")
                        .font(.title3)
                        .fontWeight(.semibold)
                    Text("HR (bpm)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                VStack {
                    Text("\(viewModel.reactionTime, specifier: "%.0f")")
                        .font(.title3)
                        .fontWeight(.semibold)
                    Text("RT (ms)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var hrvMetricsView: some View {
        VStack(alignment: .leading) {
            Text("HRV Analysis")
                .font(.headline)
                .padding(.bottom)
            
            let stats = viewModel.cognitiveTracker.getHRVStatistics()
            
            HStack(spacing: 15) {
                HRVStatView(title: "Average", value: stats.average, unit: "ms")
                HRVStatView(title: "Range", value: stats.maximum - stats.minimum, unit: "ms")
                HRVStatView(title: "Readings", value: Double(stats.readingCount), unit: "")
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var flowStateIndicatorView: some View {
        VStack(alignment: .leading) {
            Text("Flow State Zones")
                .font(.headline)
                .padding(.bottom)
            
            VStack(spacing: 10) {
                FlowZoneIndicator(
                    title: "Optimal Flow",
                    range: "0.8 - 1.0",
                    color: .green,
                    isActive: viewModel.flowStateIndex >= 0.8
                )
                
                FlowZoneIndicator(
                    title: "Good Focus",
                    range: "0.5 - 0.8",
                    color: .yellow,
                    isActive: viewModel.flowStateIndex >= 0.5 && viewModel.flowStateIndex < 0.8
                )
                
                FlowZoneIndicator(
                    title: "Needs Improvement",
                    range: "0.0 - 0.5",
                    color: .red,
                    isActive: viewModel.flowStateIndex < 0.5
                )
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var recommendationsView: some View {
        VStack(alignment: .leading) {
            Text("Recommendations")
                .font(.headline)
                .padding(.bottom)
            
            // Mock recommendations based on current state
            ForEach(mockRecommendations, id: \.self) { recommendation in
                RecommendationRow(recommendation: recommendation)
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(12)
        .shadow(radius: 5)
    }
    
    private var flowStateDescription: String {
        switch viewModel.flowStateIndex {
        case 0.8...1.0:
            return "Excellent flow state - optimal performance"
        case 0.6..<0.8:
            return "Good focus - maintain current state"
        case 0.4..<0.6:
            return "Moderate focus - room for improvement"
        default:
            return "Low flow state - consider relaxation techniques"
        }
    }
    
    private var flowStateColor: Color {
        switch viewModel.flowStateIndex {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .yellow
        default: return .red
        }
    }
    
    private var mockRecommendations: [CognitiveRecommendation] {
        var recommendations: [CognitiveRecommendation] = []
        
        if viewModel.currentHRV < 25 {
            recommendations.append(.breathingExercise)
        }
        
        if viewModel.flowStateIndex < 0.5 {
            recommendations.append(.relaxation)
        } else if viewModel.flowStateIndex > 0.8 {
            recommendations.append(.maintainFocus)
        }
        
        return recommendations.isEmpty ? [.maintainFocus] : recommendations
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @ObservedObject var viewModel: FinViewModel
    @State private var showingExportAlert = false
    @State private var showingCalibrationAlert = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("Sensor Configuration") {
                    HStack {
                        Text("IMU Status")
                        Spacer()
                        Text(viewModel.sensorManager.isIMUActive ? "Active" : "Inactive")
                            .foregroundColor(viewModel.sensorManager.isIMUActive ? .green : .red)
                    }
                    
                    HStack {
                        Text("Pressure Sensor")
                        Spacer()
                        Text(viewModel.sensorManager.isPressureSensorConnected ? "Connected" : "Disconnected")
                            .foregroundColor(viewModel.sensorManager.isPressureSensorConnected ? .green : .red)
                    }
                    
                    Button("Calibrate Sensors") {
                        showingCalibrationAlert = true
                    }
                }
                
                Section("HealthKit Integration") {
                    HStack {
                        Text("HealthKit Authorization")
                        Spacer()
                        Text(viewModel.cognitiveTracker.isHealthKitAuthorized ? "Authorized" : "Not Authorized")
                            .foregroundColor(viewModel.cognitiveTracker.isHealthKitAuthorized ? .green : .red)
                    }
                    
                    Button("Start Mock Data") {
                        viewModel.startMockDataGeneration()
                    }
                }
                
                Section("Data Management") {
                    Button("Export Sensor Data") {
                        showingExportAlert = true
                    }
                    
                    Button("Clear Data Buffer") {
                        viewModel.clearDataBuffer()
                    }
                    .foregroundColor(.red)
                }
                
                Section("Fin Configuration") {
                    HStack {
                        Text("Side Fins")
                        Spacer()
                        Text("15.00 sq.in")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Center Fin")
                        Spacer()
                        Text("14.50 sq.in")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Rake Angle")
                        Spacer()
                        Text("6.5°")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Model Type")
                        Spacer()
                        Text(viewModel.predictor.isModelLoaded ? "Core ML" : "Mathematical")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
        .alert("Calibration", isPresented: $showingCalibrationAlert) {
            Button("Calibrate") {
                viewModel.calibrateSensors()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This will calibrate the IMU sensors using the current position as reference.")
        }
        .alert("Export Complete", isPresented: $showingExportAlert) {
            Button("OK") { }
        } message: {
            Text("Sensor data has been exported successfully.")
        }
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: Double
    let unit: String
    let color: Color
    let icon: String
    
    var body: some View {
        VStack {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
            
            HStack {
                Text("\(value, specifier: "%.2f")")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.white.opacity(0.8))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

struct ActionButton: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack {
                Image(systemName: icon)
                    .font(.title2)
                Text(title)
                    .font(.caption)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.blue.opacity(0.1))
            .foregroundColor(.blue)
            .cornerRadius(10)
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }
}

struct PredictionRow: View {
    let prediction: LiftDragPrediction
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("AoA: \(prediction.angleOfAttack, specifier: "%.1f")°")
                    .font(.caption)
                Text("\(prediction.timestamp, style: .time)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing) {
                Text("L/D: \(prediction.liftToDragRatio, specifier: "%.2f")")
                    .font(.caption)
                    .fontWeight(.semibold)
                Text("\(prediction.confidence * 100, specifier: "%.0f")% conf.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct HRVStatView: View {
    let title: String
    let value: Double
    let unit: String
    
    var body: some View {
        VStack {
            Text("\(value, specifier: "%.1f")")
                .font(.title3)
                .fontWeight(.semibold)
            Text("\(title) \(unit)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }
}

struct FlowZoneIndicator: View {
    let title: String
    let range: String
    let color: Color
    let isActive: Bool
    
    var body: some View {
        HStack {
            Circle()
                .fill(isActive ? color : Color.gray.opacity(0.3))
                .frame(width: 12, height: 12)
            
            VStack(alignment: .leading) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(isActive ? .semibold : .regular)
                Text(range)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if isActive {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(color)
            }
        }
        .padding(.vertical, 4)
    }
}

struct RecommendationRow: View {
    let recommendation: CognitiveRecommendation
    
    var body: some View {
        HStack {
            Image(systemName: recommendation.icon)
                .foregroundColor(.blue)
                .frame(width: 20)
            
            Text(recommendation.description)
                .font(.subheadline)
            
            Spacer()
            
            Button("Apply") {
                // Handle recommendation action
                print("Applying recommendation: \(recommendation.description)")
            }
            .font(.caption)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.blue.opacity(0.1))
            .foregroundColor(.blue)
            .cornerRadius(8)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}