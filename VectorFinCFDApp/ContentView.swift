import SwiftUI
import SceneKit

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()
    @State private var showingPerformanceDetails = false
    @State private var showingCognitiveDetails = false
    @State private var showingDataExport = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header Section
                    headerSection
                    
                    // 3D Visualization Section
                    visualizationSection
                    
                    // Performance Metrics Section
                    performanceSection
                    
                    // Cognitive Integration Section
                    cognitiveSection
                    
                    // Control Panel Section
                    controlSection
                    
                    // Real-time Data Section
                    realTimeDataSection
                    
                    Spacer(minLength: 20)
                }
                .padding(.horizontal)
            }
            .navigationBarHidden(true)
            .background(Color(.systemGroupedBackground))
        }
        .onAppear {
            viewModel.startMonitoring()
            viewModel.visualizer.setupFinModel()
        }
        .sheet(isPresented: $showingPerformanceDetails) {
            PerformanceDetailView(viewModel: viewModel)
        }
        .sheet(isPresented: $showingCognitiveDetails) {
            CognitiveDetailView(viewModel: viewModel)
        }
        .sheet(isPresented: $showingDataExport) {
            DataExportView(viewModel: viewModel)
        }
    }
    
    // MARK: - Header Section
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Text("Vector 3/2 Fin CFD Visualizer")
                .font(.largeTitle)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
                .foregroundColor(.primary)
            
            Text("Real-time 3D visualization with cognitive integration")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            // Flow State Indicator
            HStack {
                Circle()
                    .fill(viewModel.flowState.color)
                    .frame(width: 12, height: 12)
                
                Text(viewModel.flowState.rawValue)
                    .font(.headline)
                    .foregroundColor(viewModel.flowState.color)
                
                Spacer()
                
                Text("Confidence: \(String(format: "%.0f", viewModel.predictionConfidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
        .padding(.top)
    }
    
    // MARK: - Visualization Section
    
    private var visualizationSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("3D Fin Model & Pressure Maps")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                
                Button(action: {
                    viewModel.visualizer.setupFinModel()
                }) {
                    Image(systemName: "arrow.clockwise")
                        .foregroundColor(.blue)
                }
            }
            
            Fin3DView(visualizer: viewModel.visualizer)
                .frame(height: 350)
                .background(Color(.systemBackground))
                .cornerRadius(15)
                .shadow(radius: 5)
                .overlay(
                    RoundedRectangle(cornerRadius: 15)
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                )
        }
    }
    
    // MARK: - Performance Section
    
    private var performanceSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Performance Metrics")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                
                Button("Details") {
                    showingPerformanceDetails = true
                }
                .font(.subheadline)
                .foregroundColor(.blue)
            }
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
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
                    color: .red,
                    icon: "arrow.down"
                )
                
                MetricCard(
                    title: "L/D Ratio",
                    value: viewModel.liftDrag.map { $0.lift / max($0.drag, 0.1) } ?? 0,
                    unit: "",
                    color: .blue,
                    icon: "chart.line.uptrend.xyaxis"
                )
            }
        }
    }
    
    // MARK: - Cognitive Section
    
    private var cognitiveSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Cognitive Integration")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                
                Button("Details") {
                    showingCognitiveDetails = true
                }
                .font(.subheadline)
                .foregroundColor(.blue)
            }
            
            HStack(spacing: 16) {
                CognitiveMetricCard(
                    title: "HRV",
                    value: viewModel.hrv ?? 0,
                    unit: "ms",
                    color: .purple,
                    icon: "heart.fill"
                )
                
                CognitiveMetricCard(
                    title: "Cognitive Load",
                    value: viewModel.cognitiveLoad,
                    unit: "",
                    color: .orange,
                    icon: "brain.head.profile"
                )
            }
            
            Button(action: {
                viewModel.fetchHRV()
            }) {
                HStack {
                    Image(systemName: "heart.fill")
                        .foregroundColor(.red)
                    Text("Update HRV Data")
                        .fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
        }
    }
    
    // MARK: - Control Section
    
    private var controlSection: some View {
        VStack(spacing: 16) {
            Text("Control Panel")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                HStack {
                    Text("Angle of Attack:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Spacer()
                    Text("\(Int(viewModel.turnAngle))°")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                
                Slider(value: $viewModel.turnAngle, in: 0...20, step: 1)
                    .accentColor(.blue)
                
                HStack {
                    Text("0°")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("20°")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal)
        }
    }
    
    // MARK: - Real-time Data Section
    
    private var realTimeDataSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Real-time Data")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                
                Button("Export") {
                    showingDataExport = true
                }
                .font(.subheadline)
                .foregroundColor(.blue)
            }
            
            VStack(spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("IMU Turn Angle")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(Int(viewModel.turnAngle))°")
                            .font(.title3)
                            .fontWeight(.semibold)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Last Update")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(viewModel.lastUpdateTime, style: .time)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Monitoring Status")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        HStack {
                            Circle()
                                .fill(viewModel.isMonitoring ? Color.green : Color.red)
                                .frame(width: 8, height: 8)
                            Text(viewModel.isMonitoring ? "Active" : "Inactive")
                                .font(.caption)
                                .foregroundColor(viewModel.isMonitoring ? .green : .red)
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Pressure Points")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(viewModel.pressureData.count)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(.horizontal)
        }
    }
}

// MARK: - Supporting Views

struct MetricCard: View {
    let title: String
    let value: Float
    let unit: String
    let color: Color
    let icon: String
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("\(value, specifier: "%.1f")")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(unit)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}

struct CognitiveMetricCard: View {
    let title: String
    let value: Double
    let unit: String
    let color: Color
    let icon: String
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("\(value, specifier: "%.1f")")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(unit)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}

struct Fin3DView: UIViewRepresentable {
    let visualizer: FinVisualizer
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = visualizer.scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor.systemBackground
        scnView.antialiasingMode = .multisampling4X
        
        // Add camera controls
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 15)
        visualizer.scene.rootNode.addChildNode(cameraNode)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Updates handled by the visualizer
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}