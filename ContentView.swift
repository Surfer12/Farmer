import SwiftUI
import SceneKit

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()
    @State private var showingSettings = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack {
                    Text("Vector 3/2 Blackstix+ Fin")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("CFD Visualization & Cognitive Integration")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top)
                
                // 3D Visualization
                Fin3DView(visualizer: viewModel.visualizer)
                    .frame(height: 300)
                    .background(Color.black)
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                    )
                
                // Controls Section
                VStack(alignment: .leading, spacing: 15) {
                    Text("Flow Parameters")
                        .font(.headline)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Angle of Attack:")
                            Spacer()
                            Text("\(viewModel.angleOfAttack, specifier: "%.1f")°")
                                .fontWeight(.semibold)
                        }
                        Slider(value: $viewModel.angleOfAttack, in: 0...20, step: 0.5)
                            .accentColor(.blue)
                        
                        HStack {
                            Text("Reynolds Number:")
                            Spacer()
                            Text("\(viewModel.reynoldsNumber, specifier: "%.0e")")
                                .fontWeight(.semibold)
                        }
                        Slider(value: $viewModel.reynoldsNumber, in: 100000...2000000, step: 50000)
                            .accentColor(.green)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                // Performance Metrics
                HStack(spacing: 20) {
                    MetricCard(title: "Lift", value: viewModel.liftDrag?.lift ?? 0, unit: "N", color: .green)
                    MetricCard(title: "Drag", value: viewModel.liftDrag?.drag ?? 0, unit: "N", color: .red)
                    MetricCard(title: "L/D Ratio", value: viewModel.liftToDragRatio, unit: "", color: .blue)
                }
                
                // Sensor Data Section
                VStack(alignment: .leading, spacing: 10) {
                    Text("Real-Time Sensors")
                        .font(.headline)
                    
                    HStack(spacing: 20) {
                        SensorCard(title: "Turn Angle", value: viewModel.turnAngle, unit: "°", icon: "gyroscope")
                        SensorCard(title: "HRV", value: viewModel.hrv ?? 0, unit: "ms", icon: "heart.fill")
                    }
                    
                    Button("Refresh HRV") {
                        viewModel.fetchHRV()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Fin CFD Analyzer")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Settings") {
                        showingSettings = true
                    }
                }
            }
        }
        .onAppear {
            viewModel.startMonitoring()
            viewModel.visualizer.setupFinModel()
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: Float
    let unit: String
    let color: Color
    
    var body: some View {
        VStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text("\(value, specifier: "%.2f")")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(color)
            Text(unit)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
        .shadow(radius: 2)
    }
}

struct SensorCard: View {
    let title: String
    let value: Float
    let unit: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("\(value, specifier: "%.1f") \(unit)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

struct Fin3DView: UIViewRepresentable {
    let visualizer: FinVisualizer
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = visualizer.scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor.black
        
        // Add camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 15)
        scnView.scene?.rootNode.addChildNode(cameraNode)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {}
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("Fin Configuration") {
                    HStack {
                        Text("Side Fin Area")
                        Spacer()
                        Text("15.00 sq.in.")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Center Fin Area")
                        Spacer()
                        Text("14.50 sq.in.")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Rake Angle")
                        Spacer()
                        Text("6.5°")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Rider Profile") {
                    HStack {
                        Text("Weight Range")
                        Spacer()
                        Text("125-175 lbs")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Sensors") {
                    Toggle("IMU Monitoring", isOn: .constant(true))
                    Toggle("Pressure Sensors", isOn: .constant(true))
                    Toggle("Heart Rate Variability", isOn: .constant(true))
                }
            }
            .navigationTitle("Settings")
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
}

#Preview {
    ContentView()
}