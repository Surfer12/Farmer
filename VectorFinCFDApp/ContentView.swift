// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import SwiftUI
import SceneKit

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()
    @State private var showingExportSheet = false
    @State private var exportedData = ""
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header with status indicators
                VStack(spacing: 8) {
                    Text("Vector 3/2 Fin CFD Visualizer")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                    
                    HStack(spacing: 16) {
                        StatusIndicator(
                            title: "Monitoring",
                            isActive: viewModel.isMonitoring,
                            color: .green
                        )
                        
                        StatusIndicator(
                            title: "Flow State",
                            value: viewModel.flowState.rawValue,
                            color: Color(viewModel.flowState.color)
                        )
                        
                        StatusIndicator(
                            title: "Confidence",
                            value: "\(Int(viewModel.predictionConfidence * 100))%",
                            color: viewModel.predictionConfidence > 0.8 ? .green : .orange
                        )
                    }
                }
                .padding(.top)
                
                // 3D Visualization
                VStack(spacing: 12) {
                    Text("3D Fin Model & Pressure Maps")
                        .font(.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                    
                    Fin3DView(visualizer: viewModel.visualizer)
                        .frame(height: 350)
                        .background(Color.black.opacity(0.1))
                        .cornerRadius(15)
                        .shadow(radius: 5)
                        .padding(.horizontal)
                        .overlay(
                            // Flow state overlay
                            VStack {
                                HStack {
                                    Spacer()
                                    FlowStateIndicator(flowState: viewModel.flowState)
                                        .padding()
                                }
                                Spacer()
                            }
                        )
                }
                
                // Control Panel
                VStack(spacing: 16) {
                    Text("Control Panel")
                        .font(.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                    
                    // Angle of Attack Slider
                    VStack(spacing: 8) {
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
                            .disabled(!viewModel.isMonitoring)
                    }
                    .padding(.horizontal)
                    
                    // Performance Metrics Grid
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 16) {
                        MetricCard(
                            title: "Lift",
                            value: "\(viewModel.liftDrag?.lift ?? 0, specifier: "%.1f") N",
                            color: .green
                        )
                        
                        MetricCard(
                            title: "Drag",
                            value: "\(viewModel.liftDrag?.drag ?? 0, specifier: "%.1f") N",
                            color: .red
                        )
                        
                        MetricCard(
                            title: "L/D Ratio",
                            value: "\(viewModel.performanceMetrics.liftToDragRatio, specifier: "%.1f")",
                            color: .blue
                        )
                    }
                    .padding(.horizontal)
                }
                
                // Cognitive Metrics
                VStack(spacing: 16) {
                    Text("Cognitive Integration")
                        .font(.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                    
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 12) {
                        MetricCard(
                            title: "HRV",
                            value: "\(viewModel.hrv ?? 0, specifier: "%.1f") ms",
                            color: .purple
                        )
                        
                        MetricCard(
                            title: "Cognitive Load",
                            value: "\(Int(viewModel.cognitiveTracker.cognitiveLoad * 100))%",
                            color: viewModel.cognitiveTracker.cognitiveLoad > 0.7 ? .red : .green
                        )
                        
                        MetricCard(
                            title: "Attention",
                            value: "\(Int(viewModel.cognitiveTracker.attentionLevel * 100))%",
                            color: .cyan
                        )
                        
                        MetricCard(
                            title: "Stress Level",
                            value: "\(Int(viewModel.cognitiveTracker.stressLevel * 100))%",
                            color: viewModel.cognitiveTracker.stressLevel > 0.7 ? .red : .orange
                        )
                    }
                    .padding(.horizontal)
                    
                    // Action buttons
                    HStack(spacing: 12) {
                        Button(action: {
                            viewModel.fetchHRV()
                        }) {
                            HStack {
                                Image(systemName: "heart.fill")
                                Text("Update HRV")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                        
                        Button(action: {
                            if viewModel.isMonitoring {
                                viewModel.stopMonitoring()
                            } else {
                                viewModel.startMonitoring()
                                viewModel.visualizer.setupFinModel()
                            }
                        }) {
                            HStack {
                                Image(systemName: viewModel.isMonitoring ? "stop.fill" : "play.fill")
                                Text(viewModel.isMonitoring ? "Stop" : "Start")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isMonitoring ? Color.red : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                        
                        Button(action: {
                            exportedData = viewModel.exportData()
                            showingExportSheet = true
                        }) {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("Export")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        }
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .navigationBarHidden(true)
            .alert("Error", isPresented: $viewModel.isShowingError) {
                Button("OK") {
                    viewModel.isShowingError = false
                }
            } message: {
                Text(viewModel.errorMessage ?? "Unknown error occurred")
            }
            .sheet(isPresented: $showingExportSheet) {
                ExportDataView(data: exportedData)
            }
        }
        .onAppear {
            viewModel.startMonitoring()
            viewModel.visualizer.setupFinModel()
        }
        .onDisappear {
            viewModel.stopMonitoring()
        }
    }
}

// MARK: - Supporting Views

struct Fin3DView: UIViewRepresentable {
    let visualizer: FinVisualizer
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = visualizer.scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor.systemBackground
        scnView.antialiasingMode = .multisampling4X
        
        // Add camera with better positioning
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 15)
        cameraNode.camera?.fieldOfView = 45
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        visualizer.scene.rootNode.addChildNode(cameraNode)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Updates handled by the visualizer
    }
}

struct StatusIndicator: View {
    let title: String
    var isActive: Bool = false
    var value: String = ""
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            if isActive {
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
            } else if !value.isEmpty {
                Text(value)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(color)
            }
            
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

struct FlowStateIndicator: View {
    let flowState: FlowState
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Color(flowState.color))
                .frame(width: 12, height: 12)
            
            Text(flowState.rawValue)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(Color(flowState.color))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.1))
        .cornerRadius(8)
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct ExportDataView: View {
    let data: String
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                ScrollView {
                    Text(data)
                        .font(.system(.body, design: .monospaced))
                        .padding()
                }
                
                Button("Share") {
                    let av = UIActivityViewController(activityItems: [data], applicationActivities: nil)
                    if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let rootViewController = windowScene.windows.first?.rootViewController {
                        rootViewController.present(av, animated: true)
                    }
                }
                .buttonStyle(.borderedProminent)
                .padding()
            }
            .navigationTitle("Export Data")
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

// MARK: - Color Extensions

extension Color {
    init(_ colorName: String) {
        switch colorName.lowercased() {
        case "green": self = .green
        case "blue": self = .blue
        case "orange": self = .orange
        case "red": self = .red
        case "gray": self = .gray
        case "purple": self = .purple
        case "cyan": self = .cyan
        default: self = .primary
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}