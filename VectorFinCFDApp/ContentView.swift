import SwiftUI
import SceneKit

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Vector 3/2 Fin CFD Visualizer")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                    
                    Text("Real-time 3D visualization with cognitive integration")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
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
                    }
                    .padding(.horizontal)
                    
                    // Performance Metrics
                    HStack(spacing: 20) {
                        VStack(spacing: 4) {
                            Text("Lift")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(viewModel.liftDrag?.lift ?? 0, specifier: "%.2f")")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.green)
                        }
                        .frame(maxWidth: .infinity)
                        
                        VStack(spacing: 4) {
                            Text("Drag")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(viewModel.liftDrag?.drag ?? 0, specifier: "%.2f")")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.red)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .padding(.horizontal)
                }
                
                // Real-time Data
                VStack(spacing: 16) {
                    Text("Real-time Data")
                        .font(.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                    
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
                                Text("HRV (Flow State)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("\(viewModel.hrv ?? 0, specifier: "%.1f") ms")
                                    .font(.title3)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.purple)
                            }
                        }
                        
                        Button(action: {
                            viewModel.fetchHRV()
                        }) {
                            HStack {
                                Image(systemName: "heart.fill")
                                    .foregroundColor(.red)
                                Text("Fetch HRV Data")
                                    .fontWeight(.medium)
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
        }
        .onAppear {
            viewModel.startMonitoring()
            viewModel.visualizer.setupFinModel()
        }
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