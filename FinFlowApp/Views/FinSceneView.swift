import SwiftUI
import SceneKit

#if os(iOS)
struct FinSceneView: UIViewRepresentable {
    let visualizer: FinVisualizer

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = visualizer.scene
        v.allowsCameraControl = true
        v.autoenablesDefaultLighting = true
        v.backgroundColor = .clear
        return v
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
#elseif os(macOS)
struct FinSceneView: NSViewRepresentable {
    let visualizer: FinVisualizer

    func makeNSView(context: Context) -> SCNView {
        let v = SCNView()
        v.scene = visualizer.scene
        v.allowsCameraControl = true
        v.autoenablesDefaultLighting = true
        v.backgroundColor = .clear
        return v
    }

    func updateNSView(_ nsView: SCNView, context: Context) {}
}
#endif