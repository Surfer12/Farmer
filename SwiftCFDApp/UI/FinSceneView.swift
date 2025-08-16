import SwiftUI
import SceneKit

#if os(iOS)
struct FinSceneView: UIViewRepresentable {
    let visualizer: FinVisualizer

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = visualizer.scene
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
#else
struct FinSceneView: NSViewRepresentable {
    let visualizer: FinVisualizer

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = visualizer.scene
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        return view
    }

    func updateNSView(_ nsView: SCNView, context: Context) {}
}
#endif