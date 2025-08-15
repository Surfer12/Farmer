import SceneKit
import SwiftUI

#if os(iOS)
struct FinSceneView: UIViewRepresentable {
    let visualizer: FinVisualizer

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = visualizer.scene
        view.autoenablesDefaultLighting = true
        view.allowsCameraControl = true
        view.backgroundColor = PlatformColor.black
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
#elseif os(macOS)
struct FinSceneView: NSViewRepresentable {
    let visualizer: FinVisualizer

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = visualizer.scene
        view.autoenablesDefaultLighting = true
        view.allowsCameraControl = true
        view.backgroundColor = PlatformColor.black
        return view
    }

    func updateNSView(_ nsView: SCNView, context: Context) {}
}
#endif
