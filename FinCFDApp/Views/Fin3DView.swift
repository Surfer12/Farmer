import SwiftUI
import SceneKit

struct Fin3DView: View {
	let visualizer: FinVisualizer

	var body: some View {
		#if os(iOS)
		SceneKitView_iOS(scene: visualizer.scene)
		#else
		SceneKitView_macOS(scene: visualizer.scene)
		#endif
	}
}

#if os(iOS)
struct SceneKitView_iOS: UIViewRepresentable {
	let scene: SCNScene

	func makeUIView(context: Context) -> SCNView {
		let scnView = SCNView()
		scnView.scene = scene
		scnView.allowsCameraControl = true
		scnView.autoenablesDefaultLighting = true
		return scnView
	}

	func updateUIView(_ uiView: SCNView, context: Context) {}
}
#else
struct SceneKitView_macOS: NSViewRepresentable {
	let scene: SCNScene

	func makeNSView(context: Context) -> SCNView {
		let scnView = SCNView()
		scnView.scene = scene
		scnView.allowsCameraControl = true
		scnView.autoenablesDefaultLighting = true
		return scnView
	}

	func updateNSView(_ nsView: SCNView, context: Context) {}
}
#endif