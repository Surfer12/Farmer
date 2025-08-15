import Foundation
import SceneKit
#if canImport(UIKit)
import UIKit
#endif

final class FinVisualizer {
	let scene = SCNScene()
	private var sideFinNode: SCNNode?
	private var centerFinNode: SCNNode?

	func setupFinModel() {
		let root = scene.rootNode

		// Camera
		let cameraNode = SCNNode()
		cameraNode.camera = SCNCamera()
		cameraNode.position = SCNVector3(0, 0, 12)
		root.addChildNode(cameraNode)

		// Ground
		let floor = SCNFloor()
		let floorNode = SCNNode(geometry: floor)
		root.addChildNode(floorNode)

		// Materials
		let pressureMaterial = SCNMaterial()
		#if canImport(UIKit)
		pressureMaterial.diffuse.contents = UIColor.systemBlue
		#else
		pressureMaterial.diffuse.contents = NSColor.systemBlue
		#endif

		// Side fin (Vector 3/2 foil approximation, 15.00 sq.in.)
		let sideFinGeometry = SCNBox(width: 4.63, height: 4.48, length: 0.12, chamferRadius: 0.05)
		sideFinGeometry.materials = [pressureMaterial]
		sideFinNode = SCNNode(geometry: sideFinGeometry)
		sideFinNode?.eulerAngles.z = Float(6.5 * Double.pi / 180.0)
		sideFinNode?.position = SCNVector3(1.8, 0, 0)

		// Center fin (symmetric, 14.50 sq.in.)
		let centerFinGeometry = SCNBox(width: 4.5, height: 4.4, length: 0.12, chamferRadius: 0.05)
		centerFinGeometry.materials = [pressureMaterial]
		centerFinNode = SCNNode(geometry: centerFinGeometry)
		centerFinNode?.position = SCNVector3(-1.8, 0, 0)

		if let side = sideFinNode { root.addChildNode(side) }
		if let center = centerFinNode { root.addChildNode(center) }
	}

	func updatePressureMap(pressureData: [Float]) {
		guard let sideMaterial = sideFinNode?.geometry?.materials.first else { return }
		let avg = max(0.0, min(1.0, pressureData.isEmpty ? 0.0 : pressureData.reduce(0, +) / Float(pressureData.count)))
		let color = HeatmapColorMapper.color(for: Double(avg))
		sideMaterial.diffuse.contents = color
		if let centerMaterial = centerFinNode?.geometry?.materials.first {
			centerMaterial.diffuse.contents = color
		}
	}

	func updateAngleOfAttack(aoaDegrees: Float) {
		let radians = aoaDegrees * Float(Double.pi) / 180.0
		sideFinNode?.eulerAngles.y = radians
		centerFinNode?.eulerAngles.y = radians
	}
}