import Foundation
import SceneKit
import SwiftUI

final class FinVisualizer {
    let scene: SCNScene = SCNScene()

    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var flowSystem: SCNParticleSystem?

    private let finSpec = FinSpec(
        sideAreaSqIn: 15.0,
        centerAreaSqIn: 14.5,
        cantDegrees: 6.5,
        rakeDegrees: 6.5,
        foil: "Vector 3/2"
    )

    func setupFinModel() {
        scene.rootNode.childNodes.forEach { $0.removeFromParentNode() }
        sideFinNodes.removeAll()
        centerFinNode = nil

        // Ground plane for reference
        let plane = SCNPlane(width: 20, height: 20)
        plane.firstMaterial?.diffuse.contents = NSColorOrUIColor.systemGray.withAlphaComponent(0.15)
        let planeNode = SCNNode(geometry: plane)
        planeNode.eulerAngles.x = -.pi / 2
        planeNode.position = SCNVector3(0, -2.0, 0)
        scene.rootNode.addChildNode(planeNode)

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 12)
        cameraNode.camera?.zFar = 1000
        scene.rootNode.addChildNode(cameraNode)

        // Lights
        let light = SCNLight()
        light.type = .omni
        light.intensity = 1000
        let lightNode = SCNNode()
        lightNode.light = light
        lightNode.position = SCNVector3(5, 10, 15)
        scene.rootNode.addChildNode(lightNode)

        // Side fins (proxy geometry). Dimensions chosen to approximate area.
        for i in 0..<2 {
            let width: CGFloat = 3.9
            let height: CGFloat = 4.9
            let thickness: CGFloat = 0.12
            let sideFinGeometry = SCNBox(width: width, height: height, length: thickness, chamferRadius: 0.06)
            sideFinGeometry.firstMaterial?.diffuse.contents = ColorMap.color(forNormalized: 0.2)
            let sideFinNode = SCNNode(geometry: sideFinGeometry)
            sideFinNode.pivot = SCNMatrix4MakeTranslation(0, Float(-height/2.0), 0)
            sideFinNode.position = SCNVector3(x: Float(i == 0 ? -2.0 : 2.0), y: 0.0, z: 0.0)
            sideFinNode.eulerAngles.z = Float(finSpec.cantDegrees * .pi / 180.0)
            sideFinNode.name = i == 0 ? "side_left" : "side_right"
            sideFinNodes.append(sideFinNode)
            scene.rootNode.addChildNode(sideFinNode)
        }

        // Center fin (symmetric)
        let centerGeometry = SCNBox(width: 3.7, height: 4.7, length: 0.12, chamferRadius: 0.06)
        centerGeometry.firstMaterial?.diffuse.contents = ColorMap.color(forNormalized: 0.2)
        let centerNode = SCNNode(geometry: centerGeometry)
        centerNode.pivot = SCNMatrix4MakeTranslation(0, -2.35, 0)
        centerNode.position = SCNVector3(0, 0, -0.4)
        centerNode.name = "center"
        scene.rootNode.addChildNode(centerNode)
        centerFinNode = centerNode

        // Default flow
        setFlowMode(.laminar)
    }

    func setFlowMode(_ mode: FlowMode) {
        flowSystem.flatMap { scene.rootNode.removeParticleSystem($0) }
        let system = SCNParticleSystem()
        system.particleColor = NSColorOrUIColor.white
        system.particleSize = 0.04
        system.particleVelocity = mode == .laminar ? 1.2 : 2.5
        system.particleVelocityVariation = mode == .laminar ? 0.2 : 1.0
        system.birthRate = mode == .laminar ? 80 : 180
        system.emissionDuration = 0
        system.loops = true
        system.emitterShape = SCNSphere(radius: 0.2)
        system.acceleration = SCNVector3(0, 0, -0.2)
        scene.rootNode.addParticleSystem(system)
        flowSystem = system

        // Subtle oscillation for turbulent case
        if mode == .turbulent {
            sideFinNodes.forEach { addWobble(to: $0, amplitude: 0.02) }
            centerFinNode.flatMap { addWobble(to: $0, amplitude: 0.015) }
        }
    }

    func updatePressureMap(values: [Float]) {
        guard !values.isEmpty else { return }
        let average = values.reduce(0, +) / Float(values.count)
        let mappedColor = ColorMap.color(forNormalized: average)
        sideFinNodes.forEach { $0.geometry?.firstMaterial?.diffuse.contents = mappedColor }
        centerFinNode?.geometry?.firstMaterial?.diffuse.contents = mappedColor
    }

    func tiltFins(forAoA degrees: Float) {
        let clamped = max(0, min(20, degrees))
        let radians = clamped * .pi / 180
        let target = SCNVector3(0, radians / 4.0, 0)
        let action = SCNAction.rotateTo(x: CGFloat(target.x), y: CGFloat(target.y), z: CGFloat(target.z), duration: 0.15)
        sideFinNodes.forEach { $0.runAction(action) }
        centerFinNode?.runAction(action)
    }

    private func addWobble(to node: SCNNode, amplitude: CGFloat) {
        let up = SCNAction.moveBy(x: 0, y: amplitude, z: 0, duration: 0.4)
        let down = SCNAction.moveBy(x: 0, y: -amplitude, z: 0, duration: 0.4)
        let seq = SCNAction.sequence([up, down])
        node.runAction(.repeatForever(seq))
    }
}