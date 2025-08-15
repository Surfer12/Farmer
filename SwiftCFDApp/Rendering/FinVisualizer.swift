import Foundation
import SceneKit
import SwiftUI

final class FinVisualizer {
    let scene = SCNScene()
    private let cameraNode = SCNNode()
    private let lightNode = SCNNode()
    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var particleNode: SCNNode?

    init() {
        setupCameraAndLight()
        buildFins()
        setFlowMode(.laminar)
    }

    private func setupCameraAndLight() {
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 16)
        scene.rootNode.addChildNode(cameraNode)

        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(0, 10, 10)
        scene.rootNode.addChildNode(lightNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.color = PlatformColor.rgb(0.35, 0.35, 0.35, 1)
        scene.rootNode.addChildNode(ambient)

        let ground = SCNFloor()
        ground.reflectivity = 0.05
        let groundNode = SCNNode(geometry: ground)
        #if os(macOS)
        groundNode.geometry?.firstMaterial?.diffuse.contents = NSColor.systemTeal.withAlphaComponent(0.15)
        #else
        groundNode.geometry?.firstMaterial?.diffuse.contents = UIColor.systemTeal.withAlphaComponent(0.15)
        #endif
        scene.rootNode.addChildNode(groundNode)
    }

    private func finShapePath() -> AnyObject {
        #if os(macOS)
        let path = NSBezierPath()
        path.move(to: NSPoint(x: 0, y: 0))
        path.curve(to: NSPoint(x: 0.2, y: 4.4), controlPoint1: NSPoint(x: 0.9, y: 2.2), controlPoint2: NSPoint(x: 0.9, y: 2.2))
        path.curve(to: NSPoint(x: 4.6, y: 0.1), controlPoint1: NSPoint(x: 3.9, y: 2.9), controlPoint2: NSPoint(x: 3.9, y: 2.9))
        path.close()
        return path
        #else
        let path = UIBezierPath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addQuadCurve(to: CGPoint(x: 0.2, y: 4.4), controlPoint: CGPoint(x: 0.9, y: 2.2))
        path.addQuadCurve(to: CGPoint(x: 4.6, y: 0.1), controlPoint: CGPoint(x: 3.9, y: 2.9))
        path.close()
        return path
        #endif
    }

    private func buildFins() {
        // Side fins (Vector 3/2 foil approximation)
        for i in 0..<2 {
            #if os(macOS)
            let geom = SCNShape(path: finShapePath() as! NSBezierPath, extrusionDepth: 0.12)
            #else
            let geom = SCNShape(path: finShapePath() as! UIBezierPath, extrusionDepth: 0.12)
            #endif
            geom.firstMaterial = baseMaterial()
            let node = SCNNode(geometry: geom)
            node.eulerAngles.z = (i == 0 ? -1 : 1) * Float(Defaults.rakeDegrees * .pi / 180)
            node.position = SCNVector3(x: (i == 0 ? -1.8 : 1.8), y: 2.2, z: 0)
            sideFinNodes.append(node)
            scene.rootNode.addChildNode(node)
        }
        // Center fin (symmetric)
        #if os(macOS)
        let centerGeom = SCNShape(path: finShapePath() as! NSBezierPath, extrusionDepth: 0.12)
        #else
        let centerGeom = SCNShape(path: finShapePath() as! UIBezierPath, extrusionDepth: 0.12)
        #endif
        centerGeom.firstMaterial = baseMaterial()
        let center = SCNNode(geometry: centerGeom)
        center.position = SCNVector3(x: 0, y: 1.8, z: -0.6)
        scene.rootNode.addChildNode(center)
        centerFinNode = center
    }

    private func baseMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.metalness.contents = 0.15
        m.roughness.contents = 0.65
        #if os(macOS)
        m.diffuse.contents = NSColor.systemBlue
        #else
        m.diffuse.contents = UIColor.systemBlue
        #endif
        return m
    }

    func setFlowMode(_ mode: FlowMode) {
        particleNode?.removeFromParentNode()
        let particles = SCNParticleSystem()
        particles.birthRate = (mode == .laminar) ? 160 : 420
        particles.particleSize = (mode == .laminar) ? 0.025 : 0.035
        #if os(macOS)
        particles.particleColor = (mode == .laminar) ? NSColor.white : NSColor.systemYellow
        #else
        particles.particleColor = (mode == .laminar) ? UIColor.white : UIColor.systemYellow
        #endif
        particles.emitterShape = SCNSphere(radius: 0.2)
        particles.particleVelocity = (mode == .laminar) ? 2.0 : 4.0
        particles.particleVelocityVariation = (mode == .laminar) ? 0.2 : 2.0
        particles.acceleration = (mode == .laminar) ? SCNVector3(1.4, 0, 0) : SCNVector3(2.2, 0.4, 0.3)
        let node = SCNNode()
        node.position = SCNVector3(-4.5, 2.2, 0)
        node.addParticleSystem(particles)
        scene.rootNode.addChildNode(node)
        particleNode = node
    }

    func updatePressureTexture(_ pressures: [Float]) {
        let image = ColorMap.image(from: pressures, width: 64, height: 64)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        for node in sideFinNodes + [centerFinNode].compactMap({ $0 }) {
            node.geometry?.firstMaterial?.diffuse.contents = image
        }
    }
}