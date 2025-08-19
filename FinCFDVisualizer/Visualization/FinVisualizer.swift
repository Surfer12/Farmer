// SPDX-License-Identifier: GPL-3.0-only
import Foundation
import SceneKit

final class FinVisualizer {
    let scene: SCNScene = SCNScene()

    private(set) var sideFinNode: SCNNode?
    private(set) var centerFinNode: SCNNode?

    private let cameraNode = SCNNode()
    private let lightNode = SCNNode()
    private var flowNode: SCNNode?

    init() {
        setupScene()
        setupCameraAndLight()
        setupFinModel()
        setupFlow()
    }

    func setupScene() {
        scene.background.contents = PlatformColor.black
    }

    func setupCameraAndLight() {
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 12)
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        scene.rootNode.addChildNode(cameraNode)

        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(5, 6, 10)
        scene.rootNode.addChildNode(lightNode)

        let ambientNode = SCNNode()
        ambientNode.light = SCNLight()
        ambientNode.light?.type = .ambient
        ambientNode.light?.color = PlatformColor(white: 0.2, alpha: 1.0)
        scene.rootNode.addChildNode(ambientNode)
    }

    func setupFinModel() {
        let sideFinGeometry = SCNBox(width: 4.6, height: 4.5, length: 0.08, chamferRadius: 0.02)
        let centerFinGeometry = SCNBox(width: 4.5, height: 4.5, length: 0.10, chamferRadius: 0.02)

        let initialMaterial = SCNMaterial()
        initialMaterial.diffuse.contents = PlatformColor.systemBlue
        initialMaterial.locksAmbientWithDiffuse = true
        sideFinGeometry.materials = [initialMaterial]

        let centerMaterial = SCNMaterial()
        centerMaterial.diffuse.contents = PlatformColor.systemTeal
        centerMaterial.locksAmbientWithDiffuse = true
        centerFinGeometry.materials = [centerMaterial]

        let sideFin = SCNNode(geometry: sideFinGeometry)
        sideFin.pivot = SCNMatrix4MakeTranslation(-Float(sideFinGeometry.width / 2.0), 0, 0)
        sideFin.position = SCNVector3(x: 1.8, y: 0, z: 0)
        sideFin.eulerAngles.z = degreesToRadians(6.5)

        let centerFin = SCNNode(geometry: centerFinGeometry)
        centerFin.pivot = SCNMatrix4MakeTranslation(-Float(centerFinGeometry.width / 2.0), 0, 0)
        centerFin.position = SCNVector3(x: -1.8, y: 0, z: 0)

        sideFinNode = sideFin
        centerFinNode = centerFin

        scene.rootNode.addChildNode(sideFin)
        scene.rootNode.addChildNode(centerFin)
    }

    private func setupFlow() {
        let node = SCNNode()
        let particles = SCNParticleSystem()
        particles.birthRate = 120
        particles.particleLifeSpan = 2.0
        particles.particleSize = 0.035
        particles.emissionDuration = .infinity
        particles.emitterShape = SCNSphere(radius: 0.2)
        particles.particleColor = PlatformColor.white
        node.addParticleSystem(particles)
        node.position = SCNVector3(-2.0, 0, 0)
        scene.rootNode.addChildNode(node)
        flowNode = node
    }

    func updateAoA(degrees: Float) {
        sideFinNode?.eulerAngles.y = degreesToRadians(degrees)
        // emit a synthetic pressure map tied to AoA for demo
        let width = 64
        let height = 64
        let normalized = syntheticPressureMap(width: width, height: height, aoa: Double(degrees))
        updatePressureMapImage(width: width, height: height, normalized: normalized)
        updateFlow(for: Double(degrees))
    }

    func updatePressureMap(pressureDataNormalized: [Float]) {
        // Fallback single-color update based on average if an image cannot be created
        guard let sideFinGeometry = sideFinNode?.geometry else { return }
        let avg = pressureDataNormalized.isEmpty ? 0.0 : (pressureDataNormalized.reduce(0, +) / Float(pressureDataNormalized.count))
        let clamped = max(0.0, min(1.0, avg))
        let color = colorFor(value: clamped)
        sideFinGeometry.firstMaterial?.diffuse.contents = color
    }

    private func updatePressureMapImage(width: Int, height: Int, normalized: [Float]) {
        guard let image = PressureColorMap.cgImage(width: width, height: height, normalized: normalized) else {
            updatePressureMap(pressureDataNormalized: normalized)
            return
        }
        sideFinNode?.geometry?.firstMaterial?.diffuse.contents = image
    }

    private func updateFlow(for aoa: Double) {
        guard let systems = flowNode?.particleSystems, let ps = systems.first else { return }
        let t = min(1.0, max(0.0, aoa / 20.0))
        ps.birthRate = CGFloat(80 + 160 * t)
        ps.particleVelocity = CGFloat(1.0 + 2.0 * t)
        #if os(iOS)
        ps.particleColor = PlatformColor(hue: CGFloat(0.6 - 0.6 * t), saturation: 0.8, brightness: 1.0, alpha: 1.0)
        #else
        ps.particleColor = PlatformColor(calibratedHue: CGFloat(0.6 - 0.6 * t), saturation: 0.8, brightness: 1.0, alpha: 1.0)
        #endif
    }

    private func syntheticPressureMap(width: Int, height: Int, aoa: Double) -> [Float] {
        let a = Float(min(1.0, max(0.0, aoa / 20.0)))
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let u = Float(x) / Float(width - 1)
                let v = Float(y) / Float(height - 1)
                let ridge = max(0, 1.0 - abs(u - a) * 3.0)
                let tipBoost = pow(v, 2.0)
                let value = min(1.0, ridge * (0.6 + 0.4 * tipBoost))
                out[y * width + x] = value
            }
        }
        return out
    }

    private func colorFor(value: Float) -> PlatformColor {
        let v = CGFloat(max(0.0, min(1.0, value)))
        if v < 0.25 {
            let t = v / 0.25
            return PlatformColor(red: 0, green: t, blue: 1.0, alpha: 1.0)
        } else if v < 0.5 {
            let t = (v - 0.25) / 0.25
            return PlatformColor(red: 0, green: 1.0, blue: 1.0 - t, alpha: 1.0)
        } else if v < 0.75 {
            let t = (v - 0.5) / 0.25
            return PlatformColor(red: t, green: 1.0, blue: 0, alpha: 1.0)
        } else {
            let t = (v - 0.75) / 0.25
            return PlatformColor(red: 1.0, green: 1.0 - t, blue: 0, alpha: 1.0)
        }
    }

    private func degreesToRadians(_ degrees: Float) -> Float {
        return degrees * .pi / 180.0
    }
}