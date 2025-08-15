import SceneKit
import UIKit

class FinVisualizer: ObservableObject {
    let scene = SCNScene()
    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var flowParticleSystem: SCNParticleSystem?
    
    // Fin specifications based on Vector 3/2 Blackstix+
    private let sideFinArea: Float = 15.00 // sq.in.
    private let centerFinArea: Float = 14.50 // sq.in.
    private let sideFinAngle: Float = 6.5 // degrees
    private let finThickness: Float = 0.1
    private let finChamferRadius: Float = 0.05
    
    init() {
        setupScene()
    }
    
    private func setupScene() {
        // Add ambient lighting
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 300
        scene.rootNode.addChildNode(ambientLight)
        
        // Add directional lighting
        let directionalLight = SCNNode()
        directionalLight.light = SCNLight()
        directionalLight.light?.type = .directional
        directionalLight.light?.intensity = 800
        directionalLight.position = SCNVector3(x: 10, y: 10, z: 10)
        directionalLight.eulerAngles = SCNVector3(x: -Float.pi/4, y: Float.pi/4, z: 0)
        scene.rootNode.addChildNode(directionalLight)
        
        // Add flow particle system
        setupFlowParticles()
    }
    
    func setupFinModel() {
        // Clear existing fins
        sideFinNodes.forEach { $0.removeFromParentNode() }
        centerFinNode?.removeFromParentNode()
        sideFinNodes.removeAll()
        
        // Calculate fin dimensions based on area
        let sideFinWidth = sqrt(sideFinArea * 0.0064516) // Convert sq.in. to sq.m
        let sideFinHeight = sideFinWidth * 0.97 // Aspect ratio for Vector 3/2 foil
        let centerFinWidth = sqrt(centerFinArea * 0.0064516)
        let centerFinHeight = centerFinWidth * 0.97
        
        // Create side fins (15.00 sq.in., 6.5° angle, Vector 3/2 foil)
        for i in 0..<2 {
            let sideFinGeometry = SCNBox(
                width: CGFloat(sideFinWidth),
                height: CGFloat(sideFinHeight),
                length: CGFloat(finThickness),
                chamferRadius: CGFloat(finChamferRadius)
            )
            
            let sideFinNode = SCNNode(geometry: sideFinGeometry)
            sideFinNode.eulerAngles.z = Float(sideFinAngle * .pi / 180) // 6.5° angle
            sideFinNode.position = SCNVector3(x: Float(i - 0.5) * 1.2, y: 0, z: 0)
            
            // Apply Vector 3/2 foil material
            let pressureMaterial = SCNMaterial()
            pressureMaterial.diffuse.contents = UIColor.blue // Low pressure default
            pressureMaterial.metalness.contents = 0.8
            pressureMaterial.roughness.contents = 0.2
            pressureMaterial.lightingModel = .physicallyBased
            sideFinGeometry.materials = [pressureMaterial]
            
            sideFinNodes.append(sideFinNode)
            scene.rootNode.addChildNode(sideFinNode)
            
            // Add fin labels
            addFinLabel(to: sideFinNode, text: "Side Fin \(i + 1)", position: SCNVector3(0, sideFinHeight/2 + 0.1, 0))
        }
        
        // Create center fin (14.50 sq.in., symmetric)
        let centerFinGeometry = SCNBox(
            width: CGFloat(centerFinWidth),
            height: CGFloat(centerFinHeight),
            length: CGFloat(finThickness),
            chamferRadius: CGFloat(finChamferRadius)
        )
        
        centerFinNode = SCNNode(geometry: centerFinGeometry)
        centerFinNode?.position = SCNVector3(x: 0, y: 0, z: 0)
        
        let centerMaterial = SCNMaterial()
        centerMaterial.diffuse.contents = UIColor.blue
        centerMaterial.metalness.contents = 0.8
        centerMaterial.roughness.contents = 0.2
        centerMaterial.lightingModel = .physicallyBased
        centerFinGeometry.materials = [centerMaterial]
        
        scene.rootNode.addChildNode(centerFinNode!)
        
        // Add center fin label
        if let centerFin = centerFinNode {
            addFinLabel(to: centerFin, text: "Center Fin", position: SCNVector3(0, centerFinHeight/2 + 0.1, 0))
        }
        
        // Add flow visualization
        updateFlowVisualization(angleOfAttack: 0)
    }
    
    private func addFinLabel(to node: SCNNode, text: String, position: SCNVector3) {
        let textGeometry = SCNText(string: text, extrusionDepth: 0.01)
        textGeometry.font = UIFont.systemFont(ofSize: 0.1, weight: .medium)
        
        let textNode = SCNNode(geometry: textGeometry)
        textNode.position = position
        textNode.scale = SCNVector3(0.5, 0.5, 0.5)
        
        let textMaterial = SCNMaterial()
        textMaterial.diffuse.contents = UIColor.white
        textGeometry.materials = [textMaterial]
        
        node.addChildNode(textNode)
    }
    
    private func setupFlowParticles() {
        flowParticleSystem = SCNParticleSystem()
        guard let particles = flowParticleSystem else { return }
        
        particles.particleColor = UIColor.white
        particles.particleSize = 0.02
        particles.birthRate = 200
        particles.emissionDuration = 2.0
        particles.emitterShape = createFlowEmitterShape()
        particles.particleLifeSpan = 3.0
        particles.particleVelocity = 2.0
        particles.particleVelocityVariation = 0.5
        particles.particleColorVariation = SCNVector4(0.1, 0.1, 0.1, 0.0)
        
        scene.rootNode.addParticleSystem(particles)
    }
    
    private func createFlowEmitterShape() -> SCNGeometry {
        // Create a plane emitter for flow visualization
        let emitterGeometry = SCNPlane(width: 8, height: 6)
        return emitterGeometry
    }
    
    func updatePressureMap(pressureData: [Float]) {
        guard !pressureData.isEmpty else { return }
        
        // Update side fins pressure visualization
        for (index, node) in sideFinNodes.enumerated() {
            guard let material = node.geometry?.materials.first else { continue }
            let pressureIndex = min(index, pressureData.count - 1)
            let pressure = pressureData[pressureIndex]
            updateMaterialPressure(material: material, pressure: pressure)
        }
        
        // Update center fin pressure visualization
        if let material = centerFinNode?.geometry?.materials.first {
            let centerPressure = pressureData.count > 2 ? pressureData[2] : pressureData.first ?? 0
            updateMaterialPressure(material: material, pressure: centerPressure)
        }
        
        // Update flow visualization based on pressure
        let avgPressure = pressureData.reduce(0, +) / Float(pressureData.count)
        updateFlowVisualization(angleOfAttack: avgPressure * 20) // Scale pressure to angle
    }
    
    private func updateMaterialPressure(material: SCNMaterial, pressure: Float) {
        // Create pressure-based color mapping (blue = low pressure, red = high pressure)
        let normalizedPressure = max(0, min(1, pressure))
        let color = UIColor(
            red: CGFloat(normalizedPressure),
            green: 0,
            blue: CGFloat(1.0 - normalizedPressure),
            alpha: 1.0
        )
        
        material.diffuse.contents = color
        
        // Add pressure value label
        if let node = material.diffuse.contents as? UIColor {
            // Update emission for high pressure areas
            if normalizedPressure > 0.7 {
                material.emission.contents = color.withAlphaComponent(0.3)
            } else {
                material.emission.contents = UIColor.clear
            }
        }
    }
    
    func updateFlowVisualization(angleOfAttack: Float) {
        guard let particles = flowParticleSystem else { return }
        
        // Adjust particle system based on angle of attack
        let normalizedAngle = max(0, min(1, angleOfAttack / 20.0))
        
        // Update particle properties for different flow regimes
        if normalizedAngle < 0.5 {
            // Laminar flow (0-10° AoA)
            particles.particleColor = UIColor.cyan
            particles.birthRate = 150
            particles.particleVelocity = 1.5
        } else {
            // Turbulent flow (10-20° AoA)
            particles.particleColor = UIColor.orange
            particles.birthRate = 300
            particles.particleVelocity = 3.0
        }
        
        // Update emitter position based on angle
        if let emitter = particles.emitterShape {
            let emitterNode = SCNNode(geometry: emitter)
            emitterNode.eulerAngles.z = Float(angleOfAttack * .pi / 180)
            particles.emitterShape = emitterNode.geometry
        }
    }
    
    func animateFinRotation(to angle: Float) {
        let animation = SCNAction.rotateBy(
            x: 0,
            y: 0,
            z: CGFloat(angle * .pi / 180),
            duration: 0.5
        )
        
        // Animate side fins
        for node in sideFinNodes {
            node.runAction(animation)
        }
        
        // Animate center fin (symmetric, no rotation)
        // centerFinNode remains static
    }
}