import SceneKit
import UIKit
import Metal
import MetalKit

class FinVisualizer: NSObject {
    
    // MARK: - Properties
    
    let scene = SCNScene()
    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var flowParticleSystem: SCNParticleSystem?
    private var pressureVisualizationNode: SCNNode?
    
    // Fin geometry parameters
    private let finHeight: Float = 4.48 // Based on 15.00 sq.in area
    private let finWidth: Float = 4.63
    private let finThickness: Float = 0.15
    
    // Animation properties
    private var currentAngleOfAttack: Float = 0.0
    private var isAnimating = false
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupScene()
        setupLighting()
        setupCamera()
    }
    
    // MARK: - Scene Setup
    
    private func setupScene() {
        scene.background.contents = UIColor.black
        scene.physicsWorld.gravity = SCNVector3(0, -9.8, 0)
    }
    
    private func setupLighting() {
        // Ambient light
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor(white: 0.3, alpha: 1.0)
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)
        
        // Directional light (sun)
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.color = UIColor.white
        directionalLight.intensity = 1000
        directionalLight.castsShadow = true
        directionalLight.shadowRadius = 5.0
        directionalLight.shadowColor = UIColor.black.withAlphaComponent(0.5)
        
        let lightNode = SCNNode()
        lightNode.light = directionalLight
        lightNode.position = SCNVector3(10, 10, 10)
        lightNode.look(at: SCNVector3.zero)
        scene.rootNode.addChildNode(lightNode)
        
        // Spot light for dramatic effect
        let spotLight = SCNLight()
        spotLight.type = .spot
        spotLight.color = UIColor.cyan
        spotLight.intensity = 500
        spotLight.spotInnerAngle = 30
        spotLight.spotOuterAngle = 80
        
        let spotNode = SCNNode()
        spotNode.light = spotLight
        spotNode.position = SCNVector3(0, 8, 8)
        spotNode.look(at: SCNVector3.zero)
        scene.rootNode.addChildNode(spotNode)
    }
    
    private func setupCamera() {
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        cameraNode.position = SCNVector3(0, 5, 15)
        cameraNode.look(at: SCNVector3.zero)
        scene.rootNode.addChildNode(cameraNode)
    }
    
    // MARK: - Fin Model Creation
    
    func setupFinModel() {
        createSideFins()
        createCenterFin()
        createFlowVisualization()
        createPressureVisualization()
        addEnvironmentalElements()
    }
    
    private func createSideFins() {
        // Create side fins (Vector 3/2, 15.00 sq.in., 6.5° angle)
        for i in 0..<2 {
            let finGeometry = createFinGeometry(spec: .vector32SideFin)
            let finNode = SCNNode(geometry: finGeometry)
            
            // Position side fins
            let xOffset: Float = (i == 0) ? -2.5 : 2.5
            finNode.position = SCNVector3(xOffset, 0, 0)
            
            // Apply 6.5° rake angle
            finNode.eulerAngles.z = Float(6.5 * .pi / 180)
            
            // Add materials and shaders
            setupFinMaterials(for: finNode, finType: .vector32)
            
            sideFinNodes.append(finNode)
            scene.rootNode.addChildNode(finNode)
        }
    }
    
    private func createCenterFin() {
        // Create center fin (Symmetric, 14.50 sq.in.)
        let finGeometry = createFinGeometry(spec: .vector32CenterFin)
        centerFinNode = SCNNode(geometry: finGeometry)
        
        centerFinNode?.position = SCNVector3(0, 0, 0)
        
        // Setup materials
        if let centerNode = centerFinNode {
            setupFinMaterials(for: centerNode, finType: .symmetric)
            scene.rootNode.addChildNode(centerNode)
        }
    }
    
    private func createFinGeometry(spec: FinSpecification) -> SCNGeometry {
        // Create custom fin geometry based on specifications
        let geometry: SCNGeometry
        
        switch spec.foilType {
        case .vector32:
            geometry = createVector32Geometry()
        case .symmetric:
            geometry = createSymmetricGeometry()
        case .scimitarTip:
            geometry = createScimitarGeometry()
        }
        
        return geometry
    }
    
    private func createVector32Geometry() -> SCNGeometry {
        // Create Vector 3/2 foil shape with optimized lift profile
        let vertices: [SCNVector3] = [
            // Leading edge (rounded)
            SCNVector3(0, 0, finThickness/2),
            SCNVector3(0, 0, -finThickness/2),
            
            // Mid-section (maximum thickness)
            SCNVector3(finWidth * 0.3, 0, finThickness/2),
            SCNVector3(finWidth * 0.3, 0, -finThickness/2),
            
            // Trailing edge (sharp)
            SCNVector3(finWidth, 0, 0),
            
            // Tip section
            SCNVector3(finWidth * 0.8, finHeight, 0),
            SCNVector3(0, finHeight, 0)
        ]
        
        return createMeshFromVertices(vertices)
    }
    
    private func createSymmetricGeometry() -> SCNGeometry {
        // Create symmetric foil for center fin
        let box = SCNBox(width: CGFloat(finWidth), height: CGFloat(finHeight), length: CGFloat(finThickness), chamferRadius: 0.1)
        return box
    }
    
    private func createScimitarGeometry() -> SCNGeometry {
        // Create scimitar tip geometry for reduced drag
        let cylinder = SCNCylinder(radius: CGFloat(finWidth/2), height: CGFloat(finHeight))
        return cylinder
    }
    
    private func createMeshFromVertices(_ vertices: [SCNVector3]) -> SCNGeometry {
        // Create custom mesh geometry
        let vertexData = Data(bytes: vertices, count: vertices.count * MemoryLayout<SCNVector3>.size)
        
        let source = SCNGeometrySource(data: vertexData,
                                     semantic: .vertex,
                                     vectorCount: vertices.count,
                                     usesFloatComponents: true,
                                     componentsPerVector: 3,
                                     bytesPerComponent: MemoryLayout<Float>.size,
                                     dataOffset: 0,
                                     dataStride: MemoryLayout<SCNVector3>.size)
        
        // Create simple triangulated indices
        let indices: [UInt16] = [0, 1, 2, 1, 2, 3, 2, 3, 4, 4, 5, 6]
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<UInt16>.size)
        
        let element = SCNGeometryElement(data: indexData,
                                       primitiveType: .triangles,
                                       primitiveCount: indices.count / 3,
                                       bytesPerIndex: MemoryLayout<UInt16>.size)
        
        return SCNGeometry(sources: [source], elements: [element])
    }
    
    // MARK: - Materials and Shaders
    
    private func setupFinMaterials(for node: SCNNode, finType: FoilType) {
        guard let geometry = node.geometry else { return }
        
        // Base material
        let baseMaterial = SCNMaterial()
        baseMaterial.diffuse.contents = UIColor.systemBlue
        baseMaterial.metalness.contents = 0.8
        baseMaterial.roughness.contents = 0.2
        baseMaterial.normal.contents = UIImage(named: "fin_normal_map")
        
        // Pressure visualization material
        let pressureMaterial = SCNMaterial()
        pressureMaterial.diffuse.contents = UIColor.blue // Default low pressure
        pressureMaterial.emission.contents = UIColor.blue.withAlphaComponent(0.3)
        pressureMaterial.transparency = 0.8
        
        geometry.materials = [baseMaterial, pressureMaterial]
        
        // Add custom shader for pressure visualization
        setupPressureShader(for: geometry)
    }
    
    private func setupPressureShader(for geometry: SCNGeometry) {
        // Custom Metal shader for pressure visualization
        let shaderCode = """
        #include <metal_stdlib>
        using namespace metal;
        
        vertex float4 pressure_vertex(float4 position [[attribute(0)]],
                                    constant float4x4& modelViewProjection [[buffer(0)]]) {
            return modelViewProjection * position;
        }
        
        fragment float4 pressure_fragment(float4 position [[stage_in]],
                                        constant float& pressure [[buffer(0)]]) {
            float3 lowPressureColor = float3(0.0, 0.0, 1.0);  // Blue
            float3 highPressureColor = float3(1.0, 0.0, 0.0); // Red
            
            float3 color = mix(lowPressureColor, highPressureColor, pressure);
            return float4(color, 1.0);
        }
        """
        
        if let material = geometry.materials.first {
            material.shaderModifiers = [
                .fragment: shaderCode
            ]
        }
    }
    
    // MARK: - Flow Visualization
    
    private func createFlowVisualization() {
        // Create particle system for flow visualization
        flowParticleSystem = SCNParticleSystem()
        
        guard let particles = flowParticleSystem else { return }
        
        // Configure particle system
        particles.particleColor = UIColor.cyan
        particles.particleColorVariation = SCNVector4(0.2, 0.2, 0.2, 0.0)
        particles.particleSize = 0.1
        particles.particleSizeVariation = 0.05
        particles.birthRate = 200
        particles.emissionDuration = 0
        particles.particleLifeSpan = 3.0
        particles.particleVelocity = 5.0
        particles.particleVelocityVariation = 2.0
        
        // Flow direction and behavior
        particles.acceleration = SCNVector3(2, 0, 0) // Flow from left to right
        particles.emitterShape = SCNBox(width: 0.5, height: 8, length: 0.5, chamferRadius: 0)
        
        // Add turbulence for realistic flow
        particles.dampingFactor = 0.1
        particles.spreadingAngle = 15
        
        // Create emitter node
        let emitterNode = SCNNode()
        emitterNode.position = SCNVector3(-8, 0, 0)
        emitterNode.addParticleSystem(particles)
        scene.rootNode.addChildNode(emitterNode)
    }
    
    private func createPressureVisualization() {
        // Create pressure field visualization
        pressureVisualizationNode = SCNNode()
        
        // Add pressure field grid
        for x in stride(from: -5.0, through: 5.0, by: 0.5) {
            for y in stride(from: -3.0, through: 3.0, by: 0.5) {
                let pressureIndicator = SCNSphere(radius: 0.05)
                let indicatorNode = SCNNode(geometry: pressureIndicator)
                indicatorNode.position = SCNVector3(Float(x), Float(y), 0)
                
                let material = SCNMaterial()
                material.diffuse.contents = UIColor.blue.withAlphaComponent(0.6)
                pressureIndicator.materials = [material]
                
                pressureVisualizationNode?.addChildNode(indicatorNode)
            }
        }
        
        if let pressureNode = pressureVisualizationNode {
            scene.rootNode.addChildNode(pressureNode)
        }
    }
    
    private func addEnvironmentalElements() {
        // Add water surface
        let waterGeometry = SCNPlane(width: 20, height: 20)
        let waterNode = SCNNode(geometry: waterGeometry)
        waterNode.position = SCNVector3(0, -5, 0)
        waterNode.eulerAngles.x = -.pi / 2
        
        let waterMaterial = SCNMaterial()
        waterMaterial.diffuse.contents = UIColor.blue.withAlphaComponent(0.3)
        waterMaterial.transparency = 0.7
        waterMaterial.isDoubleSided = true
        waterGeometry.materials = [waterMaterial]
        
        scene.rootNode.addChildNode(waterNode)
        
        // Add subtle wave animation
        let waveAction = SCNAction.repeatForever(
            SCNAction.sequence([
                SCNAction.rotateBy(x: 0, y: 0.1, z: 0, duration: 2.0),
                SCNAction.rotateBy(x: 0, y: -0.1, z: 0, duration: 2.0)
            ])
        )
        waterNode.runAction(waveAction)
    }
    
    // MARK: - Real-time Updates
    
    func updatePressureMap(pressureData: [Double]) {
        guard !pressureData.isEmpty else { return }
        
        // Update fin surface colors based on pressure data
        updateFinPressureColors(pressureData: pressureData)
        
        // Update pressure field visualization
        updatePressureField(pressureData: pressureData)
    }
    
    private func updateFinPressureColors(pressureData: [Double]) {
        let avgPressure = pressureData.reduce(0, +) / Double(pressureData.count)
        let normalizedPressure = max(0, min(1, avgPressure))
        
        // Create pressure color (blue = low, red = high)
        let pressureColor = UIColor(
            red: CGFloat(normalizedPressure),
            green: 0.0,
            blue: CGFloat(1.0 - normalizedPressure),
            alpha: 0.8
        )
        
        // Update all fin materials
        let allFins = sideFinNodes + [centerFinNode].compactMap { $0 }
        for finNode in allFins {
            if let material = finNode.geometry?.materials.last {
                material.diffuse.contents = pressureColor
                material.emission.contents = pressureColor.withAlphaComponent(0.3)
            }
        }
    }
    
    private func updatePressureField(pressureData: [Double]) {
        guard let pressureNode = pressureVisualizationNode else { return }
        
        let childNodes = pressureNode.childNodes
        for (index, node) in childNodes.enumerated() {
            let pressureIndex = min(index, pressureData.count - 1)
            let pressure = pressureData[pressureIndex]
            let normalizedPressure = max(0, min(1, pressure))
            
            if let sphere = node.geometry as? SCNSphere,
               let material = sphere.materials.first {
                let color = UIColor(
                    red: CGFloat(normalizedPressure),
                    green: 0.0,
                    blue: CGFloat(1.0 - normalizedPressure),
                    alpha: 0.6
                )
                material.diffuse.contents = color
                
                // Scale based on pressure intensity
                let scale = Float(0.5 + normalizedPressure * 1.5)
                node.scale = SCNVector3(scale, scale, scale)
            }
        }
    }
    
    func updateAngleOfAttack(_ angle: Float) {
        currentAngleOfAttack = max(0, min(20, angle)) // Clamp to 0-20°
        
        // Animate fin rotation
        let angleRadians = currentAngleOfAttack * .pi / 180
        
        // Rotate all fins
        let allFins = sideFinNodes + [centerFinNode].compactMap { $0 }
        for finNode in allFins {
            let rotateAction = SCNAction.rotateTo(x: 0, y: CGFloat(angleRadians), z: 0, duration: 0.3)
            rotateAction.timingMode = .easeInEaseOut
            finNode.runAction(rotateAction)
        }
        
        // Update flow visualization based on angle
        updateFlowVisualization(for: currentAngleOfAttack)
    }
    
    private func updateFlowVisualization(for angle: Float) {
        guard let particles = flowParticleSystem else { return }
        
        // Adjust particle behavior based on flow regime
        let flowRegime = determineFlowRegime(for: angle)
        
        switch flowRegime {
        case .laminar:
            particles.particleVelocityVariation = 1.0
            particles.spreadingAngle = 5
            particles.particleColor = UIColor.green
            
        case .transitional:
            particles.particleVelocityVariation = 3.0
            particles.spreadingAngle = 15
            particles.particleColor = UIColor.yellow
            
        case .turbulent:
            particles.particleVelocityVariation = 5.0
            particles.spreadingAngle = 30
            particles.particleColor = UIColor.red
        }
    }
    
    private func determineFlowRegime(for angle: Float) -> FlowRegime {
        switch angle {
        case 0...10:
            return .laminar
        case 10...15:
            return .transitional
        default:
            return .turbulent
        }
    }
    
    // MARK: - Animation Control
    
    func startAnimation() {
        guard !isAnimating else { return }
        isAnimating = true
        
        // Start continuous rotation animation for demonstration
        let rotateAction = SCNAction.repeatForever(
            SCNAction.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 10.0)
        )
        
        scene.rootNode.runAction(rotateAction, forKey: "continuous_rotation")
    }
    
    func stopAnimation() {
        isAnimating = false
        scene.rootNode.removeAction(forKey: "continuous_rotation")
    }
    
    // MARK: - Utility Methods
    
    func resetVisualization() {
        updateAngleOfAttack(0)
        updatePressureMap(pressureData: Array(repeating: 0.5, count: 10))
    }
    
    func exportScene() -> SCNScene {
        return scene.copy() as! SCNScene
    }
}