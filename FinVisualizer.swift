import SceneKit
import UIKit

class FinVisualizer: ObservableObject {
    let scene = SCNScene()
    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var pressureNodes: [SCNNode] = []
    private var flowStreamlines: [SCNNode] = []
    
    // Fin specifications
    private let sideFinArea: Float = 15.00 // sq.in.
    private let centerFinArea: Float = 14.50 // sq.in.
    private let rakeAngle: Float = 6.5 // degrees
    private let foilType = "Vector 3/2"
    
    func setupFinModel() {
        setupLighting()
        createFinGeometry()
        createPressureVisualization()
        createFlowStreamlines()
        setupCamera()
    }
    
    private func setupLighting() {
        // Ambient light
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor(white: 0.3, alpha: 1.0)
        let ambientLightNode = SCNNode()
        ambientLightNode.light = ambientLight
        scene.rootNode.addChildNode(ambientLightNode)
        
        // Directional light
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.color = UIColor.white
        directionalLight.intensity = 1000
        let directionalLightNode = SCNNode()
        directionalLightNode.light = directionalLight
        directionalLightNode.position = SCNVector3(x: 10, y: 10, z: 10)
        directionalLightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(directionalLightNode)
    }
    
    private func createFinGeometry() {
        // Create side fins (Vector 3/2 foil profile)
        createSideFins()
        
        // Create center fin (symmetric profile)
        createCenterFin()
        
        // Create fin box/mount
        createFinBox()
    }
    
    private func createSideFins() {
        let finProfile = createVector32Profile()
        
        // Left side fin
        let leftFin = SCNNode(geometry: finProfile)
        leftFin.position = SCNVector3(x: -2.5, y: 0, z: 0)
        leftFin.eulerAngles = SCNVector3(0, 0, Float(rakeAngle * .pi / 180))
        
        // Right side fin
        let rightFin = SCNNode(geometry: finProfile)
        rightFin.position = SCNVector3(x: 2.5, y: 0, z: 0)
        rightFin.eulerAngles = SCNVector3(0, 0, -Float(rakeAngle * .pi / 180))
        
        sideFinNodes = [leftFin, rightFin]
        scene.rootNode.addChildNode(leftFin)
        scene.rootNode.addChildNode(rightFin)
    }
    
    private func createCenterFin() {
        let centerProfile = createSymmetricProfile()
        centerFinNode = SCNNode(geometry: centerProfile)
        centerFinNode?.position = SCNVector3(x: 0, y: 0, z: 0)
        
        if let centerFin = centerFinNode {
            scene.rootNode.addChildNode(centerFin)
        }
    }
    
    private func createVector32Profile() -> SCNGeometry {
        // Simplified Vector 3/2 foil geometry (concave surface, Scimitar tip)
        let finHeight: Float = 4.5
        let finLength: Float = 3.5
        let finThickness: Float = 0.15
        
        // Create custom geometry for Vector 3/2 profile
        let vertices: [SCNVector3] = [
            // Leading edge (curved)
            SCNVector3(0, 0, finThickness/2),
            SCNVector3(0, 0, -finThickness/2),
            SCNVector3(0.2, 0.5, finThickness/2),
            SCNVector3(0.2, 0.5, -finThickness/2),
            
            // Mid section (concave)
            SCNVector3(1.5, 2.0, finThickness/2),
            SCNVector3(1.5, 2.0, -finThickness/2),
            SCNVector3(2.5, 3.5, finThickness/2),
            SCNVector3(2.5, 3.5, -finThickness/2),
            
            // Scimitar tip
            SCNVector3(finLength, finHeight, finThickness/2),
            SCNVector3(finLength, finHeight, -finThickness/2),
            SCNVector3(finLength + 0.5, finHeight - 0.5, 0),
        ]
        
        let indices: [Int32] = [
            // Simplified triangular faces for the fin profile
            0, 1, 2, 1, 2, 3,
            2, 3, 4, 3, 4, 5,
            4, 5, 6, 5, 6, 7,
            6, 7, 8, 7, 8, 9,
            8, 9, 10
        ]
        
        let geometrySource = SCNGeometrySource(vertices: vertices)
        let geometryElement = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        
        let geometry = SCNGeometry(sources: [geometrySource], elements: [geometryElement])
        
        // Apply fin material
        let finMaterial = SCNMaterial()
        finMaterial.diffuse.contents = UIColor.systemBlue
        finMaterial.specular.contents = UIColor.white
        finMaterial.shininess = 0.8
        finMaterial.metalness.contents = 0.3
        finMaterial.roughness.contents = 0.2
        geometry.materials = [finMaterial]
        
        return geometry
    }
    
    private func createSymmetricProfile() -> SCNGeometry {
        // Symmetric center fin profile
        let finHeight: Float = 4.2
        let finLength: Float = 3.2
        let finThickness: Float = 0.12
        
        let box = SCNBox(width: CGFloat(finLength), height: CGFloat(finHeight), length: CGFloat(finThickness), chamferRadius: 0.05)
        
        let centerMaterial = SCNMaterial()
        centerMaterial.diffuse.contents = UIColor.systemGreen
        centerMaterial.specular.contents = UIColor.white
        centerMaterial.shininess = 0.8
        box.materials = [centerMaterial]
        
        return box
    }
    
    private func createFinBox() {
        let boxGeometry = SCNBox(width: 6, height: 0.3, length: 1.5, chamferRadius: 0.05)
        let boxMaterial = SCNMaterial()
        boxMaterial.diffuse.contents = UIColor.darkGray
        boxGeometry.materials = [boxMaterial]
        
        let boxNode = SCNNode(geometry: boxGeometry)
        boxNode.position = SCNVector3(x: 0, y: -2.5, z: 0)
        scene.rootNode.addChildNode(boxNode)
    }
    
    private func createPressureVisualization() {
        // Create pressure field visualization nodes
        for i in 0..<20 {
            for j in 0..<15 {
                let pressureNode = SCNNode(geometry: SCNSphere(radius: 0.05))
                pressureNode.position = SCNVector3(
                    x: Float(i - 10) * 0.5,
                    y: Float(j - 7) * 0.4,
                    z: 2.0
                )
                
                let material = SCNMaterial()
                material.diffuse.contents = UIColor.blue.withAlphaComponent(0.3)
                pressureNode.geometry?.materials = [material]
                pressureNode.isHidden = true // Initially hidden
                
                pressureNodes.append(pressureNode)
                scene.rootNode.addChildNode(pressureNode)
            }
        }
    }
    
    private func createFlowStreamlines() {
        // Create flow visualization streamlines
        for i in 0..<10 {
            let streamline = createStreamlineGeometry(index: i)
            let streamlineNode = SCNNode(geometry: streamline)
            streamlineNode.position = SCNVector3(x: -8, y: Float(i - 5) * 0.8, z: Float(i) * 0.1)
            
            flowStreamlines.append(streamlineNode)
            scene.rootNode.addChildNode(streamlineNode)
        }
    }
    
    private func createStreamlineGeometry(index: Int) -> SCNGeometry {
        // Create curved streamline geometry
        var vertices: [SCNVector3] = []
        let numPoints = 50
        
        for i in 0..<numPoints {
            let t = Float(i) / Float(numPoints - 1)
            let x = t * 12 - 6
            let y = sin(t * .pi * 2) * 0.5 + Float(index - 5) * 0.8
            let z = cos(t * .pi) * 0.2
            vertices.append(SCNVector3(x: x, y: y, z: z))
        }
        
        var indices: [Int32] = []
        for i in 0..<(numPoints - 1) {
            indices.append(Int32(i))
            indices.append(Int32(i + 1))
        }
        
        let geometrySource = SCNGeometrySource(vertices: vertices)
        let geometryElement = SCNGeometryElement(indices: indices, primitiveType: .line)
        
        let geometry = SCNGeometry(sources: [geometrySource], elements: [geometryElement])
        
        let streamlineMaterial = SCNMaterial()
        streamlineMaterial.diffuse.contents = UIColor.cyan.withAlphaComponent(0.6)
        streamlineMaterial.emission.contents = UIColor.cyan.withAlphaComponent(0.3)
        geometry.materials = [streamlineMaterial]
        
        return geometry
    }
    
    private func setupCamera() {
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 2, z: 12)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)
    }
    
    // MARK: - Dynamic Updates
    
    func updatePressureMap(pressureData: [Float]) {
        guard pressureData.count >= pressureNodes.count else { return }
        
        for (index, node) in pressureNodes.enumerated() {
            let pressure = pressureData[index]
            let normalizedPressure = max(0, min(1, (pressure + 1) / 2)) // Normalize -1 to 1 range
            
            // Color mapping: blue (low pressure) to red (high pressure)
            let color = UIColor(
                red: CGFloat(normalizedPressure),
                green: 0,
                blue: CGFloat(1.0 - normalizedPressure),
                alpha: 0.7
            )
            
            node.geometry?.materials.first?.diffuse.contents = color
            node.isHidden = false
        }
    }
    
    func updateFlowVisualization(angleOfAttack: Float, velocity: Float) {
        // Update streamlines based on angle of attack
        for (index, streamlineNode) in flowStreamlines.enumerated() {
            let deflection = sin(angleOfAttack * .pi / 180) * 2.0
            
            // Animate streamline deflection
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 0.5
            streamlineNode.position.y = Float(index - 5) * 0.8 + deflection
            SCNTransaction.commit()
            
            // Update color based on velocity
            let velocityColor = UIColor(
                red: CGFloat(velocity / 10.0),
                green: 1.0 - CGFloat(velocity / 10.0),
                blue: 0.5,
                alpha: 0.6
            )
            streamlineNode.geometry?.materials.first?.diffuse.contents = velocityColor
        }
    }
    
    func updateFinOrientation(angleOfAttack: Float) {
        // Rotate fins based on angle of attack
        let rotationAngle = angleOfAttack * .pi / 180
        
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.3
        
        for finNode in sideFinNodes {
            finNode.eulerAngles.x = rotationAngle
        }
        
        centerFinNode?.eulerAngles.x = rotationAngle
        
        SCNTransaction.commit()
    }
    
    func generatePressureField(aoa: Float, velocity: Float) -> [Float] {
        // Generate synthetic pressure data based on CFD principles
        var pressureField: [Float] = []
        
        for i in 0..<300 { // 20x15 grid
            let x = Float(i % 20 - 10) * 0.5
            let y = Float(i / 20 - 7) * 0.4
            
            // Simplified pressure calculation around fin
            let distance = sqrt(x * x + y * y)
            let baseP = -0.5 * velocity * velocity // Dynamic pressure
            
            // Pressure variation with AoA and position
            let aoaEffect = sin(aoa * .pi / 180) * x * 0.1
            let distanceEffect = exp(-distance / 3.0)
            
            let pressure = baseP * distanceEffect + aoaEffect
            pressureField.append(pressure)
        }
        
        return pressureField
    }
}