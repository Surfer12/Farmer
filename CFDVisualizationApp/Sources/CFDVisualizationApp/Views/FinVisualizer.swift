import SceneKit
import Foundation

/// SceneKit-based 3D visualizer for Vector 3/2 Blackstix+ fins with CFD data overlay
class FinVisualizer: SCNNode {
    
    private var sideFinNodes: [SCNNode] = []
    private var centerFinNode: SCNNode?
    private var flowVectorNodes: [SCNNode] = []
    private var pressureMapTexture: SCNMaterialProperty?
    
    // Animation properties
    private var flowAnimation: SCNAction?
    private var pressureAnimation: SCNAction?
    
    override init() {
        super.init()
        setupScene()
        setupLighting()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupScene()
        setupLighting()
    }
    
    /// Initial scene setup with fin geometries
    private func setupScene() {
        setupFinModels()
        setupCoordinateSystem()
    }
    
    /// Creates the 3D fin models based on Vector 3/2 specifications
    private func setupFinModels() {
        // Create side fins (Vector 3/2 foil)
        createSideFins()
        
        // Create center fin (symmetric foil)
        createCenterFin()
        
        // Position fins in thruster configuration
        positionFins()
    }
    
    /// Creates the two side fins with 6.5° angle
    private func createSideFins() {
        let sideFinGeometry = FinGeometry.sideFin
        
        // Left side fin
        let leftFinNode = createFinNode(geometry: sideFinGeometry, name: "LeftSideFin")
        leftFinNode.position = SCNVector3(x: -1.5, y: 0, z: 0)
        leftFinNode.eulerAngles.z = Float(sideFinGeometry.angle * .pi / 180) // 6.5° angle
        sideFinNodes.append(leftFinNode)
        addChildNode(leftFinNode)
        
        // Right side fin
        let rightFinNode = createFinNode(geometry: sideFinGeometry, name: "RightSideFin")
        rightFinNode.position = SCNVector3(x: 1.5, y: 0, z: 0)
        rightFinNode.eulerAngles.z = -Float(sideFinGeometry.angle * .pi / 180) // -6.5° angle
        sideFinNodes.append(rightFinNode)
        addChildNode(rightFinNode)
    }
    
    /// Creates the center fin with symmetric foil
    private func createCenterFin() {
        let centerFinGeometry = FinGeometry.centerFin
        let centerNode = createFinNode(geometry: centerFinGeometry, name: "CenterFin")
        centerNode.position = SCNVector3(x: 0, y: 0, z: 0)
        
        centerFinNode = centerNode
        addChildNode(centerNode)
    }
    
    /// Creates a SceneKit node for a fin with the specified geometry
    private func createFinNode(geometry: FinGeometry, name: String) -> SCNNode {
        let scnGeometry = geometry.createSCNGeometry()
        let finNode = SCNNode(geometry: scnGeometry)
        finNode.name = name
        
        // Create materials for pressure visualization
        let material = SCNMaterial()
        material.diffuse.contents = createDefaultPressureTexture()
        material.specular.contents = UIColor.white
        material.shininess = 0.8
        material.transparency = 0.9
        
        scnGeometry.materials = [material]
        
        return finNode
    }
    
    /// Positions fins in the correct thruster configuration
    private func positionFins() {
        // Fins are already positioned in create methods
        // Additional positioning adjustments can be made here
        
        // Rotate entire fin system to match surfboard orientation
        self.eulerAngles.x = Float(-90 * .pi / 180) // Point fins downward
    }
    
    /// Sets up lighting for optimal 3D visualization
    private func setupLighting() {
        // Ambient light
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.color = UIColor(white: 0.3, alpha: 1.0)
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        addChildNode(ambientNode)
        
        // Key light
        let keyLight = SCNLight()
        keyLight.type = .directional
        keyLight.color = UIColor.white
        keyLight.intensity = 1000
        let keyLightNode = SCNNode()
        keyLightNode.light = keyLight
        keyLightNode.position = SCNVector3(x: 5, y: 5, z: 5)
        keyLightNode.look(at: SCNVector3Zero)
        addChildNode(keyLightNode)
        
        // Fill light
        let fillLight = SCNLight()
        fillLight.type = .directional
        fillLight.color = UIColor(white: 0.8, alpha: 1.0)
        fillLight.intensity = 500
        let fillLightNode = SCNNode()
        fillLightNode.light = fillLight
        fillLightNode.position = SCNVector3(x: -3, y: 2, z: 3)
        fillLightNode.look(at: SCNVector3Zero)
        addChildNode(fillLightNode)
    }
    
    /// Creates coordinate system visualization
    private func setupCoordinateSystem() {
        let axisLength: Float = 2.0
        let axisRadius: Float = 0.02
        
        // X-axis (red)
        let xAxis = SCNCylinder(radius: CGFloat(axisRadius), height: CGFloat(axisLength))
        let xAxisNode = SCNNode(geometry: xAxis)
        xAxis.firstMaterial?.diffuse.contents = UIColor.red
        xAxisNode.position = SCNVector3(axisLength/2, 0, 0)
        xAxisNode.eulerAngles.z = Float(-90 * .pi / 180)
        addChildNode(xAxisNode)
        
        // Y-axis (green)
        let yAxis = SCNCylinder(radius: CGFloat(axisRadius), height: CGFloat(axisLength))
        let yAxisNode = SCNNode(geometry: yAxis)
        yAxis.firstMaterial?.diffuse.contents = UIColor.green
        yAxisNode.position = SCNVector3(0, axisLength/2, 0)
        addChildNode(yAxisNode)
        
        // Z-axis (blue)
        let zAxis = SCNCylinder(radius: CGFloat(axisRadius), height: CGFloat(axisLength))
        let zAxisNode = SCNNode(geometry: zAxis)
        zAxis.firstMaterial?.diffuse.contents = UIColor.blue
        zAxisNode.position = SCNVector3(0, 0, axisLength/2)
        zAxisNode.eulerAngles.x = Float(90 * .pi / 180)
        addChildNode(zAxisNode)
    }
    
    /// Updates the pressure map visualization with CFD data
    func updatePressureMap(cfdData: CFDData) {
        let pressureTexture = createPressureTexture(from: cfdData.pressureMap)
        
        // Update all fin materials
        updateFinMaterial(sideFinNodes[0], texture: pressureTexture)
        updateFinMaterial(sideFinNodes[1], texture: pressureTexture)
        if let centerFin = centerFinNode {
            updateFinMaterial(centerFin, texture: pressureTexture)
        }
        
        // Animate pressure changes
        animatePressureTransition()
    }
    
    /// Updates flow vector visualization
    func updateFlowVectors(flowVectors: [FlowVector]) {
        // Remove existing flow vectors
        flowVectorNodes.forEach { $0.removeFromParentNode() }
        flowVectorNodes.removeAll()
        
        // Create new flow vectors
        for vector in flowVectors {
            let vectorNode = createFlowVectorNode(vector: vector)
            flowVectorNodes.append(vectorNode)
            addChildNode(vectorNode)
        }
        
        // Animate flow vectors
        animateFlowVectors()
    }
    
    /// Creates a visual representation of a flow vector
    private func createFlowVectorNode(vector: FlowVector) -> SCNNode {
        let vectorLength = vector.magnitude * 0.5
        let vectorRadius: Float = 0.01
        
        // Create arrow shaft
        let shaft = SCNCylinder(radius: CGFloat(vectorRadius), height: CGFloat(vectorLength))
        let shaftNode = SCNNode(geometry: shaft)
        
        // Create arrow head
        let head = SCNCone(topRadius: 0, bottomRadius: CGFloat(vectorRadius * 3), height: CGFloat(vectorLength * 0.2))
        let headNode = SCNNode(geometry: head)
        headNode.position = SCNVector3(0, Float(vectorLength/2 + vectorLength * 0.1), 0)
        
        // Combine shaft and head
        let arrowNode = SCNNode()
        arrowNode.addChildNode(shaftNode)
        arrowNode.addChildNode(headNode)
        
        // Position and orient the arrow
        arrowNode.position = SCNVector3(vector.position.x, vector.position.y, vector.position.z)
        
        // Calculate rotation to align with velocity vector
        let velocity = SCNVector3(vector.velocity.vx, vector.velocity.vy, vector.velocity.vz)
        let up = SCNVector3(0, 1, 0)
        arrowNode.look(at: SCNVector3(
            arrowNode.position.x + velocity.x,
            arrowNode.position.y + velocity.y,
            arrowNode.position.z + velocity.z
        ), up: up, localFront: SCNVector3(0, 1, 0))
        
        // Apply color based on magnitude
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(
            red: CGFloat(vector.color.r),
            green: CGFloat(vector.color.g),
            blue: CGFloat(vector.color.b),
            alpha: 0.8
        )
        shaft.materials = [material]
        head.materials = [material]
        
        return arrowNode
    }
    
    /// Creates a default pressure texture
    private func createDefaultPressureTexture() -> UIImage {
        let size = CGSize(width: 256, height: 256)
        let renderer = UIGraphicsImageRenderer(size: size)
        
        return renderer.image { context in
            // Create a blue-to-red gradient (low to high pressure)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let colors = [UIColor.blue.cgColor, UIColor.cyan.cgColor, UIColor.green.cgColor, UIColor.yellow.cgColor, UIColor.red.cgColor]
            let gradient = CGGradient(colorsSpace: colorSpace, colors: colors as CFArray, locations: [0, 0.25, 0.5, 0.75, 1.0])!
            
            context.cgContext.drawLinearGradient(
                gradient,
                start: CGPoint(x: 0, y: 0),
                end: CGPoint(x: size.width, y: 0),
                options: []
            )
        }
    }
    
    /// Creates pressure texture from CFD data
    private func createPressureTexture(from pressureMap: [Float]) -> UIImage {
        let gridSize = Int(sqrt(Double(pressureMap.count)))
        let size = CGSize(width: gridSize, height: gridSize)
        let renderer = UIGraphicsImageRenderer(size: size)
        
        return renderer.image { context in
            for i in 0..<gridSize {
                for j in 0..<gridSize {
                    let index = i * gridSize + j
                    let pressure = pressureMap[index]
                    
                    // Map pressure to color (blue = low, red = high)
                    let color = UIColor(
                        red: CGFloat(pressure),
                        green: CGFloat(0.5),
                        blue: CGFloat(1.0 - pressure),
                        alpha: 1.0
                    )
                    
                    color.setFill()
                    let rect = CGRect(x: j, y: i, width: 1, height: 1)
                    context.cgContext.fill(rect)
                }
            }
        }
    }
    
    /// Updates fin material with new texture
    private func updateFinMaterial(_ finNode: SCNNode, texture: UIImage) {
        guard let geometry = finNode.geometry,
              let material = geometry.materials.first else { return }
        
        material.diffuse.contents = texture
    }
    
    /// Animates pressure map transitions
    private func animatePressureTransition() {
        let fadeOut = SCNAction.fadeOpacity(to: 0.7, duration: 0.2)
        let fadeIn = SCNAction.fadeOpacity(to: 0.9, duration: 0.2)
        let sequence = SCNAction.sequence([fadeOut, fadeIn])
        
        sideFinNodes.forEach { $0.runAction(sequence) }
        centerFinNode?.runAction(sequence)
    }
    
    /// Animates flow vectors
    private func animateFlowVectors() {
        for vectorNode in flowVectorNodes {
            let moveAction = SCNAction.moveBy(x: 0.1, y: 0, z: 0, duration: 1.0)
            let repeatAction = SCNAction.repeatForever(moveAction)
            vectorNode.runAction(repeatAction)
        }
    }
    
    /// Updates visualization with complete CFD data
    func updateVisualization(with cfdData: CFDData) {
        updatePressureMap(cfdData: cfdData)
        updateFlowVectors(flowVectors: cfdData.flowVectors)
    }
    
    /// Animates angle of attack changes
    func animateAngleOfAttack(to newAngle: Float, duration: TimeInterval = 1.0) {
        let rotationAction = SCNAction.rotateTo(
            x: CGFloat(-90 * .pi / 180), // Base orientation
            y: CGFloat(newAngle * .pi / 180), // AoA rotation
            z: 0,
            duration: duration
        )
        
        runAction(rotationAction)
    }
    
    /// Resets the visualization to default state
    func resetVisualization() {
        removeAllActions()
        flowVectorNodes.forEach { $0.removeFromParentNode() }
        flowVectorNodes.removeAll()
        
        // Reset materials to default
        let defaultTexture = createDefaultPressureTexture()
        sideFinNodes.forEach { updateFinMaterial($0, texture: defaultTexture) }
        centerFinNode.map { updateFinMaterial($0, texture: defaultTexture) }
    }
}