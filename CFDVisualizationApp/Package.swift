// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CFDVisualizationApp",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "CFDVisualizationApp",
            targets: ["CFDVisualizationApp"]),
    ],
    dependencies: [
        // Add any external dependencies here if needed
    ],
    targets: [
        .target(
            name: "CFDVisualizationApp",
            dependencies: [],
            resources: [
                .process("Resources")
            ]),
        .testTarget(
            name: "CFDVisualizationAppTests",
            dependencies: ["CFDVisualizationApp"]),
    ]
)