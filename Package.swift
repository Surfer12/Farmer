// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FinCFDVisualizer",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "FinCFDVisualizer",
            targets: ["FinCFDVisualizer"]
        ),
    ],
    dependencies: [
        // Add external dependencies here if needed
        // Example: Networking, additional ML frameworks, etc.
    ],
    targets: [
        .target(
            name: "FinCFDVisualizer",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "FinCFDVisualizerTests",
            dependencies: ["FinCFDVisualizer"]
        ),
    ]
)

