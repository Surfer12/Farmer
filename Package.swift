// swift-tools-version: 5.9
// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import PackageDescription

let package = Package(
    name: "UOIF",
    platforms: [
        .macOS(.v13),
        .iOS(.v17)
    ],
    products: [
        .library(name: "UOIFCore", targets: ["UOIFCore"]),
        .executable(name: "uoif-cli", targets: ["UOIFCLI"]),
        .library(name: "VectorFinCFD", targets: ["VectorFinCFD"])
    ],
    dependencies: [
        // Add any external dependencies here
    ],
    targets: [
        .target(
            name: "UOIFCore",
            dependencies: [],
            path: "Sources/UOIFCore"
        ),
        .executableTarget(
            name: "UOIFCLI",
            dependencies: ["UOIFCore"],
            path: "Sources/UOIFCLI"
        ),
        .target(
            name: "VectorFinCFD",
            dependencies: [],
            path: "VectorFinCFDApp",
            resources: [
                .process("Assets.xcassets"),
                .process("FinCFDModel.mlmodel")
            ]
        ),
        .testTarget(
            name: "UOIFCoreTests",
            dependencies: ["UOIFCore"],
            path: "Tests/UOIFCoreTests"
        ),
        .testTarget(
            name: "VectorFinCFDTests",
            dependencies: ["VectorFinCFD"],
            path: "Tests/VectorFinCFDTests"
        )
    ]
)

