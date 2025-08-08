// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "UOIF",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "UOIFCore", targets: ["UOIFCore"]),
    .executable(name: "uoif-cli", targets: ["UOIFCLI"])
  ],
  targets: [
    .target(name: "UOIFCore"),
    .executableTarget(name: "UOIFCLI", dependencies: ["UOIFCore"]),
    .testTarget(name: "UOIFCoreTests", dependencies: ["UOIFCore"])
  ]
)

