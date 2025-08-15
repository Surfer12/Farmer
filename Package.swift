// swift-tools-version: 5.9
import PackageDescription

let package = Package(
<<<<<<< Current (Your changes)
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
=======
	name: "PINN",
	platforms: [
		.macOS(.v13)
	],
	products: [
		.executable(name: "PINN", targets: ["PINN"])
	],
	targets: [
		.executableTarget(
			name: "PINN",
			path: "Sources/PINN"
		)
	]
>>>>>>> Incoming (Background Agent changes)
)

