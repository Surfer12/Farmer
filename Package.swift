// swift-tools-version: 5.9
import PackageDescription

let package = Package(
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
)

