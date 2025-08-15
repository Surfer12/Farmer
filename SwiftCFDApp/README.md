# Vector 3/2 Fin CFD Visualizer (iOS/macOS)

A SwiftUI + SceneKit app that renders real-time 3D pressure maps on Vector 3/2 Blackstix+ fins and correlates hydrodynamic performance with cognitive metrics (IMU, HRV). The codebase is modular, includes error handling, and runs with mock data if sensors or the Core ML model are unavailable.

## Platforms
- iOS 17+ (full feature set: HealthKit, CoreMotion, Bluetooth)
- macOS 14+ (no HealthKit; HRV is simulated)

## Features
- SceneKit 3D fin rendering with laminar/turbulent particle visualizations
- Pressure map colorization (blue→red) and AoA control (0–20°)
- Core ML predictions for lift/drag using `FinCFDModel.mlmodel` (with heuristic fallback)
- Combine pipelines joining IMU angle, pressure sensors, and ML predictions
- Error reporting UI and stubs for unavailable capabilities

## Project Layout
- `App/` SwiftUI entry and app-wide environment
- `UI/` Views (SwiftUI wrappers for SceneKit)
- `Rendering/` SceneKit scene construction and animations
- `Models/` Domain types (specs, metrics)
- `ML/` Predictor wrapper with safe fallbacks
- `Sensors/` IMU and Bluetooth pressure sensor managers
- `Cognitive/` HealthKit HRV tracker with macOS stub
- `ViewModels/` Combine orchestration
- `Utilities/` Shared helpers (errors, color maps)
- `Config/` Constants and UUIDs

## Required Capabilities (iOS)
Add to your target’s Signing & Capabilities:
- HealthKit (read: Heart Rate Variability SDNN)
- Bluetooth LE

Add to `Info.plist`:
- `NSBluetoothAlwaysUsageDescription` = "Access Bluetooth for fin pressure sensors"
- `NSHealthShareUsageDescription` = "Access HealthKit for HRV data"

## Adding the ML Model
Place a compiled Core ML model at the project root and add it to the Xcode target:
- `FinCFDModel.mlmodel` (inputs: `[aoa, rake, re]`, outputs: `[lift, drag]`)

The app will automatically use a physically-informed heuristic if the model is missing or fails to load.

## Build & Run
1. Create an Xcode iOS or macOS SwiftUI App project (iOS 17+/macOS 14+).
2. Drag the folders from this repository into your Xcode project, keeping folder references.
3. (Optional iOS) Add capabilities and `Info.plist` keys listed above.
4. Run on device or simulator. Without sensors, the app simulates pressure data and HRV.

## Notes
- Flow presets: Laminar (≈10° AoA), Turbulent (≈20° AoA)
- Vector 3/2 side fins: 15.00 sq.in. at 6.5°; center: 14.50 sq.in. symmetric
- Reynolds number range: 1e5–1e6

## License
For research and prototyping purposes only. Not for medical use.