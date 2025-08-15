# FinCFDVisualizer (Vector 3/2 Blackstix+)

A SwiftUI iOS/macOS app for real-time 3D visualization of CFD-derived fin performance for the Vector 3/2 Blackstix+ fin set, integrating:

- SceneKit 3D rendering of side and center fins with dynamic pressure maps
- Core ML surrogate model predictions for lift/drag vs. AoA, rake, and Re (k-ω SST approximations assumed)
- Combine-based data flow for real-time updates
- CoreMotion IMU for turn angle/AoA estimation
- CoreBluetooth (placeholder pipeline) for external pressure sensors
- HealthKit HRV (SDNN) as a cognitive load metric to correlate rider flow state

## Requirements
- Xcode 15+
- iOS 17+ (primary)
- macOS 13+ (visualization-only; HRV and IMU pipelines are disabled by default)
- A Core ML model named `FinCFDModel.mlmodel` trained on your CFD dataset. Place it in the Xcode project root so that Xcode generates `FinCFDModel` Swift APIs.

## Project Structure
- `FinCFDVisualizerApp.swift`: SwiftUI app entry
- `Views/ContentView.swift`: Main UI
- `Views/FinSceneView.swift`: Cross-platform SceneKit view wrapper
- `Visualization/FinVisualizer.swift`: SceneKit scene, fin geometry, pressure mapping
- `ViewModels/FinViewModel.swift`: Combine-based orchestrator
- `ML/FinPredictor.swift`: Core ML wrapper with physics fallback
- `Sensors/SensorManager.swift`: IMU (iOS) and placeholder BLE pressure pipeline
- `Cognitive/CognitiveTracker.swift`: HealthKit HRV
- `Utilities/PlatformTypes.swift`: Cross-platform color/image typedefs and helpers
- `Utilities/PressureColorMap.swift`: Heatmap generation for pressure maps

## Setup in Xcode
1) Create a new SwiftUI iOS app (optionally add a macOS target).
2) Add all files from `/workspace/FinCFDVisualizer` into the Xcode project, preserving folder structure.
3) Add `FinCFDModel.mlmodel` to the project. Confirm the generated class name is `FinCFDModel`.
4) Enable Capabilities:
   - HealthKit (iOS) – read HRV SDNN
   - Bluetooth (Background Modes if you need background BLE)
5) Info.plist keys (iOS):
   - `NSBluetoothAlwaysUsageDescription` = "This app reads fin pressure sensors over Bluetooth to render pressure maps."
   - `NSMotionUsageDescription` = "Used to estimate angle of attack during turns via device motion."
   - `NSHealthShareUsageDescription` = "Used to read HRV for flow state correlation."
   - `NSHealthUpdateUsageDescription` = "Used to manage Health data access."

macOS target: SceneKit visualization is supported. HealthKit/IMU/BLE are stubbed by default. Remove those files from the macOS build target or leave conditional compilation in place.

## Notes on Physics and Ranges
- AoA range: 0°–20°; soft stall modeled near 15° in fallback approximation.
- Reynolds numbers: ~1e5–1e6 typical; default 1e6.
- Vector 3/2 Blackstix+ specs:
  - Side fins: 15.00 sq.in., 6.5° rake, 3/2 foil with concave and scimitar tip
  - Center fin: 14.50 sq.in., symmetric foil

## How to Run
- iOS: Select your device/simulator in Xcode and run. Live IMU requires a physical device.
- macOS: Build the macOS target; IMU/HealthKit are disabled; visualization and ML inference still work.

## Extensibility
- Replace placeholder fin geometry (SCNBox) with a parametric `SCNGeometry` or load a `.scn`/`.usdz` fin mesh.
- Implement your BLE pressure sensor service/characteristic UUIDs in `SensorManager`.
- Add ARKit overlays for in-water augmented visualization.
- Move heatmap generation to a Metal shader for higher performance on large textures.
