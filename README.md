# FinFlowApp (Vector 3/2 Fin CFD Visualizer)

A SwiftUI + SceneKit app for iOS 17+/macOS 14+ that renders real‑time 3D fin visuals, overlays CFD‑derived pressure maps, predicts lift/drag with Core ML, and correlates with IMU and HRV metrics.

## Project Layout

- `FinFlowApp/FinFlowApp.swift`: App entry (SwiftUI, cross‑platform)
- `FinFlowApp/Views/ContentView.swift`: Main UI
- `FinFlowApp/Views/FinSceneView.swift`: SceneKit bridge (iOS/macOS)
- `FinFlowApp/Rendering/FinVisualizer.swift`: Scene, geometry, flow animations, pressure colormap
- `FinFlowApp/ViewModels/FinViewModel.swift`: Combine orchestration
- `FinFlowApp/ML/FinPredictor.swift`: Core ML wrapper with graceful fallbacks
- `FinFlowApp/Sensors/SensorManager.swift`: CoreMotion + Bluetooth (mockable)
- `FinFlowApp/Cognitive/CognitiveTracker.swift`: HealthKit HRV (mockable)
- `FinFlowApp/Models/DomainModels.swift`: Shared domain types and errors

## Requirements

- Xcode 15+
- iOS 17+ or macOS 14+
- Add your Core ML model: `FinCFDModel.mlmodel` (inputs: `[aoa, rake, re]`; outputs: `[lift, drag]`). Place it in the target and ensure it is compiled to Swift class `FinCFDModel`.

## Entitlements & Info.plist

Enable these capabilities (as needed):

- HealthKit (read Heart Rate Variability SDNN)
- Bluetooth (CoreBluetooth)

And add to `Info.plist`:

- `NSBluetoothAlwaysUsageDescription` = "Access Bluetooth for fin pressure sensors"
- `NSHealthShareUsageDescription` = "Access HealthKit for HRV data"

## Running without sensors/model

The code includes safe fallbacks:

- Define `USE_MOCK_SENSORS` in Build Settings > Other Swift Flags to simulate IMU and pressure data.
- If `FinCFDModel` is absent, a lightweight surrogate computes plausible lift/drag based on AoA and Reynolds number so the app stays interactive.
- If HealthKit is unavailable or not authorized, HRV displays `--`.

## Notes

- The SceneKit fin is a parametric proxy (dimensions approximate 15.00 sq.in. side fins at 6.5° cant and 14.50 sq.in. center fin). Replace with CAD/SCN assets as desired.
- Flow animations illustrate laminar (≈10° AoA) vs turbulent (≈20° AoA) behavior; they are visual metaphors, not a CFD solver.
- Tune color mapping and particle systems to your CFD scale (e.g., 30% pressure differential).

## Build Targets

Create an Xcode project (iOS App and/or macOS App), add the `FinFlowApp` folder to your target(s), and include `FinCFDModel.mlmodel` if available.

## License

MIT
