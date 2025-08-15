# Fin CFD Visualization & Cognitive Integration App

A SwiftUI app for iOS and macOS that visualizes 3D hydrodynamic performance of the Vector 3/2 Blackstix+ fin setup using SceneKit, integrates Core ML surrogate CFD predictions (lift, drag, pressure), and correlates performance with cognitive load metrics (e.g., HRV). Real-time data flow is coordinated with Combine, and iOS sensor inputs (IMU, BLE pressure) are supported.

## Features
- 3D SceneKit visualization of side and center fins with pressure heatmap
- Core ML surrogate for lift/drag and pressure differential predictions
- Real-time streaming of AoA from device motion (iOS)
- Optional BLE pressure sensor ingestion (iOS)
- HealthKit HRV sampling (iOS)
- Cross-platform UI (iOS + macOS) with platform gating

## Requirements
- Xcode 15+
- iOS 17+ or macOS 14+ (Sonoma) target
- A converted Core ML model included in the project as `FinCFDModel.mlmodel`

## Project Structure
```
FinCFDApp/
  README.md
  FinCFDApp.swift                 # @main entry (SwiftUI)
  Views/
    ContentView.swift
    Fin3DView.swift
  Visualization/
    FinVisualizer.swift
  ML/
    FinPredictor.swift
  Sensors/
    SensorManager.swift
  Cognitive/
    CognitiveTracker.swift
  ViewModels/
    FinViewModel.swift
  Utilities/
    Color+Heatmap.swift
  Config/
    iOS-Info.plist
    macOS-Info.plist
    FinCFDApp.entitlements
  Models/
    FinCFDModel.mlmodel           # PLACEHOLDER – add your model
```

## Setup
1. Create an Xcode project (App, SwiftUI, SceneKit enabled) named `FinCFDApp` targeting iOS 17 and/or macOS 14.
2. Drag the contents of this folder into the Xcode project, preserving groups.
3. Add the Core ML model file: `FinCFDApp/Models/FinCFDModel.mlmodel`.
4. Configure capabilities (iOS target):
   - HealthKit (read Heart Rate Variability SDNN)
   - Bluetooth LE (Background Modes if needed)
   - Motion & Fitness
5. Merge the provided `Config/iOS-Info.plist` keys into your target `Info.plist`.
6. Set the iOS entitlements using `Config/FinCFDApp.entitlements` (or add equivalent in the target’s Signing & Capabilities).

## Info.plist Keys (iOS)
Add the following keys if not already present:
- NSBluetoothAlwaysUsageDescription
- NSBluetoothPeripheralUsageDescription
- NSHealthShareUsageDescription
- NSMotionUsageDescription

See `Config/iOS-Info.plist` for example strings.

## Model Contract
The predictor attempts to introspect the Core ML model’s input/output names. It supports a single vector input with 3 features `[aoa, rake, Re]` and expects output `[lift, drag]` (and optionally an additional pressure scalar/array). If names differ, the code falls back to the first available input and output provider.

## Notes
- SceneKit geometry approximates fin surfaces for visualization. Replace with your CAD/SCN assets for higher fidelity.
- BLE integration is scaffolded; implement your specific peripheral/service/characteristics.
- HealthKit requires user authorization on device; HRV is available on supported devices.

## AR Extensions
For AR overlays, add an iOS target with ARKit and swap `SCNView` with `ARSCNView` in `Fin3DView` (iOS-only).

## License
MIT