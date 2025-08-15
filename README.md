# Fin CFD Visualizer

A comprehensive iOS/macOS app for 3D visualization of CFD (Computational Fluid Dynamics) results for Vector 3/2 Blackstix+ fins, integrating hydrodynamic performance with cognitive load metrics to optimize flow state for surfers.

## Features

### 🌊 3D CFD Visualization
- Real-time 3D rendering of fin geometry using SceneKit
- Pressure field visualization with color-mapped intensity
- Flow streamline visualization showing fluid dynamics
- Vector 3/2 foil profile with accurate geometry (15.00 sq.in. side fins, 14.50 sq.in. center fin)
- Dynamic updates based on angle of attack (0°–20° AoA)

### 🧠 Cognitive Integration
- Heart Rate Variability (HRV) monitoring via HealthKit
- Real-time flow state detection and analysis
- Cognitive load assessment and stress level monitoring
- Performance correlation between fin efficiency and mental state
- Personalized recommendations for optimal performance

### 📱 Sensor Integration
- IMU data processing for board orientation and movement
- Bluetooth pressure sensor connectivity
- Real-time maneuver detection (turns, bottom turns, aerials)
- Sensor calibration and filtering algorithms

### 🤖 Machine Learning
- Core ML integration for CFD predictions
- Neural surrogate models trained on CFD data
- Analytical backup models using fluid dynamics principles
- Boundary layer analysis and turbulence modeling (k-ω SST approximation)

## Technical Specifications

### Fin Configuration
- **Side Fins**: 15.00 sq.in., 6.5° rake angle, Vector 3/2 foil profile
- **Center Fin**: 14.50 sq.in., symmetric profile
- **Target Rider Weight**: 125–175 lbs
- **Reynolds Number Range**: 10⁵–10⁶

### Performance Metrics
- Lift and drag force calculations
- Lift-to-drag ratio optimization
- Pressure differential mapping (up to 30%)
- Flow separation detection
- Stall angle estimation

### Cognitive Metrics
- HRV analysis for flow state detection
- Stress level quantification
- Cognitive load assessment
- Performance-mindset correlation
- Real-time feedback and recommendations

## Requirements

### iOS/macOS
- iOS 17.0+ or macOS 14.0+
- Xcode 15.0+
- Swift 5.9+

### Device Capabilities
- Gyroscope and accelerometer
- Bluetooth LE support
- HealthKit compatibility (optional)
- Metal-compatible GPU for 3D rendering

### Permissions
- Motion sensors access
- HealthKit data access
- Bluetooth peripheral access
- Background processing (for sensor monitoring)

## Installation

### Xcode Project Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fin-cfd-visualizer.git
   cd fin-cfd-visualizer
   ```

2. **Open in Xcode**
   ```bash
   open FinCFDVisualizer.xcodeproj
   ```

3. **Configure Bundle Identifier**
   - Update the bundle identifier in project settings
   - Configure signing certificates

4. **Add Core ML Model**
   - Place your trained `FinCFDModel.mlmodel` in the project
   - Ensure it's added to the target

### Dependencies

The app uses native iOS frameworks:
- **SwiftUI**: Modern UI framework
- **SceneKit**: 3D graphics and visualization
- **Core ML**: Machine learning inference
- **HealthKit**: Health data integration
- **Core Motion**: Motion sensor access
- **Core Bluetooth**: Sensor connectivity
- **Combine**: Reactive programming

## Usage

### Initial Setup

1. **Launch the app** and grant necessary permissions
2. **Calibrate sensors** by placing the device on a flat surface
3. **Connect pressure sensors** via Bluetooth (optional)
4. **Authorize HealthKit** for cognitive monitoring

### Real-Time Monitoring

1. **Angle of Attack Control**: Use the slider or rely on IMU data
2. **Reynolds Number**: Adjust based on water conditions
3. **3D Visualization**: Interact with the 3D model to explore fin geometry
4. **Performance Metrics**: Monitor lift, drag, and L/D ratio
5. **Cognitive State**: Track HRV and flow state indicators

### Session Analysis

1. **Start a session** to begin comprehensive tracking
2. **Monitor real-time feedback** during surfing
3. **Review session summary** with performance metrics
4. **Export data** for further analysis

## Architecture

### Core Components

```
FinCFDVisualizerApp (App Entry Point)
├── ContentView (Main UI)
├── FinVisualizer (3D SceneKit Rendering)
├── FinPredictor (Core ML & CFD Analysis)
├── SensorManager (IMU & Bluetooth)
├── CognitiveTracker (HealthKit & Flow State)
└── FinViewModel (Combine Data Pipeline)
```

### Data Flow

1. **Sensor Input** → Raw IMU and pressure data
2. **Processing** → Filtering and calibration
3. **ML Inference** → CFD predictions via Core ML
4. **Visualization** → 3D rendering updates
5. **Cognitive Analysis** → HRV processing and flow state detection
6. **Correlation** → Performance-mindset integration
7. **Feedback** → Real-time recommendations

## Core ML Model

### Input Format
- Angle of Attack (0-20°)
- Rake Angle (6.5° for Vector 3/2)
- Reynolds Number (1e5-1e6)
- Velocity (m/s)
- Fin Area (sq.in.)

### Output Format
- Lift Force (N)
- Drag Force (N)
- Lift Coefficient
- Drag Coefficient
- Pressure Distribution (300-point grid)
- Velocity Field (300-point grid)

### Training Data
The model should be trained on CFD simulation data covering:
- Various angles of attack (0-20°)
- Reynolds number range (1e5-2e6)
- Different fin geometries
- Boundary layer transitions
- Flow separation conditions

## Cognitive Integration

### Flow State Detection
Based on research in sports psychology and HRV analysis:

- **Flow State**: HRV > 110% baseline, low cognitive load, positive trend
- **Focused State**: HRV > 90% baseline, moderate cognitive load
- **Stressed State**: High cognitive load (>60%), elevated stress markers
- **Fatigued State**: HRV < 70% baseline, declining trend

### Performance Correlation
The app correlates fin performance with cognitive state:
- Optimal zone: High cognitive score + high performance score
- Recommendations based on correlation analysis
- Personalized feedback for improvement

## Customization

### Fin Configurations
Modify `FinConfiguration` in `FinViewModel.swift`:
```swift
private let finConfiguration = FinConfiguration(
    sideFinArea: 15.00,      // sq.in.
    centerFinArea: 14.50,    // sq.in.
    rakeAngle: 6.5,          // degrees
    foilType: .vector32
)
```

### Cognitive Parameters
Adjust baseline values in `CognitiveAnalyzer`:
```swift
private var baselineHRV: Double = 45.0 // ms
private var baselineHR: Double = 70.0  // bpm
```

### Visualization Settings
Customize 3D rendering in `FinVisualizer.swift`:
- Fin geometry parameters
- Pressure field resolution
- Streamline density
- Color mappings

## Testing

### Unit Tests
```bash
# Run unit tests
xcodebuild test -scheme FinCFDVisualizer -destination 'platform=iOS Simulator,name=iPhone 15'
```

### Integration Testing
- Test sensor integration with device
- Validate Core ML model predictions
- Verify HealthKit data access
- Test Bluetooth connectivity

### Performance Testing
- 3D rendering performance
- Real-time data processing
- Memory usage optimization
- Battery life impact

## Troubleshooting

### Common Issues

1. **Core ML Model Not Found**
   - Ensure `FinCFDModel.mlmodel` is in the project bundle
   - Check model compilation settings

2. **HealthKit Authorization Failed**
   - Verify Info.plist permissions
   - Check device HealthKit availability

3. **Sensor Data Not Updating**
   - Calibrate sensors on flat surface
   - Check motion permission settings

4. **Bluetooth Connection Issues**
   - Verify peripheral compatibility
   - Check Bluetooth permissions

### Performance Optimization

1. **3D Rendering**
   - Reduce polygon count for complex geometries
   - Optimize texture sizes
   - Use Level of Detail (LOD) techniques

2. **Data Processing**
   - Implement data decimation for high-frequency sensors
   - Use background queues for heavy computations
   - Cache frequently accessed calculations

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow Swift style guidelines
4. Add unit tests for new features
5. Submit a pull request

### Code Style
- Follow Apple's Swift API Design Guidelines
- Use meaningful variable names
- Document public interfaces
- Maintain consistent indentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CFD modeling based on fluid dynamics principles
- Cognitive flow state research from sports psychology
- Vector 3/2 fin geometry specifications
- k-ω SST turbulence modeling approaches

## Future Enhancements

### Planned Features
- **ARKit Integration**: Augmented reality fin visualization
- **CloudKit Sync**: Cross-device session synchronization
- **Advanced Analytics**: Machine learning performance insights
- **Social Features**: Session sharing and comparison
- **Wearable Integration**: Apple Watch support

### Research Areas
- Advanced turbulence modeling
- Personalized cognitive baselines
- Environmental condition integration
- Multi-fin configuration analysis

## Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Join our community discussions

---

**Note**: This app is designed for educational and research purposes. Always prioritize safety when surfing and use professional guidance for equipment selection.
