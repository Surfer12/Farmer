# Vector 3/2 Fin CFD Visualizer

A comprehensive iOS/macOS application that provides real-time 3D visualization of CFD results for Vector 3/2 Blackstix+ fins, integrating hydrodynamic performance with cognitive load metrics to optimize flow state for surfers.

## 🏄‍♂️ Overview

This app delivers real-time 3D visualization of CFD results (lift, drag, pressure maps) for the Vector 3/2 Blackstix+ fin system, featuring:
- **15.00 sq.in. side fins** with 6.5° rake angle and Vector 3/2 foil
- **14.50 sq.in. center fin** with symmetric design
- **Real-time cognitive integration** using HRV data and IMU sensors
- **Advanced flow state optimization** for riders (125–175 lbs)

## ✨ Key Features

### 🎯 3D Visualization
- **SceneKit-powered 3D rendering** of fin models with real-time pressure mapping
- **Dynamic pressure visualization** showing 30% pressure differential across fin surfaces
- **Flow animation** for laminar (10°) and turbulent (20°) flow regimes
- **Interactive camera controls** for detailed inspection

### 🧠 Cognitive Integration
- **HealthKit integration** for real-time HRV monitoring
- **Flow state assessment** based on cognitive load and physiological metrics
- **Performance optimization** recommendations based on current state
- **Cognitive load analysis** using pressure complexity metrics

### 📊 Performance Analytics
- **Real-time lift/drag predictions** using Core ML neural surrogates
- **CFD-based performance metrics** with k-ω SST turbulence model insights
- **Pressure distribution analysis** across fin surfaces
- **Efficiency scoring** and performance trends

### 🔌 Sensor Integration
- **IMU motion tracking** for turn angle measurement (0°–20° AoA)
- **Bluetooth pressure sensors** for real-time fin pressure data
- **Combine framework** for asynchronous data processing
- **Real-time data pipelines** with optimized performance

## 🏗️ Architecture

### Core Components
- **FinVisualizer**: SceneKit-based 3D rendering and pressure mapping
- **FinPredictor**: Core ML integration for lift/drag predictions
- **SensorManager**: IMU and Bluetooth sensor data management
- **CognitiveTracker**: HealthKit integration and flow state analysis
- **FinViewModel**: Combine-based data pipeline and state management

### Data Flow
```
Sensors → SensorManager → FinViewModel → UI Updates
   ↓           ↓            ↓
IMU/HRV → CognitiveTracker → Flow State
   ↓           ↓            ↓
Pressure → FinVisualizer → 3D Updates
   ↓           ↓            ↓
AoA → FinPredictor → Performance Metrics
```

## 🚀 Setup Instructions

### Prerequisites
- **Xcode 15.0+** with iOS 17.0+ deployment target
- **iOS 17.0+** or **macOS 14.0+** device
- **Apple Developer Account** for device deployment
- **Apple Watch** (optional, for HRV data)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VectorFinCFDApp
   ```

2. **Open in Xcode**
   ```bash
   open VectorFinCFDApp.xcodeproj
   ```

3. **Configure capabilities**
   - Enable **HealthKit** in Signing & Capabilities
   - Enable **Bluetooth** in Signing & Capabilities
   - Ensure **Motion & Fitness** is enabled

4. **Build and run**
   - Select target device (iPhone 14 Pro+ recommended)
   - Build and run the project

### Permissions Setup

The app will request the following permissions:
- **HealthKit**: For HRV data access
- **Bluetooth**: For pressure sensor connectivity
- **Motion & Fitness**: For device motion tracking

## 📱 Usage Guide

### Main Interface
1. **3D Visualization**: Interactive 3D fin model with pressure mapping
2. **Performance Metrics**: Real-time lift, drag, and efficiency data
3. **Cognitive Integration**: HRV monitoring and flow state assessment
4. **Control Panel**: Adjustable angle of attack (0°–20°)
5. **Real-time Data**: Live sensor feeds and monitoring status

### Performance Analysis
- **Tap "Details"** in Performance Metrics for comprehensive analysis
- **View efficiency scores** and performance trends
- **Analyze pressure distribution** across fin surfaces
- **Monitor lift/drag ratios** for optimization

### Cognitive Monitoring
- **Tap "Details"** in Cognitive Integration for deep insights
- **View HRV analysis** and flow state assessment
- **Monitor cognitive load** and performance recommendations
- **Track flow state** changes over time

### Data Export
- **Tap "Export"** in Real-time Data section
- **Choose format**: JSON, CSV, or Plain Text
- **Share data** via AirDrop, Messages, or Files app
- **Copy to clipboard** for external analysis

## 🔬 Technical Specifications

### CFD Integration
- **Reynolds Number Range**: 10⁵ – 10⁶
- **Angle of Attack**: 0° – 20°
- **Pressure Differential**: 30% across fin surfaces
- **Lift Increase**: 12% for raked fins vs. pivot fins

### Performance Metrics
- **Lift Force**: 0–100 N range
- **Drag Force**: 0–50 N range
- **Efficiency Score**: 0–100% based on L/D ratio
- **Flow State**: Optimal, Good, Moderate, Challenging

### Cognitive Metrics
- **HRV Range**: 0–100 ms (optimal >50 ms)
- **Cognitive Load**: 0–1 scale (lower is better)
- **Flow State Confidence**: 0–100% based on data quality

## 🧪 Testing

### Device Testing
- **iPhone 14 Pro+**: Optimal performance with all features
- **iPhone 13+**: Good performance with most features
- **iPad Pro**: Excellent 3D visualization experience

### Sensor Testing
- **IMU Motion**: Test device rotation for turn angle accuracy
- **HealthKit**: Verify HRV data access and permissions
- **Bluetooth**: Test pressure sensor connectivity (mock data available)

### Performance Testing
- **3D Rendering**: Verify smooth 60fps visualization
- **Data Processing**: Check real-time sensor data updates
- **Memory Usage**: Monitor app memory consumption

## 🔧 Customization

### Fin Specifications
Modify `FinVisualizer.swift` to adjust:
- Fin dimensions and areas
- Rake angles and foil types
- Pressure mapping algorithms
- Flow visualization parameters

### Cognitive Thresholds
Adjust `CognitiveTracker.swift` for:
- HRV optimal thresholds
- Flow state boundaries
- Cognitive load calculations
- Recommendation algorithms

### Performance Models
Update `FinPredictor.swift` to:
- Modify physics-based fallbacks
- Adjust Core ML model parameters
- Customize prediction confidence
- Update efficiency calculations

## 📊 Data Export Formats

### JSON Export
```json
{
  "metadata": {
    "app": "Vector 3/2 Fin CFD Visualizer",
    "version": "1.0.0",
    "export_timestamp": "2025-01-15T10:30:00Z"
  },
  "session_data": {
    "turn_angle": 12.5,
    "lift_force": 45.2,
    "drag_force": 8.7,
    "flow_state": "Good Flow"
  }
}
```

### CSV Export
```csv
Metric,Value,Unit,Description
Turn Angle,12.5,degrees,Current angle of attack
Lift Force,45.2,N,Vertical force generated by fins
Drag Force,8.7,N,Resistance force against water
```

## 🚨 Troubleshooting

### Common Issues

**HealthKit Permissions Denied**
- Check Health app permissions
- Verify device compatibility
- Restart app after permission changes

**Bluetooth Connection Issues**
- Ensure Bluetooth is enabled
- Check device compatibility
- Verify sensor proximity

**3D Rendering Performance**
- Close background apps
- Reduce device motion updates
- Check device temperature

**Data Export Failures**
- Verify file permissions
- Check available storage
- Restart app and try again

### Performance Optimization
- **Reduce update frequency** for better battery life
- **Limit 3D complexity** on older devices
- **Optimize sensor polling** based on usage
- **Cache processed data** for offline analysis

## 🔮 Future Enhancements

### Planned Features
- **ARKit integration** for augmented reality overlays
- **Machine learning optimization** for personalized recommendations
- **Cloud synchronization** for multi-device data sharing
- **Advanced analytics** with historical trend analysis

### Research Integration
- **CFD validation studies** with experimental data
- **Cognitive performance correlation** research
- **Flow state optimization** algorithms
- **Performance prediction** improvements

## 📚 References

### Technical Documentation
- [SceneKit Programming Guide](https://developer.apple.com/scenekit/)
- [Core ML Framework](https://developer.apple.com/documentation/coreml)
- [HealthKit Framework](https://developer.apple.com/healthkit/)
- [Combine Framework](https://developer.apple.com/documentation/combine)

### Research Papers
- Vector 3/2 foil performance analysis
- k-ω SST turbulence model applications
- Cognitive load and athletic performance
- Flow state optimization in sports

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature proposals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For technical support or questions:
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this README and inline code comments
- **Community**: Join our developer community discussions

---

**Built with ❤️ for the surfing community**

*Optimize your flow state, enhance your performance, and master the waves with data-driven insights.*