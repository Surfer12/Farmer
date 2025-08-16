# Vector Fin CFD App - Project Configuration Guide

This guide provides step-by-step instructions for configuring the Xcode project to ensure all features work correctly.

## 🏗️ Project Setup

### 1. Xcode Project Configuration

#### Target Settings
- **Deployment Target**: iOS 17.0+ / macOS 14.0+
- **Device Orientation**: Portrait, Landscape Left, Landscape Right
- **Status Bar**: Default style, View controller-based appearance: NO

#### Build Settings
- **Swift Language Version**: Swift 5.9+
- **iOS Deployment Target**: 17.0
- **macOS Deployment Target**: 14.0
- **Enable Bitcode**: NO (for better compatibility)

### 2. Capabilities Configuration

#### Required Capabilities
1. **HealthKit**
   - Enable in Signing & Capabilities
   - Add `NSHealthShareUsageDescription` to Info.plist
   - Add `NSHealthUpdateUsageDescription` to Info.plist

2. **Bluetooth**
   - Enable in Signing & Capabilities
   - Add `NSBluetoothAlwaysUsageDescription` to Info.plist
   - Add `NSBluetoothPeripheralUsageDescription` to Info.plist

3. **Motion & Fitness**
   - Enable in Signing & Capabilities
   - Add `NSMotionUsageDescription` to Info.plist

#### Optional Capabilities
- **Background Modes**: Background processing, Background fetch
- **App Groups**: For data sharing between app extensions

### 3. Info.plist Configuration

#### Required Keys
```xml
<key>NSBluetoothAlwaysUsageDescription</key>
<string>This app uses Bluetooth to connect to fin pressure sensors for real-time performance monitoring.</string>

<key>NSBluetoothPeripheralUsageDescription</key>
<string>This app connects to Bluetooth pressure sensors to monitor fin performance.</string>

<key>NSHealthShareUsageDescription</key>
<string>This app accesses your heart rate variability data to optimize your surfing performance and flow state.</string>

<key>NSHealthUpdateUsageDescription</key>
<string>This app may update your health data to track performance improvements over time.</string>

<key>NSMotionUsageDescription</key>
<string>This app uses device motion to track your turn angles and surfing technique for performance analysis.</string>
```

#### Optional Keys
```xml
<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>armv7</string>
    <string>bluetooth-le</string>
</array>

<key>UIBackgroundModes</key>
<array>
    <string>background-processing</string>
    <string>background-fetch</string>
</array>
```

## 📱 Device Requirements

### iOS Devices
- **iPhone 14 Pro+**: Optimal performance, all features
- **iPhone 13+**: Good performance, most features
- **iPhone 12+**: Adequate performance, basic features
- **iPad Pro**: Excellent 3D visualization experience

### macOS Devices
- **M2 MacBook**: Excellent performance, all features
- **M1 MacBook**: Good performance, most features
- **Intel MacBook**: Adequate performance, basic features

### Hardware Requirements
- **Motion Sensors**: Built-in IMU for device motion tracking
- **Bluetooth**: LE 4.0+ for pressure sensor connectivity
- **HealthKit**: Compatible with Apple Health ecosystem

## 🔧 Build Configuration

### Debug Configuration
- **Optimization Level**: None [-O0]
- **Debug Information**: DWARF with dSYM File
- **Enable Bitcode**: NO
- **Swift Compilation Mode**: Incremental

### Release Configuration
- **Optimization Level**: Fastest, Smallest [-Os]
- **Debug Information**: DWARF
- **Enable Bitcode**: NO
- **Swift Compilation Mode**: Whole Module

### Build Phases
1. **Compile Sources**: All Swift files
2. **Link Binary With Libraries**:
   - SceneKit.framework
   - CoreML.framework
   - CoreMotion.framework
   - CoreBluetooth.framework
   - HealthKit.framework
   - Combine.framework

## 📦 Dependencies

### System Frameworks
- **SceneKit**: 3D rendering and visualization
- **Core ML**: Machine learning predictions
- **Core Motion**: Device motion and IMU data
- **Core Bluetooth**: Bluetooth sensor connectivity
- **HealthKit**: Health data access
- **Combine**: Reactive programming and data flow

### Third-Party Dependencies
- **None required**: All functionality uses Apple frameworks
- **Optional**: Core ML model files (.mlmodel)

## 🧪 Testing Configuration

### Unit Tests
- **Target**: VectorFinCFDAppTests
- **Framework**: XCTest
- **Coverage**: Aim for 80%+ code coverage

### UI Tests
- **Target**: VectorFinCFDAppUITests
- **Framework**: XCTest
- **Automation**: UI element identification and interaction

### Test Devices
- **Simulator**: iOS 17.0+ for basic functionality
- **Physical Device**: Required for sensor testing
- **Multiple Devices**: Test on different screen sizes

## 🚀 Deployment

### Development
- **Provisioning Profile**: Development profile
- **Signing**: Automatic signing with development team
- **Distribution**: Ad Hoc or Development

### Production
- **Provisioning Profile**: Distribution profile
- **Signing**: Manual signing with distribution certificate
- **Distribution**: App Store Connect

### Archive Process
1. Select "Any iOS Device" as target
2. Product → Archive
3. Validate archive in Organizer
4. Upload to App Store Connect

## 🔍 Troubleshooting

### Build Issues
- **Swift Compiler Errors**: Check Swift version compatibility
- **Framework Linking**: Verify framework inclusion
- **Capability Errors**: Ensure all capabilities are properly configured

### Runtime Issues
- **Permission Denials**: Check Info.plist descriptions
- **Sensor Failures**: Verify device compatibility
- **Performance Issues**: Check device specifications

### Deployment Issues
- **Code Signing**: Verify certificates and provisioning profiles
- **App Store Rejection**: Check App Store guidelines compliance
- **Device Compatibility**: Verify minimum deployment target

## 📚 Additional Resources

### Apple Documentation
- [SceneKit Programming Guide](https://developer.apple.com/scenekit/)
- [Core ML Framework](https://developer.apple.com/documentation/coreml)
- [HealthKit Framework](https://developer.apple.com/healthkit/)
- [Combine Framework](https://developer.apple.com/documentation/combine)

### Development Tools
- **Xcode**: Primary development environment
- **Instruments**: Performance profiling and debugging
- **Simulator**: iOS device simulation
- **Device Console**: Real-time device logging

### Community Resources
- [Apple Developer Forums](https://developer.apple.com/forums/)
- [Stack Overflow](https://stackoverflow.com/)
- [GitHub Issues](https://github.com/your-repo/issues)

---

**Note**: This configuration guide assumes you're using Xcode 15.0+ and targeting iOS 17.0+. Adjust version numbers and capabilities based on your specific requirements.