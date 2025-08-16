import SwiftUI

@main
struct VectorFinCFDAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    // Request necessary permissions on app launch
                    requestPermissions()
                }
        }
    }
    
    private func requestPermissions() {
        // HealthKit permissions will be requested by CognitiveTracker
        // Bluetooth permissions will be requested by SensorManager
        print("Vector 3/2 Fin CFD Visualizer starting...")
    }
}