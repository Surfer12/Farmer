import SwiftUI

@main
struct FinCFDVisualizerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark) // Better for 3D visualization
        }
    }
}