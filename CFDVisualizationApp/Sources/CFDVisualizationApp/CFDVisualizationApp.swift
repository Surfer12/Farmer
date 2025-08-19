import SwiftUI

/// Main app entry point for CFD Visualization and Cognitive Integration
@main
struct CFDVisualizationApp: App {
    var body: some Scene {
        WindowGroup {
            CFDVisualizationView()
                .preferredColorScheme(.dark) // Optimize for 3D visualization
        }
    }
}

#if os(macOS)
// macOS-specific app configuration
extension CFDVisualizationApp {
    var body: some Scene {
        WindowGroup {
            CFDVisualizationView()
                .frame(minWidth: 1024, minHeight: 768)
        }
        .windowStyle(DefaultWindowStyle())
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Session") {
                    // Create new CFD session
                }
                .keyboardShortcut("n", modifiers: .command)
            }
            
            CommandGroup(after: .newItem) {
                Button("Export Data") {
                    // Export CFD data
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }
    }
}
#endif