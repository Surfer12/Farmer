import SwiftUI

@main
struct FinFlowApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .windowStyle(.automatic)
        #endif
    }
}