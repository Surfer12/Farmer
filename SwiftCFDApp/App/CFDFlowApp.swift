import SwiftUI

@main
struct CFDFlowApp: App {
    @StateObject private var viewModel = FinViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
        #if os(macOS)
        .windowToolbarStyle(.unifiedCompact)
        #endif
    }
}