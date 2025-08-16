// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import SwiftUI

@main
struct VectorFinCFDAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.light)
                .onAppear {
                    // Initialize app components
                    setupAppEnvironment()
                }
        }
    }
    
    private func setupAppEnvironment() {
        // Configure any app-wide settings
        print("Vector 3/2 Fin CFD Visualizer starting...")
        
        // Request necessary permissions on startup
        requestPermissions()
    }
    
    private func requestPermissions() {
        // Note: Actual permission requests are handled by individual managers
        // This is just for logging startup sequence
        print("Requesting sensor and health permissions...")
    }
}