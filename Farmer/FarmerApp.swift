//
//  FarmerApp.swift
//  Farmer
//
//  Created by Ryan David Oates on 8/7/25.
//

import SwiftUI

@main
struct FarmerApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
