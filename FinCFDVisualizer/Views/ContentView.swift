// SPDX-License-Identifier: GPL-3.0-only
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Text("Vector 3/2 Blackstix+ CFD Visualizer")
                .font(.title3)
                .fontWeight(.semibold)
                .padding(.top, 8)

            FinSceneView(visualizer: viewModel.visualizer)
                .frame(minHeight: 320)
                .cornerRadius(8)

            Picker("Mode", selection: $viewModel.isManualAoA) {
                Text("Live (IMU)").tag(false)
                Text("Manual").tag(true)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            if viewModel.isManualAoA {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Angle of Attack: \(Int(viewModel.manualAoADegrees))°")
                    Slider(value: $viewModel.manualAoADegrees, in: 0...20, step: 0.5)
                }
                .padding(.horizontal)
            } else {
                Text("AoA (IMU): \(Int(viewModel.liveAoADegrees))°")
            }

            HStack(spacing: 16) {
                VStack(alignment: .leading) {
                    Text("Lift Coefficient")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.3f", viewModel.liftCoefficient))
                        .monospacedDigit()
                }
                VStack(alignment: .leading) {
                    Text("Drag Coefficient")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.3f", viewModel.dragCoefficient))
                        .monospacedDigit()
                }
                VStack(alignment: .leading) {
                    Text("Re")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(viewModel.reynoldsDisplayString)
                        .monospacedDigit()
                }
            }

            HStack(spacing: 12) {
                Button("Laminar (10°)") { viewModel.setLaminar() }
                Button("Turbulent (20°)") { viewModel.setTurbulent() }
            }

            VStack(spacing: 6) {
                Text("Ψ Score: " + String(format: "%.3f", viewModel.psiScore))
                    .font(.headline)
                HStack {
                    Text("α")
                    Slider(value: Binding(get: { viewModel.psiParameters.alpha }, set: { viewModel.psiParameters.alpha = $0 }), in: 0...1, step: 0.05)
                }
                .padding(.horizontal)
            }

            HStack(spacing: 12) {
                Button("Fetch HRV") { viewModel.fetchHRV() }
                if let hrv = viewModel.latestHRVms {
                    Text("HRV: \(String(format: "%.1f", hrv)) ms")
                } else {
                    Text("HRV: -")
                        .foregroundColor(.secondary)
                }
            }

            Spacer(minLength: 8)
        }
        .onAppear { viewModel.startup() }
        .padding(.bottom)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}