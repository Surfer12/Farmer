import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = FinViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Text("Vector 3/2 Fin CFD Visualizer")
                .font(.title2).bold()
                .padding(.top, 8)

            FinSceneView(visualizer: viewModel.visualizer)
                .frame(height: 360)
                .background(Color.black.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 12))

            VStack(spacing: 10) {
                HStack {
                    Text("AoA: \(Int(viewModel.aoa))°")
                    Slider(value: $viewModel.aoa, in: 0...20, step: 1) { Text("AoA") }
                        .onChange(of: viewModel.aoa) { _ in viewModel.userChangedAoA() }
                }
                HStack {
                    Text("Flow: ")
                    Picker("Flow Mode", selection: $viewModel.flowMode) {
                        ForEach(FlowMode.allCases) { mode in
                            Text(mode.rawValue.capitalized).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                HStack {
                    Text("Lift: \(viewModel.liftDrag.lift, specifier: "%.2f")")
                    Spacer()
                    Text("Drag: \(viewModel.liftDrag.drag, specifier: "%.2f")")
                }
                .font(.system(.body, design: .monospaced))

                HStack {
                    Text("HRV: ")
                    Text(viewModel.hrvText)
                    Spacer()
                    Button("Fetch HRV") { viewModel.fetchHRV() }
                }
            }
            .padding(.horizontal)

            HStack {
                Text("IMU Turn: \(Int(viewModel.imuTurnAoA))°")
                Spacer()
                Text("Pressure Δ: \(viewModel.avgPressure, specifier: "%.2f")")
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
        }
        .onAppear {
            viewModel.start()
        }
        .padding()
    }
}