import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: FinViewModel

    var body: some View {
        VStack(spacing: 16) {
            Text("Vector 3/2 Fin CFD Visualizer").font(.title2).bold()

            FinSceneView(visualizer: model.visualizer)
                .frame(minHeight: 320, maxHeight: 420)
                .background(Color.black.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

            VStack(spacing: 12) {
                HStack {
                    Text("AoA")
                    Slider(value: Binding(
                        get: { Double(model.aoaDegrees) },
                        set: { model.aoaDegrees = Float($0) }
                    ), in: Double(Defaults.aoaDegreesRange.lowerBound)...Double(Defaults.aoaDegreesRange.upperBound))
                    Text("\(Int(model.aoaDegrees))°").monospacedDigit()
                }
                Picker("Flow", selection: $model.flowMode) {
                    ForEach(FlowMode.allCases) { mode in
                        Text(mode.rawValue.capitalized).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                HStack {
                    VStack(alignment: .leading) {
                        Text("Lift").foregroundStyle(.secondary)
                        Text(String(format: "%.3f", model.liftDrag.lift)).font(.title3).monospacedDigit()
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("Drag").foregroundStyle(.secondary)
                        Text(String(format: "%.3f", model.liftDrag.drag)).font(.title3).monospacedDigit()
                    }
                }

                HStack {
                    VStack(alignment: .leading) {
                        Text("HRV SDNN").foregroundStyle(.secondary)
                        Text(model.hrvSDNNms.map { String(format: "%.1f ms", $0) } ?? "—").monospacedDigit()
                    }
                    Spacer()
                    Button {
                        model.start()
                    } label: {
                        Label("Start", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(12)
            .background(.thinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .padding()
        .errorOverlay(model.appError)
    }
}

#Preview {
    ContentView().environmentObject(FinViewModel())
}