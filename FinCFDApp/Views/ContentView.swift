import SwiftUI

struct ContentView: View {
	@StateObject private var viewModel = FinViewModel()

	var body: some View {
		VStack(spacing: 12) {
			Text("Vector 3/2 Fin CFD Visualizer")
				.font(.headline)

			Fin3DView(visualizer: viewModel.visualizer)
				.frame(minHeight: 280, maxHeight: 360)

			VStack(alignment: .leading, spacing: 8) {
				HStack {
					Text("AoA")
					Spacer()
					Text("\(Int(viewModel.turnAngle))°")
				}
				Slider(value: $viewModel.turnAngle, in: 0...20, step: 0.5)
			}

			HStack {
				VStack(alignment: .leading) {
					Text("Lift")
					Text(String(format: "%.3f", viewModel.liftDrag?.lift ?? 0))
				}
				Spacer()
				VStack(alignment: .leading) {
					Text("Drag")
					Text(String(format: "%.3f", viewModel.liftDrag?.drag ?? 0))
				}
			}

			HStack {
				Text("HRV (ms)")
				Spacer()
				Text(String(format: "%.2f", viewModel.hrv ?? 0))
				Button("Fetch HRV") { viewModel.fetchHRV() }
			}
		}
		.padding()
		.onAppear {
			viewModel.startMonitoring()
			viewModel.visualizer.setupFinModel()
		}
	}
}

struct ContentView_Previews: PreviewProvider {
	static var previews: some View {
		ContentView()
	}
}