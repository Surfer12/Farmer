#if canImport(SwiftUI) && canImport(Charts)
import SwiftUI
import Charts

struct SamplePoint: Identifiable {
	let id = UUID()
	let x: Double
	let value: Double
	let series: String
}

struct SolutionChart: View {
	var body: some View {
		let xVals: [Double] = stride(from: -1.0, through: 1.0, by: 0.1).map { $0 }
		let pinnU: [Double] = [0.0, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.4, 0.1, -0.3, -0.6, -0.8, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.0]
		let rk4U:  [Double] = [0.0, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.3, 0.0, -0.4, -0.7, -0.8, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.0]

		let pinnSeries = zip(xVals, pinnU).map { SamplePoint(x: $0.0, value: $0.1, series: "PINN (t=1)") }
		let rk4Series = zip(xVals, rk4U).map { SamplePoint(x: $0.0, value: $0.1, series: "RK4 (t=1)") }
		let points = pinnSeries + rk4Series

		Chart(points) {
			LineMark(
				x: .value("x", $0.x),
				y: .value("u", $0.value)
			)
			.foregroundStyle(by: .value("Series", $0.series))
		}
		.chartLegend(.visible)
		.padding()
	}
}

#Preview {
	SolutionChart()
}
#endif