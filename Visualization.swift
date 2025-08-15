import Foundation

// MARK: - Data Structures for Visualization

/// Data point for plotting
struct PlotPoint {
    let x: Double
    let y: Double
    let z: Double? // For 3D plots (time dimension)
    
    init(x: Double, y: Double, z: Double? = nil) {
        self.x = x
        self.y = y
        self.z = z
    }
}

/// Plot series containing multiple data points
struct PlotSeries {
    let name: String
    let points: [PlotPoint]
    let color: String
    let style: PlotStyle
    
    enum PlotStyle {
        case solid, dashed, dotted
    }
}

// MARK: - Visualization Engine

/// Visualization engine for PINN results
class PINNVisualizer {
    
    /// Generate solution data for visualization
    static func generateSolutionData(pinn: PINN, 
                                   xRange: (Double, Double), 
                                   tRange: (Double, Double),
                                   nx: Int = 100, 
                                   nt: Int = 50) -> [[Double]] {
        
        var solution = Array(repeating: Array(repeating: 0.0, count: nx), count: nt)
        
        let dx = (xRange.1 - xRange.0) / Double(nx - 1)
        let dt = (tRange.1 - tRange.0) / Double(nt - 1)
        
        for i in 0..<nt {
            let t = tRange.0 + Double(i) * dt
            for j in 0..<nx {
                let x = xRange.0 + Double(j) * dx
                solution[i][j] = pinn.forward(x: x, t: t)
            }
        }
        
        return solution
    }
    
    /// Compare PINN and RK4 solutions at specific time
    static func compareSolutions(pinn: PINN, 
                               rk4Solution: [[Double]],
                               time: Double,
                               xRange: (Double, Double),
                               tRange: (Double, Double),
                               nx: Int = 100,
                               nt: Int = 50) -> (pinnSeries: PlotSeries, rk4Series: PlotSeries, errorSeries: PlotSeries) {
        
        let dx = (xRange.1 - xRange.0) / Double(nx - 1)
        let dt = (tRange.1 - tRange.0) / Double(nt - 1)
        let timeIndex = Int(round((time - tRange.0) / dt))
        
        var pinnPoints: [PlotPoint] = []
        var rk4Points: [PlotPoint] = []
        var errorPoints: [PlotPoint] = []
        
        for j in 0..<nx {
            let x = xRange.0 + Double(j) * dx
            let pinnValue = pinn.forward(x: x, t: time)
            let rk4Value = timeIndex < nt ? rk4Solution[timeIndex][j] : 0.0
            let error = abs(pinnValue - rk4Value)
            
            pinnPoints.append(PlotPoint(x: x, y: pinnValue))
            rk4Points.append(PlotPoint(x: x, y: rk4Value))
            errorPoints.append(PlotPoint(x: x, y: error))
        }
        
        let pinnSeries = PlotSeries(name: "PINN", points: pinnPoints, color: "blue", style: .dashed)
        let rk4Series = PlotSeries(name: "RK4", points: rk4Points, color: "red", style: .solid)
        let errorSeries = PlotSeries(name: "Error", points: errorPoints, color: "green", style: .dotted)
        
        return (pinnSeries, rk4Series, errorSeries)
    }
    
    /// Generate heatmap data for 2D visualization
    static func generateHeatmapData(pinn: PINN,
                                  xRange: (Double, Double),
                                  tRange: (Double, Double),
                                  nx: Int = 50,
                                  nt: Int = 50) -> (data: [[Double]], xAxis: [Double], tAxis: [Double]) {
        
        let dx = (xRange.1 - xRange.0) / Double(nx - 1)
        let dt = (tRange.1 - tRange.0) / Double(nt - 1)
        
        var data = Array(repeating: Array(repeating: 0.0, count: nx), count: nt)
        var xAxis: [Double] = []
        var tAxis: [Double] = []
        
        for j in 0..<nx {
            xAxis.append(xRange.0 + Double(j) * dx)
        }
        
        for i in 0..<nt {
            let t = tRange.0 + Double(i) * dt
            tAxis.append(t)
            
            for j in 0..<nx {
                let x = xAxis[j]
                data[i][j] = pinn.forward(x: x, t: t)
            }
        }
        
        return (data, xAxis, tAxis)
    }
    
    /// Export data to CSV format for external plotting
    static func exportToCSV(series: [PlotSeries], filename: String) {
        var csvContent = "x"
        for s in series {
            csvContent += ",\(s.name)"
        }
        csvContent += "\n"
        
        let maxPoints = series.map { $0.points.count }.max() ?? 0
        
        for i in 0..<maxPoints {
            if i < series[0].points.count {
                csvContent += "\(series[0].points[i].x)"
            }
            
            for s in series {
                if i < s.points.count {
                    csvContent += ",\(s.points[i].y)"
                } else {
                    csvContent += ","
                }
            }
            csvContent += "\n"
        }
        
        do {
            try csvContent.write(toFile: filename, atomically: true, encoding: .utf8)
            print("📊 Data exported to \(filename)")
        } catch {
            print("❌ Failed to export data: \(error)")
        }
    }
    
    /// Export heatmap data to CSV
    static func exportHeatmapToCSV(data: [[Double]], 
                                 xAxis: [Double], 
                                 tAxis: [Double], 
                                 filename: String) {
        var csvContent = "t\\x"
        for x in xAxis {
            csvContent += ",\(x)"
        }
        csvContent += "\n"
        
        for (i, t) in tAxis.enumerated() {
            csvContent += "\(t)"
            for j in 0..<xAxis.count {
                csvContent += ",\(data[i][j])"
            }
            csvContent += "\n"
        }
        
        do {
            try csvContent.write(toFile: filename, atomically: true, encoding: .utf8)
            print("🔥 Heatmap data exported to \(filename)")
        } catch {
            print("❌ Failed to export heatmap data: \(error)")
        }
    }
    
    /// Generate ASCII art visualization (for terminal display)
    static func generateASCIIPlot(series: PlotSeries, width: Int = 80, height: Int = 20) -> String {
        guard !series.points.isEmpty else { return "No data to plot" }
        
        let minX = series.points.map { $0.x }.min()!
        let maxX = series.points.map { $0.x }.max()!
        let minY = series.points.map { $0.y }.min()!
        let maxY = series.points.map { $0.y }.max()!
        
        var plot = Array(repeating: Array(repeating: " ", count: width), count: height)
        
        // Plot points
        for point in series.points {
            let x = Int((point.x - minX) / (maxX - minX) * Double(width - 1))
            let y = Int((1.0 - (point.y - minY) / (maxY - minY)) * Double(height - 1))
            
            if x >= 0 && x < width && y >= 0 && y < height {
                plot[y][x] = "*"
            }
        }
        
        // Add axes
        let zeroY = Int((1.0 - (0 - minY) / (maxY - minY)) * Double(height - 1))
        if zeroY >= 0 && zeroY < height {
            for x in 0..<width {
                if plot[zeroY][x] == " " {
                    plot[zeroY][x] = "-"
                }
            }
        }
        
        var result = "📈 \(series.name) (x: \(String(format: "%.2f", minX)) to \(String(format: "%.2f", maxX)), y: \(String(format: "%.2f", minY)) to \(String(format: "%.2f", maxY)))\n"
        result += String(repeating: "=", count: width) + "\n"
        
        for row in plot {
            result += row.joined() + "\n"
        }
        
        result += String(repeating: "=", count: width) + "\n"
        
        return result
    }
    
    /// Generate training loss visualization
    static func visualizeTrainingLoss(losses: [Double]) -> String {
        guard !losses.isEmpty else { return "No loss data available" }
        
        let series = PlotSeries(
            name: "Training Loss",
            points: losses.enumerated().map { PlotPoint(x: Double($0.offset), y: $0.element) },
            color: "blue",
            style: .solid
        )
        
        return generateASCIIPlot(series: series, width: 60, height: 15)
    }
    
    /// Generate hybrid framework metrics visualization
    static func visualizeHybridMetrics(pinn: PINN, 
                                     xRange: (Double, Double), 
                                     tRange: (Double, Double),
                                     numPoints: Int = 50) -> String {
        
        var sValues: [PlotPoint] = []
        var nValues: [PlotPoint] = []
        var alphaValues: [PlotPoint] = []
        var psiValues: [PlotPoint] = []
        
        let dt = (tRange.1 - tRange.0) / Double(numPoints - 1)
        
        for i in 0..<numPoints {
            let t = tRange.0 + Double(i) * dt
            let x = 0.0 // Sample at x = 0
            
            let S = HybridFramework.stateInference(pinn: pinn, x: x, t: t)
            let N = HybridFramework.neuralApproximation(pinn: pinn, x: x, t: t)
            let alpha = HybridFramework.validationFlow(t: t)
            let hybridOutput = HybridFramework.hybridOutput(S: S, N: N, alpha: alpha)
            
            let rCognitive = HybridFramework.cognitiveRegularization(pinn: pinn, x: [x], t: [t])
            let rEfficiency = HybridFramework.efficiencyRegularization(computationTime: 0.05)
            let penalty = exp(-(0.6 * rCognitive + 0.4 * rEfficiency))
            let adjustedProb = HybridFramework.probabilityAdjustment(baseProb: 0.8, beta: 1.2)
            let psi = hybridOutput * penalty * adjustedProb
            
            sValues.append(PlotPoint(x: t, y: S))
            nValues.append(PlotPoint(x: t, y: N))
            alphaValues.append(PlotPoint(x: t, y: alpha))
            psiValues.append(PlotPoint(x: t, y: psi))
        }
        
        var result = "🔬 Hybrid Framework Metrics Evolution\n"
        result += "=====================================\n\n"
        
        result += generateASCIIPlot(series: PlotSeries(name: "S(x) - State Inference", points: sValues, color: "red", style: .solid), width: 50, height: 10)
        result += "\n"
        
        result += generateASCIIPlot(series: PlotSeries(name: "N(x) - Neural Approximation", points: nValues, color: "blue", style: .solid), width: 50, height: 10)
        result += "\n"
        
        result += generateASCIIPlot(series: PlotSeries(name: "α(t) - Validation Flow", points: alphaValues, color: "green", style: .solid), width: 50, height: 10)
        result += "\n"
        
        result += generateASCIIPlot(series: PlotSeries(name: "Ψ(x) - Overall Performance", points: psiValues, color: "purple", style: .solid), width: 50, height: 10)
        
        return result
    }
}

// MARK: - Extended Visualization Functions

extension PINNVisualizer {
    
    /// Create comprehensive visualization report
    static func generateVisualizationReport(pinn: PINN,
                                          rk4Solution: [[Double]]?,
                                          losses: [Double],
                                          xRange: (Double, Double),
                                          tRange: (Double, Double)) -> String {
        
        var report = """
        
        🎨 COMPREHENSIVE PINN VISUALIZATION REPORT
        ==========================================
        
        📊 Domain Information:
        • Spatial domain: x ∈ [\(xRange.0), \(xRange.1)]
        • Temporal domain: t ∈ [\(tRange.0), \(tRange.1)]
        • PDE: Burgers' equation (u_t + u*u_x = 0)
        • Initial condition: u(x,0) = -sin(πx)
        
        """
        
        // Training loss visualization
        if !losses.isEmpty {
            report += "\n📉 Training Loss Evolution:\n"
            report += visualizeTrainingLoss(losses: losses)
            report += "\n"
        }
        
        // Hybrid framework metrics
        report += "\n🔬 Hybrid Framework Metrics:\n"
        report += visualizeHybridMetrics(pinn: pinn, xRange: xRange, tRange: tRange)
        report += "\n"
        
        // Solution comparison at different times
        let testTimes = [0.1, 0.3, 0.5]
        
        for testTime in testTimes {
            if let rk4 = rk4Solution {
                let comparison = compareSolutions(pinn: pinn, 
                                                rk4Solution: rk4,
                                                time: testTime,
                                                xRange: xRange,
                                                tRange: tRange)
                
                report += "\n📈 Solution Comparison at t = \(testTime):\n"
                report += generateASCIIPlot(series: comparison.pinnSeries, width: 60, height: 12)
                report += "\n"
                report += generateASCIIPlot(series: comparison.rk4Series, width: 60, height: 12)
                report += "\n"
                report += generateASCIIPlot(series: comparison.errorSeries, width: 60, height: 12)
                report += "\n"
            }
        }
        
        // Export instructions
        report += """
        
        📁 Data Export Instructions:
        ============================================
        
        To create publication-quality plots, use the exported CSV data with:
        
        • Python/Matplotlib:
          ```python
          import pandas as pd
          import matplotlib.pyplot as plt
          
          data = pd.read_csv('pinn_solution.csv')
          plt.plot(data['x'], data['PINN'], '--', label='PINN')
          plt.plot(data['x'], data['RK4'], '-', label='RK4')
          plt.legend()
          plt.show()
          ```
        
        • R:
          ```r
          data <- read.csv('pinn_solution.csv')
          plot(data$x, data$PINN, type='l', lty=2, col='blue')
          lines(data$x, data$RK4, type='l', lty=1, col='red')
          ```
        
        • Julia:
          ```julia
          using CSV, Plots
          data = CSV.read("pinn_solution.csv", DataFrame)
          plot(data.x, data.PINN, label="PINN", linestyle=:dash)
          plot!(data.x, data.RK4, label="RK4", linestyle=:solid)
          ```
        
        """
        
        return report
    }
}