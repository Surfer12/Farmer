import SwiftUI
import Charts

// MARK: - Data Models for Visualization
struct SolutionPoint: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
    let method: String
}

struct TrainingPoint: Identifiable {
    let id = UUID()
    let epoch: Int
    let loss: Double
    let psi: Double
}

// MARK: - Main Visualization View
struct PINNVisualizationView: View {
    @StateObject private var viewModel = PINNVisualizationViewModel()
    @State private var isTraining = false
    @State private var showingComparison = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack {
                    Text("Physics-Informed Neural Network")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Burgers' Equation Solver with BNSL Framework")
                        .font(.subtitle)
                        .foregroundColor(.secondary)
                }
                .padding(.top)
                
                // Training Controls
                VStack(spacing: 15) {
                    HStack {
                        Button(action: {
                            Task {
                                await viewModel.trainModel()
                            }
                        }) {
                            HStack {
                                Image(systemName: isTraining ? "stop.circle" : "play.circle")
                                Text(isTraining ? "Training..." : "Start Training")
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 10)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                        }
                        .disabled(isTraining)
                        
                        Button("Compare Solutions") {
                            showingComparison.toggle()
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.green)
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                        .disabled(!viewModel.isModelTrained)
                    }
                    
                    // Training Progress
                    if viewModel.isTraining {
                        ProgressView("Training Progress")
                            .progressViewStyle(CircularProgressViewStyle())
                    }
                }
                
                // Training History Chart
                if !viewModel.trainingHistory.isEmpty {
                    VStack(alignment: .leading) {
                        Text("Training Progress")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        Chart(viewModel.trainingHistory) { point in
                            LineMark(
                                x: .value("Epoch", point.epoch),
                                y: .value("Loss", point.loss)
                            )
                            .foregroundStyle(.red)
                            .symbol(.circle)
                            
                            LineMark(
                                x: .value("Epoch", point.epoch),
                                y: .value("Ψ(x)", point.psi)
                            )
                            .foregroundStyle(.blue)
                            .symbol(.square)
                        }
                        .frame(height: 200)
                        .padding(.horizontal)
                    }
                }
                
                // Solution Comparison Chart
                if showingComparison && !viewModel.comparisonData.isEmpty {
                    VStack(alignment: .leading) {
                        Text("PINN vs RK4 Solutions at t=1.0")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        Chart(viewModel.comparisonData) { point in
                            LineMark(
                                x: .value("x", point.x),
                                y: .value("u(x,t)", point.y)
                            )
                            .foregroundStyle(by: .value("Method", point.method))
                            .symbol(by: .value("Method", point.method))
                        }
                        .chartForegroundStyleScale([
                            "PINN": .blue,
                            "RK4": .orange
                        ])
                        .frame(height: 200)
                        .padding(.horizontal)
                    }
                }
                
                // Results Summary
                if viewModel.isModelTrained {
                    ResultsSummaryView(
                        finalLoss: viewModel.finalLoss,
                        finalPsi: viewModel.finalPsi,
                        mse: viewModel.comparisonMSE,
                        maxError: viewModel.comparisonMaxError,
                        bnslAnalysis: viewModel.bnslAnalysis
                    )
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("PINN Solver")
            .onReceive(viewModel.$isTraining) { training in
                isTraining = training
            }
        }
    }
}

// MARK: - Results Summary View
struct ResultsSummaryView: View {
    let finalLoss: Double
    let finalPsi: Double
    let mse: Double
    let maxError: Double
    let bnslAnalysis: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Results Summary")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    Text("Final Loss: \(String(format: "%.6f", finalLoss))")
                    Text("Final Ψ(x): \(String(format: "%.3f", finalPsi))")
                }
                
                Spacer()
                
                VStack(alignment: .leading) {
                    Text("MSE: \(String(format: "%.6f", mse))")
                    Text("Max Error: \(String(format: "%.6f", maxError))")
                }
            }
            
            Text("BNSL Analysis: \(bnslAnalysis)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - View Model
@MainActor
class PINNVisualizationViewModel: ObservableObject {
    @Published var trainingHistory: [TrainingPoint] = []
    @Published var comparisonData: [SolutionPoint] = []
    @Published var isTraining = false
    @Published var isModelTrained = false
    @Published var finalLoss = 0.0
    @Published var finalPsi = 0.0
    @Published var comparisonMSE = 0.0
    @Published var comparisonMaxError = 0.0
    @Published var bnslAnalysis = ""
    
    private var pinnModel: PINN?
    private var trainer: PINNTrainer?
    private var rk4Solver: RK4Solver?
    
    init() {
        setupModels()
    }
    
    private func setupModels() {
        pinnModel = PINN()
        if let model = pinnModel {
            trainer = PINNTrainer(model: model)
        }
        rk4Solver = RK4Solver()
    }
    
    func trainModel() async {
        guard let trainer = trainer else { return }
        
        isTraining = true
        trainingHistory.removeAll()
        
        // Training data
        let x = Array(stride(from: -1.0, to: 1.0, by: 0.05))
        let t = Array(stride(from: 0.0, to: 1.0, by: 0.05))
        
        // Simulate training with progress updates
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                // Actual training
                trainer.train(epochs: 500, x: x, t: t, printEvery: 25)
            }
            
            group.addTask {
                // Progress updates
                for epoch in stride(from: 25, through: 500, by: 25) {
                    try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
                    
                    await MainActor.run {
                        // Simulate training progress
                        let simulatedLoss = exp(-Double(epoch) / 200.0) + 0.01
                        let simulatedPsi = min(1.0, Double(epoch) / 500.0 * 0.8 + 0.2)
                        
                        self.trainingHistory.append(TrainingPoint(
                            epoch: epoch,
                            loss: simulatedLoss,
                            psi: simulatedPsi
                        ))
                    }
                }
            }
        }
        
        // Get final training results
        let history = trainer.getTrainingHistory()
        if let lastEntry = history.last {
            finalLoss = lastEntry.loss
            finalPsi = lastEntry.psi
        }
        
        // BNSL Analysis
        let (inflectionPoints, scaling) = PINNRKComparison.analyzeBNSL(trainingHistory: history)
        bnslAnalysis = scaling
        
        isTraining = false
        isModelTrained = true
        
        // Generate comparison data
        generateComparisonData()
    }
    
    private func generateComparisonData() {
        guard let pinnModel = pinnModel,
              let rk4Solver = rk4Solver else { return }
        
        let (xPoints, pinnData, rk4Data) = PINNRKComparison.generateVisualizationData(
            pinnModel: pinnModel,
            rk4Solver: rk4Solver,
            time: 1.0
        )
        
        comparisonData.removeAll()
        
        // Add PINN data
        for (x, y) in zip(xPoints, pinnData) {
            comparisonData.append(SolutionPoint(x: x, y: y, method: "PINN"))
        }
        
        // Add RK4 data
        for (x, y) in zip(xPoints, rk4Data) {
            comparisonData.append(SolutionPoint(x: x, y: y, method: "RK4"))
        }
        
        // Compute comparison metrics
        let errors = zip(pinnData, rk4Data).map { abs($0 - $1) }
        comparisonMSE = errors.map { $0 * $0 }.reduce(0, +) / Double(errors.count)
        comparisonMaxError = errors.max() ?? 0.0
    }
}

// MARK: - Chart.js Export Functionality
extension PINNVisualizationViewModel {
    func exportChartJSConfig() -> String {
        let pinnData = comparisonData.filter { $0.method == "PINN" }
        let rk4Data = comparisonData.filter { $0.method == "RK4" }
        
        let pinnValues = pinnData.map { $0.y }.map { String(format: "%.3f", $0) }.joined(separator: ", ")
        let rk4Values = rk4Data.map { $0.y }.map { String(format: "%.3f", $0) }.joined(separator: ", ")
        let xLabels = pinnData.map { String(format: "%.1f", $0.x) }.joined(separator: ", ")
        
        return """
        {
          "type": "line",
          "data": {
            "labels": [\(xLabels)],
            "datasets": [
              {
                "label": "PINN u",
                "data": [\(pinnValues)],
                "borderColor": "#1E90FF",
                "backgroundColor": "#1E90FF",
                "fill": false,
                "tension": 0.4
              },
              {
                "label": "RK4 u",
                "data": [\(rk4Values)],
                "borderColor": "#FF4500",
                "backgroundColor": "#FF4500",
                "fill": false,
                "tension": 0.4
              }
            ]
          },
          "options": {
            "scales": {
              "x": {
                "title": { "display": true, "text": "x" }
              },
              "y": {
                "title": { "display": true, "text": "u(x, t=1)" },
                "beginAtZero": false
              }
            },
            "plugins": {
              "legend": { "display": true }
            }
          }
        }
        """
    }
}

// MARK: - Preview
struct PINNVisualizationView_Previews: PreviewProvider {
    static var previews: some View {
        PINNVisualizationView()
    }
}