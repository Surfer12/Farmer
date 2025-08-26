// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import SwiftUI

/// Main application entry point for the Inverse HB Model demo
@main
struct ReverseHierarchicalBayesianDemoApp: App {

    var body: some Scene {
        WindowGroup {
            InverseHBModelView()
                .environmentObject(ReverseHierarchicalBayesianViewModel())
        }
    }
}

/// Main view for the Inverse Hierarchical Bayesian Model demo
struct InverseHBModelView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel

    var body: some View {
        TabView {
            // Data Management Tab
            DataManagementView()
                .tabItem {
                    Label("Data", systemImage: "list.bullet")
                }

            // Parameter Recovery Tab
            ParameterRecoveryView()
                .tabItem {
                    Label("Parameters", systemImage: "function")
                }

            // Structure Learning Tab
            StructureLearningView()
                .tabItem {
                    Label("Structure", systemImage: "tree")
                }

            // Validation Tab
            ValidationView()
                .tabItem {
                    Label("Validation", systemImage: "checkmark.seal")
                }

            // Settings Tab
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
        .navigationTitle("Inverse HB Model")
    }
}

/// Data management view for adding and viewing observations
struct DataManagementView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel
    @State private var showingAddObservation = false
    @State private var sampleCount = 20

    var body: some View {
        NavigationView {
            VStack {
                // Header with stats
                HStack {
                    VStack(alignment: .leading) {
                        Text("Observations: \(viewModel.observations.count)")
                            .font(.headline)
                        if viewModel.hasResults {
                            Text("Results Available")
                                .font(.subheadline)
                                .foregroundColor(.green)
                        }
                    }
                    Spacer()

                    // Generate sample data button
                    Button(action: {
                        viewModel.generateSampleData(count: sampleCount)
                    }) {
                        Label("Generate Sample", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.bordered)
                }
                .padding()

                // Sample count picker
                HStack {
                    Text("Sample Count:")
                    Picker("", selection: $sampleCount) {
                        ForEach([10, 20, 50, 100], id: \.self) { count in
                            Text("\(count)").tag(count)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 150)
                }
                .padding(.horizontal)

                // Observations list
                if viewModel.observations.isEmpty {
                    VStack {
                        Spacer()
                        Text("No observations yet")
                            .foregroundColor(.secondary)
                        Text("Add observations or generate sample data")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                } else {
                    List {
                        ForEach(viewModel.observations.indices, id: \.self) { index in
                            ObservationRow(observation: viewModel.observations[index], index: index)
                        }
                        .onDelete { indices in
                            for index in indices {
                                viewModel.removeObservation(at: index)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Data Management")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        showingAddObservation = true
                    }) {
                        Image(systemName: "plus")
                    }
                }

                ToolbarItem(placement: .navigationBarLeading) {
                    if !viewModel.observations.isEmpty {
                        Button(action: viewModel.clearAll) {
                            Image(systemName: "trash")
                        }
                        .foregroundColor(.red)
                    }
                }
            }
            .sheet(isPresented: $showingAddObservation) {
                AddObservationView(isPresented: $showingAddObservation)
            }
        }
    }
}

/// Row view for displaying an observation
struct ObservationRow: View {
    let observation: ReverseHierarchicalBayesianModel.Observation
    let index: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Observation \(index + 1)")
                    .font(.headline)
                Spacer()
                Text(String(format: "Ψ: %.3f", observation.observedPsi))
                    .font(.caption)
                    .foregroundColor(.secondary)
                Image(systemName: observation.verificationOutcome ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(observation.verificationOutcome ? .green : .red)
            }

            Text("ID: \(observation.claim.id)")
                .font(.caption)
                .foregroundColor(.secondary)

            HStack {
                Text(String(format: "Authenticity: %.2f", observation.claim.riskAuthenticity))
                Text(String(format: "Virality: %.2f", observation.claim.riskVirality))
                Text(String(format: "P(H|E): %.2f", observation.claim.probabilityHgivenE))
            }
            .font(.caption2)
            .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

/// View for adding new observations
struct AddObservationView: View {

    @Binding var isPresented: Bool
    @State private var claimId = ""
    @State private var isVerified = false
    @State private var riskAuthenticity = 0.5
    @State private var riskVirality = 0.5
    @State private var probabilityHgivenE = 0.5
    @State private var observedPsi = 0.5
    @State private var verificationOutcome = false

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Claim Data")) {
                    TextField("Claim ID", text: $claimId)
                    Toggle("Verified True", isOn: $isVerified)

                    VStack {
                        HStack {
                            Text("Authenticity Risk")
                            Spacer()
                            Text(String(format: "%.2f", riskAuthenticity))
                        }
                        Slider(value: $riskAuthenticity, in: 0...1)
                    }

                    VStack {
                        HStack {
                            Text("Virality Risk")
                            Spacer()
                            Text(String(format: "%.2f", riskVirality))
                        }
                        Slider(value: $riskVirality, in: 0...1)
                    }

                    VStack {
                        HStack {
                            Text("P(H|E)")
                            Spacer()
                            Text(String(format: "%.2f", probabilityHgivenE))
                        }
                        Slider(value: $probabilityHgivenE, in: 0...1)
                    }
                }

                Section(header: Text("Observation")) {
                    VStack {
                        HStack {
                            Text("Observed Ψ")
                            Spacer()
                            Text(String(format: "%.3f", observedPsi))
                        }
                        Slider(value: $observedPsi, in: 0...1)
                    }

                    Toggle("Verification Outcome", isOn: $verificationOutcome)
                }
            }
            .navigationTitle("Add Observation")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add") {
                        let claim = ReverseHierarchicalBayesianModel.ClaimData(
                            id: claimId.isEmpty ? UUID().uuidString : claimId,
                            isVerifiedTrue: isVerified,
                            riskAuthenticity: riskAuthenticity,
                            riskVirality: riskVirality,
                            probabilityHgivenE: probabilityHgivenE
                        )

                        viewModel.addObservation(
                            claim: claim,
                            observedPsi: observedPsi,
                            verificationOutcome: verificationOutcome
                        )

                        isPresented = false
                    }
                    .disabled(claimId.isEmpty)
                }
            }
        }
    }
}

/// Parameter recovery view
struct ParameterRecoveryView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel

    var body: some View {
        VStack {
            // Control buttons
            HStack {
                Button(action: viewModel.recoverParameters) {
                    Label("Recover Parameters", systemImage: "function")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canRecoverParameters)

                if viewModel.isProcessing {
                    ProgressView(value: viewModel.processingProgress) {
                        Text(viewModel.currentOperation)
                    }
                    .progressViewStyle(.linear)
                }
            }
            .padding()

            // Results display
            if let result = viewModel.inverseResult {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Recovered Parameters")
                            .font(.title2)
                            .bold()

                        ParameterCard(title: "S (Internal Signal)", value: result.recoveredParameters.S)
                        ParameterCard(title: "N (Canonical Evidence)", value: result.recoveredParameters.N)
                        ParameterCard(title: "α (Evidence Allocation)", value: result.recoveredParameters.alpha)
                        ParameterCard(title: "β (Uplift Factor)", value: result.recoveredParameters.beta)

                        Divider()

                        VStack(alignment: .leading) {
                            Text("Analysis Results")
                                .font(.headline)

                            HStack {
                                Text("Confidence:")
                                Text(String(format: "%.2f%%", result.confidence * 100))
                                    .foregroundColor(result.confidence > 0.8 ? .green : .orange)
                            }

                            Text("Processing Time: \(String(format: "%.2f", result.processingTime))s")
                            Text("Log Evidence: \(String(format: "%.4f", result.logEvidence))")

                            Text("Parameter Uncertainties:")
                                .font(.headline)
                                .padding(.top)

                            ForEach(result.parameterUncertainties.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                                HStack {
                                    Text(key)
                                    Spacer()
                                    Text(String(format: "%.4f", value))
                                }
                            }
                        }

                        Divider()

                        VStack(alignment: .leading) {
                            Text("Posterior Samples")
                                .font(.headline)

                            Text("\(result.posteriorSamples.count) samples generated")
                                .foregroundColor(.secondary)

                            // Simple histogram-like visualization
                            Text("Sample Statistics:")
                                .font(.subheadline)
                                .padding(.top, 4)

                            let sValues = result.posteriorSamples.map { $0.S }
                            let nValues = result.posteriorSamples.map { $0.N }
                            let alphaValues = result.posteriorSamples.map { $0.alpha }
                            let betaValues = result.posteriorSamples.map { $0.beta }

                            SampleStatsView(title: "S", values: sValues)
                            SampleStatsView(title: "N", values: nValues)
                            SampleStatsView(title: "α", values: alphaValues)
                            SampleStatsView(title: "β", values: betaValues)
                        }
                    }
                    .padding()
                }
            } else if !viewModel.isProcessing {
                VStack {
                    Spacer()
                    Text("No parameter recovery results")
                        .foregroundColor(.secondary)
                    Text("Add observations and run parameter recovery")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .navigationTitle("Parameter Recovery")
    }
}

/// Card for displaying parameter values
struct ParameterCard: View {
    let title: String
    let value: Double

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(String(format: "%.4f", value))
                .font(.system(.body, design: .monospaced))
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

/// View for displaying sample statistics
struct SampleStatsView: View {
    let title: String
    let values: [Double]

    var body: some View {
        let mean = values.reduce(0, +) / Double(values.count)
        let std = sqrt(values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count))

        HStack {
            Text(title)
            Spacer()
            Text(String(format: "μ=%.3f, σ=%.3f", mean, std))
                .font(.system(.caption, design: .monospaced))
        }
    }
}

/// Structure learning view
struct StructureLearningView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel

    var body: some View {
        VStack {
            // Control buttons
            HStack {
                Button(action: viewModel.learnStructure) {
                    Label("Learn Structure", systemImage: "tree")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canLearnStructure)

                if viewModel.isProcessing {
                    ProgressView(value: viewModel.processingProgress) {
                        Text(viewModel.currentOperation)
                    }
                    .progressViewStyle(.linear)
                }
            }
            .padding()

            // Results display
            if let result = viewModel.structureResult {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Learned Structure")
                            .font(.title2)
                            .bold()

                        VStack(alignment: .leading) {
                            Text("Hierarchy Levels")
                                .font(.headline)

                            ForEach(result.learnedStructure.levels, id: \.self) { level in
                                HStack {
                                    Text("• \(level)")
                                    Spacer()
                                    if let weight = result.learnedStructure.levelWeights[level] {
                                        Text(String(format: "%.2f", weight))
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                        }

                        Divider()

                        VStack(alignment: .leading) {
                            Text("Inferred Relationships")
                                .font(.headline)

                            ForEach(result.inferredRelationships, id: \.self) { relationship in
                                Text("• \(relationship)")
                            }
                        }

                        Divider()

                        VStack(alignment: .leading) {
                            Text("Relationship Strengths")
                                .font(.headline)

                            ForEach(result.relationshipStrengths.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                                HStack {
                                    Text(key)
                                    Spacer()
                                    Text(String(format: "%.2f", value))
                                        .foregroundColor(.secondary)
                                }
                            }
                        }

                        Divider()

                        HStack {
                            Text("Structure Confidence:")
                            Text(String(format: "%.2f%%", result.structureConfidence * 100))
                                .foregroundColor(result.structureConfidence > 0.8 ? .green : .orange)
                        }
                    }
                    .padding()
                }
            } else if !viewModel.isProcessing {
                VStack {
                    Spacer()
                    Text("No structure learning results")
                        .foregroundColor(.secondary)
                    Text("Add observations and run structure learning")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .navigationTitle("Structure Learning")
    }
}

/// Validation view
struct ValidationView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel
    @State private var groundTruth = ReverseHierarchicalBayesianModel.ModelParameters.defaultParams

    var body: some View {
        VStack {
            // Ground truth parameters input
            VStack(alignment: .leading) {
                Text("Ground Truth Parameters")
                    .font(.headline)
                    .padding(.bottom, 4)

                ParameterInputRow(title: "S", value: $groundTruth.S)
                ParameterInputRow(title: "N", value: $groundTruth.N)
                ParameterInputRow(title: "α", value: $groundTruth.alpha)
                ParameterInputRow(title: "β", value: $groundTruth.beta, minValue: 1.0)
            }
            .padding()

            // Validation button
            Button(action: {
                viewModel.validateAgainst(groundTruth: groundTruth)
            }) {
                Label("Validate Results", systemImage: "checkmark.seal")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!viewModel.canValidate)
            .padding()

            // Results display
            if let result = viewModel.validationResult {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Validation Results")
                            .font(.title2)
                            .bold()

                        VStack(alignment: .leading) {
                            Text("Overall Performance")
                                .font(.headline)

                            HStack {
                                Text("Overall Score:")
                                Text(String(format: "%.2f%%", result.overallScore * 100))
                                    .foregroundColor(result.overallScore > 0.8 ? .green : .orange)
                            }

                            HStack {
                                Text("Parameter Recovery Error:")
                                Text(String(format: "%.4f", result.parameterRecoveryError))
                                    .foregroundColor(result.parameterRecoveryError < 0.1 ? .green : .red)
                            }

                            HStack {
                                Text("Confidence Accuracy:")
                                Text(String(format: "%.2f%%", result.confidenceAccuracy * 100))
                                    .foregroundColor(result.confidenceAccuracy > 0.8 ? .green : .orange)
                            }
                        }

                        Divider()

                        VStack(alignment: .leading) {
                            Text("Parameter Errors")
                                .font(.headline)

                            ForEach(result.parameterErrors.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                                HStack {
                                    Text(key)
                                    Spacer()
                                    Text(String(format: "%.4f", value))
                                        .foregroundColor(value < 0.1 ? .green : .red)
                                }
                            }
                        }
                    }
                    .padding()
                }
            } else if viewModel.inverseResult != nil {
                VStack {
                    Spacer()
                    Text("Ready for validation")
                        .foregroundColor(.secondary)
                    Text("Set ground truth parameters and validate")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            } else {
                VStack {
                    Spacer()
                    Text("No results to validate")
                        .foregroundColor(.secondary)
                    Text("Run parameter recovery first")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .navigationTitle("Validation")
    }
}

/// Row for parameter input
struct ParameterInputRow: View {
    let title: String
    @Binding var value: Double
    var minValue: Double = 0.0
    var maxValue: Double = 1.0

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(String(format: "%.3f", value))
                .font(.system(.body, design: .monospaced))
            Slider(value: $value, in: minValue...maxValue)
                .frame(width: 100)
        }
    }
}

/// Settings view for configuration
struct SettingsView: View {

    @EnvironmentObject var viewModel: ReverseHierarchicalBayesianViewModel
    @State private var showingExport = false

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Algorithm Configuration")) {
                    Stepper("Max Iterations: \(viewModel.config.maxIterations)",
                           value: Binding(
                               get: { viewModel.config.maxIterations },
                               set: { viewModel.config = ReverseHierarchicalBayesianModel.Configuration(
                                   maxIterations: $0,
                                   tolerance: viewModel.config.tolerance,
                                   populationSize: viewModel.config.populationSize,
                                   mutationRate: viewModel.config.mutationRate,
                                   crossoverRate: viewModel.config.crossoverRate,
                                   useParallel: viewModel.config.useParallel,
                                   parallelThreshold: viewModel.config.parallelThreshold
                               )}
                           ),
                           in: 100...5000, step: 100)

                    VStack {
                        HStack {
                            Text("Tolerance")
                            Spacer()
                            Text(String(format: "%.2e", viewModel.config.tolerance))
                        }
                        Slider(
                            value: Binding(
                                get: { viewModel.config.tolerance },
                                set: { viewModel.config = ReverseHierarchicalBayesianModel.Configuration(
                                    maxIterations: viewModel.config.maxIterations,
                                    tolerance: $0,
                                    populationSize: viewModel.config.populationSize,
                                    mutationRate: viewModel.config.mutationRate,
                                    crossoverRate: viewModel.config.crossoverRate,
                                    useParallel: viewModel.config.useParallel,
                                    parallelThreshold: viewModel.config.parallelThreshold
                                )}
                            ),
                            in: 1e-8...1e-4
                        )
                    }

                    Stepper("Population Size: \(viewModel.config.populationSize)",
                           value: Binding(
                               get: { viewModel.config.populationSize },
                               set: { viewModel.config = ReverseHierarchicalBayesianModel.Configuration(
                                   maxIterations: viewModel.config.maxIterations,
                                   tolerance: viewModel.config.tolerance,
                                   populationSize: $0,
                                   mutationRate: viewModel.config.mutationRate,
                                   crossoverRate: viewModel.config.crossoverRate,
                                   useParallel: viewModel.config.useParallel,
                                   parallelThreshold: viewModel.config.parallelThreshold
                               )}
                           ),
                           in: 50...500, step: 50)
                }

                Section(header: Text("Export & Actions")) {
                    Button(action: {
                        if let data = viewModel.exportResults() {
                            // In a real app, you'd save this to a file
                            print("Exported \(data.count) bytes of JSON data")
                            showingExport = true
                        }
                    }) {
                        Label("Export Results", systemImage: "square.and.arrow.up")
                    }
                    .disabled(!viewModel.hasResults)

                    Button(action: viewModel.clearAll) {
                        Label("Clear All Data", systemImage: "trash")
                            .foregroundColor(.red)
                    }
                    .disabled(!viewModel.hasObservations)
                }

                Section(header: Text("About")) {
                    VStack(alignment: .leading) {
                        Text("Inverse Hierarchical Bayesian Model")
                            .font(.headline)

                        Text("This app demonstrates parameter recovery and structure learning from observed Ψ scores using evolutionary optimization.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)

                        HStack {
                            Text("Version:")
                            Text("1.0.0")
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 4)
                    }
                }
            }
            .navigationTitle("Settings")
            .alert("Export Complete", isPresented: $showingExport) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("Results have been exported to the console.")
            }
        }
    }
}
