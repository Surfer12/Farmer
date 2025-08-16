import SwiftUI
import UniformTypeIdentifiers

struct DataExportView: View {
    @ObservedObject var viewModel: FinViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var exportFormat: ExportFormat = .json
    @State private var showingShareSheet = false
    @State private var exportData: String = ""
    @State private var exportFileName: String = ""
    
    enum ExportFormat: String, CaseIterable {
        case json = "JSON"
        case csv = "CSV"
        case text = "Plain Text"
        
        var fileExtension: String {
            switch self {
            case .json: return "json"
            case .csv: return "csv"
            case .text: return "txt"
            }
        }
        
        var mimeType: String {
            switch self {
            case .json: return "application/json"
            case .csv: return "text/csv"
            case .text: return "text/plain"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Export Options
                    exportOptionsSection
                    
                    // Data Preview
                    dataPreviewSection
                    
                    // Export Actions
                    exportActionsSection
                    
                    Spacer(minLength: 20)
                }
                .padding(.horizontal)
            }
            .navigationTitle("Data Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .onAppear {
            generateExportData()
        }
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(activityItems: [exportData], applicationActivities: nil)
        }
    }
    
    // MARK: - Export Options Section
    
    private var exportOptionsSection: some View {
        VStack(spacing: 16) {
            Text("Export Options")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                HStack {
                    Text("Export Format:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Picker("Format", selection: $exportFormat) {
                        ForEach(ExportFormat.allCases, id: \.self) { format in
                            Text(format.rawValue).tag(format)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: exportFormat) { _ in
                        generateExportData()
                    }
                }
                
                HStack {
                    Text("File Extension:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(".\(exportFormat.fileExtension)")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.blue)
                }
                
                HStack {
                    Text("MIME Type:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(exportFormat.mimeType)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
        }
    }
    
    // MARK: - Data Preview Section
    
    private var dataPreviewSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Data Preview")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                Button("Refresh") {
                    generateExportData()
                }
                .font(.subheadline)
                .foregroundColor(.blue)
            }
            
            ScrollView {
                Text(exportData)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            }
            .frame(maxHeight: 300)
        }
    }
    
    // MARK: - Export Actions Section
    
    private var exportActionsSection: some View {
        VStack(spacing: 16) {
            Text("Export Actions")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 12) {
                Button(action: {
                    showingShareSheet = true
                }) {
                    HStack {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(.white)
                        Text("Share Data")
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
                }
                
                Button(action: {
                    copyToClipboard()
                }) {
                    HStack {
                        Image(systemName: "doc.on.clipboard")
                            .foregroundColor(.white)
                        Text("Copy to Clipboard")
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .cornerRadius(10)
                }
                
                Button(action: {
                    saveToFiles()
                }) {
                    HStack {
                        Image(systemName: "folder")
                            .foregroundColor(.white)
                        Text("Save to Files")
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .cornerRadius(10)
                }
            }
        }
    }
    
    // MARK: - Data Generation
    
    private func generateExportData() {
        let timestamp = Date().ISO8601String()
        exportFileName = "VectorFinCFD_\(timestamp).\(exportFormat.fileExtension)"
        
        switch exportFormat {
        case .json:
            exportData = generateJSONExport()
        case .csv:
            exportData = generateCSVExport()
        case .text:
            exportData = generateTextExport()
        }
    }
    
    private func generateJSONExport() -> String {
        let metrics = viewModel.getPerformanceMetrics()
        let data: [String: Any] = [
            "metadata": [
                "app": "Vector 3/2 Fin CFD Visualizer",
                "version": "1.0.0",
                "export_timestamp": Date().ISO8601String(),
                "export_format": "JSON"
            ],
            "session_data": [
                "turn_angle": metrics.turnAngle,
                "lift_force": metrics.lift,
                "drag_force": metrics.drag,
                "lift_drag_ratio": metrics.liftToDragRatio,
                "flow_state": metrics.flowState.rawValue,
                "prediction_confidence": metrics.confidence
            ],
            "cognitive_data": [
                "hrv": viewModel.hrv ?? 0,
                "cognitive_load": metrics.cognitiveLoad,
                "pressure_average": metrics.averagePressure,
                "pressure_range": metrics.pressureRange
            ],
            "performance_metrics": [
                "efficiency_score": metrics.efficiency,
                "last_update": viewModel.lastUpdateTime.ISO8601String(),
                "monitoring_active": viewModel.isMonitoring
            ],
            "pressure_data": viewModel.pressureData.map { ["value": $0] }
        ]
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: data, options: .prettyPrinted)
            return String(data: jsonData, encoding: .utf8) ?? "Error generating JSON"
        } catch {
            return "Error generating JSON: \(error.localizedDescription)"
        }
    }
    
    private func generateCSVExport() -> String {
        let metrics = viewModel.getPerformanceMetrics()
        
        var csv = "Metric,Value,Unit,Description\n"
        csv += "Turn Angle,\(metrics.turnAngle),degrees,Current angle of attack\n"
        csv += "Lift Force,\(metrics.lift),N,Vertical force generated by fins\n"
        csv += "Drag Force,\(metrics.drag),N,Resistance force against water\n"
        csv += "Lift/Drag Ratio,\(metrics.liftToDragRatio),,Aerodynamic efficiency metric\n"
        csv += "Flow State,\(metrics.flowState.rawValue),,Current performance state\n"
        csv += "Prediction Confidence,\(String(format: "%.1f", metrics.confidence * 100)),%,Model confidence level\n"
        csv += "HRV,\(viewModel.hrv ?? 0),ms,Heart rate variability\n"
        csv += "Cognitive Load,\(String(format: "%.3f", metrics.cognitiveLoad)),,Mental effort score\n"
        csv += "Pressure Average,\(String(format: "%.3f", metrics.averagePressure)),,Average pressure across fins\n"
        csv += "Pressure Range,\(String(format: "%.3f", metrics.pressureRange)),,Pressure variation\n"
        csv += "Efficiency Score,\(String(format: "%.1f", metrics.efficiency * 100)),%,Overall performance efficiency\n"
        csv += "Export Timestamp,\(Date().ISO8601String()),,Data export time\n"
        
        // Add pressure data points
        csv += "\nPressure Data Points\n"
        csv += "Index,Value\n"
        for (index, pressure) in viewModel.pressureData.enumerated() {
            csv += "\(index),\(pressure)\n"
        }
        
        return csv
    }
    
    private func generateTextExport() -> String {
        let metrics = viewModel.getPerformanceMetrics()
        
        var text = """
        Vector 3/2 Fin CFD Visualizer - Session Data Export
        ===================================================
        
        EXPORT INFORMATION
        ------------------
        Export Format: \(exportFormat.rawValue)
        Export Time: \(Date().ISO8601String())
        App Version: 1.0.0
        
        SESSION PERFORMANCE
        -------------------
        Turn Angle: \(metrics.turnAngle)°
        Lift Force: \(metrics.lift) N
        Drag Force: \(metrics.drag) N
        Lift/Drag Ratio: \(String(format: "%.2f", metrics.liftToDragRatio))
        Flow State: \(metrics.flowState.rawValue)
        Prediction Confidence: \(String(format: "%.1f", metrics.confidence * 100))%
        
        COGNITIVE METRICS
        -----------------
        HRV: \(viewModel.hrv?.description ?? "N/A") ms
        Cognitive Load: \(String(format: "%.3f", metrics.cognitiveLoad))
        Pressure Average: \(String(format: "%.3f", metrics.averagePressure))
        Pressure Range: \(String(format: "%.3f", metrics.pressureRange))
        
        PERFORMANCE ANALYSIS
        --------------------
        Efficiency Score: \(String(format: "%.1f", metrics.efficiency * 100))%
        Last Update: \(viewModel.lastUpdateTime.ISO8601String())
        Monitoring Status: \(viewModel.isMonitoring ? "Active" : "Inactive")
        
        PRESSURE DATA POINTS
        --------------------
        Total Points: \(viewModel.pressureData.count)
        
        """
        
        // Add pressure data
        for (index, pressure) in viewModel.pressureData.enumerated() {
            text += "Point \(index): \(String(format: "%.3f", pressure))\n"
        }
        
        text += "\nEND OF EXPORT"
        return text
    }
    
    // MARK: - Export Actions
    
    private func copyToClipboard() {
        UIPasteboard.general.string = exportData
        // You could add a success message here
    }
    
    private func saveToFiles() {
        // This would typically use UIDocumentPickerViewController
        // For now, we'll just show the share sheet
        showingShareSheet = true
    }
}

// MARK: - Supporting Views

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]?
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: applicationActivities
        )
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct DataExportView_Previews: PreviewProvider {
    static var previews: some View {
        DataExportView(viewModel: FinViewModel())
    }
}