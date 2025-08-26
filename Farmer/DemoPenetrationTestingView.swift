// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  DemoPenetrationTestingView.swift
//  Farmer
//
//  Created by Ryan David Oates on 8/26/25.
//  Demonstration of reverse koopman penetration testing

import SwiftUI
import Combine

struct DemoPenetrationTestingView: View {
    @StateObject private var securityManager = SecurityManager()
    @State private var selectedTest: PenetrationTestType = .iosGeneral
    @State private var isRunning = false
    @State private var testResults: [SecurityFinding] = []
    @State private var progress: Double = 0.0

    enum PenetrationTestType: String, CaseIterable {
        case iosGeneral = "iOS General Security"
        case itAssistant = "IT Assistant Analysis"
        case cfdSystem = "CFD System Analysis"
    }

    var body: some View {
        NavigationView {
            VStack {
                // Test Selection
                Picker("Test Type", selection: $selectedTest) {
                    ForEach(PenetrationTestType.allCases, id: \.self) { testType in
                        Text(testType.rawValue).tag(testType)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()

                // Progress Indicator
                if isRunning {
                    ProgressView(value: progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding()

                    Text("Running \(selectedTest.rawValue)...")
                        .font(.headline)
                }

                // Run Test Button
                Button(action: runSelectedTest) {
                    HStack {
                        Image(systemName: "shield.fill")
                        Text("Run Penetration Test")
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isRunning)
                .padding()

                // Results Summary
                if !testResults.isEmpty {
                    VStack(alignment: .leading) {
                        Text("Test Results Summary")
                            .font(.headline)
                            .padding(.bottom, 5)

                        HStack {
                            ForEach([SeverityLevel.critical, .high, .medium, .low], id: \.self) { severity in
                                let count = testResults.filter { $0.severity == severity }.count
                                Text("\(severity.rawValue): \(count)")
                                    .foregroundColor(severityColor(severity))
                                    .font(.caption)
                            }
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .padding()
                }

                // Results List
                List(testResults, id: \.id) { finding in
                    SecurityFindingRow(finding: finding)
                }
                .listStyle(PlainListStyle())

                Spacer()
            }
            .navigationTitle("Reverse Koopman Penetration Testing")
            .toolbar {
                Button(action: exportResults) {
                    Image(systemName: "square.and.arrow.up")
                }
            }
        }
    }

    private func runSelectedTest() {
        isRunning = true
        progress = 0.0
        testResults.removeAll()

        // Run test asynchronously
        DispatchQueue.global(qos: .userInitiated).async {
            let results = performTest(selectedTest)

            DispatchQueue.main.async {
                self.testResults = results
                self.isRunning = false
                self.progress = 1.0
            }
        }

        // Simulate progress updates
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            if self.isRunning {
                self.progress += 0.1
                if self.progress >= 1.0 {
                    timer.invalidate()
                }
            } else {
                timer.invalidate()
            }
        }
    }

    private func performTest(_ testType: PenetrationTestType) -> [SecurityFinding] {
        let reverseKoopman = ReverseKoopmanOperator()
        let ksValidator = KSPenetrationTestingValidator()

        switch testType {
        case .iosGeneral:
            let penetrationTester = iOSPenetrationTesting(
                koopmanOperator: reverseKoopman
            )
            return penetrationTester.performSecurityAssessment()

        case .itAssistant:
            let penetrationTester = iOSPenetrationTesting(
                koopmanOperator: reverseKoopman
            )
            return penetrationTester.testITAssistantSecurity()

        case .cfdSystem:
            let penetrationTester = CFDSystemPenetrationTesting(
                koopmanOperator: reverseKoopman,
                ksValidator: ksValidator
            )
            return penetrationTester.performCFDSecurityAssessment()
        }
    }

    private func severityColor(_ severity: SeverityLevel) -> Color {
        switch severity {
        case .critical: return .red
        case .high: return .orange
        case .medium: return .yellow
        case .low: return .green
        case .info: return .blue
        }
    }

    private func exportResults() {
        let exporter = SecurityReportExporter()
        let report = exporter.generateConsolidatedReport(
            iosFindings: selectedTest == .iosGeneral ? testResults : [],
            itAssistantFindings: selectedTest == .itAssistant ? testResults : [],
            cfdFindings: selectedTest == .cfdSystem ? testResults : []
        )

        // In a real app, this would save to file or share
        print("Exported report:\n\(report)")
    }
}

struct SecurityFindingRow: View {
    let finding: SecurityFinding

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(severityColor(finding.severity))
                    .frame(width: 12, height: 12)
                Text(finding.title)
                    .font(.headline)
                Spacer()
                Text(finding.severity.rawValue)
                    .font(.caption)
                    .padding(4)
                    .background(severityColor(finding.severity).opacity(0.2))
                    .cornerRadius(4)
            }

            Text(finding.description)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Text("Location: \(finding.location)")
                .font(.caption)
                .foregroundColor(.gray)

            if finding.koopmanStability > 0 {
                Text("Koopman Stability: \(String(format: "%.3f", finding.koopmanStability))")
                    .font(.caption)
                    .foregroundColor(.blue)
            }
        }
        .padding(.vertical, 8)
    }

    private func severityColor(_ severity: SeverityLevel) -> Color {
        switch severity {
        case .critical: return .red
        case .high: return .orange
        case .medium: return .yellow
        case .low: return .green
        case .info: return .blue
        }
    }
}

class SecurityReportExporter {
    func generateConsolidatedReport(
        iosFindings: [SecurityFinding],
        itAssistantFindings: [SecurityFinding],
        cfdFindings: [SecurityFinding]
    ) -> String {

        var report = """
        Reverse Koopman Penetration Testing Report
        ========================================

        Generated: \(Date())

        Executive Summary:
        """

        let totalFindings = iosFindings.count + itAssistantFindings.count + cfdFindings.count
        report += "\nTotal Security Findings: \(totalFindings)\n"

        // iOS General Findings
        if !iosFindings.isEmpty {
            report += "\n\niOS General Security Assessment:"
            report += generateSummarySection(findings: iosFindings)
        }

        // IT Assistant Findings
        if !itAssistantFindings.isEmpty {
            report += "\n\nIT Assistant Security Analysis:"
            report += generateSummarySection(findings: itAssistantFindings)
        }

        // CFD System Findings
        if !cfdFindings.isEmpty {
            report += "\n\nCFD System Security Analysis:"
            report += generateSummarySection(findings: cfdFindings)
        }

        report += "\n\nTechnical Methodology:"
        report += "\n- Reverse Koopman Operator Analysis for system behavior modeling"
        report += "\n- K-S Statistical Validation for result confidence"
        report += "\n- Spectral decomposition for anomaly detection"
        report += "\n- Progressive validation gates for quality assurance"

        return report
    }

    private func generateSummarySection(findings: [SecurityFinding]) -> String {
        var section = "\n  Total Findings: \(findings.count)"

        let severityCounts = Dictionary(grouping: findings) { $0.severity }
            .mapValues { $0.count }

        for severity in [SeverityLevel.critical, .high, .medium, .low, .info] {
            if let count = severityCounts[severity] {
                section += "\n  \(severity.rawValue): \(count)"
            }
        }

        // Add top critical findings
        let criticalFindings = findings.filter { $0.severity == .critical }
        if !criticalFindings.isEmpty {
            section += "\n\n  Critical Issues:"
            for finding in criticalFindings.prefix(3) {
                section += "\n  • \(finding.title)"
            }
        }

        return section
    }
}

// MARK: - Demo Extension

extension ReverseKoopmanOperator {
    /// Demo method to generate sample trajectory for testing
    func generateDemoTrajectory() -> [[Double]] {
        return generateVanDerPolTrajectory(nPoints: 100)
    }
}

extension KSPenetrationTestingValidator {
    /// Demo method to generate sample validation results
    func generateDemoValidation() -> KSValidationResult {
        let data = (0..<50).map { _ in Double.random(in: 0...1) }
        return oneSampleKSTest(data: data)
    }
}
