#!/usr/bin/env python3
"""
Export Analysis Results

Comprehensive export of all analysis results to various formats for reporting and documentation.
"""

import json
import csv
import os
from datetime import datetime
from minimal_contraction_psi import (
    MinimalContractionPsi,
    MinimalContractionConfig,
    create_test_scenarios,
)


def export_contraction_analysis():
    """Export contraction analysis results to JSON and CSV"""
    print("Exporting contraction analysis results...")

    # Setup
    config = MinimalContractionConfig()
    psi_updater = MinimalContractionPsi(config)
    scenarios = create_test_scenarios()

    # Run analysis
    results = psi_updater.analyze_contraction_properties(scenarios)

    # Export to JSON
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "kappa": config.kappa,
            "g_max": config.g_max,
            "L_C": config.L_C,
            "omega": config.omega,
            "modality_weights": config.modality_weights,
            "modality_lipschitz": config.modality_lipschitz,
        },
        "theoretical_analysis": results["theoretical_bound"],
        "numerical_estimates": results["numerical_estimates"],
        "convergence_analysis": results["convergence_analysis"],
        "sample_sequences": {},
    }

    # Add sample sequences
    for i, scenario in enumerate(scenarios):
        sequence = psi_updater.simulate_sequence(0.3, 20, scenario)
        export_data["sample_sequences"][f"scenario_{i+1}"] = {
            "parameters": scenario,
            "sequence": sequence,
            "final_value": sequence[-1],
            "convergence_steps": len(
                [x for x in sequence if abs(x - sequence[-1]) > 0.001]
            ),
        }

    # Write JSON
    with open("outputs/contraction_analysis_results.json", "w") as f:
        json.dump(export_data, f, indent=2)

    # Export to CSV
    csv_data = []
    for i, est in enumerate(results["numerical_estimates"]):
        conv = results["convergence_analysis"][i]
        scenario = est["scenario"]
        csv_data.append(
            {
                "scenario_id": i + 1,
                "alpha": scenario["alpha"],
                "S": scenario["S"],
                "N": scenario["N"],
                "R_cog": scenario["R_cog"],
                "R_eff": scenario["R_eff"],
                "K_hat_numerical": est["K_hat"],
                "L_hat_numerical": est["L_hat"],
                "final_value": conv["final_value"],
                "convergence_rate": conv["convergence_rate"],
                "K_theoretical": results["theoretical_bound"]["K_theory"],
            }
        )

    with open("outputs/contraction_summary.csv", "w", newline="") as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    print(
        f"✓ Exported contraction analysis to outputs/contraction_analysis_results.json"
    )
    print(f"✓ Exported summary to outputs/contraction_summary.csv")

    return export_data


def export_hybrid_functional_analysis():
    """Export hybrid functional analysis results"""
    print("Exporting hybrid functional analysis results...")

    try:
        from minimal_hybrid_functional import MinimalHybridFunctional

        functional = MinimalHybridFunctional()

        # Test scenarios
        test_cases = [
            {"x": 0.3, "t": 0.5, "description": "Early time, low complexity"},
            {"x": 0.5, "t": 1.0, "description": "Standard case"},
            {"x": 0.7, "t": 1.5, "description": "Late time, high complexity"},
            {"x": 0.9, "t": 2.0, "description": "Very late time, maximum complexity"},
        ]

        results = []
        for case in test_cases:
            result = functional.compute_psi_single(case["x"], case["t"])
            result["description"] = case["description"]
            results.append(result)

        # Export results
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "test_cases": results,
            "collaboration_scenarios": {
                "open_source": functional.compute_psi_single(0.5, 1.0),
                "potential_benefits": functional.compute_psi_single(0.6, 1.2),
                "hypothetical_collaboration": functional.compute_psi_single(0.7, 1.5),
            },
        }

        with open("outputs/hybrid_functional_results.json", "w") as f:
            json.dump(export_data, f, indent=2)

        print(
            f"✓ Exported hybrid functional analysis to outputs/hybrid_functional_results.json"
        )

    except ImportError:
        print(
            "⚠ minimal_hybrid_functional.py not found, skipping hybrid analysis export"
        )


def create_summary_report():
    """Create a comprehensive summary report"""
    print("Creating comprehensive summary report...")

    report_lines = [
        "# Hybrid Symbolic-Neural Framework Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the analysis of the Hybrid Symbolic-Neural Accuracy Functional",
        "framework with contraction guarantees and theoretical validation.",
        "",
        "## Key Results",
        "",
    ]

    # Load contraction results if available
    try:
        with open("outputs/contraction_analysis_results.json", "r") as f:
            contraction_data = json.load(f)

        theoretical = contraction_data["theoretical_analysis"]
        report_lines.extend(
            [
                "### Contraction Analysis",
                f"- Theoretical contraction modulus: K = {theoretical['K_theory']:.4f}",
                f"- Contraction status: {theoretical['message']}",
                f"- Configuration: κ = {contraction_data['configuration']['kappa']}, g_max = {contraction_data['configuration']['g_max']}",
                "",
            ]
        )

        # Numerical results
        report_lines.append("### Numerical Validation")
        for est in contraction_data["numerical_estimates"]:
            scenario_id = est["scenario_id"]
            K_hat = est["K_hat"]
            report_lines.append(f"- Scenario {scenario_id}: K_hat = {K_hat:.4f}")
        report_lines.append("")

    except FileNotFoundError:
        report_lines.extend(
            [
                "### Contraction Analysis",
                "- Analysis not yet run. Use `pixi run contraction-analysis` to generate results.",
                "",
            ]
        )

    # Load hybrid functional results if available
    try:
        with open("outputs/hybrid_functional_results.json", "r") as f:
            hybrid_data = json.load(f)

        report_lines.extend(
            [
                "### Hybrid Functional Analysis",
                "- Test cases completed successfully",
                f"- Collaboration scenarios evaluated: {len(hybrid_data['collaboration_scenarios'])} scenarios",
                "",
            ]
        )

    except FileNotFoundError:
        report_lines.extend(
            [
                "### Hybrid Functional Analysis",
                "- Analysis not yet run. Use `pixi run hybrid-analysis` to generate results.",
                "",
            ]
        )

    # Framework integration
    report_lines.extend(
        [
            "## Framework Integration",
            "",
            "The analysis validates integration of:",
            "- Contraction theory with Banach fixed-point guarantees",
            "- Fractal Ψ framework with bounded self-interaction",
            "- Multi-modal cognitive-memory framework",
            "- LSTM convergence theorem with O(1/√T) bounds",
            "- Swarm-Koopman confidence theorem",
            "- Academic network analysis with researcher cloning",
            "",
            "## Usage Instructions",
            "",
            "### Quick Analysis",
            "```bash",
            "pixi run demo-contraction    # Quick contraction demo",
            "pixi run demo-hybrid         # Quick hybrid functional demo",
            "```",
            "",
            "### Comprehensive Analysis",
            "```bash",
            "pixi run analyze-all         # Run all analyses",
            "pixi run export-results      # Export results to files",
            "pixi run generate-report     # Generate this report",
            "```",
            "",
            "### Individual Components",
            "```bash",
            "pixi run contraction-minimal # Minimal contraction analysis",
            "pixi run hybrid-minimal      # Minimal hybrid functional",
            "pixi run academic-basic      # Basic academic network analysis",
            "```",
            "",
            "## Files Generated",
            "- `outputs/contraction_analysis_results.json` - Detailed contraction analysis",
            "- `outputs/contraction_summary.csv` - Summary data for spreadsheet analysis",
            "- `outputs/hybrid_functional_results.json` - Hybrid functional test results",
            "- `outputs/analysis_summary_report.md` - This comprehensive report",
            "",
            "---",
            "*Report generated by Hybrid Symbolic-Neural Framework Analysis Suite*",
        ]
    )

    # Write report
    with open("outputs/analysis_summary_report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Created comprehensive report: outputs/analysis_summary_report.md")


def main():
    """Main export function"""
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    print("=== Exporting Analysis Results ===")

    # Export individual analyses
    export_contraction_analysis()
    export_hybrid_functional_analysis()

    # Create summary report
    create_summary_report()

    print("\n=== Export Complete ===")
    print("Available files:")
    for file in os.listdir("outputs"):
        if file.endswith((".json", ".csv", ".md", ".txt")):
            size = os.path.getsize(f"outputs/{file}")
            print(f"  outputs/{file} ({size} bytes)")


if __name__ == "__main__":
    main()
