// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation
import UOIFCore

func printEvaluation(_ evaluation: Evaluation) {
  let outcome = PsiModel.computePsi(inputs: evaluation.inputs)
  let overallConf = ConfidenceHeuristics.overall(
    sources: evaluation.confidence.sources,
    hybrid: evaluation.confidence.hybrid,
    penalty: evaluation.confidence.penalty,
    posterior: evaluation.confidence.posterior
  )
  print("=== \(evaluation.title) ===")
  print(String(format: "alpha=%.3f, S=%.2f, N=%.2f", evaluation.inputs.alpha, evaluation.inputs.S_symbolic, evaluation.inputs.N_external))
  print(String(format: "hybrid=%.4f, penalty=%.4f, posterior=%.4f", outcome.hybrid, outcome.penalty, outcome.posterior))
  print(String(format: "Psi=%.3f, dPsi/dAlpha=%.4f", outcome.psi, outcome.dPsi_dAlpha))
  print(String(format: "confidence: sources=%.2f, hybrid=%.2f, penalty=%.2f, posterior=%.2f, overall=%.2f",
               evaluation.confidence.sources,
               evaluation.confidence.hybrid,
               evaluation.confidence.penalty,
               evaluation.confidence.posterior,
               overallConf))
  print("label: \(evaluation.label)")
  print()
}

print("UOIF CLI — recompute and reflect\n")

// 2025 results, two alphas
printEvaluation(Presets.eval2025Results(alpha: 0.12))
printEvaluation(Presets.eval2025Results(alpha: 0.15))

// 2025 problems, range examples
printEvaluation(Presets.eval2025Problems(alpha: 0.17, N: 0.89))
printEvaluation(Presets.eval2025Problems(alpha: 0.15, N: 0.90))
printEvaluation(Presets.eval2025Problems(alpha: 0.20, N: 0.88))

// 2024, two alphas
printEvaluation(Presets.eval2024(alpha: 0.10))
printEvaluation(Presets.eval2024(alpha: 0.15))

// Short reflection
print("Reflection:")
print("- Hybrid linearity gives monotone, auditable responses as canonical sources arrive (alpha ↓ ⇒ Psi ↑ when N>S).")
print("- Exponential penalty and capped posterior maintain Psi in [0,1] and prevent overconfidence.")
print("- Confidence trail marks robustness at each step; promotions are tied to observable artifacts.\n")

