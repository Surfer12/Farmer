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

// Numeric Ψ(x) example from brief
print("Example — hybrid Ψ(x) calc:")
let S_example = 0.72
let N_example = 0.85
let alpha_example = 0.5
let O_hybrid = PsiModel.computeHybrid(alpha: alpha_example, S: S_example, N: N_example) // 0.785
let penalty_example = PsiModel.computePenalty(
  lambdaAuthority: 0.6,
  lambdaVerifiability: 0.4,
  riskAuthority: 0.15,
  riskVerifiability: 0.10
) // exp(-P_total) ~ 0.878
let posterior_example = PsiModel.computePosteriorCapped(basePosterior: 0.80, beta: 1.2) // ~0.96
let psi_example = O_hybrid * penalty_example * posterior_example // ~0.662
print(String(format: "O=%.3f, pen=%.3f, P*=%.3f ⇒ Ψ=%.3f", O_hybrid, penalty_example, posterior_example, psi_example))
print()

// Tiny PINN demo: single training step for 1D inviscid Burgers
print("PINN — 1D inviscid Burgers (single step demo):")
let demo = PINNDemo.singleStepDemo(hidden: [8, 8], samplePoints: 10, seed: 123)
print(String(format: "loss before=%.6f, after=%.6f", demo.lossBefore, demo.lossAfter))
for (x, t, u) in demo.sampleU {
  print(String(format: "u(x=%.2f,t=%.2f)=%.4f", x, t, u))
}
print()

// Short reflection
print("Reflection:")
print("- Hybrid linearity gives monotone, auditable responses as canonical sources arrive (alpha ↓ ⇒ Psi ↑ when N>S).")
print("- Exponential penalty and capped posterior maintain Psi in [0,1] and prevent overconfidence.")
print("- Confidence trail marks robustness at each step; promotions are tied to observable artifacts.\n")

