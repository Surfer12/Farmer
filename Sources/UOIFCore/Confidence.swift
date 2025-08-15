// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public enum ConfidenceHeuristics {
  public static func overall(
    sources: Double,
    hybrid: Double,
    penalty: Double,
    posterior: Double
  ) -> Double {
    // Simple weighted blend; tune as needed
    let weights = (sources: 0.35, hybrid: 0.25, penalty: 0.20, posterior: 0.20)
    let score = weights.sources * sources
      + weights.hybrid * hybrid
      + weights.penalty * penalty
      + weights.posterior * posterior
    return min(max(score, 0.0), 1.0)
  }
}

public enum Presets {
  // Common constants
  private static let S: Double = 0.60
  private static let lambda1: Double = 0.85
  private static let lambda2: Double = 0.15

  // 2025 results (canonical live, eased)
  public static func eval2025Results(alpha: Double) -> Evaluation {
    let inputs = PsiInputs(
      alpha: alpha,
      S_symbolic: S,
      N_external: 0.97,
      lambdaAuthority: lambda1,
      lambdaVerifiability: lambda2,
      riskAuthority: 0.12,
      riskVerifiability: 0.04,
      basePosterior: 0.90,
      betaUplift: 1.15
    )
    let conf = ConfidenceBundle(
      sources: 0.98, hybrid: 0.96, penalty: 0.85, posterior: 0.88,
      psiOverall: 0.0
    )
    let label = Label.primitiveEmpiricallyGrounded.rawValue // for results primitives
    return Evaluation(title: "2025 Results (canonical eased)", inputs: inputs, confidence: conf, label: label)
  }

  // 2025 problems (pending canonical)
  public static func eval2025Problems(alpha: Double, N: Double) -> Evaluation {
    let inputs = PsiInputs(
      alpha: alpha,
      S_symbolic: S,
      N_external: N, // 0.88...0.90
      lambdaAuthority: lambda1,
      lambdaVerifiability: lambda2,
      riskAuthority: 0.25,
      riskVerifiability: 0.10,
      basePosterior: 0.90,
      betaUplift: 1.05
    )
    let conf = ConfidenceBundle(
      sources: 0.88, hybrid: 0.85, penalty: 0.80, posterior: 0.85,
      psiOverall: 0.0
    )
    let label = Label.interpretive.rawValue
    return Evaluation(title: "2025 Problems (pending canonical)", inputs: inputs, confidence: conf, label: label)
  }

  // 2024 (DeepMind P1/P2/P4)
  public static func eval2024(alpha: Double) -> Evaluation {
    let inputs = PsiInputs(
      alpha: alpha,
      S_symbolic: S,
      N_external: 0.96,
      lambdaAuthority: lambda1,
      lambdaVerifiability: lambda2,
      riskAuthority: 0.10,
      riskVerifiability: 0.05,
      basePosterior: 0.90,
      betaUplift: 1.05
    )
    let conf = ConfidenceBundle(
      sources: 0.90, hybrid: 0.88, penalty: 0.85, posterior: 0.85,
      psiOverall: 0.0
    )
    let label = Label.primitiveEmpiricallyGrounded.rawValue
    return Evaluation(title: "2024 P1/P2/P4", inputs: inputs, confidence: conf, label: label)
  }
<<<<<<< Current (Your changes)
=======

  // User numeric example: S=0.72, N=0.85, alpha=0.5; R_cognitive=0.15, R_efficiency=0.10; lambda1=0.6, lambda2=0.4; basePosterior=0.80, beta=1.2
  public static func evalHybridPINNExample() -> Evaluation {
    let S_user = 0.72
    let N_user = 0.85
    let alpha: Double = 0.5
    let lambda1_user: Double = 0.6
    let lambda2_user: Double = 0.4
    let R_cognitive: Double = 0.15
    let R_efficiency: Double = 0.10
    let basePosterior: Double = 0.80
    let beta: Double = 1.2

    let inputs = PsiInputs(
      alpha: alpha,
      S_symbolic: S_user,
      N_external: N_user,
      lambdaAuthority: lambda1_user,
      lambdaVerifiability: lambda2_user,
      riskAuthority: R_cognitive,
      riskVerifiability: R_efficiency,
      basePosterior: basePosterior,
      betaUplift: beta
    )
    let out = PsiModel.computePsi(inputs: inputs)
    let conf = ConfidenceBundle(
      sources: 0.90,
      hybrid: out.hybrid,
      penalty: out.penalty,
      posterior: out.posterior,
      psiOverall: 0.0
    )
    return Evaluation(title: "Hybrid PINN Example (single step)", inputs: inputs, confidence: conf, label: Label.empiricallyGrounded.rawValue)
  }
>>>>>>> Incoming (Background Agent changes)
}

