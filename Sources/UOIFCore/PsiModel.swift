// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public enum PsiModel {
  public static func computeHybrid(alpha: Double, S: Double, N: Double) -> Double {
    return alpha * S + (1.0 - alpha) * N
  }

  public static func computePenalty(
    lambdaAuthority: Double,
    lambdaVerifiability: Double,
    riskAuthority: Double,
    riskVerifiability: Double
  ) -> Double {
    let exponent = -1.0 * (lambdaAuthority * riskAuthority + lambdaVerifiability * riskVerifiability)
    return exp(exponent)
  }

  public static func computePosteriorCapped(basePosterior: Double, beta: Double) -> Double {
    let scaled = basePosterior * beta
    return min(max(scaled, 0.0), 1.0)
  }

  public static func computePsi(inputs: PsiInputs) -> PsiOutcome {
    let hybrid = computeHybrid(alpha: inputs.alpha, S: inputs.S_symbolic, N: inputs.N_external)
    let penalty = computePenalty(
      lambdaAuthority: inputs.lambdaAuthority,
      lambdaVerifiability: inputs.lambdaVerifiability,
      riskAuthority: inputs.riskAuthority,
      riskVerifiability: inputs.riskVerifiability
    )
    let posterior = computePosteriorCapped(basePosterior: inputs.basePosterior, beta: inputs.betaUplift)
    let psi = hybrid * penalty * posterior
    let dPsi_dAlpha = (inputs.S_symbolic - inputs.N_external) * penalty * posterior
    return PsiOutcome(hybrid: hybrid, penalty: penalty, posterior: posterior, psi: psi, dPsi_dAlpha: dPsi_dAlpha)
  }

  // Convenience: map user-friendly terms to inputs
  public static func computePsi(
    alpha: Double,
    S: Double,
    N: Double,
    R_cognitive: Double,
    R_efficiency: Double,
    lambda1: Double,
    lambda2: Double,
    basePosterior: Double,
    beta: Double
  ) -> PsiOutcome {
    let inputs = PsiInputs(
      alpha: alpha,
      S_symbolic: S,
      N_external: N,
      lambdaAuthority: lambda1,
      lambdaVerifiability: lambda2,
      riskAuthority: R_cognitive,
      riskVerifiability: R_efficiency,
      basePosterior: basePosterior,
      betaUplift: beta
    )
    return computePsi(inputs: inputs)
  }
}

