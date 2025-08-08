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
}

