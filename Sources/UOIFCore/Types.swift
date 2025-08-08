// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import Foundation

public struct PsiInputs {
  public let alpha: Double
  public let S_symbolic: Double
  public let N_external: Double
  public let lambdaAuthority: Double
  public let lambdaVerifiability: Double
  public let riskAuthority: Double
  public let riskVerifiability: Double
  public let basePosterior: Double
  public let betaUplift: Double

  public init(
    alpha: Double,
    S_symbolic: Double,
    N_external: Double,
    lambdaAuthority: Double,
    lambdaVerifiability: Double,
    riskAuthority: Double,
    riskVerifiability: Double,
    basePosterior: Double,
    betaUplift: Double
  ) {
    self.alpha = alpha
    self.S_symbolic = S_symbolic
    self.N_external = N_external
    self.lambdaAuthority = lambdaAuthority
    self.lambdaVerifiability = lambdaVerifiability
    self.riskAuthority = riskAuthority
    self.riskVerifiability = riskVerifiability
    self.basePosterior = basePosterior
    self.betaUplift = betaUplift
  }
}

public struct PsiOutcome {
  public let hybrid: Double
  public let penalty: Double
  public let posterior: Double
  public let psi: Double
  public let dPsi_dAlpha: Double
}

public struct ConfidenceBundle {
  public let sources: Double
  public let hybrid: Double
  public let penalty: Double
  public let posterior: Double
  public let psiOverall: Double
}

public struct Evaluation {
  public let title: String
  public let inputs: PsiInputs
  public let confidence: ConfidenceBundle
  public let label: String
}

public enum Label: String {
  case interpretive = "Interpretive/Contextual"
  case empiricallyGrounded = "Empirically Grounded"
  case primitiveEmpiricallyGrounded = "Primitive/Empirically Grounded"
}

