// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
import XCTest
@testable import UOIFCore

final class PsiModelTests: XCTestCase {
  func testBoundsAndSensitivity() {
    let inputs = PsiInputs(
      alpha: 0.12, S_symbolic: 0.60, N_external: 0.97,
      lambdaAuthority: 0.85, lambdaVerifiability: 0.15,
      riskAuthority: 0.12, riskVerifiability: 0.04,
      basePosterior: 0.90, betaUplift: 1.15
    )
    let out = PsiModel.computePsi(inputs: inputs)
    XCTAssert(out.hybrid >= 0 && out.hybrid <= 1)
    XCTAssert(out.penalty > 0 && out.penalty <= 1)
    XCTAssert(out.posterior >= 0 && out.posterior <= 1)
    XCTAssert(out.psi >= 0 && out.psi <= 1)
    // Sensitivity sign: S - N < 0
    XCTAssertLessThan(out.dPsi_dAlpha, 0.0)
  }
}

