// SPDX-License-Identifier: GPL-3.0-only
import Foundation

struct PsiParameters {
    var alpha: Double = 0.5
    var lambda1: Double = 0.6
    var lambda2: Double = 0.4
    var beta: Double = 1.45
    var riskCognitive: Double = 0.11
    var riskEfficiency: Double = 0.06
}

enum PsiModel {
    static func compute(S: Double,
                        N: Double,
                        parameters: PsiParameters) -> Double {
        let s = clamp01(S)
        let n = clamp01(N)
        let a = clamp01(parameters.alpha)
        let penalty = parameters.lambda1 * max(0, parameters.riskCognitive) + parameters.lambda2 * max(0, parameters.riskEfficiency)
        let expPenalty = exp(-penalty)
        let blended = a * s + (1.0 - a) * n
        let raw = parameters.beta * expPenalty * blended
        return min(1.0, max(0.0, raw))
    }

    private static func clamp01(_ v: Double) -> Double {
        return min(1.0, max(0.0, v))
    }
}