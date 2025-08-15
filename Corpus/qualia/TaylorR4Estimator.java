// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.List;

/**
 * Fourth-order Taylor estimator scaffold for mean Ψ with trust region and remainder bound.
 * Currently provides 1st-order implementation with hooks for higher-order terms and remainder.
 */
public final class TaylorR4Estimator {
    public static final class Estimate {
        public final double value;     // estimated mean Ψ
        public final double remainder; // bound on truncation error
        public final boolean withinTrust;
        Estimate(double value, double remainder, boolean withinTrust) {
            this.value = value; this.remainder = Math.max(0.0, remainder); this.withinTrust = withinTrust;
        }
    }

    private final HierarchicalBayesianModel model;
    private final double trustRadiusL2;

    public TaylorR4Estimator(HierarchicalBayesianModel model, double trustRadiusL2) {
        this.model = model;
        this.trustRadiusL2 = Math.max(0.0, trustRadiusL2);
    }

    public Estimate estimate(List<ClaimData> dataset, ModelParameters current, ModelParameters anchor) {
        HierarchicalBayesianModel.Prepared prep = model.precompute(dataset);
        boolean ok = l2Delta(current, anchor) <= trustRadiusL2;
        if (!ok) {
            double exact = PsiSensitivity.meanPsi(prep, current);
            return new Estimate(exact, 0.0, false);
        }
        // First-order now; placeholders for 2nd..4th and remainder
        double value = taylorFirst(prep, current, anchor);
        double rem = estimateRemainderUpperBound(current, anchor);
        return new Estimate(value, rem, true);
    }

    private double taylorFirst(HierarchicalBayesianModel.Prepared prep, ModelParameters current, ModelParameters anchor) {
        double psi0 = PsiSensitivity.meanPsi(prep, anchor);
        PsiSensitivity.Partials g = PsiSensitivity.partials(prep, anchor);
        double dS = current.S() - anchor.S();
        double dN = current.N() - anchor.N();
        double dA = current.alpha() - anchor.alpha();
        double dB = current.beta() - anchor.beta();
        double lin = psi0 + g.dS * dS + g.dN * dN + g.dAlpha * dA + g.dBeta * dB;
        return clamp01(lin);
    }

    private double estimateRemainderUpperBound(ModelParameters current, ModelParameters anchor) {
        // Placeholder: scale with squared norm and a conservative Lipschitz-like constant
        double r = l2Delta(current, anchor);
        double L = 2.0; // tunable
        return Math.min(1.0, L * r * r);
    }

    private static double l2Delta(ModelParameters a, ModelParameters b) {
        double dS = a.S() - b.S();
        double dN = a.N() - b.N();
        double dA = a.alpha() - b.alpha();
        double dB = Math.log(Math.max(1e-12, a.beta())) - Math.log(Math.max(1e-12, b.beta()));
        return Math.sqrt(dS*dS + dN*dN + dA*dA + dB*dB);
    }

    private static double clamp01(double x) { if (x < 0) return 0; if (x > 1) return 1; return x; }
}


