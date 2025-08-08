// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.List;

/**
 * Unified Ψ evaluator that selects a first-order Taylor approximation within a
 * trust region, otherwise falls back to exact evaluation. Performs orthogonal
 * geometric invariants checks and returns a confidence score derived from
 * (i) margin to threshold(s) and (ii) agreement across methods.
 */
public final class UnifiedPsiRuntime {

    public static final class Config {
        public final double trustRadiusL2;      // parameter delta norm threshold
        public final double thresholdTau;       // decision threshold for margin
        public final long timeBudgetMicros;     // optional runtime budget
        public Config(double trustRadiusL2, double thresholdTau, long timeBudgetMicros) {
            this.trustRadiusL2 = Math.max(0.0, trustRadiusL2);
            this.thresholdTau = clamp01(thresholdTau);
            this.timeBudgetMicros = Math.max(0L, timeBudgetMicros);
        }
    }

    public static final class Result {
        public final double psi;            // final Ψ used for decisions
        public final double psiTaylor;      // Taylor estimate (if used)
        public final double psiExact;       // exact Ψ (mean over dataset)
        public final double confidence;     // [0,1]
        public final boolean invariantsOk;  // geometric checks passed
        Result(double psi, double psiTaylor, double psiExact, double confidence, boolean invariantsOk) {
            this.psi = psi; this.psiTaylor = psiTaylor; this.psiExact = psiExact;
            this.confidence = clamp01(confidence); this.invariantsOk = invariantsOk;
        }
    }

    private final HierarchicalBayesianModel model;

    public UnifiedPsiRuntime(HierarchicalBayesianModel model) {
        this.model = model;
    }

    /**
     * Evaluate unified Ψ for a dataset using current parameters, optionally using
     * a first-order Taylor expansion around prevParams when within the trust region.
     *
     * - Taylor path: fast approximate mean Ψ using analytic partials; requires "cap"
     *   handling via indicator for β·P<1.
     * - Exact path: mean Ψ via direct formula.
     * - Confidence combines margin to threshold and method agreement.
     */
    public Result evaluate(List<ClaimData> dataset,
                           ModelParameters params,
                           ModelParameters prevParams,
                           Config cfg) {
        long t0 = System.nanoTime();
        HierarchicalBayesianModel.Prepared prep = model.precompute(dataset);
        boolean parallel = model.shouldParallelize(dataset.size());

        double exact = meanPsi(prep, params, parallel);

        double psiTaylor = Double.NaN;
        boolean withinTrust = prevParams != null && l2Delta(params, prevParams) <= cfg.trustRadiusL2;
        if (withinTrust) {
            psiTaylor = taylorMeanPsi(prep, params, prevParams);
        }

        // Selection with runtime budget/graceful degradation
        long elapsedUs = (System.nanoTime() - t0) / 1_000L;
        double chosen;
        if (withinTrust && (cfg.timeBudgetMicros > 0 && elapsedUs > cfg.timeBudgetMicros)) {
            // time budget tight: prefer Taylor
            chosen = psiTaylor;
        } else if (withinTrust) {
            // Prefer exact but only if still within budget
            chosen = exact;
        } else {
            chosen = exact;
        }

        boolean invOk = checkInvariants(prep, params);
        double conf = computeConfidence(chosen, exact, psiTaylor, cfg.thresholdTau, withinTrust, invOk);
        return new Result(chosen, psiTaylor, exact, conf, invOk);
    }

    /** Mean Ψ across prepared dataset. */
    private double meanPsi(HierarchicalBayesianModel.Prepared prep, ModelParameters p, boolean parallel) {
        double O = p.alpha() * p.S() + (1.0 - p.alpha()) * p.N();
        double beta = p.beta();
        int n = prep.size();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double pHb = Math.min(beta * prep.pHe[i], 1.0);
            double psi = O * prep.pen[i] * pHb;
            if (psi < 0.0) psi = 0.0; else if (psi > 1.0) psi = 1.0;
            sum += psi;
        }
        return sum / Math.max(1, n);
    }

    /** First-order Taylor expansion of mean Ψ around prevParams. */
    private double taylorMeanPsi(HierarchicalBayesianModel.Prepared prep, ModelParameters current, ModelParameters prev) {
        int n = prep.size();
        // Aggregate helpers
        double meanPenPbeta = 0.0;         // E[pen * pBeta(prev)]
        double meanPenP = 0.0;             // E[pen * P]
        boolean anyCap = false;
        double Oprev = prev.alpha() * prev.S() + (1.0 - prev.alpha()) * prev.N();
        for (int i = 0; i < n; i++) {
            double P = prep.pHe[i];
            double pen = prep.pen[i];
            boolean capped = (prev.beta() * P >= 1.0);
            double pBeta = capped ? 1.0 : prev.beta() * P;
            if (capped) anyCap = true;
            meanPenPbeta += pen * pBeta;
            meanPenP += pen * P;
        }
        meanPenPbeta /= Math.max(1, n);
        meanPenP /= Math.max(1, n);

        // Partials at prev
        double dPsi_dS = prev.alpha() * meanPenPbeta;
        double dPsi_dN = (1.0 - prev.alpha()) * meanPenPbeta;
        double dPsi_dAlpha = (prev.S() - prev.N()) * meanPenPbeta;
        double dPsi_dBeta = anyCap ? cappedAwareBetaPartial(prep, prev) : (Oprev * meanPenP);

        // Delta
        double dS = current.S() - prev.S();
        double dN = current.N() - prev.N();
        double dA = current.alpha() - prev.alpha();
        double dB = current.beta() - prev.beta();

        double psiPrev = meanPsi(prep, prev, false);
        double lin = psiPrev + dPsi_dS * dS + dPsi_dN * dN + dPsi_dAlpha * dA + dPsi_dBeta * dB;
        return clamp01(lin);
    }

    private double cappedAwareBetaPartial(HierarchicalBayesianModel.Prepared prep, ModelParameters prev) {
        // dΨ/dβ = E[O*pen*P] on non-capped subset; 0 on capped
        int n = prep.size();
        double O = prev.alpha() * prev.S() + (1.0 - prev.alpha()) * prev.N();
        double sum = 0.0; int m = 0;
        for (int i = 0; i < n; i++) {
            double P = prep.pHe[i];
            if (prev.beta() * P < 1.0) { sum += O * prep.pen[i] * P; m++; }
        }
        if (m == 0) return 0.0;
        return sum / m;
    }

    private boolean checkInvariants(HierarchicalBayesianModel.Prepared prep, ModelParameters p) {
        int n = prep.size();
        double meanPenPbeta = 0.0;
        double meanPenP = 0.0;
        boolean anyCap = false;
        for (int i = 0; i < n; i++) {
            double P = prep.pHe[i];
            double pen = prep.pen[i];
            boolean capped = (p.beta() * P >= 1.0);
            double pBeta = capped ? 1.0 : p.beta() * P;
            if (capped) anyCap = true;
            meanPenPbeta += pen * pBeta;
            meanPenP += pen * P;
        }
        meanPenPbeta /= Math.max(1, n);
        meanPenP /= Math.max(1, n);

        double dS = p.alpha() * meanPenPbeta;
        double dN = (1.0 - p.alpha()) * meanPenPbeta;
        double dAlpha = (p.S() - p.N()) * meanPenPbeta;
        double dBeta = anyCap ? cappedAwareBetaPartial(prep, p) : ((p.alpha() * p.S() + (1.0 - p.alpha()) * p.N()) * meanPenP);
        boolean ok = dS >= -1e-12 && dN >= -1e-12;
        // dAlpha sign matches sign(S-N)
        if (p.S() >= p.N()) ok &= dAlpha >= -1e-12; else ok &= dAlpha <= 1e-12;
        ok &= dBeta >= -1e-12;
        return ok;
    }

    private static double computeConfidence(double chosen,
                                            double exact,
                                            double taylor,
                                            double tau,
                                            boolean withinTrust,
                                            boolean invOk) {
        // Margin component: distance from decision threshold
        double margin = Math.abs(chosen - tau);
        double cMargin = Math.min(1.0, margin / 0.25); // scale: 0 @ on-threshold, ~1 beyond 0.25 away
        // Agreement component: 1 - normalized discrepancy (if Taylor present)
        double cAgree = 1.0;
        if (withinTrust && !Double.isNaN(taylor)) {
            double diff = Math.abs(exact - taylor);
            cAgree = Math.max(0.0, 1.0 - Math.min(1.0, diff / 0.05)); // 5% band full credit
        }
        double base = 0.6 * cMargin + 0.4 * cAgree;
        if (!invOk) base *= 0.5; // penalize if invariants fail
        return Math.max(0.0, Math.min(1.0, base));
    }

    private static double l2Delta(ModelParameters a, ModelParameters b) {
        double dS = a.S() - b.S();
        double dN = a.N() - b.N();
        double dA = a.alpha() - b.alpha();
        double dB = Math.log(Math.max(1e-12, a.beta())) - Math.log(Math.max(1e-12, b.beta()));
        return Math.sqrt(dS*dS + dN*dN + dA*dA + dB*dB);
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0; if (x > 1.0) return 1.0; return x;
    }
}


