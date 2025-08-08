// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;

/**
 * Triad gating: combines RK4 band, Taylor trust-region estimate, and geometry invariants
 * into a single accept/reject decision under an error-budget gate.
 */
public final class TriadGate {
    public static final class Config {
        public final double tau;
        public final double epsTotal;
        public final double trustRadiusL2;
        public final long timeBudgetMicros;
        public Config(double tau, double epsTotal, double trustRadiusL2, long timeBudgetMicros) {
            this.tau = clamp01(tau);
            this.epsTotal = Math.max(0.0, epsTotal);
            this.trustRadiusL2 = Math.max(0.0, trustRadiusL2);
            this.timeBudgetMicros = Math.max(0L, timeBudgetMicros);
        }
    }

    public static final class Result {
        public final double psiExact;
        public final double psiTaylor;
        public final double rk4Low, rk4High;
        public final double psiChosen;
        public final double epsRk4, epsTaylor, epsGeom, epsTotal;
        public final boolean gatePassed;
        public final boolean invariantsOk;
        public final double marginAbs;
        public final List<String> reasons;
        Result(double psiExact, double psiTaylor, double rk4Low, double rk4High,
               double psiChosen, double epsRk4, double epsTaylor, double epsGeom, double epsTotal,
               boolean gatePassed, boolean invariantsOk, double marginAbs, List<String> reasons) {
            this.psiExact = psiExact; this.psiTaylor = psiTaylor; this.rk4Low = rk4Low; this.rk4High = rk4High;
            this.psiChosen = psiChosen; this.epsRk4 = epsRk4; this.epsTaylor = epsTaylor; this.epsGeom = epsGeom;
            this.epsTotal = epsTotal; this.gatePassed = gatePassed; this.invariantsOk = invariantsOk;
            this.marginAbs = marginAbs; this.reasons = reasons;
        }
    }

    private final HierarchicalBayesianModel model;
    private final RK4Controller rk4;
    private final TaylorR4Estimator taylor;

    public TriadGate(HierarchicalBayesianModel model, double trustRadiusL2) {
        this.model = model;
        this.rk4 = new RK4Controller(0.9, 0.1);
        this.taylor = new TaylorR4Estimator(model, trustRadiusL2);
    }

    public Result evaluate(List<ClaimData> dataset,
                           ModelParameters current,
                           ModelParameters previous,
                           Config cfg) {
        HierarchicalBayesianModel.Prepared prep = model.precompute(dataset);
        boolean parallel = model.shouldParallelize(dataset.size());

        // Exact
        double psiExact = PsiSensitivity.meanPsi(prep, current);

        // Taylor R4 (first-order + remainder scaffold)
        TaylorR4Estimator.Estimate te = taylor.estimate(dataset, current, previous != null ? previous : current);
        double psiTaylor = te.value;
        double epsTaylor = te.remainder;

        // RK4 band via step-doubling on linear path Ψ(t) from prev->current using gradient dot delta
        double[] delta = new double[] {
                (current.S() - (previous != null ? previous.S() : current.S())),
                (current.N() - (previous != null ? previous.N() : current.N())),
                (current.alpha() - (previous != null ? previous.alpha() : current.alpha())),
                (current.beta() - (previous != null ? previous.beta() : current.beta()))
        };
        PsiSensitivity.Partials gPrev = PsiSensitivity.partials(prep, previous != null ? previous : current);
        // f(t) ≈ dΨ/dt ≈ gradPsi(prev + t·delta) · delta  (use prev gradient as proxy)
        java.util.function.DoubleUnaryOperator f = (double t) ->
                gPrev.dS * delta[0] + gPrev.dN * delta[1] + gPrev.dAlpha * delta[2] + gPrev.dBeta * delta[3];
        RK4Controller.StepResult stepFull = rk4.step(f, 0.0, PsiSensitivity.meanPsi(prep, previous != null ? previous : current), 1.0, cfg.epsTotal);
        // half stepping: controller already did step-doubling inside; use err as epsRk4
        double epsRk4 = Double.isNaN(stepFull.errEst) ? 0.0 : Math.abs(stepFull.errEst);
        double rk4Low = clamp01(psiExact - epsRk4);
        double rk4High = clamp01(psiExact + epsRk4);

        // Geometry invariants
        GeometryInvariants.Check gc = GeometryInvariants.verify(prep, current);
        double epsGeom = gc.epsGeom;

        double epsTotal = epsRk4 + epsTaylor + epsGeom;
        boolean gate = epsTotal <= cfg.epsTotal;
        double chosen = psiExact; // prefer exact when available
        double margin = Math.abs(chosen - cfg.tau);
        List<String> reasons = new ArrayList<>();
        if (!gate) reasons.add("eps_total_exceeded");
        if (!gc.ok) reasons.add("invariants_failed");

        return new Result(psiExact, psiTaylor, rk4Low, rk4High, chosen, epsRk4, epsTaylor, epsGeom, epsTotal, gate, gc.ok, margin, reasons);
    }

    private static double clamp01(double x) { if (x < 0) return 0; if (x > 1) return 1; return x; }
}


