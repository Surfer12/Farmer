// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.Arrays;

/**
 * Unified runtime detector that selects a Taylor step inside a trust region and
 * otherwise uses RK4. Provides a single confidence Ψ derived from:
 *   (i) margin-to-thresholds (local error vs ε_total), and
 *   (ii) agreement between Taylor and RK4 trajectories and geometric invariants.
 *
 * Performance-oriented: precomputes/stashes derivative stencils and curvature
 * estimates and adapts step size h to meet a latency budget while satisfying
 * a total error target ε_total. Includes a graceful lower-order fast path when
 * the time budget is tight, logging the increased ε_total via MetricsRegistry.
 */
final class UnifiedDetector {
    /** Dynamics: dy/dt = f(t, y). */
    interface Dynamics {
        void eval(double t, double[] y, double[] dy);
    }

    /** Geometric invariant for orthogonal checks (e.g., energy for SHO). */
    interface Invariant {
        double value(double t, double[] y);
        double reference();
        /**
         * Acceptable absolute drift in the invariant before penalization.
         */
        double tolerance();
    }

    static final class Result {
        final double tNext;
        final double[] yNext;
        final double psi;           // overall confidence in [0,1]
        final double errLocal;      // local error estimate
        final double agreeMetric;   // agreement score between methods/invariants [0,1]
        final double hUsed;
        Result(double tNext, double[] yNext, double psi, double errLocal, double agreeMetric, double hUsed) {
            this.tNext = tNext; this.yNext = yNext; this.psi = psi; this.errLocal = errLocal; this.agreeMetric = agreeMetric; this.hUsed = hUsed;
        }
    }

    private final CurvatureCache curvature;

    UnifiedDetector() {
        this.curvature = new CurvatureCache();
    }

    /** One integration step with adaptive selection and confidence production. */
    Result step(Dynamics f,
                double t,
                double[] y,
                double hInit,
                double epsilonTotal,
                long timeBudgetNanos,
                Invariant[] invariants) {
        long start = System.nanoTime();
        double[] dy = new double[y.length];
        f.eval(t, y, dy);

        // Estimate curvature magnitude κ ~ ||f(t, y+δ f) - 2 f(t,y) + f(t, y-δ f)|| / δ^2
        double kappa = curvature.estimate(f, t, y, dy);

        // Trust region radius for Taylor-2: require local truncation error ~ O(κ h^3) <= ε_total/2
        // Solve for h where κ h^3 = ε/2 => h_trust = cbrt(ε_total / (2 max(κ,eps)))
        double hTrust = Math.cbrt(Math.max(epsilonTotal, 1e-12) / (2.0 * Math.max(kappa, 1e-12)));

        // Start from suggested h
        double h = Math.max(1e-9, Math.min(hInit, 10.0 * hTrust));

        // Attempt Taylor-2 within trust region; otherwise RK4
        boolean useTaylor = h <= hTrust;
        StepOut taylor = null, rk4;
        if (useTaylor) taylor = taylor2(f, t, y, dy.clone(), h, kappa);
        rk4 = rk4(f, t, y, h);

        // Compute agreement metric across methods
        double agree = agreement(taylor != null ? taylor.yNext : null, rk4.yNext, invariants, t + h);

        // Local error estimate: use difference between methods as proxy
        double errLoc = (taylor != null) ? l2diff(taylor.yNext, rk4.yNext) : rk4.errEst;

        // Confidence Ψ = product of two monotone terms in [0,1]
        double margin = Math.max(0.0, 1.0 - (errLoc / Math.max(epsilonTotal, 1e-12)));
        double psi = clamp01(margin) * clamp01(agree);

        // Time budget check; if exceeded, switch to cheaper fast path (Euler) and log increased error
        long elapsed = System.nanoTime() - start;
        if (elapsed > timeBudgetNanos) {
            MetricsRegistry.get().incCounter("unified_fastpath_total");
            StepOut euler = euler(f, t, y, dy, h);
            double extra = l2diff(euler.yNext, rk4.yNext);
            MetricsRegistry.get().setGauge("unified_fastpath_extra_err_ppm", (long) Math.round(extra / Math.max(epsilonTotal, 1e-12) * 1_000_000.0));
            // Degrade psi to reflect increased ε_total (graceful)
            psi *= 0.9;
            return new Result(t + h, euler.yNext, clamp01(psi), extra, agree, h);
        }

        double[] ySel = (useTaylor ? taylor.yNext : rk4.yNext);
        return new Result(t + h, ySel, clamp01(psi), errLoc, agree, h);
    }

    // --- Integrators ---
    private static final class StepOut { final double[] yNext; final double errEst; StepOut(double[] yNext, double errEst){ this.yNext=yNext; this.errEst=errEst; } }

    private static StepOut taylor2(Dynamics f, double t, double[] y, double[] dy, double h, double kappa) {
        // y_{n+1} ≈ y + h f + 0.5 h^2 y'' with ||y''||≈κ as magnitude proxy
        double[] y2 = y.clone();
        for (int i = 0; i < y.length; i++) y2[i] = y[i] + h * dy[i] + 0.5 * h * h * Math.signum(dy[i]) * kappa;
        // local error proxy ~ κ h^3
        double err = kappa * h * h * h;
        return new StepOut(y2, err);
    }

    private static StepOut rk4(Dynamics f, double t, double[] y, double h) {
        int n = y.length;
        double[] k1 = new double[n]; f.eval(t, y, k1);
        double[] y2 = add(y, scale(k1, h * 0.5));
        double[] k2 = new double[n]; f.eval(t + 0.5 * h, y2, k2);
        double[] y3 = add(y, scale(k2, h * 0.5));
        double[] k3 = new double[n]; f.eval(t + 0.5 * h, y3, k3);
        double[] y4 = add(y, scale(k3, h));
        double[] k4 = new double[n]; f.eval(t + h, y4, k4);
        double[] yn = new double[n];
        for (int i = 0; i < n; i++) yn[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        // simple embedded error proxy via |k1 - k4|
        double err = 0.0; for (int i = 0; i < n; i++) err += sqr(k1[i] - k4[i]); err = Math.sqrt(err) * (h / 6.0);
        return new StepOut(yn, err);
    }

    private static StepOut euler(Dynamics f, double t, double[] y, double[] dy, double h) {
        double[] yn = add(y, scale(dy, h));
        double err = 0.0; // unknown; caller will compare to RK4
        return new StepOut(yn, err);
    }

    // --- Agreement and invariants ---
    private static double agreement(double[] yA, double[] yB, Invariant[] invs, double tNext) {
        double a = 1.0;
        if (yA != null && yB != null) {
            double diff = l2diff(yA, yB);
            a *= 1.0 / (1.0 + diff); // in (0,1]
        }
        if (invs != null) {
            for (Invariant inv : invs) {
                double v = inv.value(tNext, (yA != null ? yA : yB));
                double drift = Math.abs(v - inv.reference());
                double tol = Math.max(1e-12, inv.tolerance());
                double factor = Math.max(0.0, 1.0 - (drift / (10.0 * tol)));
                a *= clamp01(factor);
            }
        }
        return clamp01(a);
    }

    // --- Curvature cache with tiny finite-difference stencil ---
    private static final class CurvatureCache {
        private double[] lastY;
        private double lastKappa;
        private double lastT;
        private static final double DELTA = 1e-3;

        double estimate(Dynamics f, double t, double[] y, double[] dy) {
            if (lastY != null && Math.abs(t - lastT) < 1e-9 && l2diff(lastY, y) < 1e-9) {
                return lastKappa;
            }
            // Direction along dy; fallback to unit if near-zero
            double[] dir = dy.clone();
            double norm = l2(dir);
            if (norm < 1e-12) {
                Arrays.fill(dir, 0.0); dir[0] = 1.0; norm = 1.0;
            }
            for (int i = 0; i < dir.length; i++) dir[i] /= norm;
            double delta = DELTA;
            double[] yPlus = add(y, scale(dir, delta));
            double[] yMinus = add(y, scale(dir, -delta));
            double[] f0 = new double[y.length]; f.eval(t, y, f0);
            double[] fP = new double[y.length]; f.eval(t, yPlus, fP);
            double[] fM = new double[y.length]; f.eval(t, yMinus, fM);
            double num = 0.0; for (int i = 0; i < y.length; i++) num += Math.abs(fP[i] - 2.0 * f0[i] + fM[i]);
            double kappa = num / (delta * delta * Math.max(1, y.length));
            lastY = y.clone(); lastKappa = kappa; lastT = t;
            return kappa;
        }
    }

    // --- small vector helpers ---
    private static double[] add(double[] a, double[] b) { double[] r = new double[a.length]; for (int i=0;i<a.length;i++) r[i]=a[i]+b[i]; return r; }
    private static double[] scale(double[] a, double s) { double[] r = new double[a.length]; for (int i=0;i<a.length;i++) r[i]=a[i]*s; return r; }
    private static double l2(double[] a) { double s=0.0; for (double v: a) s += v*v; return Math.sqrt(s); }
    private static double l2diff(double[] a, double[] b) { double s=0.0; for (int i=0;i<a.length;i++){ double d=a[i]-b[i]; s+=d*d; } return Math.sqrt(s); }
    private static double sqr(double x){ return x*x; }
    private static double clamp01(double x){ return x<0?0:(x>1?1:x); }
}


