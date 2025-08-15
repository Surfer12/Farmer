// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.*;

public final class PsiMcdaTest {
    public static void main(String[] args) {
        int passed = 0, total = 0;

        total++; if (testComputePsiDeterminismAndRanges()) passed++; else throw new AssertionError("testComputePsiDeterminismAndRanges");
        total++; if (testComputePsiTemporalMean()) passed++; else throw new AssertionError("testComputePsiTemporalMean");
        total++; if (testThresholdTransfer()) passed++; else throw new AssertionError("testThresholdTransfer");
        total++; if (testNormalizeCriterion()) passed++; else throw new AssertionError("testNormalizeCriterion");
        total++; if (testMapGovernanceWeights()) passed++; else throw new AssertionError("testMapGovernanceWeights");
        total++; if (testGateByPsi()) passed++; else throw new AssertionError("testGateByPsi");
        total++; if (testWSM_WPM()) passed++; else throw new AssertionError("testWSM_WPM");
        total++; if (testTopsisTrivial()) passed++; else throw new AssertionError("testTopsisTrivial");
        total++; if (testAHPIdentity()) passed++; else throw new AssertionError("testAHPIdentity");
        total++; if (testBounds()) passed++; else throw new AssertionError("testBounds");
        total++; if (testGradPsiSigns()) passed++; else throw new AssertionError("testGradPsiSigns");
        total++; if (testTieBreak()) passed++; else throw new AssertionError("testTieBreak");

        System.out.println("PsiMcdaTest: passed " + passed + "/" + total + " tests");
        if (passed != total) System.exit(1);
    }

    static boolean testComputePsiDeterminismAndRanges() {
        PsiMcda.PsiResult r1 = PsiMcda.computePsi(0.4, 0.8, 0.6, 0.2, 0.1, 0.5, 0.3, 1.2);
        PsiMcda.PsiResult r2 = PsiMcda.computePsi(0.4, 0.8, 0.6, 0.2, 0.1, 0.5, 0.3, 1.2);
        return close(r1.psi(), r2.psi()) && inRange01(r1.psi());
    }

    static boolean testComputePsiTemporalMean() {
        double[] w = {0.25, 0.25, 0.5};
        double[] S = {0.4, 0.4, 0.4};
        double[] N = {0.8, 0.8, 0.8};
        double[] a = {0.6, 0.6, 0.6};
        double[] Ra = {0.2, 0.2, 0.2};
        double[] Rv = {0.1, 0.1, 0.1};
        double[] beta = {1.2, 1.2, 1.2};
        double lambda1 = 0.5, lambda2 = 0.3;
        double mean = 0.0;
        for (int i = 0; i < w.length; i++) {
            mean += w[i] * PsiMcda.computePsi(S[i], N[i], a[i], Ra[i], Rv[i], lambda1, lambda2, beta[i]).psi();
        }
        double got = PsiMcda.computePsiTemporal(w, S, N, a, Ra, Rv, beta, lambda1, lambda2, PsiMcda.Aggregator.MEAN);
        return close(mean, got) && inRange01(got);
    }

    static boolean testThresholdTransfer() {
        double t1 = PsiMcda.thresholdTransfer(1.5, 0.4, "subcap"); // 0.6
        double t2 = PsiMcda.thresholdTransfer(2.0, 0.5, "softcap"); // 0.75
        return close(t1, 0.6) && close(t2, 0.75) && inRange01(t1) && inRange01(t2);
    }

    static boolean testNormalizeCriterion() {
        double[] vals = {0.0, 5.0, 10.0};
        double[] zb = PsiMcda.normalizeCriterion(vals, PsiMcda.Direction.BENEFIT);
        double[] zc = PsiMcda.normalizeCriterion(vals, PsiMcda.Direction.COST);
        return close(zb[0], 0.0) && close(zb[2], 1.0) && close(zc[0], 1.0) && close(zc[2], 0.0)
                && allIn01(zb) && allIn01(zc);
    }

    static boolean testMapGovernanceWeights() {
        double[] base = {0.7, 0.3};
        double[] w = PsiMcda.mapGovernanceWeights(base, 0.8, 0.5);
        double sum = Arrays.stream(w).sum();
        return close(sum, 1.0) && w.length == 3 && w[0] > 0.0;
    }

    static boolean testGateByPsi() {
        List<Double> xs = Arrays.asList(0.2, 0.5, 0.8);
        List<Double> out = PsiMcda.gateByPsi(xs, d -> d, 0.5);
        return out.size() == 2 && out.get(0) == 0.5 && out.get(1) == 0.8;
    }

    static boolean testWSM_WPM() {
        double[] w = {0.2, 0.8};
        double[] z = {1.0, 0.0};
        double u = PsiMcda.wsmScore(w, z);
        boolean ok1 = close(u, 0.2);
        double[] w2 = {0.5, 0.5};
        double[] z2 = {0.25, 1.0};
        double wp = PsiMcda.wpmScore(w2, z2);
        boolean ok2 = close(wp, 0.5);
        return ok1 && ok2;
    }

    static boolean testTopsisTrivial() {
        double[][] z = {{1.0}, {0.0}};
        double[] w = {1.0};
        PsiMcda.Direction[] dir = {PsiMcda.Direction.BENEFIT};
        double[] cc = PsiMcda.topsisCloseness(z, w, dir);
        return close(cc[0], 1.0) && close(cc[1], 0.0);
    }

    static boolean testAHPIdentity() {
        double[][] I = {{1,1,1},{1,1,1},{1,1,1}}; // perfectly consistent and symmetric
        PsiMcda.AHPResult res = PsiMcda.ahpWeights(I);
        boolean eq = close(res.weights()[0], res.weights()[1]) && close(res.weights()[1], res.weights()[2]);
        return eq && res.consistencyRatio() >= 0.0;
    }

    static boolean testBounds() {
        PsiMcda.Bounds b = PsiMcda.zBoundsFromXBounds(2.0, 4.0, 0.0, 10.0, PsiMcda.Direction.BENEFIT);
        PsiMcda.Bounds c = PsiMcda.zBoundsFromXBounds(2.0, 4.0, 0.0, 10.0, PsiMcda.Direction.COST);
        return close(b.lower(), 0.2) && close(b.upper(), 0.4) && close(c.lower(), 0.6) && close(c.upper(), 0.8);
    }

    static boolean testGradPsiSigns() {
        PsiMcda.GradPsi g = PsiMcda.gradPsi(0.4, 0.8, 0.5, 0.2, 0.1, 0.5, 0.3, 1.0);
        return g.dAlpha() < 0.0 && g.dRa() < 0.0 && g.dRv() < 0.0 && g.dS() > 0.0 && g.dN() > 0.0;
    }

    static boolean testTieBreak() {
        class A { final double u, psi, cost; A(double u,double p,double c){ this.u=u; this.psi=p; this.cost=c; } }
        List<A> xs = Arrays.asList(new A(1.0, 0.7, 5.0), new A(0.9, 0.9, 1.0));
        Optional<A> best = PsiMcda.tieBreak(xs, a -> a.u, a -> a.psi, a -> a.cost);
        return best.isPresent() && close(best.get().u, 1.0);
    }

    // helpers
    static boolean inRange01(double v) { return v >= -1e-12 && v <= 1.0 + 1e-12; }
    static boolean allIn01(double[] z) { for (double v: z) if (!inRange01(v)) return false; return true; }
    static boolean close(double a, double b) { return Math.abs(a - b) <= 1e-9; }
}


