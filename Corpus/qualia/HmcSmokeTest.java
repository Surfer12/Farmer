// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal smoke test for deterministic multi-chain HMC and basic diagnostics bounds.
 * Exits with non-zero code on failure; prints brief results on success.
 */
public final class HmcSmokeTest {
    public static void main(String[] args) {
        int assertions = 0;
        try {
            // 1) Synthetic dataset (fixed RNG)
            List<ClaimData> dataset = new ArrayList<>();
            java.util.Random rng = new java.util.Random(123);
            int n = 60;
            for (int i = 0; i < n; i++) {
                dataset.add(new ClaimData(
                        "t-" + i,
                        rng.nextBoolean(),
                        Math.abs(rng.nextGaussian()) * 0.3,
                        Math.abs(rng.nextGaussian()) * 0.3,
                        Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()))
                ));
            }

            // 2) Model
            HierarchicalBayesianModel model = new HierarchicalBayesianModel();

            // 3) Config (fixed seeds)
            int chains = 2;
            int warm = 400;
            int iters = 1200;
            int thin = 3;
            long baseSeed = 13579L;
            double[] z0 = new double[] { logit(0.7), logit(0.6), logit(0.5), Math.log(1.0) };
            double eps0 = 0.01;
            int leap = 24;
            double target = 0.75;

            // 4) Run twice (no disk persistence to avoid FS flakiness)
            HmcMultiChainRunner r1 = new HmcMultiChainRunner(model, dataset, chains, warm, iters, thin, baseSeed, z0, eps0, leap, target, null);
            HmcMultiChainRunner r2 = new HmcMultiChainRunner(model, dataset, chains, warm, iters, thin, baseSeed, z0, eps0, leap, target, null);
            HmcMultiChainRunner.Summary s1 = r1.run();
            HmcMultiChainRunner.Summary s2 = r2.run();

            // 5) Determinism: per-chain acceptance, divergences, sample counts match closely
            if (s1.chains.size() != s2.chains.size()) throw new AssertionError("chain count mismatch"); assertions++;
            for (int i = 0; i < s1.chains.size(); i++) {
                var c1 = s1.chains.get(i);
                var c2 = s2.chains.get(i);
                if (c1.samples.size() != c2.samples.size()) throw new AssertionError("samples size mismatch at chain " + i); assertions++;
                if (!close(c1.acceptanceRate, c2.acceptanceRate, 1e-12)) throw new AssertionError("acceptance mismatch at chain " + i); assertions++;
                if (c1.divergences != c2.divergences) throw new AssertionError("divergences mismatch at chain " + i); assertions++;
                if (!close(c1.tunedStepSize, c2.tunedStepSize, 1e-12)) throw new AssertionError("eps* mismatch at chain " + i); assertions++;
            }

            // 6) Diagnostics sanity
            Diagnostics d = s1.diagnostics;
            // R-hat near 1 (loose bounds for smoke test)
            if (!within(d.rHatS, 0.8, 1.2)) throw new AssertionError("rHatS out of range: " + d.rHatS); assertions++;
            if (!within(d.rHatN, 0.8, 1.2)) throw new AssertionError("rHatN out of range: " + d.rHatN); assertions++;
            if (!within(d.rHatAlpha, 0.8, 1.2)) throw new AssertionError("rHatAlpha out of range: " + d.rHatAlpha); assertions++;
            if (!within(d.rHatBeta, 0.8, 1.2)) throw new AssertionError("rHatBeta out of range: " + d.rHatBeta); assertions++;
            // ESS positive
            if (!(d.essS > 0 && d.essN > 0 && d.essAlpha > 0 && d.essBeta > 0)) throw new AssertionError("ESS non-positive"); assertions++;

            System.out.println("HmcSmokeTest: OK (" + assertions + " checks)" );
        } catch (Throwable t) {
            t.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static boolean close(double a, double b, double tol) {
        return Math.abs(a - b) <= tol;
    }

    private static boolean within(double x, double lo, double hi) {
        return x >= lo && x <= hi;
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double c = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(c / (1.0 - c));
    }
}


