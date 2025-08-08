// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal smoke tests for HMC determinism and basic diagnostics.
 * Run via: java -cp out-qualia qualia.HmcSmokeTests
 */
public final class HmcSmokeTests {
    public static void main(String[] args) {
        boolean ok = true;

        // dataset
        List<ClaimData> dataset = new ArrayList<>();
        Random rng = new Random(101);
        for (int i = 0; i < 40; i++) {
            dataset.add(new ClaimData("t-" + i, rng.nextBoolean(),
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()))));
        }

        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        HmcSampler hmc = new HmcSampler(model, dataset);
        double[] z0 = new double[] { logit(0.7), logit(0.6), logit(0.5), Math.log(1.0) };

        // determinism with fixed seeds
        HmcSampler.AdaptiveResult a1 = hmc.sampleAdaptive(500, 1000, 2, 777L, z0, 0.02, 20, 0.75);
        HmcSampler.AdaptiveResult a2 = hmc.sampleAdaptive(500, 1000, 2, 777L, z0, 0.02, 20, 0.75);
        if (a1.samples.size() != a2.samples.size()) { System.err.println("size mismatch"); ok = false; }
        int n = Math.min(a1.samples.size(), a2.samples.size());
        for (int i = 0; i < n; i++) {
            ModelParameters p = a1.samples.get(i);
            ModelParameters q = a2.samples.get(i);
            if (!almostEq(p, q, 1e-12)) { System.err.println("determinism mismatch at " + i); ok = false; break; }
        }

        // basic diagnostics across two distinct chains
        HmcSampler.AdaptiveResult b1 = hmc.sampleAdaptive(500, 1000, 2, 1001L, z0, 0.02, 20, 0.75);
        HmcSampler.AdaptiveResult b2 = hmc.sampleAdaptive(500, 1000, 2, 2002L, z0, 0.02, 20, 0.75);
        List<List<ModelParameters>> chains = List.of(b1.samples, b2.samples);
        Diagnostics d = model.diagnose(chains);
        if (!(d.rHatS > 0.8 && d.rHatS < 1.2)) { System.err.println("rHatS out of band: " + d.rHatS); ok = false; }
        if (!(d.essS > 50)) { System.err.println("ESS too low: " + d.essS); ok = false; }

        System.out.println(ok ? "SMOKE OK" : "SMOKE FAIL");
        if (!ok) System.exit(1);
    }

    private static boolean almostEq(ModelParameters a, ModelParameters b, double eps) {
        return Math.abs(a.S() - b.S()) < eps
                && Math.abs(a.N() - b.N()) < eps
                && Math.abs(a.alpha() - b.alpha()) < eps
                && Math.abs(a.beta() - b.beta()) < eps;
    }

    private static double logit(double x) {
        double e = 1e-12; double c = Math.max(e, Math.min(1 - e, x));
        return Math.log(c / (1 - c));
    }
}


