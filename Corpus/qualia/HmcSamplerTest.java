// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal sanity tests for HMC dual-averaging and mass estimation. Run via `java qualia.HmcSamplerTest`.
 */
public final class HmcSamplerTest {
    public static void main(String[] args) {
        System.out.println("HmcSamplerTest: " + (testDualAveragingConverges() && testMassEstimationPositive()) );
    }

    static boolean testDualAveragingConverges() {
        List<ClaimData> dataset = syntheticDataset(80, 123);
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        HmcSampler hmc = new HmcSampler(model, dataset);
        double[] z0 = new double[] { logit(0.7), logit(0.6), logit(0.5), Math.log(1.0) };
        HmcSampler.AdaptiveResult res = hmc.sampleAdaptive(600, 400, 2, 42L, z0, 0.02, 20, 0.75);
        return res.tunedStepSize > 1e-4 && res.tunedStepSize < 0.3 && res.acceptanceRate > 0.55 && res.acceptanceRate < 0.95;
    }

    static boolean testMassEstimationPositive() {
        List<ClaimData> dataset = syntheticDataset(80, 321);
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        HmcSampler hmc = new HmcSampler(model, dataset);
        double[] z0 = new double[] { logit(0.6), logit(0.4), logit(0.5), Math.log(1.2) };
        HmcSampler.AdaptiveResult res = hmc.sampleAdaptive(400, 200, 2, 7L, z0, 0.03, 15, 0.7);
        for (double m : res.massDiag) if (!(m > 0.0) || !Double.isFinite(m)) return false;
        return true;
    }

    private static List<ClaimData> syntheticDataset(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        List<ClaimData> ds = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ds.add(new ClaimData("t-"+i, rng.nextBoolean(), Math.abs(rng.nextGaussian())*0.4, Math.abs(rng.nextGaussian())*0.4,
                    Math.min(1.0, Math.max(0.0, 0.5 + 0.2*rng.nextGaussian()))));
        }
        return ds;
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double c = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(c / (1.0 - c));
    }
}


