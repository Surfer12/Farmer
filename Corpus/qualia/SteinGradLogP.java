// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;


/**
 * Provides ∇ log p(x) for a posterior over parameters X given dataset Y.
 * For this scaffold, we approximate d=|theta| gradient using finite differences
 * around a ModelParameters vector mapped to R^4 as [S,N,alpha,beta].
 */
final class SteinGradLogP {
    private final HierarchicalBayesianModel model;
    private final java.util.List<ClaimData> dataset;
    private final HierarchicalBayesianModel.Prepared prep;
    private final boolean parallel;

    SteinGradLogP(HierarchicalBayesianModel model, java.util.List<ClaimData> dataset) {
        this.model = model;
        this.dataset = dataset;
        this.prep = model.precompute(dataset);
        this.parallel = model.shouldParallelize(dataset.size());
    }

    private static ModelParameters toParams(double[] x) {
        double S = clamp01(x[0]);
        double N = clamp01(x[1]);
        double a = clamp01(x[2]);
        double b = Math.max(1e-6, x[3]);
        return new ModelParameters(S, N, a, b);
    }

    private static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    /** Returns ∇ log p(x) where x∈R^4 corresponds to ModelParameters. */
    double[] gradLogPosterior(double[] x) {
        // x is in parameter space θ = [S, N, alpha, beta]
        return model.gradientLogPosteriorPrepared(prep, toParams(x), parallel);
    }
}


