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

    /** Returns ∇ log p(x) where x∈R^4 corresponds to ModelParameters. */
    double[] gradLogPosterior(double[] x) {
        // Interpret x as z-coordinates: z = [logit(S), logit(N), logit(alpha), log(beta)]
        return model.gradLogTargetZ(prep, x, parallel);
    }
}


