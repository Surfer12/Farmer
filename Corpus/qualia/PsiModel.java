// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.List;

/**
 * Minimal model interface exposing Ψ scoring and log-posterior evaluation.
 */
public interface PsiModel {
    double calculatePsi(ClaimData claim, ModelParameters params);
    double logLikelihood(ClaimData claim, ModelParameters params);
    double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params);
    double logPriors(ModelParameters params);
    double logPosterior(List<ClaimData> dataset, ModelParameters params);
    boolean shouldParallelize(int datasetSize);

    // Optional advanced API (default no-op) to allow samplers to use adaptive/z-space paths
    default List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount) { return List.of(); }
    default HmcSampler.AdaptiveResult hmcAdaptive(List<ClaimData> dataset,
                                                  int warmupIters,
                                                  int samplingIters,
                                                  int thin,
                                                  long seed,
                                                  double[] z0,
                                                  double initStepSize,
                                                  int leapfrogSteps,
                                                  double targetAccept) {
        HmcSampler h = new HmcSampler((HierarchicalBayesianModel) this, dataset);
        return h.sampleAdaptive(warmupIters, samplingIters, thin, seed, z0, initStepSize, leapfrogSteps, targetAccept);
    }
}


