// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

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
}


