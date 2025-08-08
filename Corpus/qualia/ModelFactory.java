// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

/**
 * Simple factory for constructing model instances with dependency injection.
 * In larger projects, replace with a DI container; here we keep a tiny wire-up.
 */
public interface ModelFactory {
    PsiModel createPsiModel(ModelPriors priors, int parallelThreshold);

    static ModelFactory defaultFactory() {
        return (priors, threshold) -> new HierarchicalBayesianModel(priors, threshold);
    }
}


