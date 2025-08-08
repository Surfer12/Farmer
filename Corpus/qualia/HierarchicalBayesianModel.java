package qualia;

import java.util.List;
import java.util.Objects;

/**
 * Core model implementation for Hierarchical Bayesian inference.
 */
public final class HierarchicalBayesianModel {
    private final ModelPriors priors;

    public HierarchicalBayesianModel() {
        this(ModelPriors.defaults());
    }

    public HierarchicalBayesianModel(ModelPriors priors) {
        this.priors = Objects.requireNonNull(priors, "priors must not be null");
    }

    /**
     * Calculates Ψ(x) for a given claim and parameter sample.
     * Result is clamped to [0,1] for numerical stability.
     */
    public double calculatePsi(ClaimData claim, ModelParameters params) {
        Objects.requireNonNull(claim, "claim must not be null");
        Objects.requireNonNull(params, "params must not be null");

        double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        double penaltyExponent = -(
                priors.lambda1() * claim.riskAuthenticity() +
                priors.lambda2() * claim.riskVirality()
        );
        double pen = Math.exp(penaltyExponent);
        double p_H_given_E_beta = Math.min(params.beta() * claim.probabilityHgivenE(), 1.0);
        double psi = O * pen * p_H_given_E_beta;
        if (psi < 0.0) return 0.0;
        if (psi > 1.0) return 1.0;
        return psi;
    }

    /**
     * log-likelihood for a single observation: y | Ψ ~ Bernoulli(Ψ).
     */
    public double logLikelihood(ClaimData claim, ModelParameters params) {
        double psi = calculatePsi(claim, params);
        double epsilon = 1e-9;
        double clamped = Math.max(epsilon, Math.min(1.0 - epsilon, psi));
        return claim.isVerifiedTrue() ? Math.log(clamped) : Math.log(1.0 - clamped);
    }

    /**
     * Sum log-likelihood over a dataset.
     */
    public double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.stream()
                .mapToDouble(claim -> logLikelihood(claim, params))
                .sum();
    }

    /**
     * Log-priors for a given parameter draw. Placeholder (returns 0.0).
     * In a full implementation, provide log-PDFs for Beta, LogNormal, etc.
     */
    public double logPriors(ModelParameters params) {
        // Placeholder: treat as uniform for now
        return 0.0;
    }

    /**
     * Log-posterior up to an additive constant: logLik + logPrior.
     */
    public double logPosterior(List<ClaimData> dataset, ModelParameters params) {
        return totalLogLikelihood(dataset, params) + logPriors(params);
    }

    /**
     * Posterior inference via MCMC (placeholder).
     *
     * <p>Integrate with a Java sampling library if needed.
     */
    public List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount) {
        System.out.println("Starting inference (placeholder)...");
        // TODO: Integrate HMC/NUTS or other samplers
        System.out.println("Inference placeholder finished.");
        return List.of();
    }
}
