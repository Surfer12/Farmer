package qualia;

/**
 * Hyperparameters for prior distributions.
 *
 * <p>Intended as an immutable container for Beta, LogNormal, and Gamma shapes
 * and scales used by the hierarchical model.
 */
public record ModelPriors(
        double s_alpha,
        double s_beta,
        double n_alpha,
        double n_beta,
        double alpha_alpha,
        double alpha_beta,
        double beta_mu,
        double beta_sigma,
        double ra_shape,
        double ra_scale,
        double rv_shape,
        double rv_scale,
        double lambda1,
        double lambda2
) {
    /**
     * Returns a reasonable set of weakly-informative defaults.
     */
    public static ModelPriors defaults() {
        return new ModelPriors(
                1.0, 1.0, // S ~ Beta(1,1)
                1.0, 1.0, // N ~ Beta(1,1)
                1.0, 1.0, // alpha ~ Beta(1,1)
                0.0, 1.0, // beta ~ LogNormal(mu=0, sigma=1)
                1.0, 1.0, // R_a ~ Gamma(k=1, theta=1)
                1.0, 1.0, // R_v ~ Gamma(k=1, theta=1)
                0.1, 0.1  // penalty weights λ1, λ2
        );
    }
}
