package jumping.qualia;

/**
 * Represents a single sample of model parameters from the posterior.
 */
public record ModelParameters(
        double S,
        double N,
        double alpha,
        double beta
) {
    public ModelParameters {
        if (S < 0.0 || S > 1.0) {
            throw new IllegalArgumentException("S must be in [0,1]");
        }
        if (N < 0.0 || N > 1.0) {
            throw new IllegalArgumentException("N must be in [0,1]");
        }
        if (alpha < 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("alpha must be in [0,1]");
        }
        if (beta <= 0.0) {
            throw new IllegalArgumentException("beta must be > 0");
        }
    }
}
