// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal Hamiltonian Monte Carlo (HMC) sampler over an unconstrained
 * reparameterization z ∈ R^4 mapped to model parameters θ = (S,N,alpha,beta):
 *   S = sigmoid(z0), N = sigmoid(z1), alpha = sigmoid(z2), beta = exp(z3).
 *
 * Target density in z includes the transform Jacobian:
 *   logTarget(z) = logPosterior(θ(z)) + log|J(z)|, where
 *   log|J| = log(sig(z0)(1-sig(z0)))
 *          + log(sig(z1)(1-sig(z1)))
 *          + log(sig(z2)(1-sig(z2)))
 *          + z3.
 *
 * Gradients are estimated via finite differences on logTarget(z) for clarity
 * and zero-dependency simplicity. For production use, provide analytic
 * gradients or AD for better performance and stability.
 */
final class HmcSampler {
    private final HierarchicalBayesianModel model;
    private final List<ClaimData> dataset;

    public HmcSampler(HierarchicalBayesianModel model, List<ClaimData> dataset) {
        this.model = model;
        this.dataset = dataset;
    }

    public static final class Result {
        public final List<ModelParameters> samples;
        public final double acceptanceRate;
        public Result(List<ModelParameters> samples, double acceptanceRate) {
            this.samples = samples;
            this.acceptanceRate = acceptanceRate;
        }
    }

    /** Samples using HMC with identity mass matrix. */
    public Result sample(int totalIters,
                         int burnIn,
                         int thin,
                         long seed,
                         double[] z0,
                         double stepSize,
                         int leapfrogSteps) {
        if (totalIters <= 0) return new Result(List.of(), 0.0);
        Random rng = new Random(seed);
        double[] z = z0.clone();
        List<ModelParameters> kept = new ArrayList<>();
        int accepted = 0;

        for (int iter = 0; iter < totalIters; iter++) {
            // Momentum p ~ N(0, I)
            double[] p = new double[4];
            for (int i = 0; i < 4; i++) p[i] = rng.nextGaussian();

            // Cache current state
            double[] zCur = z.clone();
            double[] pCur = p.clone();

            // Compute initial grad and Hamiltonian
            double logT0 = logTarget(z);
            double H0 = -logT0 + 0.5 * dot(p, p);

            // Leapfrog
            double[] grad = gradLogTarget(z);
            for (int l = 0; l < leapfrogSteps; l++) {
                // p_{t+1/2} = p_t + (ε/2) * ∇ logTarget(z_t)
                axpy(stepSize * 0.5, grad, p);
                // z_{t+1} = z_t + ε * p_{t+1/2}
                axpy(stepSize, p, z);
                // p_{t+1} = p_{t+1/2} + (ε/2) * ∇ logTarget(z_{t+1})
                grad = gradLogTarget(z);
                axpy(stepSize * 0.5, grad, p);
            }

            // Metropolis accept
            double logT1 = logTarget(z);
            double H1 = -logT1 + 0.5 * dot(p, p);
            double acceptProb = Math.min(1.0, Math.exp(H0 - H1));
            if (rng.nextDouble() < acceptProb) {
                accepted++;
            } else {
                // reject
                z = zCur;
            }

            // collect after burn-in/thinning
            if (iter >= burnIn && ((iter - burnIn) % Math.max(1, thin) == 0)) {
                kept.add(zToParams(z));
            }
        }

        double accRate = accepted / (double) totalIters;
        return new Result(kept, accRate);
    }

    /** logTarget(z) = logPosterior(θ(z)) + log|J(z)| */
    private double logTarget(double[] z) {
        ModelParameters params = zToParams(z);
        double logPost = model.logPosterior(dataset, params);
        // Jacobian terms
        double s0 = sigmoid(z[0]);
        double s1 = sigmoid(z[1]);
        double s2 = sigmoid(z[2]);
        double logJ = Math.log(s0 * (1.0 - s0))
                    + Math.log(s1 * (1.0 - s1))
                    + Math.log(s2 * (1.0 - s2))
                    + z[3]; // d/dz exp(z) = exp(z)
        return logPost + logJ;
    }

    private double[] gradLogTarget(double[] z) {
        double[] g = new double[4];
        double eps = 1e-5;
        double base = logTarget(z);
        for (int i = 0; i < 4; i++) {
            double old = z[i];
            z[i] = old + eps; double up = logTarget(z);
            z[i] = old - eps; double dn = logTarget(z);
            z[i] = old;
            g[i] = (up - dn) / (2.0 * eps);
        }
        return g;
    }

    private static ModelParameters zToParams(double[] z) {
        double S = sigmoid(z[0]);
        double N = sigmoid(z[1]);
        double a = sigmoid(z[2]);
        double b = Math.exp(z[3]);
        return new ModelParameters(S, N, a, b);
    }

    private static double sigmoid(double x) {
        if (x >= 0) {
            double ex = Math.exp(-x);
            return 1.0 / (1.0 + ex);
        } else {
            double ex = Math.exp(x);
            return ex / (1.0 + ex);
        }
    }

    private static void axpy(double a, double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) y[i] += a * x[i];
    }
    private static double dot(double[] a, double[] b) {
        double s = 0.0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
    }
}


