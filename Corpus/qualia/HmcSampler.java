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
    private final HierarchicalBayesianModel.Prepared prep;
    private final boolean parallel;

    public HmcSampler(HierarchicalBayesianModel model, List<ClaimData> dataset) {
        this.model = model;
        this.dataset = dataset;
        this.prep = model.precompute(dataset);
        this.parallel = model.shouldParallelize(dataset.size());
    }

    public static final class Result {
        public final List<ModelParameters> samples;
        public final double acceptanceRate;
        public Result(List<ModelParameters> samples, double acceptanceRate) {
            this.samples = samples;
            this.acceptanceRate = acceptanceRate;
        }
    }

    public static final class AdaptiveResult {
        public final List<ModelParameters> samples;
        public final double acceptanceRate;
        public final double tunedStepSize;
        public final double[] massDiag;
        public final int divergenceCount;
        public AdaptiveResult(List<ModelParameters> samples,
                              double acceptanceRate,
                              double tunedStepSize,
                              double[] massDiag,
                              int divergenceCount) {
            this.samples = samples;
            this.acceptanceRate = acceptanceRate;
            this.tunedStepSize = tunedStepSize;
            this.massDiag = massDiag;
            this.divergenceCount = divergenceCount;
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

        double[] massDiag = new double[] {1.0, 1.0, 1.0, 1.0};
        for (int iter = 0; iter < totalIters; iter++) {
            // Momentum p ~ N(0, I)
            double[] p = new double[4];
            for (int i = 0; i < 4; i++) p[i] = rng.nextGaussian() * Math.sqrt(massDiag[i]);

            // Cache current state
            double[] zCur = z.clone();
            double[] pCur = p.clone();

            // Compute initial grad and Hamiltonian
            double logT0 = logTarget(z);
            double H0 = -logT0 + 0.5 * dotInvMass(p, massDiag);

            // Leapfrog
            double[] grad = gradLogTarget(z);
            for (int l = 0; l < leapfrogSteps; l++) {
                // p_{t+1/2} = p_t + (ε/2) * ∇ logTarget(z_t)
                axpy(stepSize * 0.5, grad, p);
                // z_{t+1} = z_t + ε * p_{t+1/2}
                axpyWithInvMass(stepSize, p, massDiag, z);
                // p_{t+1} = p_{t+1/2} + (ε/2) * ∇ logTarget(z_{t+1})
                grad = gradLogTarget(z);
                axpy(stepSize * 0.5, grad, p);
            }

            // Metropolis accept
            double logT1 = logTarget(z);
            double H1 = -logT1 + 0.5 * dotInvMass(p, massDiag);
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

    /**
     * Adaptive HMC with three warmup phases: step-size find, dual-averaging of ε, and diagonal mass estimation.
     * After warmup, samples are drawn using tuned ε and massDiag.
     */
    public AdaptiveResult sampleAdaptive(int warmupIters,
                                         int samplingIters,
                                         int thin,
                                         long seed,
                                         double[] z0,
                                         double initStepSize,
                                         int leapfrogSteps,
                                         double targetAccept) {
        Random rng = new Random(seed);
        double[] z = z0.clone();
        // Mass diag initialised to ones
        double[] massDiag = new double[] {1.0, 1.0, 1.0, 1.0};

        int findWindow = Math.max(10, (int) Math.round(warmupIters * 0.15));
        int adaptWindow = Math.max(10, (int) Math.round(warmupIters * 0.60));
        int massWindow = Math.max(10, warmupIters - findWindow - adaptWindow);
        int phase1End = findWindow;
        int phase2End = findWindow + adaptWindow;

        double mu = Math.log(10.0 * Math.max(1e-6, initStepSize));
        double logEps = Math.log(Math.max(1e-6, initStepSize));
        double h = 0.0;
        double t0 = 10.0;
        double kappa = 0.75;        // weights decay
        double gamma = 0.25;         // shrink factor (higher -> smaller updates)
        double logEpsBar = 0.0;
        int tCount = 0;

        double[] meanZ = new double[4];
        double[] m2Z = new double[4];
        int massN = 0;
        int divergenceCount = 0;
        double divThreshold = getEnvDouble("HMC_DIVERGENCE_THRESHOLD", 50.0);

        // Warmup loop
        for (int iter = 0; iter < warmupIters; iter++) {
            // Sample momentum
            double[] p = new double[4];
            for (int i = 0; i < 4; i++) p[i] = rng.nextGaussian() * Math.sqrt(massDiag[i]);

            double[] zCur = z.clone();
            double[] pCur = p.clone();

            double logT0 = logTarget(z);
            double H0 = -logT0 + 0.5 * dotInvMass(p, massDiag);

            double[] grad = gradLogTarget(z);
            double eps = Math.exp(logEps);
            int L = Math.max(1, leapfrogSteps);
            for (int l = 0; l < L; l++) {
                axpy(eps * 0.5, grad, p);
                axpyWithInvMass(eps, p, massDiag, z);
                grad = gradLogTarget(z);
                axpy(eps * 0.5, grad, p);
            }

            double logT1 = logTarget(z);
            double H1 = -logT1 + 0.5 * dotInvMass(p, massDiag);
            double acceptProb = Math.min(1.0, Math.exp(H0 - H1));
            boolean accept = rng.nextDouble() < acceptProb;
            if (!accept) {
                z = zCur;
            }

            // Divergence check: large energy error or NaN
            if (!Double.isFinite(H0) || !Double.isFinite(H1) || (H1 - H0) > divThreshold) {
                divergenceCount++;
            }

            // Phase-specific adaptation
            if (iter < phase1End) {
                // Coarse step-size finder around target acceptance
                double low = Math.max(0.10, targetAccept * 0.9);
                double high = Math.min(0.95, targetAccept * 1.1);
                if (acceptProb < low) {
                    // too low acceptance -> decrease step
                    logEps -= 0.1;
                } else if (acceptProb > high) {
                    // too high acceptance -> increase step
                    logEps += 0.1;
                }
            } else if (iter < phase2End) {
                // Dual-averaging towards targetAccept
                tCount++;
                double eta = 1.0 / (tCount + t0);
                h = (1.0 - eta) * h + eta * (targetAccept - acceptProb);
                double logEpsRaw = mu - (Math.sqrt(tCount) / gamma) * h;
                double w = Math.pow(tCount, -kappa);
                logEps = (1.0 - w) * logEps + w * logEpsRaw;
                logEpsBar = (tCount == 1) ? logEps : ((tCount - 1) / (double) tCount) * logEpsBar + (1.0 / tCount) * logEps;
            } else {
                // Accumulate z variance for diag mass
                massN++;
                for (int i = 0; i < 4; i++) {
                    double delta = z[i] - meanZ[i];
                    meanZ[i] += delta / massN;
                    double delta2 = z[i] - meanZ[i];
                    m2Z[i] += delta * delta2;
                }
            }
        }

        // Final tuned ε and mass diag
        double tunedStep = Math.exp((tCount > 0) ? logEpsBar : logEps);
        // Clamp tuned step into a reasonable band
        tunedStep = Math.max(1e-4, Math.min(0.25, tunedStep));
        double[] tunedMass = massDiag;
        if (massN > 10) {
            tunedMass = new double[4];
            for (int i = 0; i < 4; i++) {
                double var = m2Z[i] / Math.max(1, massN - 1);
                tunedMass[i] = Math.max(1e-6, var);
            }
        }

        // Sampling with tuned parameters
        List<ModelParameters> kept = new ArrayList<>();
        int accepted = 0;
        int keepEvery = Math.max(1, thin);
        int keptCount = 0;
        for (int iter = 0; iter < samplingIters; iter++) {
            double[] p = new double[4];
            for (int i = 0; i < 4; i++) p[i] = rng.nextGaussian() * Math.sqrt(tunedMass[i]);
            double[] zCur = z.clone();
            double logT0 = logTarget(z);
            double H0 = -logT0 + 0.5 * dotInvMass(p, tunedMass);

            double[] grad = gradLogTarget(z);
            int L = Math.max(1, leapfrogSteps);
            for (int l = 0; l < L; l++) {
                axpy(tunedStep * 0.5, grad, p);
                axpyWithInvMass(tunedStep, p, tunedMass, z);
                grad = gradLogTarget(z);
                axpy(tunedStep * 0.5, grad, p);
            }
            double logT1 = logTarget(z);
            double H1 = -logT1 + 0.5 * dotInvMass(p, tunedMass);

            double acceptProb = Math.min(1.0, Math.exp(H0 - H1));
            if (rng.nextDouble() < acceptProb) {
                accepted++;
            } else {
                z = zCur;
            }

            if ((iter % keepEvery) == 0) {
                kept.add(zToParams(z));
                keptCount++;
            }
        }

        double accRate = accepted / (double) Math.max(1, samplingIters);
        return new AdaptiveResult(kept, accRate, tunedStep, tunedMass, divergenceCount);
    }

    private static double getEnvDouble(String key, double fallback) {
        try {
            String v = System.getenv(key);
            if (v == null || v.isEmpty()) return fallback;
            return Double.parseDouble(v);
        } catch (Exception e) {
            return fallback;
        }
    }

    /** logTarget(z) = logPosterior(θ(z)) + log|J(z)| */
    private double logTarget(double[] z) {
        ModelParameters params = zToParams(z);
        double logPost = model.logPosteriorPrepared(prep, params, parallel);
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
        // Analytic gradient via chain rule: ∇_z logPost(θ(z)) + ∇_z log|J|
        ModelParameters p = zToParams(z);
        double[] dLogPost_dTheta = model.gradientLogPosteriorPrepared(prep, p, parallel);
        double S = p.S(), N = p.N(), A = p.alpha(), B = p.beta();

        // Jacobians dθ/dz
        double dSdz = S * (1.0 - S);
        double dNdz = N * (1.0 - N);
        double dAdz = A * (1.0 - A);
        double dBdz = B; // beta = exp(z3)

        // Chain rule for logPosterior term
        double g0 = dLogPost_dTheta[0] * dSdz; // d/dz0
        double g1 = dLogPost_dTheta[1] * dNdz; // d/dz1
        double g2 = dLogPost_dTheta[2] * dAdz; // d/dz2
        double g3 = dLogPost_dTheta[3] * dBdz; // d/dz3

        // Plus gradient of log|J| wrt z: for sigmoid, d/dz log(sig(1-sig)) = 1 - 2*sig(z)
        g0 += (1.0 - 2.0 * S);
        g1 += (1.0 - 2.0 * N);
        g2 += (1.0 - 2.0 * A);
        g3 += 1.0; // d/dz log exp(z) = 1

        return new double[] { g0, g1, g2, g3 };
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
    private static void axpyWithInvMass(double eps, double[] p, double[] massDiag, double[] z) {
        for (int i = 0; i < p.length; i++) z[i] += eps * (p[i] / Math.max(1e-12, massDiag[i]));
    }
    private static double dot(double[] a, double[] b) {
        double s = 0.0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
    }
    private static double dotInvMass(double[] p, double[] massDiag) {
        double s = 0.0; for (int i = 0; i < p.length; i++) s += (p[i] * p[i]) / Math.max(1e-12, massDiag[i]); return s;
    }
}


