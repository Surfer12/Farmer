// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal RMALA sampler with position-dependent step size policy and CDLB reward metric.
 *
 * - Dimension fixed to 4 via mapping: x = [S, N, alpha, beta]
 * - Preconditioner G0 = I (identity) for simplicity
 * - Gradients computed via {@link SteinGradLogP}
 */
public final class RmalaSampler {
    @FunctionalInterface
    public interface StepSizePolicy {
        double stepSize(double[] x);
    }

    public static final class Result {
        public final List<ModelParameters> samples;
        public final double acceptanceRate;
        public final double avgCdlb;
        public Result(List<ModelParameters> samples, double acceptanceRate, double avgCdlb) {
            this.samples = samples;
            this.acceptanceRate = acceptanceRate;
            this.avgCdlb = avgCdlb;
        }
    }

    private final HierarchicalBayesianModel model;
    private final List<ClaimData> dataset;
    private final SteinGradLogP gradProvider;
    private final HierarchicalBayesianModel.Prepared prep;
    private final boolean parallel;

    public RmalaSampler(HierarchicalBayesianModel model, List<ClaimData> dataset) {
        this.model = model;
        this.dataset = dataset;
        this.gradProvider = new SteinGradLogP(model, dataset);
        this.prep = model.precompute(dataset);
        this.parallel = model.shouldParallelize(dataset.size());
    }

    public Result sample(int totalIters, int burnIn, int thin, long seed, double[] x0, StepSizePolicy policy) {
        Random rng = new Random(seed);
        double[] x = x0.clone();

        List<ModelParameters> kept = new ArrayList<>();
        int accepted = 0;
        double sumCdlb = 0.0;

        for (int iter = 0; iter < totalIters; iter++) {
            // Current stats
            double eps = policy.stepSize(x);
            double[] grad = gradProvider.gradLogPosterior(x.clone());
            double[] mu = add(x, scale(grad, eps));
            double[] z = new double[4];
            for (int i = 0; i < 4; i++) z[i] = rng.nextGaussian();
            double[] xProp = add(mu, scale(z, Math.sqrt(2.0 * eps)));

            // Reverse proposal terms
            double epsProp = policy.stepSize(xProp);
            double[] gradProp = gradProvider.gradLogPosterior(xProp.clone());
            double[] muProp = add(xProp, scale(gradProp, epsProp));

            // Log posterior
            double lpCur = logPosterior(x);
            double lpProp = logPosterior(xProp);

            // q densities (log) with diag covariance 2*eps I
            double lqForward = logGaussianDiag(xProp, mu, 2.0 * eps);
            double lqReverse = logGaussianDiag(x, muProp, 2.0 * epsProp);

            // MH acceptance
            double logRatio = (lpProp - lpCur) + (lqReverse - lqForward);
            double alpha = Math.min(1.0, Math.exp(logRatio));

            if (rng.nextDouble() < alpha) {
                x = xProp;
                accepted++;
            }

            // CDLB reward (Rao-Blackwellised form)
            double a = clamp(alpha, 1e-12, 1.0 - 1e-12);
            double cdlb = a * (lpProp - lpCur)
                    + ( - a * Math.log(a) - (1.0 - a) * Math.log(1.0 - a) )
                    + ( - a * lqForward );
            sumCdlb += cdlb;

            // Collect
            if (iter >= burnIn && ((iter - burnIn) % thin == 0)) {
                kept.add(toParams(x));
            }
        }

        double accRate = accepted / (double) totalIters;
        double avgCdlb = sumCdlb / Math.max(1, totalIters);
        return new Result(kept, accRate, avgCdlb);
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private double logPosterior(double[] x) {
        return model.logPosteriorPrepared(prep, toParams(x), parallel);
    }

    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    private static double[] scale(double[] a, double s) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * s;
        return r;
    }

    private static double logGaussianDiag(double[] x, double[] mean, double var) {
        // diag covariance var * I (var > 0)
        int d = x.length;
        double logDet = d * Math.log(var);
        double quad = 0.0;
        double inv = 1.0 / var;
        for (int i = 0; i < d; i++) {
            double diff = x[i] - mean[i];
            quad += diff * diff * inv;
        }
        return -0.5 * (d * Math.log(2.0 * Math.PI) + logDet + quad);
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
}


