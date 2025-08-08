// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadLocalRandom;
import java.security.SecureRandom;
import java.util.stream.IntStream;

/**
 * Core model implementation for Hierarchical Bayesian inference.
 *
 * <p>Defines Ψ using the affine+exponential+cap structure:
 * O(α) = α·S + (1−α)·N,
 * pen = exp(−[λ1·R_a + λ2·R_v]),
 * P(H|E, β) = min{β·P(H|E), 1},
 * Ψ = O · pen · P(H|E, β).
 *
 * <p>Thread-safety: stateless aside from immutable {@link ModelPriors}; safe for
 * concurrent use.
 */
public final class HierarchicalBayesianModel {
    private final ModelPriors priors;
    private final int parallelThreshold;

    /**
     * Constructs a model with default priors.
     */
    public HierarchicalBayesianModel() {
        this(ModelPriors.defaults(), 2048);
    }

    /**
     * Constructs a model with explicit priors.
     * @param priors hyperparameters (non-null)
     */
    public HierarchicalBayesianModel(ModelPriors priors) {
        this(priors, 2048);
    }

    /**
     * Constructs a model with explicit priors and a parallelization threshold.
     * If dataset size >= parallelThreshold, likelihood sums use parallel streams.
     */
    public HierarchicalBayesianModel(ModelPriors priors, int parallelThreshold) {
        this.priors = Objects.requireNonNull(priors, "priors must not be null");
        this.parallelThreshold = Math.max(0, parallelThreshold);
    }

    /**
     * Calculates Ψ(x) for a given claim and parameter sample.
     * Result is clamped to [0,1] for numerical stability.
     *
     * @param claim observed claim data (non-null)
     * @param params parameter draw (non-null)
     * @return Ψ in [0,1]
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
     *
     * @param claim observation
     * @param params parameter draw
     * @return log p(y|Ψ)
     */
    public double logLikelihood(ClaimData claim, ModelParameters params) {
        double psi = calculatePsi(claim, params);
        double epsilon = 1e-9;
        double clamped = Math.max(epsilon, Math.min(1.0 - epsilon, psi));
        return claim.isVerifiedTrue() ? Math.log(clamped) : Math.log(1.0 - clamped);
    }

    /**
     * Sum log-likelihood over a dataset.
     *
     * @param dataset list of observations
     * @param params parameter draw
     * @return Σ log p(y_i|Ψ_i)
     */
    public double totalLogLikelihood(List<ClaimData> dataset, ModelParameters params) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return dataset.stream()
                .mapToDouble(claim -> logLikelihood(claim, params))
                .sum();
    }

    // --- Prepared dataset fast-paths ---
    private static final class Prepared {
        final double[] pen;       // exp(-[λ1 Ra + λ2 Rv]) per-claim
        final double[] pHe;       // P(H|E) per-claim
        final boolean[] y;        // labels per-claim
        Prepared(double[] pen, double[] pHe, boolean[] y) {
            this.pen = pen; this.pHe = pHe; this.y = y;
        }
        int size() { return pen.length; }
    }

    private Prepared precompute(List<ClaimData> dataset) {
        int n = dataset.size();
        double[] pen = new double[n];
        double[] pHe = new double[n];
        boolean[] y = new boolean[n];
        final double l1 = priors.lambda1();
        final double l2 = priors.lambda2();
        for (int i = 0; i < n; i++) {
            ClaimData c = dataset.get(i);
            pen[i] = Math.exp(-(l1 * c.riskAuthenticity() + l2 * c.riskVirality()));
            pHe[i] = c.probabilityHgivenE();
            y[i] = c.isVerifiedTrue();
        }
        return new Prepared(pen, pHe, y);
    }

    private double totalLogLikelihoodPrepared(Prepared prep, ModelParameters params, boolean parallel) {
        final double O = params.alpha() * params.S() + (1.0 - params.alpha()) * params.N();
        final double beta = params.beta();
        final double epsilon = 1e-9;
        final int n = prep.size();

        if (!parallel) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double pHb = Math.min(beta * prep.pHe[i], 1.0);
                double psi = O * prep.pen[i] * pHb;
                // clamp
                double clamped = Math.max(epsilon, Math.min(1.0 - epsilon, psi));
                sum += prep.y[i] ? Math.log(clamped) : Math.log(1.0 - clamped);
            }
            return sum;
        }

        return IntStream.range(0, n).parallel().mapToDouble(i -> {
            double pHb = Math.min(beta * prep.pHe[i], 1.0);
            double psi = O * prep.pen[i] * pHb;
            double clamped = Math.max(epsilon, Math.min(1.0 - epsilon, psi));
            return prep.y[i] ? Math.log(clamped) : Math.log(1.0 - clamped);
        }).sum();
    }

    private double logPosteriorPrepared(Prepared prep, ModelParameters params, boolean parallel) {
        return totalLogLikelihoodPrepared(prep, params, parallel) + logPriors(params);
    }

    /**
     * Log-priors for a given parameter draw. Placeholder (returns 0.0).
     * In a full implementation, provide log-PDFs for Beta, LogNormal, etc.
     *
     * @param params parameter draw
     * @return log p(params)
     */
    public double logPriors(ModelParameters params) {
        double lpS = Stats.logBetaPdf(params.S(), priors.s_alpha(), priors.s_beta());
        double lpN = Stats.logBetaPdf(params.N(), priors.n_alpha(), priors.n_beta());
        double lpAlpha = Stats.logBetaPdf(params.alpha(), priors.alpha_alpha(), priors.alpha_beta());
        double lpBeta = Stats.logLogNormalPdf(params.beta(), priors.beta_mu(), priors.beta_sigma());
        return lpS + lpN + lpAlpha + lpBeta;
    }

    /**
     * Log-posterior up to an additive constant: logLik + logPrior.
     *
     * @param dataset observations
     * @param params parameter draw
     * @return log p(params|data) + C
     */
    public double logPosterior(List<ClaimData> dataset, ModelParameters params) {
        return totalLogLikelihood(dataset, params) + logPriors(params);
    }

    /**
     * Posterior inference via a simple Random-Walk Metropolis-Hastings prototype.
     *
     * <p>This is a minimal, dependency-free scaffold useful for tests and
     * baselines. For production use, prefer HMC/NUTS via a dedicated library.
     */
    public List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount) {
        long seed = ThreadLocalRandom.current().nextLong();
        System.out.println("performInference: seed=" + seed + ", samples=" + sampleCount);
        Prepared prep = precompute(dataset);
        boolean par = dataset.size() >= parallelThreshold;
        return runMhChain(prep, sampleCount, seed, par);
    }

    /**
     * Overload allowing explicit seed control for reproducibility.
     */
    public List<ModelParameters> performInference(List<ClaimData> dataset, int sampleCount, long seed) {
        System.out.println("performInference(explicit): seed=" + seed + ", samples=" + sampleCount);
        Prepared prep = precompute(dataset);
        boolean par = dataset.size() >= parallelThreshold;
        return runMhChain(prep, sampleCount, seed, par);
    }

    /**
     * Runs multiple independent MH chains in parallel and returns per-chain samples.
     */
    public List<List<ModelParameters>> performInferenceParallel(List<ClaimData> dataset, int chains, int samplesPerChain) {
        if (chains <= 0 || samplesPerChain <= 0) return List.of();
        // Choose a cryptographically-strong base seed to reduce collision risk across invocations
        long baseSeed = new SecureRandom().nextLong();
        System.out.println("performInferenceParallel: baseSeed=" + baseSeed + ", chains=" + chains + ", samplesPerChain=" + samplesPerChain);
        return performInferenceParallel(dataset, chains, samplesPerChain, baseSeed);
    }

    /**
     * Overload allowing an explicit baseSeed. Per-chain seeds are derived as
     * baseSeed + c * PHI64 where PHI64 is a large odd constant to avoid collisions.
     */
    public List<List<ModelParameters>> performInferenceParallel(List<ClaimData> dataset, int chains, int samplesPerChain, long baseSeed) {
        if (chains <= 0 || samplesPerChain <= 0) return List.of();
        final Prepared prep = precompute(dataset);
        final boolean par = dataset.size() >= parallelThreshold;
        int poolSize = Math.min(chains, Math.max(1, Runtime.getRuntime().availableProcessors()));
        ExecutorService pool = Executors.newFixedThreadPool(poolSize, r -> {
            Thread t = new Thread(r, "qualia-mh-");
            t.setDaemon(true);
            return t;
        });

        try {
            List<CompletableFuture<List<ModelParameters>>> futures = new ArrayList<>(chains);
            for (int c = 0; c < chains; c++) {
                // Use the 64-bit golden ratio odd constant for good spacing in Z/2^64Z
                final long PHI64 = 0x9E3779B97F4A7C15L;
                final long seed = baseSeed + (PHI64 * (long) c);
                futures.add(CompletableFuture.supplyAsync(() -> runMhChain(prep, samplesPerChain, seed, par), pool));
            }
            List<List<ModelParameters>> results = new ArrayList<>(chains);
            for (CompletableFuture<List<ModelParameters>> f : futures) {
                results.add(f.join());
            }
            return results;
        } finally {
            pool.shutdown();
            try { pool.awaitTermination(5, TimeUnit.SECONDS); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        }
    }

    /**
     * Convenience wrapper to compute convergence diagnostics from chains.
     */
    public Diagnostics diagnose(List<List<ModelParameters>> chains) {
        return Diagnostics.fromChains(chains);
    }

    /**
     * HMC-based posterior inference. Returns a single-chain list of samples.
     *
     * @param dataset observations (non-null)
     * @param totalIters total HMC iterations
     * @param burnIn number of warmup iterations to discard
     * @param thin keep 1 of every {@code thin} samples after burn-in
     * @param seed RNG seed
     * @param initial starting point for parameters
     * @param stepSize leapfrog step size
     * @param leapfrogSteps number of leapfrog steps per iteration
     * @return posterior samples as {@link ModelParameters}
     */
    public List<ModelParameters> performInferenceHmc(List<ClaimData> dataset,
                                                     int totalIters,
                                                     int burnIn,
                                                     int thin,
                                                     long seed,
                                                     ModelParameters initial,
                                                     double stepSize,
                                                     int leapfrogSteps) {
        HmcSampler hmc = new HmcSampler(this, dataset);
        double[] z0 = new double[] {
                logit(initial.S()),
                logit(initial.N()),
                logit(initial.alpha()),
                Math.log(Math.max(initial.beta(), 1e-12))
        };
        HmcSampler.Result res = hmc.sample(totalIters, burnIn, thin, seed, z0, stepSize, leapfrogSteps);
        return res.samples;
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double c = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(c / (1.0 - c));
    }

    private List<ModelParameters> runMhChain(List<ClaimData> dataset, int sampleCount, long seed) {
        Prepared prep = precompute(dataset);
        boolean par = dataset.size() >= parallelThreshold;
        return runMhChain(prep, sampleCount, seed, par);
    }

    private List<ModelParameters> runMhChain(Prepared prep, int sampleCount, long seed, boolean parallel) {
        System.out.println("Starting MH chain (seed=" + seed + ")...");
        if (sampleCount <= 0) return List.of();

        ModelParameters current = new ModelParameters(0.7, 0.6, 0.5, 1.0);
        double currentLogPost = logPosteriorPrepared(prep, current, parallel);
        ArrayList<ModelParameters> samples = new ArrayList<>(sampleCount);

        java.util.Random rng = new java.util.Random(seed);
        double stepS = 0.05, stepN = 0.05, stepAlpha = 0.05, stepBeta = 0.10;

        int burn = Math.min(500, Math.max(0, sampleCount / 5));
        int thin = Math.max(1, sampleCount / Math.max(1, Math.min(100, sampleCount)));
        int kept = 0;
        for (int iter = 0; kept < sampleCount; ) {
            double propS = clamp01(current.S() + stepS * (rng.nextDouble() * 2 - 1));
            double propN = clamp01(current.N() + stepN * (rng.nextDouble() * 2 - 1));
            double propAlpha = clamp01(current.alpha() + stepAlpha * (rng.nextDouble() * 2 - 1));
            double propBeta = Math.max(1e-6, current.beta() * Math.exp(stepBeta * (rng.nextDouble() * 2 - 1)));

            ModelParameters proposal = new ModelParameters(propS, propN, propAlpha, propBeta);
            double propLogPost = logPosteriorPrepared(prep, proposal, parallel);
            double acceptProb = Math.min(1.0, Math.exp(propLogPost - currentLogPost));
            if (rng.nextDouble() < acceptProb) {
                current = proposal;
                currentLogPost = propLogPost;
            }

            iter++;
            if (iter > burn && (iter - burn) % thin == 0) {
                samples.add(current);
                kept++;
            }
        }
        return samples;
    }

    private static double clamp01(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }
}
