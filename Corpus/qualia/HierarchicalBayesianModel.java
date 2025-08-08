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
import java.util.concurrent.atomic.DoubleAdder;

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
public final class HierarchicalBayesianModel implements PsiModel {
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

    /** Returns whether likelihood evaluation should use parallel execution for a dataset of given size. */
    public boolean shouldParallelize(int datasetSize) {
        return datasetSize >= parallelThreshold;
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
    static final class Prepared {
        final double[] pen;       // exp(-[λ1 Ra + λ2 Rv]) per-claim
        final double[] pHe;       // P(H|E) per-claim
        final boolean[] y;        // labels per-claim
        Prepared(double[] pen, double[] pHe, boolean[] y) {
            this.pen = pen; this.pHe = pHe; this.y = y;
        }
        int size() { return pen.length; }
    }

    // Cache configuration and layers
    private static final CacheConfig CACHE_CFG = CacheConfig.fromEnv();
    private static final SimpleCaffeineLikeCache<DatasetCacheKey, Prepared> PREP_CACHE =
            CACHE_CFG.memoryEnabled ?
                    new SimpleCaffeineLikeCache<>(
                            CACHE_CFG.maxEntries,
                            CACHE_CFG.ttlDuration(),
                            true,
                            CACHE_CFG.maxWeightBytes,
                            prep -> prep == null ? 0L : estimatePreparedWeight(prep)
                    ) : null;
    private static final DatasetPreparedDiskStore DISK_STORE =
            CACHE_CFG.diskEnabled ? new DatasetPreparedDiskStore(new java.io.File(CACHE_CFG.diskDir)) : null;

    private static long estimatePreparedWeight(Prepared p) {
        if (p == null) return 0L;
        // Approximate: 3 arrays of length n: double(8B), double(8B), boolean(1B ~ padded but approx)
        long n = p.size();
        return n * (8L + 8L + 1L);
    }

    /** Export a snapshot of cache stats to MetricsRegistry gauges. No-op if disabled. */
    public static void exportCacheStats() {
        MetricsRegistry mr = MetricsRegistry.get();
        if (PREP_CACHE != null) {
            var s = PREP_CACHE.stats();
            mr.setGauge("prep_cache_hits", s.hitCount);
            mr.setGauge("prep_cache_misses", s.missCount);
            mr.setGauge("prep_cache_load_success", s.loadSuccessCount);
            mr.setGauge("prep_cache_load_failure", s.loadFailureCount);
            mr.setGauge("prep_cache_evictions", s.evictionCount);
            mr.setGauge("prep_cache_total_load_time_ns", s.totalLoadTimeNanos);
        }
    }

    Prepared precompute(List<ClaimData> dataset) {
        // Build cache key (algoVersion ties to preparation logic and priors affecting pen)
        String datasetId = "anon"; // caller can extend API later to pass a real id
        String datasetHash = DatasetCacheKey.hashDatasetContent(dataset);
        String paramsHash = Long.toHexString(Double.doubleToLongBits(priors.lambda1())) + ":" +
                Long.toHexString(Double.doubleToLongBits(priors.lambda2()));
        DatasetCacheKey key = new DatasetCacheKey(datasetId, datasetHash, paramsHash, "precompute-v1");

        // Decide flow based on enabled layers
        final boolean mem = PREP_CACHE != null;
        final boolean disk = DISK_STORE != null;

        try {
            if (mem) {
                return PREP_CACHE.get(key, k -> {
                    if (disk) {
                        Prepared fromDisk = DISK_STORE.readIfPresent(k);
                        if (fromDisk != null) return fromDisk;
                    }
                    Prepared fresh = computePrepared(dataset);
                    if (disk) { try { DISK_STORE.write(k, fresh); } catch (Exception ignored) {} }
                    return fresh;
                });
            } else {
                if (disk) {
                    Prepared fromDisk = DISK_STORE.readIfPresent(key);
                    if (fromDisk != null) return fromDisk;
                }
                Prepared fresh = computePrepared(dataset);
                if (disk) { try { DISK_STORE.write(key, fresh); } catch (Exception ignored) {} }
                return fresh;
            }
        } catch (Exception e) {
            return computePrepared(dataset);
        }
    }

    private Prepared computePrepared(List<ClaimData> dataset) {
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

    double totalLogLikelihoodPrepared(Prepared prep, ModelParameters params, boolean parallel) {
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

    double logPosteriorPrepared(Prepared prep, ModelParameters params, boolean parallel) {
        return totalLogLikelihoodPrepared(prep, params, parallel) + logPriors(params);
    }

    /**
     * Analytic gradient of log-posterior with respect to (S, N, alpha, beta), using a prepared dataset.
     * Parallelizes across observations when {@code parallel} is true.
     */
    public double[] gradientLogPosteriorPrepared(Prepared prep, ModelParameters params, boolean parallel) {
        double dS = 0.0, dN = 0.0, dA = 0.0, dB = 0.0;

        // Likelihood gradient contributions
        final double S = params.S();
        final double N = params.N();
        final double A = params.alpha();
        final double B = params.beta();

        final int n = prep.size();
        if (!parallel) {
            for (int i = 0; i < n; i++) {
                double O = A * S + (1.0 - A) * N;
                double pen = prep.pen[i];
                double P = prep.pHe[i];
                boolean capped = (B * P >= 1.0);
                double pBeta = capped ? 1.0 : (B * P);
                double psi = O * pen * pBeta;

                double eps = 1e-12;
                double denomPos = Math.max(eps, psi);
                double denomNeg = Math.max(eps, 1.0 - psi);

                double dpsi_dS = A * pen * pBeta;
                double dpsi_dN = (1.0 - A) * pen * pBeta;
                double dpsi_dA = (S - N) * pen * pBeta;
                double dpsi_dB = capped ? 0.0 : (O * pen * P);

                if (prep.y[i]) {
                    dS += dpsi_dS / denomPos;
                    dN += dpsi_dN / denomPos;
                    dA += dpsi_dA / denomPos;
                    dB += dpsi_dB / denomPos;
                } else {
                    dS += -dpsi_dS / denomNeg;
                    dN += -dpsi_dN / denomNeg;
                    dA += -dpsi_dA / denomNeg;
                    dB += -dpsi_dB / denomNeg;
                }
            }
        } else {
            DoubleAdder aS = new DoubleAdder();
            DoubleAdder aN = new DoubleAdder();
            DoubleAdder aA = new DoubleAdder();
            DoubleAdder aB = new DoubleAdder();
            IntStream.range(0, n).parallel().forEach(i -> {
                double O = A * S + (1.0 - A) * N;
                double pen = prep.pen[i];
                double P = prep.pHe[i];
                boolean capped = (B * P >= 1.0);
                double pBeta = capped ? 1.0 : (B * P);
                double psi = O * pen * pBeta;

                double eps = 1e-12;
                double denomPos = Math.max(eps, psi);
                double denomNeg = Math.max(eps, 1.0 - psi);

                double dpsi_dS = A * pen * pBeta;
                double dpsi_dN = (1.0 - A) * pen * pBeta;
                double dpsi_dA = (S - N) * pen * pBeta;
                double dpsi_dB = capped ? 0.0 : (O * pen * P);

                if (prep.y[i]) {
                    aS.add(dpsi_dS / denomPos);
                    aN.add(dpsi_dN / denomPos);
                    aA.add(dpsi_dA / denomPos);
                    aB.add(dpsi_dB / denomPos);
                } else {
                    aS.add(-dpsi_dS / denomNeg);
                    aN.add(-dpsi_dN / denomNeg);
                    aA.add(-dpsi_dA / denomNeg);
                    aB.add(-dpsi_dB / denomNeg);
                }
            });
            dS = aS.sum();
            dN = aN.sum();
            dA = aA.sum();
            dB = aB.sum();
        }

        // Prior gradients (same as non-prepared)
        double epsUnit = 1e-12;
        double Sa = priors.s_alpha();
        double Sb = priors.s_beta();
        double sClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, S));
        dS += (Sa - 1.0) / sClamped - (Sb - 1.0) / (1.0 - sClamped);

        double Na = priors.n_alpha();
        double Nb = priors.n_beta();
        double nClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, N));
        dN += (Na - 1.0) / nClamped - (Nb - 1.0) / (1.0 - nClamped);

        double Aa = priors.alpha_alpha();
        double Ab = priors.alpha_beta();
        double aClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, A));
        dA += (Aa - 1.0) / aClamped - (Ab - 1.0) / (1.0 - aClamped);

        double mu = priors.beta_mu();
        double sigma = priors.beta_sigma();
        double t = Math.log(Math.max(B, epsUnit));
        dB += (-(t - mu) / (sigma * sigma) - 1.0) * (1.0 / Math.max(B, epsUnit));

        return new double[] { dS, dN, dA, dB };
    }

    /**
     * Gradient in z-space of logTarget(z) = logPosterior(θ(z)) + log|J(z)| where
     * θ(z) = (sigmoid(z0), sigmoid(z1), sigmoid(z2), exp(z3)).
     */
    public double[] gradLogTargetZ(Prepared prep, double[] z, boolean parallel) {
        ModelParameters theta = new ModelParameters(
                sigmoid(z[0]),
                sigmoid(z[1]),
                sigmoid(z[2]),
                Math.exp(z[3])
        );
        double[] dLogPost_dTheta = gradientLogPosteriorPrepared(prep, theta, parallel);
        double S = theta.S();
        double N = theta.N();
        double A = theta.alpha();
        double B = theta.beta();

        // dθ/dz
        double dSdz = S * (1.0 - S);
        double dNdz = N * (1.0 - N);
        double dAdz = A * (1.0 - A);
        double dBdz = B; // beta = exp(z3)

        // Chain rule for logPosterior term
        double g0 = dLogPost_dTheta[0] * dSdz;
        double g1 = dLogPost_dTheta[1] * dNdz;
        double g2 = dLogPost_dTheta[2] * dAdz;
        double g3 = dLogPost_dTheta[3] * dBdz;

        // Gradient of log|J| wrt z
        g0 += (1.0 - 2.0 * S);
        g1 += (1.0 - 2.0 * N);
        g2 += (1.0 - 2.0 * A);
        g3 += 1.0;

        return new double[] { g0, g1, g2, g3 };
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

    /**
     * Analytic gradient of log-posterior with respect to (S, N, alpha, beta).
     * Includes both likelihood and prior terms. Piecewise handles beta cap where
     * P(H|E, β) = min{β·P(H|E), 1}.
     */
    public double[] gradientLogPosterior(List<ClaimData> dataset, ModelParameters params) {
        double dS = 0.0, dN = 0.0, dA = 0.0, dB = 0.0;

        // Likelihood gradient
        double S = params.S();
        double N = params.N();
        double A = params.alpha();
        double B = params.beta();

        for (ClaimData c : dataset) {
            double O = A * S + (1.0 - A) * N;
            double pen = Math.exp(-(priors.lambda1() * c.riskAuthenticity() + priors.lambda2() * c.riskVirality()));
            double P = c.probabilityHgivenE();
            boolean capped = (B * P >= 1.0);
            double pBeta = capped ? 1.0 : (B * P);
            double psi = O * pen * pBeta;

            // Clamp to avoid division by zero in log-gradient
            double eps = 1e-12;
            double denomPos = Math.max(eps, psi);
            double denomNeg = Math.max(eps, 1.0 - psi);

            double dpsi_dS = A * pen * pBeta;
            double dpsi_dN = (1.0 - A) * pen * pBeta;
            double dpsi_dA = (S - N) * pen * pBeta;
            double dpsi_dB = capped ? 0.0 : (O * pen * P);

            if (c.isVerifiedTrue()) {
                dS += dpsi_dS / denomPos;
                dN += dpsi_dN / denomPos;
                dA += dpsi_dA / denomPos;
                dB += dpsi_dB / denomPos;
            } else {
                dS += -dpsi_dS / denomNeg;
                dN += -dpsi_dN / denomNeg;
                dA += -dpsi_dA / denomNeg;
                dB += -dpsi_dB / denomNeg;
            }
        }

        // Prior gradients
        double epsUnit = 1e-12;
        // Beta prior for S
        double Sa = priors.s_alpha();
        double Sb = priors.s_beta();
        double sClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, S));
        dS += (Sa - 1.0) / sClamped - (Sb - 1.0) / (1.0 - sClamped);

        // Beta prior for N
        double Na = priors.n_alpha();
        double Nb = priors.n_beta();
        double nClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, N));
        dN += (Na - 1.0) / nClamped - (Nb - 1.0) / (1.0 - nClamped);

        // Beta prior for alpha
        double Aa = priors.alpha_alpha();
        double Ab = priors.alpha_beta();
        double aClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, A));
        dA += (Aa - 1.0) / aClamped - (Ab - 1.0) / (1.0 - aClamped);

        // LogNormal prior for beta
        double mu = priors.beta_mu();
        double sigma = priors.beta_sigma();
        double t = Math.log(Math.max(B, epsUnit));
        dB += (-(t - mu) / (sigma * sigma) - 1.0) * (1.0 / Math.max(B, epsUnit));

        return new double[] { dS, dN, dA, dB };
    }

    /** Prepared fast-path gradient of log-posterior with respect to (S,N,alpha,beta). */
    double[] gradientLogPosteriorPrepared(Prepared prep, ModelParameters params, boolean parallel) {
        double dS = 0.0, dN = 0.0, dA = 0.0, dB = 0.0;

        double S = params.S();
        double N = params.N();
        double A = params.alpha();
        double B = params.beta();

        final double O = A * S + (1.0 - A) * N;
        final int n = prep.size();
        final double eps = 1e-12;

        for (int i = 0; i < n; i++) {
            double pen = prep.pen[i];
            double P = prep.pHe[i];
            boolean capped = (B * P >= 1.0);
            double pBeta = capped ? 1.0 : (B * P);
            double psi = O * pen * pBeta;

            double denomPos = Math.max(eps, psi);
            double denomNeg = Math.max(eps, 1.0 - psi);

            double dpsi_dS = A * pen * pBeta;
            double dpsi_dN = (1.0 - A) * pen * pBeta;
            double dpsi_dA = (S - N) * pen * pBeta;
            double dpsi_dB = capped ? 0.0 : (O * pen * P);

            if (prep.y[i]) {
                dS += dpsi_dS / denomPos;
                dN += dpsi_dN / denomPos;
                dA += dpsi_dA / denomPos;
                dB += dpsi_dB / denomPos;
            } else {
                dS += -dpsi_dS / denomNeg;
                dN += -dpsi_dN / denomNeg;
                dA += -dpsi_dA / denomNeg;
                dB += -dpsi_dB / denomNeg;
            }
        }

        // Priors
        double epsUnit = 1e-12;
        double Sa = priors.s_alpha();
        double Sb = priors.s_beta();
        double sClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, S));
        dS += (Sa - 1.0) / sClamped - (Sb - 1.0) / (1.0 - sClamped);

        double Na = priors.n_alpha();
        double Nb = priors.n_beta();
        double nClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, N));
        dN += (Na - 1.0) / nClamped - (Nb - 1.0) / (1.0 - nClamped);

        double Aa = priors.alpha_alpha();
        double Ab = priors.alpha_beta();
        double aClamped = Math.max(epsUnit, Math.min(1.0 - epsUnit, A));
        dA += (Aa - 1.0) / aClamped - (Ab - 1.0) / (1.0 - aClamped);

        double mu = priors.beta_mu();
        double sigma = priors.beta_sigma();
        double t = Math.log(Math.max(B, epsUnit));
        dB += (-(t - mu) / (sigma * sigma) - 1.0) * (1.0 / Math.max(B, epsUnit));

        return new double[] { dS, dN, dA, dB };
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
