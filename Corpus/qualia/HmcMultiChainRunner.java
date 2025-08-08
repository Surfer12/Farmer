// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Runs multiple adaptive HMC chains in parallel, persists JSONL draws per chain,
 * and emits a summary with acceptance, divergences, tuned step size/mass, and diagnostics.
 */
public final class HmcMultiChainRunner {
    private final HierarchicalBayesianModel model;
    private final List<ClaimData> dataset;
    private final int chains;
    private final int warmupIters;
    private final int samplingIters;
    private final int thin;
    private final long baseSeed;
    private final double[] z0;
    private final double initStepSize;
    private final int leapfrogSteps;
    private final double targetAccept;
    private final File outDir;

    public HmcMultiChainRunner(HierarchicalBayesianModel model,
                               List<ClaimData> dataset,
                               int chains,
                               int warmupIters,
                               int samplingIters,
                               int thin,
                               long baseSeed,
                               double[] z0,
                               double initStepSize,
                               int leapfrogSteps,
                               double targetAccept,
                               File outDir) {
        this.model = Objects.requireNonNull(model);
        this.dataset = Objects.requireNonNull(dataset);
        this.chains = Math.max(1, chains);
        this.warmupIters = Math.max(0, warmupIters);
        this.samplingIters = Math.max(0, samplingIters);
        this.thin = Math.max(1, thin);
        this.baseSeed = baseSeed;
        this.z0 = Objects.requireNonNull(z0).clone();
        this.initStepSize = Math.max(1e-8, initStepSize);
        this.leapfrogSteps = Math.max(1, leapfrogSteps);
        this.targetAccept = Math.min(0.95, Math.max(0.55, targetAccept));
        this.outDir = outDir;
    }

    public Summary run() {
        if (outDir != null) outDir.mkdirs();
        final long PHI64 = 0x9E3779B97F4A7C15L;
        ExecutorService pool = Executors.newFixedThreadPool(Math.min(chains, Math.max(1, Runtime.getRuntime().availableProcessors())));
        try {
            List<CompletableFuture<ChainResult>> futures = new ArrayList<>(chains);
            for (int c = 0; c < chains; c++) {
                final int chainId = c;
                final long seed = baseSeed + PHI64 * (long) c;
                futures.add(CompletableFuture.supplyAsync(() -> runChain(chainId, seed), pool));
            }
            List<ChainResult> results = new ArrayList<>(chains);
            for (CompletableFuture<ChainResult> f : futures) results.add(f.join());

            // Collect draws for diagnostics
            List<List<ModelParameters>> all = new ArrayList<>(chains);
            for (ChainResult r : results) all.add(r.samples);
            Diagnostics diag = model.diagnose(all);

            // Write summary JSON if requested and export metrics
            Summary summary = new Summary(results, diag);
            if (outDir != null) {
                writeText(new File(outDir, "summary.json"), summary.toJson());
                // Also write per-chain meta JSONs
                for (ChainResult r : results) {
                    File mf = new File(outDir, String.format("chain-%02d.meta.json", r.chainId));
                    writeText(mf, r.toJson());
                }
            }
            exportMetrics(results, diag);
            return summary;
        } finally {
            pool.shutdown();
            try { pool.awaitTermination(5, TimeUnit.SECONDS); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        }
    }

    private ChainResult runChain(int chainId, long seed) {
        HmcSampler hmc = new HmcSampler(model, dataset);
        HmcSampler.AdaptiveResult res = hmc.sampleAdaptive(
                warmupIters,
                samplingIters,
                thin,
                seed,
                z0,
                initStepSize,
                leapfrogSteps,
                targetAccept
        );
        // Persist draws
        if (outDir != null) {
            File f = new File(outDir, String.format("chain-%02d.jsonl", chainId));
            try (BufferedWriter w = new BufferedWriter(new FileWriter(f))) {
                int i = 0;
                for (ModelParameters p : res.samples) {
                    w.write(toJsonLine(chainId, i++, p));
                    w.newLine();
                }
            } catch (IOException ignored) { }
        }
        return new ChainResult(chainId, res.samples, res.acceptanceRate, res.tunedStepSize, res.massDiag, res.divergenceCount);
    }

    private static void writeText(File f, String text) {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(f))) {
            w.write(text);
        } catch (IOException ignored) { }
    }

    private static String toJsonLine(int chainId, int index, ModelParameters p) {
        return String.format(java.util.Locale.ROOT,
                "{\"chain\":%d,\"i\":%d,\"S\":%.8f,\"N\":%.8f,\"alpha\":%.8f,\"beta\":%.8f}",
                chainId, index, p.S(), p.N(), p.alpha(), p.beta());
    }

    public static final class ChainResult {
        public final int chainId;
        public final List<ModelParameters> samples;
        public final double acceptanceRate;
        public final double tunedStepSize;
        public final double[] massDiag;
        public final int divergences;
        ChainResult(int chainId,
                    List<ModelParameters> samples,
                    double acceptanceRate,
                    double tunedStepSize,
                    double[] massDiag,
                    int divergences) {
            this.chainId = chainId;
            this.samples = samples;
            this.acceptanceRate = acceptanceRate;
            this.tunedStepSize = tunedStepSize;
            this.massDiag = massDiag;
            this.divergences = divergences;
        }
        String toJson() {
            return String.format(java.util.Locale.ROOT,
                    "{\"chain\":%d,\"acceptance\":%.4f,\"divergences\":%d,\"eps\":%.6f,\"mass\":[%.6f,%.6f,%.6f,%.6f],\"n\":%d}",
                    chainId, acceptanceRate, divergences, tunedStepSize, massDiag[0], massDiag[1], massDiag[2], massDiag[3], samples.size());
        }
    }

    public static final class Summary {
        public final List<ChainResult> chains;
        public final Diagnostics diagnostics;
        Summary(List<ChainResult> chains, Diagnostics diagnostics) {
            this.chains = chains;
            this.diagnostics = diagnostics;
        }
        public String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"chains\":[");
            for (int i = 0; i < chains.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(chains.get(i).toJson());
            }
            sb.append("],\"diagnostics\":");
            sb.append(String.format(java.util.Locale.ROOT,
                    "{\"rHatS\":%.4f,\"rHatN\":%.4f,\"rHatAlpha\":%.4f,\"rHatBeta\":%.4f,\"essS\":%.1f,\"essN\":%.1f,\"essAlpha\":%.1f,\"essBeta\":%.1f}",
                    diagnostics.rHatS, diagnostics.rHatN, diagnostics.rHatAlpha, diagnostics.rHatBeta,
                    diagnostics.essS, diagnostics.essN, diagnostics.essAlpha, diagnostics.essBeta));
            sb.append('}');
            return sb.toString();
        }
    }

    private static void exportMetrics(List<ChainResult> results, Diagnostics diag) {
        MetricsRegistry mr = MetricsRegistry.get();
        int n = results.size();
        mr.setGauge("hmc_chains", n);
        long totalDiv = 0L;
        long accPpmSum = 0L;
        for (ChainResult r : results) {
            long accPpm = (long) Math.round(r.acceptanceRate * 1_000_000.0);
            long epsPpm = (long) Math.round(r.tunedStepSize * 1_000_000.0);
            mr.setGauge(String.format("hmc_chain_%02d_acceptance_ppm", r.chainId), accPpm);
            mr.setGauge(String.format("hmc_chain_%02d_divergences", r.chainId), r.divergences);
            mr.setGauge(String.format("hmc_chain_%02d_samples", r.chainId), r.samples.size());
            mr.setGauge(String.format("hmc_chain_%02d_eps_1e6", r.chainId), epsPpm);
            totalDiv += r.divergences;
            accPpmSum += accPpm;
        }
        mr.setGauge("hmc_total_divergences", totalDiv);
        if (n > 0) mr.setGauge("hmc_mean_acceptance_ppm", accPpmSum / n);
        // Basic diagnostics as gauges (scaled by 1e6 for doubles)
        mr.setGauge("hmc_diag_rhatS_1e6", (long) Math.round(diag.rHatS * 1_000_000.0));
        mr.setGauge("hmc_diag_rhatN_1e6", (long) Math.round(diag.rHatN * 1_000_000.0));
        mr.setGauge("hmc_diag_rhatAlpha_1e6", (long) Math.round(diag.rHatAlpha * 1_000_000.0));
        mr.setGauge("hmc_diag_rhatBeta_1e6", (long) Math.round(diag.rHatBeta * 1_000_000.0));
    }
}


