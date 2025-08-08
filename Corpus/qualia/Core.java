// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.io.File;
import java.util.Date;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Core entrypoint for quick demos of available AuditSink implementations.
 *
 * Usage:
 *   java -cp <cp> qualia.Core console
 *   java -cp <cp> qualia.Core file
 *   java -cp <cp> qualia.Core jdbc
 *
 * For JDBC mode, set env vars: JDBC_URL, JDBC_USER (optional), JDBC_PASS (optional).
 */
public final class Core {
    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            printUsageAndExit();
        }
        String mode = args[0].toLowerCase();
        switch (mode) {
            case "console" -> runConsole();
            case "file" -> runFile();
            case "jdbc" -> runJdbc();
            case "stein" -> runStein();
            case "hmc" -> runHmc();
            case "mcda" -> runMcda();
            case "rmala" -> runRmala();
            default -> printUsageAndExit();
        }
    }

    private static void runConsole() {
        AuditSink sink = new ConsoleAuditSink(AuditOptions.builder().dryRun(false).build());
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        sink.write(rec, AuditOptions.builder().dryRun(false).build()).join();
        System.out.println("console: wrote record " + rec);
    }

    private static void runFile() {
        File dir = new File("audit-logs");
        AuditSink sink = new FileAuditSink(dir, "audit", 10_000_000, 60_000, 1024);
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        sink.write(rec, AuditOptions.builder().dryRun(false).build()).join();
        System.out.println("file: wrote record to " + dir.getAbsolutePath());
    }

    private static void runJdbc() {
        String url = System.getenv("JDBC_URL");
        if (url == null || url.isEmpty()) {
            System.err.println("Missing JDBC_URL env var. Example: jdbc:postgresql://localhost:5432/qualia");
            System.exit(2);
        }
        String user = System.getenv("JDBC_USER");
        String pass = System.getenv("JDBC_PASS");

        AuditSink sink = new JdbcAuditSink(url, user, pass);
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        AuditOptions opts = AuditOptions.builder().idempotencyKey(rec.id()).dryRun(false).build();
        sink.write(rec, opts).join();
        System.out.println("jdbc: wrote record " + rec + " to " + url);

        if (sink instanceof JdbcAuditSink j) {
            j.close();
        }
    }

    private static void runStein() {
        // 1) Small synthetic dataset for speed
        List<ClaimData> dataset = new ArrayList<>();
        Random rng = new Random(42);
        int n = 50;
        for (int i = 0; i < n; i++) {
            String id = "c-" + i;
            boolean y = rng.nextBoolean();
            double ra = Math.abs(rng.nextGaussian()) * 0.5; // small risks
            double rv = Math.abs(rng.nextGaussian()) * 0.5;
            double pHe = Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()));
            dataset.add(new ClaimData(id, y, ra, rv, pHe));
        }

        // 2) Model and MCMC samples
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        int sampleCount = 60; // keep small for demo speed
        List<ModelParameters> samples = model.performInference(dataset, sampleCount);

        // 3) Define integrand f(params): average Psi over dataset
        double[] fvals = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            ModelParameters p = samples.get(i);
            double sumPsi = 0.0;
            for (ClaimData c : dataset) sumPsi += model.calculatePsi(c, p);
            fvals[i] = sumPsi / dataset.size();
        }

        // 4) Build Stein estimator and compute c_N
        double lengthScale = 0.5; // demo value
        SteinEstimator est = new SteinEstimator(lengthScale, model, dataset, samples);
        double cN = est.estimate(fvals, 50, 1e-3);

        // 5) Compare to plain MC average
        double mc = 0.0; for (double v : fvals) mc += v; mc /= fvals.length;

        System.out.println("stein: c_N (Stein) = " + cN);
        System.out.println("stein: MC average  = " + mc);
    }

    private static void printUsageAndExit() {
        System.err.println("Usage: java -cp <cp> qualia.Core <console|file|jdbc|stein|hmc|mcda|rmala>");
        System.exit(1);
    }

    private static void runRmala() {
        // small synthetic dataset
        List<ClaimData> dataset = new ArrayList<>();
        Random rng = new Random(7);
        for (int i = 0; i < 60; i++) {
            dataset.add(new ClaimData("r-" + i, rng.nextBoolean(), Math.abs(rng.nextGaussian()) * 0.3,
                    Math.abs(rng.nextGaussian()) * 0.3, Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()))));
        }

        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        RmalaSampler sampler = new RmalaSampler(model, dataset);

        // constant step size baseline policy
        RmalaSampler.StepSizePolicy constant = x -> 0.05;
        double[] x0 = new double[] {0.6, 0.5, 0.5, 1.0};

        RmalaSampler.Result res = sampler.sample(5000, 1000, 10, 123L, x0, constant);
        System.out.println("rmala: kept " + res.samples.size() + " samples");
        System.out.println("rmala: acceptanceRate = " + String.format("%.3f", res.acceptanceRate));
        System.out.println("rmala: avgCDLB = " + String.format("%.6f", res.avgCdlb));

        // report a simple f average (mean Ψ) from RMALA samples
        double meanPsi = 0.0;
        for (ModelParameters p : res.samples) {
            double sum = 0.0;
            for (ClaimData c : dataset) sum += model.calculatePsi(c, p);
            meanPsi += sum / dataset.size();
        }
        if (!res.samples.isEmpty()) meanPsi /= res.samples.size();
        System.out.println("rmala: mean Ψ over RMALA samples = " + String.format("%.6f", meanPsi));
    }

    private static void runHmc() {
        // small synthetic dataset
        java.util.List<ClaimData> dataset = new java.util.ArrayList<>();
        java.util.Random rng = new java.util.Random(11);
        for (int i = 0; i < 60; i++) {
            dataset.add(new ClaimData("h-" + i, rng.nextBoolean(),
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()))));
        }

        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        HmcSampler hmc = new HmcSampler(model, dataset);

        // Unconstrained z0 corresponding to params ~ [0.7, 0.6, 0.5, 1.0]
        double[] z0 = new double[] { logit(0.7), logit(0.6), logit(0.5), Math.log(1.0) };
        double stepSize = getEnvDouble("HMC_STEP_SIZE", 0.01);
        int leap = getEnvInt("HMC_LEAP", 30);
        HmcSampler.Result res1 = hmc.sample(
                3000,      // total iterations
                1000,      // burn-in
                3,         // thinning
                20240808L, // seed
                z0,
                stepSize,
                leap
        );
        // Second chain (different seed)
        HmcSampler.Result res2 = hmc.sample(
                3000, 1000, 3, 20240809L, z0,
                stepSize, leap
        );

        // Report acceptance and mean Ψ per chain
        double meanPsi1 = 0.0; int m1 = 0;
        for (ModelParameters p : res1.samples) {
            double s = 0.0;
            for (ClaimData c : dataset) s += model.calculatePsi(c, p);
            meanPsi1 += s / dataset.size();
            m1++;
        }
        if (m1 > 0) meanPsi1 /= m1;
        double meanPsi2 = 0.0; int m2 = 0;
        for (ModelParameters p : res2.samples) {
            double s = 0.0;
            for (ClaimData c : dataset) s += model.calculatePsi(c, p);
            meanPsi2 += s / dataset.size();
            m2++;
        }
        if (m2 > 0) meanPsi2 /= m2;
        System.out.println("hmc: chain1 kept=" + res1.samples.size() + ", acc=" + String.format("%.3f", res1.acceptanceRate) + ", meanΨ=" + String.format("%.6f", meanPsi1));
        System.out.println("hmc: chain2 kept=" + res2.samples.size() + ", acc=" + String.format("%.3f", res2.acceptanceRate) + ", meanΨ=" + String.format("%.6f", meanPsi2));

        // Diagnostics R̂/ESS on Ψ across chains (quick scalar view)
        java.util.List<java.util.List<ModelParameters>> chains = java.util.List.of(res1.samples, res2.samples);
        Diagnostics diag = model.diagnose(chains);
        System.out.println("hmc: R̂ S=" + String.format("%.3f", diag.rHatS)
                + ", N=" + String.format("%.3f", diag.rHatN)
                + ", alpha=" + String.format("%.3f", diag.rHatAlpha)
                + ", beta=" + String.format("%.3f", diag.rHatBeta));
        System.out.println("hmc: ESS S=" + String.format("%.1f", diag.essS)
                + ", N=" + String.format("%.1f", diag.essN)
                + ", alpha=" + String.format("%.1f", diag.essAlpha)
                + ", beta=" + String.format("%.1f", diag.essBeta));
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double clamped = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(clamped / (1.0 - clamped));
    }

    private static void runMcda() {
        // Define alternatives with raw criteria and Ψ
        Mcda.Alternative a = new Mcda.Alternative(
                "A",
                java.util.Map.of("cost", 100.0, "value", 0.8),
                0.85 // Ψ(A)
        );
        Mcda.Alternative b = new Mcda.Alternative(
                "B",
                java.util.Map.of("cost", 80.0, "value", 0.7),
                0.80 // Ψ(B)
        );
        Mcda.Alternative c = new Mcda.Alternative(
                "C",
                java.util.Map.of("cost", 120.0, "value", 0.9),
                0.78 // Ψ(C)
        );

        java.util.List<Mcda.Alternative> alts = java.util.List.of(a, b, c);

        // Gate by confidence threshold τ
        double tau = 0.79;
        java.util.List<Mcda.Alternative> Ftau = Mcda.gateByPsi(alts, tau);
        System.out.println("mcda: feasible after gate(τ=" + tau + ") = " + Ftau);

        // Define criteria specs; include Ψ as criterion "psi"
        java.util.List<Mcda.CriterionSpec> specs = java.util.List.of(
                new Mcda.CriterionSpec("psi", Mcda.Direction.BENEFIT, 0.4),
                new Mcda.CriterionSpec("value", Mcda.Direction.BENEFIT, 0.4),
                new Mcda.CriterionSpec("cost", Mcda.Direction.COST, 0.2)
        );
        Mcda.validateWeights(specs);

        // Rank by WSM and TOPSIS
        java.util.List<Mcda.Ranked> wsm = Mcda.rankByWSM(Ftau, specs);
        java.util.List<Mcda.Ranked> topsis = Mcda.rankByTOPSIS(Ftau, specs);
        System.out.println("mcda: WSM ranking: " + wsm);
        System.out.println("mcda: TOPSIS ranking: " + topsis);
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

    private static int getEnvInt(String key, int fallback) {
        try {
            String v = System.getenv(key);
            if (v == null || v.isEmpty()) return fallback;
            return Integer.parseInt(v);
        } catch (Exception e) {
            return fallback;
        }
    }
}



