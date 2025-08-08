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
        System.err.println("Usage: java -cp <cp> qualia.Core <console|file|jdbc|stein|hmc|rmala>");
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
        HmcSampler.Result res = hmc.sample(
                4000,      // total iterations
                1000,      // burn-in
                5,         // thinning
                20240808L, // seed
                z0,
                0.02,      // step size
                20         // leapfrog steps
        );

        // Report acceptance and mean Ψ
        double meanPsi = 0.0; int m = 0;
        for (ModelParameters p : res.samples) {
            double s = 0.0;
            for (ClaimData c : dataset) s += model.calculatePsi(c, p);
            meanPsi += s / dataset.size();
            m++;
        }
        if (m > 0) meanPsi /= m;
        System.out.println("hmc: kept " + res.samples.size() + " samples");
        System.out.println("hmc: acceptanceRate = " + String.format("%.3f", res.acceptanceRate));
        System.out.println("hmc: mean Ψ over HMC samples = " + String.format("%.6f", meanPsi));
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double clamped = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(clamped / (1.0 - clamped));
    }
}



