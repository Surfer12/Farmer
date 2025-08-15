// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Smoke test for JSON/JSONL persistence from HmcMultiChainRunner.
 */
public final class HmcPersistenceTest {
    public static void main(String[] args) throws Exception {
        int checks = 0;

        // 1) Synthetic dataset
        List<ClaimData> dataset = new ArrayList<>();
        java.util.Random rng = new java.util.Random(99);
        for (int i = 0; i < 40; i++) {
            dataset.add(new ClaimData("p-" + i, rng.nextBoolean(),
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.abs(rng.nextGaussian()) * 0.3,
                    Math.min(1.0, Math.max(0.0, 0.5 + 0.2 * rng.nextGaussian()))));
        }

        // 2) Output directory (temp under project root)
        File outDir = new File("hmc-out-test");
        if (outDir.exists()) deleteRecursive(outDir);
        outDir.mkdirs();

        // 3) Runner with small settings
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        int chains = 2, warm = 200, iters = 600, thin = 3, leap = 20;
        long seed = 4242L;
        double[] z0 = new double[] { logit(0.7), logit(0.6), logit(0.5), Math.log(1.0) };
        HmcMultiChainRunner runner = new HmcMultiChainRunner(
                model, dataset, chains, warm, iters, thin, seed, z0, 0.01, leap, 0.75, outDir);
        HmcMultiChainRunner.Summary summary = runner.run();

        // 4) Assertions on files
        File summaryJson = new File(outDir, "summary.json");
        if (!summaryJson.isFile()) throw new AssertionError("summary.json missing"); checks++;
        String summaryText = Files.readString(summaryJson.toPath());
        if (!summaryText.contains("\"chains\"") || !summaryText.contains("\"diagnostics\""))
            throw new AssertionError("summary.json missing keys");
        checks++;

        // per-chain files
        for (int c = 0; c < chains; c++) {
            File meta = new File(outDir, String.format("chain-%02d.meta.json", c));
            File lines = new File(outDir, String.format("chain-%02d.jsonl", c));
            if (!meta.isFile() || !lines.isFile()) throw new AssertionError("missing chain files for " + c);
            checks++;
            String metaText = Files.readString(meta.toPath());
            if (!metaText.contains("\"chain\":" + c) || !metaText.contains("\"acceptance\""))
                throw new AssertionError("meta.json missing keys for chain " + c);
            checks++;

            int lineCount = countLines(lines);
            int expected = summary.chains.get(c).samples.size();
            if (lineCount != expected)
                throw new AssertionError("jsonl line count mismatch for chain " + c + ": " + lineCount + " != " + expected);
            checks++;

            // check first line has required keys
            try (BufferedReader br = new BufferedReader(new FileReader(lines))) {
                String first = br.readLine();
                if (first == null) throw new AssertionError("empty jsonl for chain " + c);
                if (!(first.contains("\"chain\":" + c) && first.contains("\"S\"") && first.contains("\"N\"")
                        && first.contains("\"alpha\"") && first.contains("\"beta\"")))
                    throw new AssertionError("jsonl first line missing keys for chain " + c);
                checks++;
            }
        }

        // 5) Clean up
        deleteRecursive(outDir);
        System.out.println("HmcPersistenceTest: OK (" + checks + " checks)");
    }

    private static int countLines(File f) throws Exception {
        int n = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            while (br.readLine() != null) n++;
        }
        return n;
    }

    private static void deleteRecursive(File f) {
        if (f == null || !f.exists()) return;
        if (f.isDirectory()) {
            File[] kids = f.listFiles();
            if (kids != null) for (File k : kids) deleteRecursive(k);
        }
        try { Files.deleteIfExists(f.toPath()); } catch (Exception ignored) {}
    }

    private static double logit(double x) {
        double eps = 1e-12;
        double c = Math.max(eps, Math.min(1.0 - eps, x));
        return Math.log(c / (1.0 - c));
    }
}


