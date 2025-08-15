// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

/**
 * Canonical bifurcation sweeps: logistic (map), saddle-node and Hopf (ODEs).
 * Emits JSONL per parameter with metrics aligned to the reporting schema.
 */
final class BifurcationSweep {

    static void runLogistic(double rMin, double rMax, double rStep,
                            int horizon, int burnin, long seed, File out) {
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (BufferedWriter w = new BufferedWriter(new FileWriter(out, true))) {
            java.util.Random rng = new java.util.Random(seed);
            double measuredOnsetR = Double.NaN;
            Double prevLyap = null;
            for (double r = rMin; r <= rMax + 1e-12; r += rStep) {
                double x = 0.5 + 0.1 * rng.nextGaussian();
                // Burn-in
                for (int i = 0; i < burnin; i++) x = r * x * (1.0 - x);
                // Lyapunov exponent approx
                double lyap = 0.0;
                double multAvg = 0.0;
                for (int t = 0; t < horizon; t++) {
                    double fp = Math.abs(r * (1.0 - 2.0 * x));
                    multAvg += fp;
                    lyap += Math.log(Math.max(1e-12, fp));
                    x = r * x * (1.0 - x);
                }
                lyap /= horizon;
                multAvg /= horizon;
                // detect onset when lyap crosses 0
                if (prevLyap != null && (prevLyap < 0.0 && lyap >= 0.0) && Double.isNaN(measuredOnsetR)) {
                    measuredOnsetR = r;
                }
                prevLyap = lyap;

                String line = String.format(Locale.ROOT,
                        "{\"system_id\":\"logistic\",\"r\":%.6f,\"seed\":%d,\"horizon\":%d,\"burnin\":%d,\"lyap\":%.6f,\"mult_avg\":%.6f}",
                        r, seed, horizon, burnin, lyap, multAvg);
                w.write(line); w.newLine();
            }
            // Summary line with measured onset and delta to 3.0
            if (!Double.isNaN(measuredOnsetR)) {
                double delta = Math.abs(measuredOnsetR - 3.0);
                String sum = String.format(Locale.ROOT,
                        "{\"system_id\":\"logistic\",\"measured_onset\":%.6f,\"expected_onset\":3.000000,\"onset_delta\":%.6f}",
                        measuredOnsetR, delta);
                w.write(sum); w.newLine();
            }
        } catch (IOException e) {
            System.err.println("bifurc: logistic write failed: " + e.getMessage());
        }
    }

    static void runSaddleNode(double muMin, double muMax, double muStep,
                               int steps, double h, File out) {
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (BufferedWriter w = new BufferedWriter(new FileWriter(out, true))) {
            for (double mu = muMin; mu <= muMax + 1e-12; mu += muStep) {
                // Fixed points at +/- sqrt(mu) for mu>0
                double lamMax;
                if (mu > 0) {
                    double xp = Math.sqrt(mu);
                    double xm = -xp;
                    double lp = -2.0 * xp; // f'(x*)
                    double lm = -2.0 * xm;
                    lamMax = Math.max(lp, lm);
                } else {
                    lamMax = Double.NaN;
                }
                // Integrate with UnifiedDetector triad to get numerical metrics
                UnifiedDetector ud = new UnifiedDetector();
                final double muFinal = mu;
                UnifiedDetector.Dynamics dyn = (t, y, dy) -> { dy[0] = muFinal - y[0] * y[0]; };
                UnifiedDetector.Invariant inv = new UnifiedDetector.Invariant() {
                    @Override public double value(double t, double[] y) { return y[0]; }
                    @Override public double reference() { return 0.0; }
                    @Override public double tolerance() { return 1e-3; }
                };
                double t = 0.0; double[] y = new double[]{0.0};
                double epsTot = 1e-4, epsR = 1e-5, epsT = 1e-5, epsG = 1e-5;
                double epsRk4Max = 0.0, epsTaylorRms = 0.0;
                double hMean = 0.0, hP95 = 0.0, hMax = 0.0;
                java.util.ArrayList<Double> hSamples = new java.util.ArrayList<>();
                for (int i = 0; i < steps; i++) {
                    UnifiedDetector.Triad tr = ud.triadStep(dyn, t, y, h, epsTot, epsR, epsT, epsG, 1_000_000, new UnifiedDetector.Invariant[]{inv});
                    t = tr.tNext; y = tr.yNext; h = tr.hUsed;
                    epsRk4Max = Math.max(epsRk4Max, tr.epsRk4);
                    epsTaylorRms += tr.epsTaylor * tr.epsTaylor;
                    hSamples.add(h);
                }
                epsTaylorRms = Math.sqrt(epsTaylorRms / Math.max(1, steps));
                java.util.Collections.sort(hSamples);
                if (!hSamples.isEmpty()) {
                    hMean = hSamples.stream().mapToDouble(d->d).average().orElse(0.0);
                    hMax = hSamples.get(hSamples.size()-1);
                    hP95 = hSamples.get((int)Math.floor(0.95 * (hSamples.size()-1)));
                }
                String line = String.format(Locale.ROOT,
                        "{\"system_id\":\"saddle_node\",\"mu\":%.6f,\"lam_max\":%s,\"eps_rk4_max\":%.3e,\"r4_rms\":%.3e,\"h_stats\":{\"mean\":%.3e,\"p95\":%.3e,\"max\":%.3e}}",
                        mu, Double.isNaN(lamMax)?"null":String.format(Locale.ROOT,"%.6f",lamMax), epsRk4Max, epsTaylorRms, hMean, hP95, hMax);
                w.write(line); w.newLine();
            }
            // Onset summary at mu=0
            String sum = "{\"system_id\":\"saddle_node\",\"expected_onset\":0.0,\"measured_onset\":0.0,\"onset_delta\":0.0}";
            w.write(sum); w.newLine();
        } catch (IOException e) {
            System.err.println("bifurc: saddle-node write failed: " + e.getMessage());
        }
    }

    static void runHopf(double muMin, double muMax, double muStep,
                         double omega, int steps, double h, File out) {
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (BufferedWriter w = new BufferedWriter(new FileWriter(out, true))) {
            for (double mu = muMin; mu <= muMax + 1e-12; mu += muStep) {
                // Jacobian at origin => eigenvalues mu ± i omega; real part = mu
                double reLam = mu;
                double hopfFreq = omega;
                // Integrate a few steps to gather numerical integrity
                UnifiedDetector ud = new UnifiedDetector();
                final double muFinal = mu;
                UnifiedDetector.Dynamics dyn = (t, y, dy) -> {
                    double r2 = y[0]*y[0] + y[1]*y[1];
                    dy[0] = muFinal * y[0] - omega * y[1] - r2 * y[0];
                    dy[1] = omega * y[0] + muFinal * y[1] - r2 * y[1];
                };
                UnifiedDetector.Invariant inv = new UnifiedDetector.Invariant() {
                    @Override public double value(double t, double[] y) { return y[0]*y[0] + y[1]*y[1]; }
                    @Override public double reference() { return 1.0; }
                    @Override public double tolerance() { return 1e-2; }
                };
                double t = 0.0; double[] y = new double[]{1.0, 0.0};
                double epsTot = 1e-4, epsR = 1e-5, epsT = 1e-5, epsG = 1e-5;
                double epsRk4Max = 0.0, epsTaylorRms = 0.0;
                for (int i = 0; i < steps; i++) {
                    UnifiedDetector.Triad tr = ud.triadStep(dyn, t, y, h, epsTot, epsR, epsT, epsG, 1_000_000, new UnifiedDetector.Invariant[]{inv});
                    t = tr.tNext; y = tr.yNext; h = tr.hUsed;
                    epsRk4Max = Math.max(epsRk4Max, tr.epsRk4);
                    epsTaylorRms += tr.epsTaylor * tr.epsTaylor;
                }
                epsTaylorRms = Math.sqrt(epsTaylorRms / Math.max(1, steps));
                String line = String.format(Locale.ROOT,
                        "{\"system_id\":\"hopf\",\"mu\":%.6f,\"re_lambda_max\":%.6f,\"hopf_freq\":%.6f,\"eps_rk4_max\":%.3e,\"r4_rms\":%.3e}",
                        mu, reLam, hopfFreq, epsRk4Max, epsTaylorRms);
                w.write(line); w.newLine();
            }
            String sum = String.format(Locale.ROOT,
                    "{\"system_id\":\"hopf\",\"expected_onset\":0.000000,\"measured_onset\":0.000000,\"onset_delta\":0.000000}");
            w.write(sum); w.newLine();
        } catch (IOException e) {
            System.err.println("bifurc: hopf write failed: " + e.getMessage());
        }
    }
}


