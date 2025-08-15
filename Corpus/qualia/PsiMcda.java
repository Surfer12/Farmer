// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.*;
import java.util.function.ToDoubleFunction;

/**
 * Public, side-effect free methods for Ψ scoring and MCDA utilities.
 *
 * All methods validate inputs, are deterministic, and optimize for clarity.
 */
public final class PsiMcda {
    private PsiMcda() {}

    public enum Direction { BENEFIT, COST }
    public enum Aggregator { MEAN, SOFTCAP }

    public static record PsiResult(double psi, double O, double pen, double post) {}
    public static record GradWSM(double[] dU_dw, double[] dU_dz) {}
    public static record GradPsi(double dAlpha, double dRa, double dRv, double dS, double dN) {}
    public static record AHPResult(double[] weights, double consistencyRatio) {}

    // --- Ψ scoring ---
    public static PsiResult computePsi(double S, double N, double alpha,
                                       double Ra, double Rv, double lambda1, double lambda2, double beta) {
        requireRange(S, 0.0, 1.0, "S");
        requireRange(N, 0.0, 1.0, "N");
        requireRange(alpha, 0.0, 1.0, "alpha");
        if (Ra < 0 || Rv < 0) throw new IllegalArgumentException("Risks must be nonnegative");
        if (lambda1 <= 0 || lambda2 <= 0) throw new IllegalArgumentException("Risk weights must be > 0");
        if (beta < 1.0) throw new IllegalArgumentException("beta must be >= 1");

        double O = alpha * S + (1.0 - alpha) * N;
        double pen = Math.exp(-(lambda1 * Ra + lambda2 * Rv));
        double post = Math.min(1.0, beta * 1.0); // placeholder: caller can fold P(H|E) externally if needed
        // In our framework post is min{beta * P(H|E), 1}. If caller provides P(H|E), overload as needed.
        double psi = Math.min(1.0, O * pen * post);
        psi = Math.max(0.0, Math.min(1.0, psi));
        return new PsiResult(psi, O, pen, post);
    }

    /** Temporal aggregation over time steps with weights w (sum=1). 
     *  MEAN: weighted mean of Ψ_t; SOFTCAP: compute raw R_t=beta_t O_t pen_t then Ψ=1-exp(-∑ w_t R_t). */
    public static double computePsiTemporal(double[] w, double[] S, double[] N, double[] alpha,
                                            double[] Ra, double[] Rv, double[] beta,
                                            double lambda1, double lambda2, Aggregator aggregator) {
        Objects.requireNonNull(w); Objects.requireNonNull(S); Objects.requireNonNull(N);
        Objects.requireNonNull(alpha); Objects.requireNonNull(Ra); Objects.requireNonNull(Rv);
        Objects.requireNonNull(beta); Objects.requireNonNull(aggregator);
        int n = S.length;
        if (! (N.length==n && alpha.length==n && Ra.length==n && Rv.length==n && beta.length==n && w.length==n))
            throw new IllegalArgumentException("Arrays must have same length");
        double sumW = 0.0; for (double v: w) sumW += v; if (Math.abs(sumW - 1.0) > 1e-9) throw new IllegalArgumentException("w must sum to 1");

        switch (aggregator) {
            case MEAN -> {
                double acc = 0.0;
                for (int i = 0; i < n; i++) {
                    PsiResult r = computePsi(S[i], N[i], alpha[i], Ra[i], Rv[i], lambda1, lambda2, beta[i]);
                    acc += w[i] * r.psi();
                }
                return clamp01(acc);
            }
            case SOFTCAP -> {
                double raw = 0.0;
                for (int i = 0; i < n; i++) {
                    double O = alpha[i] * S[i] + (1.0 - alpha[i]) * N[i];
                    double pen = Math.exp(-(lambda1 * Ra[i] + lambda2 * Rv[i]));
                    double R = beta[i] * O * pen; // raw belief channel, uncapped
                    raw += w[i] * Math.max(0.0, R);
                }
                return 1.0 - Math.exp(-raw);
            }
            default -> throw new IllegalArgumentException("Unknown aggregator");
        }
    }

    public static double thresholdTransfer(double kappa, double tau, String mode) {
        requireRange(tau, 0.0, 1.0, "tau");
        if (kappa <= 0) throw new IllegalArgumentException("kappa must be > 0");
        return switch (Objects.requireNonNull(mode).toLowerCase(Locale.ROOT)) {
            case "subcap" -> clamp01(kappa * tau);
            case "softcap" -> clamp01(1.0 - Math.pow(1.0 - tau, kappa));
            default -> throw new IllegalArgumentException("mode must be 'subcap' or 'softcap'");
        };
    }

    // --- Normalization ---
    public static double[] normalizeCriterion(double[] values, Direction direction) {
        Objects.requireNonNull(values); Objects.requireNonNull(direction);
        if (values.length == 0) return new double[0];
        double min = Arrays.stream(values).min().orElse(0.0);
        double max = Arrays.stream(values).max().orElse(0.0);
        double denom = Math.max(1e-12, max - min);
        double[] z = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            double v = values[i];
            double zi = (direction == Direction.BENEFIT) ? (v - min) / denom : (max - v) / denom;
            z[i] = clamp01(zi);
        }
        return z;
    }

    // --- Governance weights ---
    public static double[] mapGovernanceWeights(double[] baseWeights, double gPsi, double eta) {
        Objects.requireNonNull(baseWeights);
        if (eta <= 0 || eta > 1) throw new IllegalArgumentException("eta in (0,1]");
        requireRange(gPsi, 0.0, 1.0, "gPsi");
        double sum = Arrays.stream(baseWeights).sum();
        if (Math.abs(sum - 1.0) > 1e-9) throw new IllegalArgumentException("baseWeights must sum to 1");
        // Allocate a dynamic share to Ψ, distribute the remainder across base weights so the total sums to 1
        double wPsi = eta * gPsi;                 // dynamic governance share for Ψ in [0, eta]
        double remaining = Math.max(0.0, 1.0 - wPsi);
        double[] w = new double[baseWeights.length + 1];
        w[0] = wPsi;
        for (int i = 0; i < baseWeights.length; i++) {
            w[i + 1] = remaining * baseWeights[i];
        }
        return w;
    }

    // --- Gating ---
    public static <T> List<T> gateByPsi(List<T> alternatives, ToDoubleFunction<T> psiLowerFn, double tau) {
        Objects.requireNonNull(alternatives); Objects.requireNonNull(psiLowerFn);
        requireRange(tau, 0.0, 1.0, "tau");
        List<T> out = new ArrayList<>();
        for (T a : alternatives) {
            if (psiLowerFn.applyAsDouble(a) >= tau) out.add(a);
        }
        return out;
    }

    // --- Scores ---
    public static double wsmScore(double[] w, double[] z) {
        checkSameLength(w, z);
        double sum = 0.0;
        for (int i = 0; i < w.length; i++) sum += w[i] * z[i];
        return sum;
    }
    public static double wsmScoreRobust(double[] w, double[] zLower) { return wsmScore(w, zLower); }

    public static double wpmScore(double[] w, double[] z) {
        checkSameLength(w, z);
        double prod = 1.0; double eps = 1e-12;
        for (int i = 0; i < w.length; i++) prod *= Math.pow(Math.max(z[i], eps), w[i]);
        return prod;
    }
    public static double wpmScoreRobust(double[] w, double[] zLower) { return wpmScore(w, zLower); }

    // --- TOPSIS (expects inputs already normalized to [0,1] per criterion) ---
    public static double[] topsisCloseness(double[][] z, double[] w, Direction[] directions) {
        Objects.requireNonNull(z); Objects.requireNonNull(w); Objects.requireNonNull(directions);
        int n = z.length; if (n == 0) return new double[0];
        int m = z[0].length; if (w.length != m || directions.length != m) throw new IllegalArgumentException("shape");
        // Weighted matrix V
        double[][] V = new double[n][m];
        for (int i = 0; i < n; i++) {
            if (z[i].length != m) throw new IllegalArgumentException("ragged z");
            for (int j = 0; j < m; j++) V[i][j] = w[j] * z[i][j];
        }
        // Ideals
        double[] vStar = new double[m], vMinus = new double[m];
        for (int j = 0; j < m; j++) {
            double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) { double v = V[i][j]; if (v < min) min = v; if (v > max) max = v; }
            if (directions[j] == Direction.BENEFIT) { vStar[j] = max; vMinus[j] = min; }
            else { vStar[j] = min; vMinus[j] = max; }
        }
        // Distances and closeness
        double[] closeness = new double[n];
        for (int i = 0; i < n; i++) {
            double dPos = 0.0, dNeg = 0.0;
            for (int j = 0; j < m; j++) {
                double dvPos = V[i][j] - vStar[j]; dPos += dvPos * dvPos;
                double dvNeg = V[i][j] - vMinus[j]; dNeg += dvNeg * dvNeg;
            }
            dPos = Math.sqrt(dPos); dNeg = Math.sqrt(dNeg);
            closeness[i] = (dPos + dNeg) <= 1e-12 ? 0.0 : dNeg / (dPos + dNeg);
        }
        return closeness;
    }

    // --- AHP principal eigenvector (power iteration) and CR ---
    public static AHPResult ahpWeights(double[][] pairwise) {
        Objects.requireNonNull(pairwise);
        int n = pairwise.length; if (n == 0) return new AHPResult(new double[0], 0.0);
        for (double[] row : pairwise) if (row.length != n) throw new IllegalArgumentException("matrix not square");
        double[] w = new double[n]; Arrays.fill(w, 1.0 / n);
        double[] tmp = new double[n];
        int iters = 0; double lambda = 0.0;
        while (iters++ < 1000) {
            Arrays.fill(tmp, 0.0);
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) tmp[i] += pairwise[i][j] * w[j];
            double sum = Arrays.stream(tmp).sum(); if (sum <= 0) throw new IllegalArgumentException("invalid matrix");
            for (int i = 0; i < n; i++) w[i] = tmp[i] / sum;
            // Rayleigh quotient approximation of lambda_max
            double[] Aw = new double[n];
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Aw[i] += pairwise[i][j] * w[j];
            double num = 0.0, den = 0.0; for (int i = 0; i < n; i++) { num += Aw[i] * w[i]; den += w[i] * w[i]; }
            lambda = num / Math.max(den, 1e-12);
            if (l1diff(w, tmpNormalize(Aw)) < 1e-9) break;
        }
        double ci = (lambda - n) / Math.max(n - 1, 1);
        double ri = randomIndexFor(n);
        double cr = (ri <= 0) ? 0.0 : Math.max(0.0, ci / ri);
        return new AHPResult(w, cr);
    }

    // --- Bounds mapping ---
    public static Bounds zBoundsFromXBounds(double xLower, double xUpper, double minX, double maxX, Direction dir) {
        if (maxX - minX <= 1e-12) return new Bounds(0.0, 0.0);
        if (dir == Direction.BENEFIT) {
            double zl = (xLower - minX) / (maxX - minX);
            double zu = (xUpper - minX) / (maxX - minX);
            return new Bounds(clamp01(zl), clamp01(zu));
        } else {
            double zl = (maxX - xUpper) / (maxX - minX);
            double zu = (maxX - xLower) / (maxX - minX);
            return new Bounds(clamp01(zl), clamp01(zu));
        }
    }
    public static record Bounds(double lower, double upper) {}

    // --- Sensitivities ---
    public static GradWSM gradWSM(double[] w, double[] z) {
        checkSameLength(w, z);
        double[] dU_dw = z.clone();
        double[] dU_dz = w.clone();
        return new GradWSM(dU_dw, dU_dz);
    }

    /** Sub-cap partials for Ψ. Includes dS and dN. */
    public static GradPsi gradPsi(double S, double N, double alpha,
                                  double Ra, double Rv, double lambda1, double lambda2, double beta) {
        double O = alpha * S + (1.0 - alpha) * N;
        double pen = Math.exp(-(lambda1 * Ra + lambda2 * Rv));
        double dAlpha = (S - N) * beta * pen;
        double dRa = -beta * O * lambda1 * pen;
        double dRv = -beta * O * lambda2 * pen;
        double dS = alpha * beta * pen;
        double dN = (1.0 - alpha) * beta * pen;
        return new GradPsi(dAlpha, dRa, dRv, dS, dN);
    }

    // --- Tie-break (lexicographic) ---
    public static <T> Optional<T> tieBreak(List<T> alternatives,
                                           ToDoubleFunction<T> primaryDesc, // e.g., U_min desc
                                           ToDoubleFunction<T> secondaryDesc, // e.g., psiLower desc
                                           ToDoubleFunction<T> costAsc) {
        Objects.requireNonNull(alternatives);
        // Build a comparator that makes the desired option the maximum:
        // - Higher primary is better (ascending comparator + max)
        // - Higher secondary is better (ascending comparator + max)
        // - Lower cost is better (compare negative cost so higher -cost wins)
        Comparator<T> cmp = Comparator
                .comparingDouble(primaryDesc)
                .thenComparingDouble(secondaryDesc)
                .thenComparingDouble(a -> -costAsc.applyAsDouble(a));
        return alternatives.stream().max(cmp);
    }

    // --- helpers ---
    private static void requireRange(double v, double lo, double hi, String name) {
        if (v < lo || v > hi) throw new IllegalArgumentException(name + " must be in [" + lo + "," + hi + "]");
    }
    private static void checkSameLength(double[] a, double[] b) {
        Objects.requireNonNull(a); Objects.requireNonNull(b);
        if (a.length != b.length) throw new IllegalArgumentException("length mismatch");
    }
    private static double clamp01(double x) { return (x < 0.0) ? 0.0 : (x > 1.0 ? 1.0 : x); }
    private static double l1diff(double[] a, double[] b) {
        if (a.length != b.length) return Double.POSITIVE_INFINITY;
        double s = 0.0; for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s;
    }
    private static double[] tmpNormalize(double[] v) {
        double s = Arrays.stream(v).sum(); if (s <= 0) return v.clone();
        double[] r = new double[v.length]; for (int i = 0; i < v.length; i++) r[i] = v[i] / s; return r;
    }
    private static double randomIndexFor(int n) {
        // Saaty's RI values for n=1..10
        double[] RI = {0.0, 0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49};
        return (n >= 0 && n < RI.length) ? RI[n] : 1.49; // default to 10-size RI for larger n
    }
}


