package qualia;

import java.util.ArrayList;
import java.util.List;

/**
 * Convergence diagnostics and effective sample size estimates for MCMC chains.
 */
public final class Diagnostics {
    public final double rHatS;
    public final double rHatN;
    public final double rHatAlpha;
    public final double rHatBeta;

    public final double essS;
    public final double essN;
    public final double essAlpha;
    public final double essBeta;

    public Diagnostics(double rHatS, double rHatN, double rHatAlpha, double rHatBeta,
                       double essS, double essN, double essAlpha, double essBeta) {
        this.rHatS = rHatS;
        this.rHatN = rHatN;
        this.rHatAlpha = rHatAlpha;
        this.rHatBeta = rHatBeta;
        this.essS = essS;
        this.essN = essN;
        this.essAlpha = essAlpha;
        this.essBeta = essBeta;
    }

    /**
     * Computes R̂ (Gelman-Rubin) and ESS for each parameter across multiple chains.
     */
    public static Diagnostics fromChains(List<List<ModelParameters>> chains) {
        if (chains == null || chains.isEmpty()) {
            return new Diagnostics(Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                    Double.NaN, Double.NaN, Double.NaN, Double.NaN);
        }

        List<double[]> sChains = new ArrayList<>();
        List<double[]> nChains = new ArrayList<>();
        List<double[]> aChains = new ArrayList<>();
        List<double[]> bChains = new ArrayList<>();

        for (List<ModelParameters> ch : chains) {
            int n = ch.size();
            double[] s = new double[n];
            double[] nArr = new double[n];
            double[] a = new double[n];
            double[] b = new double[n];
            for (int i = 0; i < n; i++) {
                ModelParameters p = ch.get(i);
                s[i] = p.S();
                nArr[i] = p.N();
                a[i] = p.alpha();
                b[i] = p.beta();
            }
            sChains.add(s);
            nChains.add(nArr);
            aChains.add(a);
            bChains.add(b);
        }

        double rS = rHat(sChains);
        double rN = rHat(nChains);
        double rA = rHat(aChains);
        double rB = rHat(bChains);

        double essS = ess(flatten(sChains));
        double essN = ess(flatten(nChains));
        double essA = ess(flatten(aChains));
        double essB = ess(flatten(bChains));

        return new Diagnostics(rS, rN, rA, rB, essS, essN, essA, essB);
    }

    private static double rHat(List<double[]> chains) {
        int m = chains.size();
        if (m < 2) return Double.NaN; // requires >= 2 chains
        int n = chains.get(0).length;
        if (n < 2) return Double.NaN;

        double[] means = new double[m];
        double meanOfMeans = 0.0;
        for (int j = 0; j < m; j++) {
            means[j] = mean(chains.get(j));
            meanOfMeans += means[j];
        }
        meanOfMeans /= m;

        double B = 0.0;
        for (int j = 0; j < m; j++) {
            double diff = means[j] - meanOfMeans;
            B += diff * diff;
        }
        B *= (double) n / (m - 1);

        double W = 0.0;
        for (int j = 0; j < m; j++) {
            W += variance(chains.get(j));
        }
        W /= m;

        double varPlus = ((n - 1.0) / n) * W + (B / n);
        return Math.sqrt(varPlus / W);
    }

    private static double[] flatten(List<double[]> chains) {
        int total = 0;
        for (double[] c : chains) total += c.length;
        double[] out = new double[total];
        int k = 0;
        for (double[] c : chains) {
            System.arraycopy(c, 0, out, k, c.length);
            k += c.length;
        }
        return out;
    }

    private static double ess(double[] series) {
        int n = series.length;
        if (n < 3) return Double.NaN;
        double[] centered = series.clone();
        double mu = mean(centered);
        for (int i = 0; i < n; i++) centered[i] -= mu;
        double var = variance(centered);
        if (var == 0.0) return n;

        int maxLag = Math.min(1000, n / 2);
        double sum = 0.0;
        for (int lag = 1; lag <= maxLag; lag++) {
            double rho = autocorr(centered, lag) / var;
            if (rho <= 0) break; // initial-positive sequence truncation
            sum += rho;
        }
        double ess = n / (1.0 + 2.0 * sum);
        return Math.max(1.0, Math.min(ess, n));
    }

    private static double autocorr(double[] x, int lag) {
        int n = x.length;
        double s = 0.0;
        for (int i = lag; i < n; i++) s += x[i] * x[i - lag];
        return s / (n - lag);
    }

    private static double mean(double[] x) {
        double s = 0.0;
        for (double v : x) s += v;
        return s / x.length;
    }

    private static double variance(double[] x) {
        double mu = mean(x);
        double s2 = 0.0;
        for (double v : x) {
            double d = v - mu;
            s2 += d * d;
        }
        return s2 / (x.length - 1);
    }
}


