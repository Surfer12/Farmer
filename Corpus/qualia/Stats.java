// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

/**
 * Lightweight statistical utilities for priors and special functions.
 */
final class Stats {
    private Stats() {}

    // Lanczos coefficients for logGamma (g=7, n=9)
    private static final double[] LANCZOS = new double[] {
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
    };

    /**
     * Natural log of the Gamma function using the Lanczos approximation.
     */
    static double logGamma(double z) {
        if (Double.isNaN(z)) return Double.NaN;
        if (z < 0.5) {
            // Reflection formula: Γ(z)Γ(1−z) = π / sin(πz)
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1.0 - z);
        }
        z -= 1.0;
        double x = LANCZOS[0];
        for (int i = 1; i < LANCZOS.length; i++) {
            x += LANCZOS[i] / (z + i);
        }
        double t = z + LANCZOS.length - 0.5;
        return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }

    static double logBeta(double a, double b) {
        return logGamma(a) + logGamma(b) - logGamma(a + b);
    }

    /**
     * Log-density for Beta(x; a, b) for x in (0,1).
     */
    static double logBetaPdf(double x, double a, double b) {
        if (!(a > 0.0) || !(b > 0.0)) return Double.NEGATIVE_INFINITY;
        // Clamp to avoid log(0)
        double eps = 1e-12;
        if (x <= 0.0) x = eps;
        if (x >= 1.0) x = 1.0 - eps;
        return (a - 1.0) * Math.log(x) + (b - 1.0) * Math.log(1.0 - x) - logBeta(a, b);
    }

    /**
     * Log-density for LogNormal(x; mu, sigma) with x>0, sigma>0.
     */
    static double logLogNormalPdf(double x, double mu, double sigma) {
        if (!(x > 0.0) || !(sigma > 0.0)) return Double.NEGATIVE_INFINITY;
        double logx = Math.log(x);
        double z = (logx - mu) / sigma;
        return -0.5 * z * z - Math.log(x * sigma * Math.sqrt(2.0 * Math.PI));
    }
}


