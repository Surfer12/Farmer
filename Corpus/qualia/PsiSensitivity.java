// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


/**
 * Sensitivities of mean Ψ over a prepared dataset with respect to parameters (S,N,alpha,beta).
 * Assumes Ψ_i = O * pen_i * min{beta * P_i, 1} and returns partials of the mean Ψ.
 */
final class PsiSensitivity {

    static double meanPsi(HierarchicalBayesianModel.Prepared prep, ModelParameters p) {
        int n = prep.size();
        double O = p.alpha() * p.S() + (1.0 - p.alpha()) * p.N();
        double beta = p.beta();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double pHb = Math.min(beta * prep.pHe[i], 1.0);
            double psi = O * prep.pen[i] * pHb;
            if (psi < 0.0) psi = 0.0; else if (psi > 1.0) psi = 1.0;
            sum += psi;
        }
        return sum / Math.max(1, n);
    }

    static Partials partials(HierarchicalBayesianModel.Prepared prep, ModelParameters p) {
        int n = prep.size();
        double alpha = p.alpha();
        double S = p.S();
        double N = p.N();
        double beta = p.beta();

        double meanPenPbeta = 0.0;
        double meanPenP = 0.0;
        boolean anyCap = false;
        double O = alpha * S + (1.0 - alpha) * N;
        for (int i = 0; i < n; i++) {
            double P = prep.pHe[i];
            double pen = prep.pen[i];
            boolean capped = (beta * P >= 1.0);
            double pBeta = capped ? 1.0 : (beta * P);
            if (capped) anyCap = true;
            meanPenPbeta += pen * pBeta;
            meanPenP += pen * P;
        }
        meanPenPbeta /= Math.max(1, n);
        meanPenP /= Math.max(1, n);

        double dS = alpha * meanPenPbeta;
        double dN = (1.0 - alpha) * meanPenPbeta;
        double dAlpha = (S - N) * meanPenPbeta;
        double dBeta = anyCap ? dBetaCappedAware(prep, p) : (O * meanPenP);
        return new Partials(dS, dN, dAlpha, dBeta);
    }

    private static double dBetaCappedAware(HierarchicalBayesianModel.Prepared prep, ModelParameters p) {
        int n = prep.size();
        double O = p.alpha() * p.S() + (1.0 - p.alpha()) * p.N();
        double sum = 0.0; int m = 0;
        for (int i = 0; i < n; i++) {
            double P = prep.pHe[i];
            if (p.beta() * P < 1.0) { sum += O * prep.pen[i] * P; m++; }
        }
        if (m == 0) return 0.0;
        return sum / m;
    }

    static final class Partials {
        final double dS, dN, dAlpha, dBeta;
        Partials(double dS, double dN, double dAlpha, double dBeta) {
            this.dS = dS; this.dN = dN; this.dAlpha = dAlpha; this.dBeta = dBeta;
        }
    }
}


