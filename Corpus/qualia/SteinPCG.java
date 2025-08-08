// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.Arrays;
import java.util.List;

/**
 * Minimal preconditioned CG solver for the Stein linear system K_p w = 1.
 * Uses only matrix-vector products via the action callback.
 */
final class SteinPCG {
    interface Action {
        void apply(double[] v, double[] out);
    }

    interface Preconditioner {
        void applyInv(double[] r, double[] out);
    }

    /** Conjugate gradients with right preconditioning. */
    static double[] solve(Action A, Preconditioner M, int n, int maxIter, double tol) {
        double[] w = new double[n];
        double[] r = new double[n];
        double[] z = new double[n];
        double[] p = new double[n];
        double[] Ap = new double[n];

        // b = 1
        Arrays.fill(r, 1.0);
        A.apply(w, Ap); // zero -> 0
        for (int i = 0; i < n; i++) r[i] -= Ap[i];

        M.applyInv(r, z);
        System.arraycopy(z, 0, p, 0, n);

        double rz = dot(r, z);
        double bnorm = norm1(r);
        for (int it = 0; it < maxIter; it++) {
            A.apply(p, Ap);
            double alpha = rz / dot(p, Ap);
            axpy(alpha, p, w);
            axpy(-alpha, Ap, r);
            if (norm1(r) / bnorm < tol) break;
            M.applyInv(r, z);
            double rzNew = dot(r, z);
            double beta = rzNew / rz;
            for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
            rz = rzNew;
        }
        return w;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
    }
    private static void axpy(double a, double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) y[i] += a * x[i];
    }
    private static double norm1(double[] v) {
        double s = 0.0; for (double x : v) s += Math.abs(x); return s;
    }
}


