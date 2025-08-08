package qualia;

import java.util.List;

/**
 * Point-estimator c_N from Proposition 1: c_N = (f^T w) / (1^T w) with K_p w = 1.
 * Uses Gaussian RBF base kernel and canonical Stein kernel.
 */
public final class SteinEstimator {
    private final SteinKernel base;
    private final SteinGradLogP gradLogP;
    private final double[][] nodes; // collocation nodes in R^4 (S,N,alpha,beta)

    public SteinEstimator(double lengthScale,
                          HierarchicalBayesianModel model,
                          List<ClaimData> dataset,
                          List<ModelParameters> samples) {
        this.base = new GaussianRBFKernel(lengthScale);
        this.gradLogP = new SteinGradLogP(model, dataset);
        this.nodes = new double[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            ModelParameters p = samples.get(i);
            nodes[i] = new double[] { p.S(), p.N(), p.alpha(), p.beta() };
        }
    }

    /** Applies K_p to vector v. */
    private void actKp(double[] v, double[] out) {
        int n = nodes.length;
        java.util.Arrays.fill(out, 0.0);
        for (int i = 0; i < n; i++) {
            double[] xi = nodes[i];
            double[] gi = gradLogP.gradLogPosterior(xi.clone());
            for (int j = 0; j < n; j++) {
                double[] xj = nodes[j];
                double[] gj = gradLogP.gradLogPosterior(xj.clone());
                double k = base.k(xi, xj);
                double[] g1 = new double[gi.length];
                double[] g2 = new double[gi.length];
                base.grad1(xi, xj, g1);
                base.grad2(xi, xj, g2);
                double term = base.div12(xi, xj);
                double dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;
                for (int d = 0; d < gi.length; d++) {
                    dot1 += g1[d] * gj[d];
                    dot2 += gi[d] * g2[d];
                    dot3 += gi[d] * gj[d];
                }
                double kp = term + dot1 + dot2 + k * dot3;
                out[i] += kp * v[j];
            }
        }
    }

    /** Jacobi preconditioner using diag(K_p). */
    private void applyJacobiInv(double[] r, double[] out) {
        int n = nodes.length;
        for (int i = 0; i < n; i++) {
            double[] xi = nodes[i];
            double[] gi = gradLogP.gradLogPosterior(xi.clone());
            double diag = 0.0;
            // Approximate diag by kp(xi,xi)
            double[] g1 = new double[gi.length];
            double[] g2 = new double[gi.length];
            base.grad1(xi, xi, g1);
            base.grad2(xi, xi, g2);
            double k = base.k(xi, xi);
            double term = base.div12(xi, xi);
            double dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;
            for (int d = 0; d < gi.length; d++) {
                dot1 += g1[d] * gi[d];
                dot2 += gi[d] * g2[d];
                dot3 += gi[d] * gi[d];
            }
            diag = term + dot1 + dot2 + k * dot3;
            double pre = (Math.abs(diag) > 1e-12) ? (1.0 / diag) : 1.0;
            out[i] = pre * r[i];
        }
    }

    /** Returns c_N given values f(x_i). */
    public double estimate(double[] fValues, int maxIter, double tol) {
        int n = nodes.length;
        SteinPCG.Action A = this::actKp;
        SteinPCG.Preconditioner M = this::applyJacobiInv;
        double[] w = SteinPCG.solve(A, M, n, maxIter, tol);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; i++) { num += fValues[i] * w[i]; den += w[i]; }
        return num / den;
    }
}


