// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.util.Arrays;

/**
 * Gaussian RBF kernel k(x,y) = exp(-||x-y||^2 / (2 l^2)).
 * Provides derivatives needed for Stein kernel construction.
 */
final class GaussianRBFKernel implements SteinKernel {
    private final double lengthScale;
    private final double invTwoL2;

    GaussianRBFKernel(double lengthScale) {
        if (!(lengthScale > 0.0)) throw new IllegalArgumentException("lengthScale must be > 0");
        this.lengthScale = lengthScale;
        this.invTwoL2 = 1.0 / (2.0 * lengthScale * lengthScale);
    }

    @Override
    public double k(double[] x, double[] y) {
        double s = 0.0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            s += d * d;
        }
        return Math.exp(-s * invTwoL2);
    }

    @Override
    public void grad1(double[] x, double[] y, double[] out) {
        Arrays.fill(out, 0.0);
        double kval = k(x, y);
        double c = -kval / (lengthScale * lengthScale);
        for (int i = 0; i < x.length; i++) {
            out[i] = c * (x[i] - y[i]);
        }
    }

    @Override
    public void grad2(double[] x, double[] y, double[] out) {
        Arrays.fill(out, 0.0);
        double kval = k(x, y);
        double c = kval / (lengthScale * lengthScale);
        for (int i = 0; i < x.length; i++) {
            out[i] = c * (x[i] - y[i]);
        }
    }

    @Override
    public double div12(double[] x, double[] y) {
        int d = x.length;
        double kval = k(x, y);
        double l2 = lengthScale * lengthScale;
        double sum = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = x[i] - y[i];
            // ∂^2/∂x_i∂y_i k(x,y) = (1/l^2 - diff^2/l^4) k(x,y)
            sum += (1.0 / l2 - (diff * diff) / (l2 * l2)) * kval;
        }
        return sum;
    }
}


