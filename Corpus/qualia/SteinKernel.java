// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

/**
 * Base interface for kernels used in Stein reproducing kernels.
 */
interface SteinKernel {
    /** Returns k(x, y). */
    double k(double[] x, double[] y);

    /** Writes ∇_x k(x, y) into out (length d). */
    void grad1(double[] x, double[] y, double[] out);

    /** Writes ∇_y k(x, y) into out (length d). */
    void grad2(double[] x, double[] y, double[] out);

    /** Returns ∇_x · ∇_y k(x, y). */
    double div12(double[] x, double[] y);
}


