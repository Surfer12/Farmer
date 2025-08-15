// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.function.DoubleUnaryOperator;

/**
 * Minimal RK4 integrator with PI step-size controller targeting a total error budget ε_total.
 * For this scaffold, we expose a single-step estimate function and a controller that adjusts h.
 */
public final class RK4Controller {
    public static final class StepResult {
        public final double yNew;
        public final double hUsed;
        public final double errEst;
        StepResult(double yNew, double hUsed, double errEst) { this.yNew = yNew; this.hUsed = hUsed; this.errEst = errEst; }
    }

    private final double kP;
    private final double kI;
    private double integral;

    public RK4Controller(double kP, double kI) {
        this.kP = kP; this.kI = kI; this.integral = 0.0;
    }

    /** One adaptive step attempting to meet eps budget by adjusting h. */
    public StepResult step(DoubleUnaryOperator f, double t, double y, double hInit, double epsBudget) {
        double h = Math.max(1e-9, hInit);
        for (int it = 0; it < 8; it++) {
            double yRK4 = rk4(f, t, y, h);
            double yHalf = rk4(f, t, y, h * 0.5);
            yHalf = rk4(f, t + 0.5 * h, yHalf, h * 0.5);
            double err = Math.abs(yRK4 - yHalf);
            double e = (err - epsBudget);
            integral += e;
            double scale = Math.exp(-(kP * e + kI * integral));
            double hNext = Math.max(1e-9, h * Math.min(2.0, Math.max(0.5, scale)));
            if (err <= epsBudget) {
                return new StepResult(yHalf, h, err);
            }
            h = hNext;
        }
        double yRK4 = rk4(f, t, y, h);
        return new StepResult(yRK4, h, Double.NaN);
    }

    private static double rk4(DoubleUnaryOperator f, double t, double y, double h) {
        double k1 = f.applyAsDouble(t);
        double k2 = f.applyAsDouble(t + 0.5 * h);
        double k3 = f.applyAsDouble(t + 0.5 * h);
        double k4 = f.applyAsDouble(t + h);
        return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
    }
}


