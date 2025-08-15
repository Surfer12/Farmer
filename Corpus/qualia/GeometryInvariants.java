// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions;

/**
 * Geometric invariants checks for Ψ: monotonicity signs and band constraints.
 */
public final class GeometryInvariants {
    public static final class Check {
        public final boolean ok;
        public final double epsGeom; // penalty contribution to error budget
        Check(boolean ok, double epsGeom) { this.ok = ok; this.epsGeom = Math.max(0.0, epsGeom); }
    }

    /**
     * Returns whether sensitivities match expected signs and computes a small penalty if not.
     */
    public static Check verify(HierarchicalBayesianModel.Prepared prep, ModelParameters p) {
        PsiSensitivity.Partials g = PsiSensitivity.partials(prep, p);
        boolean ok = true;
        if (g.dS < -1e-9) ok = false;
        if (g.dN < -1e-9) ok = false;
        if (p.S() >= p.N()) { if (g.dAlpha < -1e-9) ok = false; }
        else { if (g.dAlpha > 1e-9) ok = false; }
        if (g.dBeta < -1e-9) ok = false;
        double penalty = ok ? 0.0 : 0.05; // 5% budget hit if violated
        return new Check(ok, penalty);
    }
}


