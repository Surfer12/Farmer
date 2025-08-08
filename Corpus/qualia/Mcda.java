// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.*;

/**
 * Minimal MCDA utilities with Ψ integration.
 *
 * Pipeline:
 * - Gate by Ψ: F_tau = { a | psi(a) >= tau }
 * - Normalize criteria per Theorem 1 using min–max; cost criteria are flipped
 * - Aggregators: WSM, WPM, TOPSIS (monotone in Ψ when w_psi>0)
 */
public final class Mcda {
    private Mcda() {}

    public enum Direction { BENEFIT, COST }

    public record CriterionSpec(String name, Direction direction, double weight) {
        public CriterionSpec {
            Objects.requireNonNull(name);
            Objects.requireNonNull(direction);
            if (!(weight >= 0.0)) throw new IllegalArgumentException("weight must be >= 0");
        }
    }

    public static final class Alternative {
        public final String id;
        public final Map<String, Double> rawScores; // name -> x_j(a)
        public final double psi; // Ψ(a) in [0,1]

        public Alternative(String id, Map<String, Double> rawScores, double psi) {
            this.id = Objects.requireNonNull(id);
            this.rawScores = Collections.unmodifiableMap(new HashMap<>(rawScores));
            if (psi < 0.0 || psi > 1.0) throw new IllegalArgumentException("psi must be in [0,1]");
            this.psi = psi;
        }

        @Override public String toString() { return id; }
    }

    public record Ranked(Alternative alternative, double score) {}

    public static List<Alternative> gateByPsi(Collection<Alternative> alts, double tau) {
        ArrayList<Alternative> out = new ArrayList<>();
        for (Alternative a : alts) if (a.psi >= tau) out.add(a);
        return out;
    }

    /**
     * Normalizes each criterion to [0,1]. For name "psi", returns Ψ directly.
     */
    public static Map<Alternative, Map<String, Double>> normalize(Collection<Alternative> alts,
                                                                  List<CriterionSpec> specs) {
        Map<Alternative, Map<String, Double>> z = new HashMap<>();
        if (alts.isEmpty()) return z;

        // Precompute min/max per criterion across alts
        Map<String, double[]> minMax = new HashMap<>(); // name -> [min,max]
        for (CriterionSpec c : specs) {
            if ("psi".equals(c.name())) { continue; }
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (Alternative a : alts) {
                Double xv = a.rawScores.get(c.name());
                if (xv == null || xv.isNaN() || xv.isInfinite()) continue;
                min = Math.min(min, xv);
                max = Math.max(max, xv);
            }
            minMax.put(c.name(), new double[] { min, max });
        }

        // Normalize per alternative
        for (Alternative a : alts) {
            Map<String, Double> zi = new HashMap<>();
            for (CriterionSpec c : specs) {
                double val;
                if ("psi".equals(c.name())) {
                    val = a.psi;
                } else {
                    Double x = a.rawScores.get(c.name());
                    if (x == null) x = 0.0;
                    double[] mm = minMax.get(c.name());
                    double min = mm[0], max = mm[1];
                    if (!(max > min)) {
                        // degenerate criterion; treat as constant 0.5
                        val = 0.5;
                    } else if (c.direction() == Direction.BENEFIT) {
                        val = (x - min) / (max - min);
                    } else {
                        val = (max - x) / (max - min);
                    }
                    // Clamp for numerical safety
                    if (val < 0.0) val = 0.0; else if (val > 1.0) val = 1.0;
                }
                zi.put(c.name(), val);
            }
            z.put(a, Collections.unmodifiableMap(zi));
        }
        return z;
    }

    public static List<Ranked> rankByWSM(Collection<Alternative> alts,
                                          List<CriterionSpec> specs) {
        Map<Alternative, Map<String, Double>> z = normalize(alts, specs);
        ArrayList<Ranked> out = new ArrayList<>(z.size());
        for (Alternative a : alts) {
            double s = 0.0;
            for (CriterionSpec c : specs) s += c.weight() * z.get(a).get(c.name());
            out.add(new Ranked(a, s));
        }
        out.sort((r1, r2) -> -Double.compare(r1.score, r2.score));
        return out;
    }

    public static List<Ranked> rankByWPM(Collection<Alternative> alts,
                                          List<CriterionSpec> specs) {
        Map<Alternative, Map<String, Double>> z = normalize(alts, specs);
        double eps = 1e-12;
        ArrayList<Ranked> out = new ArrayList<>(z.size());
        for (Alternative a : alts) {
            double sumWLog = 0.0;
            for (CriterionSpec c : specs) {
                double zj = Math.max(z.get(a).get(c.name()), eps);
                sumWLog += c.weight() * Math.log(zj);
            }
            double s = Math.exp(sumWLog);
            out.add(new Ranked(a, s));
        }
        out.sort((r1, r2) -> -Double.compare(r1.score, r2.score));
        return out;
    }

    public static List<Ranked> rankByTOPSIS(Collection<Alternative> alts,
                                             List<CriterionSpec> specs) {
        Map<Alternative, Map<String, Double>> z = normalize(alts, specs);
        // Weighted matrix V_j(a) = w_j z_j(a)
        Map<Alternative, Map<String, Double>> V = new HashMap<>();
        Map<String, Double> vStar = new HashMap<>();
        Map<String, Double> vMinus = new HashMap<>();
        for (CriterionSpec c : specs) {
            double vmax = Double.NEGATIVE_INFINITY;
            double vmin = Double.POSITIVE_INFINITY;
            for (Alternative a : alts) {
                double v = c.weight() * z.get(a).get(c.name());
                V.computeIfAbsent(a, _k -> new HashMap<>()).put(c.name(), v);
                vmax = Math.max(vmax, v);
                vmin = Math.min(vmin, v);
            }
            vStar.put(c.name(), vmax);
            vMinus.put(c.name(), vmin);
        }
        ArrayList<Ranked> out = new ArrayList<>(alts.size());
        for (Alternative a : alts) {
            double dPlus = 0.0, dMinus = 0.0;
            for (CriterionSpec c : specs) {
                double v = V.get(a).get(c.name());
                double vp = vStar.get(c.name());
                double vm = vMinus.get(c.name());
                dPlus += (v - vp) * (v - vp);
                dMinus += (v - vm) * (v - vm);
            }
            dPlus = Math.sqrt(dPlus);
            dMinus = Math.sqrt(dMinus);
            double cc = dMinus / Math.max(dPlus + dMinus, 1e-12);
            out.add(new Ranked(a, cc));
        }
        out.sort((r1, r2) -> -Double.compare(r1.score, r2.score));
        return out;
    }

    /** Validates weights sum to ~1 and w_j>=0. */
    public static void validateWeights(List<CriterionSpec> specs) {
        double sum = 0.0;
        for (CriterionSpec c : specs) {
            if (!(c.weight() >= 0.0)) throw new IllegalArgumentException("negative weight: " + c);
            sum += c.weight();
        }
        if (Math.abs(sum - 1.0) > 1e-6) throw new IllegalArgumentException("weights must sum to 1 (got " + sum + ")");
    }
}


