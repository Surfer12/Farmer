// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Aggregates stakeholder votes with support for explicit overrides.
 *
 * <p>Rules:
 * - Any override REJECT trumps everything ⇒ REJECT.
 * - Else any override APPROVE trumps weighted tally ⇒ APPROVE.
 * - Else weighted majority on APPROVE/REJECT; ESCALATE if tie/insufficient.
 */
public final class VotingPolicy {
    private VotingPolicy() {}

    /**
     * Aggregates votes into a single decision.
     */
    public static Decision decide(Collection<Vote> votes) {
        Objects.requireNonNull(votes, "votes");
        if (votes.isEmpty()) return Decision.ESCALATE;

        boolean anyOverrideReject = votes.stream().anyMatch(v -> v.override() && v.decision() == Decision.REJECT);
        if (anyOverrideReject) return Decision.REJECT;

        boolean anyOverrideApprove = votes.stream().anyMatch(v -> v.override() && v.decision() == Decision.APPROVE);
        if (anyOverrideApprove) return Decision.APPROVE;

        double approveWeight = votes.stream().filter(v -> v.decision() == Decision.APPROVE).mapToDouble(Vote::weight).sum();
        double rejectWeight  = votes.stream().filter(v -> v.decision() == Decision.REJECT ).mapToDouble(Vote::weight).sum();

        if (approveWeight > rejectWeight) return Decision.APPROVE;
        if (rejectWeight > approveWeight) return Decision.REJECT;
        return Decision.ESCALATE;
    }

    /** Supermajority threshold on weighted votes after overrides. */
    public static Decision decideSupermajority(Collection<Vote> votes, double threshold) {
        Objects.requireNonNull(votes, "votes");
        if (votes.isEmpty()) return Decision.ESCALATE;
        Decision d = applyOverrides(votes);
        if (d != Decision.ESCALATE) return d;
        double approve = votes.stream().filter(v -> v.decision()==Decision.APPROVE).mapToDouble(Vote::weight).sum();
        double reject  = votes.stream().filter(v -> v.decision()==Decision.REJECT ).mapToDouble(Vote::weight).sum();
        double total = approve + reject;
        if (total == 0.0) return Decision.ESCALATE;
        if (approve / total >= threshold) return Decision.APPROVE;
        if (reject  / total >= threshold) return Decision.REJECT;
        return Decision.ESCALATE;
    }

    /** Quorum with optional role veto, then fallback to simple decide. */
    public static Decision decideWithQuorum(Collection<Vote> votes, int minApprovers, int minRejectors, Set<Role> vetoRoles) {
        Objects.requireNonNull(votes, "votes");
        if (votes.isEmpty()) return Decision.ESCALATE;
        boolean veto = votes.stream().anyMatch(v -> v.override() && v.decision()==Decision.REJECT && vetoRoles!=null && vetoRoles.contains(v.role()));
        if (veto) return Decision.REJECT;
        long approvers = votes.stream().filter(v -> v.decision()==Decision.APPROVE).count();
        long rejectors = votes.stream().filter(v -> v.decision()==Decision.REJECT ).count();
        if (approvers < minApprovers && rejectors < minRejectors) return Decision.ESCALATE;
        return decide(votes);
    }

    /** Time-decayed weighting based on vote expiry proximity; overrides first. */
    public static Decision decideTimeDecay(Collection<Vote> votes, java.time.Duration halfLife, java.time.Instant now) {
        Objects.requireNonNull(votes, "votes");
        if (votes.isEmpty()) return Decision.ESCALATE;
        Decision od = applyOverrides(votes);
        if (od != Decision.ESCALATE) return od;
        double lambda = Math.log(2.0) / Math.max(1L, halfLife.toSeconds());
        java.util.function.ToDoubleFunction<Vote> w = v -> v.weight() * Math.exp(-lambda * Math.abs(java.time.Duration.between((v.expiresAt()!=null?v.expiresAt():now), now).toSeconds()));
        double approve = votes.stream().filter(v -> v.decision()==Decision.APPROVE).mapToDouble(w).sum();
        double reject  = votes.stream().filter(v -> v.decision()==Decision.REJECT ).mapToDouble(w).sum();
        if (approve > reject) return Decision.APPROVE;
        if (reject  > approve) return Decision.REJECT;
        return Decision.ESCALATE;
    }

    /** Chainable rules; returns first non-ESCALATE decision. */
    public interface VotingRule { Decision apply(Collection<Vote> votes); }

    public static Decision decide(Collection<Vote> votes, List<VotingRule> rules) {
        Objects.requireNonNull(votes, "votes");
        for (VotingRule r : rules) {
            Decision d = r.apply(votes);
            if (d != Decision.ESCALATE) return d;
        }
        return Decision.ESCALATE;
    }

    private static Decision applyOverrides(Collection<Vote> votes) {
        boolean anyOverrideReject = votes.stream().anyMatch(v -> v.override() && v.decision() == Decision.REJECT);
        if (anyOverrideReject) return Decision.REJECT;
        boolean anyOverrideApprove = votes.stream().anyMatch(v -> v.override() && v.decision() == Decision.APPROVE);
        if (anyOverrideApprove) return Decision.APPROVE;
        return Decision.ESCALATE;
    }

    /**
     * Returns per-decision total weights for reporting.
     */
    public static Map<Decision, Double> weightBreakdown(Collection<Vote> votes) {
        return votes.stream().collect(Collectors.groupingBy(Vote::decision, Collectors.summingDouble(Vote::weight)));
    }
}


