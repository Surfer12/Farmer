package qualia;

import java.util.Collection;
import java.util.Map;
import java.util.Objects;
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

    /**
     * Returns per-decision total weights for reporting.
     */
    public static Map<Decision, Double> weightBreakdown(Collection<Vote> votes) {
        return votes.stream().collect(Collectors.groupingBy(Vote::decision, Collectors.summingDouble(Vote::weight)));
    }
}


