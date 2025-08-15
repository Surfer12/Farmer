// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.time.Instant;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Ready-made voting rule stacks for common agent policies.
 */
@SuppressWarnings("unused")
public final class AgentPresets {
    private AgentPresets() {}

    /**
     * Safety-critical: overrideReject(Security) → quorum(SEC+LEGAL) → supermajority(0.8) → ESCALATE.
     */
    public static Decision safetyCritical(Collection<Vote> votes) {
        VotingPolicy.VotingRule overrideRejectSecurity = vs ->
                vs.stream().anyMatch(v -> v.override() && v.decision()==Decision.REJECT && v.role()==Role.SECURITY)
                        ? Decision.REJECT : Decision.ESCALATE;

        VotingPolicy.VotingRule quorumSecLegalApprove = vs -> {
            boolean secOk = vs.stream().anyMatch(v -> v.role()==Role.SECURITY && v.decision()==Decision.APPROVE);
            boolean legalOk = vs.stream().anyMatch(v -> v.role()==Role.LEGAL && v.decision()==Decision.APPROVE);
            return (secOk && legalOk) ? Decision.APPROVE : Decision.ESCALATE;
        };

        VotingPolicy.VotingRule supermajority08 = vs -> {
            Decision d = VotingPolicy.decideSupermajority(vs, 0.80);
            return d != Decision.ESCALATE ? d : Decision.ESCALATE;
        };

        return VotingPolicy.decide(votes, List.of(overrideRejectSecurity, quorumSecLegalApprove, supermajority08));
    }

    /**
     * Fast path: overrideReject → supermajority(0.67) → simple majority → ESCALATE.
     */
    public static Decision fastPath(Collection<Vote> votes) {
        VotingPolicy.VotingRule overrideReject = vs ->
                vs.stream().anyMatch(v -> v.override() && v.decision()==Decision.REJECT)
                        ? Decision.REJECT : Decision.ESCALATE;

        VotingPolicy.VotingRule supermajority067 = vs -> {
            Decision d = VotingPolicy.decideSupermajority(vs, 0.67);
            return d != Decision.ESCALATE ? d : Decision.ESCALATE;
        };

        VotingPolicy.VotingRule simpleMajority = VotingPolicy::decide;

        return VotingPolicy.decide(votes, List.of(overrideReject, supermajority067, simpleMajority));
    }

    /**
     * Consensus: all-approve in quorum(Role=OWNER,ENGINEERING,SECURITY) else ESCALATE.
     */
    public static Decision consensus(Collection<Vote> votes) {
        Set<Role> quorum = Set.of(Role.OWNER, Role.ENGINEERING, Role.SECURITY);
        VotingPolicy.VotingRule allApproveInQuorum = vs -> {
            for (Role r : quorum) {
                boolean any = vs.stream().anyMatch(v -> v.role()==r);
                if (!any) return Decision.ESCALATE;
                boolean anyReject = vs.stream().anyMatch(v -> v.role()==r && v.decision()==Decision.REJECT);
                if (anyReject) return Decision.ESCALATE;
                boolean anyApprove = vs.stream().anyMatch(v -> v.role()==r && v.decision()==Decision.APPROVE);
                if (!anyApprove) return Decision.ESCALATE;
            }
            return Decision.APPROVE;
        };
        return VotingPolicy.decide(votes, List.of(allApproveInQuorum));
    }
}


