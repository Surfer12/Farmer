package qualia;

import java.util.Objects;

/**
 * Immutable data record representing a single claim with its features and
 * verification outcome.
 *
 * <p>Field semantics:
 * - {@code id}: application-unique identifier
 * - {@code isVerifiedTrue}: observed label for supervision/evaluation
 * - {@code riskAuthenticity}, {@code riskVirality}: non-negative risk indicators
 * - {@code probabilityHgivenE}: P(H|E) in [0,1]
 */
public record ClaimData(
        String id,
        boolean isVerifiedTrue,
        double riskAuthenticity,
        double riskVirality,
        double probabilityHgivenE
) {
    public ClaimData {
        Objects.requireNonNull(id, "id must not be null");
        if (probabilityHgivenE < 0.0 || probabilityHgivenE > 1.0) {
            throw new IllegalArgumentException("probabilityHgivenE must be in [0,1]");
        }
        if (riskAuthenticity < 0.0) {
            throw new IllegalArgumentException("riskAuthenticity must be >= 0");
        }
        if (riskVirality < 0.0) {
            throw new IllegalArgumentException("riskVirality must be >= 0");
        }
    }
}
