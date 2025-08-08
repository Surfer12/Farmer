// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Single stakeholder vote with optional weight, override, role, confidence, expiry, and metadata.
 */
public record Vote(
        String stakeholderId,
        Role role,
        String quorumId,
        Decision decision,
        double weight,
        boolean override,
        double confidence,
        Instant expiresAt,
        String idempotencyKey,
        Map<String, String> meta
) {
    public Vote {
        Objects.requireNonNull(stakeholderId, "stakeholderId");
        Objects.requireNonNull(decision, "decision");
        if (!(weight > 0.0)) {
            throw new IllegalArgumentException("weight must be > 0");
        }
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("confidence must be in [0,1]");
        }
    }
}


