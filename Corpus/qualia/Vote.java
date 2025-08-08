package qualia;

import java.util.Objects;

/**
 * Single stakeholder vote with optional weight and override flag.
 */
public record Vote(String stakeholderId, Decision decision, double weight, boolean override)
{
    public Vote {
        Objects.requireNonNull(stakeholderId, "stakeholderId");
        Objects.requireNonNull(decision, "decision");
        if (!(weight > 0.0)) {
            throw new IllegalArgumentException("weight must be > 0");
        }
    }
}


