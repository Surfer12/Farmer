// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;
import java.util.Objects;

/**
 * Immutable options for audit operations.
 *
 * <p>These options travel alongside each write to an {@link AuditSink} and do not
 * change after creation. Instances are thread-safe and may be shared across
 * threads without additional synchronization.
 *
 * <p>Use the {@link Builder} to construct instances; the builder is mutable and
 * not thread-safe.
 */
public final class AuditOptions {
    private final String idempotencyKey;
    private final boolean dryRun;

    private AuditOptions(Builder builder) {
        this.idempotencyKey = builder.idempotencyKey;
        this.dryRun = builder.dryRun;
    }

    /**
     * Returns the idempotency key associated with this operation, if any.
     *
     * <p>Sinks may use this to de-duplicate writes. Semantics are sink-defined;
     * absence of a key implies no idempotency guarantee.
     */
    public String idempotencyKey() {
        return idempotencyKey;
    }

    /**
     * Returns whether the operation should be a dry run.
     *
     * <p>When {@code true}, sinks should execute side-effect-free write paths
     * (e.g., log instead of persisting) but still validate inputs.
     */
    public boolean dryRun() {
        return dryRun;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link AuditOptions}. Not thread-safe.
     */
    public static final class Builder {
        private String idempotencyKey;
        private boolean dryRun = false;

        /**
         * Sets an optional idempotency key for the write.
         * @param idempotencyKey a caller-provided stable key, or {@code null}
         * @return this builder
         */
        public Builder idempotencyKey(String idempotencyKey) {
            this.idempotencyKey = idempotencyKey;
            return this;
        }

        /**
         * Sets whether to execute the write in dry-run mode.
         * @param dryRun if {@code true}, sinks should avoid external side effects
         * @return this builder
         */
        public Builder dryRun(boolean dryRun) {
            this.dryRun = dryRun;
            return this;
        }

        /**
         * Builds an immutable {@link AuditOptions} instance.
         */
        public AuditOptions build() {
            return new AuditOptions(this);
        }
    }

    @Override
    public String toString() {
        return "AuditOptions{idempotencyKey=" + idempotencyKey + ", dryRun=" + dryRun + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AuditOptions)) return false;
        AuditOptions that = (AuditOptions) o;
        return dryRun == that.dryRun &&
                Objects.equals(idempotencyKey, that.idempotencyKey);
    }

    @Override
    public int hashCode() {
        return Objects.hash(idempotencyKey, dryRun);
    }
}