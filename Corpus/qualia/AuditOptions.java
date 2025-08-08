package qualia;
import java.util.Objects;

/**
 * Immutable options for audit operations.
 * <p>
 * Use the builder for flexible construction.
 */
public final class AuditOptions {
    private final String idempotencyKey;
    private final boolean dryRun;

    private AuditOptions(Builder builder) {
        this.idempotencyKey = builder.idempotencyKey;
        this.dryRun = builder.dryRun;
    }

    public String idempotencyKey() {
        return idempotencyKey;
    }

    public boolean dryRun() {
        return dryRun;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String idempotencyKey;
        private boolean dryRun = false;

        public Builder idempotencyKey(String idempotencyKey) {
            this.idempotencyKey = idempotencyKey;
            return this;
        }

        public Builder dryRun(boolean dryRun) {
            this.dryRun = dryRun;
            return this;
        }

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