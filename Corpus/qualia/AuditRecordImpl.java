package qualia;

import java.util.Date;
import java.util.Objects;

/**
 * Immutable implementation of {@link AuditRecord}.
 *
 * <p>Thread-safe: fields are final; mutable inputs/outputs are defensively copied.
 */
public final class AuditRecordImpl implements AuditRecord {
    private final String id;
    private final Date timestamp;

    /**
     * Constructs an immutable audit record.
     * @param id unique identifier (non-null)
     * @param timestamp creation time (non-null); will be defensively copied
     */
    public AuditRecordImpl(String id, Date timestamp) {
        this.id = Objects.requireNonNull(id, "id must not be null");
        this.timestamp = new Date(Objects.requireNonNull(timestamp, "timestamp must not be null").getTime());
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public Date timestamp() {
        return new Date(timestamp.getTime());
    }

    @Override
    public String toString() {
        return "AuditRecord{id='" + id + "', timestamp=" + timestamp + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AuditRecord)) return false;
        AuditRecord that = (AuditRecord) o;
        return Objects.equals(id, that.id()) &&
               Objects.equals(timestamp, that.timestamp());
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, timestamp);
    }
}