package qualia;

import java.util.Date;
import java.util.Objects;

/**
 * Immutable implementation of the AuditRecord interface.
 */
public final class AuditRecordImpl implements AuditRecord {
    private final String id;
    private final Date timestamp;

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