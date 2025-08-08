package qualia;

import java.util.Date;
import java.util.Objects;

/**
 * Optional extension of {@link AuditRecord} that carries structured decision context
 * as pre-redacted JSON strings.
 */
public interface ExtendedAuditRecord extends AuditRecord {
    String event();
    String schemaVersion();
    String inputsJson();
    String outputsJson();
    String paramsJson();
}

/**
 * Simple implementation of {@link ExtendedAuditRecord} with JSON-encoded payloads.
 */
final class ExtendedAuditRecordImpl implements ExtendedAuditRecord {
    private final String id;
    private final Date timestamp;
    private final String event;
    private final String schemaVersion;
    private final String inputsJson;
    private final String outputsJson;
    private final String paramsJson;

    ExtendedAuditRecordImpl(String id,
                            Date timestamp,
                            String event,
                            String schemaVersion,
                            String inputsJson,
                            String outputsJson,
                            String paramsJson) {
        this.id = Objects.requireNonNull(id);
        this.timestamp = new Date(Objects.requireNonNull(timestamp).getTime());
        this.event = Objects.requireNonNull(event);
        this.schemaVersion = Objects.requireNonNull(schemaVersion);
        this.inputsJson = inputsJson;
        this.outputsJson = outputsJson;
        this.paramsJson = paramsJson;
    }

    @Override public String id() { return id; }
    @Override public Date timestamp() { return new Date(timestamp.getTime()); }
    @Override public String event() { return event; }
    @Override public String schemaVersion() { return schemaVersion; }
    @Override public String inputsJson() { return inputsJson; }
    @Override public String outputsJson() { return outputsJson; }
    @Override public String paramsJson() { return paramsJson; }
}


