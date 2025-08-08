package qualia;

import java.util.Date;

/**
 * Represents an audit record.
 */
public interface AuditRecord {
    String id();
    Date timestamp();
}


