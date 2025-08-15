// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


/**
 * Base unchecked exception for the Qualia package with a lightweight
 * severity and code for governance and metrics.
 */
public class QualiaException extends RuntimeException {
    public enum Severity { INFO, WARN, ERROR, FATAL }

    private final Severity severity;
    private final String code;

    public QualiaException(String message) {
        this(message, null, Severity.ERROR, null);
    }

    public QualiaException(String message, Throwable cause) {
        this(message, cause, Severity.ERROR, null);
    }

    public QualiaException(String message, Throwable cause, Severity severity, String code) {
        super(message, cause);
        this.severity = severity == null ? Severity.ERROR : severity;
        this.code = code;
    }

    public Severity severity() { return severity; }
    public String code() { return code; }
}

/** Configuration/initialization problems. */
class ConfigurationException extends QualiaException {
    public ConfigurationException(String message) { super(message, null, Severity.ERROR, "CONFIG"); }
    public ConfigurationException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "CONFIG"); }
}

/** Input validation errors (caller mistakes). */
class ValidationException extends QualiaException {
    public ValidationException(String message) { super(message, null, Severity.ERROR, "VALIDATION"); }
    public ValidationException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "VALIDATION"); }
}

/** Network/IO with remote services. */
class NetworkException extends QualiaException {
    public NetworkException(String message) { super(message, null, Severity.ERROR, "NETWORK"); }
    public NetworkException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "NETWORK"); }
}

/** Local persistence/IO failures. */
class PersistenceException extends QualiaException {
    public PersistenceException(String message) { super(message, null, Severity.ERROR, "PERSISTENCE"); }
    public PersistenceException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "PERSISTENCE"); }
}

/** Computation/algorithmic failures. */
class ComputationException extends QualiaException {
    public ComputationException(String message) { super(message, null, Severity.ERROR, "COMPUTE"); }
    public ComputationException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "COMPUTE"); }
}

/** Audit emission failures. */
class AuditWriteException extends QualiaException {
    public AuditWriteException(String message) { super(message, null, Severity.ERROR, "AUDIT_WRITE"); }
    public AuditWriteException(String message, Throwable cause) { super(message, cause, Severity.ERROR, "AUDIT_WRITE"); }
}


