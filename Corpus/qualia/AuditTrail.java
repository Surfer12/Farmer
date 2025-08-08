package qualia;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Pure-decision boundary for emitting audit records. All decision methods remain pure;
 * this utility provides a single, isolated side-effect via an injected {@link AuditSink}.
 *
 * <p>Fail-open: errors are contained and do not affect the decision path.
 * <p>Idempotent: callers can provide an idempotency key to ensure at-most-once semantics.
 * <p>Non-blocking: returns immediately with a {@link CompletableFuture} that callers
 *     may ignore or observe.
 */
public final class AuditTrail {
    private final AuditSink sink;

    /**
     * Creates an audit trail that writes to the given sink. Pass {@link AuditSink#noop()}
     * in unit tests for hermetic purity.
     */
    public AuditTrail(AuditSink sink) {
        this.sink = Objects.requireNonNull(sink, "sink");
    }

    /**
     * Fire-and-forget write of an audit record. Exceptions are captured in the returned future.
     * The decision caller should not be blocked by this call; do not join on critical paths.
     */
    public CompletableFuture<Void> emit(AuditRecord record, AuditOptions options) {
        Objects.requireNonNull(record, "record");
        AuditOptions opts = options != null ? options : AuditOptions.builder().dryRun(false).build();
        try {
            return sink.write(record, opts).exceptionally(ex -> null);
        } catch (RuntimeException ex) {
            // Fail-open: swallow to preserve decision purity
            return CompletableFuture.completedFuture(null);
        }
    }
}


