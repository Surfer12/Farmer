package qualia;

import java.util.concurrent.CompletableFuture;
import java.util.Objects;

/**
 * Represents a sink for audit records.
 * <p>
 * Implementations must be thread-safe and non-blocking where possible.
 */
public interface AuditSink {
    /**
     * Writes an audit record to the sink.
     *
     * @param rec  the audit record to write (must not be null)
     * @param opts the audit options (must not be null)
     * @return a CompletableFuture that completes when the write is done
     * @throws NullPointerException if rec or opts is null
     */
    CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts);

    /**
     * Returns a no-op AuditSink that completes immediately.
     */
    static AuditSink noop() {
        return (rec, opts) -> CompletableFuture.completedFuture(null);
    }

    /**
     * Returns an AuditSink that delegates to the given sinks in order.
     * If any sink fails, the returned future completes exceptionally.
     *
     * @param sinks the sinks to delegate to
     * @return a composite AuditSink
     */
    static AuditSink composite(AuditSink... sinks) {
        Objects.requireNonNull(sinks, "sinks must not be null");
        return (rec, opts) -> {
            CompletableFuture<Void> future = CompletableFuture.completedFuture(null);
            for (AuditSink sink : sinks) {
                future = future.thenCompose(v -> sink.write(rec, opts));
            }
            return future;
        };
    }
}