// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.util.concurrent.CompletableFuture;
import java.util.Objects;

/**
 * Asynchronous sink for {@link AuditRecord} persistence or forwarding.
 *
 * <p>Implementations should be thread-safe and non-blocking where possible. The
 * returned {@link CompletableFuture} signals completion or failure of the write.
 */
public interface AuditSink {
    /**
     * Writes an audit record to the sink asynchronously.
     *
     * @param rec  the audit record to write (must not be null)
     * @param opts the audit options (must not be null)
     * @return a CompletableFuture that completes when the write is done
     * @throws NullPointerException if rec or opts is null
     */
    CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts);

    /**
     * Returns a no-op {@link AuditSink} that completes immediately.
     */
    static AuditSink noop() {
        return (rec, opts) -> CompletableFuture.completedFuture(null);
    }

    /**
     * Returns a composite {@link AuditSink} that writes sequentially to the given sinks.
     * If any sink fails, the returned future completes exceptionally and subsequent
     * sinks are not invoked.
     *
     * @param sinks the sinks to delegate to (non-null)
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