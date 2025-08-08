// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * HTTP {@link AuditSink} that POSTs JSON with retry and exponential backoff.
 *
 * <p>Non-blocking using Java's async {@link HttpClient}. The sink retries
 * transient failures up to {@code maxRetries}, doubling backoff each attempt
 * up to a small cap.
 */
public final class HttpAuditSink implements AuditSink {
    private final HttpClient client;
    private final URI endpoint;
    private final int maxRetries;
    private final Duration initialBackoff;

    /**
     * Creates an HTTP sink.
     * @param endpoint target URI for POST requests
     * @param maxRetries maximum number of retries on failure
     * @param initialBackoff initial backoff duration; defaults to 200ms if null
     */
    public HttpAuditSink(URI endpoint, int maxRetries, Duration initialBackoff) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        this.maxRetries = Math.max(0, maxRetries);
        this.initialBackoff = initialBackoff == null ? Duration.ofMillis(200) : initialBackoff;
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    @Override
    public CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts) {
        String json = toJson(rec);
        if (opts != null && opts.dryRun()) {
            return CompletableFuture.completedFuture(null);
        }
        return sendWithRetry(json, 0, initialBackoff);
    }

    /**
     * Sends a JSON payload with retry and backoff.
     */
    private CompletableFuture<Void> sendWithRetry(String json, int attempt, Duration backoff) {
        HttpRequest req = HttpRequest.newBuilder(endpoint)
                .timeout(Duration.ofSeconds(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

        return client.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                .thenCompose(resp -> {
                    int code = resp.statusCode();
                    if (code >= 200 && code < 300) {
                        MetricsRegistry.get().incCounter("http_audit_post_success_total");
                        return CompletableFuture.completedFuture(null);
                    }
                    if (attempt >= maxRetries) {
                        MetricsRegistry.get().incCounter("http_audit_post_fail_total");
                        return CompletableFuture.failedFuture(new NetworkException("HTTP " + code));
                    }
                    MetricsRegistry.get().incCounter("http_retry_total");
                    return delayed(backoff).thenCompose(v -> sendWithRetry(json, attempt + 1, nextBackoff(backoff)));
                })
                .exceptionallyCompose(ex -> {
                    if (attempt >= maxRetries) {
                        MetricsRegistry.get().incCounter("http_audit_post_fail_total");
                        return CompletableFuture.failedFuture(new NetworkException("HTTP send failed", ex));
                    }
                    MetricsRegistry.get().incCounter("http_retry_total");
                    return delayed(backoff).thenCompose(v -> sendWithRetry(json, attempt + 1, nextBackoff(backoff)));
                });
    }

    /**
     * Exponential backoff with a bounded upper limit.
     */
    private static Duration nextBackoff(Duration current) {
        long next = Math.min(current.toMillis() * 2, TimeUnit.SECONDS.toMillis(5));
        return Duration.ofMillis(next);
    }

    /**
     * Creates a future that completes after the given delay.
     */
    private static CompletableFuture<Void> delayed(Duration d) {
        return CompletableFuture.runAsync(() -> {}, CompletableFuture.delayedExecutor(d.toMillis(), TimeUnit.MILLISECONDS))
                .thenApply(v -> null);
    }

    /**
     * Serializes a record to JSON.
     */
    private static String toJson(AuditRecord rec) {
        String id = JsonUtil.escape(rec.id());
        String ts = JsonUtil.toIso8601(rec.timestamp());
        if (rec instanceof ExtendedAuditRecord e) {
            String event = JsonUtil.escape(e.event());
            String schema = JsonUtil.escape(e.schemaVersion());
            String inputs = e.inputsJson();
            String outputs = e.outputsJson();
            String params = e.paramsJson();
            return "{" +
                    "\"id\":\"" + id + "\"," +
                    "\"timestamp\":\"" + ts + "\"," +
                    "\"event\":\"" + event + "\"," +
                    "\"schemaVersion\":\"" + schema + "\"," +
                    "\"inputs\":" + (inputs == null ? "null" : inputs) + "," +
                    "\"outputs\":" + (outputs == null ? "null" : outputs) + "," +
                    "\"params\":" + (params == null ? "null" : params) +
                    "}";
        }
        return "{\"id\":\"" + id + "\",\"timestamp\":\"" + ts + "\"}";
    }
}


