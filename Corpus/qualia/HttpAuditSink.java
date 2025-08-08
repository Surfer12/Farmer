package qualia;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * HTTP sink that POSTs JSON with retry/backoff. Non-blocking using async HttpClient.
 */
public final class HttpAuditSink implements AuditSink {
    private final HttpClient client;
    private final URI endpoint;
    private final int maxRetries;
    private final Duration initialBackoff;

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
                        return CompletableFuture.completedFuture(null);
                    }
                    if (attempt >= maxRetries) {
                        return CompletableFuture.failedFuture(new IOException("HTTP " + code));
                    }
                    return delayed(backoff).thenCompose(v -> sendWithRetry(json, attempt + 1, nextBackoff(backoff)));
                })
                .exceptionallyCompose(ex -> {
                    if (attempt >= maxRetries) {
                        return CompletableFuture.failedFuture(ex);
                    }
                    return delayed(backoff).thenCompose(v -> sendWithRetry(json, attempt + 1, nextBackoff(backoff)));
                });
    }

    private static Duration nextBackoff(Duration current) {
        long next = Math.min(current.toMillis() * 2, TimeUnit.SECONDS.toMillis(5));
        return Duration.ofMillis(next);
    }

    private static CompletableFuture<Void> delayed(Duration d) {
        return CompletableFuture.runAsync(() -> {}, CompletableFuture.delayedExecutor(d.toMillis(), TimeUnit.MILLISECONDS))
                .thenApply(v -> null);
    }

    private static String toJson(AuditRecord rec) {
        String id = JsonUtil.escape(rec.id());
        String ts = JsonUtil.toIso8601(rec.timestamp());
        return "{\"id\":\"" + id + "\",\"timestamp\":\"" + ts + "\"}";
    }
}


