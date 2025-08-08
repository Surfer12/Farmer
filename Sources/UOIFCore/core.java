import java.time.Instant;
import java.util.Date;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.Objects;

// Option class for optional parameters (Java 24+ has Optional, but for clarity, using simple POJO)
final class AuditOptions {
    private final String idempotencyKey;
    private final boolean dryRun;

    public AuditOptions(String idempotencyKey, boolean dryRun) {
        this.idempotencyKey = idempotencyKey;
        this.dryRun = dryRun;
    }

    public AuditOptions(String idempotencyKey) {
        this(idempotencyKey, false);
    }

    public AuditOptions() {
        this(null, false);
    }

    public String idempotencyKey() { return idempotencyKey; }
    public boolean dryRun() { return dryRun; }
}

final class AuditRecord {
    private final String id;
    private final Date timestamp;

    public AuditRecord(String id, Date timestamp) {
        this.id = Objects.requireNonNull(id);
        this.timestamp = new Date(timestamp.getTime());
    }

    public String id() { return id; }
    public Date timestamp() { return new Date(timestamp.getTime()); }

    @Override
    public String toString() {
        return "AuditRecord{id='" + id + "', timestamp=" + timestamp + '}';
    }
}

interface AuditSink {
    CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts);
}

final class ConsoleAuditSink implements AuditSink {
    @Override
    public CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts) {
        return CompletableFuture.runAsync(() -> {
            if (opts != null && opts.dryRun()) {
                System.out.println("Dry run: would write audit record: " + rec);
            } else {
                System.out.println("Writing audit record: " + rec);
            }
        });
    }
}

public class Core {
    private static final AuditSink auditSink = new ConsoleAuditSink();

    private static final AuditRecord auditRecord = new AuditRecord(
            "abc123",
            Date.from(Instant.now())
    );

    public static void main(String[] args) {
        // Write audit record with idempotencyKey
        auditSink.write(auditRecord, new AuditOptions("abc123"))
            .thenRun(() -> System.out.println("Audit record written successfully"))
            .exceptionally(error -> {
                System.err.println("Failed to write audit record: " + error);
                return null;
            }).join();

        // Dry run
        try {
            dryRun();
            System.out.println("Dry run completed");
        } catch (Exception e) {
            System.err.println("Dry run failed: " + e + " " + auditRecord + " opts: {idempotencyKey: \"abc123\", dryRun: true}");
        }
    }

    public static void dryRun() throws ExecutionException, InterruptedException {
        auditSink.write(auditRecord, new AuditOptions("abc123", true)).get();
    }

    public static void withIdempotencyKey(AuditOptions opts) throws ExecutionException, InterruptedException {
        auditSink.write(auditRecord, opts).get();
    }

    public static void withDryRun(AuditOptions opts) throws ExecutionException, InterruptedException {
        auditSink.write(auditRecord, opts).get();
    }
}
