package qualia; // TODO: move to UOIFCore

import java.util.concurrent.CompletableFuture;

/**
 * Console implementation of AuditSink.
 */
public final class ConsoleAuditSink implements AuditSink {
    private final AuditOptions options;

    public ConsoleAuditSink(AuditOptions options) {
        this.options = options;
    }

    @Override
    public CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts) {
        AuditOptions effectiveOpts = opts != null ? opts : this.options;
        return CompletableFuture.runAsync(() -> {
            if (effectiveOpts != null && effectiveOpts.dryRun()) {
                System.out.println("Dry run: would write audit record: " + rec);
            } else {
                System.out.println("Writing audit record: " + rec);
            }
        });
    }
}