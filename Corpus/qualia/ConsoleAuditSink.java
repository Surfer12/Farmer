// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia; // TODO: move to UOIFCore

import java.util.concurrent.CompletableFuture;

/**
 * Development {@link AuditSink} that logs records to standard output.
 *
 * <p>Non-blocking: uses a default async execution for printing.
 */
public final class ConsoleAuditSink implements AuditSink {
    private final AuditOptions options;

    /**
     * Creates a console sink with default behavior controlled by {@code options}.
     * @param options default options applied when per-call opts are null
     */
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