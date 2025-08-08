package qualia;

import java.util.Date;

/**
 * Minimal demo entrypoint; prints a record via Console sink.
 */
final class Main {
    public static void main(String[] args) {
        AuditSink sink = new ConsoleAuditSink(AuditOptions.builder().dryRun(true).build());
        AuditRecord rec = new AuditRecordImpl("rec-123", new Date());
        sink.write(rec, AuditOptions.builder().dryRun(true).build()).join();
    }
}