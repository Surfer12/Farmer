package qualia;

import java.util.Date;

/**
 * Minimal demo entrypoint; prints a record via Console sink.
 */
final class Main {
    public static void main(String[] args) {
        AuditSink jdbcSink = new JdbcAuditSink(
            "jdbc:postgresql://localhost:5432/qualia", // or MySQL URL
            "dbuser",
            "dbpass"
        );
        
        AuditRecord rec1 = new AuditRecordImpl("rec-123", new Date());
        AuditOptions opts1 = AuditOptions.builder()
            .idempotencyKey("rec-123") // optional
            .dryRun(false)
            .build();
        
        jdbcSink.write(rec1, opts1).join(); // wait if you need to ensure durability now
        
        // Optional: when shutting down your app
        if (jdbcSink instanceof JdbcAuditSink j) j.close();
        AuditSink consoleSink = new ConsoleAuditSink(AuditOptions.builder().dryRun(true).build());
        AuditRecord rec2 = new AuditRecordImpl("rec-123", new Date());
        consoleSink.write(rec2, AuditOptions.builder().dryRun(true).build()).join();
        // AuditSink sink = new ConsoleAuditSink(AuditOptions.builder().dryRun(true).build());
        // AuditRecord rec = new AuditRecordImpl("rec-123", new Date());
        // sink.write(rec, AuditOptions.builder().dryRun(true).build()).join();
    }
}