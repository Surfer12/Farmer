// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.Date;

/**
 * Minimal demo entrypoint; prints a record via Console sink.
 */
final class Main {
    public static void main(String[] args) {
        // Start services from environment and enable metrics if requested
        ServiceLocator sl = ServiceLocator.builder()
                .fromEnvironment()
                .withMetricsServer(true)
                .build();

        String dbUrl = System.getenv("QUALIA_DB_URL");
        JdbcAuditSink jdbcSink = null;
        if (dbUrl != null && !dbUrl.isEmpty()) {
            jdbcSink = new JdbcAuditSink(
                dbUrl,
                System.getenv("QUALIA_DB_USER"),
                System.getenv("QUALIA_DB_PASS")
            );
            AuditRecord rec1 = new AuditRecordImpl("rec-123", new Date());
            AuditOptions opts1 = AuditOptions.builder()
                .idempotencyKey("rec-123")
                .dryRun(false)
                .build();
            try {
                jdbcSink.write(rec1, opts1).join();
            } catch (Exception e) {
                System.err.println("JDBC sink disabled (" + e.getMessage() + ")");
            }
        }
        
        // Minimal smoke tests for agent presets
        var now = java.time.Instant.now();
        java.util.List<Vote> votes = java.util.List.of(
                new Vote("u1", Role.SECURITY, "q1", Decision.REJECT, 1.0, true, 1.0, now.plusSeconds(3600), null, java.util.Map.of()),
                new Vote("u2", Role.LEGAL, "q1", Decision.APPROVE, 1.0, false, 0.9, now.plusSeconds(3600), null, java.util.Map.of()),
                new Vote("u3", Role.ENGINEERING, "q1", Decision.APPROVE, 1.0, false, 0.9, now.plusSeconds(3600), null, java.util.Map.of())
        );
        System.out.println("safetyCritical: " + AgentPresets.safetyCritical(votes));
        System.out.println("fastPath: " + AgentPresets.fastPath(votes));
        System.out.println("consensus: " + AgentPresets.consensus(votes));
        
        // Optional: shutdown JDBC sink
        if (jdbcSink != null) jdbcSink.close();
    }
}