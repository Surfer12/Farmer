
/**
 * Minimal health checks for sinks and samplers.
 */
public final class Health {
    private Health() {}

    public static boolean isJdbcHealthy(JdbcAuditSink sink) {
        try {
            // Try a no-op write in dryRun; if exception bubbles, consider unhealthy
            sink.write(new AuditRecordImpl("health", new java.util.Date()),
                    AuditOptions.builder().dryRun(true).build()).join();
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}


