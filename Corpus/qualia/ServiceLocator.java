// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.Objects;

/**
 * Minimal service locator / DI container. For simplicity, only wires a few bindings.
 */
public final class ServiceLocator {
    private final ModelFactory modelFactory;
    private final AuditSink auditSink;
    private MetricsServer metricsServer;

    private ServiceLocator(ModelFactory modelFactory, AuditSink auditSink) {
        this.modelFactory = Objects.requireNonNull(modelFactory);
        this.auditSink = Objects.requireNonNull(auditSink);
    }

    public static Builder builder() { return new Builder(); }

    public PsiModel psiModel(ModelPriors priors, int parallelThreshold) {
        return modelFactory.createPsiModel(priors, parallelThreshold);
    }

    public AuditSink auditSink() { return auditSink; }

    public static final class Builder {
        private ModelFactory modelFactory = ModelFactory.defaultFactory();
        private AuditSink auditSink = AuditSink.noop();
        private boolean startMetrics = false;

        public Builder modelFactory(ModelFactory f) { this.modelFactory = Objects.requireNonNull(f); return this; }
        public Builder auditSink(AuditSink s) { this.auditSink = Objects.requireNonNull(s); return this; }

        public Builder fromEnvironment() {
            String sinkKind = System.getenv().getOrDefault("AUDIT_SINK", "noop").toLowerCase(java.util.Locale.ROOT);
            switch (sinkKind) {
                case "console" -> this.auditSink = new ConsoleAuditSink(AuditOptions.builder().dryRun(false).build());
                case "file" -> {
                    String dir = System.getenv().getOrDefault("AUDIT_DIR", "audit-logs");
                    String prefix = System.getenv().getOrDefault("AUDIT_PREFIX", "audit");
                    long maxBytes = parseLong(System.getenv("AUDIT_MAX_BYTES"), 10_000_000L);
                    long rotateMs = parseLong(System.getenv("AUDIT_ROTATE_MS"), 60_000L);
                    int flush = (int) parseLong(System.getenv("AUDIT_FLUSH_BYTES"), 1024L);
                    this.auditSink = new FileAuditSink(new java.io.File(dir), prefix, maxBytes, rotateMs, flush);
                }
                case "http" -> {
                    String url = System.getenv("AUDIT_HTTP_URL");
                    if (url == null || url.isEmpty()) url = System.getenv("HTTP_AUDIT_URL");
                    if (url == null || url.isEmpty()) {
                        this.auditSink = AuditSink.noop();
                    } else {
                        try {
                            java.net.URI uri = java.net.URI.create(url);
                            this.auditSink = new HttpAuditSink(uri, 3, java.time.Duration.ofMillis(500));
                        } catch (Exception e) {
                            this.auditSink = AuditSink.noop();
                        }
                    }
                }
                case "jdbc" -> {
                    String url = System.getenv("JDBC_URL");
                    String user = System.getenv("JDBC_USER");
                    String pass = System.getenv("JDBC_PASS");
                    if (url == null || url.isEmpty()) this.auditSink = AuditSink.noop();
                    else this.auditSink = new JdbcAuditSink(url, user, pass);
                }
                default -> this.auditSink = AuditSink.noop();
            }
            return this;
        }

        public Builder withMetricsServer(boolean enable) { this.startMetrics = enable; return this; }

        private static long parseLong(String s, long def) { try { return s==null?def:Long.parseLong(s); } catch (Exception e) { return def; } }

        public ServiceLocator build() {
            ServiceLocator sl = new ServiceLocator(modelFactory, auditSink);
            if (startMetrics) {
                try {
                    sl.metricsServer = MetricsServer.startFromEnv();
                } catch (RuntimeException ex) {
                    ErrorReporter.report("ServiceLocator.metricsServer", ex);
                }
            }
            // Shutdown hook for graceful close
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    if (sl.auditSink instanceof AutoCloseable c) c.close();
                } catch (Exception ignored) {}
                try {
                    if (sl.metricsServer != null) sl.metricsServer.close();
                } catch (Exception ignored) {}
            }, "qualia-shutdown"));
            return sl;
        }
    }
}


