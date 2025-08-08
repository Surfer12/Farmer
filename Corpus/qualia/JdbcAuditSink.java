package qualia;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLIntegrityConstraintViolationException;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Types;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

/**
 * JDBC-backed {@link AuditSink} for durable persistence.
 *
 * <p>Non-blocking and thread-safe: writes are serialized via a single-threaded
 * executor with a bounded queue. Connections are created lazily and reused.
 * The sink can optionally auto-create the audit table if it does not exist.
 *
 * <p>Schema (cross-DB friendly):
 * <pre>
 *   CREATE TABLE IF NOT EXISTS audit_records (
 *     id VARCHAR(255) PRIMARY KEY,
 *     ts_ms BIGINT NOT NULL,
 *     idempotency_key VARCHAR(255)
 *   );
 * </pre>
 */
public final class JdbcAuditSink implements AuditSink {
    private final String jdbcUrl;
    private final String username;
    private final String password;
    private final String tableName;
    private final boolean createTableIfMissing;
    private final ExecutorService executor;

    private volatile Connection connection;

    /**
     * Creates a JDBC sink with sensible defaults.
     * @param jdbcUrl JDBC connection URL (non-null)
     * @param username DB username (nullable if URL supplies it)
     * @param password DB password (nullable if URL supplies it)
     */
    public JdbcAuditSink(String jdbcUrl, String username, String password) {
        this(jdbcUrl, username, password, "audit_records", true, 1024);
    }

    /**
     * Creates a JDBC sink with full control over behavior.
     * @param jdbcUrl JDBC connection URL (non-null)
     * @param username DB username (nullable if URL supplies it)
     * @param password DB password (nullable if URL supplies it)
     * @param tableName audit table name (non-null)
     * @param createTableIfMissing whether to auto-create table at startup
     * @param maxQueue maximum queued writes before applying back-pressure
     */
    public JdbcAuditSink(String jdbcUrl,
                         String username,
                         String password,
                         String tableName,
                         boolean createTableIfMissing,
                         int maxQueue) {
        this.jdbcUrl = Objects.requireNonNull(jdbcUrl, "jdbcUrl");
        this.username = username;
        this.password = password;
        this.tableName = Objects.requireNonNullElse(tableName, "audit_records");
        this.createTableIfMissing = createTableIfMissing;
        this.executor = newBoundedExecutor(maxQueue);
    }

    private static ExecutorService newBoundedExecutor(int maxQueue) {
        LinkedBlockingQueue<Runnable> queue = new LinkedBlockingQueue<>(Math.max(128, maxQueue));
        ThreadFactory tf = r -> {
            Thread t = new Thread(r, "audit-jdbc-");
            t.setDaemon(true);
            return t;
        };
        // Single-threaded to avoid concurrent JDBC access on a shared connection
        return new ThreadPoolExecutorCompat(1, 1, 60, TimeUnit.SECONDS, queue, tf);
    }

    @Override
    public CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts) {
        Objects.requireNonNull(rec, "rec must not be null");
        final AuditOptions effectiveOpts = opts;
        if (effectiveOpts != null && effectiveOpts.dryRun()) {
            return CompletableFuture.completedFuture(null);
        }
        return CompletableFuture.runAsync(() -> {
            try {
                ensureConnection();
                insertRecord(rec, effectiveOpts);
            } catch (SQLException e) {
                throw new RuntimeException("JDBC write failed", e);
            }
        }, executor);
    }

    private void ensureConnection() throws SQLException {
        try {
            if (connection == null || connection.isClosed() || !isValid(connection)) {
                closeQuietly();
                connection = (username == null && password == null)
                        ? DriverManager.getConnection(jdbcUrl)
                        : DriverManager.getConnection(jdbcUrl, username, password);
                connection.setAutoCommit(true);
                if (createTableIfMissing) {
                    createTableIfNotExists(connection);
                }
            }
        } catch (SQLException e) {
            closeQuietly();
            throw e;
        }
    }

    private static boolean isValid(Connection c) {
        try {
            return c.isValid(1);
        } catch (SQLException ignored) {
            return false;
        }
    }

    private void createTableIfNotExists(Connection c) throws SQLException {
        String ddl = "CREATE TABLE IF NOT EXISTS " + tableName + " (" +
                "id VARCHAR(255) PRIMARY KEY, " +
                "ts_ms BIGINT NOT NULL, " +
                "idempotency_key VARCHAR(255)" +
                ")";
        try (Statement st = c.createStatement()) {
            st.executeUpdate(ddl);
        }
    }

    private void insertRecord(AuditRecord rec, AuditOptions opts) throws SQLException {
        String sql = "INSERT INTO " + tableName + " (id, ts_ms, idempotency_key) VALUES (?, ?, ?)";
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            ps.setString(1, rec.id());
            ps.setLong(2, rec.timestamp().getTime());
            if (opts != null && opts.idempotencyKey() != null) {
                ps.setString(3, opts.idempotencyKey());
            } else {
                ps.setNull(3, Types.VARCHAR);
            }
            ps.executeUpdate();
        } catch (SQLIntegrityConstraintViolationException dup) {
            // Duplicate primary key (id) — treat as idempotent no-op
        } catch (SQLException e) {
            if (isDuplicateKey(e)) {
                // Cross-DB duplicate key (e.g., SQLState 23*, 23505)
                return;
            }
            throw e;
        }
    }

    private static boolean isDuplicateKey(SQLException e) {
        if (e instanceof SQLIntegrityConstraintViolationException) return true;
        String state = e.getSQLState();
        // 23*** is integrity constraint violation (ANSI), 23505 is Postgres unique_violation
        return state != null && (state.startsWith("23") || "23505".equals(state));
    }

    /**
     * Closes resources. Not part of {@link AuditSink} interface but provided for lifecycle management.
     */
    public synchronized void close() {
        try {
            executor.shutdown();
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        } finally {
            closeQuietly();
        }
    }

    private synchronized void closeQuietly() {
        if (connection != null) {
            try { connection.close(); } catch (SQLException ignored) {}
            connection = null;
        }
    }
}


