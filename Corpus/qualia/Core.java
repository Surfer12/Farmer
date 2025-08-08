package qualia;

import java.io.File;
import java.util.Date;

/**
 * Core entrypoint for quick demos of available AuditSink implementations.
 *
 * Usage:
 *   java -cp <cp> qualia.Core console
 *   java -cp <cp> qualia.Core file
 *   java -cp <cp> qualia.Core jdbc
 *
 * For JDBC mode, set env vars: JDBC_URL, JDBC_USER (optional), JDBC_PASS (optional).
 */
public final class Core {
    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            printUsageAndExit();
        }
        String mode = args[0].toLowerCase();
        switch (mode) {
            case "console" -> runConsole();
            case "file" -> runFile();
            case "jdbc" -> runJdbc();
            default -> printUsageAndExit();
        }
    }

    private static void runConsole() {
        AuditSink sink = new ConsoleAuditSink(AuditOptions.builder().dryRun(false).build());
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        sink.write(rec, AuditOptions.builder().dryRun(false).build()).join();
        System.out.println("console: wrote record " + rec);
    }

    private static void runFile() {
        File dir = new File("audit-logs");
        AuditSink sink = new FileAuditSink(dir, "audit", 10_000_000, 60_000, 1024);
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        sink.write(rec, AuditOptions.builder().dryRun(false).build()).join();
        System.out.println("file: wrote record to " + dir.getAbsolutePath());
    }

    private static void runJdbc() {
        String url = System.getenv("JDBC_URL");
        if (url == null || url.isEmpty()) {
            System.err.println("Missing JDBC_URL env var. Example: jdbc:postgresql://localhost:5432/qualia");
            System.exit(2);
        }
        String user = System.getenv("JDBC_USER");
        String pass = System.getenv("JDBC_PASS");

        AuditSink sink = new JdbcAuditSink(url, user, pass);
        AuditRecord rec = new AuditRecordImpl("rec-" + System.currentTimeMillis(), new Date());
        AuditOptions opts = AuditOptions.builder().idempotencyKey(rec.id()).dryRun(false).build();
        sink.write(rec, opts).join();
        System.out.println("jdbc: wrote record " + rec + " to " + url);

        if (sink instanceof JdbcAuditSink j) {
            j.close();
        }
    }

    private static void printUsageAndExit() {
        System.err.println("Usage: java -cp <cp> qualia.Core <console|file|jdbc>");
        System.exit(1);
    }
}



