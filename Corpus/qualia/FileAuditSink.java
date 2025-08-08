package qualia;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

/**
 * JSON Lines file sink with size/time-based rotation.
 *
 * <p>Thread-safe and non-blocking: writes are dispatched to a bounded executor.
 * Rotation occurs when either file size exceeds {@code maxBytes} or file age
 * exceeds {@code maxMillis}. Each line is a single JSON object describing an
 * {@link AuditRecord}. No external JSON library is required.
 */
public final class FileAuditSink implements AuditSink {
    private final File directory;
    private final String filePrefix;
    private final long maxBytes;
    private final long maxMillis;
    private final ExecutorService executor;

    private volatile BufferedWriter writer;
    private volatile File currentFile;
    private volatile long createdAtMs;

    /**
     * Creates a file sink.
     * @param directory output directory (created if missing)
     * @param filePrefix prefix for rotated files
     * @param maxBytes maximum file size before rotation; defaults to 10MB if <=0
     * @param maxMillis maximum file age before rotation; defaults to 1h if <=0
     * @param maxQueue maximum queued writes before back-pressure via executor rejection
     */
    public FileAuditSink(File directory, String filePrefix, long maxBytes, long maxMillis, int maxQueue) {
        this.directory = Objects.requireNonNull(directory, "directory");
        this.filePrefix = Objects.requireNonNullElse(filePrefix, "audit");
        this.maxBytes = maxBytes <= 0 ? 10_000_000 : maxBytes; // default 10MB
        this.maxMillis = maxMillis <= 0 ? TimeUnit.HOURS.toMillis(1) : maxMillis; // default 1h
        this.executor = newBoundedExecutor(maxQueue);
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IllegalStateException("Could not create directory: " + directory);
        }
        rotate();
    }

    private static ExecutorService newBoundedExecutor(int maxQueue) {
        LinkedBlockingQueue<Runnable> queue = new LinkedBlockingQueue<>(Math.max(128, maxQueue));
        ThreadFactory tf = r -> {
            Thread t = new Thread(r, "audit-file-");
            t.setDaemon(true);
            return t;
        };
        return new ThreadPoolExecutorCompat(1, 2, 60, TimeUnit.SECONDS, queue, tf);
    }

    /**
     * Rotates the output file, closing the previous writer if open.
     */
    private synchronized void rotate() {
        closeQuietly();
        createdAtMs = System.currentTimeMillis();
        String name = filePrefix + "-" + createdAtMs + ".jsonl";
        currentFile = new File(directory, name);
        try {
            writer = new BufferedWriter(new FileWriter(currentFile, true));
        } catch (IOException e) {
            throw new RuntimeException("Failed to open audit file", e);
        }
    }

    /**
     * Flushes and closes the current writer, ignoring secondary IO errors.
     */
    private synchronized void closeQuietly() {
        if (writer != null) {
            try { writer.flush(); writer.close(); } catch (IOException ignored) {}
            writer = null;
        }
    }

    /**
     * Writes a single line and triggers rotation if thresholds are exceeded.
     */
    private synchronized void writeLine(String line) throws IOException {
        if (writer == null) rotate();
        writer.write(line);
        writer.write('\n');
        writer.flush();
        if (currentFile.length() >= maxBytes || System.currentTimeMillis() - createdAtMs >= maxMillis) {
            rotate();
        }
    }

    @Override
    public CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts) {
        return CompletableFuture.runAsync(() -> {
            String json = toJson(rec);
            try {
                writeLine(json);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }, executor);
    }

    /**
     * Serializes a record to a compact JSON object.
     */
    private static String toJson(AuditRecord rec) {
        String id = JsonUtil.escape(rec.id());
        String ts = JsonUtil.toIso8601(rec.timestamp());
        return "{\"id\":\"" + id + "\",\"timestamp\":\"" + ts + "\"}";
    }
}

/**
 * Minimal compatibility class to avoid a hard dependency surface in this module.
 */
final class ThreadPoolExecutorCompat extends java.util.concurrent.ThreadPoolExecutor {
    ThreadPoolExecutorCompat(int corePoolSize, int maximumPoolSize, long keepAliveTime, TimeUnit unit,
                             java.util.concurrent.BlockingQueue<Runnable> workQueue, ThreadFactory threadFactory) {
        super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory, new AbortPolicy());
        allowCoreThreadTimeOut(true);
    }
}


