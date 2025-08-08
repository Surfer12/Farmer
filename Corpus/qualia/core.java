package jumping.qualia;

import java.util.concurrent.CompletableFuture;
import java.util.Objects;
import java.util.Date;

/**
 * Represents a sink for audit records.
 */
public interface AuditSink {
    CompletableFuture<Void> write(AuditRecord rec, AuditOptions opts);
}

/**
 * Options for audit operations.
 */
public interface AuditOptions {
    boolean dryRun();
}

/**
 * Represents an audit record.
 */
public interface AuditRecord {
    String id();
    Date timestamp();
}

/**
 * Builder for AuditRecord.
 */
public interface AuditRecordBuilder {
    AuditRecordBuilder withId(String id);
    AuditRecordBuilder withTimestamp(Date timestamp);
    AuditRecord build();
}

/**
 * Factory for creating AuditSink instances.
 */
public interface AuditSinkFactory {
    AuditSink create(AuditOptions opts);
}

/**
 * Builder for AuditSinkFactory.
 */
public interface AuditSinkFactoryBuilder {
    AuditSinkFactoryBuilder withSink(AuditSink sink);
    AuditSinkFactoryBuilder withOptions(AuditOptions opts);
    AuditSinkFactory build();
}

/**
 * Abstract base for AuditSinkFactoryBuilder implementations.
 */
public abstract class AbstractAuditSinkFactoryBuilder implements AuditSinkFactoryBuilder {
    @Override
    public AuditSinkFactoryBuilder withSink(AuditSink sink) {
        // Default no-op, override in concrete builder
        return this;
    }

    @Override
    public AuditSinkFactoryBuilder withOptions(AuditOptions opts) {
        // Default no-op, override in concrete builder
        return this;
    }
}

/**
 * Console implementation of AuditSinkFactoryBuilder.
 */
public class ConsoleAuditSinkFactoryBuilder extends AbstractAuditSinkFactoryBuilder {
    private AuditOptions options;

    @Override
    public AuditSinkFactoryBuilder withOptions(AuditOptions opts) {
        this.options = opts;
        return this;
    }

    @Override
    public AuditSinkFactory build() {
        return new ConsoleAuditSinkFactory(options);
    }
}

/**
 * Console implementation of AuditSinkFactory.
 */
public class ConsoleAuditSinkFactory implements AuditSinkFactory {
    private final AuditOptions options;

    public ConsoleAuditSinkFactory(AuditOptions options) {
        this.options = options;
    }

    @Override
    public AuditSink create(AuditOptions opts) {
        // Prefer provided options, fallback to factory's options
        return new ConsoleAuditSink(opts != null ? opts : this.options);
    }
}

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
