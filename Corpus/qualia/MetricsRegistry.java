;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Minimal metrics registry supporting counters and gauges.
 */
public final class MetricsRegistry {
    private static final MetricsRegistry INSTANCE = new MetricsRegistry();

    public static MetricsRegistry get() { return INSTANCE; }

    private final Map<String, AtomicLong> counters = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> gauges = new ConcurrentHashMap<>();

    private MetricsRegistry() {}

    public void incCounter(String name) {
        counters.computeIfAbsent(name, k -> new AtomicLong()).incrementAndGet();
    }

    public void addCounter(String name, long delta) {
        counters.computeIfAbsent(name, k -> new AtomicLong()).addAndGet(delta);
    }

    public void setGauge(String name, long value) {
        gauges.computeIfAbsent(name, k -> new AtomicLong()).set(value);
    }

    public String toPrometheus() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, AtomicLong> e : counters.entrySet()) {
            sb.append(e.getKey()).append("_total").append(' ').append(e.getValue().get()).append('\n');
        }
        for (Map.Entry<String, AtomicLong> e : gauges.entrySet()) {
            sb.append(e.getKey()).append(' ').append(e.getValue().get()).append('\n');
        }
        return sb.toString();
    }
}


