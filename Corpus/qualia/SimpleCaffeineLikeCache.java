// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.ToLongFunction;

/**
 * A lightweight, thread-safe cache with LRU eviction, expire-after-write TTL, optional
 * max-weight, single-flight loading, and basic stats — designed to mimic the common
 * pattern: cache.get(key, k -> compute(...)).
 *
 * Notes:
 * - Values should be immutable or defensively copied by the loader to ensure safety.
 * - For max-weight, provide a non-null weigher and positive maxWeightBytes; otherwise
 *   weight-based eviction is disabled.
 */
public final class SimpleCaffeineLikeCache<K, V> {

    private static final class Entry<V> {
        final V value;
        final long writeNanos;
        final long weight;
        Entry(V value, long writeNanos, long weight) {
            this.value = value;
            this.writeNanos = writeNanos;
            this.weight = weight;
        }
    }

    public static final class StatsSnapshot {
        public final long hitCount;
        public final long missCount;
        public final long loadSuccessCount;
        public final long loadFailureCount;
        public final long evictionCount;
        public final long totalLoadTimeNanos;
        StatsSnapshot(long h, long m, long ls, long lf, long e, long t) {
            this.hitCount = h; this.missCount = m; this.loadSuccessCount = ls;
            this.loadFailureCount = lf; this.evictionCount = e; this.totalLoadTimeNanos = t;
        }
    }

    private static final class Stats {
        final LongAdder hit = new LongAdder();
        final LongAdder miss = new LongAdder();
        final LongAdder loadOk = new LongAdder();
        final LongAdder loadErr = new LongAdder();
        final LongAdder evict = new LongAdder();
        final LongAdder loadTime = new LongAdder();
        StatsSnapshot snapshot() {
            return new StatsSnapshot(hit.sum(), miss.sum(), loadOk.sum(), loadErr.sum(), evict.sum(), loadTime.sum());
        }
    }

    @FunctionalInterface
    public interface ThrowingFunction<K, V> {
        V apply(K key) throws Exception;
    }

    private final LinkedHashMap<K, Entry<V>> lru;
    private final ConcurrentHashMap<K, CompletableFuture<V>> inFlight = new ConcurrentHashMap<>();
    private final long ttlNanos;
    private final int maxEntries;
    private final long maxWeightBytes;
    private final ToLongFunction<V> weigher; // may be null if weight disabled
    private final boolean recordStats;
    private final Stats stats = new Stats();
    private long currentWeightBytes = 0L;

    public SimpleCaffeineLikeCache(int maxEntries, Duration ttl, boolean recordStats) {
        this(maxEntries, ttl, recordStats, 0L, null);
    }

    public SimpleCaffeineLikeCache(int maxEntries, Duration ttl, boolean recordStats, long maxWeightBytes, ToLongFunction<V> weigher) {
        if (maxEntries <= 0) throw new IllegalArgumentException("maxEntries must be > 0");
        this.maxEntries = maxEntries;
        this.ttlNanos = ttl == null ? 0L : ttl.toNanos();
        this.recordStats = recordStats;
        this.maxWeightBytes = Math.max(0L, maxWeightBytes);
        this.weigher = weigher;
        // access-order LRU
        this.lru = new LinkedHashMap<>(16, 0.75f, true);
    }

    /** Returns a point-in-time stats snapshot. */
    public StatsSnapshot stats() { return stats.snapshot(); }

    /** Clears all entries. */
    public void clear() {
        synchronized (lru) {
            lru.clear();
            currentWeightBytes = 0L;
        }
    }

    /** Removes a specific entry. */
    public void invalidate(K key) {
        Objects.requireNonNull(key, "key");
        synchronized (lru) {
            Entry<V> e = lru.remove(key);
            if (e != null) currentWeightBytes -= e.weight;
        }
    }

    /**
     * Get-or-load with single-flight. Multiple callers for the same key will wait on
     * a single computation. If the value is present and not expired, it is returned.
     */
    public V get(K key, ThrowingFunction<K, V> mappingFunction) throws Exception {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(mappingFunction, "mappingFunction");

        // Fast path: try to return a live entry
        Entry<V> hit;
        synchronized (lru) {
            hit = lru.get(key);
            if (hit != null && !isExpired(hit)) {
                if (recordStats) stats.hit.increment();
                return hit.value;
            }
            if (hit != null) {
                // expired
                removeInternal(key, hit);
            }
            if (recordStats) stats.miss.increment();
        }

        // Single-flight loading
        final CompletableFuture<V> loader = new CompletableFuture<>();
        final CompletableFuture<V> existing = inFlight.putIfAbsent(key, loader);
        if (existing == null) {
            // We are the loader
            long t0 = System.nanoTime();
            try {
                V value = mappingFunction.apply(key);
                long loadNs = System.nanoTime() - t0;
                if (recordStats) { stats.loadOk.increment(); stats.loadTime.add(loadNs); }
                // Insert into cache
                synchronized (lru) {
                    long weight = weigh(value);
                    Entry<V> prior = lru.put(key, new Entry<>(value, System.nanoTime(), weight));
                    if (prior != null) currentWeightBytes -= prior.weight;
                    currentWeightBytes += weight;
                    evictIfNecessary();
                }
                loader.complete(value);
            } catch (Throwable t) {
                if (recordStats) stats.loadErr.increment();
                loader.completeExceptionally(t);
            } finally {
                inFlight.remove(key);
            }
            return loader.get();
        } else {
            // A load is already in progress; wait for it
            return existing.get();
        }
    }

    private boolean isExpired(Entry<V> e) {
        if (ttlNanos <= 0L) return false;
        return (System.nanoTime() - e.writeNanos) > ttlNanos;
    }

    private void removeInternal(K key, Entry<V> e) {
        lru.remove(key);
        currentWeightBytes -= e.weight;
    }

    private void evictIfNecessary() {
        // Size-based eviction first
        while (lru.size() > maxEntries) {
            Map.Entry<K, Entry<V>> eldest = eldest();
            if (eldest == null) break;
            removeInternal(eldest.getKey(), eldest.getValue());
            if (recordStats) stats.evict.increment();
        }
        // Weight-based eviction if enabled
        if (maxWeightBytes > 0L) {
            while (currentWeightBytes > maxWeightBytes) {
                Map.Entry<K, Entry<V>> eldest = eldest();
                if (eldest == null) break;
                removeInternal(eldest.getKey(), eldest.getValue());
                if (recordStats) stats.evict.increment();
            }
        }
    }

    private Map.Entry<K, Entry<V>> eldest() {
        // LinkedHashMap in access-order: eldest is iteration first element
        var it = lru.entrySet().iterator();
        return it.hasNext() ? it.next() : null;
    }

    private long weigh(V value) {
        if (weigher == null) return 0L;
        try {
            long w = weigher.applyAsLong(value);
            return Math.max(0L, w);
        } catch (Throwable t) {
            return 0L;
        }
    }
}


