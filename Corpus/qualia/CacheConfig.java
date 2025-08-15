// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions


import java.time.Duration;

/**
 * Configuration for the Prepared-dataset cache layers, sourced from system properties
 * or environment variables.
 *
 * Property/env names (system property takes precedence):
 * - QUALIA_PREP_CACHE_ENABLED: true|false (default true)
 * - QUALIA_PREP_DISK_ENABLED: true|false (default true)
 * - QUALIA_PREP_CACHE_MAX_ENTRIES: int (default 256)
 * - QUALIA_PREP_CACHE_TTL_SECONDS: long (default 21600 = 6h)
 * - QUALIA_PREP_CACHE_MAX_WEIGHT_BYTES: long (default 134217728 = 128MB)
 * - QUALIA_PREP_DISK_DIR: string path (default "prep-cache")
 */
public final class CacheConfig {
    public final boolean memoryEnabled;
    public final boolean diskEnabled;
    public final int maxEntries;
    public final long ttlSeconds;
    public final long maxWeightBytes;
    public final String diskDir;

    private CacheConfig(boolean memoryEnabled,
                        boolean diskEnabled,
                        int maxEntries,
                        long ttlSeconds,
                        long maxWeightBytes,
                        String diskDir) {
        this.memoryEnabled = memoryEnabled;
        this.diskEnabled = diskEnabled;
        this.maxEntries = maxEntries;
        this.ttlSeconds = ttlSeconds;
        this.maxWeightBytes = maxWeightBytes;
        this.diskDir = diskDir;
    }

    public Duration ttlDuration() { return Duration.ofSeconds(Math.max(0, ttlSeconds)); }

    public static CacheConfig fromEnv() {
        boolean mem = getBool("QUALIA_PREP_CACHE_ENABLED", true);
        boolean disk = getBool("QUALIA_PREP_DISK_ENABLED", true);
        int entries = getInt("QUALIA_PREP_CACHE_MAX_ENTRIES", 256);
        long ttl = getLong("QUALIA_PREP_CACHE_TTL_SECONDS", 6L * 60L * 60L);
        long weight = getLong("QUALIA_PREP_CACHE_MAX_WEIGHT_BYTES", 128L * 1024L * 1024L);
        String dir = getStr("QUALIA_PREP_DISK_DIR", "prep-cache");
        return new CacheConfig(mem, disk, Math.max(1, entries), Math.max(0L, ttl), Math.max(0L, weight), dir);
    }

    private static String get(String key) {
        String sys = System.getProperty(key);
        if (sys != null && !sys.isBlank()) return sys;
        String env = System.getenv(key);
        return env;
    }

    private static boolean getBool(String key, boolean dflt) {
        String v = get(key);
        if (v == null) return dflt;
        return switch (v.trim().toLowerCase()) {
            case "true", "1", "yes", "y", "on" -> true;
            case "false", "0", "no", "n", "off" -> false;
            default -> dflt;
        };
    }

    private static int getInt(String key, int dflt) {
        String v = get(key);
        if (v == null) return dflt;
        try { return Integer.parseInt(v.trim()); } catch (Exception e) { return dflt; }
    }

    private static long getLong(String key, long dflt) {
        String v = get(key);
        if (v == null) return dflt;
        try { return Long.parseLong(v.trim()); } catch (Exception e) { return dflt; }
    }

    private static String getStr(String key, String dflt) {
        String v = get(key);
        return (v == null || v.isBlank()) ? dflt : v;
    }
}


