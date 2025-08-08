// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

/**
 * Minimal JSON utilities for audit sinks (no third-party deps).
 *
 * <p>Provides escaping and ISO-8601 timestamp formatting. Methods are static
 * and thread-safe; date formatting uses a {@link ThreadLocal} formatter.
 */
final class JsonUtil {
    private static final ThreadLocal<SimpleDateFormat> ISO8601 = ThreadLocal.withInitial(() -> {
        SimpleDateFormat fmt = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
        fmt.setTimeZone(TimeZone.getTimeZone("UTC"));
        return fmt;
    });

    private JsonUtil() {}

    /**
     * Escapes a string for safe inclusion in JSON string literal context.
     * @param s input string or {@code null}
     * @return escaped text (or literal {@code "null"} if input is null)
     */
    static String escape(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }

    /**
     * Formats a {@link Date} to ISO-8601 using UTC timezone.
     */
    static String toIso8601(Date date) {
        return ISO8601.get().format(date);
    }
}


