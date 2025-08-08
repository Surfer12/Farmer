// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.time.Instant;

/**
 * Minimal error reporter that logs to stderr and increments metrics counters.
 * Kept dead-simple to avoid external logging dependencies.
 */
final class ErrorReporter {
    private ErrorReporter() {}

    static void report(String where, Throwable ex) {
        String code = ex instanceof QualiaException qe && qe.code() != null ? qe.code() : "UNCLASSIFIED";
        String sev = ex instanceof QualiaException qe ? qe.severity().name() : QualiaException.Severity.ERROR.name();
        MetricsRegistry.get().incCounter("errors_total");
        MetricsRegistry.get().addCounter("errors_by_code_" + code, 1);
        System.err.println("[" + Instant.now() + "] [" + sev + "] [" + code + "] at " + where + ": " + ex.getMessage());
    }
}


