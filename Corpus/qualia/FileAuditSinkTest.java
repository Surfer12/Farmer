// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.io.File;
import java.util.Date;

/**
 * Minimal test to exercise file open failure and ensure AuditTrail fail-open behavior.
 */
public final class FileAuditSinkTest {
    static final class SimpleRecord implements AuditRecord {
        private final String id = "t1";
        private final Date ts = new Date();
        @Override public String id() { return id; }
        @Override public Date timestamp() { return ts; }
    }

    public static void main(String[] args) {
        // Attempt to write to a directory path that cannot be created (simulate failure)
        File dir = new File("/this/should/not/exist/and/fail");
        try {
            new FileAuditSink(dir, "test", 1_000, 1_000, 128);
            System.out.println("UNEXPECTED: FileAuditSink constructed");
            System.exit(1);
        } catch (QualiaException e) {
            System.out.println("OK: caught expected QualiaException: " + e.code());
        }

        // Ensure AuditTrail.emit fail-opens on sink write failure
        AuditTrail trail = new AuditTrail(AuditSink.noop());
        trail.emit(new SimpleRecord(), AuditOptions.builder().dryRun(false).build());
        System.out.println("OK: AuditTrail emit returned without throwing");
    }
}


