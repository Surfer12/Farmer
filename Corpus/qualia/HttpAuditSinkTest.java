// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.net.URI;
import java.time.Duration;
import java.util.Date;

/**
 * Minimal test to exercise HTTP 5xx retries path. Uses localhost port unlikely to be open.
 */
public final class HttpAuditSinkTest {
    static final class SimpleRecord implements AuditRecord {
        private final String id = "h1";
        private final Date ts = new Date();
        @Override public String id() { return id; }
        @Override public Date timestamp() { return ts; }
    }

    public static void main(String[] args) throws Exception {
        HttpAuditSink sink = new HttpAuditSink(new URI("http://127.0.0.1:59999/doesnotexist"), 2, Duration.ofMillis(50));
        try {
            sink.write(new SimpleRecord(), null).join();
            System.out.println("UNEXPECTED: HTTP write succeeded");
            System.exit(1);
        } catch (Exception e) {
            Throwable cause = e.getCause() != null ? e.getCause() : e;
            if (cause instanceof NetworkException) {
                System.out.println("OK: caught NetworkException after retries");
            } else {
                System.out.println("UNEXPECTED exception type: " + cause.getClass());
                System.exit(1);
            }
        }
    }
}


