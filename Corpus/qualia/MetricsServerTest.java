// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/** Simple smoke test: start MetricsServer (HTTP dev mode), GET /metrics, check content. */
public final class MetricsServerTest {
    public static void main(String[] args) throws Exception {
        // ensure we have some counters
        MetricsRegistry.get().incCounter("test_counter");
        MetricsServer ms = MetricsServer.startFromEnv();
        try {
            URL url = new URL("http://127.0.0.1:" + ms.port() + "/metrics");
            HttpURLConnection c = (HttpURLConnection) url.openConnection();
            c.setConnectTimeout(1000);
            c.setReadTimeout(1000);
            int code = c.getResponseCode();
            if (code != 200) { System.out.println("UNEXPECTED HTTP code: " + code); System.exit(1); }
            try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream()))) {
                String all = r.lines().reduce("", (a,b) -> a + b + "\n");
                if (!all.contains("test_counter_total")) {
                    System.out.println("UNEXPECTED: counter missing in metrics");
                    System.exit(1);
                }
            }
            System.out.println("OK: metrics endpoint responded with counters");
        } finally {
            ms.close();
        }
    }
}


