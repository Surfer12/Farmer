// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpsConfigurator;
import com.sun.net.httpserver.HttpsServer;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.time.Duration;
import java.util.Objects;

/**
 * Minimal metrics server exposing Prometheus text format at /metrics.
 *
 * Supports HTTPS when a keystore is provided via environment:
 *   METRICS_PORT (default 9095)
 *   METRICS_TLS_KEYSTORE (path to JKS/PFX)
 *   METRICS_TLS_PASSWORD (password)
 *
 * If TLS env is not provided, falls back to HTTP for local development.
 */
public final class MetricsServer implements AutoCloseable {
    private final HttpServer server;
    private final int port;
    private final boolean https;

    private MetricsServer(HttpServer server, int port, boolean https) {
        this.server = server;
        this.port = port;
        this.https = https;
    }

    public static MetricsServer startFromEnv() {
        int port = parseInt(System.getenv("METRICS_PORT"), 9095);
        String ksPath = System.getenv("METRICS_TLS_KEYSTORE");
        String ksPass = System.getenv("METRICS_TLS_PASSWORD");

        try {
            final HttpServer srv;
            final boolean https;
            if (ksPath != null && !ksPath.isEmpty() && ksPass != null) {
                SSLContext ctx = createSslContext(ksPath, ksPass.toCharArray());
                HttpsServer hs = HttpsServer.create(new InetSocketAddress(port), 0);
                hs.setHttpsConfigurator(new HttpsConfigurator(ctx));
                srv = hs; https = true;
            } else {
                srv = HttpServer.create(new InetSocketAddress(port), 0);
                https = false;
            }
            srv.createContext("/metrics", new MetricsHandler());
            srv.setExecutor(java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "metrics-server"); t.setDaemon(true); return t;
            }));
            srv.start();
            return new MetricsServer(srv, port, https);
        } catch (IOException e) {
            throw new PersistenceException("Failed to start MetricsServer", e);
        }
    }

    public void stop(Duration grace) {
        int secs = (int) Math.max(0, grace == null ? 0 : grace.getSeconds());
        server.stop(secs);
        MetricsRegistry.get().incCounter("metrics_server_stopped_total");
    }

    public int port() { return port; }
    public boolean isHttps() { return https; }

    private static int parseInt(String s, int def) {
        if (s == null) return def;
        try { return Integer.parseInt(s.trim()); } catch (NumberFormatException e) { return def; }
    }

    private static SSLContext createSslContext(String ksPath, char[] password) {
        Objects.requireNonNull(ksPath);
        Objects.requireNonNull(password);
        try (FileInputStream fis = new FileInputStream(ksPath)) {
            KeyStore ks = KeyStore.getInstance(ksPath.endsWith(".p12") || ksPath.endsWith(".pfx") ? "PKCS12" : "JKS");
            ks.load(fis, password);

            KeyManagerFactory kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            kmf.init(ks, password);

            TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            tmf.init(ks);

            SSLContext ctx = SSLContext.getInstance("TLS");
            ctx.init(kmf.getKeyManagers(), tmf.getTrustManagers(), null);
            return ctx;
        } catch (Exception e) {
            throw new ConfigurationException("Invalid TLS keystore for MetricsServer", e);
        }
    }

    private static final class MetricsHandler implements HttpHandler {
        @Override public void handle(HttpExchange ex) throws IOException {
            String body = MetricsRegistry.get().toPrometheus();
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
        }
    }

    @Override
    public void close() {
        stop(Duration.ofSeconds(1));
    }
}


