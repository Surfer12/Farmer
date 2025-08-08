package qualia;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Field redaction and hashing utilities for audit payloads.
 */
public final class Redactor {
    private final Set<String> redactedFields;
    private final Set<String> hashedFields;

    public Redactor(Set<String> redactedFields, Set<String> hashedFields) {
        this.redactedFields = Objects.requireNonNull(redactedFields);
        this.hashedFields = Objects.requireNonNull(hashedFields);
    }

    public Map<String, Object> apply(Map<String, Object> in) {
        Map<String, Object> out = new HashMap<>();
        for (Map.Entry<String, Object> e : in.entrySet()) {
            String k = e.getKey();
            Object v = e.getValue();
            if (redactedFields.contains(k)) {
                out.put(k, "[REDACTED]");
            } else if (hashedFields.contains(k) && v != null) {
                out.put(k, hash(String.valueOf(v)));
            } else {
                out.put(k, v);
            }
        }
        return out;
    }

    private static String hash(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] h = md.digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(h.length * 2);
            for (byte b : h) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }
}


