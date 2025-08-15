// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
;

import java.util.Map;

/**
 * JSON serializer that applies redaction/hashing and optional field-level encryption.
 */
final class SecureJsonSerializer {
    private final Redactor redactor;
    private final byte[] encryptionKey; // 32 bytes if provided; null for no encryption

    SecureJsonSerializer(Redactor redactor, byte[] encryptionKey) {
        this.redactor = redactor;
        this.encryptionKey = encryptionKey;
    }

    String toJson(ExtendedAuditRecord rec, Map<String, Object> inputs, Map<String, Object> outputs, Map<String, Object> params) {
        Map<String, Object> redIn = redactor.apply(inputs);
        Map<String, Object> redOut = redactor.apply(outputs);
        Map<String, Object> redPar = redactor.apply(params);

        StringBuilder sb = new StringBuilder(256);
        sb.append('{');
        appendField(sb, "id", rec.id()); sb.append(',');
        appendField(sb, "timestamp", JsonUtil.toIso8601(rec.timestamp())); sb.append(',');
        appendField(sb, "event", rec.event()); sb.append(',');
        appendField(sb, "schemaVersion", rec.schemaVersion()); sb.append(',');
        appendObject(sb, "inputs", redIn); sb.append(',');
        appendObject(sb, "outputs", redOut); sb.append(',');
        appendObject(sb, "params", redPar);
        sb.append('}');
        return sb.toString();
    }

    private void appendField(StringBuilder sb, String key, String value) {
        sb.append('"').append(JsonUtil.escape(key)).append('"').append(':');
        sb.append('"').append(JsonUtil.escape(value)).append('"');
    }

    private void appendObject(StringBuilder sb, String key, Map<String, Object> map) {
        sb.append('"').append(JsonUtil.escape(key)).append('"').append(':');
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(JsonUtil.escape(e.getKey())).append('"').append(':');
            Object v = e.getValue();
            if (v == null) {
                sb.append("null");
            } else if (v instanceof Number || v instanceof Boolean) {
                sb.append(v.toString());
            } else {
                String s = String.valueOf(v);
                if (encryptionKey != null && shouldEncryptField(e.getKey())) {
                    s = Crypto.encryptToBase64(s, encryptionKey);
                }
                sb.append('"').append(JsonUtil.escape(s)).append('"');
            }
        }
        sb.append('}');
    }

    private boolean shouldEncryptField(String fieldName) {
        // Minimal example: encrypt any field ending with ":enc"; production would use a policy
        return fieldName.endsWith(":enc");
    }
}


