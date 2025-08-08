// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/** Cache key composed of dataset identity, content hash, params hash, and algo version. */
public record DatasetCacheKey(String datasetId,
                              String datasetHash,
                              String paramsHash,
                              String algoVersion) {

    public DatasetCacheKey {
        Objects.requireNonNull(datasetId, "datasetId");
        Objects.requireNonNull(datasetHash, "datasetHash");
        Objects.requireNonNull(paramsHash, "paramsHash");
        Objects.requireNonNull(algoVersion, "algoVersion");
    }

    public String fingerprint() {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(datasetId.getBytes(StandardCharsets.UTF_8));
            md.update((byte) 0);
            md.update(datasetHash.getBytes(StandardCharsets.UTF_8));
            md.update((byte) 0);
            md.update(paramsHash.getBytes(StandardCharsets.UTF_8));
            md.update((byte) 0);
            md.update(algoVersion.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override public String toString() { return fingerprint(); }

    public static String hashDatasetContent(List<ClaimData> dataset) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            for (ClaimData c : dataset) {
                // Stable serialization: id|y|ra|rv|pHe; avoid PII other than id already in dataset
                md.update(c.id().getBytes(StandardCharsets.UTF_8));
                md.update((byte) '|');
                md.update((byte) (c.isVerifiedTrue() ? 1 : 0));
                md.update(Double.toString(c.riskAuthenticity()).getBytes(StandardCharsets.UTF_8)); md.update((byte) '|');
                md.update(Double.toString(c.riskVirality()).getBytes(StandardCharsets.UTF_8)); md.update((byte) '|');
                md.update(Double.toString(c.probabilityHgivenE()).getBytes(StandardCharsets.UTF_8));
                md.update((byte) '\n');
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}


