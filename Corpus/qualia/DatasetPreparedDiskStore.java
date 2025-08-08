// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.UUID;

/**
 * Simple disk store for HierarchicalBayesianModel.Prepared arrays.
 * Files are written to <dir>/<fingerprint>.bin with a small header for versioning.
 */
public final class DatasetPreparedDiskStore {
    private static final String MAGIC = "DPDS1"; // Disk Prepared Dataset Store v1
    private static final int VERSION = 1;

    private final Path directory;

    public DatasetPreparedDiskStore(File dir) { this(dir.toPath()); }

    public DatasetPreparedDiskStore(Path dir) {
        Objects.requireNonNull(dir, "dir");
        this.directory = dir;
        try { Files.createDirectories(directory); } catch (IOException e) { throw new IllegalStateException(e); }
    }

    private Path pathFor(DatasetCacheKey key) {
        return directory.resolve(key.fingerprint() + ".bin");
    }

    /** Returns null if not present or on a read error. */
    public HierarchicalBayesianModel.Prepared readIfPresent(DatasetCacheKey key) {
        Path p = pathFor(key);
        if (!Files.isReadable(p)) return null;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(p)))) {
            // Header
            byte[] magic = new byte[MAGIC.length()];
            in.readFully(magic);
            if (!MAGIC.equals(new String(magic))) return null;
            int ver = in.readInt();
            if (ver != VERSION) return null;
            int n = in.readInt();
            if (n < 0 || n > (1 << 27)) return null; // basic sanity bound

            double[] pen = new double[n];
            double[] pHe = new double[n];
            boolean[] y = new boolean[n];
            for (int i = 0; i < n; i++) pen[i] = in.readDouble();
            for (int i = 0; i < n; i++) pHe[i] = in.readDouble();
            for (int i = 0; i < n; i++) y[i] = in.readBoolean();
            return new HierarchicalBayesianModel.Prepared(pen, pHe, y);
        } catch (EOFException eof) {
            // treat as missing/corrupt
            return null;
        } catch (IOException ex) {
            return null;
        }
    }

    public void write(DatasetCacheKey key, HierarchicalBayesianModel.Prepared prep) throws IOException {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(prep, "prep");
        Path target = pathFor(key);
        String tmpName = key.fingerprint() + ".tmp-" + UUID.randomUUID();
        Path tmp = directory.resolve(tmpName);
        int n = prep.size();
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp)))) {
            // Header
            out.write(MAGIC.getBytes());
            out.writeInt(VERSION);
            out.writeInt(n);
            // Payload
            for (int i = 0; i < n; i++) out.writeDouble(prep.pen[i]);
            for (int i = 0; i < n; i++) out.writeDouble(prep.pHe[i]);
            for (int i = 0; i < n; i++) out.writeBoolean(prep.y[i]);
        }
        // Atomic move if supported; otherwise fallback to replace existing
        try {
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException atomicUnsupported) {
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}


