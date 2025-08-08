// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.Date;

/**
 * Represents an immutable audit record.
 *
 * <p>Implementations must defensively copy mutable fields (e.g., {@link Date})
 * to preserve immutability. Records should be safe to share across threads.
 */
public interface AuditRecord {
    /**
     * Unique identifier for the record, assigned by the caller.
     */
    String id();

    /**
     * Timestamp of the record creation in UTC or caller-defined timezone.
     * Implementations should return a defensive copy to avoid external mutation.
     */
    Date timestamp();
}


