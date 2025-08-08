
Creating new non-blocking audit sinks: a JSON-lines file sink with size/time rotation and an HTTP sink with retry/backoff. Also resolving duplicate audit type definitions by introducing `AuditRecord.java`, adding a separate `ConsoleAuditSink.java`, and removing the conflicting `core.java`. Then I'll add a small JSON utility used by both sinks.
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool

Planning to: 1) delete `Corpus/qualia/core.java` to remove duplicate audit type definitions, 2) adjust `FileAuditSink` to use a bounded platform-thread executor (safer than virtual threads for a worker pool), 3) adjust `HttpAuditSink` to use `CompletableFuture.delayedExecutor` for backoff, and 4) update `TODO.md` checkboxes for the two sinks and serialization. Proceeding to read current files and then apply edits.
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool

Ran tool

Updating packages to `qualia` for model classes and cleaning unused imports to resolve linter errors. Then I’ll re-run lint checks.
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Quick status: Implemented the initial persistent audit sinks and cleaned up the audit API. Removed the legacy aggregator to avoid duplicates, unified packages under `qualia`, and added a minimal JSON utility. Updated `TODO.md` to reflect progress.

### What I built
- File sink (JSON Lines) with rotation
  - `qualia.FileAuditSink`: writes one JSON object per line; rotates by size or time; non-blocking via a small bounded executor.
- HTTP sink (POST + retry/backoff)
  - `qualia.HttpAuditSink`: async `HttpClient`, exponential backoff with a cap, uses `CompletableFuture.delayedExecutor`.
- Console sink separated
  - `qualia.ConsoleAuditSink`: retains dev-friendly behavior; non-blocking.
- Shared utilities and interfaces
  - `qualia.AuditRecord` (interface) and `qualia.AuditRecordImpl` (existing).
  - `qualia.JsonUtil` for escaping and ISO-8601 timestamps.
  - Removed `Corpus/qualia/core.java` (duplicate/legacy audit bundle).
- Package consistency
  - Moved `HierarchicalBayesianModel`, `ModelParameters`, `ModelPriors`, and `HB` to package `qualia`.

### Files added/edited (highlights)
- Added: `qualia/AuditRecord.java`, `qualia/ConsoleAuditSink.java`, `qualia/FileAuditSink.java`, `qualia/HttpAuditSink.java`, `qualia/JsonUtil.java`
- Edited: `qualia/HierarchicalBayesianModel.java`, `qualia/ModelParameters.java`, `qualia/ModelPriors.java`, `qualia/HB.java`
- Removed: `qualia/core.java` (eliminated duplicates)
- Updated: `Corpus/qualia/TODO.md` to check off file sink, HTTP sink, and serialization.

### How to use (minimal examples)
- File sink:
  ```java
  AuditSink sink = new FileAuditSink(new File("logs/audit"), "audit", 10_000_000, 3_600_000, 1024);
  sink.write(new AuditRecordImpl("id-1", new Date()), AuditOptions.builder().dryRun(false).build());
  ```
- HTTP sink:
  ```java
  AuditSink sink = new HttpAuditSink(URI.create("https://example.com/audit"), 5, Duration.ofMillis(200));
  sink.write(new AuditRecordImpl("id-2", new Date()), AuditOptions.builder().dryRun(false).build());
  ```

- `TODO.md` changes:
  - File-based sink: done
  - HTTP sink: done (with retry/backoff)
  - Serialization: done (JSON Lines)

- Implemented `qualia.FileAuditSink` (JSONL + rotation), `qualia.HttpAuditSink` (async + retry/backoff), and `qualia.ConsoleAuditSink`.
- Added `qualia.AuditRecord` and `qualia.JsonUtil`. Removed `qualia/core.java`. Unified packages to `qualia`.
- Updated `Corpus/qualia/TODO.md` to mark sinks and serialization complete.