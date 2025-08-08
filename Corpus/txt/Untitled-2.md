Reviewing current Java files in `Corpus/qualia` and adding comprehensive JavaDoc across each public type and member, including class purpose, thread-safety, parameter/return semantics, invariants, and usage notes. Then I’ll run a lint check to catch any doc formatting or import issues.
Ran tool

Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool

Adding comprehensive JavaDoc across all `qualia` Java files: documenting classes, methods, record components, thread-safety, parameters/returns, invariants, and usage notes. Then I’ll run lints.
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool
Quick status: Added comprehensive JavaDoc across `qualia` classes/interfaces and tightened docs on behavior, invariants, thread-safety, params/returns, and usage. Lint passes clean.

### Updated files (JavaDoc highlights)
- `qualia.AuditOptions`
  - Class-level: immutability, builder semantics, thread-safety.
  - Methods: `idempotencyKey()`, `dryRun()`, builder setters with param docs.
- `qualia.AuditRecord`
  - Class-level: immutability requirements, defensive copies.
  - Methods: `id()`, `timestamp()` with invariants.
- `qualia.AuditRecordImpl`
  - Class-level: thread-safety and defensive copying.
  - Constructor and `toString()` docs.
- `qualia.AuditSink`
  - Class-level: async semantics, non-blocking guidance.
  - Methods: `write(...)` params/returns, `noop()`, `composite(...)` behavior.
- `qualia.ConsoleAuditSink`
  - Class-level: dev use, non-blocking print.
  - Constructor and `write(...)` details.
- `qualia.FileAuditSink`
  - Class-level: JSONL format, rotation policy, thread-safety.
  - Constructor parameters, `rotate()`, `closeQuietly()`, `writeLine()`, and JSON serialization docs.
- `qualia.HttpAuditSink`
  - Class-level: async POST, retry/backoff behavior.
  - Constructor params, `sendWithRetry(...)`, backoff and delay helpers, serialization docs.
- `qualia.JsonUtil`
  - Class-level: scope, thread-safety.
  - `escape(...)` and `toIso8601(...)` docs.
- `qualia.ModelParameters`
  - Class-level: invariants S,N,α∈[0,1]; β>0.
- `qualia.ModelPriors`
  - Class-level: role of hyperparameters.
  - `defaults()` docs.
- `qualia.HB`
  - Class-level: placeholder/shim note directing to real classes.
- `qualia.HierarchicalBayesianModel`
  - Class-level: full Ψ structure, thread-safety.
  - Docs for `calculatePsi`, `logLikelihood`, `totalLogLikelihood`, `logPriors`, `logPosterior`, `performInference`.

- Added detailed JavaDoc across all `qualia` Java files.
- Ensured thread-safety and invariants are documented; async behavior clarified for sinks.
- Lint is green.