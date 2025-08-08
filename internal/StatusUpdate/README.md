SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

# StatusUpdate (Internal)

Classification: Confidential — Internal Use Only

Purpose
- Single, coherent place for incremental state logs across the codebase (HMC, MCDA, UPOCF triad, Audit, Docs, Security/Ops).
- Append-only `status.jsonl` with one JSON object per line conforming to `status.schema.json`.

Where
- Directory: `internal/StatusUpdate/`
  - `status.jsonl` — canonical log (append-only)
  - `status.schema.json` — schema for entries
  - `STATUS_UPDATE_TEMPLATE.md` — copy/paste template for quick updates

Entry fields (see schema for full details)
- `ts` (ISO-8601): Timestamp of the update
- `component` (string): Area being updated (e.g., "HMC Adaptive Warmup", "UPOCF Triangulation")
- `status` (enum): one of `planned | in_progress | done | blocked`
- `summary` (string): 1–3 sentence summary
- `changes` ([string]): Notable edits/landed work since last entry
- `next` ([string]): Next concrete actions
- `blockers` ([string]): True blockers only, or empty
- `metrics` (object): Key metrics (e.g., acceptance, R̂, ESS, divergences, latency_ms, eps_rk4_max, taylor_r4, curvature_ci)
- `refs` ([{type, path|url, lines?, note?}]): Pointers to files/PRs/commits/URLs
- `classification` (string): Must be "Confidential — Internal Use Only"
- `version` (int): Schema version (start at 1)

Conventions
- Append entries (do not rewrite). Correct errors with a follow-up entry.
- Prefer concise bullets in `changes`/`next`.
- Use `refs` to cite the exact files or sections updated.
- Mirror CI/job metrics where applicable for traceability.

CLI (examples)
```sh
# Append an entry (macOS/BSD sed-safe)
cat >> internal/StatusUpdate/status.jsonl <<'JSON'
{"ts":"2025-08-08T12:34:56Z","component":"HMC Adaptive Warmup","status":"in_progress","summary":"Dual-averaging + diag mass in place; tuning toward target acc.","changes":["Added dual-averaging step size","Added diag mass adaptation","JSONL/CLI wired"],"next":["Sweep (gamma, leap, phase)","Regress acc into 0.7–0.8 band","Add tests (grad check, diagnostics)"],"blockers":[],"metrics":{"acceptance":0.62,"divergences":3},"refs":[{"type":"file","path":"Corpus/qualia/HmcSampler.java","note":"sampleAdaptive"}],"classification":"Confidential — Internal Use Only","version":1}
JSON
```

Validation
- Validate locally using any JSON Schema validator against `status.schema.json`.

Notes
- `REUSE.toml` marks `internal/**` as `LicenseRef-Internal-Use-Only`; no SPDX line inside JSONL needed.
- Keep entries terse; the goal is coherence and quick situational awareness.


