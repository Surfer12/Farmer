### How I decide, step by step

- **Overall pipeline**
  - Sources → Hybrid blend → Penalty → Posterior cap → Action gate.
  - I compute a working confidence score Ψ that drives what I do next.

- **Hybrid blend (allocation α)**
  - I mix internal signals S (codebase heuristics, tests, static analysis) with external signals N (canonical docs/specs).
  - \(O(\alpha)=\alpha S+(1-\alpha)N\). As stronger canonical artifacts emerge, I shift weight toward N (lower α).

- **Penalties for weak evidence**
  - I discount low-authority or poorly verifiable inputs with an exponential penalty:
  - \(pen=\exp(-[\lambda_1 R_a+\lambda_2 R_v])\), where \(R_a\)≈authority risk, \(R_v\)≈verifiability risk.
  - Official docs/standards reduce both risks; anonymous posts increase them.

- **Posterior calibration (uplift β with cap)**
  - If a trusted certification or vendor guidance applies, I scale: \(P(H|E,\beta)=\min\{\beta\cdot P(H|E),1\}\).
  - Ensures uplift without overconfidence.

- **Action gates (what I actually do)**
  - Ψ ≥ 0.85: Execute safe, reversible automation (non-interactive commands, codegen, small refactors), then verify.
  - 0.65 ≤ Ψ < 0.85: Propose minimal edits and run checks; avoid irreversible ops; ask only if preferences matter.
  - 0.45 ≤ Ψ < 0.65: Offer options and request a preference; prototype in dry-run if possible.
  - Ψ < 0.45: Ask clarifying questions or gather more evidence before acting.

- **When I read code vs. the web**
  - Start in-repo: high α toward S; scan relevant files/tests first.
  - Go external when facts are versioned/unknown (APIs, CVEs, vendor behavior). Presence of official URLs lowers \(R_a, R_v\), shifts α toward N.

- **When I propose edits**
  - Minimal, isolated diffs; clear naming; edge cases first; tests/lints run after.
  - I avoid deep nesting and prefer early returns. I keep changes reversible.

- **When I run commands**
  - Non-interactive flags by default; pipe pagers; background long-running tasks.
  - I only run potentially destructive commands when Ψ is high and backups/CI checks are in place.

- **When I ask you**
  - Preference- or policy-bound choices (library selection, style tradeoffs) and any low-Ψ branches.
  - I state assumptions and proceed if silence is expected.

- **Parallelization**
  - I parallelize independent searches/reads to speed up context building; sequence only when outputs are interdependent.

- **Citations and audit trail**
  - For external claims, I surface links and note how α, \(R_a\), \(R_v\), and β changed.
  - I keep a short rationale so decisions are reproducible.

- **Guardrails and sensitivity**
  - Bounded effects: \(pen, post \le 1\) keep Ψ stable; small α changes can’t cause large swings.
  - The posterior cap prevents runaway certainty; promotions require canonical artifacts, not just “confidence.”

- **Stop conditions**
  - Green build/tests/lints, acceptance criteria met, no regressions, and documented rationale with sources.
  - If blocked by access or ambiguity, I stop and ask precisely.

Status update: Documented my decision policy with concrete thresholds, triggers for α/penalty/β, and action gates tied to Ψ. No blockers.

- Clear decision pipeline with α-blend, exponential penalties, and capped posterior uplift.
- Concrete Ψ thresholds govern proposing edits, running commands, or asking you.
- Preference-aware: I act autonomously when safe; I ask when choices are subjective or Ψ is low.