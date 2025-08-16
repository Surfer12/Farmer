### How I ended up with this way to decide (vs. alternatives)

- Constraint‑driven design
  - Need fast, auditable updates; monotone upgrades when canon appears; bounded confidence; reversible actions.
  - Map levers to observable events (α, risks, β) and gate actions by a single score.

- What I tried and why I moved on
  - Naive averages/fixed weights: no authority/verifiability discount; non‑monotone with canon.
  - Softmax/logistic blends: opaque sensitivity; hard to audit; monotonicity not guaranteed.
  - Additive penalties: can break bounds; awkward penalty interactions.
  - Full Bayes: principled but slower/opaque; needs priors/data; overkill for fast loops.
  - RL/bandits: needs reward + safe exploration; not for one‑shot, defensible steps.
  - MCDA/rules: great for constraints/priorities, not continuous confidence.

- Synthesis (why Ψ)
  - Linear blend O(α)=αS+(1−α)N for transparent allocation.
  - Exponential penalty exp(−[λ1Ra+λ2Rv]) for independent risks and boundedness.
  - Capped uplift min{β·O·pen,1} for calibrated promotions without runaway certainty.
  - Single, monotone, bounded score Ψ → clear action gates (execute/propose/ask/gather).

- What made it stick
  - Gauge freedom: parameter renames/defaults don’t change behavior—structure is what matters.
  - Threshold transfer: if β shifts, adjust τ to preserve decisions (sub‑cap).
  - Sensitivity invariants: tuning directions never flip (e.g., more canon always helps).

- When I defer or compose
  - Full Bayes/RL: when labeled data or rewards exist and latency/opacity are acceptable (compose around Ψ).
  - MCDA/rules: encode non‑negotiable constraints and stakeholder weights; feed α and hard gates.

- Net
  - Ψ gives safe, fast, reproducible decisions by default; other methods plug in when their preconditions hold.

- Clear constraints led to the affine+exponential+cap design; alternatives failed monotonicity, auditability, or speed.
- Ψ’s gauge freedom, threshold transfer, and sensitivity invariants keep behavior stable; Bayes/RL/MCDA are composed when justified.