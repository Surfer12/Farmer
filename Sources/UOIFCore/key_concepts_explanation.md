
Quick status: I’ll give plain‑English explanations with tiny examples for each concept. No blockers.

### Gauge freedom
- What it is: Renaming or trivially reparameterizing the knobs doesn’t change behavior. If two setups compute the same score function, they’re the same policy—just different labels.
- Intuition: Swap `α`↔`α_g`, `λ_i`↔`λ_{ig}`, `β`↔`β_g`; outputs and rankings stay identical.
- Why it matters: Avoids arguing about names/defaults when the structure (linear blend + exponential penalty + cap) is unchanged.

### Threshold transfer
- What it is: If the uplift factor changes, you can rescale the decision threshold to keep the same accept/reject decisions (while not capped).
- Rule of thumb: With uplift going from β to β′, use τ′ = τ · (β/β′).
- Example: β: 1.2 → 1.5 and τ = 0.80 ⇒ τ′ = 0.80·(1.2/1.5) = 0.64. Same decisions in the sub‑cap region.

### Sensitivity invariants
- What it is: The directions of change (signs/zeros of derivatives) don’t flip under trivial reparameterizations or threshold transfers.
- Intuition:
  - Moving weight toward the better signal raises the score: if N>S, then ∂Ψ/∂α < 0 (so lowering α increases Ψ).
  - Increasing risks always lowers the score: ∂Ψ/∂R_a < 0, ∂Ψ/∂R_v < 0.
  - Changing β scales sensitivities but never changes their sign or where they’re zero.
- Why it matters: Tuning remains predictable; monotonicity with canonical evidence is stable.

- Defined each concept simply and gave a quick example for threshold transfer.
- Emphasized why they matter: labels don’t alter behavior, thresholds can be adjusted to preserve decisions, and tuning directions remain stable.