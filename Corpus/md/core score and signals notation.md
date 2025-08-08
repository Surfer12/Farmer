
### Core score and signals
- S(x) ∈ [0,1]: internal signal (in‑repo/tests/static analysis) for item x
- N(x) ∈ [0,1]: external/canonical signal (specs/official docs) for x
- α ∈ [0,1]: allocation weight toward S (1−α toward N)
- O(α) = α S + (1−α) N ∈ [0,1]: hybrid (affine) evidence blend
- R_a ≥ 0, R_v ≥ 0: authority and verifiability/version risks
- λ1>0, λ2>0: risk weights
- pen = exp(−[λ1 R_a + λ2 R_v]) ∈ (0,1]: multiplicative risk penalty
- P(H|E) ∈ [0,1]: base posterior for hypothesis H given evidence E
- β ≥ 1: calibrated uplift factor (for certified/canonical evidence)
- post = min{β · P(H|E), 1} ∈ (0,1]: capped posterior
- Ψ = min{β · O(α) · pen, 1} ∈ [0,1]: guardrailed confidence score (pointwise)

### Thresholds and transfer
- τ ∈ (0,1]: action threshold (gate)
- κ > 0: uniform scaling of β
- Sub‑cap region U(β) = {β · O(α) · pen < 1}
- Threshold transfer (sub‑cap): if β→κ·β, preserve accepts by τ′ = κ · τ

### Sensitivities (sub‑cap)
- ∂Ψ/∂α = (S − N) · β · pen
- ∂Ψ/∂R_a = −β · O(α) · λ1 · pen
- ∂Ψ/∂R_v = −β · O(α) · λ2 · pen
- Monotone: increasing N or decreasing risks raises Ψ; decreasing α raises Ψ when N>S

### Temporal extension (optional, normalized)
- t ∈ T (time/index); w(t) ≥ 0 with ∫_T w(t) dt = 1 (or ∑ w_t = 1)
- O_t = α(t) S(t) + (1−α(t)) N(t); pen_t, β(t), P_t analogous
- Instantaneous Ψ_t = O_t · pen_t · min{β(t)·P_t, 1}
- Averaged score Ψ̄ = ∫ w(t) Ψ_t dt (or ∑ w_t Ψ_t) ∈ [0,1]
- Soft‑cap aggregator: R_t = β(t) O_t pen_t; I = ∫ w(t) R_t dt; Ψ̄ = 1 − exp(−I)
- Threshold transfer under β→κ·β: averaged (sub‑cap) τ′ = κ·τ; soft‑cap τ′ = 1 − (1 − τ)^κ

### MCDA/rules (when selecting among discrete options)
- A: option set; C: hard constraints; F = {a∈A | constraints satisfied}
- c(a) = [Ψ(a), cost(a), time(a), ROI(a), …]: criteria vector
- w (MCDA weights), M(c,w): monotone aggregator in the Ψ coordinate
- Decision: pick argmax_{a∈F} M(c(a), w)

- Defined every symbol used: core score, risks, uplift, thresholds, temporal weights/aggregators, and MCDA/rules terms.