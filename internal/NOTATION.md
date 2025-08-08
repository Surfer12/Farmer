SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

© 2025 Jumping Quail Solutions. All rights reserved.
Classification: Confidential — Internal Use Only

## Ψ Framework Notation and JSON/Metric Key Conventions (v0.1)

Purpose: canonicalize names, symbols, and JSON/metric keys used across the codebase (UnifiedDetector, HMC samplers, MCDA glue) to ensure consistency and auditability.

### Global conventions
- **Ranges**: probabilities/confidences in [0,1]; error budgets ε ≥ 0; times in seconds (h) or nanoseconds (latency budgets) as stated.
- **JSON field casing**: snake_case (e.g., `eps_geom`).
- **Java identifiers**: lowerCamelCase for fields/locals; UPPER_SNAKE_CASE for constants.
- **Metrics names**: snake_case with subsystem prefix (e.g., `unified_fastpath_total`).
- **Symbols**: keep Greek names where standard (α, λ, β, Ψ) in docs; map to code/JSON per keys below.

### Core Ψ quantities
- **O(α)** (`hybrid`): affine blend O(α)=α·S+(1−α)·N.
- **pen(r)** (`penalty`): exp(−[λ1·R_a + λ2·R_v]).
- **post** (`posterior`): min{β·P(H|E), 1} (or soft-cap variant where applicable).
- **Ψ** (`psi`): bounded policy confidence in [0,1].
- **R** (`belief_raw`, optional): uncapped belief R=β·O·pen.

JSON keys (when emitted together): `psi`, `hybrid`, `penalty`, `posterior`, optionally `belief_raw`.

### UnifiedDetector: triad gating terms
- **eps_rk4**: RK4 local error estimate from step-doubling controller for the accepted step size h.
- **eps_taylor**: Taylor remainder proxy at order 4: ε_Taylor ≈ (L_eff · h^5)/120, with L_eff=max(Ĺ, κ).
- **geom_drift**: raw drift of invariants; sum over invariants of |value − reference| (absolute drift, no tolerance applied).
- **eps_geom**: invariant violation, not raw drift. Defined as sum(max(0, drift − tol)) over invariants; zero when drift ≤ tol for all invariants.
- **eps_total**: total allowable error budget for the step; gate requires component sums within budget.
- **accepted**: boolean; true iff eps_rk4 ≤ ε_RK4_budget, eps_taylor ≤ ε_Taylor_budget, eps_geom ≤ ε_geom_budget, and (eps_rk4+eps_taylor+eps_geom) ≤ eps_total.
- **agree** (`agreement`): agreement metric in (0,1]; default implementation multiplies 1/(1+||y_A−y_B||) by invariant band factors.
- **h** (`h_used`): step size used for the accepted step, in seconds.

JSON keys: `t`, `x`, `v` (or state components), `psi`, `h`, `eps_rk4`, `eps_taylor`, `eps_geom`, `accepted`, optionally `agree`.

### UnifiedDetector: auxiliary numerics
- **κ (kappa)** (`curvature_kappa`): curvature magnitude estimate from second finite difference of f along a normalized direction of dy.
- **Ĺ (L_hat)** (`lipschitz_hat`): crude Lipschitz bound estimate via finite differences; used in ε_Taylor proxy.
- **trust_region_radius** (`h_trust`): cubic root bound h_trust = cbrt(ε_total / (2·max(κ,ε))): Taylor path only if h ≤ h_trust.
- **fast path**: Euler fallback when time budget exceeded; metrics: `unified_fastpath_total`, `unified_fastpath_extra_err_ppm`.

#### Curvature estimation method (NEW)
- **curvature_kappa_estimator**: estimator for Ollivier–Ricci curvature’s W1 distance.
  - Allowed: `sinkhorn` (entropic-regularized OT; fast, approximate), `network_simplex` (exact W1 via min-cost flow; slower).
  - Default: `sinkhorn`.
  - Replaces: prior implicit/unspecified estimator for `curvature_kappa`; emit this key when logging κ to document the method used.

JSON keys (optional): `curvature_kappa_estimator` with values `sinkhorn` or `network_simplex`.

### HMC/NUTS-related
- **ε (epsilon)** (`step_size`): leapfrog integrator step size.
- **mass_diag**: diagonal elements of the mass matrix (preconditioning) used for momenta.
- **L** (`leapfrog_steps`): number of leapfrog steps in a trajectory (or variable in NUTS).
- **acceptance_rate**: mean acceptance over a (sub)run.
- **divergences** (`divergence_count`): count of divergent trajectories based on energy error/non-finite Hamiltonian.
- **target_accept**: target acceptance rate used by dual-averaging.
- **tuned_step_size**: ε after dual-averaging warmup.
- **R̂ (Rhat)** (`rhat`): split-R̂ convergence diagnostic.
- **ESS** (`ess_bulk`, `ess_tail`): effective sample sizes.
- **z**: unconstrained parameter vector; transforms: S,N,α via logit; β via log; include Jacobian `log_jacobian` in target.

JSON keys (draws): `chain`, `i`, `S`, `N`, `alpha`, `beta` (plus any additional params); meta/summary: `acceptance_rate`, `divergence_count`, `tuned_step_size`, `mass_diag`, `rhat`, `ess_bulk`, `ess_tail`.

### MCDA and gating
- **τ (tau)** (`threshold_tau`): decision threshold for Ψ.
- **threshold_transfer**: when β→β′ in sub-cap, τ′ = τ·(β/β′) to preserve accept/reject.
- **feasible_set**: options with Ψ≥τ and constraints satisfied.

### Invariant interface (geometry)
- **invariant value**: function v(t,y) evaluated on state.
- **reference**: expected invariant value (e.g., initial energy E0 for SHO).
- **tolerance** (`tol`): allowed absolute drift band; used to compute `eps_geom` as violation beyond tolerance.

Recommended JSON for invariants (optional): `invariants`: array of objects `{name, value, reference, tolerance, drift, violation}` where `violation = max(0, drift − tolerance)`.

### Naming rules (MUST)
- JSON/metrics MUST use the exact keys specified above when emitting these quantities.
- `eps_geom` MUST mean invariant violation (post-tolerance), not raw drift; use `geom_drift` for pre-tolerance drift.
- `eps_taylor` MUST denote the remainder proxy at the accepted h; document the order and constant if it changes.
- `eps_rk4` MUST be derived from step-doubling or embedded estimate tied to the accepted h.
- `psi` MUST remain bounded in [0,1]; if publishing uncapped belief, name it `belief_raw`.
- HMC field `step_size` MUST be ε at emission time; `tuned_step_size` MUST be the post-warmup value.

### Units and ranges
- `psi` in [0,1]; `agree` in (0,1].
- `eps_*` nonnegative, dimensionless (same scale as `eps_total`).
- `h` in seconds; latency budgets in nanoseconds.
- Probabilities S,N,α in (0,1); β ≥ 1 (log-space in z parameterization).

### Versioning and change control
- Any addition or change to names/semantics in this document MUST:
  1) update this file with a new minor version;
  2) include a short rationale;
  3) update emitters and tests to match;
  4) note changes in `CHANGELOG.md`.

### Examples
Unified triad JSON row (abbreviated):
```json
{"t":0.001, "x":0.99998, "v":-0.039, "psi":0.942, "h":1.0e-3, "eps_rk4":1.6e-6, "eps_taylor":2.3e-6, "eps_geom":0.0, "accepted":true}
```

HMC summary JSON (abbreviated):
```json
{"chains":4, "acceptance_rate":0.74, "divergence_count":1, "tuned_step_size":0.0125, "mass_diag":[1.1,0.9,2.0,0.8], "rhat":1.01, "ess_bulk":1450, "ess_tail":1600}
```


