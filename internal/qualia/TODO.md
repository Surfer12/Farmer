# Qualia Package TODO

SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

## 🚀 High Priority

### Architecture & DI
- [x] Introduce `PsiModel` interface and make `HierarchicalBayesianModel` implement it
- [x] Add simple DI: `ModelFactory` and `ServiceLocator` for wiring
- [ ] Introduce abstract base classes where useful (e.g., `AbstractAuditSink`, `AbstractModel`)
- [x] DI configuration via environment: select `AuditSink` (console/file/http/jdbc) and params
- [ ] Unit tests: factory creation paths and service locator bindings

### Core Model Implementation
- [ ] **Complete MCMC inference in `HierarchicalBayesianModel`**
  - [ ] Integrate HMC/NUTS sampler library (e.g., Stan4j, JAGS Java bindings)
  - [x] Implement internal HMC sampler (unconstrained reparam + Jacobian)
  - [x] Env-based tuning knobs for `stepSize`/`leapfrogSteps` (HMC_STEP_SIZE/HMC_LEAP)
  - [x] Dual-averaging step-size adaptation to target 0.65–0.80 acceptance
  - [x] Diagonal mass-matrix adaptation (momentum preconditioning)
  - [ ] Dynamic trajectory length (NUTS-style)
  - [x] Simple divergence checks (energy error, non-finite H) in HMC
  - [x] Multi-chain runner: warmup schedule, seed control (PHI64 spacing), persistent draws (JSONL/meta), summary JSON
  - [x] Metrics export from multi-chain: per-chain acceptance/divergences/ε; diagnostics gauges (R̂)
  - [x] Analytic gradient for SteinGradLogP (replace finite-diff; reuse dlogPost)
  - [x] Switch Stein to z-space origin (logit/exp) with Jacobian-corrected score; expose `gradLogTargetZ` and use prepared, thresholded parallel gradients
  - [ ] Warmup hyperparam sweep: γ∈[0.2,0.3], leap∈[8,15], phase split 15%/60%/25% → pick defaults to hit acc∈[0.70,0.80]
  - [ ] Early-divergence backoff in warmup: reduce ε by 10% on divergence; log per-iteration
  - [ ] RK4 smoothness guard: empirical Lipschitz estimate L̂ for f(Ψ,t); adapt h to ensure global O(h^4) ≤ ε_RK4
  - [ ] Taylor divergence guard: remainder/ratio tests on |R4|; auto-switch to RK4 when out-of-trust-region
  - [ ] Geometry robustness: k-NN smoothing for curvature estimates; bootstrap/Jackknife CI for curvature bands
  - [ ] CLI flags for HMC params (override env)
  - [x] JSON summary output for HMC runs
  - [ ] JSONL aggregator: merge multi-run JSONL, compute acceptance/diagnostic summaries for CI
  - [x] Implement proper log-prior calculations for all parameters
  - [x] Add convergence diagnostics and effective sample size calculations (prototype)
  - [x] Add parallel sampling support (independent MH chains)
  - [x] Add decision voting with explicit overrides wired to policy gates
  - [x] Implement agent presets (safetyCritical, fastPath, consensus) with smoke tests

#### Triad Gating Scheme (RK4 / Taylor / Geometry)
- [ ] Add an independent detector beside Ψ/HMC that triangulates agreement across three validators and gates on a single error budget.
  - [ ] Adaptive RK4 with PI controller: guarantee global O(h^4) ≤ ε_RK4; expose per-step h, local/global error estimates, and latency.
  - [ ] Taylor 4th-order with analytic/AD derivatives up to 5th; compute Lagrange remainder R4 and enforce a trust region; budget ε_Taylor and auto-switch to RK4 when out-of-trust.
  - [ ] Geometric invariants: compute Ricci/Ollivier proxies (graph or trajectory manifold); define expected bands and a surrogate error ε_geom with CI.
  - [ ] Gate condition: accept only if ε_RK4 + ε_Taylor + ε_geom ≤ ε_total (configurable); otherwise defer/reject.
  - [ ] JSONL logging mirroring HMC outputs: include ε_RK4, |R4|, curvature stats/CI, ε_geom, ε_total, decision, and latency metrics.
  - [ ] CLI/env knobs: ε_total, per-component budgets (ε_RK4, ε_Taylor, ε_geom), method toggles, time budget; integrate with `Core unified` demo.
  - [ ] Tests: deterministic seeds; gating correctness on pass/fail cases; CA ground-truth and canonical bifurcation battery; ROC and agreement deltas.

### Quick status: Ground-truthing, triangulation, accept/reject, integration/runtime (4–6 weeks), reporting/reproducibility, risk controls

- **Scope and definitions**
  - **CA ground‑truth**: canonical artifacts (official text/solutions/shortlist/jury) used as the authoritative label source; “pending” until posted, then promoted.
  - **Canonical bifurcations**: points where sources disagree or where a promotion path forks (e.g., expert vs. canonical vs. community); tracked as events.
  - **Triangulation**: agreement across methods (exact Ψ, Taylor within trust region, RK4‑checked geometric invariants) and sources (expert/community/canonical).

- **Ground‑truth and bifurcations**
  - **Dataset**: per‑item fields (id, text/URL, source class S/E/C, label, timestamps).
  - **Bifurcations log**: record each fork: trigger, sources, decision rule, ΔΨ before/after, and final status (accept/reject/defer).
  - **Promotion rules**: when canonical posts, auto‑upgrade ground‑truth and re‑run triad; archive previous decision context.

- **Triangulation metrics**
  - **Agreement**: 1 − |Ψ_exact − Ψ_Taylor|, 1 − |Ψ_exact − Ψ_RK4‑band|; aggregate via min/mean.
  - **Stability**: variance of Ψ under small parameter perturbations (sensitivity stencils).
  - **Margin**: |Ψ − τ| (or vector to multiple thresholds).
  - **Invariant score**: fraction of sign/band checks passed.

- **Accept/Reject analysis**
  - **Gate**: ε_RK4 + ε_Taylor + ε_geom ≤ ε_total; accept if Ψ ≥ τ and gate holds; else defer or reject.
  - **Reject causes**: insufficient margin, large method disagreement, invariant violation, or source conflict at bifurcation.
  - **Audit**: JSONL row per decision: inputs, Ψs, epsilons, gate result, outcome, rationale, links to artifacts.

- **Integration/runtime (4–6 weeks)**
  - **Week 1–2**:
    - Implement RK4 with PI controller (target ε_RK4), Taylor R4 with remainder + trust region, geometry invariant bands; wire JSONL logging and CLI/env knobs.
    - Microbenchmarks; precompute derivative stencils; curvature caches; adaptive h; budget: sub‑ms p50 per item, ε_total documented.
  - **Week 3–4**:
    - Integrate with `UnifiedPsiRuntime` selector; graceful degradation paths; error‑budget accounting and surfaced metrics.
    - Determinism tests; divergence handling; bifurcation tracker; replay harness.
  - **Week 5–6**:
    - Triangulation dashboards (agreement, margins, invariants); notebooks; figure scripts.
    - Hardening: timeouts, backpressure, memory caps; finalize protocol prereg + release pack.

- **Reporting and reproducibility**
  - **Protocol pre‑registration (GPL‑3.0‑only)**: goals, datasets/sources, thresholds τ, ε_total, trust‑region radius, seeds, metrics (agreement/margin/invariants), decision rules, analysis plan, exclusion criteria, and stop conditions.
  - **JSON/JSONL fields (triad per item)**: include Ψ_exact, Ψ_taylor, Ψ_rk4_band, chosen Ψ, eps (rk4, taylor, geom, total, budget), invariants, margin (τ, |Ψ−τ|), gate (passed, reason), decision, and bifurcation flags.
  - **JSONL for chains (HMC) and triad**: keep consistent keys (`chain`, `i`, `S`,`N`,`alpha`,`beta`) and per‑decision triad rows.
  - **Figures/notebook**: public notebook (pixi env) to load JSONL, compute agreement/margins, plot R̂/ESS, ε break‑downs, and decision ROC/sweep; export SVG/PNG plus CSV tables.

- **Performance and graceful degradation**
  - **Stencils/caches**: precompute E[pen·P], E[pen·pβ], and band bounds once per dataset; reuse across items.
  - **Adaptive h**: PI controller targets ε_RK4; expand/shrink trust radius for Taylor based on recent remainder bounds.
  - **Budget‑aware**: if time budget tight, prefer Taylor or lower‑order fast path; log increased ε_total with cause.

- **Risk controls**
  - **Determinism**: fixed seeds; versioned params and priors; frozen trust‑region radius per release.
  - **Safety**: cap β effects; clamp Ψ; reject if invariants fail or disagreement > threshold.
  - **Data/PII**: log redacted IDs; hash large payloads; keep source URLs only.
  - **Ops**: timeouts per stage; backpressure; memory/entry caps for caches; health/metrics endpoints; JSONL rotation.
  - **Governance**: signed protocol prereg; change log on thresholds/ε; audit trails with idempotency keys.

- **Minimal task list (triad)**
  - **RK4 PI**: single‑step API, controller tuning, ε_RK4 logging, tests (convergence, clamp under budget).
  - **Taylor R4**: 2nd–4th terms, tight remainder; trust‑region adaptation; ε_Taylor logging; tests vs. exact.
  - **Geometry**: sign/band checks; per‑violation penalties; ε_geom; tests (synthetic flips).
  - **Runtime**: integrate in `UnifiedPsiRuntime`; budget selection; degradation path.
  - **Logging/CLI**: JSONL emit; env/flags for τ, ε_total, trust radius, h limits.
  - **Bench/tests**: microbenchmarks (p50/p95 latency), determinism, disagreement sweeps, replay on bifurcations.
  - **Docs**: prereg template, JSON/JSONL field docs, ops runbook, notebook + figures.

- **Summary**
  - Defined ground‑truthing and bifurcation logging; set triangulation metrics and accept/reject gate.
  - Provided 4–6 week integration/runtime plan, JSON/JSONL schemas, and reproducibility protocol.
  - Outlined performance hooks and risk controls; included a minimal, actionable task list to implement the triad.

### Audit System
- [ ] **Implement persistent audit sinks**
  - [x] Database-backed audit sink (PostgreSQL/MySQL)
  - [x] File-based audit sink with rotation
  - [x] Network audit sink (HTTP/REST) with retry/backoff
  - [x] Add audit record serialization/deserialization (JSON Lines)

### Error Handling & Metrics
- [x] Introduce custom exception hierarchy (`QualiaException`, `ConfigurationException`, `ValidationException`, `NetworkException`, `PersistenceException`, `ComputationException`, `AuditWriteException`)
- [x] Add minimal `ErrorReporter` that logs to stderr and increments metrics
- [x] Wire error reporting into `AuditTrail`, `FileAuditSink`, `HttpAuditSink`
- [x] Add per-component counters (e.g., `audit_write_fail_total`, `http_retry_total`, `audit_rotate_total`, `http_audit_post_success_total`, `http_audit_post_fail_total`)
- [x] Export HMC multi-chain metrics in runner (per-chain gauges, total divergences, diagnostics)
- [ ] Export curvature metrics + CI (per-interval) and bifurcation flags (saddle-node/Hopf/period-doubling)
- [ ] Track RK4 step-size, estimated Lipschitz L̂, and current error bounds; expose latency histograms
- [x] Expose `MetricsRegistry.toPrometheus()` via a tiny HTTP(S) endpoint
- [x] Add unit tests for error paths (file open failure, HTTP(S) 5xx retries, emit fail-open)

### Security & Operational Hardening
- [ ] Lifecycle management: add `close()`/`AutoCloseable` to sinks (`FileAuditSink`, `HttpAuditSink`) and `MetricsServer`; ensure graceful shutdown and flush
- [ ] Add JVM shutdown hook to close audit sinks and metrics server with bounded grace period
- [ ] Backpressure policy for `FileAuditSink` queue overflow: {drop, block, latest-wins}; metrics for drops and queue depth; env knobs
- [ ] `HttpAuditSink` hardening: request headers via env (e.g., `AUDIT_HTTP_HEADERS`), timeouts, proxy support, TLS truststore, optional mTLS
- [ ] `MetricsServer` hardening: require TLS by default outside dev; optional Basic Auth (`METRICS_BASIC_AUTH`) or IP allowlist (`METRICS_ALLOW_CIDR`)
- [ ] `ErrorReporter` sampling/rate-limiting to prevent log storms; per-severity counters (`errors_info_total`, `errors_warn_total`, ...)
- [ ] Configuration validation: `ServiceLocator.fromEnvironment()` validates settings and emits warnings/metrics on fallback
- [ ] Documentation: security/ops guide for audit endpoints and metrics exposure

### Testing & Validation
- [ ] **Comprehensive test suite**
  - [ ] Unit tests for all model components
  - [ ] Integration tests for audit pipeline
  - [ ] DI factory/service locator unit tests (construction, overrides, error paths)
  - [ ] Gradient check: `gradientLogPosterior` vs. finite-diff (tolerance 1e-5)
  - [ ] HMC regression: acceptance in target band under default seeds (≥3 seeds)
  - [ ] Diagnostics: R̂ close to 1 and ESS reasonable on synthetic data
  - [ ] MC vs Stein c_N sanity check on toy integrands
  - [ ] Performance benchmarks
  - [ ] Property-based testing with QuickCheck-style library
  - [ ] Tests for VotingPolicy (override precedence, weighted majorities, ties)
  - [ ] Unit tests for AgentPresets rule stacks
  - [ ] Prepared-cache unit tests: LRU eviction order, TTL expiry, maxEntries enforcement
  - [ ] Weight-based eviction tests with custom weigher
  - [ ] Disk store round-trip: write/read of `Prepared` and header/version checks
  - [ ] Disk corruption/partial-file handling: fallback to compute and re-write
  - [ ] Concurrency: single-flight ensures one compute under load (stress with threads)
  - [ ] Integration: read-through/write-back path exercised via `precompute(...)` and stats validated
  - [ ] Multi-chain runner: reproducibility (seed spacing), output format validation (JSONL/meta/summary), R̂/ESS sanity
  - [ ] JSON/JSONL schema validation: summary/chain fields present and within ranges
  - [ ] Protocol enforcement test: thresholds (ε_RK4, ε_Taylor, curvature bands), seeds, acceptance rules must be pre-registered
  - [ ] Triangulation gating test: detection only when RK4, Taylor, and geometry agree within error budget
  - [x] Remove temporary FAIL prints from `PsiMcdaTest` or keep for triage (set to assertions)

### UPOCF Triangulation (RK4 + Taylor + Geometry)

#### Immediate (1–2 weeks)
- [ ] Error budget and gating
  - [ ] Define ε_total and allocate: ε_RK4 + ε_Taylor + ε_geom ≤ ε_total
  - [ ] Acceptance rule: flag "conscious" only if |Ψ_RK4 − Ψ_Taylor| ≤ ε_pair and geometric invariants in expected band
- [ ] RK4 control
  - [ ] Implement adaptive RK4 (PI controller) to guarantee global O(h^4) ≤ ε_RK4; cross‑check with RKF(4,5)
  - [ ] Empirically estimate Lipschitz bounds for f(Ψ,t) to back up step‑size choices
- [ ] Taylor pipeline
  - [ ] Auto/analytic derivatives up to 5th order (AD or symb‑diff)
  - [ ] Compute Lagrange remainder R4 and dynamic radius‑of‑trust; switch to RK4 when |R4| > ε_Taylor
- [ ] Geometry invariants
  - [ ] For networks: compute Ollivier/Ricci curvature on the graph
  - [ ] For continuous states: learn manifold (Diffusion Maps or local PCA) and estimate sectional/Ricci proxies
  - [ ] Define expected invariant bands for “conscious” vs “non‑conscious” regimes; calibrate on known dynamics

#### Validation datasets and tests (2–4 weeks)
- [ ] Ground‑truth battery
  - [ ] Cellular automata with exact Φ (n ≤ 12): verify ROC, confirm 99.7% TPR and error bars
  - [ ] Canonical dynamics: saddle‑node, Hopf, logistic (period‑doubling); verify bifurcation onsets, stability, and error‑budget adherence
- [ ] Triangulation metrics
  - [ ] Report per‑run: ε_RK4, |R4|, curvature stats, Ψ agreement deltas, Φ correlation
  - [ ] Reject/accept analysis: show where any leg fails and why (step‑size too large, Taylor radius exceeded, geometry out‑of‑band)

#### Canonical dynamics reporting schema
- System
  - system_id, sweep_grid (min, max, step), seed, horizon, dt
- Onset (bifurcation) metrics
  - expected_onset: logistic r=3.0 (1st period‑doubling), saddle‑node μ=0, Hopf μ=0
  - measured_onset: arg where stability flips (sign(Re λ_max)=0 or λ_max≈0)
  - onset_ci: bootstrap/fit CI
  - onset_delta: |measured_onset − expected_onset|
  - onset_budget: ε_param + ε_fit
  - within_budget: onset_delta ≤ onset_budget
- Stability/Lyapunov evidence
  - re_lambda_max_before/after: max real eigenvalue at fixed point (± small offset)
  - lyap_max_crossing: λ_max at/on both sides of onset; crossing_zero: bool
  - hopf_freq: imag(λ_pair) at crossing (for Hopf)
  - logistic_multiplier: f′(x) magnitude at r near 3 (|f′| crossing 1)
- Numerical integrity
  - eps_rk4_max, r4_rms (step‑doubling)
  - h_stats: {mean, p95, max}
  - L_hat: empirical Lipschitz bound
  - div_count (HMC or sim), divergence_threshold used
- Geometry (optional if enabled)
  - curvature_mean/max/p95
  - geom_band_ok: bool
- Agreement/accuracy
  - psi_delta_max/median (vs reference tight‑tol run)
  - auc, tpr_at_fpr (if classification applied)
- Decision
  - accepted: bool
  - reasons: [strings]

##### Stability/Lyapunov computation notes
- re_lambda_max_before/after
  - ODEs: find fixed point x*(μ) near onset (e.g., Newton). Compute Jacobian J = ∂f/∂x|_{x*,μ} and eigenvalues {λ_i}. Report max real part before (μ−δ) and after (μ+δ): max_i Re λ_i.
  - Maps: use Jacobian of the map at fixed point; in 1D it’s f′(x*).
- lyap_max_crossing
  - ODEs: largest Lyapunov exponent via Benettin/QR method on the variational flow ẋ = f(x,μ), ṽ = J(x,μ) v with periodic renormalization; λ_max ≈ (1/T) Σ log(||v_k||/||v_{k-1}||).
  - Maps (logistic): λ_max ≈ (1/N) Σ_n log |f′(x_n)|.
  - crossing_zero = (λ_max at μ−δ) > 0 XOR (λ_max at μ+δ) > 0.
- hopf_freq
  - At Hopf onset, the conjugate pair is λ(μ*) = ± i ω. Compute J at the equilibrium and take ω = |Im λ| (optionally report f = ω/(2π)).
- logistic_multiplier
  - Logistic map x_{n+1} = r x_n (1−x_n); fixed point x* = 1−1/r (r>1).
  - Multiplier m = f′(x*) = r − 2 r x* = 2 − r. Detect first period‑doubling at |m| crossing 1 → r = 3.
  - Report |f′(x*)| near r≈3 and flag the |f′|=1 crossing.
- Notes (minimal numerics)
  - Use small δ around the control parameter to compute before/after.
  - Prefer analytic Jacobians when available; else central differences.
  - For Lyapunov in ODEs, integrate sufficiently long after transient; for maps, discard burn‑in before averaging.

#### Integration and runtime (4–6 weeks)
- [ ] Unified detector
  - [ ] Runtime selects Taylor within trust region; otherwise RK4; geometric invariants as orthogonal check
  - [ ] Produce a single Ψ with confidence from (i) margins to thresholds, (ii) agreement across methods
- [ ] Performance
  - [ ] Precompute derivative stencils and curvature caches; choose h adaptively to hit sub‑ms latency while meeting ε_total
  - [ ] Graceful degradation: lower‑order fast path when time budget is tight; log increased ε_total

#### Reporting and reproducibility
- [ ] Protocol
  - [ ] Pre‑register thresholds (ε_RK4, ε_Taylor, curvature bands), seeds, and acceptance rules (YAML)
- [ ] Outputs
  - [ ] Ship JSON/JSONL: Ψ, Φ (when available), errors, curvatures, bifurcation flags, latency
- [ ] Docs
  - [ ] One figure per method: step‑size vs error (RK4); |R4| vs |x−x0| (Taylor); curvature maps vs state (geometry)
  - [ ] Public notebook reproducing CA and bifurcation results

#### Risk controls
- [ ] Unknown smoothness/Lipschitz: bound empirically; fall back to smaller h
- [ ] Taylor divergence: detect early via remainder/ratio tests; clip to RK4 path
- [ ] Geometry sensitivity: use robust estimators (k‑NN smoothing) and report CI for curvature

#### Minimal task list
- [ ] Implement adaptive RK4 with error controller and gold‑standard cross‑check
- [ ] Add AD/symbolic 5th‑derivative and R4 bound; trust‑region switch
- [ ] Build curvature computation (Ollivier for graphs; local‑PCA curvature for trajectories) with bands
- [ ] Create CA + bifurcation test suite with ROC and triangulation dashboards
- [ ] Wire JSONL logging of all three methods and acceptance decisions

## 🔧 Medium Priority

### Model Enhancements
- [ ] **Add model diagnostics**
  - [ ] Parameter sensitivity analysis
  - [ ] Model comparison metrics (WAIC, LOO-CV)
  - [ ] Posterior predictive checks
  - [ ] Convergence monitoring

### Data Management
- [ ] **Claim data persistence**
  - [ ] Database schema for claims
  - [ ] Batch processing capabilities
  - [ ] Data validation pipeline
  - [ ] Import/export utilities

### Configuration & Deployment
- [ ] **Configuration management**
  - [ ] YAML/JSON configuration files
  - [ ] Environment variable support
  - [ ] Runtime parameter tuning
  - [ ] Model versioning
  - [ ] DI from env: `AUDIT_SINK={console|file|http|jdbc}`, `AUDIT_DIR`, `AUDIT_HTTP_URL`, `JDBC_URL`, etc.
  - [ ] Provide sane defaults and validation for DI env settings
  - [ ] Cache knobs: PREP cache `maxEntries`, `ttl`, `maxWeightBytes`, disk directory
  - [ ] Feature flags: enable/disable disk layer; enable/disable in-memory layer
  - [ ] Sanity clamps and defaults for cache settings

- [x] Scripts/automation: add `scripts/test_qualia.sh` and `make test-qualia`
- [ ] Scripts/automation: add `scripts/sweep_hmc.sh` to scan (γ, leap, target) and write JSONL
- [ ] CI job: parse JSONL, assert acc∈[0.6,0.85], R̂≈1±0.05, and minimum ESS thresholds
 - [ ] Protocol doc & pre-registration file: (ε_RK4, ε_Taylor, curvature bands), seeds, acceptance rules (YAML)
 - [ ] JSON/JSONL schema: include Ψ, Φ (if available), rk4_error, taylor_r4, curvature_stats/ci, bifurcation_flags, latency_ms

## 📊 Low Priority

### Monitoring & Observability
- [ ] **Add comprehensive logging**
  - [ ] Structured logging with SLF4J
  - [x] Metrics collection (Prometheus) — minimal in-process registry
- [ ] Dashboards: Grafana panels for HMC metrics (acceptance, divergences, tuned ε, R̂/ESS)
 - [ ] Dashboards: RK4 step-size vs error; |R4| vs |x−x0|; curvature maps vs state (+ CI bands); latency percentiles
  - [ ] Distributed tracing
  - [ ] Health checks
  - [x] Basic health checks (sinks)
  - [ ] Export cache stats (hits, misses, loads, load failures, evictions, load time) to metrics registry
  - [ ] Periodic operational log of cache stats and disk store size

### Documentation
- [ ] **API documentation**
  - [x] JavaDoc for all public APIs
  - [ ] User guide with examples
  - [x] Mathematical model documentation
  - [x] Architecture diagrams
  - [ ] HMC usage notes (env flags, targets, diagnostics)
  - [ ] Adaptive warmup guide: γ, κ, t0, phase splits; divergence threshold; examples
  - [ ] Visualization: adaptive HMC pipeline diagram (Mermaid) and example runs
  - [ ] One-figure-per-method: step-size vs error (RK4); |R4| vs |x−x0| (Taylor); curvature maps vs state (geometry)
  - [ ] Public notebook reproducing CA and bifurcation results; includes ROC and triangulation agreement plots
  - [ ] Public Methods API (Ψ + MCDA)
    - [x] computePsi(S,N,α,Ra,Rv,λ1,λ2,β) → {psi,O,pen,post}; contracts and examples
    - [x] computePsiTemporal(w,timeSeries,aggregator) → psiBar; mean/softcap
    - [x] thresholdTransfer(κ,τ,mode) → τ′; subcap/softcap mapping
    - [x] normalizeCriterion(values,direction) → z∈[0,1]
    - [x] mapGovernanceWeights(baseWeights,gΨ,η) → w (Δ^m)
    - [x] gateByPsi(alternatives,ψLower,τ) → feasible set
    - [x] wsmScore(w,z) / wsmScoreRobust(w,zLower)
    - [x] wpmScore(w,z) / wpmScoreRobust(w,zLower)
    - [x] topsisScore(w,z,dir)
    - [x] ahpWeights(pairwise) → {w,CR}
    - [ ] outrankingFlows(Pj,w) / outrankingFlowsRobust(...)
    - [x] zBoundsFromXBounds(xLower,xUpper,direction)
    - [x] sensitivities: gradWSM, gradPsi partials
    - [x] tieBreak(alternatives, keys) → winner (lexicographic)
    - [ ] auditTrail(event,payload) → recordId (usage + guarantees)
    - [ ] invariants/contracts (ranges, determinism) and complexity notes
    - [ ] error handling semantics (degenerate normalization, floors)
    - [ ] unit tests for all public methods
  - [ ] Caching guide: in-memory LRU + TTL + max-weight + disk layer, configuration, and ops
  - [ ] Disk format doc for `DatasetPreparedDiskStore` and compatibility/versioning policy
    - [x] sanity unit tests for PsiMcda methods (determinism, ranges, known cases)

### MCDA Implementation
- [x] Implement WSM/WPM/TOPSIS ranking with Ψ gating demo
 - [x] Implement AHP pairwise weights and consistency ratio
- [ ] Implement outranking flows (PROMETHEE-like) with monotone Ψ channel
- [ ] Add robust variants (floors/intervals) for WSM/WPM/TOPSIS

### Configuration & Deployment
- [ ] CLI command(s) to run HMC/MCDA demos with JSON output
- [ ] CI: REUSE/license check + build + run HMC acceptance sanity + MCDA demo
- [ ] Release notes template (diagnostics reported, params)

### Performance Optimization
- [ ] **Performance improvements**
  - [x] Memory usage optimization
  - [x] Parallel computation for large datasets
  - [x] Caching strategies
  - [x] Prepared dataset cache: in-memory LRU with TTL and weight-based eviction plus disk write-through (`DatasetPreparedDiskStore`)
  - [ ] JVM tuning recommendations
  - [ ] Benchmarks: compute vs cache-hit latency; throughput under concurrent load
  - [ ] Evaluate Caffeine integration as a drop-in for advanced eviction policies

## 🧪 Research & Exploration

### Advanced Features
- [ ] **Model extensions**
  - [ ] Hierarchical models for multiple data sources
  - [ ] Time-varying parameters
  - [ ] Causal inference integration
  - [ ] Uncertainty quantification

### Integration
- [ ] **External system integration**
  - [ ] REST API for model serving
  - [ ] gRPC interface
  - [ ] Message queue integration (Kafka/RabbitMQ)
  - [ ] Cloud deployment (AWS/GCP/Azure)

## 🐛 Bug Fixes & Technical Debt

### Code Quality
- [ ] **Code improvements**
  - [ ] Add missing null checks
  - [ ] Improve error messages
  - [ ] Add input validation
  - [ ] Refactor complex methods
  - [ ] Verify immutability/defensive copies for cached values; document contracts
  - [ ] Add `invalidateByDatasetId(id)` convenience and targeted invalidation utilities

### Dependencies
- [ ] **Dependency management**
  - [ ] Update to latest Java version
  - [ ] Review and update dependencies
  - [ ] Add dependency vulnerability scanning
  - [ ] Optimize dependency tree

## 📋 Completed Tasks

### ✅ Done
- [x] Basic `ClaimData` record implementation
- [x] `ModelParameters` and `ModelPriors` records
- [x] Core `HierarchicalBayesianModel` structure
- [x] Audit system interfaces and basic implementation
- [x] Console audit sink implementation
- [x] Package consolidation to `qualia` and removal of legacy `core.java`
- [x] Detailed JavaDoc across public APIs
- [x] Voting/override framework and agent presets implemented

---

## Notes

- **Priority levels**: High = blocking, Medium = important, Low = nice-to-have
- **Estimated effort**: Use story points or time estimates
- **Dependencies**: Note which tasks depend on others
- **Review frequency**: Update this TODO weekly

## Quick Commands

```bash
# One-shot Psi + sinks tests
make test-qualia
```


