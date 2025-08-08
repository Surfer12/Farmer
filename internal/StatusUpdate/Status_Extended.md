SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

# Status_Extended (internal)

Classification: Confidential — Internal Use Only

This file collects extended session logs and narrative notes that are too long for the one-line JSONL status entries. The canonical, append-only log remains `internal/StatusUpdate/status.jsonl`.

## Session log (example)

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

```bash
cd /Users/ryan_david_oates/Farmer && javac -d out-qualia Corpus/qualia/*.java | cat
```

```bash
cd /Users/ryan_david_oates/Farmer && java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.01 horizon=2000 burnin=1000 seed=42 out=bifurc-logistic.jsonl | cat
```

Ran tool

```bash
cd /Users/ryan_david_oates/Farmer && javac -d out-qualia Corpus/qualia/*.java | cat
```

```bash
cd /Users/ryan_david_oates/Farmer && java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02 horizon=1000 burnin=500 seed=42 out=bifurc-logistic.jsonl | cat
```

#### Session log (example – numerical guidance insertion)

Adding the numerical guidance section to `internal/qualia/TODO.md` under the canonical dynamics notes, then saving.

Ran tool

Made a quick plan: insert a new “Numerical guidance (δ sweep, Jacobian choice, burn‑in/transients)” subsection under the canonical dynamics notes in `internal/qualia/TODO.md`. Applying the edit now.

Ran tool

Ran tool

Added the requested numerical guidance block to `internal/qualia/TODO.md` under the canonical dynamics section, covering δ sweep, Jacobian choice, burn‑in/transients, stability checks, and integrator hygiene. No lint issues.

- Inserted subsection: “Numerical guidance (δ sweep, Jacobian choice, burn‑in/transients)”
- Content includes:
  - δ sweep defaults and robustness check
  - Jacobian step sizing and sparsity notes
  - Burn‑in rules for ODEs/maps and Lyapunov (Benettin/QR)
  - Cross‑validation of onset via eigenvalues and Lyapunov
  - RK4 step‑doubling hygiene and h_stats stability check

### Status

- Implemented canonical bifurcation tasks and CA-grounded triangulation wiring alongside the triad detector.

### What I added

- `UnifiedDetector`:
  - Triad gating step `triadStep(...)`: Adaptive RK4 with PI controller (ε_RK4), Taylor 4th remainder proxy and trust region (ε_Taylor), geometric invariant drift surrogate (ε_geom), plus gate acceptance and ψ.
- `Core`:
  - New subcommand `bifurc`:
    - `kind=logistic|saddle|hopf`
    - Outputs JSONL with schema-aligned fields (onset, Lyapunov, Jacobian real part, RK4/Taylor error metrics).
  - Updated `unified` to write triad JSONL per step with ε components and accept/reject.
- `BifurcationSweep`:
  - `runLogistic(...)`: sweeps r, computes Lyapunov exponent and average multiplier, writes JSONL; summary includes measured onset vs expected 3.0.
  - `runSaddleNode(...)`: sweeps μ, fixed-point eigenvalue proxy, runs triad for numerical integrity (ε_rk4_max, r4_rms, h stats) and writes JSONL; summary at μ=0.
  - `runHopf(...)`: sweeps μ with ω, reports real part of eigenvalue and triad integrity; writes JSONL; summary at μ=0.
  - Safe output dir handling (no NPE if writing to cwd).

### How to run

- Logistic (CA-style map):

```bash
java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02 horizon=1000 burnin=500 seed=42 out=bifurc-logistic.jsonl
```

- Saddle-node (ODE):

```bash
java -cp out-qualia qualia.Core bifurc kind=saddle muMin=-0.5 muMax=0.5 muStep=0.01 steps=500 h=1e-3 out=bifurc-saddle.jsonl
```

- Hopf (ODE):

```bash
java -cp out-qualia qualia.Core bifurc kind=hopf muMin=-0.2 muMax=0.2 muStep=0.01 omega=1.0 steps=500 h=1e-3 out=bifurc-hopf.jsonl
```

- Triad unified stepper (harmonic oscillator, JSONL per step):

```bash
java -cp out-qualia qualia.Core unified triad=true eps=1e-4 epsRk4=1e-5 epsTaylor=1e-5 epsGeom=1e-5 omega=6.28318 jsonl=unified.jsonl
```

- Added triad gating (RK4 PI, Taylor remainder, geometry) with accept/reject and ψ.
- Added `bifurc` CLI to sweep logistic, saddle-node, Hopf, writing schema-aligned JSONL and summaries.
- Verified build and ran a logistic sweep; output file written.

Ran tool

```bash
cd /Users/ryan_david_oates/Farmer && javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java') 2>&1 | cat
```

Quick status: Implemented the next step by adding a minimal triad gate and scaffolds; compiles cleanly.

- New runtime pieces
  - `qualia/TriadGate.java`: combines RK4 band (step-doubling error as ε_RK4), Taylor trust-region estimate with remainder (ε_Taylor), and geometry invariants (ε_geom) into a single gate; outputs chosen Ψ, epsilons, gate status, margin, and reasons.
  - `qualia/TaylorR4Estimator.java`: trust-region Taylor estimator scaffold (first-order now) with placeholder R4 remainder bound.
  - `qualia/RK4Controller.java`: RK4 integrator with a PI controller and built-in step-doubling error estimate.
  - `qualia/GeometryInvariants.java`: sensitivity-sign checks and a small penalty when violated.
  - `qualia/PsiSensitivity.java`: mean Ψ and partials used by Taylor and geometry modules.

- Build
  - Compiled all sources; no errors.

Summary
- Triad gating core is in place and ready to wire to JSONL logging/CLI per TODO.
- Next: tighten R4 bound and add proper RK4 target function/context, JSONL emit for triad results, and CLI/env knobs for τ, ε_total, trust radius, and time budgets.

### Additional example note

Creating an internal notation/glossary to standardize terms and JSON/metrics keys across UnifiedDetector and HMC, marked with the internal-use SPDX and avoiding CC0 as you prefer [[memory:5595133]].

Ran tool
Ran tool

- eps_geom is zero when invariants are within tolerance by definition. It measures violation, not raw drift. If you want to log raw drift too, add a parallel key like geom_drift.

- I added a canonical notation/glossary to standardize these terms and JSON keys across the codebase at internal/NOTATION.md with internal-use SPDX. It defines eps_geom, geom_drift, eps_rk4, eps_taylor, psi, acceptance, HMC keys, units, and naming rules.

- If you want emitters to include geom_drift alongside eps_geom in UnifiedDetector.triadStep, I can wire that next.

- Changes
  - Added internal/NOTATION.md documenting:
    - eps_geom: sum(max(0, drift − tol)) over invariants
    - geom_drift: sum(|value − reference|)
    - JSON/metrics keys for UnifiedDetector and HMC
    - Required naming/units rules and examples

SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

### Component: Bifurcations & Triad Detector
Status: done

Summary
- Integrated triad gating (Adaptive RK4 with PI controller, Taylor 4th remainder proxy with trust region, geometric invariant drift surrogate) and added CLI sweeps for logistic (CA), saddle-node, and Hopf bifurcations. Emitting JSONL with schema-aligned fields; verified logistic run.

Changes
- UnifiedDetector: added `triadStep(...)` producing ε_RK4, ε_Taylor, ε_geom, ψ, accepted flag
- Core: new `bifurc` subcommand (`kind=logistic|saddle|hopf`) and updated `unified` to log triad fields
- BifurcationSweep: implemented `runLogistic(...)`, `runSaddleNode(...)`, `runHopf(...)` with integrity metrics (e.g., ε_rk4_max, r4_rms, h stats) and onset summaries

How to run
- Logistic (CA):
  `java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02 horizon=1000 burnin=500 seed=42 out=bifurc-logistic.jsonl`
- Saddle-node (ODE):
  `java -cp out-qualia qualia.Core bifurc kind=saddle muMin=-0.5 muMax=0.5 muStep=0.01 steps=500 h=1e-3 out=bifurc-saddle.jsonl`
- Hopf (ODE):
  `java -cp out-qualia qualia.Core bifurc kind=hopf muMin=-0.2 muMax=0.2 muStep=0.01 omega=1.0 steps=500 h=1e-3 out=bifurc-hopf.jsonl`
- Triad unified stepper (oscillator):
  `java -cp out-qualia qualia.Core unified triad=true eps=1e-4 epsRk4=1e-5 epsTaylor=1e-5 epsGeom=1e-5 omega=6.28318 jsonl=unified.jsonl`

Next
- Add JSONL schema checks for triad fields; emit optional per-invariant breakdown
- Validate CA exact Φ datasets and produce ROC; estimate onset CIs for Hopf and logistic

Refs
- `Corpus/qualia/UnifiedDetector.java`
- `Corpus/qualia/BifurcationSweep.java`
- `Corpus/qualia/Core.java`
- `internal/qualia/TODO.md`


