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


