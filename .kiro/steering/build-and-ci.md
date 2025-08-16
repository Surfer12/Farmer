---
inclusion: always
---
SPDX-License-Identifier: LicenseRef-Internal-Use-Only

## Build & Run (Java 21)

- Compile (Java):
  ```bash
  javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')
  ```
- Run help:
  ```bash
  java -cp out-qualia qualia.Core help | cat
  ```
- HMC (adaptive warmup):
  ```bash
  java -cp out-qualia qualia.Core hmc_adapt chains=2 warmup=500 iters=1000 thin=2 seed=42 out=hmc.jsonl | cat
  ```
- Bifurcation sweeps (logistic):
  ```bash
  java -cp out-qualia qualia.Core bifurc kind=logistic rMin=2.8 rMax=3.2 rStep=0.02 horizon=1000 burnin=500 seed=42 out=bifurc-logistic.jsonl | cat
  ```

- Metrics server (optional): set `METRICS_ENABLE=1` before running to expose `/metrics`.

