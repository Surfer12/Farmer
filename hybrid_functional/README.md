# Hybrid Symbolic–Neural Accuracy Functional

This module implements a bounded, normalized accuracy functional Ψ(x) that blends symbolic accuracy `S(x,t)` and neural accuracy `N(x,t)` with an adaptive weight `α(t)`, regularization penalties, and calibrated probability of correctness.

## Install

No external dependencies beyond NumPy.

## Usage

- As a library:

```python
from hybrid_functional.core import compute_psi, apply_beta_bias
import numpy as np

S = np.array([0.65])
N = np.array([0.85])
alpha = np.array([0.3])
Rcog = np.array([0.20])
Reff = np.array([0.15])
lam1 = 0.75
lam2 = 0.25
P_base = np.array([0.81])
P_corr = apply_beta_bias(P_base, beta=1.2)
psi = compute_psi(S, N, alpha, Rcog, Reff, lam1, lam2, P_corr)
print(psi)
```

- From CLI examples:

```bash
python -m hybrid_functional.cli --which runtime
python -m hybrid_functional.cli --which open_source
```