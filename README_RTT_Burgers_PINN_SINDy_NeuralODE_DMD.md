# Reynolds Transport Theorem + Cauchy Momentum (1D Burgers proxy) with PINN, SINDy, Neural ODE, and DMD

This project provides a single Python script implementing:

- PINN solving inviscid Burgers' equation u_t + u u_x = 0 (RTT/conservation form) with periodic BCs and IC u(x,0) = -sin(pi x).
- RK4 finite-difference solver for validation.
- SINDy sparse regression of dynamics from simulated snapshots.
- Neural ODE that learns coarse dynamics, using RK4 integration.
- DMD mode decomposition and reconstruction error as a Koopman-like analysis.

Requirements are already in `requirements.txt` (PyTorch, NumPy, SciPy, etc.).

## How to run

```bash
python3 scripts/rtt_burgers_ml.py
```

The script will:
- Simulate Burgers with a periodic finite-difference scheme and RK4 time stepping (ground-truth surrogate).
- Train a PINN with RTT residual (conservative form) and periodic BC consistency.
- Fit SINDy on the simulated data and print discovered sparse coefficients.
- Train a small Neural ODE on a downsampled spatial grid.
- Run DMD, printing relative reconstruction error.

## Notes
- Viscosity `nu` is set to 0.0 (inviscid). You can set `nu > 0` in `main()` to include the viscous term (u_t + u u_x - nu u_xx = 0).
- The PINN uses autograd for u_x and u_t; viscous mode also computes u_xx.
- RK4 is used consistently for time integration to align with the validation request.
- This 1D proxy captures the convective nonlinearity of Cauchy momentum; RTT provides the conservation form used in both the solver and the PINN residual.