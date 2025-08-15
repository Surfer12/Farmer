# Mojo-based CFD (2D prototype) for Vector 3/2 Blackstix+ Fin

This folder contains a minimal, runnable 2D incompressible Navier–Stokes (projection) solver tailored to a fin-like obstacle, plus plotting utilities. It is written in Python for immediate execution, with a Mojo stub to migrate core loops later.

## Quick start (Python)

```bash
# 1) Run solver (laminar, projection method)
MPLBACKEND=Agg python3 /workspace/cfd_mojo/solver_python.py \
  --nx 300 --ny 150 --u 2.0 --aoa 10.0 --nu 1e-6 \
  --steps 1200 --save_every 300 --out /workspace/cfd_mojo/out \
  --chord 0.28 --thickness 0.06 --camber 0.02 --fin_angle 6.5

# 2) Plot latest state
MPLBACKEND=Agg python3 /workspace/cfd_mojo/plot_results.py \
  --in_dir /workspace/cfd_mojo/out \
  --out_png /workspace/cfd_mojo/pressure_map.png
```

Outputs:
- `out/state_XXXXXX.npz`: compressed arrays `u, v, p, mask`
- `pressure_map.png`: pressure heatmap, speed magnitude, and velocity vectors

## Notes
- Grid: collocated, central differences, explicit advection + diffusion, Jacobi pressure solve.
- Fin geometry: rasterized cambered airfoil (NACA-like) with rotation to emulate 3/2 foil and 6.5° cant. Adjust `--camber`, `--thickness`, `--fin_angle`.
- Boundary conditions: uniform inflow with AoA at left, Neumann outflow at right, zero-gradient top/bottom, no-slip on fin mask.
- This is a laminar baseline. Turbulence (e.g., k-ω SST) can be added via an eddy-viscosity term.

## Mojo migration
See `solver.mojo` for a skeleton mirroring the Python algorithm (advection-diffusion, pressure projection, Jacobi Poisson). Replace inner loops with Mojo `@parallel` constructs and vectorized math once Mojo is available in your environment.