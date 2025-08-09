Hybrid AI-Physics UQ (minimal stubs)

Contents
- Physics stub `S(x)` with vorticity/divergence diagnostics
- Neural residual `N(x)` with heteroscedastic head
- Hybridization `O(α) = α S + (1−α) N`
- Penalties `R_cog, R_eff` and governance `pen = exp(-[λ1 R_cog + λ2 R_eff])`
- Calibrated posterior `post = min{β P(H|E), 1}`
- Confidence `Ψ(x) = O · pen · post`
- Objective: `MSE + NLL + penalties`
- α scheduler and split conformal calibrator

Files
- `core.py`: all PyTorch components
- `__init__.py`: public API
- `example_numeric_check.py`: quick float-only check of the numerical example

Quick start
1) Optional: install PyTorch to instantiate the model
   - `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU wheels)
2) Run the numeric check (no dependencies):
   - `python -m hybrid_uq.example_numeric_check`

Notes
- Replace the identity mapping in `PhysicsInterpolator` with your domain-specific surface→sigma operator (metric terms, masks).
- Decoder and Eq. B4 can be added similarly; keep residual scaling small (e.g., 0.02·σ_surface) to mitigate shocks.
- Use a small ensemble or SWAG/Laplace to populate `pred_var` for the α scheduler.