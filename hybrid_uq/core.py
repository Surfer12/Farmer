import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- finite differences and diagnostics --------------------
def central_diff(x: torch.Tensor, dx: float, dim: int) -> torch.Tensor:
    """Central difference along dimension dim with periodic roll.

    Args:
        x: Tensor of shape [...]
        dx: Grid spacing along dim
        dim: Dimension to differentiate along
    Returns:
        Tensor of same shape as x
    """
    return (x.roll(shifts=-1, dims=dim) - x.roll(shifts=1, dims=dim)) / (2.0 * dx)


def vorticity_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D vorticity and divergence on an (H, W) grid for batched fields.

    Args:
        u, v: [B, 1, H, W]
        dx, dy: grid spacings
    Returns:
        (zeta, div): both [B, 1, H, W]
    """
    du_dx = central_diff(u, dx, dim=-2)
    du_dy = central_diff(u, dy, dim=-1)
    dv_dx = central_diff(v, dx, dim=-2)
    dv_dy = central_diff(v, dy, dim=-1)
    zeta = dv_dx - du_dy
    div = du_dx + dv_dy
    return zeta, div


# -------------------- physics interpolation S(x) --------------------
class PhysicsInterpolator(nn.Module):
    """Placeholder physics mapping from surface plane to sigma/vector space.

    Replace the identity mapping with a domain-specific linear operator
    that encodes metric terms, masks, and sigma-level transforms.
    """

    def __init__(self, grid_metrics: Dict[str, float]):
        super().__init__()
        self.grid_metrics = grid_metrics  # expects keys: 'dx', 'dy' (floats)

    def forward(self, x_surface: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the physics-based interpolation and diagnostics.

        Args:
            x_surface: [B, C, H, W] surface-plane fields; channels 0,1 treated as u,v
        Returns:
            S: [B, C, H, W] sigma-space fields (placeholder = identity)
            diagnostics: dict with vorticity and divergence [B,1,H,W]
        """
        S = x_surface  # identity placeholder
        u = S[:, 0:1]
        v = S[:, 1:2]
        dx = float(self.grid_metrics.get("dx", 1.0))
        dy = float(self.grid_metrics.get("dy", 1.0))
        zeta, div = vorticity_divergence(u, v, dx, dy)
        return S, {"vorticity": zeta, "divergence": div}


# -------------------- neural correction N(x) --------------------
class ResidualNet(nn.Module):
    """Small CNN producing residual mean and heteroscedastic log_sigma."""

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_mu = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        self.conv_log_sigma = nn.Conv2d(hidden, out_ch, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(feats))
        h = F.relu(self.conv2(h))
        mu = self.conv_mu(h)
        log_sigma = self.conv_log_sigma(h).clamp(min=-6.0, max=3.0)
        sigma = torch.exp(log_sigma)
        return mu, sigma, log_sigma


# -------------------- hybrid model with penalties and Ψ --------------------
class HybridModel(nn.Module):
    def __init__(
        self,
        grid_metrics: Dict[str, float],
        in_ch: int,
        out_ch: int,
        residual_scale: float = 0.02,
        init_alpha: float = 0.5,
        lambda_cog: float = 0.5,
        lambda_eff: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.phys = PhysicsInterpolator(grid_metrics)
        self.nn = ResidualNet(in_ch=in_ch, out_ch=out_ch)
        self.residual_scale = residual_scale
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))  # global scalar alpha
        # governance/penalty weights managed externally by scheduler
        self.lambdas: Dict[str, float] = {"cog": float(lambda_cog), "eff": float(lambda_eff)}
        self.register_buffer("beta", torch.tensor(float(beta)), persistent=True)

    def compute_penalties(self, diagnostics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        zeta = diagnostics["vorticity"]
        div = diagnostics["divergence"]
        # Cognitive: penalize divergence and encourage rotational smoothness
        r_div = (div ** 2).mean()
        zeta_dx = central_diff(zeta, 1.0, dim=-2)
        zeta_dy = central_diff(zeta, 1.0, dim=-1)
        r_rot_smooth = (zeta_dx ** 2 + zeta_dy ** 2).mean()
        R_cog = r_div + 0.1 * r_rot_smooth
        # Efficiency: surrogate anti-oscillation via gradient magnitude of zeta
        R_eff = r_rot_smooth
        return R_cog, R_eff

    def forward(
        self,
        x: torch.Tensor,
        external_P: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing S, N, O, penalties, pen, post, and Ψ.

        Args:
            x: [B, C, H, W] inputs
            external_P: optional tensor scalar or [B,1,1,1] with P(H|E)
        Returns:
            Dict of tensors including S, N, O, psi, sigma_res, R_cog, R_eff, pen, post
        """
        S, diagnostics = self.phys(x)
        mu_res, sigma_res, log_sigma = self.nn(x)
        N = S + self.residual_scale * mu_res
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        O = alpha * S + (1.0 - alpha) * N

        R_cog, R_eff = self.compute_penalties(diagnostics)
        pen = torch.exp(- (self.lambdas["cog"] * R_cog + self.lambdas["eff"] * R_eff))

        if external_P is None:
            external_P = torch.as_tensor(0.5, device=O.device)
        if not torch.is_tensor(external_P):
            external_P = torch.as_tensor(float(external_P), device=O.device)
        post = torch.clamp(self.beta * external_P, max=1.0)

        psi = O * pen * post
        return {
            "S": S,
            "N": N,
            "O": O,
            "psi": psi,
            "sigma_res": sigma_res,
            "log_sigma": log_sigma,
            "R_cog": R_cog,
            "R_eff": R_eff,
            "pen": pen,
            "post": post,
        }


# -------------------- objective (MSE + heteroscedastic NLL + penalties) --------------------
def loss_objective(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    w_rec: float = 1.0,
    w_nll: float = 1.0,
    w_cog: float = 1.0,
    w_eff: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    O = outputs["O"]
    sigma = outputs["sigma_res"]

    mse_grid = F.mse_loss(O, target)
    # Gaussian NLL per element
    var = sigma.pow(2).clamp_min(1e-8)
    nll = 0.5 * (torch.log(2.0 * math.pi * var) + (target - O).pow(2) / var)
    nll = nll.mean()

    loss = (
        w_rec * mse_grid
        + w_nll * nll
        + w_cog * outputs["R_cog"]
        + w_eff * outputs["R_eff"]
    )
    metrics = {
        "mse": mse_grid.detach(),
        "nll": nll.detach(),
        "R_cog": outputs["R_cog"].detach(),
        "R_eff": outputs["R_eff"].detach(),
        "pen": outputs["pen"].detach(),
        "post": outputs["post"].detach(),
    }
    return loss, metrics


# -------------------- α scheduler (variance- and stability-aware) --------------------
class AlphaScheduler:
    def __init__(
        self,
        alpha_min: float = 0.1,
        alpha_max: float = 0.95,
        var_hi: float = 0.02,
        var_lo: float = 0.005,
        k_up: float = 0.15,
        k_dn: float = 0.08,
    ) -> None:
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.var_hi = float(var_hi)
        self.var_lo = float(var_lo)
        self.k_up = float(k_up)
        self.k_dn = float(k_dn)

    @torch.no_grad()
    def step(
        self,
        model: HybridModel,
        pred_var: torch.Tensor,
        resid_stability: float,
        bifurcation_flag: bool = False,
    ) -> bool:
        """Update alpha and tighten/relax lambdas; return abstain flag."""
        alpha = model.alpha.clamp(self.alpha_min, self.alpha_max)
        var_mean = float(pred_var.mean().item()) if pred_var.numel() > 0 else 0.0

        risky = bifurcation_flag or (var_mean > self.var_hi) or (resid_stability < 0.0)
        if risky:
            alpha = torch.clamp(alpha + self.k_up * (self.alpha_max - alpha), self.alpha_min, self.alpha_max)
        elif (var_mean < self.var_lo) and (resid_stability > 0.5):
            alpha = torch.clamp(alpha - self.k_dn * (alpha - self.alpha_min), self.alpha_min, self.alpha_max)
        model.alpha.copy_(alpha)

        # tighten/relax penalties
        if risky:
            model.lambdas["cog"] = min(1.5, model.lambdas["cog"] * 1.10)
            model.lambdas["eff"] = min(1.5, model.lambdas["eff"] * 1.10)
        else:
            model.lambdas["cog"] = max(0.2, model.lambdas["cog"] * 0.98)
            model.lambdas["eff"] = max(0.2, model.lambdas["eff"] * 0.98)

        abstain = bifurcation_flag or (var_mean > 3.0 * self.var_hi)
        return bool(abstain)


# -------------------- simple split conformal calibration --------------------
class SplitConformal:
    def __init__(self, quantile: float = 0.9) -> None:
        self.quantile = float(quantile)
        self.q: Optional[float] = None

    def fit(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        residuals = torch.abs(targets - preds).reshape(-1).detach().cpu().numpy()
        self.q = float(torch.tensor(residuals).quantile(self.quantile).item())

    def intervals(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.q is not None, "Call fit() first"
        q = torch.tensor(self.q, device=preds.device, dtype=preds.dtype)
        return preds - q, preds + q