#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

# Try to import torch; make optional
try:
	import torch
	from torch import nn
	from torch.nn import functional as F
	HAS_TORCH = True
except Exception:
	HAS_TORCH = False
	nn = None
	F = None

def set_seed(seed: int = 42) -> None:
	if HAS_TORCH:
		torch.manual_seed(seed)
	np.random.seed(seed)


def device() -> str:
	if HAS_TORCH and torch.cuda.is_available():
		return "cuda"
	return "cpu"


# -------------------------------
# Burgers PDE helpers (RTT form)
# -------------------------------

def burgers_flux(u: np.ndarray) -> np.ndarray:
	"""Conservative flux F(u) = u^2 / 2 for inviscid Burgers."""
	return 0.5 * u * u


def np_periodic_diff_1d(u: np.ndarray, dx: float) -> np.ndarray:
	"""First derivative using second-order central differences with periodic BCs."""
	u_left = np.roll(u, 1, axis=-1)
	u_right = np.roll(u, -1, axis=-1)
	return (u_right - u_left) / (2.0 * dx)


def np_periodic_diff2_1d(u: np.ndarray, dx: float) -> np.ndarray:
	"""Second derivative using second-order central differences with periodic BCs."""
	u_left = np.roll(u, 1, axis=-1)
	u_right = np.roll(u, -1, axis=-1)
	return (u_left - 2.0 * u + u_right) / (dx * dx)


# -------------------------------
# RK4 time stepper (NumPy)
# -------------------------------

def rk4_step_np(f: Callable[[np.ndarray, float], np.ndarray], y: np.ndarray, t: float, dt: float) -> np.ndarray:
	k1 = f(y, t)
	k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
	k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
	k4 = f(y + dt * k3, t + dt)
	return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# -------------------------------
# Optional: RK4 for torch tensors
# -------------------------------
if HAS_TORCH:
	def rk4_step(f: Callable[["torch.Tensor", float], "torch.Tensor"], y: "torch.Tensor", t: float, dt: float) -> "torch.Tensor":
		k1 = f(y, t)
		k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
		k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
		k4 = f(y + dt * k3, t + dt)
		return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# -------------------------------
# PINN for Burgers (optional)
# -------------------------------

if HAS_TORCH:
	class MLP(nn.Module):
		def __init__(self, in_dim: int, hidden: int = 64, depth: int = 4, out_dim: int = 1) -> None:
			super().__init__()
			layers = []
			dim = in_dim
			for _ in range(depth):
				layers.append(nn.Linear(dim, hidden))
				layers.append(nn.Tanh())
				dim = hidden
			layers.append(nn.Linear(dim, out_dim))
			self.net = nn.Sequential(*layers)

		def forward(self, x: "torch.Tensor") -> "torch.Tensor":
			return self.net(x)

@dataclass
class PINNConfig:
	nu: float = 0.0
	x_low: float = -1.0
	x_high: float = 1.0
	t_low: float = 0.0
	t_high: float = 1.0
	n_collocation: int = 4096
	n_ic: int = 256
	n_bc: int = 256
	epochs: int = 4000
	lr: float = 1e-3
	hidden: int = 64
	depth: int = 3

if HAS_TORCH:
	class BurgersPINN(nn.Module):
		def __init__(self, cfg: PINNConfig) -> None:
			super().__init__()
			self.cfg = cfg
			self.model = MLP(in_dim=2, hidden=cfg.hidden, depth=cfg.depth, out_dim=1)

		def u(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
			xt = torch.cat([x, t], dim=-1)
			return self.model(xt)

		def pde_residual(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
			x.requires_grad_(True)
			t.requires_grad_(True)
			u = self.u(x, t)
			grads = torch.ones_like(u)
			u_x = torch.autograd.grad(u, x, grads, create_graph=True)[0]
			u_t = torch.autograd.grad(u, t, grads, create_graph=True)[0]
			if self.cfg.nu != 0.0:
				u_xx = torch.autograd.grad(u_x, x, grads, create_graph=True)[0]
				f = u_t + u * u_x - self.cfg.nu * u_xx
			else:
				f = u_t + u * u_x
			return (f ** 2).mean()

		def ic_loss(self, x: "torch.Tensor") -> "torch.Tensor":
			t0 = torch.zeros_like(x)
			u0 = -torch.sin(math.pi * x)
			u_pred = self.u(x, t0)
			return ((u_pred - u0) ** 2).mean()

		def bc_loss(self, t: "torch.Tensor") -> "torch.Tensor":
			xL = torch.full_like(t, self.cfg.x_low)
			xR = torch.full_like(t, self.cfg.x_high)
			uL = self.u(xL, t)
			uR = self.u(xR, t)
			gradsL = torch.ones_like(uL)
			gradsR = torch.ones_like(uR)
			u_x_L = torch.autograd.grad(uL, xL, gradsL, create_graph=True)[0]
			u_x_R = torch.autograd.grad(uR, xR, gradsR, create_graph=True)[0]
			return ((uL - uR) ** 2).mean() + ((u_x_L - u_x_R) ** 2).mean()


def train_pinn(cfg: PINNConfig):
	if not HAS_TORCH:
		print("[PINN] Skipped (PyTorch not available)")
		return None
	model = BurgersPINN(cfg).to(device())
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	def sample_collocation(n: int):
		x = torch.rand(n, 1, device=device()) * (cfg.x_high - cfg.x_low) + cfg.x_low
		t = torch.rand(n, 1, device=device()) * (cfg.t_high - cfg.t_low) + cfg.t_low
		return x, t

	def sample_ic(n: int):
		x = torch.rand(n, 1, device=device()) * (cfg.x_high - cfg.x_low) + cfg.x_low
		return x

	def sample_bc(n: int):
		t = torch.rand(n, 1, device=device()) * (cfg.t_high - cfg.t_low) + cfg.t_low
		return t

	start = time.time()
	for epoch in range(cfg.epochs):
		opt.zero_grad()
		x_f, t_f = sample_collocation(cfg.n_collocation)
		x_ic = sample_ic(cfg.n_ic)
		t_bc = sample_bc(cfg.n_bc)

		loss_pde = model.pde_residual(x_f, t_f)
		loss_ic = model.ic_loss(x_ic)
		loss_bc = model.bc_loss(t_bc)
		loss = loss_pde + loss_ic + 0.1 * loss_bc
		loss.backward()
		opt.step()

		if (epoch + 1) % 500 == 0 or epoch == 0:
			elapsed = time.time() - start
			print(f"[PINN] Epoch {epoch+1}/{cfg.epochs} | Loss {loss.item():.4e} | PDE {loss_pde.item():.3e} | IC {loss_ic.item():.3e} | BC {loss_bc.item():.3e} | {elapsed:.1f}s")

	return model


# -------------------------------
# PDE solver (semi-discrete + RK4, NumPy)
# -------------------------------

def solve_burgers_rk4(nx: int = 256, nt: int = 200, t_end: float = 1.0, nu: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	x_low, x_high = -1.0, 1.0
	x = np.linspace(x_low, x_high, nx, endpoint=False)
	dx = (x_high - x_low) / nx

	u = -np.sin(np.pi * x)
	U = np.zeros((nt + 1, nx), dtype=np.float64)
	U[0] = u

	dt = t_end / nt

	def f(u_vec: np.ndarray, _t: float) -> np.ndarray:
		dudx = np_periodic_diff_1d(u_vec, dx)
		if nu != 0.0:
			d2udx2 = np_periodic_diff2_1d(u_vec, dx)
			return -(u_vec * dudx) + nu * d2udx2
		return -(u_vec * dudx)

	for n in range(nt):
		u = rk4_step_np(f, u, n * dt, dt)
		U[n + 1] = u

	t = np.linspace(0.0, t_end, nt + 1)
	return x, t, U


# -------------------------------
# SINDy (STLSQ, NumPy)
# -------------------------------

def build_library(u: np.ndarray, ux: np.ndarray, uxx: np.ndarray | None = None) -> np.ndarray:
	terms = [np.ones_like(u), u, u**2, ux, u * ux]
	if uxx is not None:
		terms.append(uxx)
	return np.stack(terms, axis=-1)


def stlsq(X: np.ndarray, y: np.ndarray, threshold: float = 0.05, max_iter: int = 10) -> np.ndarray:
	Xi, *_ = np.linalg.lstsq(X, y, rcond=None)
	for _ in range(max_iter):
		small = np.abs(Xi) < threshold
		Xi[small] = 0.0
		big = ~small
		if not np.any(big):
			break
		Xi[big], *_ = np.linalg.lstsq(X[:, big], y, rcond=None)
	return Xi


def run_sindy_from_snapshots(x: np.ndarray, t: np.ndarray, U: np.ndarray, nu: float = 0.0) -> None:
	dt = t[1] - t[0]
	Ut = (U[2:] - U[:-2]) / (2.0 * dt)
	U_mid = U[1:-1]

	dx = x[1] - x[0]
	Ux = np_periodic_diff_1d(U_mid, dx)
	Uxx = np_periodic_diff2_1d(U_mid, dx) if nu != 0.0 else None

	Theta = build_library(U_mid, Ux, Uxx)
	y = Ut

	ns, nx = U_mid.shape
	Xmat = Theta.reshape(ns * nx, -1)
	yvec = y.reshape(ns * nx)
	Xi = stlsq(Xmat, yvec, threshold=0.02, max_iter=10)

	print("[SINDy] Coefficients (1, u, u^2, ux, u*ux, uxx?):")
	print(np.round(Xi, 3))


# -------------------------------
# Neural ODE (optional, requires torch)
# -------------------------------
if HAS_TORCH:
	class NeuralVectorField(nn.Module):
		def __init__(self, dim: int, hidden: int = 128, depth: int = 3) -> None:
			super().__init__()
			layers = []
			in_dim = dim
			for _ in range(depth - 1):
				layers.append(nn.Linear(in_dim, hidden))
				layers.append(nn.Tanh())
				in_dim = hidden
			layers.append(nn.Linear(in_dim, dim))
			self.net = nn.Sequential(*layers)

		def forward(self, z: "torch.Tensor", t: float | "torch.Tensor") -> "torch.Tensor":
			return self.net(z)

if HAS_TORCH:
	def rk4_integrate_vecfield(f: Callable[["torch.Tensor", float], "torch.Tensor"], z0: "torch.Tensor", t: np.ndarray) -> "torch.Tensor":
		z = z0
		for i in range(len(t) - 1):
			dt = float(t[i + 1] - t[i])
			z = rk4_step(lambda zz, tt: f(zz, tt), z, float(t[i]), dt)
		return z


def train_neural_ode_from_data(U: np.ndarray, t: np.ndarray, downsample: int = 4, epochs: int = 500) -> None:
	if not HAS_TORCH:
		print("[NeuralODE] Skipped (PyTorch not available)")
		return
	Uc = torch.from_numpy(U[:, ::downsample]).float().to(device())
	Tsteps = Uc.shape[0]
	z_dim = Uc.shape[1]

	vec = NeuralVectorField(dim=z_dim).to(device())
	opt = torch.optim.Adam(vec.parameters(), lr=1e-3)

	for epoch in range(epochs):
		perm = torch.randperm(Tsteps - 1)
		total_loss = 0.0
		for idx in perm:
			z0 = Uc[idx]
			z1 = Uc[idx + 1]
			z0 = z0.unsqueeze(0)
			z1 = z1.unsqueeze(0)
			pred = rk4_integrate_vecfield(vec.forward, z0, t[idx:idx + 2])
			loss = F.mse_loss(pred, z1)
			opt.zero_grad()
			loss.backward()
			opt.step()
			total_loss += float(loss.item())
		if (epoch + 1) % 100 == 0 or epoch == 1:
			print(f"[NeuralODE] Epoch {epoch+1}/{epochs} | loss {total_loss / (Tsteps-1):.4e}")


# -------------------------------
# DMD (NumPy)
# -------------------------------

def run_dmd(U: np.ndarray, r: int = 10) -> None:
	X = U[:-1].T
	Xp = U[1:].T

	Umat, S, Vh = np.linalg.svd(X, full_matrices=False)
	Ur = Umat[:, :r]
	Sr = np.diag(S[:r])
	Vr = Vh[:r, :]

	A_tilde = Ur.T @ Xp @ Vr.T @ np.linalg.inv(Sr)
	w, W = np.linalg.eig(A_tilde)
	Phi = Xp @ Vr.T @ np.linalg.inv(Sr) @ W

	b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
	time_dynamics = np.zeros((r, X.shape[1]), dtype=np.complex128)
	for i in range(X.shape[1]):
		time_dynamics[:, i] = (w ** i) * b
	Xrec = (Phi @ time_dynamics).real

	rel_err = np.linalg.norm(X - Xrec) / np.linalg.norm(X)
	print(f"[DMD] r={r} | relative reconstruction error {rel_err:.3e}")


# -------------------------------
# Main
# -------------------------------

def main() -> None:
	set_seed(0)
	print(f"Using device: {device()}")

	# 1) Solve PDE by RK4 finite differences (ground truth surrogate)
	nx, nt, t_end, nu = 256, 300, 1.0, 0.0
	x, tgrid, U = solve_burgers_rk4(nx=nx, nt=nt, t_end=t_end, nu=nu)
	print(f"[RK4 PDE] U shape: {U.shape} | sample u(x=0,t=1): {U[-1, nx//2]:.4f}")

	# 2) Train PINN with RTT residual (optional)
	cfg = PINNConfig(nu=nu, epochs=500, n_collocation=1024, n_ic=128, n_bc=128, hidden=32, depth=2)
	pinn = train_pinn(cfg)
	if HAS_TORCH and pinn is not None:
		xt = torch.from_numpy(x).reshape(-1, 1).float()
		tt = torch.full_like(xt, float(cfg.t_high))
		with torch.no_grad():
			upred = pinn.u(xt, tt).cpu().numpy().squeeze()
		l2_rel = np.linalg.norm(upred - U[-1]) / np.linalg.norm(U[-1])
		print(f"[PINN] L2 relative error at t=1.0: {l2_rel:.3e}")

	# 3) SINDy on simulated data (NumPy)
	run_sindy_from_snapshots(x, tgrid, U, nu=nu)

	# 4) Neural ODE learned coarse dynamics (optional)
	train_neural_ode_from_data(U, tgrid, downsample=4, epochs=100)

	# 5) DMD decomposition and reconstruction
	run_dmd(U, r=10)


if __name__ == "__main__":
	main()