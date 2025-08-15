#!/usr/bin/env python3
import argparse
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class CFDParams:
    nx: int = 200
    ny: int = 100
    lx: float = 1.0
    ly: float = 0.5
    rho: float = 1.0
    nu: float = 1e-6
    u_inlet: float = 2.0
    aoa_deg: float = 6.5
    dt: float = 5e-4
    n_steps: int = 4000
    n_poisson: int = 120
    poiss_omega: float = 1.0
    save_every: int = 400


class FinMask:
    def __init__(self, nx: int, ny: int, cx: float = 0.25, cy: float = 0.5, chord: float = 0.25, thickness: float = 0.06, camber: float = 0.02, angle_deg: float = 0.0):
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.chord = chord
        self.thickness = thickness
        self.camber = camber
        self.angle = math.radians(angle_deg)

    def _naca4_xy(self, n: int = 200):
        m = max(self.camber, 1e-6)
        p = 0.4
        t = max(self.thickness, 5e-4)
        x = np.linspace(0.0, 1.0, n)
        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        yc = np.where(x < p, m/(p**2) * (2*p*x - x**2), m/((1-p)**2) * ((1 - 2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/(p**2) * (p - x), 2*m/((1-p)**2) * (p - x))
        theta = np.arctan(dyc)
        xu = x - yt*np.sin(theta)
        yu = yc + yt*np.cos(theta)
        xl = x + yt*np.sin(theta)
        yl = yc - yt*np.cos(theta)
        xb = np.concatenate([xu[::-1], xl[1:]])
        yb = np.concatenate([yu[::-1], yl[1:]])
        return xb, yb

    def rasterize(self) -> np.ndarray:
        xb, yb = self._naca4_xy()
        # Scale and rotate then translate to domain
        # Convert to physical coords relative to domain [0,1]x[0,1]
        x = self.chord * xb
        y = self.chord * yb
        c, s = math.cos(self.angle), math.sin(self.angle)
        xr = c*x - s*y
        yr = s*x + c*y
        xr += self.cx
        yr += self.cy
        # Rasterize into mask (True = solid)
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        # Use matplotlib-like path containment via ray casting
        poly = np.stack([xr, yr], axis=1)
        # Grid cell centers
        gx = (np.arange(self.nx) + 0.5) / self.nx
        gy = (np.arange(self.ny) + 0.5) / self.ny
        GX, GY = np.meshgrid(gx, gy)
        points = np.stack([GX.ravel(), GY.ravel()], axis=1)
        mask_flat = _points_in_polygon(points, poly)
        mask[:, :] = mask_flat.reshape(self.ny, self.nx)
        return mask


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    # Vectorized ray casting algorithm
    x, y = points[:, 0], points[:, 1]
    xp, yp = polygon[:, 0], polygon[:, 1]
    n = len(xp)
    inside = np.zeros_like(x, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi)
        inside ^= intersect
        j = i
    return inside


# Upwind derivative helpers for stability
def _upwind_x(f: np.ndarray, vel_x: np.ndarray, dx: float) -> np.ndarray:
    # Non-wrapping neighbors (edge replicated)
    f_left = np.concatenate([f[:, :1], f[:, :-1]], axis=1)
    f_right = np.concatenate([f[:, 1:], f[:, -1:]], axis=1)
    pos = vel_x >= 0.0
    dfdx = np.where(pos, (f - f_left) / dx, (f_right - f) / dx)
    return dfdx


def _upwind_y(f: np.ndarray, vel_y: np.ndarray, dy: float) -> np.ndarray:
    # Non-wrapping neighbors (edge replicated)
    f_down = np.concatenate([f[:1, :], f[:-1, :]], axis=0)
    f_up = np.concatenate([f[1:, :], f[-1:, :]], axis=0)
    pos = vel_y >= 0.0
    dfdy = np.where(pos, (f - f_down) / dy, (f_up - f) / dy)
    return dfdy


class ProjectionCFD:
    def __init__(self, params: CFDParams):
        self.p = params
        self.dx = self.p.lx / self.p.nx
        self.dy = self.p.ly / self.p.ny
        shape = (self.p.ny, self.p.nx)
        self.u = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.pf = np.zeros(shape, dtype=np.float64)
        self.mask = np.zeros(shape, dtype=bool)

    def set_fin_mask(self, mask: np.ndarray):
        assert mask.shape == self.u.shape
        self.mask = mask

    def step(self):
        rho, nu, dt_max = self.p.rho, self.p.nu, self.p.dt
        dx, dy = self.dx, self.dy
        u, v, p = self.u, self.v, self.pf
        # Inflow boundary with AoA
        U = self.p.u_inlet
        aoa = math.radians(self.p.aoa_deg)
        u_in, v_in = U*math.cos(aoa), U*math.sin(aoa)
        u[:, 0] = u_in
        v[:, 0] = v_in
        # Neumann on outflow (copy from previous interior)
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        # No-slip on solid mask
        u[self.mask] = 0.0
        v[self.mask] = 0.0
        # Adaptive time step (CFL + diffusion)
        umax = max(1e-6, float(np.nanmax(np.abs(u))))
        vmax = max(1e-6, float(np.nanmax(np.abs(v))))
        dt_cfl = 0.25 * min(dx/umax, dy/vmax)
        dt_diff = 0.25 * min(dx*dx, dy*dy) / max(1e-12, nu)
        dt = min(dt_max, dt_cfl, dt_diff)
        # Compute intermediate velocity u* (upwind advection + diffusion + pressure from last step)
        u_star = u.copy()
        v_star = v.copy()
        # Upwind advection
        dudx_up = _upwind_x(u, u, dx)
        dudy_up = _upwind_y(u, v, dy)
        dvdx_up = _upwind_x(v, u, dx)
        dvdy_up = _upwind_y(v, v, dy)
        u_adv = u * dudx_up + v * dudy_up
        v_adv = u * dvdx_up + v * dvdy_up
        # Diffusion (central Laplacian) without wrap-around via edge padding
        def laplace(f: np.ndarray) -> np.ndarray:
            fpad = np.pad(f, ((1, 1), (1, 1)), mode='edge')
            center = fpad[1:-1, 1:-1]
            left = fpad[1:-1, 0:-2]
            right = fpad[1:-1, 2:]
            down = fpad[0:-2, 1:-1]
            up = fpad[2:, 1:-1]
            return (right - 2*center + left) / (dx*dx) + (up - 2*center + down) / (dy*dy)

        lap_u = laplace(u)
        lap_v = laplace(v)
        # Pressure gradients (central) without wrap-around via edge padding
        def grad_central(f: np.ndarray):
            fpad = np.pad(f, ((1, 1), (1, 1)), mode='edge')
            fx = (fpad[1:-1, 2:] - fpad[1:-1, 0:-2]) / (2*dx)
            fy = (fpad[2:, 1:-1] - fpad[0:-2, 1:-1]) / (2*dy)
            return fx, fy

        dpdx, dpdy = grad_central(p)
        u_star += dt * (-dpdx / rho - u_adv + nu * lap_u)
        v_star += dt * (-dpdy / rho - v_adv + nu * lap_v)
        # Clip to prevent overflow
        clip_mag = 10.0 * max(U, 1.0)
        np.clip(u_star, -clip_mag, clip_mag, out=u_star)
        np.clip(v_star, -clip_mag, clip_mag, out=v_star)
        # Enforce no-slip again on u*
        u_star[self.mask] = 0.0
        v_star[self.mask] = 0.0
        # Pressure Poisson: div(u*) -> solve for p^{n+1}
        # Divergence (central, no wrap)
        uxp, uyp = grad_central(u_star)
        vxp, vyp = grad_central(v_star)
        div_u = uxp + vyp
        rhs = (rho / dt) * div_u
        # Clamp RHS to prevent blow-up from transient spikes
        rhs_clip = 1e4
        np.clip(rhs, -rhs_clip, rhs_clip, out=rhs)
        p_new = p.copy()
        beta = - (1.0 / (2.0/(dx*dx) + 2.0/(dy*dy)))
        omega = self.p.poiss_omega
        for _ in range(self.p.n_poisson):
            # Jacobi using padded neighbors (no wrap)
            p_old = p_new
            ppad = np.pad(p_old, ((1, 1), (1, 1)), mode='edge')
            pxx = (ppad[1:-1, 2:] + ppad[1:-1, 0:-2]) / (dx*dx)
            pyy = (ppad[2:, 1:-1] + ppad[0:-2, 1:-1]) / (dy*dy)
            p_new = (pxx + pyy - rhs) * beta
            # Solid cells: set pressure to zero (reference inside solid)
            p_new[self.mask] = 0.0
            # Over-relaxation
            p_new = omega * p_new + (1 - omega) * p_old
            # Pressure BC: Neumann at outlet, zero-gradient top/bottom, Dirichlet reference at inlet corner
            p_new[:, -1] = p_new[:, -2]
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[0, 0] = 0.0
        # Subtract mean to avoid drift
        p[:] = p_new - np.nanmean(p_new)
        # Projection: u^{n+1} = u* - dt/rho grad p
        dpdx, dpdy = grad_central(p)
        self.u = u_star - (dt / rho) * dpdx
        self.v = v_star - (dt / rho) * dpdy
        np.clip(self.u, -clip_mag, clip_mag, out=self.u)
        np.clip(self.v, -clip_mag, clip_mag, out=self.v)
        # Re-enforce BCs and mask
        self.u[:, 0] = u_in
        self.v[:, 0] = v_in
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        self.u[self.mask] = 0.0
        self.v[self.mask] = 0.0

    def run(self, out_path: str):
        for step in range(1, self.p.n_steps + 1):
            self.step()
            if step % self.p.save_every == 0 or step == self.p.n_steps:
                self._save(out_path, step)

    def _save(self, out_path: str, step: int):
        np.savez_compressed(
            f"{out_path}/state_{step:06d}.npz",
            u=self.u, v=self.v, p=self.pf, mask=self.mask,
            dx=self.dx, dy=self.dy, rho=self.p.rho, nu=self.p.nu,
        )


def main():
    parser = argparse.ArgumentParser(description="2D Incompressible CFD solver (projection method) with fin mask")
    parser.add_argument("--nx", type=int, default=300)
    parser.add_argument("--ny", type=int, default=150)
    parser.add_argument("--u", type=float, default=2.0, help="Inlet speed m/s")
    parser.add_argument("--aoa", type=float, default=10.0, help="Angle of attack in degrees")
    parser.add_argument("--nu", type=float, default=1e-6, help="Kinematic viscosity")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=300)
    parser.add_argument("--out", type=str, default="/workspace/cfd_mojo/out")
    parser.add_argument("--chord", type=float, default=0.25)
    parser.add_argument("--thickness", type=float, default=0.06)
    parser.add_argument("--camber", type=float, default=0.02)
    parser.add_argument("--fin_angle", type=float, default=6.5)
    args = parser.parse_args()

    params = CFDParams(nx=args.nx, ny=args.ny, u_inlet=args.u, aoa_deg=args.aoa, nu=args.nu, n_steps=args.steps, save_every=args.save_every)
    solver = ProjectionCFD(params)
    fin = FinMask(nx=args.nx, ny=args.ny, cx=0.28, cy=0.5, chord=args.chord, thickness=args.thickness, camber=args.camber, angle_deg=args.fin_angle)
    mask = fin.rasterize()
    solver.set_fin_mask(mask)

    import os
    os.makedirs(args.out, exist_ok=True)
    solver.run(args.out)


if __name__ == "__main__":
    main()