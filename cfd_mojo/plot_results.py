#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


def plot_npz(npz_path: str, out_png: str):
    z = np.load(npz_path)
    u, v, p, mask = z["u"], z["v"], z["p"], z["mask"]
    speed = np.sqrt(u*u + v*v)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    im0 = axs[0].imshow(p, origin="lower", cmap="coolwarm")
    axs[0].set_title("Pressure")
    axs[0].contour(mask, levels=[0.5], colors='k', linewidths=1)
    fig.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(speed, origin="lower", cmap="viridis")
    axs[1].set_title("Speed |u|")
    axs[1].contour(mask, levels=[0.5], colors='k', linewidths=1)
    fig.colorbar(im1, ax=axs[1], shrink=0.8)

    skip = max(1, speed.shape[0] // 25)
    Y, X = np.mgrid[0:speed.shape[0], 0:speed.shape[1]]
    axs[2].quiver(X[::skip, ::skip], Y[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], scale=50)
    axs[2].contour(mask, levels=[0.5], colors='k', linewidths=1)
    axs[2].set_title("Velocity Vectors")
    axs[2].invert_yaxis()

    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot latest CFD state")
    parser.add_argument("--in_dir", type=str, default="/workspace/cfd_mojo/out")
    parser.add_argument("--out_png", type=str, default="/workspace/cfd_mojo/pressure_map.png")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.in_dir}/state_*.npz"))
    if not files:
        raise SystemExit("No state_*.npz found. Run the solver first.")
    plot_npz(files[-1], args.out_png)


if __name__ == "__main__":
    main()