import matplotlib.pyplot as plt
from pathlib import Path

def plot_discrepancy(grid, scores, theta_next, outpath, ylabel="T-opt discrepancy"):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(grid, scores, linewidth=2)
    plt.axvline(theta_next, linestyle="--", linewidth=1.5)
    plt.xlabel(r"$\theta_0$ (rad)")
    plt.ylabel(ylabel)
    plt.title("Model-discrimination landscape")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_trajectories(t, truth, sindy, linear, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.plot(t, truth[:,0], label="truth θ", linewidth=2)
    plt.plot(t, sindy[:,0], "--", label="SINDy θ", linewidth=1.8)
    plt.plot(t, linear[:,0], ":", label="Linear θ", linewidth=1.8)
    plt.xlabel("t (s)")
    plt.ylabel(r"$\theta$ (rad)")
    plt.legend(frameon=False)
    plt.title("Trajectory comparison at suggested amplitude")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
