import json
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import importlib

from noetheragent.data.generate_pendulum import (
    PendulumParams, stack_trajectories, simulate_pendulum
)
from noetheragent.modeling.sindy_model import fit_sindy, sindy_predict
from noetheragent.modeling.baselines import LinearizedPendulum, simulate as simulate_lin
from noetheragent.design.oed_botorch import discrepancy_objective, suggest_next_theta0
from noetheragent.util.metrics import long_horizon_rmse
from noetheragent.util.plotting import plot_discrepancy, plot_trajectories

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def dump_versions():
    pkgs = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sympy": "sympy",
        "pysindy": "pysindy",
        "torch": "torch",
        "gpytorch": "gpytorch",
        "botorch": "botorch",
    }
    lines = []
    for name, mod in pkgs.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
        except Exception:
            v = "not-importable"
        lines.append(f"{name}=={v}")
    (ART / "VERSIONS.txt").write_text("\n".join(lines))

def discover_with_sindy(params: PendulumParams, seeds=6, T=6.0, dt=0.01):
    rng = np.random.default_rng(123)
    thetas0 = rng.uniform(low=0.05, high=1.2, size=seeds) * rng.choice([-1,1], size=seeds)
    omegas0 = np.zeros_like(thetas0)
    _, X, Xdot = stack_trajectories(thetas0, omegas0, T, dt, params)
    # Construct a synthetic time vector matching concatenation
    t = np.tile(np.linspace(0.0, T, int(T/dt)+1), seeds)

    model = fit_sindy(X, t, library_kind="fourier", n_frequencies=1, threshold=0.05)
    (ART/"sindy_model.txt").write_text(model.equations())
    return model

def rival_sim_fn_factory(sindy_model, params: PendulumParams, T, dt):
    lin = LinearizedPendulum(g=params.g, L=params.L)
    def sim(theta0):
        _, xs = sindy_predict(sindy_model, np.array([theta0, 0.0]), T, dt)
        _, xl = simulate_lin(lin, theta0, 0.0, T, dt)
        t = np.linspace(0.0, T, int(T/dt)+1)
        return t, xs, xl
    return sim

def parse_args():
    p = argparse.ArgumentParser(description="NoetherAgent pipeline")
    p.add_argument("--T", type=float, default=6.0, help="trajectory length (s)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (s)")
    p.add_argument("--seeds", type=int, default=8, help="num initial conditions for discovery")
    p.add_argument("--grid", type=int, default=20, help="grid points for initial θ0 sweep")
    return p.parse_args()

def main():
    args = parse_args()
    dump_versions()

    params = PendulumParams(g=9.81, L=1.0)
    T, dt = args.T, args.dt

    print("1) Discovery with SINDy...")
    sindy_model = discover_with_sindy(params, seeds=args.seeds, T=T, dt=dt)
    print("Discovered equations:\n", sindy_model.equations())

    print("2) Model discrimination — pick θ0 to separate rival laws...")
    rival_sim = rival_sim_fn_factory(sindy_model, params, T=T, dt=dt)

    # Evaluate initial θ0 grid
    grid = np.linspace(0.05, 2.5, args.grid)
    scores = [discrepancy_objective(th, rival_sim, T=T, dt=dt) for th in tqdm(grid)]
    np.save(ART/"initial_thetas.npy", grid)
    np.save(ART/"initial_scores.npy", scores)

    # Suggest next θ0 via BoTorch
    theta_next = suggest_next_theta0(bounds=(0.05, 2.5),
                                     initial_thetas=grid.tolist(),
                                     initial_scores=scores,
                                     n_candidates=1)[0]
    print(f"Suggested next θ0 (rad): {theta_next:.4f}")

    # 3) Validate & compare trajectories
    t, x_truth = simulate_pendulum(theta_next, 0.0, T=T, dt=dt, params=params)
    _, x_sindy = sindy_predict(sindy_model, np.array([theta_next, 0.0]), T, dt)
    lin = LinearizedPendulum(g=params.g, L=params.L)
    _, x_lin = simulate_lin(lin, theta_next, 0.0, T, dt)

    rmse_sindy = long_horizon_rmse(x_truth[:,0], x_sindy[:,0])
    rmse_lin   = long_horizon_rmse(x_truth[:,0], x_lin[:,0])

    # Plots
    plot_discrepancy(grid, scores, theta_next, ART/"discrepancy.png")
    plot_trajectories(t, x_truth, x_sindy, x_lin, ART/"trajectories.png")

    out = {
        "theta_next": float(theta_next),
        "rmse_angle_truth_vs_sindy": rmse_sindy,
        "rmse_angle_truth_vs_linear": rmse_lin,
        "sindy_equations": sindy_model.equations(),
        "T": T, "dt": dt, "seeds": args.seeds, "grid_points": args.grid,
    }
    (ART/"summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
