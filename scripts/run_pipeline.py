import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import importlib
import argparse
import sys

from noetheragent.data.generate_pendulum import (
    PendulumParams, stack_trajectories, simulate_pendulum
)
from noetheragent.modeling.sindy_model import fit_sindy, sindy_predict
from noetheragent.modeling.baselines import LinearizedPendulum, simulate as simulate_lin
from noetheragent.modeling.hnn_lnn import HNN, HNNConfig, LNNMinimal, LNNConfig
from noetheragent.design.oed_botorch import objective_score, suggest_next_theta0
from noetheragent.util.metrics import long_horizon_rmse
from noetheragent.util.plotting import plot_discrepancy, plot_trajectories

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def dump_versions():
    pkgs = ["numpy","scipy","pandas","matplotlib","sympy","pysindy","torch","gpytorch","botorch"]
    lines = []
    for mod in pkgs:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
        except Exception:
            v = "not-importable"
        lines.append(f"{mod}=={v}")
    (ART / "VERSIONS.txt").write_text("\n".join(lines))

def check_not_empty(arr, name):
    if arr is None or not hasattr(arr, 'size') or arr.size == 0:
        print(f"ERROR: {name} is empty or None. Halting execution.", file=sys.stderr)
        sys.exit(1)

def discover_with_sindy(params: PendulumParams, seeds=6, T=6.0, dt=0.01):
    if seeds < 1:
        print(f"ERROR: seeds must be >= 1, got {seeds}.", file=sys.stderr)
        sys.exit(1)
    rng = np.random.default_rng(123)
    thetas0 = rng.uniform(low=0.1, high=1.6, size=seeds) * rng.choice([-1,1], size=seeds)
    omegas0 = rng.uniform(low=-1.5, high=1.5, size=seeds)
    _, X, _ = stack_trajectories(thetas0, omegas0, T, dt, params)
    check_not_empty(X, 'X from stack_trajectories')
    t = np.tile(np.linspace(0.0, T, int(T/dt)+1), seeds)
    if t.size == 0:
        print("ERROR: t (time vector) is empty after tiling.", file=sys.stderr)
        sys.exit(1)
    model = fit_sindy(X, t, library_kind="custom", threshold=0.08)
    (ART/"sindy_model.txt").write_text("\n".join(model.equations()))
    return model, (X, t)

def rival_sim_fn_factory(primary_model, rival_kind, trained, params: PendulumParams, T, dt):
    lin = trained.get("linear")
    hnn = trained.get("hnn")
    lnn = trained.get("lnn")
    def sim(theta0):
        _, xs = sindy_predict(primary_model, np.array([theta0, 0.0]), T, dt)
        check_not_empty(xs, "xs from sindy_predict")
        if rival_kind == "linear":
            _, xr = simulate_lin(lin, theta0, 0.0, T, dt)
        elif rival_kind == "hnn":
            if hnn is None:
                print("ERROR: HNN rival requested but not trained.", file=sys.stderr)
                sys.exit(1)
            _, xr = hnn.simulate(np.array([theta0, 0.0]), T, dt)
        elif rival_kind == "lnn":
            if lnn is None:
                print("ERROR: LNN rival requested but not trained.", file=sys.stderr)
                sys.exit(1)
            _, xr = lnn.simulate(np.array([theta0, 0.0]), T, dt)
        else:
            print(f"ERROR: unknown rival_kind '{rival_kind}'", file=sys.stderr)
            sys.exit(1)
        check_not_empty(xr, "xr from rival simulation")
        t = np.linspace(0.0, T, int(T/dt)+1)
        return t, xs, xr
    return sim

def parse_args():
    p = argparse.ArgumentParser(description="NoetherAgent pipeline")
    p.add_argument("--T", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--oed_objective", choices=["topt","kl"], default="topt", help="Design objective: T-opt or Gaussian KL surrogate")
    p.add_argument("--noise_sigma", type=float, default=0.05, help="Obs noise (rad) for KL objective")
    p.add_argument("--oed_rival", choices=["linear","hnn","lnn"], default="linear", help="Rival model vs SINDy in discrimination")
    p.add_argument("--train_hnn", action="store_true")
    p.add_argument("--train_lnn", action="store_true")
    p.add_argument("--hnn_epochs", type=int, default=250)
    p.add_argument("--lnn_epochs", type=int, default=250)
    return p.parse_args()

def main():
    args = parse_args()
    if args.seeds < 1:
        print(f"ERROR: --seeds must be >= 1, got {args.seeds}", file=sys.stderr)
        sys.exit(1)
    if args.T <= 0 or args.dt <= 0:
        print(f"ERROR: --T and --dt must be > 0, got T={args.T}, dt={args.dt}", file=sys.stderr)
        sys.exit(1)
    dump_versions()
    params = PendulumParams(g=9.81, L=1.0)
    T, dt = args.T, args.dt

    print("1) Discovery with SINDy...")
    sindy_model, (X_train, t_train) = discover_with_sindy(params, seeds=args.seeds, T=T, dt=dt)
    check_not_empty(X_train, "X_train from discover_with_sindy")

    print("Discovered equations:\n", sindy_model.equations())
    print("2) Train rivals (opt-in)...")
    trained = {}
    trained["linear"] = LinearizedPendulum(g=params.g, L=params.L)
    n = int(T/dt)+1
    Xdot = np.zeros_like(X_train)
    if X_train.shape[0] < 2:
        print("ERROR: Not enough data to compute derivatives (need at least 2 timesteps).", file=sys.stderr)
        sys.exit(1)
    Xdot[1:-1] = (X_train[2:] - X_train[:-2]) / (2*dt)
    Xdot[0] = (X_train[1] - X_train[0]) / dt
    Xdot[-1] = (X_train[-1] - X_train[-2]) / dt

    if args.train_hnn or args.oed_rival == "hnn":
        hcfg = HNNConfig(epochs=args.hnn_epochs)
        hnn = HNN(hcfg)
        hnn.fit(X_train, Xdot)
        trained["hnn"] = hnn
        print("HNN trained.")
    if args.train_lnn or args.oed_rival == "lnn":
        lcfg = LNNConfig(epochs=args.lnn_epochs)
        lnn = LNNMinimal(lcfg)
        lnn.fit(X_train, Xdot)
        trained["lnn"] = lnn
        print("LNN (minimal) trained.")

    print(f"3) Model discrimination objective = {args.oed_objective}, rival = {args.oed_rival}")
    rival_sim = rival_sim_fn_factory(sindy_model, args.oed_rival, trained, params, T=T, dt=dt)

    grid = np.linspace(0.1, 2.2, args.grid)
    if grid.size == 0:
        print("ERROR: grid is empty.", file=sys.stderr)
        sys.exit(1)
    scores = []
    for th in tqdm(grid):
        try:
            score = objective_score(th, rival_sim, T=T, dt=dt, objective=args.oed_objective, noise_sigma=args.noise_sigma)
            scores.append(score)
        except Exception as e:
            print(f"WARNING: Failed to compute objective_score for theta={th}: {e}", file=sys.stderr)
            scores.append(float("nan"))
    np.save(ART/"initial_thetas.npy", grid)
    np.save(ART/"initial_scores.npy", scores)

    if np.all(np.isnan(scores)):
        print("ERROR: All scores are NaN; cannot proceed to suggest next theta.", file=sys.stderr)
        sys.exit(1)

    theta_next = suggest_next_theta0(bounds=(0.1, 2.2), thetas=grid.tolist(), scores=scores, n_candidates=1)[0]
    print(f"Suggested next θ0 (rad): {theta_next:.4f}")

    # Validate trajectories at θ0*
    t, x_truth = simulate_pendulum(theta_next, 0.0, T=T, dt=dt, params=params)
    check_not_empty(x_truth, f"x_truth for θ0={theta_next}")
    _, x_sindy = sindy_predict(sindy_model, np.array([theta_next, 0.0]), T, dt)
    check_not_empty(x_sindy, "x_sindy")
    if args.oed_rival == "linear":
        _, x_rival = simulate_lin(trained["linear"], theta_next, 0.0, T, dt)
    elif args.oed_rival == "hnn":
        _, x_rival = trained["hnn"].simulate(np.array([theta_next, 0.0]), T, dt)
    else:
        _, x_rival = trained["lnn"].simulate(np.array([theta_next, 0.0]), T, dt)
    check_not_empty(x_rival, "x_rival")

    rmse_sindy = long_horizon_rmse(x_truth[:,0], x_sindy[:,0])
    rmse_rival = long_horizon_rmse(x_truth[:,0], x_rival[:,0])

    ylabel = "T-opt discrepancy" if args.oed_objective=="topt" else "Info gain (KL surrogate)"
    plot_discrepancy(grid, scores, theta_next, ART/"discrepancy.png", ylabel=ylabel)
    lin_tmp = LinearizedPendulum(g=params.g, L=params.L)
    _, x_lin = simulate_lin(lin_tmp, theta_next, 0.0, T, dt)
    plot_trajectories(t, x_truth, x_sindy, x_lin, ART/"trajectories.png")

    out = {
        "theta_next": float(theta_next),
        "oed_objective": args.oed_objective,
        "noise_sigma": args.noise_sigma,
        "oed_rival": args.oed_rival,
        "rmse_angle_truth_vs_sindy": rmse_sindy,
        "rmse_angle_truth_vs_rival": rmse_rival,
        "sindy_equations": sindy_model.equations(),
        "T": T, "dt": dt, "seeds": args.seeds, "grid_points": args.grid,
        "hnn_trained": bool("hnn" in trained),
        "lnn_trained": bool("lnn" in trained),
    }
    (ART/"summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
