import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

def discrepancy_objective(theta0, rival_sim_fn, T, dt):
    """
    Deterministic 'T-optimal' proxy: L2 distance between rival trajectories.
    theta0: scalar initial angle (radians); omega0=0.
    rival_sim_fn: callable(theta0) -> (t, x_sindy, x_linear)
    returns scalar discrepancy J(theta0)
    """
    t, xs, xl = rival_sim_fn(theta0)
    # We emphasize angle mismatch; you can add velocity as well
    d = xs[:,0] - xl[:,0]
    return float(np.trapz(d**2, t))

def suggest_next_theta0(bounds, initial_thetas, initial_scores, n_candidates=1):
    """
    Fit a GP to (theta0 -> score) and propose next theta0 via qEI.
    bounds: tuple (lo, hi) in radians
    """
    X = torch.tensor(initial_thetas, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(initial_scores, dtype=torch.double).unsqueeze(-1)
    # normalize targets
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)

    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    lo, hi = bounds
    bounds_t = torch.tensor([[lo], [hi]], dtype=torch.double)

    qei = qExpectedImprovement(model=gp, best_f=Y.max())
    cand, _ = optimize_acqf(
        acq_function=qei,
        bounds=bounds_t,
        q=n_candidates,
        num_restarts=10,
        raw_samples=256,
    )
    return cand.detach().squeeze(-1).cpu().numpy().tolist()
