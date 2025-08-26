import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from .info_objectives import t_opt_discrepancy, kl_discrimination

def objective_score(theta0, rival_sim_fn, T, dt, objective="topt", noise_sigma=0.05):
    """
    Compute a scalar discrimination score at a proposed initial angle.
    objective: 'topt' | 'kl'  (Gaussian KL surrogate)
    """
    t, yA, yB = rival_sim_fn(theta0)  # y* are full state trajectories
    # Use angle channel for discrimination (could extend to both)
    aA, aB = yA[:, 0], yB[:, 0]
    if objective == "topt":
        return t_opt_discrepancy(t, aA, aB)
    elif objective == "kl":
        return kl_discrimination(t, aA, aB, sigma=noise_sigma)
    else:
        raise ValueError("unknown objective")

def suggest_next_theta0(bounds, thetas, scores, n_candidates=1):
    """
    Fit a GP to (theta0 -> score) and propose next theta0 via qEI.
    """
    X = torch.tensor(thetas, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(scores, dtype=torch.double).unsqueeze(-1)
    # z-score for numerical stability
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)

    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    lo, hi = bounds
    bounds_t = torch.tensor([[lo], [hi]], dtype=torch.double)

    acq = qExpectedImprovement(model=gp, best_f=Y.max())
    cand, _ = optimize_acqf(acq_function=acq, bounds=bounds_t,
                            q=n_candidates, num_restarts=10, raw_samples=256)
    return cand.detach().squeeze(-1).cpu().numpy().tolist()
