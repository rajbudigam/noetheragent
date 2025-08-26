import numpy as np

def t_opt_discrepancy(t, y1, y2):
    """
    T-opt criterion: integral of squared mean difference.
    """
    d = y1 - y2
    return float(np.trapz(d**2, t))

def kl_discrimination(t, y1, y2, sigma):
    """
    KL-opt (equal-variance Gaussian): 0.5/σ^2 * ∫ (μ1-μ2)^2 dt
    """
    return 0.5 * t_opt_discrepancy(t, y1, y2) / (sigma**2)

def jeffreys_discrimination(t, y1, y2, sigma):
    """
    Jeffreys (symmetrized KL): KL(p||q)+KL(q||p) = 1/σ^2 * ∫ (μ1-μ2)^2 dt
    """
    return t_opt_discrepancy(t, y1, y2) / (sigma**2)
