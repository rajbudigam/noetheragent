import numpy as np

def long_horizon_rmse(truth, pred):
    return float(np.sqrt(np.mean((truth - pred)**2)))
