import numpy as np
from noetheragent.data.generate_pendulum import PendulumParams, stack_trajectories
from noetheragent.modeling.sindy_model import fit_sindy

def test_sindy_fits_on_tiny_dataset():
    params = PendulumParams()
    T, dt = 1.0, 0.02
    thetas0 = np.array([0.2])
    omegas0 = np.array([0.0])
    t, X, Xdot = stack_trajectories(thetas0, omegas0, T, dt, params)
    m = fit_sindy(X, t, library_kind="fourier", n_frequencies=1, threshold=0.1)
    # Sanity: equations string exists and contains something
    eq = m.equations()
    assert isinstance(eq, (str, list)) and len(eq) > 0
