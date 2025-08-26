import numpy as np
import pysindy as ps

def _col(x, i):
    return x[:, i:i+1]

def make_pendulum_custom_library(include_interactions=True):
    """
    Features: θ, ω, sin(θ), cos(θ), (optional) θ·ω
    NO trig on ω (physically meaningless for the pendulum force law).
    """
    library_functions = [
        lambda x: _col(x, 0),                # theta
        lambda x: _col(x, 1),                # omega
        lambda x: np.sin(_col(x, 0)),        # sin(theta)
        lambda x: np.cos(_col(x, 0)),        # cos(theta)
    ]
    function_names = [
        lambda s: "theta",
        lambda s: "omega",
        lambda s: "sin(theta)",
        lambda s: "cos(theta)",
    ]
    if include_interactions:
        library_functions.append(lambda x: _col(x, 0) * _col(x, 1))  # theta*omega
        function_names.append(lambda s: "theta*omega")

    return ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names,
    )

def fit_sindy(x, t, library_kind="custom", poly_order=None, n_frequencies=None, threshold=0.08):
    """
    Fit SINDy with a pendulum-aware library by default. (PySINDy Custom/Fourier libs per docs.)
    """
    dt = t[1] - t[0]
    if library_kind == "custom":
        feature_library = make_pendulum_custom_library(include_interactions=True)
    elif library_kind == "fourier":
        feature_library = ps.FourierLibrary(n_frequencies=1)
    else:
        raise ValueError("unknown library_kind")

    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    model.fit(x, t=dt)
    return model

def sindy_predict(model, x0, T, dt):
    t = np.linspace(0.0, T, int(T/dt)+1)
    x = model.simulate(x0, t)
    return t, x
