import numpy as np
import pysindy as ps

def fit_sindy(x, t, library_kind="fourier", poly_order=2, n_frequencies=1, threshold=0.05):
    """
    Fit SINDy to pendulum data with a library that includes sin(theta).
    """
    dt = t[1] - t[0]
    if library_kind == "fourier":
        lib = ps.FourierLibrary(n_frequencies=n_frequencies)  # includes sin/cos
    elif library_kind == "custom":
        lib = ps.CustomLibrary(library_functions=[
            lambda x: x,                       # theta
            lambda x: x**2,
            lambda x: np.sin(x),               # sin(theta)
            lambda x: np.cos(x),               # cos(theta)
        ], function_names=[
            lambda s: s,
            lambda s: f"{s}^2",
            lambda s: f"sin({s})",
            lambda s: f"cos({s})",
        ])
    else:
        raise ValueError("unknown library")

    # The state is [theta, omega]; we want theta' and omega'
    feature_library = ps.feature_library.GeneralizedLibrary([lib, lib])
    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    model.fit(x, t=dt)
    return model

def sindy_predict(model, x0, T, dt):
    t = np.linspace(0.0, T, int(T/dt)+1)
    x = model.simulate(x0, t)
    return t, x
