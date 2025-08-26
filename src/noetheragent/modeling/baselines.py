import numpy as np
from dataclasses import dataclass

@dataclass
class LinearizedPendulum:
    g: float = 9.81
    L: float = 1.0

    def f(self, x):
        theta, omega = x[..., 0], x[..., 1]
        dtheta = omega
        domega = -(self.g / self.L) * theta  # small-angle approx
        return np.stack([dtheta, domega], axis=-1)

def simulate(model: LinearizedPendulum, theta0: float, omega0: float, T: float, dt: float):
    n = int(T / dt) + 1
    t = np.linspace(0.0, T, n)
    x = np.zeros((n, 2))
    x[0] = [theta0, omega0]
    for k in range(n - 1):
        s = x[k]
        k1 = model.f(s)
        k2 = model.f(s + 0.5 * dt * k1)
        k3 = model.f(s + 0.5 * dt * k2)
        k4 = model.f(s + dt * k3)
        x[k + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, x
