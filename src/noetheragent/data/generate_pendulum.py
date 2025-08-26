import numpy as np
from dataclasses import dataclass

@dataclass
class PendulumParams:
    g: float = 9.81
    L: float = 1.0

def pendulum_rhs(state, t, params: PendulumParams):
    theta, omega = state
    dtheta = omega
    domega = -(params.g / params.L) * np.sin(theta)
    return np.array([dtheta, domega])

def simulate_pendulum(theta0: float, omega0: float, T: float, dt: float, params: PendulumParams):
    n = int(T / dt) + 1
    t = np.linspace(0.0, T, n)
    x = np.zeros((n, 2))
    x[0] = [theta0, omega0]
    for k in range(n - 1):
        s = x[k]
        # RK4
        k1 = pendulum_rhs(s, t[k], params)
        k2 = pendulum_rhs(s + 0.5 * dt * k1, t[k] + 0.5 * dt, params)
        k3 = pendulum_rhs(s + 0.5 * dt * k2, t[k] + 0.5 * dt, params)
        k4 = pendulum_rhs(s + dt * k3, t[k] + dt, params)
        x[k + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, x

def stack_trajectories(thetas0, omegas0, T, dt, params: PendulumParams):
    """Return concatenated trajectories and derivatives."""
    ts, Xs, Xdots = [], [], []
    for th0, om0 in zip(thetas0, omegas0):
        t, X = simulate_pendulum(th0, om0, T, dt, params)
        # finite difference for derivatives
        Xdot = np.zeros_like(X)
        Xdot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
        Xdot[0] = (X[1] - X[0]) / dt
        Xdot[-1] = (X[-1] - X[-2]) / dt
        ts.append(t); Xs.append(X); Xdots.append(Xdot)
    return np.concatenate(ts), np.vstack(Xs), np.vstack(Xdots)
