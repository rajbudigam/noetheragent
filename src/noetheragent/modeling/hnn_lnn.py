from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# ---------- helpers ----------
def rk4_step(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate_vector_field(f, x0, T, dt):
    n = int(T / dt) + 1
    t = np.linspace(0.0, T, n)
    x = np.zeros((n, 2), dtype=np.float64)
    x[0] = x0
    for k in range(n - 1):
        step = f(torch.from_numpy(x[k].astype(np.float64))).numpy()
        # Defensive shape check
        if step is None or np.asarray(step).shape != (2,):
            raise ValueError(f"Vector field output at step {k} has shape {np.asarray(step).shape}, expected (2,)")
        x[k + 1] = rk4_step(lambda s: f(torch.from_numpy(s.astype(np.float64))).numpy(), x[k], dt)
    return t, x

# ---------- HNN ----------
class HNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (...,2) -> scalar H
        return self.net(x)

@dataclass
class HNNConfig:
    hidden: int = 64
    epochs: int = 300
    lr: float = 1e-3
    batch: int = 256
    seed: int = 42

class HNN:
    def __init__(self, cfg: HNNConfig):
        torch.manual_seed(cfg.seed)
        self.model = HNet(hidden=cfg.hidden).double()
        self.cfg = cfg

    def vector_field(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,2) [theta, p] -- here we use p := omega as a practical baseline
        x = x.requires_grad_(True)
        H = self.model(x)                             # (...,1)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dtheta_dt = dH[..., 1]                        # ∂H/∂p
        domega_dt = -dH[..., 0]                       # -∂H/∂θ
        result = torch.stack([dtheta_dt, domega_dt], dim=-1)
        # Defensive shape check
        if result.shape[-1] != 2:
            raise ValueError(f"HNN.vector_field output shape {result.shape} is not (..., 2)")
        if result.ndim == 1:
            result = result.unsqueeze(0)
        return result.squeeze(0) if result.shape[0] == 1 else result

    def fit(self, X: np.ndarray, Xdot: np.ndarray):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        X_t = torch.from_numpy(X).double()
        Y_t = torch.from_numpy(Xdot).double()
        for _ in range(cfg.epochs):
            idx = torch.randint(0, X_t.shape[0], (min(cfg.batch, X_t.shape[0]),))
            xb, yb = X_t[idx], Y_t[idx]
            pred = self.vector_field(xb)
            if pred.shape != yb.shape:
                raise ValueError(f"HNN.fit: prediction shape {pred.shape} does not match target {yb.shape}")
            loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def simulate(self, x0: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        def vf_np(s_np):
            s_t = torch.from_numpy(s_np.astype(np.float64)).double()
            out = self.vector_field(s_t).detach().numpy()
            if out.shape == (1,2):
                out = out[0]
            if out.shape != (2,):
                raise ValueError(f"HNN.simulate: vector field returned shape {out.shape}, expected (2,)")
            return out
        return simulate_vector_field(lambda s: torch.from_numpy(vf_np(s)), x0.astype(np.float64), T, dt)

# ---------- Minimal LNN ----------
class UNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, theta):  # (...,1) -> scalar U(theta)
        return self.net(theta)

@dataclass
class LNNConfig:
    hidden: int = 64
    epochs: int = 300
    lr: float = 1e-3
    batch: int = 256
    seed: int = 123

class LNNMinimal:
    """
    L(θ, θdot) = 0.5*θdot^2 - U(θ) with learnable U.
    Euler-Lagrange => θddot = -dU/dθ
    """
    def __init__(self, cfg: LNNConfig):
        torch.manual_seed(cfg.seed)
        self.U = UNet(hidden=cfg.hidden).double()
        self.cfg = cfg

    def ddot(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.requires_grad_(True)
        U = self.U(theta)
        dU = torch.autograd.grad(U.sum(), theta, create_graph=True)[0]
        return -dU  # θddot

    def vector_field(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[..., :1]
        omega = x[..., 1:2]
        dd = self.ddot(theta)
        result = torch.cat([omega, dd], dim=-1)
        # Defensive shape check
        if result.shape[-1] != 2:
            raise ValueError(f"LNNMinimal.vector_field output shape {result.shape} is not (..., 2)")
        if result.ndim == 1:
            result = result.unsqueeze(0)
        return result.squeeze(0) if result.shape[0] == 1 else result

    def fit(self, X: np.ndarray, Xdot: np.ndarray):
        """
        X = [theta, omega], Xdot = [theta_dot, omega_dot]; we match omega_dot.
        """
        cfg = self.cfg
        opt = torch.optim.Adam(self.U.parameters(), lr=cfg.lr)
        X_t = torch.from_numpy(X).double()
        dd_truth = torch.from_numpy(Xdot[:, 1:2]).double()   # omega_dot
        for _ in range(cfg.epochs):
            idx = torch.randint(0, X_t.shape[0], (min(cfg.batch, X_t.shape[0]),))
            thb = X_t[idx, :1]
            pred_dd = self.ddot(thb)
            if pred_dd.shape != dd_truth[idx].shape:
                raise ValueError(f"LNNMinimal.fit: prediction shape {pred_dd.shape} does not match target {dd_truth[idx].shape}")
            loss = ((pred_dd - dd_truth[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def simulate(self, x0: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        def vf_np(s_np):
            s_t = torch.from_numpy(s_np.astype(np.float64)).double()
            out = self.vector_field(s_t).detach().numpy()
            if out.shape == (1,2):
                out = out[0]
            if out.shape != (2,):
                raise ValueError(f"LNNMinimal.simulate: vector field returned shape {out.shape}, expected (2,)")
            return out
        return simulate_vector_field(lambda s: torch.from_numpy(vf_np(s)), x0.astype(np.float64), T, dt)
