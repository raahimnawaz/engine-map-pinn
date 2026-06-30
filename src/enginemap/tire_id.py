"""Tire-grip identification — mirroring the vehicle-dynamics PINN-B method.

Lap time is set by the tyre's *peak* grip, but a driver mostly stays below the
limit, so cornering telemetry (lateral accel from an IMU -> Fy = m*a_lat ->
mu = Fy/Fz, against slip angle) rarely samples the peak. Recovering it needs the
tyre *physics*. Two estimators, the same family as the VD repo's PINN-B:

  * fit_pacejka : GREY-BOX -- fit the analytic Pacejka (B, C, D) scalars
                  (the counterpart of VD's PacejkaNet).
  * MuNet       : FREE-FORM MLP alpha -> mu, trained with a CONCAVITY prior
                  (penalise positive d2mu/dalpha2 only) so it forms the
                  rise-peak-fall arch *without* being told Pacejka -- the key
                  structural-physics insight from VD's MuNet.

A physics-free polynomial fit, by contrast, is systematically low: it cannot
extrapolate the peak past the driven slip range.

Scope note: VD's PINN-B is the *rigorous* version -- it recovers mu from
deceleration trajectories via an ODE-residual loss (because longitudinal tyre
force isn't directly measurable) and is brake-aware. This is the lateral,
steady-cornering counterpart, where mu = m*a_lat/Fz is read more directly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

torch.manual_seed(0)


def pacejka_mu(alpha, B, C, D):
    """Normalized lateral force mu(alpha) = Fy/Fz (Pacejka magic formula)."""
    return D * np.sin(C * np.arctan(B * np.asarray(alpha)))


def simulate_corner_data(B_true=11.0, C_true=1.45, D_true=1.30,
                         alpha_max_deg=7.0, n=200, noise=0.03, seed=0):
    """A few laps of cornering: slip angle vs measured grip (Fy/Fz from IMU
    lateral accel), capped at the slip the driver used (< the ~10 deg peak)."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(rng.uniform(0.3, alpha_max_deg, n))
    mu = pacejka_mu(a, B_true, C_true, D_true) * (1 + rng.normal(0, noise, n))
    return a, mu


# --- grey-box: fit the Pacejka structure (== VD PacejkaNet) -----------------
def fit_pacejka(alpha, mu):
    def resid(p):
        return pacejka_mu(alpha, *p) - mu
    sol = least_squares(resid, x0=[8.0, 1.4, 1.0], bounds=([2, 1.0, 0.5], [25, 2.0, 2.5]))
    return tuple(sol.x)


# --- free-form: MLP alpha -> mu with a concavity prior (== VD MuNet) ---------
class MuNet(nn.Module):
    """alpha -> mu. The output is capped at mu_cap*sigmoid so the net can both
    reach and fall back from the peak; mu_cap must sit above the tyre's peak
    grip (race tyres exceed mu=1, so 1.1 -- VD's value for mu~0.9 tyres -- is
    too low here)."""

    def __init__(self, hidden: int = 32, mu_cap: float = 1.8):
        super().__init__()
        self.mu_cap = mu_cap
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, a):
        return self.mu_cap * torch.sigmoid(self.net(a.view(-1, 1))).squeeze(-1)


def train_munet(alpha, mu, *, epochs: int = 1500, lam_concave: float = 1.0,
                lam_zero: float = 2.0, lr: float = 5e-3):
    """Data fit + concavity prior (d2mu/dalpha2 <= 0) + mu(0)=0 boundary.
    The concavity prior is what lets the free-form net form a single-arch peak
    and *fall back* from it without being told the Pacejka form."""
    net = MuNet()
    a = torch.tensor(alpha, dtype=torch.float32)
    m = torch.tensor(mu, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    a_grid = torch.linspace(0, np.deg2rad(16), 80, requires_grad=True)
    for _ in range(epochs):
        opt.zero_grad()
        data = ((net(a) - m) ** 2).mean()
        mug = net(a_grid)
        d1 = torch.autograd.grad(mug.sum(), a_grid, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), a_grid, create_graph=True)[0]
        concave = torch.relu(d2).pow(2).mean()           # penalise positive curvature only
        zero = net(torch.zeros(1)).pow(2).mean()         # mu(0) = 0
        (data + lam_concave * concave + lam_zero * zero).backward()
        opt.step()
    return net


@torch.no_grad()
def munet_curve(net, alpha):
    return net(torch.tensor(np.asarray(alpha), dtype=torch.float32)).numpy()


def peak_grip(*pacejka_or_net, alpha_max_deg=20.0):
    """Peak mu over slip. Accepts (B, C, D) or a trained MuNet."""
    a = np.linspace(0, np.deg2rad(alpha_max_deg), 400)
    if len(pacejka_or_net) == 3:
        return float(pacejka_mu(a, *pacejka_or_net).max())
    return float(munet_curve(pacejka_or_net[0], a).max())


def plain_peak(alpha, mu, deg=4):
    """Physics-free polynomial fit; its peak over the *driven* range is all it
    can claim -- it cannot extrapolate past where the data stops."""
    c = np.polyfit(alpha, mu, deg)
    a = np.linspace(0, alpha.max(), 200)
    return float(np.polyval(c, a).max())
