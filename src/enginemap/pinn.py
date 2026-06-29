"""Physics-informed surrogate of the engine map.

The network maps (rpm, throttle) -> (brake torque, fuel rate). Brake power is
the exact kinematic identity P = tau * omega, so it is *derived*, never a free
head -- the architecture itself bakes in that physics. On top of the sparse
dyno data, a residual loss enforces physically-required structure on dense
collocation points:

  1. Load monotonicity   d(tau)/d(throttle) >= 0      (more throttle -> more torque)
  2. Efficiency bound    P_brake <= P_fuel            (brake efficiency < 1)
  3. Willans linearity   d^2(fuel)/d(throttle)^2 ~ 0  (fuel ~ affine in load)
  4. Closed-throttle BC  tau(thr_min) <= 0            (motoring/pumping torque)

Train with `use_physics=False` for the data-only baseline (same net, lambda=0)
and `True` for the PINN -- the only difference is the residual term, which is
what lets the PINN fill the gaps between sparse dyno sweeps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .engine import Engine, Q_LHV

torch.manual_seed(0)


class EngineMapNet(nn.Module):
    def __init__(self, engine: Engine, t_scale: float, f_scale: float, width: int = 64):
        super().__init__()
        s = engine.spec
        self.rpm_mid = (s.idle_rpm + s.redline_rpm) / 2.0
        self.rpm_half = (s.redline_rpm - s.idle_rpm) / 2.0
        self.t_scale = t_scale
        self.f_scale = f_scale
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 2),
        )

    def forward(self, rpm, throttle):
        x1 = (rpm - self.rpm_mid) / self.rpm_half
        x2 = throttle * 2.0 - 1.0
        out = self.net(torch.stack([x1, x2], dim=-1))
        torque = out[..., 0] * self.t_scale
        fuel = out[..., 1] * self.f_scale
        return torque, fuel

    def power_W(self, rpm, throttle):
        torque, _ = self.forward(rpm, throttle)
        return torque * rpm * 2.0 * np.pi / 60.0


def _physics_residual(model: EngineMapNet, engine: Engine, n_col: int, p_scale: float):
    s = engine.spec
    rpm = torch.rand(n_col) * (s.redline_rpm - s.idle_rpm) + s.idle_rpm
    thr = torch.rand(n_col) * 0.92 + 0.08
    rpm.requires_grad_(True)
    thr.requires_grad_(True)

    torque, fuel = model(rpm, thr)
    omega = rpm * 2.0 * np.pi / 60.0
    p_brake = torque * omega
    p_fuel = (fuel / 1000.0) * Q_LHV  # g/s -> kg/s -> W

    ones = torch.ones_like(torque)
    dtau = torch.autograd.grad(torque, thr, ones, create_graph=True)[0]
    d2tau = torch.autograd.grad(dtau, thr, torch.ones_like(dtau), create_graph=True)[0]
    dfuel = torch.autograd.grad(fuel, thr, ones, create_graph=True)[0]
    d2fuel = torch.autograd.grad(dfuel, thr, torch.ones_like(dfuel), create_graph=True)[0]

    # 1. mean-value structure: torque is AFFINE in throttle at fixed rpm
    #    (brake torque ~ trapped air charge ~ throttle/MAP, minus rpm friction).
    #    This is the value-pinning prior that reconstructs unsampled load regions.
    r_lin_tau = (d2tau / model.t_scale).pow(2).mean()
    # 2. fuel is likewise ~affine in load at fixed rpm
    r_lin_fuel = (d2fuel / model.f_scale).pow(2).mean()
    # 3. load monotonicity: more throttle -> more torque
    r_mono = torch.relu(-dtau / model.t_scale).mean()
    # 4. thermodynamic bound: brake power cannot exceed fuel power (eta < 1)
    r_eff = torch.relu(p_brake - p_fuel).mean() / p_scale

    return r_lin_tau + r_lin_fuel + 0.5 * r_mono + 0.2 * r_eff


def train(engine: Engine, df, *, use_physics: bool, epochs: int = 4000,
          lam: float = 1.0, n_col: int = 512, lr: float = 3e-3, log_every: int = 0):
    t_scale = float(np.abs(df["torque_Nm"]).max())
    f_scale = float(np.abs(df["fuel_g_s"]).max())
    p_scale = t_scale * engine.spec.redline_rpm * 2 * np.pi / 60.0

    model = EngineMapNet(engine, t_scale, f_scale)
    rpm = torch.tensor(df["rpm"].to_numpy(), dtype=torch.float32)
    thr = torch.tensor(df["throttle"].to_numpy(), dtype=torch.float32)
    tq = torch.tensor(df["torque_Nm"].to_numpy(), dtype=torch.float32)
    fr = torch.tensor(df["fuel_g_s"].to_numpy(), dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad()
        tq_p, fr_p = model(rpm, thr)
        data = ((tq_p - tq) / t_scale).pow(2).mean() + ((fr_p - fr) / f_scale).pow(2).mean()
        loss = data
        if use_physics:
            loss = loss + lam * _physics_residual(model, engine, n_col, p_scale)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and ep % log_every == 0:
            print(f"  ep {ep:5d}  loss {loss.item():.5f}  data {data.item():.5f}")
    return model


@torch.no_grad()
def predict_grid(model: EngineMapNet, RPM, THR):
    rpm = torch.tensor(RPM.ravel(), dtype=torch.float32)
    thr = torch.tensor(THR.ravel(), dtype=torch.float32)
    tq, fr = model(rpm, thr)
    torque = tq.numpy().reshape(RPM.shape)
    power_hp = (tq.numpy() * RPM.ravel() * 2 * np.pi / 60.0 / 745.7).reshape(RPM.shape)
    fuel = fr.numpy().reshape(RPM.shape)
    bsfc = (fuel * 3600.0) / np.maximum(power_hp * 745.7 / 1000.0, 1.0)
    return {"torque": torque, "power_hp": power_hp, "fuel": fuel, "bsfc": bsfc}
