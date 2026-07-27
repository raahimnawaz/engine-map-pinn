"""Dynamic single-track (bicycle) vehicle model with Pacejka tyres.

This is the model the MPC actually drives — unlike the quasi-steady-state lap
sim (a point mass always exactly at the grip limit), this carries real states
(lateral velocity, yaw rate) and transient dynamics, so a controller has to
*earn* its lap time rather than assume it.

State  X = [x, y, psi, vx, vy, r]   (position, heading, body-frame velocities, yaw rate)
Input  u = [delta, Fx_cmd]          (front steer angle, COMMANDED longitudinal force)

`Fx_cmd` is what the controller/driver asks for; the plant only applies what the
tyres and engine can actually deliver. A **friction circle** shares the tyre's
grip between cornering and braking/accel (`sqrt(Fx^2 + Fy^2) <= mu*Fz`), so you
cannot brake and corner past the tyre; and, given an engine `tractive` curve,
drive force is capped by the engine through the gearbox. Without this the plant
was unphysical (it could pull ~2 g combined, or 11 kN of drive at 300 km/h),
which is what made a controller impossible to stabilise on it.

The Pacejka "magic formula" gives lateral tyre force from slip angle; the same
tyre model is what the vehicle-dynamics repo estimates from real lateral-accel
telemetry, so its fitted parameters drop straight in here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import GRIP_BALANCE, MU_TYRE, RHO_AIR, G

__all__ = ["BicycleModel", "RHO_AIR", "G"]


@dataclass
class BicycleModel:
    m: float = 1620.0       # mass [kg]
    Iz: float = 2400.0      # yaw inertia [kg m^2]
    lf: float = 1.4         # CG to front axle [m]
    lr: float = 1.4         # CG to rear axle [m]
    cda: float = 0.78       # drag area
    cla: float = 1.5        # downforce area
    # Pacejka lateral (B stiffness, C shape, D peak ~ mu). The axle peaks straddle
    # the shared MU_TYRE, so peak_mu below equals the QSS sim's mu -- one car.
    Bf: float = 11.0
    Cf: float = 1.45
    Df: float = MU_TYRE - GRIP_BALANCE
    Br: float = 11.0
    Cr: float = 1.45
    Dr: float = MU_TYRE + GRIP_BALANCE   # more rear grip -> stable at the limit
    # optional engine tractive-force curve  vx -> max drive force [N]; caps drive
    # (the QSS Vehicle supplies this). None => drive limited only by tyre grip.
    tractive: Callable[[float], float] | None = None

    @property
    def peak_mu(self) -> float:
        """Effective peak grip: with equal axle loads the car's maximum lateral
        force is the mean of the axle peaks. This is the same number the QSS sim
        uses as `Vehicle.mu`, and it stays correct if Df/Dr are later replaced by
        a tyre fit."""
        return 0.5 * (self.Df + self.Dr)

    def _tyre(self, alpha, Fz, B, C, D):
        return Fz * D * np.sin(C * np.arctan(B * alpha))

    def _limit_fx(self, Fx_cmd, Fy, Fz, vx):
        """Longitudinal force the tyres and engine can actually deliver.

        Friction circle: total tyre force (cornering Fy + longitudinal Fx) cannot
        exceed grip mu*Fz, so whatever grip cornering leaves is available for
        brake/drive. Drive is additionally capped by the engine.
        """
        grip = self.peak_mu * Fz
        fx_grip = np.sqrt(max(grip * grip - Fy * Fy, 0.0))
        hi = fx_grip                                # drive: limited by grip ...
        if self.tractive is not None:
            hi = min(hi, max(float(np.ravel(self.tractive(vx))[0]), 0.0))  # ... and the engine
        return min(max(Fx_cmd, -fx_grip), hi)       # brake: limited by grip

    def continuous(self, X, u):
        x, y, psi, vx, vy, r = X
        delta, Fx_cmd = u
        vx = max(vx, 1.0)  # avoid singularity at standstill
        # axle vertical loads: static split + aero downforce
        Fz_aero = 0.5 * RHO_AIR * self.cla * vx * vx
        Fz = self.m * G + Fz_aero
        Fzf = Fz * self.lr / (self.lf + self.lr)
        Fzr = Fz * self.lf / (self.lf + self.lr)
        # slip angles
        alpha_f = delta - np.arctan2(vy + self.lf * r, vx)
        alpha_r = -np.arctan2(vy - self.lr * r, vx)
        Fyf = self._tyre(alpha_f, Fzf, self.Bf, self.Cf, self.Df)
        Fyr = self._tyre(alpha_r, Fzr, self.Br, self.Cr, self.Dr)
        # longitudinal force is limited by the tyre (friction circle, sharing grip
        # with cornering) and the engine -- it is NOT a free input.
        Fy = Fyf * np.cos(delta) + Fyr
        Fx = self._limit_fx(Fx_cmd, Fy, Fz, vx)
        drag = 0.5 * RHO_AIR * self.cda * vx * vx
        # body-frame dynamics
        ax = (Fx - Fyf * np.sin(delta) - drag) / self.m + vy * r
        ay = (Fyf * np.cos(delta) + Fyr) / self.m - vx * r
        r_dot = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz
        return np.array([
            vx * np.cos(psi) - vy * np.sin(psi),
            vx * np.sin(psi) + vy * np.cos(psi),
            r, ax, ay, r_dot,
        ])

    def step(self, X, u, dt):
        k1 = self.continuous(X, u)
        k2 = self.continuous(X + 0.5 * dt * k1, u)
        k3 = self.continuous(X + 0.5 * dt * k2, u)
        k4 = self.continuous(X + dt * k3, u)
        return X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
