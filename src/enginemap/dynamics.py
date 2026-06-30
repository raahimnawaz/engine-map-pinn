"""Dynamic single-track (bicycle) vehicle model with Pacejka tyres.

This is the model the MPC actually drives — unlike the quasi-steady-state lap
sim (a point mass always exactly at the grip limit), this carries real states
(lateral velocity, yaw rate) and transient dynamics, so a controller has to
*earn* its lap time rather than assume it.

State  X = [x, y, psi, vx, vy, r]   (position, heading, body-frame velocities, yaw rate)
Input  u = [delta, Fx]              (front steer angle, rear longitudinal force)

The Pacejka "magic formula" gives lateral tyre force from slip angle; the same
tyre model is what the vehicle-dynamics repo estimates from real lateral-accel
telemetry, so its fitted parameters drop straight in here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_AIR = 1.225
G = 9.81


@dataclass
class BicycleModel:
    m: float = 1620.0       # mass [kg]
    Iz: float = 2400.0      # yaw inertia [kg m^2]
    lf: float = 1.4         # CG to front axle [m]
    lr: float = 1.4         # CG to rear axle [m]
    cda: float = 0.78       # drag area
    cla: float = 1.5        # downforce area
    # Pacejka lateral (B stiffness, C shape, D peak ~ mu)
    Bf: float = 11.0
    Cf: float = 1.45
    Df: float = 1.45
    Br: float = 11.0
    Cr: float = 1.45
    Dr: float = 1.50        # slightly more rear grip for stability

    def _tyre(self, alpha, Fz, B, C, D):
        return Fz * D * np.sin(C * np.arctan(B * alpha))

    def continuous(self, X, u):
        x, y, psi, vx, vy, r = X
        delta, Fx = u
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
