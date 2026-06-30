"""Tire-grip identification from cornering telemetry — the PINN-B role.

Lap time is set by the tyre's *peak* grip, but a driver mostly operates *below*
the limit, so the telemetry (lateral accel vs slip angle, from an IMU + slip
estimate) rarely samples the peak. Recovering it therefore needs the tyre
*physics*: fit the Pacejka magic-formula structure, which extrapolates the grip
peak from sub-limit data — exactly the role PINN-B plays in the vehicle-dynamics
repo. A plain polynomial fit, with no tyre physics, gets the sampled range right
but mis-reads the peak, which is the number the lap sim actually needs.

Pipeline:  simulated cornering data -> grey-box Pacejka fit -> peak grip mu ->
feed into the lap sim's friction circle -> corrected lap time.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def pacejka_mu(alpha, B, C, D):
    """Normalized lateral force mu(alpha) = Fy/Fz (Pacejka magic formula)."""
    return D * np.sin(C * np.arctan(B * np.asarray(alpha)))


def simulate_corner_data(B_true=11.0, C_true=1.45, D_true=1.30,
                         alpha_max_deg=6.0, n=120, noise=0.04, seed=0):
    """A few laps of cornering: slip angle vs measured grip (Fy/Fz), capped at
    the slip the driver actually used (< the ~8 deg peak), with IMU-level noise."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(rng.uniform(0.3, alpha_max_deg, n))
    mu = pacejka_mu(a, B_true, C_true, D_true) * (1 + rng.normal(0, noise, n))
    return a, mu


def fit_pacejka(alpha, mu):
    """Grey-box: fit the Pacejka (B, C, D) structure -> extrapolates the peak."""
    def resid(p):
        return pacejka_mu(alpha, *p) - mu
    sol = least_squares(resid, x0=[8.0, 1.4, 1.0],
                        bounds=([2, 1.0, 0.5], [25, 2.0, 2.5]))
    return tuple(sol.x)


def peak_grip(B, C, D):
    """Peak mu over slip angle for a fitted Pacejka curve."""
    a = np.linspace(0, np.deg2rad(20), 400)
    return float(pacejka_mu(a, B, C, D).max())


def plain_peak(alpha, mu, deg=4):
    """A physics-free polynomial fit; its peak over the *driven* range is all it
    can claim — it cannot extrapolate past where the data stops."""
    c = np.polyfit(alpha, mu, deg)
    a = np.linspace(0, alpha.max(), 200)
    return float(np.polyval(c, a).max())
