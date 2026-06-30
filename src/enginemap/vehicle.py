"""Vehicle model for the lap simulator — the Aventador SVJ, using the engine's
torque curve to build a tractive-force-vs-speed envelope through the gearbox.

The tractive force at a given road speed is the best gear's wheel force:
    rpm(v, gear) = v / r_wheel * gear * final * 60/(2*pi)
    F(v, gear)   = T_engine(rpm) * gear * final * eff / r_wheel   (rpm in band)
    F(v)         = max over gears of F(v, gear)
This is exactly how the dyno map drives on-track acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Engine

RHO_AIR = 1.225  # kg/m^3
G = 9.81


@dataclass
class Vehicle:
    engine: Engine
    mass: float = 1620.0          # kg incl. driver
    cda: float = 0.78             # drag area Cd*A [m^2]
    cla: float = 1.5              # downforce area Cl*A [m^2]
    mu: float = 1.45              # tyre friction (track rubber)
    crr: float = 0.012            # rolling resistance
    r_wheel: float = 0.345        # m
    final_drive: float = 3.08
    driveline_eff: float = 0.88
    power_scale: float = 1.0      # scale engine output (for "more power?" sweeps)
    # SVJ 7-speed ISR gearbox (representative ratios)
    gears: tuple = (3.91, 2.44, 1.81, 1.46, 1.19, 0.97, 0.81)

    def rpm(self, v, gear_ratio):
        return v / self.r_wheel * gear_ratio * self.final_drive * 60.0 / (2 * np.pi)

    def tractive_force(self, v):
        """Best-gear wheel force at speed v [N] (vectorized over v)."""
        v = np.atleast_1d(np.asarray(v, dtype=float))
        best = np.zeros_like(v)
        s = self.engine.spec
        for g in self.gears:
            rpm = self.rpm(v, g)
            valid = (rpm >= s.idle_rpm) & (rpm <= s.redline_rpm)
            tq = self.engine.torque(np.clip(rpm, s.idle_rpm, s.redline_rpm), 1.0)
            f = tq * g * self.final_drive * self.driveline_eff / self.r_wheel
            f = np.where(valid, f, 0.0)
            best = np.maximum(best, f)
        return best * self.power_scale

    def drag(self, v):
        return 0.5 * RHO_AIR * self.cda * v * v

    def downforce(self, v):
        return 0.5 * RHO_AIR * self.cla * v * v

    def rolling(self, v):
        return self.crr * self.mass * G

    def grip_force(self, v):
        """Total tyre grip available [N] = mu * normal load (weight + downforce)."""
        return self.mu * (self.mass * G + self.downforce(v))
