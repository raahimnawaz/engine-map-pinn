"""Mean-value physics model of a naturally-aspirated V12, calibrated to the
Lamborghini Aventador SVJ (6.5 L, ~759 hp @ 8500 rpm, ~531 lb-ft @ 6750 rpm),
with an optional forced-induction ("twin-turbo build") variant.

This is the ground-truth *plant*: a first-principles brake-torque + fuel model
over the (rpm, throttle) operating space. The PINN learns a surrogate of it
from sparse, noisy samples; the physics relations here are also what the PINN's
residual loss enforces.

Four-stroke relations used throughout (n_r = 2 crank revolutions per power
stroke):

    torque   tau   = BMEP * Vd / (4*pi)                      [N*m]
    power    P     = tau * omega,   omega = 2*pi*rpm/60       [W]
    BMEP           = throttle * VE(rpm) * imep_ref - FMEP(rpm)
    fuel power     = indicated_power / eta_ind
    BSFC           = mdot_fuel / P_brake                      [g/kWh after conv]

VE(rpm) (volumetric efficiency) is a smooth hump; FMEP(rpm) (friction mean
effective pressure) grows with engine speed. The few shape constants are
auto-calibrated so the wide-open-throttle curve reproduces the SVJ's published
peak torque and peak power at the right rpm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# unit helpers ---------------------------------------------------------------
HP_PER_W = 1.0 / 745.7
NM_PER_LBFT = 1.355818
Q_LHV = 43.4e6  # gasoline lower heating value [J/kg]
P_ATM = 1.01325e5  # [Pa]


@dataclass
class EngineSpec:
    name: str
    displacement_L: float
    n_cylinders: int
    redline_rpm: float
    idle_rpm: float
    # published validation targets (wide-open throttle)
    peak_power_hp: float
    peak_power_rpm: float
    peak_torque_lbft: float
    peak_torque_rpm: float
    # forced-induction ("twin-turbo build") config; None => naturally aspirated
    boost: "BoostConfig | None" = None


@dataclass
class BoostConfig:
    """Intercooled twin-turbo bolt-on. A physically-plausible *what-if*, not
    real dyno data."""
    max_pressure_ratio: float = 2.3   # ~19 psi of boost over atmospheric
    spool_rpm: float = 3800.0          # rpm at half-spool
    spool_width: float = 900.0         # logistic width
    intercooler_eff: float = 0.75      # fraction of ideal density recovery


# SVJ naturally aspirated 6.5 L V12
SVJ = EngineSpec(
    name="Aventador SVJ (NA V12)",
    displacement_L=6.498,
    n_cylinders=12,
    redline_rpm=8700.0,
    idle_rpm=900.0,
    peak_power_hp=759.0,
    peak_power_rpm=8500.0,
    peak_torque_lbft=531.0,
    peak_torque_rpm=6750.0,
)

# Same block with an intercooled twin-turbo bolt-on ("Gintani-style" build).
SVJ_TT = EngineSpec(
    name="Aventador SVJ — twin-turbo build (what-if)",
    displacement_L=6.498,
    n_cylinders=12,
    redline_rpm=8200.0,
    idle_rpm=900.0,
    peak_power_hp=759.0,        # NA targets used only to calibrate the base block;
    peak_power_rpm=8500.0,      # boost is applied on top of the calibrated NA map.
    peak_torque_lbft=531.0,
    peak_torque_rpm=6750.0,
    boost=BoostConfig(),
)


class Engine:
    """Calibrated mean-value engine. Construct from an EngineSpec; call
    torque()/power()/bsfc() over (rpm, throttle)."""

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self.Vd = spec.displacement_L * 1e-3  # [m^3]
        # internal parameters, set by _calibrate
        self.imep_ref = 12.5e5      # peak gross IMEP scale [Pa]
        self.ve_peak_rpm = spec.peak_torque_rpm
        self.ve_width = 2600.0
        self.fmep = np.array([0.4e5, 6.0e-3 * 1e5 / 1000, 0.0])  # placeholder
        self.eta_ind_max = 0.39     # best indicated efficiency (NA gasoline)
        self._calibrate()

    # --- physics primitives -------------------------------------------------
    @staticmethod
    def omega(rpm):
        return np.asarray(rpm) * 2.0 * np.pi / 60.0

    def _ve(self, rpm):
        # smooth volumetric-efficiency hump (normalised to 1.0 at its peak)
        return np.exp(-0.5 * ((np.asarray(rpm) - self.ve_peak_rpm) / self.ve_width) ** 2)

    def _fmep(self, rpm):
        rpm = np.asarray(rpm, dtype=float)
        a0, a1, a2 = self.fmep
        return a0 + a1 * rpm + a2 * rpm ** 2  # [Pa]

    def _boost_factor(self, rpm):
        b = self.spec.boost
        if b is None:
            return np.ones_like(np.asarray(rpm, dtype=float))
        spool = 1.0 / (1.0 + np.exp(-(np.asarray(rpm, dtype=float) - b.spool_rpm) / b.spool_width))
        pr = 1.0 + (b.max_pressure_ratio - 1.0) * spool
        # intercooled charge-density gain over atmospheric
        return 1.0 + b.intercooler_eff * (pr - 1.0)

    # --- public map ---------------------------------------------------------
    def bmep(self, rpm, throttle):
        rpm = np.asarray(rpm, dtype=float)
        throttle = np.asarray(throttle, dtype=float)
        gross = throttle * self._ve(rpm) * self.imep_ref * self._boost_factor(rpm)
        return gross - self._fmep(rpm)

    def torque(self, rpm, throttle):
        """Brake torque [N*m]."""
        return self.bmep(rpm, throttle) * self.Vd / (4.0 * np.pi)

    def power(self, rpm, throttle):
        """Brake power [W]."""
        return self.torque(rpm, throttle) * self.omega(rpm)

    def power_hp(self, rpm, throttle):
        return self.power(rpm, throttle) * HP_PER_W

    def _eta_ind(self, rpm, throttle):
        # indicated efficiency: best at mid-high rpm and high (not full) load;
        # this is what carves the BSFC island.
        rpm = np.asarray(rpm, dtype=float)
        throttle = np.asarray(throttle, dtype=float)
        rpm_term = 1.0 - 0.25 * ((rpm - self.ve_peak_rpm) / 4200.0) ** 2
        load_term = 1.0 - 0.20 * (throttle - 0.85) ** 2 / 0.85 ** 2
        return self.eta_ind_max * np.clip(rpm_term, 0.4, 1.0) * np.clip(load_term, 0.5, 1.0)

    def fuel_rate(self, rpm, throttle):
        """Fuel mass flow [g/s]. Indicated power = brake + friction power."""
        fric_torque = self._fmep(rpm) * self.Vd / (4.0 * np.pi)
        p_friction = np.maximum(fric_torque * self.omega(rpm), 0.0)
        p_brake = np.maximum(self.power(rpm, throttle), 1.0)
        p_indicated = p_brake + p_friction
        p_fuel = p_indicated / self._eta_ind(rpm, throttle)
        return p_fuel / Q_LHV * 1000.0  # [g/s]

    def bsfc(self, rpm, throttle):
        """Brake-specific fuel consumption [g/kWh]."""
        p_brake_kw = np.maximum(self.power(rpm, throttle), 1.0) / 1000.0
        mdot_g_per_h = self.fuel_rate(rpm, throttle) * 3600.0
        return mdot_g_per_h / p_brake_kw

    # --- calibration --------------------------------------------------------
    def wot_curve(self, n=400):
        rpm = np.linspace(self.spec.idle_rpm, self.spec.redline_rpm, n)
        return rpm, self.torque(rpm, 1.0), self.power_hp(rpm, 1.0)

    def peaks(self):
        rpm, tq, hp = self.wot_curve(2000)
        i_t, i_p = int(np.argmax(tq)), int(np.argmax(hp))
        return {
            "peak_torque_Nm": float(tq[i_t]), "peak_torque_rpm": float(rpm[i_t]),
            "peak_power_hp": float(hp[i_p]), "peak_power_rpm": float(rpm[i_p]),
        }

    def _calibrate(self):
        """Auto-tune (imep_ref, ve_width, fmep slope) so the WOT curve hits the
        published peak torque/power at the right rpm. Boost (if any) is layered
        on the NA-calibrated block, so calibrate against the NA targets."""
        tgt_tq_Nm = self.spec.peak_torque_lbft * NM_PER_LBFT
        tgt_pw_W = self.spec.peak_power_hp / HP_PER_W
        boost_saved = self.spec.boost
        self.spec.boost = None  # calibrate the base (NA) block

        def residuals(p):
            self.imep_ref, self.ve_width, fmep_slope = p
            # friction: small offset + linear in rpm (Pa); quadratic kept off
            self.fmep = np.array([0.35e5, fmep_slope, 0.0])
            self.ve_peak_rpm = self.spec.peak_torque_rpm
            rpm = np.linspace(self.spec.idle_rpm, self.spec.redline_rpm, 800)
            tq = self.torque(rpm, 1.0)
            hp_w = self.power(rpm, 1.0)
            i_t, i_p = int(np.argmax(tq)), int(np.argmax(hp_w))
            return [
                (tq[i_t] - tgt_tq_Nm) / tgt_tq_Nm,
                (hp_w[i_p] - tgt_pw_W) / tgt_pw_W,
                (rpm[i_t] - self.spec.peak_torque_rpm) / 1000.0,
                (rpm[i_p] - self.spec.peak_power_rpm) / 1000.0,
            ]

        sol = least_squares(
            residuals, x0=[12.5e5, 2600.0, 8.0],
            bounds=([8e5, 1500.0, 0.0], [16e5, 4500.0, 40.0]), max_nfev=400,
        )
        self.imep_ref, self.ve_width, fmep_slope = sol.x
        self.fmep = np.array([0.35e5, fmep_slope, 0.0])
        self.spec.boost = boost_saved  # restore boost for subsequent queries


def grid(engine: Engine, n_rpm=120, n_thr=120, thr_min=0.08):
    """Dense (rpm, throttle) mesh over the operating envelope."""
    rpm = np.linspace(engine.spec.idle_rpm, engine.spec.redline_rpm, n_rpm)
    thr = np.linspace(thr_min, 1.0, n_thr)
    RPM, THR = np.meshgrid(rpm, thr)
    return RPM, THR
