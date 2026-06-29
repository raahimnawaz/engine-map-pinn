"""Generate a sparse, noisy 'dyno' dataset from the physics engine.

Mimics how an engine is actually mapped on a dynamometer: a handful of
steady-state wide-open and part-throttle sweeps, not a dense grid. The PINN
must reconstruct the full (rpm, throttle) map from these sparse samples; the
dense physics grid is held out as ground truth for evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Engine


def sample_dyno(engine: Engine, throttles=(0.5, 0.7, 0.85, 1.0), pts_per_sweep: int = 8,
                noise_torque_frac: float = 0.03, noise_fuel_frac: float = 0.04,
                seed: int = 0) -> pd.DataFrame:
    """A few constant-throttle rpm sweeps (as a real dyno session would run),
    with multiplicative measurement noise on torque and fuel flow.

    By default only mid-to-high throttle is sampled (0.5-1.0): the low-load
    region is left UNSAMPLED on purpose, so reconstructing it tests whether a
    model has learned the engine's physics rather than just interpolating data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for thr in throttles:
        rpm = np.linspace(engine.spec.idle_rpm + 300, engine.spec.redline_rpm - 200,
                          pts_per_sweep)
        rpm = rpm + rng.normal(0, 60, rpm.shape)  # rpm jitter between holds
        for r in rpm:
            tq = float(engine.torque(r, thr))
            fr = float(engine.fuel_rate(r, thr))
            rows.append({
                "rpm": float(r),
                "throttle": float(thr),
                "torque_Nm": tq * (1 + rng.normal(0, noise_torque_frac)),
                "fuel_g_s": fr * (1 + rng.normal(0, noise_fuel_frac)),
            })
    return pd.DataFrame(rows)


def dense_truth(engine: Engine, n_rpm: int = 120, n_thr: int = 120,
                thr_min: float = 0.08) -> dict:
    """Dense ground-truth map for evaluation/plots (no noise)."""
    rpm = np.linspace(engine.spec.idle_rpm, engine.spec.redline_rpm, n_rpm)
    thr = np.linspace(thr_min, 1.0, n_thr)
    RPM, THR = np.meshgrid(rpm, thr)
    return {
        "rpm": rpm, "thr": thr, "RPM": RPM, "THR": THR,
        "torque": engine.torque(RPM, THR),
        "power_hp": engine.power_hp(RPM, THR),
        "bsfc": engine.bsfc(RPM, THR),
    }
