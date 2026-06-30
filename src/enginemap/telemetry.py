"""Telemetry sources — one schema, two backends.

A telemetry source just *produces operating-point rows*; it never predicts.
Columns: time_s, rpm, throttle, torque_Nm, fuel_g_s. The PINN consumes these
exactly like the dyno data — so a simulated drive and a real OBD-II log are
interchangeable inputs.

  SimulatedOBD  drives the physics engine along a synthetic drive cycle
                (idle/city/highway/pulls). Produces the *non-uniform* coverage
                a real OBD log has: lots of low-load cruise, rare WOT.
  RealOBD       reads an ELM327/OBDLink adapter via python-obd (untested
                without hardware; see notes). Same output schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Engine


# --- simulated drive-cycle backend -----------------------------------------
# (segment, weight, rpm_lo, rpm_hi, thr_lo, thr_hi) -- weights set how much of
# a normal drive is spent in each regime (cruise dominates; WOT is rare).
_SEGMENTS = [
    ("idle",     0.10,  850, 1000, 0.04, 0.08),
    ("city",     0.40, 1400, 3000, 0.10, 0.32),
    ("highway",  0.34, 2400, 3600, 0.14, 0.34),
    ("pull",     0.16, 2200, 7800, 0.55, 1.00),  # the rare wide-open acceleration
]


class SimulatedOBD:
    def __init__(self, engine: Engine, hz: float = 5.0,
                 noise_torque_frac: float = 0.03, noise_fuel_frac: float = 0.04,
                 seed: int = 0):
        self.engine = engine
        self.hz = hz
        self.nt = noise_torque_frac
        self.nf = noise_fuel_frac
        self.rng = np.random.default_rng(seed)

    def _drive_cycle(self, minutes: float):
        n = int(minutes * 60 * self.hz)
        weights = np.array([s[1] for s in _SEGMENTS])
        weights = weights / weights.sum()
        rpm = np.empty(n); thr = np.empty(n)
        i = 0
        while i < n:
            seg = _SEGMENTS[self.rng.choice(len(_SEGMENTS), p=weights)]
            _, _, rlo, rhi, tlo, thi = seg
            hold = int(self.rng.integers(3, 12) * self.hz)  # a few-second hold
            r0, r1 = self.rng.uniform(rlo, rhi), self.rng.uniform(rlo, rhi)
            t0, t1 = self.rng.uniform(tlo, thi), self.rng.uniform(tlo, thi)
            k = min(hold, n - i)
            ramp = np.linspace(0, 1, k)
            rpm[i:i + k] = r0 + (r1 - r0) * ramp
            thr[i:i + k] = t0 + (t1 - t0) * ramp
            i += k
        # measurement jitter
        rpm += self.rng.normal(0, 30, n)
        thr = np.clip(thr + self.rng.normal(0, 0.01, n), 0.03, 1.0)
        return rpm, thr

    def log(self, minutes: float = 25.0) -> pd.DataFrame:
        rpm, thr = self._drive_cycle(minutes)
        tq = self.engine.torque(rpm, thr) * (1 + self.rng.normal(0, self.nt, rpm.shape))
        fr = self.engine.fuel_rate(rpm, thr) * (1 + self.rng.normal(0, self.nf, rpm.shape))
        return pd.DataFrame({
            "time_s": np.arange(len(rpm)) / self.hz,
            "rpm": rpm, "throttle": thr, "torque_Nm": tq, "fuel_g_s": fr,
        })


# --- real OBD-II backend (hardware; not exercised in CI) --------------------
class RealOBD:
    """Read a live ELM327 / OBDLink adapter via python-obd.

    Notes / honesty:
      * Requires `pip install obd` and a plugged-in adapter.
      * Standard PIDs give rpm + throttle directly. *Actual* torque needs the
        car to support PID 62 (actual torque %) and 63 (reference torque, N*m);
        torque = pct/100 * reference. If absent, estimate power from MAF
        (PID 10) instead -- left as a TODO so this stays honest about what the
        specific car exposes.
      * Untested here: I have no adapter on hand. The schema matches
        SimulatedOBD so the rest of the pipeline is identical.
    """
    def __init__(self, portstr: str | None = None):
        self.portstr = portstr

    def log(self, minutes: float = 5.0):  # pragma: no cover - needs hardware
        import time
        import obd  # type: ignore

        conn = obd.OBD(self.portstr)
        supported = conn.supported_commands
        ref_cmd = obd.commands.get("ENGINE_REFERENCE_TORQUE")
        act_cmd = obd.commands.get("ACTUAL_ENGINE_TORQUE")  # PID 62 if present
        has_torque = ref_cmd in supported and act_cmd in supported

        rows, t0 = [], time.time()
        while time.time() - t0 < minutes * 60:
            rpm = conn.query(obd.commands.RPM).value
            thr = conn.query(obd.commands.THROTTLE_POS).value
            tq = None
            if has_torque:
                ref = conn.query(ref_cmd).value
                pct = conn.query(act_cmd).value
                if ref is not None and pct is not None:
                    tq = float(ref.magnitude) * float(pct.magnitude) / 100.0
            rows.append({
                "time_s": time.time() - t0,
                "rpm": float(rpm.magnitude) if rpm else np.nan,
                "throttle": float(thr.magnitude) / 100.0 if thr else np.nan,
                "torque_Nm": tq, "fuel_g_s": np.nan,  # TODO: MAF-based estimate
            })
            time.sleep(0.1)
        return pd.DataFrame(rows)
