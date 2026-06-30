"""Closed-loop path-tracking controller on the dynamic vehicle model.

This is the controller that actually *works* (the MPCC in mpcc.py is the more
ambitious scaffold). It drives the dynamic bicycle model around the lap by
following the QSS racing line + speed profile in closed loop:

  steering   = pure pursuit toward a speed-scaled lookahead point on the line
  long force = speed feed-forward (track v_ref) + proportional speed error, drag-comp

Pure pursuit is self-stabilizing (it aims ahead rather than reacting to the
nearest-point error), which is what lets it follow the line at racing speed on
the full dynamic model where a naive feedback law spins.

It's evaluated against the QSS optimal lap as the baseline: the lap-time gap is
the cost of real dynamics (tyre slip, yaw) plus tracking at a grip margin. This
is a genuine closed-loop result on the dynamic model — not a planning sim.
"""
from __future__ import annotations

import numpy as np

from .dynamics import BicycleModel
from .track import Track


def track_lap(line: Track, v_ref, model: BicycleModel, *,
              speed_factor: float = 0.92, dt: float = 0.02,
              ld_gain: float = 0.6, ld_min: float = 8.0, ld_max: float = 32.0,
              k_v: float = 2500.0, fx_max: float = 11000.0, fx_min: float = -24000.0,
              delta_max: float = 0.45):
    """Drive the dynamic model one lap along `line` at `speed_factor * v_ref`.
    Returns dict with the driven path, speed, lap time, and tracking error."""
    n = len(line.s)
    refxy = np.column_stack([line.x, line.y])
    phi_ref = np.unwrap(np.arctan2(np.gradient(line.y), np.gradient(line.x)))
    L = model.lf + model.lr

    # start aligned on the line
    X = np.array([line.x[0], line.y[0], phi_ref[0], max(v_ref[0] * speed_factor, 10.0), 0.0, 0.0])
    i = 0                       # current reference index (advances monotonically)
    t = 0.0
    xs, ys, vs, lat = [], [], [], []
    laps_pts = 0
    while laps_pts < n - 1:
        # local nearest-index search (forward window)
        win = np.arange(i, i + 40) % n
        di = win[np.argmin(np.sum((refxy[win] - X[:2]) ** 2, axis=1))]
        if (di - i) % n < n // 2:
            laps_pts += (di - i) % n
            i = di
        # signed lateral error (for reporting); + = car right of path
        ph = phi_ref[i]
        e_lat = np.sin(ph) * (X[0] - line.x[i]) - np.cos(ph) * (X[1] - line.y[i])
        # PURE PURSUIT steering: aim at a lookahead point on the line
        ld = np.clip(ld_gain * X[3], ld_min, ld_max)
        j = i
        while (line.s[j % n] - line.s[i]) % line.length < ld:
            j += 1
        tx, ty = line.x[j % n], line.y[j % n]
        alpha = np.arctan2(ty - X[1], tx - X[0]) - X[2]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.clip(np.arctan(2 * L * np.sin(alpha) / ld), -delta_max, delta_max)
        # longitudinal: track reference speed with feed-forward + PI, drag comp
        v_target = speed_factor * v_ref[i]
        drag = 0.5 * 1.225 * model.cda * X[3] ** 2
        Fx = np.clip(k_v * (v_target - X[3]) + drag, fx_min, fx_max)
        X = model.step(X, np.array([delta, Fx]), dt)
        t += dt
        xs.append(X[0]); ys.append(X[1]); vs.append(X[3]); lat.append(abs(e_lat))
        if t > 1200:           # safety: bail if it never completes
            break
    return {
        "x": np.array(xs), "y": np.array(ys), "v": np.array(vs),
        "lap_time": t, "top_speed": float(np.max(vs)) if vs else 0.0,
        "mean_lat_err": float(np.mean(lat)) if lat else 0.0,
        "max_lat_err": float(np.max(lat)) if lat else 0.0,
    }
