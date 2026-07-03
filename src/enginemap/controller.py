"""Closed-loop path-tracking controller on the dynamic vehicle model.

This is the controller that actually *works* (the MPCC in mpcc.py is the more
ambitious scaffold). It drives the dynamic bicycle model around the lap by
following the QSS racing line + speed profile in closed loop:

  steering   = pure pursuit toward a speed-scaled lookahead point on the line
  long force = track an ANTICIPATORY braking envelope (the fastest speed from
               which it can still brake to every reference speed within a
               lookahead) + proportional speed error, drag-comp

Pure pursuit is self-stabilizing (it aims ahead rather than reacting to the
nearest-point error), which is what lets it follow the line at racing speed on
the full dynamic model where a naive feedback law spins. The braking is
anticipatory because the plant is grip-limited (see dynamics.py): a reactive
speed law can't shed speed AT a corner, so it must slow BEFORE one or it runs
wide. Even so, this simple tracker still spikes wide at the tightest corners --
the honest limit of pure pursuit, and the motivation for the MPCC.

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
              delta_max: float = 0.45, a_brake: float = 10.0, brake_ld: float = 260.0):
    """Drive the dynamic model one lap along `line` at `speed_factor * v_ref`.
    Returns dict with the driven path, speed, lap time, and tracking error."""
    n = len(line.s)
    refxy = np.column_stack([line.x, line.y])
    phi_ref = np.unwrap(np.arctan2(np.gradient(line.y), np.gradient(line.x)))
    L = model.lf + model.lr
    # anticipatory braking envelope: at each point, the fastest speed from which
    # the car can still brake (at a_brake) to every reference speed within a
    # brake_ld lookahead. On the grip-limited plant it must slow BEFORE a corner,
    # not react at it -- a reactive tracker overshoots and leaves the track.
    w = max(1, int(brake_ld / line.ds))
    dahead = np.arange(1, w + 1) * line.ds
    vr = np.asarray(v_ref, float)
    v_env = vr.copy()
    for k in range(n):
        js = (k + np.arange(1, w + 1)) % n
        v_env[k] = min(vr[k], np.sqrt(vr[js] ** 2 + 2 * a_brake * dahead).min())

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
        # longitudinal: track the anticipatory braking envelope (feed-forward + P, drag comp)
        v_target = speed_factor * v_env[i]
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
