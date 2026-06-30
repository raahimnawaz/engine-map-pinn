"""Quasi-steady-state lap-time simulator.

For each point on the track:
  1. corner-speed limit from the friction circle (grip + downforce),
  2. a forward pass — acceleration limited by the engine's tractive force AND
     the grip left over after cornering (friction circle),
  3. a backward pass — braking limited by grip,
take the minimum, wrap around the closed loop until converged, and integrate
dt = ds / v for the lap time. This is the standard first-order tool real
motorsport engineers use for concept work.

Honest scope: point-mass, quasi-steady (no transient weight transfer or tyre
thermal), and the *centerline* is not the racing line, so absolute lap times
are approximate. Relative comparisons — track vs track, power vs grip limited,
+power gains — are the trustworthy outputs.
"""
from __future__ import annotations

import numpy as np

from .track import Track
from .vehicle import Vehicle, G, RHO_AIR

VMAX = 120.0  # m/s hard cap (~432 km/h), never reached


def _corner_speed(track: Track, veh: Vehicle):
    # m v^2 kappa = mu (m g + 0.5 rho ClA v^2)  ->  solve for v
    denom = veh.mass * track.kappa - veh.mu * 0.5 * RHO_AIR * veh.cla
    v = np.where(denom > 0, np.sqrt(veh.mu * veh.mass * G / np.maximum(denom, 1e-9)), VMAX)
    return np.minimum(v, VMAX)


def simulate(track: Track, veh: Vehicle, n_iter: int = 4):
    n = len(track.s)
    ds = track.ds
    v_corner = _corner_speed(track, veh)
    v = v_corner.copy()

    def long_grip(vv, i):
        a_lat = vv * vv * track.kappa[i]
        fg = veh.grip_force(vv)
        return np.sqrt(np.maximum(fg * fg - (veh.mass * a_lat) ** 2, 0.0))

    for _ in range(n_iter):
        # forward (accel), wrapping around the loop
        for i in range(n):
            j = (i - 1) % n
            vv = v[j]
            f_drive = min(veh.tractive_force(vv)[0], long_grip(vv, j))
            a = (f_drive - veh.drag(vv) - veh.rolling(vv)) / veh.mass
            v_next = np.sqrt(max(v[j] ** 2 + 2 * a * ds, 1.0))
            v[i] = min(v[i], v_next, v_corner[i])
        # backward (braking), wrapping around the loop
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            vv = v[j]
            f_brake = long_grip(vv, j) + veh.drag(vv) + veh.rolling(vv)
            a = f_brake / veh.mass
            v_prev = np.sqrt(max(v[j] ** 2 + 2 * a * ds, 1.0))
            v[i] = min(v[i], v_prev)

    # lap time via segment midpoint speed
    vmid = 0.5 * (v + np.roll(v, -1))
    dt = ds / np.maximum(vmid, 1.0)
    lap_time = float(dt.sum())

    # classify the limiting factor at each point. grip = held back by a real
    # corner (v_corner binding and below the VMAX cap); power = accelerating
    # with the engine, not grip, as the bottleneck; otherwise braking.
    f_eng = veh.tractive_force(v)
    f_grip_long = long_grip(v, np.arange(n))
    accelerating = np.roll(v, -1) >= v
    at_corner = (v <= v_corner * 1.02) & (v_corner < VMAX * 0.98)
    limit = np.where(at_corner, "grip",
                     np.where(accelerating & (f_eng <= f_grip_long * 1.02), "power", "brake"))

    return {
        "track": track, "v": v, "s": track.s, "lap_time": lap_time,
        "top_speed": float(v.max()), "avg_speed": float(track.length / lap_time),
        "v_corner": v_corner, "limit": limit,
        "frac_power": float((limit == "power").mean()),
        "frac_grip": float((limit == "grip").mean()),
    }
