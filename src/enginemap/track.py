"""Load a real circuit centerline (lon/lat) and turn it into the curvature
profile the lap simulator needs.

Geometry comes from real data: Silverstone and Spa from the f1-circuits
dataset, the Nürburgring Nordschleife stitched from OpenStreetMap raceway ways
(~20.8 km, matches reality). We project lon/lat to local metres, resample to a
uniform step, and estimate curvature kappa(s) with light smoothing (a GPS
centerline is noisy, and raw curvature would spike).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

TRACK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tracks")


@dataclass
class Track:
    name: str
    s: np.ndarray        # arc length [m]
    x: np.ndarray        # local east [m]
    y: np.ndarray        # local north [m]
    kappa: np.ndarray    # curvature [1/m]
    ds: float

    @property
    def length(self):
        return float(self.s[-1])


def load(track_id: str, ds: float = 4.0, smooth_m: float = 28.0) -> Track:
    raw = json.load(open(os.path.join(TRACK_DIR, f"{track_id}.json")))
    lon, lat = np.array(raw["coords"]).T
    lat0, lon0 = lat.mean(), lon.mean()
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111320.0
    return build_track(raw["name"], x, y, ds, smooth_m)


def smooth_window(ds: float, smooth_m: float = 28.0) -> int:
    """Odd-length moving-average window covering `smooth_m` metres at step `ds`."""
    return max(3, int(smooth_m / ds) | 1)


def periodic_smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Moving average that wraps around a closed loop (no seam at start/finish)."""
    k = np.ones(w) / w
    return np.convolve(np.r_[a[-w:], a, a[:w]], k, "same")[w:-w]


def build_track(name: str, x: np.ndarray, y: np.ndarray,
                ds: float = 4.0, smooth_m: float = 28.0) -> Track:
    """Resample an (x, y) polyline [m] to uniform ds and compute curvature."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    # ensure closed loop
    if math.hypot(x[0] - x[-1], y[0] - y[-1]) > 1.0:
        x = np.append(x, x[0]); y = np.append(y, y[0])

    # cumulative arc length, then resample to uniform ds
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(s[-1] / ds)
    su = np.linspace(0, s[-1], n)
    xu = np.interp(su, s, x)
    yu = np.interp(su, s, y)

    # smooth (periodic moving average) before differentiating
    w = smooth_window(ds, smooth_m)
    xs = periodic_smooth(xu, w)
    ys = periodic_smooth(yu, w)

    # curvature kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5
    dx = np.gradient(xs, su); dy = np.gradient(ys, su)
    ddx = np.gradient(dx, su); ddy = np.gradient(dy, su)
    kappa = np.abs(dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5) + 1e-6
    # final light smooth of curvature itself
    kappa = periodic_smooth(kappa, w)

    return Track(name, su, xs, ys, kappa, ds)
