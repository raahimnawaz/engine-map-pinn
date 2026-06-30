"""Minimum-curvature racing line.

The centerline is not the fast way round — a driver straightens corners by
using the track width. This finds the lateral offset alpha(s) within the track
boundaries that minimizes path curvature (the standard minimum-curvature racing
line, e.g. TUMFTM / Heilmeier et al.). A smoother line carries more speed
through corners, so re-simulating it gives a faster lap.

Honest scope: minimum-curvature is a proxy for the true minimum-*time* line,
and we assume a constant track width per circuit (real per-point width isn't in
the centerline data). It captures most of the gain and is the established
method; it is not a full minimum-time trajectory optimization.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .track import Track, build_track

# approximate track widths [m] (constant-width assumption)
TRACK_WIDTH = {
    "Silverstone Circuit": 12.0,
    "Circuit de Spa-Francorchamps": 12.0,
    "Nürburgring Nordschleife": 8.0,
}


def _normals(x, y):
    dx = np.gradient(x); dy = np.gradient(y)
    t = np.hypot(dx, dy)
    # left-hand normal (unit)
    return -dy / t, dx / t


def optimize(track: Track, width: float, ds: float = 4.0):
    """Return a new Track following the minimum-curvature line, plus the offsets."""
    x, y = track.x, track.y
    nx, ny = _normals(x, y)
    n = len(x)
    half = width / 2.0 - 0.5  # keep half a metre off the kerb

    # centered second difference of the centerline (wrapped), matching the gradient
    cx = np.roll(x, 1) - 2 * x + np.roll(x, -1)
    cy = np.roll(y, 1) - 2 * y + np.roll(y, -1)

    def dvec(a):
        # centered second difference of the offset path position
        pax = a * nx; pay = a * ny
        dx = cx + np.roll(pax, 1) - 2 * pax + np.roll(pax, -1)
        dy = cy + np.roll(pay, 1) - 2 * pay + np.roll(pay, -1)
        return dx, dy

    def obj(a):
        dx, dy = dvec(a)
        return float(np.sum(dx * dx + dy * dy))

    def grad(a):
        dx, dy = dvec(a)
        # dJ/da_k = 2 n_k . (d_{k-1} - 2 d_k + d_{k+1})  (wrapped)
        lap_dx = np.roll(dx, 1) - 2 * dx + np.roll(dx, -1)
        lap_dy = np.roll(dy, 1) - 2 * dy + np.roll(dy, -1)
        return 2.0 * (nx * lap_dx + ny * lap_dy)

    res = minimize(obj, np.zeros(n), jac=grad, method="L-BFGS-B",
                   bounds=[(-half, half)] * n, options={"maxiter": 400})
    a = res.x
    rx = x + a * nx; ry = y + a * ny
    line = build_track(track.name + " (racing line)", rx, ry, ds=ds)
    return line, a
