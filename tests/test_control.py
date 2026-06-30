"""Dynamic model + MPCC scaffold smoke tests.

The MPCC is a scaffold (see enginemap/mpcc.py): these check the model and the
controller *run and produce sane outputs*, not that the controller laps.
"""
import numpy as np
import pytest

from enginemap.dynamics import BicycleModel


def test_bicycle_model_steady_cornering():
    """A constant steer at constant-ish speed traces a circle whose radius
    matches the Ackermann estimate (~L/delta)."""
    m = BicycleModel()
    X = np.array([0, 0, 0, 40.0, 0, 0])
    delta = np.deg2rad(3.0)
    for _ in range(1500):
        drag = 0.5 * 1.225 * m.cda * X[3] ** 2
        X = m.step(X, np.array([delta, drag]), 0.01)
    assert np.isfinite(X).all()
    assert abs(X[5]) > 0                       # yaw rate developed -> it's turning
    R = X[3] / abs(X[5])
    L = m.lf + m.lr
    assert 0.4 * (L / delta) < R < 2.5 * (L / delta)   # right ballpark radius


def test_mpcc_runs_and_returns_a_control():
    ca = pytest.importorskip("casadi")  # noqa: F841
    from enginemap import track as T, raceline
    from enginemap.mpcc import MPCC

    trk = T.load("silverstone", ds=8.0)
    line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=8.0)
    mpc = MPCC(line, BicycleModel(), raceline.TRACK_WIDTH[trk.name], N=15, dt=0.08)

    phi0 = np.arctan2(line.y[1] - line.y[0], line.x[1] - line.x[0])
    X = np.array([line.x[0], line.y[0], phi0, 35.0, 0.0, 0.0])
    u, _converged = mpc.step(X, 0.0, np.array([0.0, 3000.0]))
    assert u.shape == (2,)
    assert np.isfinite(u).all()
    assert abs(u[0]) <= 0.46                   # steering within bound
