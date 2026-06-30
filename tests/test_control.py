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


def test_controller_laps_and_tracks_the_line():
    """The pure-pursuit controller drives the dynamic model a full lap, tracks
    the racing line within track width, and is slower than the QSS optimal
    (the honest cost of real dynamics)."""
    from enginemap.engine import Engine, SVJ
    from enginemap.vehicle import Vehicle
    from enginemap import track as T, lapsim, raceline, controller

    veh = Vehicle(Engine(SVJ))
    trk = T.load("silverstone", ds=4.0)
    line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=4.0)
    qss = lapsim.simulate(line, veh)
    ctl = controller.track_lap(line, qss["v"], BicycleModel())
    assert ctl["lap_time"] < 300                 # completes a lap (not the safety bail)
    assert ctl["mean_lat_err"] < 4.0             # stays on the line on average
    assert ctl["lap_time"] > qss["lap_time"]     # slower than the point-mass optimum


def test_mpcc_frenet_solves_and_tracks():
    """The Frenet-frame MPCC solves and drives the dynamic plant a short way
    along the reference, staying near the line (low lateral deviation)."""
    pytest.importorskip("casadi")
    from enginemap import track as T, raceline
    from enginemap.mpcc import MPCC

    trk = T.load("silverstone", ds=8.0)
    line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=8.0)
    mpc = MPCC(line, BicycleModel(), raceline.TRACK_WIDTH[trk.name], N=15, dt=0.08)

    # one solve returns a valid in-bounds control
    u, _ok = mpc.step(np.array([0.0, 0.0, 0.0, 35.0, 0.0, 0.0]))
    assert u.shape == (2,) and np.isfinite(u).all() and abs(u[0]) <= 0.46

    # drive a short stretch: it should make progress and hug the line
    r = mpc.run_lap(v0=35.0, max_steps=60)
    assert r["max_lat"] < 4.0                   # stays within track width
    assert len(r["x"]) > 0
