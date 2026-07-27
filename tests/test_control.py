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


def test_qss_and_dynamic_model_agree_on_grip():
    """One car, one grip: the QSS point-mass mu and the dynamic model's effective
    peak grip (mean of the Pacejka axle peaks) must be the same number (v2 #2)."""
    from enginemap.engine import Engine, SVJ
    from enginemap.vehicle import Vehicle

    veh = Vehicle(Engine(SVJ))
    m = BicycleModel()
    assert abs(m.peak_mu - veh.mu) < 1e-9
    assert m.Dr > m.Df          # rear grip bias kept (stable at the limit) ...
    assert abs(0.5 * (m.Df + m.Dr) - veh.mu) < 1e-9   # ... without changing grip


def test_plant_is_built_from_the_identified_tyre():
    """The tyre fit is LOAD-BEARING: an identified (B, C, D) becomes the plant the
    controller drives, and its peak grip is the identified grip (v2 #3)."""
    from enginemap import tire_id

    alpha, mu = tire_id.simulate_corner_data(11.0, 1.45, 1.30, alpha_max_deg=7.0,
                                             n=200, noise=0.03)
    B, C, D = tire_id.fit_pacejka(alpha, mu)
    m = BicycleModel.from_pacejka(B, C, D)

    assert abs(m.peak_mu - D) < 1e-9          # the plant's grip IS the fitted grip
    assert m.Dr > m.Df                        # rear bias preserved
    # identifying from telemetry beats trusting the published number
    true_peak = tire_id.peak_grip(11.0, 1.45, 1.30)
    assert abs(m.peak_mu - true_peak) < abs(BicycleModel().peak_mu - true_peak)


def test_mpcc_curvature_is_the_smoothed_track_profile():
    """The MPCC steers against the track's smoothed curvature, not a second,
    spikier estimate of its own (v2 #2)."""
    pytest.importorskip("casadi")
    from enginemap import track as T
    from enginemap.mpcc import signed_curvature

    trk = T.load("silverstone", ds=4.0)
    ks, _ = signed_curvature(trk)
    raw = np.gradient(
        np.unwrap(np.arctan2(np.gradient(trk.y), np.gradient(trk.x))), trk.s)
    # same magnitude as the track's own kappa, but smoother than the raw estimate
    assert abs(np.abs(ks).max() - np.abs(trk.kappa).max()) < 0.01
    assert np.abs(np.diff(ks)).mean() < np.abs(np.diff(raw)).mean()


def test_plant_longitudinal_is_grip_and_engine_limited():
    """Longitudinal force is a real tyre/engine force, not a free input: it stays
    inside the friction circle and drive is capped by the engine (v2 fix #1)."""
    m = BicycleModel()
    Fz = 20000.0
    mu = 0.5 * (m.Df + m.Dr)
    # friction circle: with big lateral demand, an absurd drive command still
    # leaves total tyre force within grip.
    Fy = 0.9 * mu * Fz
    Fx = m._limit_fx(1e6, Fy, Fz, 50.0)
    assert Fx ** 2 + Fy ** 2 <= (mu * Fz) ** 2 + 1.0
    # straight line: full grip is available for braking (not more).
    assert abs(m._limit_fx(-1e6, 0.0, Fz, 50.0) + mu * Fz) < 1.0
    # engine cap: drive can't exceed the tractive curve, but braking is grip-limited.
    m2 = BicycleModel(tractive=lambda v: 5000.0)
    assert abs(m2._limit_fx(1e6, 0.0, Fz, 50.0) - 5000.0) < 1.0     # drive capped by engine
    assert abs(m2._limit_fx(-1e6, 0.0, Fz, 50.0) + mu * Fz) < 1.0   # brake still grip-limited


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

    # drive a short stretch: it should make progress and stay on the track.
    # Threshold is the actual corridor half-width (what "within track width"
    # means) rather than a magic number tuned to the old, unphysical plant: the
    # plant is now grip-limited (v2 #1) while the NLP's own model is not yet, so
    # tracking is looser until that is aligned in v2 #4.
    r = mpc.run_lap(v0=35.0, max_steps=60)
    assert r["max_lat"] < raceline.TRACK_WIDTH[trk.name] / 2 - 0.6
    assert len(r["x"]) > 0
