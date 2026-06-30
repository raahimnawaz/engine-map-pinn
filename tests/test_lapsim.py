"""Lap-sim sanity tests on real track geometry."""
import numpy as np

from enginemap.engine import Engine, SVJ
from enginemap.vehicle import Vehicle
from enginemap import track as T, lapsim, raceline


def test_track_geometry_loads():
    nords = T.load("nordschleife")
    assert 20000 < nords.length < 21500   # real Nordschleife is ~20.8 km
    sil = T.load("silverstone")
    assert 5500 < sil.length < 6200        # ~5.9 km


def test_lap_times_in_sane_range():
    veh = Vehicle(Engine(SVJ))
    nords = lapsim.simulate(T.load("nordschleife"), veh)
    # SVJ real record ~6:45; a centerline QSS sim should be in the 6-8 min range
    assert 360 < nords["lap_time"] < 480
    assert 300 < nords["top_speed"] * 3.6 < 380   # top speed km/h


def test_racing_line_is_faster_than_centerline():
    veh = Vehicle(Engine(SVJ))
    trk = T.load("silverstone")
    base = lapsim.simulate(trk, veh)
    line, a = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name])
    rl = lapsim.simulate(line, veh)
    assert rl["lap_time"] < base["lap_time"]      # straightening corners is faster
    assert np.abs(a).max() > 1.0                   # the line actually uses the width


def test_active_aero_beats_fixed_setups():
    veh = Vehicle(Engine(SVJ))
    trk = T.load("nordschleife")
    n = len(trk.s)
    deployed = lapsim.simulate(trk, veh, deploy=np.ones(n, bool))["lap_time"]
    stalled = lapsim.simulate(trk, veh, deploy=np.zeros(n, bool))["lap_time"]
    active = lapsim.simulate(trk, veh, deploy=lapsim.active_aero_schedule(trk, veh))["lap_time"]
    assert active <= deployed and active < stalled   # best of both worlds
