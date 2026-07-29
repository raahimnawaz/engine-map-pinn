"""Physics calibration + PINN-vs-baseline regression tests."""
import numpy as np

from enginemap import pinn
from enginemap.dataset import dense_truth, sample_dyno
from enginemap.engine import NM_PER_LBFT, SVJ, Engine


def test_calibrates_to_published_peaks():
    """The mean-value model must reproduce the SVJ's published figures."""
    p = Engine(SVJ).peaks()
    assert abs(p["peak_power_hp"] - 759) / 759 < 0.03         # within 3% of 759 hp
    assert abs(p["peak_torque_Nm"] / NM_PER_LBFT - 531) / 531 < 0.03  # within 3% of 531 lb-ft
    assert abs(p["peak_torque_rpm"] - 6750) < 400
    assert 6000 < p["peak_power_rpm"] < 9000


def test_bsfc_in_physical_range():
    eng = Engine(SVJ)
    best = eng.bsfc(6500, 0.9)
    assert 200 < best < 300   # realistic best BSFC for a perf NA gasoline engine


def test_pinn_beats_data_only_baseline():
    """With sparse data leaving the low-load region unsampled, the physics
    residuals must reconstruct it better than a data-only network."""
    eng = Engine(SVJ)
    df = sample_dyno(eng, seed=1)
    t = dense_truth(eng, n_rpm=60, n_thr=60)

    base = pinn.train(eng, df, use_physics=False, epochs=1500)
    model = pinn.train(eng, df, use_physics=True, epochs=1500, lam=2.0)
    be = np.abs(pinn.predict_grid(base, t["RPM"], t["THR"])["torque"] - t["torque"])
    pe = np.abs(pinn.predict_grid(model, t["RPM"], t["THR"])["torque"] - t["torque"])

    low = t["THR"] < 0.45
    assert pe.mean() < be.mean()            # better overall
    assert pe[low].mean() < 0.6 * be[low].mean()  # decisively better where unsampled


def test_pinn_engine_drives_the_lap_sim():
    """The PINN map is LOAD-BEARING: wrapped as an engine it drives Vehicle and
    the lap simulator wherever the analytic engine did (v2 #3)."""
    from enginemap import lapsim
    from enginemap import track as T
    from enginemap.vehicle import Vehicle

    analytic = Engine(SVJ)
    df = sample_dyno(analytic, seed=0)
    net = pinn.train(analytic, df, use_physics=True, epochs=400, lam=2.0)
    eng = pinn.PINNEngine(net, analytic.spec)

    # same interface the Vehicle needs, vectorised over rpm
    tq = eng.torque(np.linspace(2000, 8000, 16), 1.0)
    assert tq.shape == (16,) and np.isfinite(tq).all()

    # and it actually produces a lap
    trk = T.load("silverstone", ds=8.0)
    lap = lapsim.simulate(trk, Vehicle(eng))
    assert np.isfinite(lap["lap_time"]) and 60 < lap["lap_time"] < 600
    assert lap["top_speed"] > 50          # it drives, it doesn't crawl
