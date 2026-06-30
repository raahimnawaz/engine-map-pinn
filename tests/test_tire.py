"""Tire-grip identification tests (the PINN-B role)."""
import numpy as np

from enginemap import tire_id


def test_physics_fit_recovers_peak_unbiased():
    """Across telemetry sessions the Pacejka fit recovers the true peak grip on
    average; a plain polynomial fit is systematically low (it can't extrapolate
    past the driven slip range)."""
    B, C, D = 11.0, 1.45, 1.30
    true_peak = tire_id.peak_grip(B, C, D)
    pac, pln = [], []
    for s in range(20):
        a, mu = tire_id.simulate_corner_data(B, C, D, alpha_max_deg=7.0, n=200, noise=0.03, seed=s)
        pac.append(tire_id.peak_grip(*tire_id.fit_pacejka(a, mu)))
        pln.append(tire_id.plain_peak(a, mu))
    pac, pln = np.array(pac), np.array(pln)
    assert abs(pac.mean() - true_peak) < 0.03          # physics fit ~unbiased
    assert pln.mean() < true_peak - 0.01               # plain fit biased low
    assert pac.mean() > pln.mean()                     # physics closer to the peak


def test_pacejka_curve_is_physical():
    mu = tire_id.pacejka_mu(np.deg2rad([0, 3, 6]), 11.0, 1.45, 1.30)
    assert mu[0] == 0                                  # no force at zero slip
    assert mu[1] < mu[2] <= 1.31                       # rises toward the peak
