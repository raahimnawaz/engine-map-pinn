"""Tire-grip calibration: recover peak grip from cornering telemetry and feed it
into the lap sim. Mirrors the vehicle-dynamics PINN-B role.

Run:  PYTHONPATH=src python scripts/make_tire_calibration.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap.engine import Engine, SVJ
from enginemap.vehicle import Vehicle
from enginemap import track as T, lapsim, raceline, tire_id

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    # TRUE tyre (e.g. worn): peak grip 1.30; the sim's published spec assumes 1.45
    B, C, D = 11.0, 1.45, 1.30
    a, mu = tire_id.simulate_corner_data(B, C, D, alpha_max_deg=7.0, n=200, noise=0.03)
    true_peak = tire_id.peak_grip(B, C, D)
    Bf, Cf, Df = tire_id.fit_pacejka(a, mu)
    pac_peak = tire_id.peak_grip(Bf, Cf, Df)
    plain_peak = tire_id.plain_peak(a, mu)

    # bias across many telemetry sessions (the honest, robust result)
    pac_s, pln_s = [], []
    for s in range(40):
        aa_, mu_ = tire_id.simulate_corner_data(B, C, D, alpha_max_deg=7.0, n=200, noise=0.03, seed=s)
        pac_s.append(tire_id.peak_grip(*tire_id.fit_pacejka(aa_, mu_)))
        pln_s.append(tire_id.plain_peak(aa_, mu_))
    pac_s, pln_s = np.array(pac_s), np.array(pln_s)

    # lap-time impact (using the unbiased physics-fit mean vs the published spec)
    eng = Engine(SVJ)
    trk = T.load("silverstone", ds=5.0)
    line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=5.0)

    def lap(mu):
        return lapsim.simulate(line, Vehicle(eng, mu=mu))["lap_time"]
    lap_assumed, lap_truth, lap_calib = lap(1.45), lap(true_peak), lap(pac_s.mean())
    print(f"true peak {true_peak:.3f} | physics mean {pac_s.mean():.3f}±{pac_s.std():.3f} "
          f"| plain mean {pln_s.mean():.3f}±{pln_s.std():.3f}")
    print(f"lap: assumed(1.45) {fmt(lap_assumed)} | calibrated {fmt(lap_calib)} | truth {fmt(lap_truth)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    aa = np.linspace(0, np.deg2rad(14), 300)
    ax1.scatter(np.rad2deg(a), mu, s=8, color="#5b6b7a", alpha=0.5, label="telemetry (sub-limit)")
    ax1.plot(np.rad2deg(aa), tire_id.pacejka_mu(aa, B, C, D), color="#0f1b2d", lw=2, label="true tyre curve")
    ax1.plot(np.rad2deg(aa), tire_id.pacejka_mu(aa, Bf, Cf, Df), "--", color="#15803d", lw=2,
             label=f"physics (Pacejka) fit → peak {pac_peak:.2f}")
    cpoly = np.polyfit(a, mu, 4)
    ax1.plot(np.rad2deg(aa[aa < a.max() * 1.4]), np.polyval(cpoly, aa[aa < a.max() * 1.4]),
             ":", color="#c2410c", lw=2, label=f"plain poly fit → peak {plain_peak:.2f}")
    ax1.axvline(np.rad2deg(a.max()), color="gray", ls=":", lw=1)
    ax1.text(np.rad2deg(a.max()) + 0.1, 0.2, "data stops here", fontsize=8, color="gray")
    ax1.axhline(true_peak, color="#0f1b2d", lw=0.6, alpha=0.4)
    ax1.set_xlabel("slip angle (deg)"); ax1.set_ylabel("grip  μ = Fy/Fz")
    ax1.set_title("Recovering peak grip from sub-limit telemetry\n(physics extrapolates the peak; the polynomial can't)")
    ax1.legend(fontsize=8); ax1.set_ylim(0, 1.5)

    # peak-grip estimate distribution over 40 telemetry sessions
    ax2.axhline(true_peak, color="#0f1b2d", lw=1.5, label=f"true peak grip ({true_peak:.2f})")
    ax2.axhline(1.45, color="#c2410c", lw=1.2, ls="--", label="assumed spec (1.45) — optimistic")
    bp = ax2.boxplot([pac_s, pln_s], positions=[0, 1], widths=0.5, patch_artist=True,
                     tick_labels=["physics\n(Pacejka) fit", "plain\npoly fit"])
    for patch, c in zip(bp["boxes"], ["#15803d", "#c2410c"]):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    ax2.set_ylabel("estimated peak grip μ  (40 telemetry sessions)")
    ax2.set_title("Physics fit is unbiased (centered on truth);\nthe plain fit is systematically low — it can't see the peak")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.text(0.5, 1.45 - 0.12,
             f"using the spec mis-predicts the lap by {lap_assumed - lap_truth:+.1f}s;\n"
             f"calibrating from telemetry cuts that to {lap_calib - lap_truth:+.1f}s",
             ha="center", fontsize=8, color="#333",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    fig.tight_layout(); fig.savefig(f"{FIG}/tire_calibration.png", bbox_inches="tight"); plt.close(fig)
    print(f"figure written to {FIG}/tire_calibration.png")


if __name__ == "__main__":
    main()
