"""The calibrated twin — the surrogates actually drive the pipeline.

Both PINNs used to be side-panels: you could delete them and every lap time and
control result would be unchanged. This script closes that gap end to end.

  engine:  sparse noisy dyno samples -> PINN map -> Vehicle -> LAP TIME
  tyre:    cornering telemetry -> Pacejka fit -> BicycleModel -> CONTROLLED LAP

Nothing here is assumed: the engine map is reconstructed from the (sparse) dyno
data, and the tyre grip is identified from telemetry rather than taken from a
published number. Swap the simulated inputs for a real dyno/OBD log and real
IMU telemetry and the same path calibrates a twin of an *actual* car.

Run:  PYTHONPATH=src python scripts/make_calibrated_twin.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap import lapsim, pinn, raceline, tire_id
from enginemap import track as T
from enginemap.constants import MU_TYRE
from enginemap.controller import track_lap
from enginemap.dataset import sample_dyno
from enginemap.dynamics import BicycleModel
from enginemap.engine import SVJ, Engine
from enginemap.vehicle import Vehicle

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TRACK = "silverstone"
# the tyre the car "really" has -- what the telemetry is generated from. It is
# NOT the published/assumed MU_TYRE, which is the point: calibration finds it.
B_TRUE, C_TRUE, D_TRUE = 11.0, 1.45, 1.30


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    trk = T.load(TRACK, ds=4.0)
    line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=4.0)

    # --- stage 1: the ENGINE map comes from the PINN, not the equations -------
    analytic = Engine(SVJ)
    df = sample_dyno(analytic, seed=0)
    net = pinn.train(analytic, df, use_physics=True, epochs=4000, lam=2.0)
    pinn_engine = pinn.PINNEngine(net, analytic.spec)

    veh_analytic = Vehicle(analytic)
    veh_pinn = Vehicle(pinn_engine)
    lap_analytic = lapsim.simulate(line, veh_analytic)
    lap_pinn = lapsim.simulate(line, veh_pinn)

    # --- stage 2: the TYRE grip comes from telemetry, not a published number --
    alpha, mu_meas = tire_id.simulate_corner_data(B_TRUE, C_TRUE, D_TRUE,
                                                  alpha_max_deg=7.0, n=200, noise=0.03)
    B_fit, C_fit, D_fit = tire_id.fit_pacejka(alpha, mu_meas)

    model_assumed = BicycleModel(tractive=veh_pinn.tractive_force)          # mu = published
    model_calib = BicycleModel.from_pacejka(B_fit, C_fit, D_fit,
                                            tractive=veh_pinn.tractive_force)
    ctl_assumed = track_lap(line, lap_pinn["v"], model_assumed)
    ctl_calib = track_lap(line, lap_pinn["v"], model_calib)

    print(f"--- {line.name} ---")
    print("ENGINE (stage 1): the lap time comes from the reconstructed map")
    print(f"  analytic engine        lap {fmt(lap_analytic['lap_time'])}  "
          f"top {lap_analytic['top_speed'] * 3.6:.0f} km/h")
    print(f"  PINN map (from dyno)   lap {fmt(lap_pinn['lap_time'])}  "
          f"top {lap_pinn['top_speed'] * 3.6:.0f} km/h   "
          f"[delta {lap_pinn['lap_time'] - lap_analytic['lap_time']:+.2f} s]")
    print("\nTYRE (stage 2): the plant the controller drives comes from telemetry")
    print(f"  true peak grip                 mu = {tire_id.peak_grip(B_TRUE, C_TRUE, D_TRUE):.3f}")
    print(f"  assumed (published)            mu = {model_assumed.peak_mu:.3f}   "
          f"-> lap {fmt(ctl_assumed['lap_time'])}  (mean err {ctl_assumed['mean_lat_err']:.2f} m)")
    print(f"  identified from telemetry      mu = {model_calib.peak_mu:.3f}   "
          f"-> lap {fmt(ctl_calib['lap_time'])}  (mean err {ctl_calib['mean_lat_err']:.2f} m)")
    print(f"\n  grip error if you just trust the published number: "
          f"{model_assumed.peak_mu - tire_id.peak_grip(B_TRUE, C_TRUE, D_TRUE):+.3f} mu")

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    v = np.linspace(10, 100, 200)
    axes[0].plot(v * 3.6, veh_analytic.tractive_force(v) / 1000, lw=2.2,
                 color="#1d4ed8", label="analytic engine")
    axes[0].plot(v * 3.6, veh_pinn.tractive_force(v) / 1000, lw=1.8, ls="--",
                 color="#c2410c", label="PINN map (from sparse dyno)")
    axes[0].set_xlabel("speed (km/h)"); axes[0].set_ylabel("tractive force (kN)")
    axes[0].set_title("Stage 1 — the engine map that drives the lap")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    a = np.linspace(0, np.deg2rad(16), 300)
    axes[1].plot(np.rad2deg(a), tire_id.pacejka_mu(a, B_TRUE, C_TRUE, D_TRUE),
                 lw=2.2, color="#0b7285", label=f"true tyre (peak {D_TRUE:.2f})")
    axes[1].plot(np.rad2deg(a), tire_id.pacejka_mu(a, B_fit, C_fit, D_fit),
                 lw=1.8, ls="--", color="#c2410c",
                 label=f"identified from telemetry ({D_fit:.2f})")
    axes[1].axhline(MU_TYRE, color="#868e96", ls=":", lw=1.6,
                    label=f"assumed / published ({MU_TYRE:.2f})")
    axes[1].scatter(np.rad2deg(alpha), mu_meas, s=8, c="#adb5bd", zorder=1,
                    label="telemetry (sub-limit)")
    axes[1].set_xlabel("slip angle (deg)"); axes[1].set_ylabel(r"grip $\mu = F_y/F_z$")
    axes[1].set_title("Stage 2 — the tyre the controller's plant uses")
    axes[1].legend(fontsize=7.5, loc="lower right"); axes[1].grid(alpha=0.25)

    names = ["QSS\n(PINN map)", "controlled\n(assumed tyre)", "controlled\n(identified tyre)"]
    times = [lap_pinn["lap_time"], ctl_assumed["lap_time"], ctl_calib["lap_time"]]
    bars = axes[2].bar(names, times, color=["#1d4ed8", "#868e96", "#c2410c"])
    for b, t in zip(bars, times):
        axes[2].text(b.get_x() + b.get_width() / 2, t + 1, fmt(t),
                     ha="center", fontsize=9)
    axes[2].set_ylabel("lap time (s)"); axes[2].set_ylim(0, max(times) * 1.18)
    axes[2].set_title("Lap time, end to end from the data")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("The calibrated twin: engine map and tyre grip come from data, "
                 "and drive the lap", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/calibrated_twin.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {FIG}/calibrated_twin.png")


if __name__ == "__main__":
    main()
