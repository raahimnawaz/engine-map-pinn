"""3D surface views of the engine map + where a simulated drive actually lands.

Run:  PYTHONPATH=src python scripts/make_3d.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap.engine import Engine, SVJ, NM_PER_LBFT
from enginemap.dataset import dense_truth
from enginemap.telemetry import SimulatedOBD

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def _style(ax, zlabel):
    ax.set_xlabel("rpm", labelpad=2)
    ax.set_ylabel("throttle (%)", labelpad=2)
    ax.set_zlabel(zlabel, labelpad=2)
    ax.view_init(elev=28, azim=-58)
    ax.tick_params(labelsize=7)


def fig_surfaces(t):
    fig = plt.figure(figsize=(14, 4.8))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(t["RPM"], t["THR"] * 100, t["torque"] / NM_PER_LBFT,
                     cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("Brake torque (lb-ft)")
    _style(ax1, "lb-ft")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(t["RPM"], t["THR"] * 100, t["power_hp"],
                     cmap="magma", linewidth=0, antialiased=True)
    ax2.set_title("Power (hp)")
    _style(ax2, "hp")

    # BSFC as a valley: lower = more efficient, so the island is the basin
    ax3 = fig.add_subplot(133, projection="3d")
    bsfc = np.clip(t["bsfc"], 200, 520)
    ax3.plot_surface(t["RPM"], t["THR"] * 100, bsfc, cmap="RdYlGn_r",
                     linewidth=0, antialiased=True)
    ax3.set_title("BSFC valley (g/kWh) — basin = efficiency island")
    _style(ax3, "g/kWh")
    ax3.invert_zaxis()  # basin points up = most efficient

    fig.tight_layout()
    fig.savefig(f"{FIG}/surfaces_3d.png", bbox_inches="tight")
    plt.close(fig)


def fig_drive_on_map(eng, t):
    """Where a real-style drive actually operates, scattered on the torque
    surface. The point: a normal drive crowds the low-load floor and barely
    touches the high-load slope -- which is exactly why an unconstrained model
    has nothing to learn from up there."""
    log = SimulatedOBD(eng, seed=3).log(minutes=20)
    fig = plt.figure(figsize=(8.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(t["RPM"], t["THR"] * 100, t["torque"] / NM_PER_LBFT,
                    cmap="viridis", alpha=0.45, linewidth=0, antialiased=True)
    sc = ax.scatter(log["rpm"], log["throttle"] * 100, log["torque_Nm"] / NM_PER_LBFT,
                    c=log["throttle"], cmap="autumn_r", s=6, depthshade=False)
    ax.set_title("Where a 20-min drive actually operates, on the torque map\n"
                 "(crowds the low-load floor; rarely climbs the high-load slope)")
    _style(ax, "torque (lb-ft)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("throttle")
    fig.tight_layout()
    fig.savefig(f"{FIG}/drive_on_map_3d.png", bbox_inches="tight")
    plt.close(fig)


def main():
    eng = Engine(SVJ)
    t = dense_truth(eng)
    fig_surfaces(t)
    fig_drive_on_map(eng, t)
    print(f"3D figures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
