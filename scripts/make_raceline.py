"""Racing-line optimization figure: centerline vs minimum-curvature line.

Run:  PYTHONPATH=src python scripts/make_raceline.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from enginemap.engine import Engine, SVJ
from enginemap.vehicle import Vehicle
from enginemap import track as T, lapsim, raceline

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TRACKS = ["silverstone", "spa", "nordschleife"]
# real-world SVJ-class reference lap times for context
REAL = {"Nürburgring Nordschleife": "6:44.97 (SVJ record)"}


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    veh = Vehicle(Engine(SVJ))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    vmax = 0
    data = {}
    for tid in TRACKS:
        trk = T.load(tid)
        base = lapsim.simulate(trk, veh)
        line, a = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name])
        rl = lapsim.simulate(line, veh)
        data[tid] = (trk, base, line, rl)
        vmax = max(vmax, rl["top_speed"] * 3.6)

    print(f"{'track':16s} {'centerline':>10s} {'racing line':>12s} {'gain':>7s}  real")
    for ax, tid in zip(axes, TRACKS):
        trk, base, line, rl = data[tid]
        ax.plot(trk.x, trk.y, color="#b9c2cc", lw=1.0, zorder=1, label="centerline")
        pts = np.column_stack([line.x, line.y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(40, vmax), lw=2.6, zorder=2)
        lc.set_array(rl["v"][:-1] * 3.6)
        ax.add_collection(lc)
        ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        ax.set_title(f"{trk.name}\ncenterline {fmt(base['lap_time'])} → racing line {fmt(rl['lap_time'])}"
                     f"  (−{base['lap_time']-rl['lap_time']:.0f}s)", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7)
        print(f"{trk.name[:16]:16s} {fmt(base['lap_time']):>10s} {fmt(rl['lap_time']):>12s} "
              f"{base['lap_time']-rl['lap_time']:6.1f}s  {REAL.get(trk.name,'—')}")
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02); cb.set_label("speed (km/h)")
    fig.suptitle("Minimum-curvature racing line (colored by speed) vs the centerline (grey)", y=0.98)
    fig.savefig(f"{FIG}/raceline.png", bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure written to {FIG}/raceline.png")


if __name__ == "__main__":
    main()
