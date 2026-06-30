"""Closed-loop control: drive the dynamic model around the lap, vs the QSS optimal.

Run:  PYTHONPATH=src python scripts/make_control.py
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
from enginemap import track as T, lapsim, raceline, controller
from enginemap.dynamics import BicycleModel

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TRACKS = ["silverstone", "spa", "nordschleife"]


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    veh = Vehicle(Engine(SVJ)); model = BicycleModel()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    vmax = 0; data = {}
    for tid in TRACKS:
        trk = T.load(tid, ds=4.0)
        line, _ = raceline.optimize(trk, raceline.TRACK_WIDTH[trk.name], ds=4.0)
        qss = lapsim.simulate(line, veh)
        ctl = controller.track_lap(line, qss["v"], model)
        data[tid] = (line, qss, ctl)
        vmax = max(vmax, ctl["top_speed"] * 3.6)

    print(f"{'track':14s} {'QSS optimal':>11s} {'controller':>11s} {'gap':>8s}  mean err")
    for ax, tid in zip(axes, TRACKS):
        line, qss, ctl = data[tid]
        ax.plot(line.x, line.y, color="#b9c2cc", lw=1.0, zorder=1, label="racing line (reference)")
        pts = np.column_stack([ctl["x"], ctl["y"]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(40, vmax), lw=2.4, zorder=2)
        lc.set_array(ctl["v"][:-1] * 3.6)
        ax.add_collection(lc); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        gap = ctl["lap_time"] - qss["lap_time"]
        ax.set_title(f"{line.name.replace(' (racing line)','')}\noptimal {fmt(qss['lap_time'])} → "
                     f"controller {fmt(ctl['lap_time'])}  (+{gap:.0f}s)", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7)
        print(f"{line.name[:14]:14s} {fmt(qss['lap_time']):>11s} {fmt(ctl['lap_time']):>11s} "
              f"{'+%.1fs'%gap:>8s}  {ctl['mean_lat_err']:.2f}m")
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02); cb.set_label("speed (km/h)")
    fig.suptitle("Closed-loop control: a pure-pursuit controller drives the DYNAMIC model "
                 "(colored by speed) along the optimal line", y=0.99)
    fig.savefig(f"{FIG}/control_lap.png", bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure written to {FIG}/control_lap.png")


if __name__ == "__main__":
    main()
