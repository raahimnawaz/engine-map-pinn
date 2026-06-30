"""Active aero: where the wing deploys (downforce) vs stalls (low drag).

Run:  PYTHONPATH=src python scripts/make_aero.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from enginemap.engine import Engine, SVJ
from enginemap.vehicle import Vehicle
from enginemap import track as T, lapsim

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TRACKS = ["silverstone", "spa", "nordschleife"]
CMAP = ListedColormap(["#1d4ed8", "#c2410c"])  # stalled (blue), deployed (orange)


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    veh = Vehicle(Engine(SVJ))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    print(f"{'track':14s} {'deployed':>9s} {'stalled':>9s} {'ACTIVE':>9s}  gain   wing-up")
    for ax, tid in zip(axes, TRACKS):
        trk = T.load(tid)
        n = len(trk.s)
        dep = lapsim.simulate(trk, veh, deploy=np.ones(n, bool))
        sta = lapsim.simulate(trk, veh, deploy=np.zeros(n, bool))
        sched = lapsim.active_aero_schedule(trk, veh)
        act = lapsim.simulate(trk, veh, deploy=sched)

        pts = np.column_stack([trk.x, trk.y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=plt.Normalize(0, 1), lw=3)
        lc.set_array(sched[:-1].astype(float))
        ax.add_collection(lc); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        ax.set_title(f"{trk.name}\nactive {fmt(act['lap_time'])}  "
                     f"(−{dep['lap_time']-act['lap_time']:.2f}s vs fixed, "
                     f"top {dep['top_speed']*3.6:.0f}→{act['top_speed']*3.6:.0f} km/h)", fontsize=9)
        print(f"{trk.name[:14]:14s} {fmt(dep['lap_time']):>9s} {fmt(sta['lap_time']):>9s} "
              f"{fmt(act['lap_time']):>9s}  {dep['lap_time']-act['lap_time']:+.2f}s  {sched.mean()*100:.0f}%")
    handles = [plt.Line2D([], [], color="#c2410c", lw=5, label="wing DEPLOYED (downforce, corners)"),
               plt.Line2D([], [], color="#1d4ed8", lw=5, label="wing STALLED (low drag, straights)")]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Active aero — when to lift the wing (deploy) vs drop it (stall)", y=0.98)
    fig.savefig(f"{FIG}/active_aero.png", bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure written to {FIG}/active_aero.png")


if __name__ == "__main__":
    main()
