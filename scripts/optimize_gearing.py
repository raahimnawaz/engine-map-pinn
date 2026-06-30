"""Stage 3 — use the model to GO FASTER.

The engine map + lap sim are an evaluator: given a car setup, they return a lap
time. Wrap an optimizer around them and you can search setups for the fastest
one. Here we optimize the final-drive ratio per track (taller = more top speed,
shorter = more acceleration; the best trade-off is track-specific) and report
the lap time gained over the stock SVJ.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap.engine import Engine, SVJ
from enginemap.vehicle import Vehicle
from enginemap import track as T, lapsim

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TRACKS = ["silverstone", "spa", "nordschleife"]
STOCK_FD = 3.08


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    eng = Engine(SVJ)
    fds = np.linspace(2.4, 4.2, 19)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    print(f"{'track':16s} {'stock':>9s} {'optimized':>10s} {'best FD':>8s} {'gain':>7s}")
    for tid, color in zip(TRACKS, ["#1d4ed8", "#c2410c", "#15803d"]):
        trk = T.load(tid)
        times = np.array([lapsim.simulate(trk, Vehicle(eng, final_drive=fd))["lap_time"]
                          for fd in fds])
        stock_t = lapsim.simulate(trk, Vehicle(eng, final_drive=STOCK_FD))["lap_time"]
        i_best = int(np.argmin(times))
        best_fd, best_t = fds[i_best], times[i_best]
        ax.plot(fds, times - stock_t, "-", color=color, lw=2, label=trk.name)
        ax.scatter([best_fd], [best_t - stock_t], color=color, zorder=5, s=45, edgecolor="white")
        print(f"{trk.name[:16]:16s} {fmt(stock_t):>9s} {fmt(best_t):>10s} "
              f"{best_fd:7.2f} {stock_t - best_t:6.2f}s")
    ax.axvline(STOCK_FD, color="gray", ls=":", lw=1)
    ax.text(STOCK_FD + 0.03, ax.get_ylim()[1] * 0.8, "stock 3.08", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("final-drive ratio  (taller ←→ shorter)")
    ax.set_ylabel("lap-time change vs stock (s)  — lower = faster")
    ax.set_title("Optimizing one setup parameter with the model: best final drive per track")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/optimize_gearing.png"); plt.close(fig)
    print(f"\nfigure written to {FIG}/optimize_gearing.png")


if __name__ == "__main__":
    main()
