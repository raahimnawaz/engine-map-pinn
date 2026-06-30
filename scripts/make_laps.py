"""Lap simulation across three real circuits + the figures and data.

Run:  PYTHONPATH=src python scripts/make_laps.py
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
from enginemap import track as T, lapsim

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
TRACKS = ["silverstone", "spa", "nordschleife"]
LIMIT_COLORS = {"grip": "#c2410c", "power": "#1d4ed8", "brake": "#15803d"}


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def fig_track_maps(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    vmax = max(r["top_speed"] for r in results.values()) * 3.6
    for ax, tid in zip(axes, TRACKS):
        r = results[tid]; trk = r["track"]
        pts = np.column_stack([trk.x, trk.y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(40, vmax), lw=3)
        lc.set_array(r["v"][:-1] * 3.6)
        ax.add_collection(lc)
        ax.set_aspect("equal"); ax.autoscale()
        ax.set_title(f"{trk.name}\n{fmt(r['lap_time'])}  ·  top {r['top_speed']*3.6:.0f} km/h")
        ax.axis("off")
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("speed (km/h)")
    fig.suptitle("SVJ lap simulation on real circuit geometry — colored by speed", y=0.98)
    fig.savefig(f"{FIG}/lap_track_maps.png", bbox_inches="tight"); plt.close(fig)


def fig_speed_traces(results):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    for ax, tid in zip(axes, TRACKS):
        r = results[tid]
        s_km = r["s"] / 1000.0
        ax.plot(s_km, r["v"] * 3.6, color="#0f1b2d", lw=1.2)
        # shade by limiting factor
        for lab, col in LIMIT_COLORS.items():
            mask = r["limit"] == lab
            ax.fill_between(s_km, 0, r["v"] * 3.6, where=mask, color=col, alpha=0.18, step="mid")
        ax.set_ylabel("speed (km/h)")
        ax.set_title(f"{r['track'].name}  —  {fmt(r['lap_time'])}  "
                     f"({r['frac_grip']*100:.0f}% grip-limited, {r['frac_power']*100:.0f}% power-limited)")
        ax.margins(x=0)
    axes[-1].set_xlabel("distance around lap (km)")
    handles = [plt.Line2D([], [], color=c, lw=6, alpha=0.4, label=f"{lab}-limited")
               for lab, c in LIMIT_COLORS.items()]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(f"{FIG}/lap_speed_traces.png"); plt.close(fig)


def fig_power_sensitivity(veh_base):
    """Does more power actually help? Lap time vs peak power per track."""
    scales = np.linspace(0.7, 2.0, 10)
    peak_hp = 759 * scales
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for tid, color in zip(TRACKS, ["#1d4ed8", "#c2410c", "#15803d"]):
        trk = T.load(tid)
        times = []
        for sc in scales:
            veh = Vehicle(veh_base.engine, power_scale=sc)
            times.append(lapsim.simulate(trk, veh)["lap_time"])
        times = np.array(times)
        base_t = times[np.argmin(np.abs(scales - 1.0))]
        ax.plot(peak_hp, 100 * (times - base_t) / base_t, "o-", color=color, lw=2, label=trk.name)
    ax.axvline(759, color="gray", ls=":", lw=1); ax.text(765, -1.0, "stock SVJ\n759 hp", fontsize=8)
    ax.axvline(1530, color="gray", ls=":", lw=1); ax.text(1380, -1.0, "twin-turbo\n1530 hp", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("peak power (hp)"); ax.set_ylabel("lap-time change vs stock (%)  — lower = faster")
    ax.set_title("Doubling the power buys only ~5–6%: most of a lap is grip-limited, not power-limited")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/lap_power_sensitivity.png"); plt.close(fig)


def main():
    veh = Vehicle(Engine(SVJ))
    results = {tid: lapsim.simulate(T.load(tid), veh) for tid in TRACKS}

    print(f"\n{'track':16s} {'length':>8s} {'lap time':>9s} {'top':>9s} {'avg':>9s}  grip/power")
    for tid in TRACKS:
        r = results[tid]
        print(f"{r['track'].name[:16]:16s} {r['track'].length/1000:6.2f}km {fmt(r['lap_time']):>9s} "
              f"{r['top_speed']*3.6:6.0f}kmh {r['avg_speed']*3.6:6.0f}kmh  "
              f"{r['frac_grip']*100:3.0f}%/{r['frac_power']*100:2.0f}%")

    fig_track_maps(results)
    fig_speed_traces(results)
    fig_power_sensitivity(veh)
    print(f"\nfigures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
