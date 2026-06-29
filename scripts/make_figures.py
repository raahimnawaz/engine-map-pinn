"""End-to-end: calibrate engine, generate sparse dyno data, train PINN vs
data-only baseline, and render every figure in figures/.

Run:  PYTHONPATH=src python scripts/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap.engine import Engine, SVJ, SVJ_TT, NM_PER_LBFT
from enginemap.dataset import sample_dyno, dense_truth
from enginemap import pinn

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titlesize": 11})


def fig_wot_validation(na: Engine, tt: Engine):
    rpm, tq, hp = na.wot_curve()
    _, _, hp_tt = tt.wot_curve()
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(rpm, tq / NM_PER_LBFT, color="#1d4ed8", lw=2.2, label="torque (lb-ft)")
    ax2.plot(rpm, hp, color="#c2410c", lw=2.2, label="power (hp)")
    ax2.plot(rpm, hp_tt, color="#c2410c", lw=1.6, ls="--", alpha=0.7,
             label="power — twin-turbo build (hp)")
    # published peak markers
    ax1.scatter([na.spec.peak_torque_rpm], [na.spec.peak_torque_lbft], color="#1d4ed8",
                zorder=5, s=40, edgecolor="white")
    ax2.scatter([na.spec.peak_power_rpm], [na.spec.peak_power_hp], color="#c2410c",
                zorder=5, s=40, edgecolor="white")
    ax1.annotate(f"published {na.spec.peak_torque_lbft:.0f} lb-ft @ {na.spec.peak_torque_rpm:.0f}",
                 (na.spec.peak_torque_rpm, na.spec.peak_torque_lbft), textcoords="offset points",
                 xytext=(8, -16), color="#1d4ed8", fontsize=8.5)
    ax2.annotate(f"published {na.spec.peak_power_hp:.0f} hp @ {na.spec.peak_power_rpm:.0f}",
                 (na.spec.peak_power_rpm, na.spec.peak_power_hp), textcoords="offset points",
                 xytext=(-150, 6), color="#c2410c", fontsize=8.5)
    ax1.set_xlabel("engine speed (rpm)"); ax1.set_ylabel("torque (lb-ft)", color="#1d4ed8")
    ax2.set_ylabel("power (hp)", color="#c2410c")
    ax1.set_title("Calibrated SVJ V12 — wide-open-throttle curve vs published peaks")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{FIG}/wot_validation.png"); plt.close(fig)


def _heat(ax, t, field, cmap, label, levels=18):
    cf = ax.contourf(t["RPM"], t["THR"] * 100, field, levels=levels, cmap=cmap)
    ax.set_xlabel("engine speed (rpm)"); ax.set_ylabel("throttle (%)")
    cb = plt.colorbar(cf, ax=ax); cb.set_label(label)
    return cf


def fig_maps(na: Engine, t, df):
    # torque + power maps with dyno sample coverage
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _heat(axes[0], t, t["torque"] / NM_PER_LBFT, "viridis", "brake torque (lb-ft)")
    axes[0].scatter(df["rpm"], df["throttle"] * 100, s=10, c="white", edgecolor="black",
                    lw=0.4, label="dyno samples", zorder=5)
    axes[0].set_title("Brake-torque map (ground truth) + sparse dyno coverage")
    axes[0].legend(loc="lower right", fontsize=8)
    _heat(axes[1], t, t["power_hp"], "magma", "power (hp)")
    # constant-power lines
    cs = axes[1].contour(t["RPM"], t["THR"] * 100, t["power_hp"],
                         levels=[150, 300, 450, 600, 750], colors="white", linewidths=0.7, alpha=0.7)
    axes[1].clabel(cs, fmt="%d hp", fontsize=7)
    axes[1].set_title("Power map with constant-power contours")
    fig.tight_layout(); fig.savefig(f"{FIG}/torque_power_maps.png"); plt.close(fig)


def fig_bsfc_island(na: Engine, t):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    bsfc = np.clip(t["bsfc"], 200, 600)
    cf = ax.contourf(t["RPM"], t["THR"] * 100, bsfc, levels=24, cmap="RdYlGn_r")
    cs = ax.contour(t["RPM"], t["THR"] * 100, bsfc, levels=[220, 240, 270, 320, 400, 500],
                    colors="black", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, fmt="%d", fontsize=7)
    plt.colorbar(cf, ax=ax).set_label("BSFC (g/kWh) — lower = more efficient")
    # optimal operating line: for each power demand, the min-BSFC point
    powers = np.linspace(60, t["power_hp"].max() * 0.95, 40)
    ool_rpm, ool_thr = [], []
    for P in powers:
        band = np.abs(t["power_hp"] - P) < (t["power_hp"].max() * 0.02)
        if band.sum() == 0:
            continue
        idx = np.argmin(np.where(band, t["bsfc"], 1e9))
        ool_rpm.append(t["RPM"].ravel()[idx]); ool_thr.append(t["THR"].ravel()[idx] * 100)
    ax.plot(ool_rpm, ool_thr, color="#0b2e6b", lw=2.4, label="optimal operating line\n(min BSFC per power)")
    ax.set_xlabel("engine speed (rpm)"); ax.set_ylabel("throttle (%)")
    ax.set_title("BSFC efficiency island + optimal operating line")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/bsfc_island.png"); plt.close(fig)


def fig_pinn_vs_baseline(na: Engine, t, df, base_pred, pinn_pred):
    truth = t["torque"] / NM_PER_LBFT
    be = np.abs(base_pred["torque"] / NM_PER_LBFT - truth)
    pe = np.abs(pinn_pred["torque"] / NM_PER_LBFT - truth)
    vmax = float(np.percentile(be, 99))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    cf0 = axes[0].contourf(t["RPM"], t["THR"] * 100, truth, levels=18, cmap="viridis")
    axes[0].scatter(df["rpm"], df["throttle"] * 100, s=10, c="white", edgecolor="black", lw=0.4, zorder=5)
    axes[0].axhspan(8, 45, color="red", alpha=0.08)
    axes[0].set_title("Ground truth + dyno coverage\n(red band = UNSAMPLED low load)")
    plt.colorbar(cf0, ax=axes[0]).set_label("torque (lb-ft)")
    for ax, e, name in [(axes[1], be, "data-only baseline"), (axes[2], pe, "PINN (data + physics)")]:
        cf = ax.contourf(t["RPM"], t["THR"] * 100, e, levels=18, cmap="inferno", vmin=0, vmax=vmax)
        ax.axhline(45, color="white", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"|torque error| — {name}\nmean {e.mean():.1f} lb-ft")
        plt.colorbar(cf, ax=ax).set_label("abs error (lb-ft)")
    for ax in axes:
        ax.set_xlabel("rpm"); ax.set_ylabel("throttle (%)")
    fig.suptitle("PINN reconstructs the unsampled low-load region; the data-only net does not", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/pinn_vs_baseline.png", bbox_inches="tight"); plt.close(fig)


def main():
    na, tt = Engine(SVJ), Engine(SVJ_TT)
    print("NA peaks:", {k: round(v) for k, v in na.peaks().items()})
    t = dense_truth(na)
    df = sample_dyno(na, seed=0)

    base = pinn.train(na, df, use_physics=False, epochs=4000)
    model = pinn.train(na, df, use_physics=True, epochs=4000, lam=2.0)
    base_pred = pinn.predict_grid(base, t["RPM"], t["THR"])
    pinn_pred = pinn.predict_grid(model, t["RPM"], t["THR"])

    low = t["THR"] < 0.45
    be = np.abs(base_pred["torque"] - t["torque"]); pe = np.abs(pinn_pred["torque"] - t["torque"])
    print(f"torque MAE  baseline {be.mean():.1f} / PINN {pe.mean():.1f} N*m  "
          f"(extrapolation: {be[low].mean():.1f} / {pe[low].mean():.1f})")

    fig_wot_validation(na, tt)
    fig_maps(na, t, df)
    fig_bsfc_island(na, t)
    fig_pinn_vs_baseline(na, t, df, base_pred, pinn_pred)
    print(f"figures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
