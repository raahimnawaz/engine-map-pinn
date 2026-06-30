"""When is the physics actually worth it?

Trains the PINN and the identical data-only net on increasing amounts of
drive-cycle telemetry, and plots torque MAE vs dataset size. The honest story:
physics dominates when data is scarce, and a plain net catches up (and wins)
once you have lots of logged points -- because by then the data covers the map
on its own.

Run:  PYTHONPATH=src python scripts/make_crossover.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enginemap.engine import Engine, SVJ
from enginemap.dataset import dense_truth
from enginemap.telemetry import SimulatedOBD
from enginemap import pinn

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    eng = Engine(SVJ)
    t = dense_truth(eng)
    full = SimulatedOBD(eng, seed=3).log(minutes=40)
    sizes = [40, 80, 160, 320, 800, 2000, 6000]
    rng = np.random.default_rng(0)

    n_seeds = 4
    base_mae, pinn_mae, base_sd, pinn_sd = [], [], [], []
    for n in sizes:
        bs, ps = [], []
        for _ in range(n_seeds):
            df = full.sample(n=n, random_state=int(rng.integers(1e6)))
            b = pinn.train(eng, df, use_physics=False, epochs=2500)
            p = pinn.train(eng, df, use_physics=True, epochs=2500, lam=2.0)
            bs.append(np.abs(pinn.predict_grid(b, t["RPM"], t["THR"])["torque"] - t["torque"]).mean())
            ps.append(np.abs(pinn.predict_grid(p, t["RPM"], t["THR"])["torque"] - t["torque"]).mean())
        base_mae.append(np.mean(bs)); pinn_mae.append(np.mean(ps))
        base_sd.append(np.std(bs)); pinn_sd.append(np.std(ps))
        print(f"  n={n:5d}  baseline {np.mean(bs):5.1f}±{np.std(bs):.1f}  "
              f"PINN {np.mean(ps):5.1f}±{np.std(ps):.1f}  N*m")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(sizes, base_mae, yerr=base_sd, fmt="o-", color="#c2410c", lw=2,
                capsize=3, label="data-only net")
    ax.errorbar(sizes, pinn_mae, yerr=pinn_sd, fmt="s-", color="#1d4ed8", lw=2,
                capsize=3, label="PINN (data + physics)")
    ax.set_xscale("log")
    ax.set_xlabel("logged telemetry samples (log scale)")
    ax.set_ylabel("torque MAE vs ground truth (N·m)")
    ax.set_title("Does physics help on a well-distributed drive log? Mostly no.\n"
                 "Tied (high variance) when tiny; a plain net wins once you log a few hundred points")
    ax.axvspan(sizes[0], 300, color="gray", alpha=0.06)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{FIG}/when_physics_helps.png", bbox_inches="tight")
    plt.close(fig)
    print(f"crossover figure written to {FIG}/when_physics_helps.png")


if __name__ == "__main__":
    main()
