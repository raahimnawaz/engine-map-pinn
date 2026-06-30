"""Model Predictive Contouring Control (MPCC) — the control/autonomy layer.

The QSS lap sim gives the *optimal reference lap* (planning). MPCC is the
closed-loop controller that drives the **dynamic** bicycle model (`dynamics.py`)
to realize it: at each step it solves a short-horizon optimization that
maximizes progress along the reference racing line while staying inside the
track and respecting the tyre/actuator limits, choosing its own speed and line.
It is scored **against the QSS optimal lap as the baseline**.

The formulation is in the **Frenet frame** — states are progress `s` along the
reference, lateral deviation `n`, heading error, and the body velocities — which
is what makes the NLP well-conditioned enough to converge every step (a global-
Cartesian version did not). It's the standard autonomous-racing MPCC (Liniger
et al.). Runs slower than real time in Python (IPOPT per step).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .dynamics import BicycleModel, RHO_AIR, G
from .track import Track


def signed_curvature(line: Track):
    phi = np.unwrap(np.arctan2(np.gradient(line.y), np.gradient(line.x)))
    return np.gradient(phi, line.s), phi


def _frenet_ode_np(m: BicycleModel, kappa_s, X, u):
    s, n, al, vx, vy, r = X
    delta, Fx = u
    vx = max(vx, 3.0)
    k = kappa_s(s)
    L = m.lf + m.lr
    Fz = m.m * G + 0.5 * RHO_AIR * m.cla * vx ** 2
    Fzf, Fzr = Fz * m.lr / L, Fz * m.lf / L
    af = delta - np.arctan2(vy + m.lf * r, vx)
    ar = -np.arctan2(vy - m.lr * r, vx)
    Fyf = Fzf * m.Df * np.sin(m.Cf * np.arctan(m.Bf * af))
    Fyr = Fzr * m.Dr * np.sin(m.Cr * np.arctan(m.Br * ar))
    drag = 0.5 * RHO_AIR * m.cda * vx ** 2
    ax = (Fx - Fyf * np.sin(delta) - drag) / m.m + vy * r
    ay = (Fyf * np.cos(delta) + Fyr) / m.m - vx * r
    rd = (m.lf * Fyf * np.cos(delta) - m.lr * Fyr) / m.Iz
    sd = (vx * np.cos(al) - vy * np.sin(al)) / (1 - n * k)
    nd = vx * np.sin(al) + vy * np.cos(al)
    return np.array([sd, nd, r - k * sd, ax, ay, rd])


class MPCC:
    def __init__(self, line: Track, model: BicycleModel, width: float,
                 N: int = 20, dt: float = 0.06, q_lat: float = 1.5, gamma: float = 6.0,
                 fx_max: float = 11000.0, fx_min: float = -24000.0, delta_max: float = 0.45):
        self.line, self.m, self.dt, self.N = line, model, dt, N
        ks, _ = signed_curvature(line)
        self.ks_np = lambda s: float(np.interp(s % line.length, line.s, ks))
        kappa = ca.interpolant("k", "bspline", [line.s], ks)
        half = width / 2 - 0.6
        L = model.lf + model.lr

        def ode(X, u):
            s, n, al, vx, vy, r = ca.vertsplit(X)
            delta, Fx = ca.vertsplit(u)
            vx = ca.fmax(vx, 3.0); k = kappa(s)
            Fz = model.m * G + 0.5 * RHO_AIR * model.cla * vx ** 2
            Fzf, Fzr = Fz * model.lr / L, Fz * model.lf / L
            af = delta - ca.atan2(vy + model.lf * r, vx)
            ar = -ca.atan2(vy - model.lr * r, vx)
            Fyf = Fzf * model.Df * ca.sin(model.Cf * ca.atan(model.Bf * af))
            Fyr = Fzr * model.Dr * ca.sin(model.Cr * ca.atan(model.Br * ar))
            drag = 0.5 * RHO_AIR * model.cda * vx ** 2
            ax = (Fx - Fyf * ca.sin(delta) - drag) / model.m + vy * r
            ay = (Fyf * ca.cos(delta) + Fyr) / model.m - vx * r
            rd = (model.lf * Fyf * ca.cos(delta) - model.lr * Fyr) / model.Iz
            sd = (vx * ca.cos(al) - vy * ca.sin(al)) / (1 - n * k)
            return ca.vertcat(sd, vx * ca.sin(al) + vy * ca.cos(al), r - k * sd, ax, ay, rd)

        opti = ca.Opti()
        X = opti.variable(6, N + 1); U = opti.variable(2, N)
        X0 = opti.parameter(6)
        opti.subject_to(X[:, 0] == X0)
        cost = 0
        for k in range(N):
            k1 = ode(X[:, k], U[:, k]); k2 = ode(X[:, k] + dt / 2 * k1, U[:, k])
            k3 = ode(X[:, k] + dt / 2 * k2, U[:, k]); k4 = ode(X[:, k] + dt * k3, U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            cost += q_lat * X[1, k] ** 2 - gamma * (X[0, k + 1] - X[0, k]) + 0.3 * U[0, k] ** 2
            if k > 0:
                cost += 6.0 * (U[0, k] - U[0, k - 1]) ** 2
            opti.subject_to(opti.bounded(-half, X[1, k], half))
            opti.subject_to(opti.bounded(-delta_max, U[0, k], delta_max))
            opti.subject_to(opti.bounded(fx_min, U[1, k], fx_max))
            opti.subject_to(X[3, k] >= 3.0)
        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 300,
                              "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 6})
        self.opti, self.X, self.U, self.X0 = opti, X, U, X0
        self._warm = False

    def step(self, x_now):
        o = self.opti
        o.set_value(self.X0, x_now)
        if not self._warm:
            o.set_initial(self.X[0, :], x_now[0] + np.arange(self.N + 1) * max(x_now[3], 5) * self.dt)
            o.set_initial(self.X[3, :], max(x_now[3], 5)); self._warm = True
        try:
            sol = o.solve()
            o.set_initial(self.X, sol.value(self.X)); o.set_initial(self.U, sol.value(self.U))
            return np.array(sol.value(self.U[:, 0])).ravel(), True
        except Exception:
            try:
                return np.array(o.debug.value(self.U[:, 0])).ravel(), False
            except Exception:
                return np.array([0.0, 0.0]), False

    def run_lap(self, v0: float = 35.0, max_steps: int = 6000):
        """Receding-horizon drive of the Frenet dynamic plant around one lap.
        Returns lap time, the (x, y) path, speed, and max lateral deviation."""
        X = np.array([0.0, 0.0, 0.0, v0, 0.0, 0.0])
        u = np.array([0.0, 3000.0])
        _, phi = signed_curvature(self.line)
        xs, ys, vs = [], [], []; nmax = 0.0; fails = 0
        for it in range(max_steps):
            u, ok = self.step(X)
            fails += not ok
            X = self._plant_step(X, u)
            nmax = max(nmax, abs(X[1]))
            i = int(np.searchsorted(self.line.s, X[0] % self.line.length)) % len(self.line.s)
            xs.append(self.line.x[i] - X[1] * np.sin(phi[i]))
            ys.append(self.line.y[i] + X[1] * np.cos(phi[i]))
            vs.append(X[3])
            if X[0] >= self.line.length:
                return {"lap_time": (it + 1) * self.dt, "x": np.array(xs), "y": np.array(ys),
                        "v": np.array(vs), "max_lat": nmax, "fails": fails, "finished": True}
        return {"lap_time": max_steps * self.dt, "x": np.array(xs), "y": np.array(ys),
                "v": np.array(vs), "max_lat": nmax, "fails": fails, "finished": False}

    def _plant_step(self, X, u):
        h = self.dt
        k1 = _frenet_ode_np(self.m, self.ks_np, X, u)
        k2 = _frenet_ode_np(self.m, self.ks_np, X + h / 2 * k1, u)
        k3 = _frenet_ode_np(self.m, self.ks_np, X + h / 2 * k2, u)
        k4 = _frenet_ode_np(self.m, self.ks_np, X + h * k3, u)
        return X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
