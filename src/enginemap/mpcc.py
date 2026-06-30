"""Model Predictive Contouring Control (MPCC) — the control/autonomy layer.

The QSS lap sim gives the *optimal reference lap* (planning). MPCC is the
closed-loop controller that drives the **dynamic** bicycle model (`dynamics.py`)
to realize it: at each step it solves a short-horizon optimization that
maximizes progress along the reference racing line while staying inside the
track and respecting the tyre/actuator limits. This is the standard autonomous-
racing formulation (Liniger et al., model predictive contouring control), and it
is meant to be evaluated **against the QSS optimal lap as the baseline** — the
lap-time gap is the cost of real dynamics and finite-horizon control.

STATUS — SCAFFOLD. The dynamic model, the MPCC formulation (contouring + lag
cost, progress maximization, track-boundary and tyre/actuator constraints,
IPOPT via CasADi), the warm-started receding-horizon driver, and the
QSS-baseline evaluation harness are all in place and run. The controller tracks
the reference on straighter stretches, but IPOPT does **not yet converge
robustly enough to lap a full circuit** — it loses the line through hard corners.
Making it lap is the documented next step and needs the usual MPCC tuning:
coordinate scaling, a faster real-time solver (acados/OSQP), a Frenet-frame
reformulation, and warm-start chaining. It is honestly a foundation to build on,
not a race-winning controller — and it is reported as such rather than faked.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .dynamics import BicycleModel, RHO_AIR, G
from .track import Track


def build_reference(line: Track):
    """Arc-length interpolants for the reference path (x, y, heading)."""
    th = line.s
    phi = np.unwrap(np.arctan2(np.gradient(line.y), np.gradient(line.x)))
    xr = ca.interpolant("xr", "bspline", [th], line.x)
    yr = ca.interpolant("yr", "bspline", [th], line.y)
    pr = ca.interpolant("pr", "bspline", [th], phi)
    return th, xr, yr, pr


def _ode(m: BicycleModel, X, u):
    x, y, psi, vx, vy, r = ca.vertsplit(X)
    delta, Fx = ca.vertsplit(u)
    vx = ca.fmax(vx, 2.0)
    Fz = m.m * G + 0.5 * RHO_AIR * m.cla * vx ** 2
    Fzf = Fz * m.lr / (m.lf + m.lr); Fzr = Fz * m.lf / (m.lf + m.lr)
    af = delta - ca.atan2(vy + m.lf * r, vx)
    ar = -ca.atan2(vy - m.lr * r, vx)
    Fyf = Fzf * m.Df * ca.sin(m.Cf * ca.atan(m.Bf * af))
    Fyr = Fzr * m.Dr * ca.sin(m.Cr * ca.atan(m.Br * ar))
    drag = 0.5 * RHO_AIR * m.cda * vx ** 2
    ax = (Fx - Fyf * ca.sin(delta) - drag) / m.m + vy * r
    ay = (Fyf * ca.cos(delta) + Fyr) / m.m - vx * r
    rd = (m.lf * Fyf * ca.cos(delta) - m.lr * Fyr) / m.Iz
    return ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi),
                      vx * ca.sin(psi) + vy * ca.cos(psi), r, ax, ay, rd)


class MPCC:
    def __init__(self, line: Track, model: BicycleModel, width: float,
                 N: int = 25, dt: float = 0.06, fx_max: float = 11000.0,
                 fx_min: float = -22000.0, delta_max: float = 0.45,
                 max_iter: int = 400):
        self.line, self.m, self.dt, self.N = line, model, dt, N
        self._warm = False
        self.th, self.xr, self.yr, self.pr = build_reference(line)
        half = width / 2 - 0.6

        opti = ca.Opti()
        X = opti.variable(6, N + 1)      # [x, y, psi, vx, vy, r]
        Th = opti.variable(1, N + 1)     # progress along the reference
        U = opti.variable(2, N)          # [delta, Fx]
        Vth = opti.variable(1, N)        # progress speed
        X0 = opti.parameter(6); T0 = opti.parameter(1); Uprev = opti.parameter(2)

        opti.subject_to(X[:, 0] == X0)
        opti.subject_to(Th[0] == T0)
        cost = 0
        for k in range(N):
            k1 = _ode(model, X[:, k], U[:, k])
            k2 = _ode(model, X[:, k] + dt / 2 * k1, U[:, k])
            k3 = _ode(model, X[:, k] + dt / 2 * k2, U[:, k])
            k4 = _ode(model, X[:, k] + dt * k3, U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            opti.subject_to(Th[k + 1] == Th[k] + Vth[k] * dt)
            xr, yr, ph = self.xr(Th[k]), self.yr(Th[k]), self.pr(Th[k])
            ex, ey = X[0, k] - xr, X[1, k] - yr
            ec = ca.sin(ph) * ex - ca.cos(ph) * ey      # contouring (lateral) error
            el = -ca.cos(ph) * ex - ca.sin(ph) * ey     # lag (along-track) error
            cost += 1.2 * ec ** 2 + 12.0 * el ** 2 - 0.9 * Vth[k] * dt
            cost += 0.4 * U[0, k] ** 2 + 1e-7 * U[1, k] ** 2
            cost += 8.0 * (U[0, k] - (U[0, k - 1] if k > 0 else Uprev[0])) ** 2
            opti.subject_to(ec ** 2 <= half ** 2)
            opti.subject_to(opti.bounded(-delta_max, U[0, k], delta_max))
            opti.subject_to(opti.bounded(fx_min, U[1, k], fx_max))
            opti.subject_to(Vth[k] >= 0)
            opti.subject_to(X[3, k] >= 2.0)
        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                              "ipopt.max_iter": max_iter, "ipopt.tol": 1e-3,
                              "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 8})
        self.opti, self.X, self.Th, self.U, self.Vth = opti, X, Th, U, Vth
        self.X0, self.T0, self.Uprev = X0, T0, Uprev

    def _warm_start(self, x_now, th_now):
        v = max(float(x_now[3]), 5.0)
        ths = np.clip(th_now + np.arange(self.N + 1) * v * self.dt, self.th[0], self.th[-1])
        Xg = np.vstack([np.array(self.xr(ths)).ravel(), np.array(self.yr(ths)).ravel(),
                        np.array(self.pr(ths)).ravel(), np.full(self.N + 1, v),
                        np.zeros(self.N + 1), np.zeros(self.N + 1)])
        Xg[:, 0] = x_now
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.Th, ths)
        self.opti.set_initial(self.Vth, np.full(self.N, v))

    def step(self, x_now, th_now, u_prev):
        """One receding-horizon solve. Returns (control, converged)."""
        o = self.opti
        o.set_value(self.X0, x_now); o.set_value(self.T0, th_now); o.set_value(self.Uprev, u_prev)
        if not self._warm:
            self._warm_start(x_now, th_now); self._warm = True
        try:
            sol = o.solve()
            u0 = sol.value(self.U[:, 0])
            for v, s in ((self.X, sol.value(self.X)), (self.Th, sol.value(self.Th)),
                         (self.U, sol.value(self.U)), (self.Vth, sol.value(self.Vth))):
                o.set_initial(v, s)
            return np.array(u0).ravel(), True
        except Exception:
            try:                      # use the last (non-converged) iterate
                return np.array(o.debug.value(self.U[:, 0])).ravel(), False
            except Exception:
                return np.array(u_prev).ravel(), False
