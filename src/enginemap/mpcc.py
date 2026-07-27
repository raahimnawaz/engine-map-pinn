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

What the controller is told about the car (v2 #4, in progress). All of these are
the SAME limits the plant enforces, so the plan is physically achievable:

  * friction circle   sqrt(Fx^2 + Fy^2) <= mu*Fz   (grip shared with cornering)
  * engine limit      Fx <= tractive(vx)           (the map, through the gearbox)
  * slip angles       |alpha_f|, |alpha_r| <= slip_frac * peak-slip
                      -> keeps both axles on the stable side of the tyre curve
  * steering rate     |d(delta)| <= ddelta_max*dt, enforced ACROSS solves too
  * track width       hard bound, from k >= 1
  * speed reference   a strong PENALTY toward the grip-feasible QSS profile

Two formulation details matter more than any weight:

  1. State constraints apply from k >= 1 only. X[:, 0] is pinned to the measured
     state, so bounding it makes the NLP infeasible-by-construction the moment
     the car is momentarily outside a limit -- and one unrecoverable solve loses
     the car. It must be free to plan its way back inside.
  2. The speed reference is a penalty, not a bound. As a hard bound it demanded
     impossible deceleration whenever the car was over it. The physical limits
     above stay hard; this one is a planning hint about road the horizon cannot
     see, not a law.

Status: NOT yet a full lap. It drives ~39% of Silverstone cleanly (sub-metre
tracking, <1% solver failures) and then oscillates in the long braking zone into
Stowe, where it is trail-braking at the grip limit. Remaining suspects, in order:
the 0.06 s control interval may be too coarse to stabilise the yaw mode at the
limit (the pure-pursuit controller that does complete laps runs at 0.02 s), and
IPOPT-per-step makes each experiment minutes long -- acados would make this
tractable to tune. See `docs/vehicle-model-v2.md`.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .dynamics import BicycleModel, RHO_AIR, G
from .track import Track, periodic_smooth, smooth_window


def signed_curvature(line: Track):
    """Signed curvature kappa(s) and heading phi(s) for the Frenet frame.

    Smoothed with the SAME window `track.build_track` uses, so this is the track's
    curvature profile with a sign attached -- not a second, spikier estimate. A
    raw finite-difference kappa off a GPS centreline is noisy, and steering
    against those spikes is what made the controller thrash (it can also blow up
    the 1/(1 - n*kappa) Frenet term).
    """
    phi = np.unwrap(np.arctan2(np.gradient(line.y), np.gradient(line.x)))
    raw = np.gradient(phi, line.s)
    return periodic_smooth(raw, smooth_window(line.ds)), phi


def _frenet_ode_np(m: BicycleModel, kappa_s, X, u):
    s, n, al, vx, vy, r = X
    delta, Fx_cmd = u
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
    # same grip/engine limit as dynamics.BicycleModel: the MPCC's plant is the
    # SAME car the pure-pursuit controller drives, not a more permissive one.
    Fx = m._limit_fx(Fx_cmd, Fyf * np.cos(delta) + Fyr, Fz, vx)
    ax = (Fx - Fyf * np.sin(delta) - drag) / m.m + vy * r
    ay = (Fyf * np.cos(delta) + Fyr) / m.m - vx * r
    rd = (m.lf * Fyf * np.cos(delta) - m.lr * Fyr) / m.Iz
    sd = (vx * np.cos(al) - vy * np.sin(al)) / (1 - n * k)
    nd = vx * np.sin(al) + vy * np.cos(al)
    return np.array([sd, nd, r - k * sd, ax, ay, rd])


class MPCC:
    def __init__(self, line: Track, model: BicycleModel, width: float,
                 N: int = 20, dt: float = 0.06, q_lat: float = 1.5, gamma: float = 6.0,
                 fx_max: float = 11000.0, fx_min: float = -24000.0, delta_max: float = 0.45,
                 v_cap=None, v_cap_factor: float = 1.0, slip_frac: float = 0.9,
                 ddelta_max: float = 0.9, q_vcap: float = 8.0):
        self.line, self.m, self.dt, self.N = line, model, dt, N
        ks, _ = signed_curvature(line)
        self.ks_np = lambda s: float(np.interp(s % line.length, line.s, ks))
        kappa = ca.interpolant("k", "bspline", [line.s], ks)
        half = width / 2 - 0.6
        self.half = half  # used by run_lap's off-track divergence guard
        L = model.lf + model.lr

        # Grip-feasible speed ceiling (the QSS profile). This is NOT a softening:
        # v_qss is the friction-circle corner-speed limit with the lap sim's
        # backward braking pass already applied, i.e. the same physics computed
        # with GLOBAL lookahead instead of inside a 1-2 s horizon. It is what lets
        # the controller brake for a corner it cannot yet see -- the finite-horizon
        # blind spot that stopped it getting round. It still has to *achieve* the
        # speed dynamically on the real plant, so the lap-time gap stays honest.
        vcap = None
        if v_cap is not None:
            vcap = ca.interpolant("vcap", "bspline", [line.s],
                                  np.maximum(np.asarray(v_cap, float) * v_cap_factor, 4.0))
        # engine limit, as the plant applies it: drive force capped by the map
        traction = None
        if model.tractive is not None:
            vgrid = np.linspace(0.0, 130.0, 140)
            fgrid = np.array([float(np.ravel(model.tractive(v))[0]) for v in vgrid])
            traction = ca.interpolant("trac", "bspline", [vgrid], fgrid)

        # Slip angle at which the Pacejka curve peaks: sin(C*atan(B*a)) is maximal
        # at a = tan(pi/2C)/B. Past it the tyre gives LESS grip the harder you ask
        # -- the unstable side, where a spin comes from.
        a_peak = min(float(np.tan(np.pi / (2 * model.Cf)) / model.Bf),
                     float(np.tan(np.pi / (2 * model.Cr)) / model.Br))
        alpha_max = slip_frac * a_peak
        self.alpha_max = alpha_max

        def lateral(X, u):
            """Axle slip angles, lateral forces and vertical load — shared by the
            dynamics and the constraints so both see the same tyre."""
            _, _, _, vx, vy, r = ca.vertsplit(X)
            delta, _ = ca.vertsplit(u)
            vx = ca.fmax(vx, 3.0)
            Fz = model.m * G + 0.5 * RHO_AIR * model.cla * vx ** 2
            Fzf, Fzr = Fz * model.lr / L, Fz * model.lf / L
            af = delta - ca.atan2(vy + model.lf * r, vx)
            ar = -ca.atan2(vy - model.lr * r, vx)
            Fyf = Fzf * model.Df * ca.sin(model.Cf * ca.atan(model.Bf * af))
            Fyr = Fzr * model.Dr * ca.sin(model.Cr * ca.atan(model.Br * ar))
            return Fyf, Fyr, Fz, af, ar

        def ode(X, u):
            s, n, al, vx, vy, r = ca.vertsplit(X)
            delta, Fx = ca.vertsplit(u)
            vx = ca.fmax(vx, 3.0); k = kappa(s)
            Fyf, Fyr, _, _, _ = lateral(X, u)
            drag = 0.5 * RHO_AIR * model.cda * vx ** 2
            ax = (Fx - Fyf * ca.sin(delta) - drag) / model.m + vy * r
            ay = (Fyf * ca.cos(delta) + Fyr) / model.m - vx * r
            rd = (model.lf * Fyf * ca.cos(delta) - model.lr * Fyr) / model.Iz
            sd = (vx * ca.cos(al) - vy * ca.sin(al)) / (1 - n * k)
            return ca.vertcat(sd, vx * ca.sin(al) + vy * ca.cos(al), r - k * sd, ax, ay, rd)

        opti = ca.Opti()
        X = opti.variable(6, N + 1); U = opti.variable(2, N)
        X0 = opti.parameter(6)
        # last steering angle actually applied: the rate limit must hold ACROSS
        # solves too, or the applied steering can jump between MPC steps even
        # though each individual plan looks smooth.
        Dprev = opti.parameter(1)
        opti.subject_to(X[:, 0] == X0)
        dd = ddelta_max * dt          # max change in road-wheel angle per step
        opti.subject_to(opti.bounded(-dd, U[0, 0] - Dprev, dd))
        cost = 0
        for k in range(N):
            k1 = ode(X[:, k], U[:, k]); k2 = ode(X[:, k] + dt / 2 * k1, U[:, k])
            k3 = ode(X[:, k] + dt / 2 * k2, U[:, k]); k4 = ode(X[:, k] + dt * k3, U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            cost += q_lat * X[1, k] ** 2 - gamma * (X[0, k + 1] - X[0, k]) + 0.3 * U[0, k] ** 2
            if k > 0:
                cost += 6.0 * (U[0, k] - U[0, k - 1]) ** 2
                # STEERING RATE limit: a real steering actuator has finite speed.
                # Without it the optimiser swings ~0.3 rad (17 deg) in one 60 ms
                # step to chase the line, which flips the yaw rate and spins the
                # car -- the soft penalty above is far too weak to stop it.
                opti.subject_to(opti.bounded(-dd, U[0, k] - U[0, k - 1], dd))
            # --- control bounds: always applicable, U is free at every step ----
            opti.subject_to(opti.bounded(-delta_max, U[0, k], delta_max))
            opti.subject_to(opti.bounded(fx_min, U[1, k], fx_max))
            # FRICTION CIRCLE, the same limit the plant enforces: the controller
            # may only ask for longitudinal force the tyre can still deliver
            # alongside the cornering force it is already using.
            Fyf_k, Fyr_k, Fz_k, af_k, ar_k = lateral(X[:, k], U[:, k])
            Fy_k = Fyf_k * ca.cos(U[0, k]) + Fyr_k
            opti.subject_to(U[1, k] ** 2 + Fy_k ** 2 <= (model.peak_mu * Fz_k) ** 2)
            if traction is not None:                      # and the engine limit
                opti.subject_to(U[1, k] <= traction(ca.fmax(X[3, k], 0.0)))

            # --- state constraints: from k >= 1 ONLY ---------------------------
            # X[:, 0] is pinned to the measured state, so constraining it would
            # make the NLP infeasible-by-construction the moment the car is
            # momentarily outside a bound (over the speed cap, a wheel off the
            # line, rear slip built up). That is a solver failure the controller
            # can never recover from -- it must instead be free to plan its way
            # BACK inside. Rear slip in particular is a pure state quantity: it
            # does not depend on the control at all at k = 0.
            if k >= 1:
                opti.subject_to(opti.bounded(-half, X[1, k], half))   # stay on track
                opti.subject_to(X[3, k] >= 3.0)
                # SLIP-ANGLE limits: keep both axles on the stable, rising side of
                # the tyre curve. Without this the optimiser will command full lock
                # to erase a small contouring error at 190 km/h -- saturating the
                # front, snapping the rear, and spinning the car.
                opti.subject_to(opti.bounded(-alpha_max, af_k, alpha_max))
                opti.subject_to(opti.bounded(-alpha_max, ar_k, alpha_max))

            # GLOBAL SPEED REFERENCE, as a strong PENALTY rather than a bound.
            # The physical limits above (grip, slip, track) stay hard -- this one
            # is a *planning hint* about what is achievable further down the road
            # than the horizon can see, not a law of physics. As a hard bound it
            # made the NLP infeasible whenever the car was momentarily over the
            # cap (it demanded impossible deceleration), and one unrecoverable
            # solve is enough to lose the car. As a penalty it shapes the plan
            # without ever being able to make it unsolvable.
            if vcap is not None:
                cost += q_vcap * ca.fmax(0.0, X[3, k] - vcap(X[0, k])) ** 2
        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 300,
                              "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 6})
        self.opti, self.X, self.U, self.X0 = opti, X, U, X0
        self.Dprev, self._dd = Dprev, dd
        self._warm = False
        self._last_u = np.array([0.0, 0.0])
        self._plan = None      # last feasible input trajectory
        self._plan_k = 0       # how far into it we've fallen back

    def step(self, x_now):
        o = self.opti
        o.set_value(self.X0, x_now)
        o.set_value(self.Dprev, self._last_u[0])
        if not self._warm:
            o.set_initial(self.X[0, :], x_now[0] + np.arange(self.N + 1) * max(x_now[3], 5) * self.dt)
            o.set_initial(self.X[3, :], max(x_now[3], 5)); self._warm = True
        try:
            sol = o.solve()
            o.set_initial(self.X, sol.value(self.X)); o.set_initial(self.U, sol.value(self.U))
            self._plan = np.atleast_2d(sol.value(self.U))   # keep the feasible plan
            self._plan_k = 0
            self._last_u = np.array(sol.value(self.U[:, 0])).ravel()
            return self._last_u, True
        except Exception:
            # Solver failed. Do NOT return the raw debug iterate (often max-lock
            # garbage), and do NOT straighten up: going straight in a corner drives
            # you off the outside, and zeroing the steer also traps the next solve
            # behind the steering-rate limit, so it can never recover.
            # Standard MPC recovery: keep executing the TAIL of the last feasible
            # plan, which was a valid trajectory for very nearly this state.
            self._plan_k += 1
            if self._plan is not None and self._plan_k < self._plan.shape[1]:
                self._last_u = np.array(self._plan[:, self._plan_k]).ravel()
            else:                       # plan exhausted: hold the wheel, ease off
                self._last_u = np.array([self._last_u[0], min(self._last_u[1], 0.0)])
            return self._last_u, False

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
            # bail honestly if the plant diverges (spin / off-track / stalled)
            # rather than grinding out max_steps of garbage solves.
            if not np.isfinite(X).all() or X[3] < 1.0 or abs(X[1]) > self.half + 3.0:
                return {"lap_time": float("inf"), "x": np.array(xs), "y": np.array(ys),
                        "v": np.array(vs), "max_lat": nmax, "fails": fails,
                        "finished": False, "reason": "diverged", "s_reached": float(X[0])}
            if X[0] >= self.line.length:
                return {"lap_time": (it + 1) * self.dt, "x": np.array(xs), "y": np.array(ys),
                        "v": np.array(vs), "max_lat": nmax, "fails": fails,
                        "finished": True, "reason": "completed", "s_reached": float(X[0])}
        return {"lap_time": max_steps * self.dt, "x": np.array(xs), "y": np.array(ys),
                "v": np.array(vs), "max_lat": nmax, "fails": fails,
                "finished": False, "reason": "max_steps", "s_reached": float(X[0])}

    def _plant_step(self, X, u):
        h = self.dt
        k1 = _frenet_ode_np(self.m, self.ks_np, X, u)
        k2 = _frenet_ode_np(self.m, self.ks_np, X + h / 2 * k1, u)
        k3 = _frenet_ode_np(self.m, self.ks_np, X + h / 2 * k2, u)
        k4 = _frenet_ode_np(self.m, self.ks_np, X + h * k3, u)
        return X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
