# Vehicle model v2 — making the twin coherent

Planning doc, decided before touching core physics. The individual pieces of this
repo (engine calibration, QSS sim, min-curvature line, tyre-ID method) are each
sound. What's wrong is the **integration**: the ML surrogates don't feed the
pipeline, the dynamic plant is missing half its tyre physics, and the modules
disagree on the car's own parameters. This is why the MPCC has been unfixable
in place — you cannot build a realistic controller on a plant that isn't itself
realistic.

This doc records what's wrong, the fixes in order, and the acceptance criteria
for each, so the work is agreed on paper first.

## The core contradiction

The project is presented as a *dyno → track → control digital twin with ML in
the loop*. In the code as it stands:

- the lap sim gets torque from the **analytic** `Engine`, not the PINN;
- every `BicycleModel` is built with **hardcoded** Pacejka defaults, not the
  fitted tyre;
- so **deleting both PINNs changes no lap time and no control output.** The ML
  is illustrative, not load-bearing.

And the plant the controller drives is only half-physical (longitudinal force
bypasses the tyre). The goal of v2 is to make the twin *actually* coherent, so
the claim becomes accurate rather than aspirational.

## Findings (current state)

| # | Problem | Where | Symptom it causes |
|---|---|---|---|
| 1 | `Fx` applied as a direct body force — no friction ellipse, no engine limit | `dynamics.py: continuous`, `mpcc.py: _frenet_ode_np` | plant reaches ~2 g combined (brake+corner); spins / `vy` blow-ups; 11 kN drive at 300 km/h |
| 2 | Engine PINN unused downstream | only `make_figures.py`, `make_crossover.py` | ML not load-bearing |
| 3 | Fitted Pacejka never loaded into a model | only `make_tire_calibration.py` | tyre calibration is a demo |
| 4 | Three grip values for one car | `vehicle.mu=1.45`, `dynamics.Df=1.45/Dr=1.50`, `tire_id.D_true=1.30` | QSS reference and plant disagree on grip; lap-time gap is partly an artifact |
| 5 | Curvature computed two ways | `track.build_track` (smoothed) vs `mpcc.signed_curvature` (unsmoothed, spiky) | erratic MPCC steering; `1/(1−n·κ)` blow-up path |
| 6 | Objective vs horizon mismatch | `mpcc.py` progress term `−γ·Δs` | can't anticipate corners beyond the horizon ("anticipation wall") |

## The fixes, in order

### 1. `Fx` as a grip- and engine-limited tyre force — the keystone

**Now:** `ax = (Fx − Fyf·sinδ − drag)/m`, with `Fx` bounded only by control
constants. Longitudinal force never touches the tyre or the engine.

**Change:** make longitudinal a real tyre force coupled to lateral by a friction
ellipse, and capped by the engine map:

```
Fx_grip = sqrt( max(0, (mu·Fzr)^2 − Fyr^2) )     # combined-slip limit at rear
Fx_drive_max = tractive_force(vx)                # engine map through the gearbox
Fx = clip(Fx_cmd, -Fx_brake_max, min(Fx_grip, Fx_drive_max))
```

(Or, more completely, a longitudinal slip-ratio model feeding a combined-slip
Pacejka. The ellipse-clip above is the minimal faithful version.)

**Why:** the spins came from the plant *permitting* unphysical combined g. Once
the plant can't exceed the tyre, control becomes tractable and the friction-circle
constraint in the controller finally matches the truth it's driving.

**Acceptance:**
- a test asserting that for *any* `(Fx_cmd, delta)`, the realized combined
  acceleration stays within `~mu·g` (+ downforce) — the plant cannot exceed grip;
- braking deceleration capped at a realistic value (grip-limited, not 1.5 g flat);
- drive force at high speed limited by the engine map (no constant 11 kN);
- the existing pure-pursuit controller still completes all three laps.

### 2. Unify the car's parameters

**Change:** one grip source (single `mu`/`D` derived once, shared by the QSS sim
and the dynamic model), and one curvature (use the smoothed `line.kappa`; if the
MPCC needs a signed curvature, sign the smoothed one rather than re-differencing
`x, y`).

**Why:** so "MPCC vs QSS" measures the cost of real dynamics, not a μ mismatch,
and so the controller stops chasing a spiky κ.

**Acceptance:**
- one place defines tyre grip; `vehicle` and `dynamics` both read it;
- QSS and dynamic model agree on peak lateral g to within rounding;
- MPCC curvature is the smoothed profile (bounded `|dκ/ds|`);
- QSS lap times unchanged (this is a consistency fix, not a physics change to the
  reference).

### 3. Wire the surrogates in

**Change:** let `vehicle`/plant take a PINN-predicted engine map, and let
`BicycleModel` be constructed from fitted Pacejka (`tire_id.fit_pacejka`).

**Why:** makes the ML load-bearing and opens the real-data path (swap simulated
telemetry for OBD/IMU and the twin calibrates to an actual car).

**Acceptance:**
- a `BicycleModel.from_pacejka(fit)` (or equivalent) path exists and is used;
- the lap sim can run on the PINN map, not only the analytic engine;
- the "delete the PINN and nothing changes" property no longer holds — a script
  demonstrates the calibrated params flowing through to a lap/control result.

> Note: wiring the tyre-ID in does **not** fix its circularity (it still fits the
> Pacejka form to data generated by that form). Real calibration needs real,
> independent telemetry — that stays a separate roadmap item.

### 4. MPCC on the realistic plant

**Change:** with the plant now physical, give the MPCC global speed anticipation
(a grip-feasible speed ceiling from the QSS profile, or a longer horizon via
acados) instead of pure progress-max over a short window.

**Why:** closes the repo's weakest claim — turns "converges but spins" into a
real full-lap time.

**Acceptance:**
- completes a full clean lap on Silverstone **within the hard track bounds**
  (no slack), reporting gap vs QSS;
- the number is documented in the README, replacing the current hedge.

## What v2 fixes vs. what stays open

**Fixed:** one physically-coherent SVJ (friction ellipse, engine-limited
traction, consistent grip + curvature); the ML becomes load-bearing; the MPCC
drives a realistic plant and reports a defensible lap time. The "digital twin"
claim becomes accurate.

**Still open (honest scope):**
- tyre-ID stays circular until fed real, independent telemetry;
- the engine map is still simulated (no real dyno log exists);
- full-lap MPCC on the Nordschleife is achievable but not guaranteed first-try;
  real-time in Python needs acados;
- absolute lap times remain approximate (bicycle model; no tyre thermal / detailed
  weight transfer);
- the perception → μ(s) idea (predict grip ahead from a forward scan) is a
  separate future layer on top.

## Effort and recommended order

This is a v2 of the vehicle model — days of focused work, not a session. Leverage
is uneven:

- **#1 + #2** deliver ~80% of the benefit (realistic, coherent plant; control
  tractable; honest numbers). **Do these first, reassess.**
- **#3** is the narrative win (ML becomes real); interfaces mostly exist.
- **#4** is the payoff (the lap number); real work regardless.

Start: **#1 → #2**, then decide on #3 / #4.
