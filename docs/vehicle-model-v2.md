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

## Status

- [x] **#1** `Fx` as a grip- and engine-limited tyre force
- [x] **#2** unify grip + curvature (one car)
- [x] **#3** wire the surrogates in (`pinn.PINNEngine`, `BicycleModel.from_pacejka`,
      demonstrated end to end by `scripts/make_calibrated_twin.py`)
- [~] **#4** MPCC on the realistic plant — *in progress, see below*

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

## #4 progress log (where to pick it up)

Each of these was a real defect, not a tuning knob. Silverstone, measured as the
fraction of the lap completed before the car left the track:

| Change | Lap reached |
|---|---|
| starting point (after #1–#3) | 3% |
| slip-angle limits on both axles | 4% |
| state constraints from `k >= 1` (recursive feasibility) | 4%, solver failures halved |
| steering-rate limit, enforced across solves | 15% |
| fall back on the tail of the last feasible plan | 15% |
| speed reference as a penalty instead of a hard bound | **38%** |
| stability margin (`slip_frac` 0.75, `gamma` 2.0) | 39% |

At 26% of the lap it was running with **1 solver failure in 500 steps and 0.6 m
tracking** — when the NLP is feasible the controller genuinely works. What is
left is localised, not general.

**Where it fails.** The long braking zone into Stowe (s ≈ 2050–2250 m): 80 → 35
m/s over ~200 m while curvature builds to R ≈ 85 m. Curvature there is smooth, so
it is not a geometry artefact — it is the hardest combined braking-and-cornering
case on the lap. The car enters an oscillation (yaw rate swinging ±1 rad/s at
~0.6 s period, body slip ±8°, steering slewing at its rate limit every step),
then loses feasibility and drifts off.

**Next things to try, in order:**

1. **Finer control interval.** The MPCC runs at `dt = 0.06 s`; the pure-pursuit
   controller that *does* complete laps runs at `0.02 s`. At the limit, 0.06 s is
   plausibly too coarse to stabilise the yaw mode — which would explain an
   oscillation that sits exactly on the steering-rate limit. Try `dt = 0.03`,
   `N = 40` (same horizon, double the rate). This was queued but never ran.
2. **acados instead of IPOPT.** Not a fix in itself, but every experiment above
   costs 3–5 minutes of wall time, which is what makes this slow to converge.
   Codegen + SQP would make tuning tractable.
3. **Revisit the reference line through that corner.** The minimum-*curvature*
   line is a proxy; a min-time line would carry a different (probably lower)
   entry speed into a long corner feeding a straight.

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
  tractable; honest numbers). **Done.**
- **#3** is the narrative win (ML becomes real); interfaces mostly exist. **Done.**
- **#4** is the payoff (the lap number); real work regardless. **In progress.**

## After v2 — where this should go

Ranked by what each actually buys, not by novelty.

### 1. Online grip adaptation (adaptive MPC) — the strongest next idea

Right now `μ` is a fixed number the controller *assumes*. But the car is
continuously generating the exact data needed to measure it: over the last few
seconds it has produced lateral force at known slip angles, which is precisely
what `tire_id.fit_pacejka` consumes.

```
last ~5-10 s of (slip angle, lateral g)  ->  Pacejka fit  ->  updated mu
     ->  MPCC friction circle + slip limits update  ->  drives to the REAL grip
```

Why it matters: on a real car μ is not constant — tyres warm and wear, fuel
burns off, the surface changes, it rains. This also promotes the tyre PINN from
a setup-time calibration to something *in the control loop*.

Note on what this is **not**: feeding raw state history into the NLP would not
help prediction. The plant is Markovian — the current state already summarises
the past, so history adds no predictive information. The value is in estimating
*parameters that drift*, not in remembering trajectories. (The controller does
already use structured history where it matters: warm-starting from the previous
solution, the steering-rate continuity parameter `Dprev`, and the plan-tail
fallback.)

### 2. Hardware

Everything here is simulation. Putting the controller + a calibrated model on a
real small vehicle / Jetson is the capstone, and for a robotics portfolio one
working hardware loop is worth more than several more simulated features.

### 3. Real-data loop (the original roadmap #2)

Wire a real OBD-II + IMU/CAN logger (`telemetry.py` is built for it): real engine
map, real drag from coastdown, real grip from a measured g-g diagram. That turns
this from a generic SVJ into a twin of an *actual* car, and it is also what makes
`tire_id` non-circular (it currently fits the Pacejka form to data generated by
that same form).

### 4. Perception → μ(s) ahead

The ambitious version: predict grip *ahead* from a forward scan (camera/LiDAR
surface classification), feed a spatially-varying `μ(s)` into the MPCC's
constraints, and correct that perception model against the grip actually realised
(section 1 above). This is a genuine perception→action loop.

Honest sequencing: this needs (a) spatially-varying grip in the sim, (b) a scan
observation, and (c) a new vision model — and it should sit on top of an MPCC
that can already complete a lap at *constant* grip. A cheap, honest intermediate
demo: make `μ(s)` spatially varying with a hand-authored low-grip patch and show
the controller slow for it and speed back up. That proves the plumbing without
pretending a perception model exists.
