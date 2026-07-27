"""Shared physical constants — one definition, imported everywhere.

These used to be duplicated per module, which let the QSS lap sim and the dynamic
model drift apart (three different tyre-grip values for the same car). Anything
describing *the car* or *the air* belongs here so the twin stays coherent.
"""
from __future__ import annotations

RHO_AIR = 1.225   # air density [kg/m^3]
G = 9.81          # gravity [m/s^2]

# Peak tyre friction for the SVJ on track rubber. The QSS sim uses this directly
# as a point-mass mu; the dynamic model's Pacejka axle peaks (Df, Dr) straddle it
# so that its *effective* peak grip -- mean of the axle peaks at equal axle load
# -- comes out to the same number. One tyre, one grip.
MU_TYRE = 1.45

# Rear-minus-front grip bias: the rear axle gets slightly more grip than the
# front so the car understeers at the limit (stable) instead of snapping into
# oversteer. Centred on MU_TYRE, so it changes balance without changing grip.
GRIP_BALANCE = 0.025
