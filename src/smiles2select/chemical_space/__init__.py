"""Chemical space projections.

Turns molecules into 2D coordinates a human can navigate. Every projection
records the method, the representation, the parameters and the seed, because a
map without those is not reproducible and its distances are not comparable
with anyone else's.

Proximity on these maps is an approximation. A visual island is not, by itself,
a validated chemical cluster.
"""

PROXIMITY_DISCLAIMER = (
    "Map proximity is an approximate representation. Visual islands do not "
    "automatically constitute validated chemical clusters."
)
