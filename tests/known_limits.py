"""Reasons for the tests marked as known failures after the 2026-10-18 prop change (strict xfail).

Each mark is strict: when the cause goes away (a taller obstacle, a respaced drop area, a regenerated grasp
site, a fix in arm/real_robot.py) the test passes, the strict mark fails it, and the mark must be removed.
"""

SHORT_OBSTACLE = (
    "2026-10-18 props: the obstacle is a 5.7 cm tape measure, so a candidate without transit_z no longer "
    "hits it; the demo's obstacle-collision story needs an obstacle about 10 cm or taller in the "
    "pick-to-tray path. Remove this mark when such an obstacle returns or the story is rewritten."
)
CROWDED_DROP_AREA = (
    "2026-10-18 props: the three larger props crowd the taped drop area and every clear-table fixture bumps "
    "one (a known limitation, accepted by the human)."
)
FIXED_FINGER_CLEARANCE = (
    "2026-10-18 props: the 4.45 cm pill bottle and box sit 2.75 mm from the fixed finger at the grasp site "
    "(guard 5 mm); fix by moving GRASP_FROM_FIXED_FACE to >= 0.0275 and regenerating the grasp site."
)
RAISED_HAND_HOME = (
    "2026-10-18 props: the lower carry height (0.17 m) makes home refuse after a hand-over to a raised hand "
    "at (0.36, 0, 0.20); the arm stops after releasing and nothing leaves the box. Fix in arm/real_robot.py "
    "(back-off IK branch or a ceiling-aware home plan)."
)
