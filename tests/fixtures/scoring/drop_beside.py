"""Scoring fixture: carries the juice box over the cereal box but drops it beside the tray, so the spec
"juice_box in the tray" is not satisfied without any collision. Prints, to show stdout stays clean.
The drop point (0.32, 0.17) is 4 cm past the +x rim of the taped tray zone of 2026-10-18 (center
(0.19, 0.17), r 0.09); the old point (0.17, 0.22) lies inside that zone."""
import math


def run(robot, args):
    print("drop_beside: picking the juice box")
    if not robot.pick("juice_box", transit_z=0.28):
        return False
    print("drop_beside: dropping it beside the tray")
    return robot.place(0.32, 0.17, 0.07, transit_z=0.28)
