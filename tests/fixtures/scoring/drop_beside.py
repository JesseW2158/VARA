"""Scoring fixture: carries the juice box over the cereal box but drops it beside the tray, so the spec
"juice_box in the tray" is not satisfied without any collision. Prints, to show stdout stays clean."""
import math


def run(robot, args):
    print("drop_beside: picking the juice box")
    if not robot.pick("juice_box", transit_z=0.28):
        return False
    print("drop_beside: dropping it beside the tray")
    return robot.place(0.17, 0.22, 0.07, transit_z=0.28)
