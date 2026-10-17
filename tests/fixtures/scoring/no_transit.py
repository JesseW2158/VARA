"""Scoring fixture: a pick and place without transit_z, which sweeps through the cereal box."""
import math


def run(robot, args):
    if not robot.pick("juice_box"):
        return False
    return robot.place_in_zone("tray")
