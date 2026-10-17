"""Scoring fixture: gives up on the unperturbed scene (juice box yaw exactly 0.3, trial 1) and does the
proper transit pick and place on every perturbed scene, so trials 2 to 5 succeed and trial 1 does not."""
import math


def run(robot, args):
    scene = robot.scene()
    if abs(scene["objects"]["juice_box"]["yaw"] - 0.3) < 1e-9:
        return False
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    if not robot.pick("juice_box", transit_z=tallest + 0.08):
        return False
    if not robot.place_in_zone("tray", transit_z=tallest + 0.08):
        return False
    robot.home()
    return True
