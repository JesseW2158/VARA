"""Scoring fixture: trial 1 (the unperturbed scene, juice box yaw exactly 0.3) raises an exception carrying a
megabyte of text; every later trial raises one whose str() recurses past the recursion limit. Both must end
as short trial errors (exit 0), never as a broken runner or a megabyte score line."""
import math


def run(robot, args):
    if abs(robot.scene()["objects"]["juice_box"]["yaw"] - 0.3) < 1e-9:
        raise ValueError("x" * 1000000)
    nested = {}
    depth = 0
    while depth < 200000:
        nested = {1: nested}
        depth += 1
    raise ValueError(nested)
