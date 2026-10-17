"""Clear the table: nearest object first, default parameters."""

import math


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    todo = [name for name in objs if objs[name]["seen"]]
    todo.sort(key=lambda name: math.hypot(objs[name]["x"], objs[name]["y"]))
    for name in todo:
        if not robot.pick(name):
            return False
        if not robot.place_in_zone("tray"):
            return False
    robot.home()
    return True
