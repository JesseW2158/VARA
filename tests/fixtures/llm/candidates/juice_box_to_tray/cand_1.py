"""Put the juice box in the tray: nearest object first, default parameters."""

import math

TARGETS = ["juice_box"]


def distance_from_base(obj):
    return math.hypot(obj["x"], obj["y"])


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    todo = [name for name in TARGETS if name in objs and objs[name]["seen"]]
    if not todo:
        return False
    todo.sort(key=lambda name: distance_from_base(objs[name]))
    for name in todo:
        if not robot.pick(name):
            return False
        if not robot.place_in_zone("tray"):
            return False
    robot.home()
    return True
