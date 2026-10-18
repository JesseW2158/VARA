"""Clear the table: farthest object first, rising above the tallest obstacle on every move."""

import math


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    todo = [name for name in objs if objs[name]["seen"]]
    todo.sort(key=lambda name: -math.hypot(objs[name]["x"], objs[name]["y"]))
    for name in todo:
        if not robot.pick(name, transit_z=transit):
            return False
        if not robot.place_in_zone("tray", transit_z=transit):
            return False
    robot.home()
    return True
