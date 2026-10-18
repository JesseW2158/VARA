"""Put the juice box in the tray: farthest object first, set down just past the tray's far rim."""

import math

TARGETS = ["juice_box"]
CLEARANCE = 0.02  # m between the tray's rim and the drop point


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    todo = [name for name in TARGETS if name in objs and objs[name]["seen"]]
    if not todo:
        return False
    todo.sort(key=lambda name: -math.hypot(objs[name]["x"], objs[name]["y"]))
    # Set the object down on the table 2 cm past the tray's far rim, on the tray's center line.
    drop_x = tray["x"] + tray["r"] + CLEARANCE
    drop_y = tray["y"]
    for name in todo:
        if not robot.pick(name, transit_z=transit):
            return False
        if not robot.place(drop_x, drop_y, transit_z=transit):
            return False
    robot.home()
    return True
