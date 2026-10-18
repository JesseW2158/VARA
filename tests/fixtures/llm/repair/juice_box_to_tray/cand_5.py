"""Repair: put the juice box in the tray slowly, with every move at transit height above the obstacles."""

TARGET = "juice_box"
SLOW = 0.08


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    box = objs[TARGET]
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    if not robot.pick(TARGET, transit_z=transit):
        return False
    if not robot.move_to(box["x"], box["y"], transit, speed=SLOW):
        return False
    if not robot.move_to(tray["x"], tray["y"] - tray["r"], transit, speed=SLOW):
        return False
    if not robot.place_in_zone("tray", transit_z=transit):
        return False
    robot.home()
    return True
