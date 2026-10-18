"""Repair: put the juice box in the tray going around the obstacle at transit height."""

TARGET = "juice_box"
OUT_X = 0.32


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
    if not robot.move_to(box["x"], box["y"], transit):
        return False
    if not robot.move_to(OUT_X, box["y"], transit):
        return False
    if not robot.move_to(OUT_X, tray["y"] - tray["r"], transit):
        return False
    if not robot.place_in_zone("tray", transit_z=transit):
        return False
    robot.home()
    return True
