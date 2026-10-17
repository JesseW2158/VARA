"""Clear the table by going around the obstacles: rise, move out to x = 0.32 m, then sideways to the tray."""

OUT_X = 0.32
CRUISE_SPEED = 0.12


def carry_around(robot, obj, tray, transit):
    if not robot.move_to(obj["x"], obj["y"], transit, speed=CRUISE_SPEED):
        return False
    if not robot.move_to(OUT_X, obj["y"], transit, speed=CRUISE_SPEED):
        return False
    return robot.move_to(OUT_X, tray["y"] - tray["r"], transit, speed=CRUISE_SPEED)


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    for name in sorted(objs, key=lambda n: objs[n]["y"]):
        if not objs[name]["seen"]:
            continue
        if not robot.pick(name, transit_z=transit):
            return False
        if not carry_around(robot, objs[name], tray, transit):
            return False
        if not robot.place_in_zone("tray", transit_z=transit):
            return False
    robot.home()
    return True
