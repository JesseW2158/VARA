"""Put the juice box in the tray by going around the obstacle: rise, move out to x = 0.32 m, then sideways."""

TARGET = "juice_box"
OUT_X = 0.32
CRUISE_SPEED = 0.12


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    box = objs[TARGET]
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    if not robot.pick(TARGET, transit_z=transit):
        return False
    # Rise above every obstacle, go out to OUT_X, then sideways toward the tray at that height.
    if not robot.move_to(box["x"], box["y"], transit, speed=CRUISE_SPEED):
        return False
    if not robot.move_to(OUT_X, box["y"], transit, speed=CRUISE_SPEED):
        return False
    if not robot.move_to(OUT_X, tray["y"] - tray["r"], transit, speed=CRUISE_SPEED):
        return False
    if not robot.place_in_zone("tray", transit_z=transit):
        return False
    robot.home()
    return True
