"""Repair: put the juice box in the first tray slot, with a transit height above every obstacle."""

TARGET = "juice_box"
FIRST_SLOT = 0


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    if not robot.pick(TARGET, transit_z=transit):
        return False
    if not robot.place_in_zone("tray", slot=FIRST_SLOT, transit_z=transit):
        return False
    robot.home()
    return True
