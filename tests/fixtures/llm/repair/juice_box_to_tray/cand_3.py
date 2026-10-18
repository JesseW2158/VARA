"""Repair: put the juice box in the tray with a transit height, retrying the grasp once."""

TARGET = "juice_box"


def pick_with_retry(robot, name, transit):
    if robot.pick(name, transit_z=transit):
        return True
    robot.open_gripper()
    return robot.pick(name, transit_z=transit)


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    if not pick_with_retry(robot, TARGET, transit):
        return False
    if not robot.place_in_zone("tray", transit_z=transit):
        return False
    robot.home()
    return True
