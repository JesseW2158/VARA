"""Clear the table via a raised outside waypoint, shortest first, clearing existing tray contents."""

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
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    todo = [name for name in objs if objs[name]["seen"]]
    release = tray["wall"] + max([objs[name]["h"] for name in todo] + [0.0]) + 0.01
    for name in sorted(todo, key=lambda n: objs[n]["h"]):
        if not robot.pick(name, transit_z=transit):
            return False
        if not carry_around(robot, objs[name], tray, transit):
            return False
        if not robot.place(tray["x"] - 0.02, tray["y"], release_z=release, approach=0.05, transit_z=transit):
            return False
    robot.home()
    return True
