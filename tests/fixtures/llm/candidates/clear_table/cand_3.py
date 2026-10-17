"""Clear the table: every pick and place rises to the tallest obstacle plus 0.08 m."""


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    for name in sorted(objs, key=lambda n: objs[n]["x"]):
        if not objs[name]["seen"]:
            continue
        if not robot.pick(name, transit_z=transit):
            return False
        if not robot.place_in_zone("tray", transit_z=transit):
            return False
    robot.home()
    return True
