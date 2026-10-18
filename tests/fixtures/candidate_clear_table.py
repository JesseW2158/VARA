import math

def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    for name in sorted(objs, key=lambda n: objs[n]["x"]):
        if not robot.pick(name, transit_z=tallest + 0.08):
            return False
        if not robot.place_in_zone("tray", transit_z=tallest + 0.08):
            return False
    robot.home()
    return True
