"""Clear the table: shortest first, with clearance above objects already in the tray."""


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    tallest = max([o["h"] for o in [*scene["obstacles"].values(), *scene["objects"].values()]] + [0.0])
    transit = tallest + 0.08
    todo = [name for name in objs if objs[name]["seen"]]
    # The far tray slots require the forearm to cross the cereal box. Use the near-center
    # region and release above the accumulated contents, instead of reaching into a full slot.
    release = tray["wall"] + max([objs[name]["h"] for name in todo] + [0.0]) + 0.01
    for name in sorted(todo, key=lambda n: objs[n]["h"]):
        if not robot.pick(name, transit_z=transit):
            return False
        if not robot.place(tray["x"] - 0.02, tray["y"], release_z=release, approach=0.05, transit_z=transit):
            return False
    robot.home()
    return True
