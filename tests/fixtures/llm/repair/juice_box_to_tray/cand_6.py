"""Repair: juice box to the far side of the tray at transit height, nearer slots as the fallback."""

TARGET = "juice_box"


def slots_far_first(tray):
    # The far side of the tray is the row of slots farthest from the arm (largest y); within a row the
    # nearer slot comes first. A slot the arm cannot reach makes place_in_zone return False and is skipped.
    order = list(range(len(tray["slots"])))
    order.sort(key=lambda i: (-tray["slots"][i][1], tray["slots"][i][0]))
    return order


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    if not robot.pick(TARGET, transit_z=transit):
        return False
    for slot in slots_far_first(tray):
        if robot.place_in_zone("tray", slot=slot, transit_z=transit):
            robot.home()
            return True
    return False
