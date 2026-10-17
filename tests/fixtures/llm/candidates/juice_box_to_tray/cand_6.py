"""Put the juice box in the tray: fill the far side of the tray first and retry a failed grasp once."""

TARGET = "juice_box"


def slots_far_first(tray):
    # The far side of the tray is the row of slots farthest from the arm (largest y); within a row the
    # nearer slot comes first. A slot the arm cannot reach makes place_in_zone return False and is skipped.
    order = list(range(len(tray["slots"])))
    order.sort(key=lambda i: (-tray["slots"][i][1], tray["slots"][i][0]))
    return order


def pick_with_retry(robot, name, transit):
    if robot.pick(name, transit_z=transit):
        return True
    robot.open_gripper()
    return robot.pick(name, transit_z=transit)


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])
    transit = tallest + 0.08
    if not pick_with_retry(robot, TARGET, transit):
        return False
    placed = False
    for slot in slots_far_first(tray):
        if robot.place_in_zone("tray", slot=slot, transit_z=transit):
            placed = True
            break
    if not placed:
        return False
    robot.home()
    return True
