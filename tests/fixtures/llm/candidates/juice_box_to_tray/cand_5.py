"""Put the juice box in the tray slowly: 0.08 m/s everywhere and pitch -pi/4 for targets beyond 0.30 m."""

import math

TARGET = "juice_box"
SLOW = 0.08
FAR = 0.30


def pitch_for(x, y):
    if math.hypot(x, y) > FAR:
        return -math.pi / 4
    return -math.pi / 2


def far_row_slot(tray):
    # The nearest slot of the tray's far row: the arm stretches away from the obstacle to reach it.
    return min(tray["slots"], key=lambda s: (-s[1], s[0]))


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    if TARGET not in objs or not objs[TARGET]["seen"]:
        return False
    box = objs[TARGET]
    cruise = max([o["h"] for o in scene["obstacles"].values()] + [0.0]) + 0.08
    if not robot.pick(TARGET):
        return False
    slot = far_row_slot(tray)
    pitch = pitch_for(slot[0], slot[1])
    if not robot.move_to(box["x"], box["y"], cruise, pitch=pitch_for(box["x"], box["y"]), speed=SLOW):
        return False
    if not robot.move_to(slot[0], slot[1], cruise, pitch=pitch, speed=SLOW):
        return False
    release_z = tray["wall"] + box["h"] / 2 + 0.01
    if not robot.move_to(slot[0], slot[1], release_z, pitch=pitch, speed=SLOW):
        return False
    robot.open_gripper()
    if not robot.move_to(slot[0], slot[1], cruise, pitch=pitch, speed=SLOW):
        return False
    robot.home()
    return True
