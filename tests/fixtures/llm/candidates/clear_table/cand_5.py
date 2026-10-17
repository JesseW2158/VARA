"""Clear the table slowly: 0.08 m/s and pitch -pi/4 for targets beyond 0.30 m from the base."""

import math

CRUISE_Z = 0.15
SLOW = 0.08
FAR = 0.30


def pitch_for(x, y):
    if math.hypot(x, y) > FAR:
        return -math.pi / 4
    return -math.pi / 2


def run(robot, args):
    scene = robot.scene()
    objs = scene["objects"]
    tray = scene["zones"]["tray"]
    todo = [name for name in objs if objs[name]["seen"]]
    todo.sort(key=lambda name: math.hypot(objs[name]["x"], objs[name]["y"]))
    for index, name in enumerate(todo):
        obj = objs[name]
        if not robot.pick(name):
            return False
        slot = tray["slots"][index % len(tray["slots"])]
        pitch = pitch_for(slot[0], slot[1])
        if not robot.move_to(obj["x"], obj["y"], CRUISE_Z, pitch=pitch_for(obj["x"], obj["y"]), speed=SLOW):
            return False
        if not robot.move_to(slot[0], slot[1], CRUISE_Z, pitch=pitch, speed=SLOW):
            return False
        release_z = tray["wall"] + obj["h"] / 2 + 0.01
        if not robot.move_to(slot[0], slot[1], release_z, pitch=pitch, speed=SLOW):
            return False
        robot.open_gripper()
        if not robot.move_to(slot[0], slot[1], CRUISE_Z, pitch=pitch, speed=SLOW):
            return False
    robot.home()
    return True
