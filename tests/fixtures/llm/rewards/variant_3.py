"""Reward variant 3: a smoothness penalty that grows as the tip gets close to the target."""
import math

NEAR = 0.05
SMOOTH_MAX = 0.05


def reward(obs, action, prev_action, info):
    dist = float(info["dist"])
    jerk = sum((a - b) ** 2 for a, b in zip(action, prev_action))
    weight = SMOOTH_MAX * NEAR / (dist + NEAR)
    bonus = 1.0 if info.get("success") else 0.0
    return -dist - weight * jerk + bonus
