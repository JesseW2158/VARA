"""Reward variant 1: exponential distance shaping with a small smoothness penalty."""
import math

SCALE = 0.05
SMOOTH = 0.01


def reward(obs, action, prev_action, info):
    dist = float(info["dist"])
    jerk = sum((a - b) ** 2 for a, b in zip(action, prev_action))
    bonus = 1.0 if info.get("success") else 0.0
    return math.exp(-dist / SCALE) - SMOOTH * jerk + bonus
