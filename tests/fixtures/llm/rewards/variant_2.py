"""Reward variant 2: progress toward the target (the distance closed this step) plus a distance term."""
import math

PROGRESS_GAIN = 10.0
DIST_WEIGHT = 0.1


def reward(obs, action, prev_action, info):
    dist = float(info["dist"])
    prev_dist = float(info.get("prev_dist", dist))
    progress = prev_dist - dist
    bonus = 1.0 if info.get("success") else 0.0
    return PROGRESS_GAIN * progress - DIST_WEIGHT * dist + bonus
