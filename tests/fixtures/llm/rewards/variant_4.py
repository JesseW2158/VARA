"""Reward variant 4: a large success bonus with a per-step time penalty."""
import math

SUCCESS_BONUS = 5.0
TIME_PENALTY = 0.002


def reward(obs, action, prev_action, info):
    dist = float(info["dist"])
    step = float(info.get("step", 0))
    bonus = SUCCESS_BONUS if info.get("success") else 0.0
    return -dist - TIME_PENALTY * step + bonus
