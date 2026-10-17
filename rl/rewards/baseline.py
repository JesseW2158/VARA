"""Baseline hand-off reward (task D2): r = -d - 0.01 * |a - a_prev|^2 + 1[d < 0.03], with d = info["dist"].

A candidate-style file: only `import math`, literal constants and functions, loaded through
`sim.restricted.load_entry(src, entry="reward")` inside the training process, never imported.
`obs` is the 16-float observation of rl/handoff_env.py, `action` and `prev_action` the 4 floats in [-1, 1],
`info` the env's step info (`dist` is the true tip-to-target distance in meters).
"""

import math

SUCCESS_RADIUS = 0.03
SMOOTHNESS = 0.01


def reward(obs, action, prev_action, info):
    d = float(info["dist"])
    jerk = math.fsum((action[i] - prev_action[i]) ** 2 for i in range(len(action)))
    bonus = 1.0 if d < SUCCESS_RADIUS else 0.0
    return -d - SMOOTHNESS * jerk + bonus
