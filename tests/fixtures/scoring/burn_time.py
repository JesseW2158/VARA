"""Scoring fixture: moves back and forth until the simulated time limit (a BaseException) ends the trial.
The `except Exception` shows it cannot be swallowed."""
import math


def run(robot, args):
    while True:
        try:
            robot.move_to(0.2, 0.1, 0.15)
            robot.move_to(0.2, -0.1, 0.15)
        except Exception:
            pass
