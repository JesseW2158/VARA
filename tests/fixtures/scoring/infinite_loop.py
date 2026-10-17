"""Scoring fixture: never returns and never advances simulated time, so only the wall clock ends it."""
import math


def run(robot, args):
    while True:
        pass
