"""Scoring fixture: raises KeyboardInterrupt, a BaseException the restricted namespace keeps. It is the
candidate's failure (a trial error, exit 0), never the runner's; the later trials still run."""
import math


def run(robot, args):
    raise KeyboardInterrupt("the candidate pretends to be Ctrl+C")
