"""Scoring fixture: imports os, which check_source rejects before anything runs."""
import os


def run(robot, args):
    return True
