# Write a skill program for the robot arm

You write one Python program that drives a small robot arm in a physics simulation. The program is scored
over five randomized trials; it passes only when the success test below holds in every trial and the arm
touches nothing it should not. Only the simulated robot API below exists: there is no other library, no file
system and no network, and the program never runs on the real arm (the arm replays the path your program
produced in simulation).

## Robot API (simulation only)

```python
class Robot:
    def scene(self) -> dict: ...                 # current scene, schemas/scene.json
    # Tip to (x, y, z) in meters. pitch: jaw axis from horizontal in radians, -pi/2 = straight down.
    # speed in m/s, capped at 0.2. Returns False if unreachable or outside the workspace.
    def move_to(self, x, y, z, pitch=-math.pi/2, speed=0.15) -> bool: ...
    def set_roll(self, angle) -> None: ...       # gripper rotation about its own axis, radians
    def open_gripper(self) -> None: ...
    def close_gripper(self) -> bool: ...         # True if an object is now held
    def home(self) -> None: ...                  # tucked pose that hides no marker from the camera
    # Helpers. transit_z: if given, rise to that height before any sideways move and travel at it.
    def pick(self, name, approach=0.08, transit_z=None) -> bool: ...
    def place(self, x, y, release_z=0.05, approach=0.08, transit_z=None) -> bool: ...
    def place_in_zone(self, zone, slot=None, transit_z=None) -> bool: ...  # slot None = next free slot
    def carry_to_hand(self) -> bool: ...         # a live primitive on the real arm (see Trajectory)
    def run_skill(self, name, args) -> bool: ... # runs a saved skill; used by routines
```

- Without `transit_z`, a move to a new spot is one joint-space sweep from the current pose, which can pass low over obstacles. That default is deliberate: candidates that notice the obstacle and set `transit_z` score better (Demo script).
- `pick` descends to `max(h/2, 0.02)` with the jaws across the object's width. The sim attaches the object only if the tip is within 1.0 cm of its center line, between 20% and 80% of its height, and within 20° of its yaw; otherwise `close_gripper` returns False.
- `place_in_zone` releases at the zone's wall height plus half the held object's height plus 0.01 m.
- `carry_to_hand` in sim moves above the hand (or the `handoff` zone if the hand isn't seen), opens, backs off 5 cm, goes home, logs `handoff_done`, and records a `live` segment.
- `SimRobot` counts collisions (any contact by the robot or held object except with the object being gripped, including the user's hand box when the hand is seen), joint-limit hits (within 1° of a limit) and IK failures. A trial ends in an error after 60 s of simulated time.

Units are meters, radians and seconds. The origin is on the table directly below the arm's base; +x points
away from the arm toward the user, +y to the arm's left, +z up; the table is z = 0. The workspace box is
x 0.10 to 0.40, y -0.25 to 0.25, z 0.02 to 0.30; a move outside it returns False. The arm's reach is short:
high points (z above 0.25) are reachable only with the jaws near horizontal (the solver relaxes the pitch
for you), and the far corners of the box are out of reach. `slot` in `place_in_zone` is the 0-based index
into the zone's `slots` list. A failed `pick` leaves the object where it was and the gripper open.

## Rules for the program

The static check rejects a file that:

- imports anything but `math`, or has top-level statements other than a docstring, imports, function definitions and assignments of literal values;
- lacks its entry function, by default a top-level `run(robot, args)`;
- uses `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `input`, `breakpoint`, `help`, `dir`, `type`, `object` or `super`;
- uses any name or attribute starting with `_`;
- contains `class`, `global`, `nonlocal`, `async`, `await`, `yield` or `with`;
- has a bare `except:` or catches `BaseException` (the sim's time limit is a `BaseException`);
- is longer than 200 lines.

Traps that have rejected real programs:

- no `if __name__ == "__main__":` block: the runner imports the file and calls `run(robot, args)` itself;
- no name starting with `_`, not even a bare `_` loop variable: write `for step in ...` or `for unused in ...`;
- no `from typing import ...`, no `import numpy`, no import at all other than the exact line `import math`;
- no type annotations that use forbidden names (`args: object` is rejected); leave parameters unannotated;
- no `class`, no `with`, no `yield`, no `global`, no `async`, no `lambda` named `_`;
- no bare `except:` and no `except BaseException`; catch `Exception` if you must catch anything;
- no `return`, `break` or `continue` inside a `finally` block;
- at most 200 lines; every top-level statement is a docstring, `import math`, a `def` or `NAME = <literal>`;
- `run(robot, args)` takes exactly those two parameters and returns True only when it believes the task is done.

## Task

{{TASK_ENGLISH}}

Success test (`sim/success.py` scores every trial with it; `"*"` means every object seen before the run,
and every obstacle must stay where it was):

```json
{{TASK_JSON}}
```

## Current scene

`robot.scene()` returns this structure. Between trials objects are shifted by up to 1 cm and 10 degrees and
obstacles by 0.5 cm, so read positions from `robot.scene()` instead of hard-coding them:

```json
{{SCENE_JSON}}
```

`objects` can be picked (`h` is the height, `size` is `[length along yaw, width across]`; the gripper closes
across the width). `obstacles` are never moved and must not be touched by the arm or by a held object.
The tray has walls `wall` high around its center at radius `r`; `slots` are free spots inside it. `hand` is
the user's wristband, only relevant when the success test asks for `handoff_done`.

## Saved skills, callable through `robot.run_skill(name, args)`

{{SKILLS}}

## Worked examples

{{EXAMPLES}}

## Strategy for this program

{{HINT}}

Write `run(robot, args)` now. Check every return value: when a `pick`, `place`, `place_in_zone` or `move_to`
returns False, return False. Finish with `robot.home()` and return True. Prefer the helpers (`pick`,
`place_in_zone`) over raw `move_to` sequences unless the strategy says otherwise.

Return one Python code block and nothing else.
