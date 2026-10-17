## Fix these failures

The programs below were written for this same task and scene. Each one was scored over five trials and
failed; the score line and the failure reason say how. A collision names what was hit ("gripper hit
cereal_box", "held juice_box hit tray"), "spec not satisfied" means the success test was false at the end
of that trial, "error" is an exception inside the program, "timeout" means a trial exceeded 60 simulated
seconds, and "limit hit" means a joint came within 1 degree of its range.

{{FAILURES}}

Study why each one failed, then write a new program that avoids every failure above. Typical fixes: set
`transit_z` to the tallest obstacle plus 0.08 m on every pick and place so the arm rises above obstacles;
use `place_in_zone("tray", ...)` instead of a hand-picked drop point; check every return value; use the
scene's own positions instead of constants.

Strategy for the new program: {{HINT}}

Keep every rule above: only `import math`, no name starting with an underscore, no class or with, at most
200 lines, exactly `run(robot, args)`.

Return one Python code block and nothing else.
