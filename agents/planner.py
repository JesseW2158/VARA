"""Planner for saved skills (task C8; brief 9.3).

`plan(skill, args, scene, *, task_dir, pool)` runs a saved skill once (trial 1 only, the scene as observed)
through `sim.run_candidate` and returns `(trajectory, final_scene, score)`: the recorded trajectory
(validated against schemas/trajectory.json), the scene at the end of the trial (schemas/scene.json) and the
score line (exactly schemas/score.json). It writes `task_dir/skill.json` and `task_dir/scene.json` (the
runner's inputs), lets the runner write `task_dir/plan/` (`traj.json`, `final_scene.json`, `trials.json`,
`score.json`, `runner.out`, `runner.log`) and saves the full scored dict (with the `runner` label, "local" or
"daytona") as `task_dir/plan.json`.

Trusted skills (the two hand-written starters, `skill["trusted"]` true) run in a local subprocess through
`LocalRunner.plan`; every other skill runs through `pool.plan` (the planning sandbox, which falls back to a
local subprocess itself). `pool=None` means local. The skill's code is never imported or executed here.

`PlanError(fail_reason)` is raised when the trial failed (`score["passed"]` false), when no trajectory came
back, or when an input is invalid; `str(exc)` is the one-phrase reason for the `error` event.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any

from jsonschema import Draft202012Validator

from agents.sandbox_pool import PLAN_TIMEOUT, LocalRunner

__all__ = ["PlanError", "plan"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
DEFAULT_SKILLS_DIR = ROOT / "skills"
PLAN_DIR = "plan"

log = logging.getLogger(__name__)


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


_SKILL_VALIDATOR = _validator("skill.json")
_SCENE_VALIDATOR = _validator("scene.json")
_TRAJ_VALIDATOR = _validator("trajectory.json")


class PlanError(Exception):
    """The skill could not be planned; `fail_reason` is the one-phrase reason (also `str(exc)`)."""

    def __init__(self, fail_reason: str) -> None:
        super().__init__(fail_reason)
        self.fail_reason = fail_reason


def _check(validator: Draft202012Validator, doc: Any, what: str) -> None:
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        where = "/".join(str(p) for p in errors[0].absolute_path) or "(root)"
        raise PlanError(f"{what} is invalid at {where}: {errors[0].message}")


def _read_json(path: pathlib.Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"no {what} from the planner ({path.name}: {exc.strerror or exc})") from exc
    except ValueError as exc:
        raise PlanError(f"{what} from the planner is not JSON: {exc}") from exc


def plan(
    skill: dict,
    args: dict | None,
    scene: dict,
    *,
    task_dir: str | os.PathLike[str],
    pool: Any = None,
    skills_dir: str | os.PathLike[str] | None = None,
    timeout: float = PLAN_TIMEOUT,
) -> tuple[dict, dict, dict]:
    """Plan `skill` with `args` on `scene`: (trajectory, final_scene, score). Raises PlanError.

    `pool` is the object from `agents.sandbox_pool.get_pool()` (or anything with the same `plan` method);
    `skills_dir` is where `robot.run_skill` finds other skills (default the repo's `skills/`).
    """
    if not isinstance(skill, dict):
        raise PlanError("skill must be a dict")
    if not isinstance(scene, dict):
        raise PlanError("scene must be a dict")
    args = dict(args or {})
    _check(_SKILL_VALIDATOR, skill, "skill")
    _check(_SCENE_VALIDATOR, scene, "scene")
    task_dir = pathlib.Path(task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    skill_path = task_dir / "skill.json"
    scene_path = task_dir / "scene.json"
    skill_path.write_text(json.dumps(skill, indent=2), encoding="utf-8")
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    out_dir = task_dir / PLAN_DIR
    skills = pathlib.Path(skills_dir) if skills_dir is not None else DEFAULT_SKILLS_DIR

    trusted = bool(skill.get("trusted"))
    runner = LocalRunner() if trusted or pool is None else pool
    log.info(
        "planning %s with %s on %s",
        skill.get("name"),
        args,
        "the local runner" if trusted else type(runner).__name__,
    )
    scored = runner.plan(skill_path, args, scene_path, out_dir, skills, timeout=timeout)
    try:
        (task_dir / "plan.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("could not write plan.json: %s", exc)

    score = scored["score"]
    if not score.get("passed"):
        raise PlanError(
            str(score.get("fail_reason") or score.get("error") or "the skill failed in simulation")
        )
    traj_dir = pathlib.Path(scored.get("out_dir") or out_dir)
    trajectory = _read_json(traj_dir / "traj.json", "trajectory")
    _check(_TRAJ_VALIDATOR, trajectory, "trajectory")
    final_scene = _read_json(traj_dir / "final_scene.json", "final scene")
    _check(_SCENE_VALIDATOR, final_scene, "final scene")
    return trajectory, final_scene, score
