"""C8: `agents.planner.plan` on the local runner (trusted starters, real MuJoCo trial) and on a fake pool.

Nothing here touches xAI, Daytona, the arm or a camera; nothing writes into the repo's `skills/` or `runs/`.
"""

from __future__ import annotations

import json
import pathlib
import threading

import pytest
from jsonschema import Draft202012Validator

from agents import planner
from agents import sandbox_pool as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills" / "_starter"
SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text("utf-8"))
DROP_BESIDE = (ROOT / "tests" / "fixtures" / "scoring" / "drop_beside.py").read_text("utf-8")
TRAJ_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "trajectory.json").read_text("utf-8")))
SCENE_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "scene.json").read_text("utf-8")))
SCORE_VALIDATOR = Draft202012Validator(json.loads((ROOT / "schemas" / "score.json").read_text("utf-8")))

MINIMAL_TRAJ = {
    "version": 1,
    "skill": "learned",
    "args": {"object": "juice_box"},
    "dt": 1 / 30,
    "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    "segments": [{"kind": "path", "q": [[0.0, -1.35, 0.02, 1.45, 0.0]]}],
}


def starter(name: str) -> dict:
    return json.loads((STARTER / f"{name}.json").read_text("utf-8"))


def learned(name: str = "place_in_tray", code: str | None = None) -> dict:
    skill = starter(name)
    skill["trusted"] = False
    if code is not None:
        skill["code"] = code
    return skill


class FakePool:
    """Stands in for `SandboxPool`: records `plan` calls and writes canned outputs (or not)."""

    def __init__(
        self, passed: bool = True, write_outputs: bool = True, fail_reason: str | None = None
    ) -> None:
        self.passed = passed
        self.write_outputs = write_outputs
        self.fail_reason = fail_reason
        self.calls: list[dict] = []

    def plan(self, skill_path, args, scene_path, out_dir, skills_dir, timeout=30, *, cancel=None):
        self.calls.append(
            {
                "skill_path": pathlib.Path(skill_path),
                "args": dict(args),
                "scene_path": pathlib.Path(scene_path),
                "out_dir": pathlib.Path(out_dir),
                "skills_dir": None if skills_dir is None else pathlib.Path(skills_dir),
                "timeout": timeout,
            }
        )
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.write_outputs:
            (out / "traj.json").write_text(json.dumps(MINIMAL_TRAJ), encoding="utf-8")
            (out / "final_scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
        score = dict(
            sp.runner_error(0, 1, "unused"),
            passed=self.passed,
            successes=int(self.passed),
            nominal_success=self.passed,
            fail_reason=self.fail_reason if not self.passed else None,
            error=None,
            mean_sim_s=12.0,
            traj=(out / "traj.json").as_posix() if self.write_outputs else None,
        )
        return sp.scored_dict(0, "daytona", score, [self.passed], str(out))

    def score(self, *args, **kwargs):
        raise AssertionError("the planner never scores")


def test_trusted_starters_plan_on_the_local_runner_and_write_the_task_files(tmp_path):
    pool = FakePool()
    results: dict[str, tuple] = {}

    def run(name: str) -> None:
        results[name] = planner.plan(
            starter(name),
            {"object": "juice_box"},
            SCENE,
            task_dir=tmp_path / name,
            pool=pool,
            skills_dir=STARTER,
        )

    threads = [threading.Thread(target=run, args=(name,)) for name in ("place_in_tray", "hand_over")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert pool.calls == []  # trusted skills never go to the pool
    for name in ("place_in_tray", "hand_over"):
        traj, final, score = results[name]
        TRAJ_VALIDATOR.validate(traj)
        SCENE_VALIDATOR.validate(final)
        SCORE_VALIDATOR.validate(score)
        assert traj["skill"] == name and traj["args"] == {"object": "juice_box"}
        assert score["passed"] is True and score["trials"] == 1 and score["candidate"] == 0
        task_dir = tmp_path / name
        assert json.loads((task_dir / "skill.json").read_text("utf-8")) == starter(name)
        assert json.loads((task_dir / "scene.json").read_text("utf-8")) == SCENE
        plan_file = json.loads((task_dir / "plan.json").read_text("utf-8"))
        assert plan_file["runner"] == "local" and plan_file["score"] == score
        assert plan_file["out_dir"] == str(task_dir / "plan")
        assert json.loads((task_dir / "plan" / "traj.json").read_text("utf-8")) == traj
        assert json.loads((task_dir / "plan" / "final_scene.json").read_text("utf-8")) == final
        assert (task_dir / "plan" / "runner.log").is_file()
    assert results["hand_over"][0]["segments"][-1] == {
        "kind": "live",
        "primitive": "carry_to_hand",
        "object": "juice_box",
    }
    assert results["hand_over"][1]["objects"]["juice_box"]["seen"] is False
    assert results["place_in_tray"][1]["objects"]["juice_box"]["seen"] is True


def test_untrusted_skill_plans_through_the_pool(tmp_path):
    pool = FakePool()
    skill = learned()
    traj, final, score = planner.plan(skill, {"object": "juice_box"}, SCENE, task_dir=tmp_path, pool=pool)
    assert len(pool.calls) == 1
    call = pool.calls[0]
    assert call["skill_path"] == tmp_path / "skill.json" and call["scene_path"] == tmp_path / "scene.json"
    assert call["out_dir"] == tmp_path / "plan" and call["args"] == {"object": "juice_box"}
    assert call["skills_dir"] == ROOT / "skills" and call["timeout"] == 30
    assert json.loads((tmp_path / "skill.json").read_text("utf-8")) == skill
    assert traj == MINIMAL_TRAJ and final == SCENE
    assert score["passed"] is True and set(score) == set(sp.runner_error(0, 1, "x"))
    assert json.loads((tmp_path / "plan.json").read_text("utf-8"))["runner"] == "daytona"


def test_pool_none_means_local(tmp_path):
    traj, final, score = planner.plan(
        starter("place_in_tray"), {"object": "juice_box"}, SCENE, task_dir=tmp_path, pool=None
    )
    assert score["passed"] is True
    assert json.loads((tmp_path / "plan.json").read_text("utf-8"))["runner"] == "local"


def test_plan_error_on_a_failing_skill_from_the_pool(tmp_path):
    pool = FakePool(passed=False, fail_reason="trial 1: gripper hit cereal_box")
    with pytest.raises(planner.PlanError) as info:
        planner.plan(learned(), {"object": "juice_box"}, SCENE, task_dir=tmp_path, pool=pool)
    assert str(info.value) == "trial 1: gripper hit cereal_box"
    assert info.value.fail_reason == "trial 1: gripper hit cereal_box"
    assert (tmp_path / "plan.json").is_file()


def test_plan_error_when_no_trajectory_comes_back(tmp_path):
    pool = FakePool(passed=True, write_outputs=False)
    with pytest.raises(planner.PlanError, match="no trajectory"):
        planner.plan(learned(), {"object": "juice_box"}, SCENE, task_dir=tmp_path, pool=pool)


def test_plan_error_on_a_really_failing_trusted_skill(tmp_path):
    # the drop-beside fixture as a trusted skill: it runs locally and misses the tray
    skill = starter("place_in_tray")
    skill["code"] = DROP_BESIDE
    with pytest.raises(planner.PlanError) as info:
        planner.plan(skill, {"object": "juice_box"}, SCENE, task_dir=tmp_path, pool=FakePool())
    assert str(info.value) == "trial 1: spec not satisfied"
    plan_file = json.loads((tmp_path / "plan.json").read_text("utf-8"))
    assert plan_file["runner"] == "local" and plan_file["score"]["passed"] is False


def test_invalid_inputs_are_plan_errors(tmp_path):
    with pytest.raises(planner.PlanError, match="skill is invalid"):
        planner.plan({"name": "x"}, {}, SCENE, task_dir=tmp_path, pool=FakePool())
    with pytest.raises(planner.PlanError, match="scene is invalid"):
        planner.plan(starter("hand_over"), {}, {"t": 1.0}, task_dir=tmp_path, pool=FakePool())
    with pytest.raises(planner.PlanError):
        planner.plan("hand_over", {}, SCENE, task_dir=tmp_path, pool=FakePool())  # type: ignore[arg-type]
    assert not (tmp_path / "skill.json").exists()
