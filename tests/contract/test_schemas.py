"""Contract: every schema is valid, and every example in docs/interfaces.md matches its schema.

Don't edit without team agreement; change prose, schema and this test together.
"""

import json
import pathlib

import pytest
from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "tests" / "fixtures"

CASES = [
    ("scene.json", "scene.json"),
    ("task_spec.json", "task_spec_clear_table.json"),
    ("task_spec.json", "task_spec_hand_over.json"),
    ("trajectory.json", "trajectory_ok.json"),
    ("score.json", "score_line.json"),
    ("skill.json", "skill_clear_table.json"),
    ("event.json", "event_candidate_scored.json"),
    ("router.json", "router_run.json"),
    ("router.json", "router_learn.json"),
    ("router.json", "router_define_routine.json"),
    ("router.json", "router_queue.json"),
    ("router.json", "router_stop.json"),
]


def load(path):
    return json.loads(path.read_text())


def validator(name):
    return Draft202012Validator(load(SCHEMAS / name))


@pytest.mark.parametrize("name", sorted(p.name for p in SCHEMAS.glob("*.json")))
def test_schema_is_valid(name):
    Draft202012Validator.check_schema(load(SCHEMAS / name))


@pytest.mark.parametrize("schema,fixture", CASES)
def test_example_matches_schema(schema, fixture):
    validator(schema).validate(load(FIXTURES / fixture))


def test_fake_scene_is_a_valid_scene():
    validator("scene.json").validate(load(ROOT / "config" / "fake_scene.json"))


def test_trajectory_waypoints_need_five_joints():
    bad = load(FIXTURES / "trajectory_ok.json")
    bad["segments"][0]["q"][0] = [0.0, 0.0]
    assert list(validator("trajectory.json").iter_errors(bad))


def test_unknown_predicate_is_rejected():
    bad = load(FIXTURES / "task_spec_clear_table.json")
    bad["success"] = {"looks_tidy": {"object": "*"}}
    assert list(validator("task_spec.json").iter_errors(bad))


def test_unknown_router_action_is_rejected():
    assert list(validator("router.json").iter_errors({"action": "dance"}))


def test_unknown_event_stage_is_rejected():
    bad = load(FIXTURES / "event_candidate_scored.json")
    bad["stage"] = "thinking"
    assert list(validator("event.json").iter_errors(bad))
