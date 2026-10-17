"""Contract: sim.success.evaluate(spec, before, after, args, events) -> bool.

Spec: docs/interfaces.md, "Task spec". The same function scores every simulated trial
and checks the camera after the real run. Don't edit without team agreement.
"""

import copy
import json
import pathlib

import pytest

evaluate = pytest.importorskip("sim.success", reason="S0 implements sim/success.py").evaluate

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
TRAY = (0.30, 0.22)
CLEAR = json.loads((FIXTURES / "task_spec_clear_table.json").read_text())["success"]
HAND_OVER = json.loads((FIXTURES / "task_spec_hand_over.json").read_text())["success"]


def scene():
    return json.loads((FIXTURES / "scene.json").read_text())


def put(s, name, x, y, seen=True):
    s = copy.deepcopy(s)
    s["objects"][name].update(x=x, y=y, seen=seen)
    return s


def all_in_tray(s):
    for i, name in enumerate(sorted(s["objects"])):
        s = put(s, name, TRAY[0] + 0.02 * (i - 1), TRAY[1])
    return s


def test_clear_table_true_when_every_object_is_in_the_tray():
    before = scene()
    assert evaluate(CLEAR, before, all_in_tray(before), {}, []) is True


def test_clear_table_false_when_one_object_is_left():
    before = scene()
    after = put(all_in_tray(before), "tea_tin", 0.15, 0.03)
    assert evaluate(CLEAR, before, after, {}, []) is False


def test_spec_is_false_on_the_unchanged_scene():
    before = scene()
    assert evaluate(CLEAR, before, before, {}, []) is False


def test_wildcard_means_objects_seen_in_before():
    before = put(scene(), "tea_tin", 0.15, 0.03, seen=False)
    after = put(all_in_tray(before), "tea_tin", 0.15, 0.03, seen=False)
    assert evaluate(CLEAR, before, after, {}, []) is True


def test_object_unseen_after_makes_in_zone_false():
    before = scene()
    after = all_in_tray(before)
    after["objects"]["juice_box"]["seen"] = False
    assert evaluate(CLEAR, before, after, {}, []) is False


def test_param_substitution():
    spec = {"in_zone": {"object": "$object", "zone": "tray"}}
    before = scene()
    after = put(before, "juice_box", *TRAY)
    assert evaluate(spec, before, after, {"object": "juice_box"}, []) is True
    assert evaluate(spec, before, after, {"object": "tea_tin"}, []) is False


def test_hand_over_needs_the_move_and_the_event():
    before = scene()
    taken = copy.deepcopy(before)
    taken["objects"]["pill_bottle"]["seen"] = False  # the user took it
    args = {"object": "pill_bottle"}
    assert evaluate(HAND_OVER, before, taken, args, ["handoff_done"]) is True
    assert evaluate(HAND_OVER, before, taken, args, []) is False
    assert evaluate(HAND_OVER, before, before, args, ["handoff_done"]) is False


def test_moved_by_distance():
    before = scene()
    spec = {"moved": {"object": "tea_tin", "min": 0.05}}
    assert evaluate(spec, before, put(before, "tea_tin", 0.15, 0.10), {}, []) is True
    assert evaluate(spec, before, put(before, "tea_tin", 0.15, 0.05), {}, []) is False


def test_near_and_not_moved():
    before = scene()
    after = put(before, "tea_tin", 0.20, 0.01)
    near = {"near": {"object": "tea_tin", "x": 0.2, "y": 0.0, "tol": 0.03}}
    assert evaluate(near, before, after, {}, []) is True
    assert evaluate({"not_moved": {"object": "juice_box", "max": 0.02}}, before, after, {}, []) is True
    assert evaluate({"not_moved": {"object": "tea_tin", "max": 0.02}}, before, after, {}, []) is False


def test_all_any_not():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    a = {"in_zone": {"object": "juice_box", "zone": "tray"}}
    b = {"in_zone": {"object": "tea_tin", "zone": "tray"}}
    assert evaluate({"any": [a, b]}, before, after, {}, []) is True
    assert evaluate({"not": b}, before, after, {}, []) is True
    assert evaluate({"all": [a, b]}, before, after, {}, []) is False


def test_moving_an_obstacle_fails_any_spec():
    before = scene()
    after = all_in_tray(before)
    after["obstacles"]["cereal_box"]["x"] += 0.05
    assert evaluate(CLEAR, before, after, {}, []) is False


def test_returns_a_real_bool():
    before = scene()
    assert type(evaluate(CLEAR, before, before, {}, [])) is bool
