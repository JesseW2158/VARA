"""Extra edge cases for sim.success beyond tests/contract/test_success.py (S0)."""

import copy
import json
import pathlib

import pytest

from sim.success import OBSTACLE_MAX_MOVE, already_true, evaluate, resolve_object

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
TRAY = (0.30, 0.22)
IN_TRAY = {"in_zone": {"object": "juice_box", "zone": "tray"}}


def scene():
    return json.loads((FIXTURES / "scene.json").read_text())


def hand_over():
    return json.loads((FIXTURES / "task_spec_hand_over.json").read_text())


def put(s, name, x, y, seen=True):
    s = copy.deepcopy(s)
    s["objects"][name].update(x=x, y=y, seen=seen)
    return s


def test_full_task_spec_and_bare_predicate_agree():
    before = scene()
    after = put(before, "pill_bottle", 0.30, 0.0)
    args = {"object": "pill_bottle"}
    assert evaluate(hand_over(), before, after, args, ["handoff_done"]) is True
    assert evaluate(hand_over()["success"], before, after, args, ["handoff_done"]) is True


# --- resolve_object ---


def test_wildcard_excludes_unseen_objects_and_obstacles():
    before = put(scene(), "tea_tin", 0.15, 0.03, seen=False)
    assert resolve_object("*", before, {}) == ["pill_bottle", "juice_box"]
    assert "cereal_box" not in resolve_object("*", scene(), {})


def test_param_substitution_and_a_wildcard_argument():
    assert resolve_object("$object", scene(), {"object": "tea_tin"}) == ["tea_tin"]
    assert resolve_object("$object", scene(), {"object": "*"}) == ["pill_bottle", "juice_box", "tea_tin"]
    assert resolve_object("juice_box", scene(), None) == ["juice_box"]


@pytest.mark.parametrize(
    "ref,args",
    [
        ("$object", {}),
        ("$object", None),
        ("$object", {"object": 3}),
        ("$object", {"object": ""}),
        ("", {}),
        (None, {}),
    ],
)
def test_bad_references_raise_value_error(ref, args):
    with pytest.raises(ValueError):
        resolve_object(ref, scene(), args)


# --- already_true ---


def test_already_true_detects_a_done_task():
    assert already_true(IN_TRAY, put(scene(), "juice_box", *TRAY), {}) is True
    assert already_true(IN_TRAY, scene(), {}) is False
    assert already_true(hand_over(), scene(), {"object": "pill_bottle"}) is False  # no event, nothing moved


# --- malformed specs ---


@pytest.mark.parametrize(
    "pred",
    [{"looks_tidy": {"object": "*"}}, {}, {"all": [IN_TRAY], "any": [IN_TRAY]}, {"all": IN_TRAY}, "in_zone"],
)
def test_malformed_predicates_raise_value_error(pred):
    with pytest.raises(ValueError):
        evaluate(pred, scene(), scene(), {}, [])


def test_unknown_zone_and_zone_without_radius_raise():
    after = put(scene(), "juice_box", *TRAY)
    with pytest.raises(ValueError):
        evaluate({"in_zone": {"object": "juice_box", "zone": "sink"}}, scene(), after, {}, [])
    with pytest.raises(ValueError):
        evaluate({"in_zone": {"object": "juice_box", "zone": "handoff"}}, scene(), after, {}, [])


# --- predicate details ---


def test_in_zone_boundary_is_inclusive():
    before = scene()
    assert evaluate(IN_TRAY, before, put(before, "juice_box", TRAY[0] + 0.07, TRAY[1]), {}, []) is True
    assert evaluate(IN_TRAY, before, put(before, "juice_box", TRAY[0] + 0.0701, TRAY[1]), {}, []) is False


def test_moved_is_true_when_the_object_vanished_from_after():
    before = scene()
    after = copy.deepcopy(before)
    del after["objects"]["pill_bottle"]
    assert evaluate({"moved": {"object": "pill_bottle", "min": 0.05}}, before, after, {}, []) is True


def test_moved_is_false_for_an_object_that_was_never_there():
    assert evaluate({"moved": {"object": "banana", "min": 0.05}}, scene(), scene(), {}, []) is False


def test_near_and_not_moved_are_false_when_unseen():
    before = scene()
    near = {"near": {"object": "tea_tin", "x": 0.2, "y": 0.0, "tol": 0.03}}
    assert evaluate(near, before, put(before, "tea_tin", 0.20, 0.01, seen=False), {}, []) is False
    still = {"not_moved": {"object": "juice_box", "max": 0.02}}
    unseen = put(before, "juice_box", 0.24, -0.04, seen=False)
    assert evaluate(still, before, unseen, {}, []) is False
    assert evaluate(still, unseen, before, {}, []) is False
    assert evaluate(still, before, before, {}, []) is True


def test_wildcard_predicates_apply_to_every_seen_object():
    before = scene()
    all_moved = copy.deepcopy(before)
    for item in all_moved["objects"].values():
        item["x"] += 0.10
    spec = {"moved": {"object": "*", "min": 0.05}}
    assert evaluate(spec, before, all_moved, {}, []) is True
    assert evaluate(spec, before, put(all_moved, "tea_tin", 0.15, 0.03), {}, []) is False
    assert evaluate({"not_moved": {"object": "*", "max": 0.02}}, before, before, {}, []) is True


def test_wildcard_with_no_seen_objects_is_vacuously_true():
    before = scene()
    for item in before["objects"].values():
        item["seen"] = False
    assert evaluate({"in_zone": {"object": "*", "zone": "tray"}}, before, before, {}, []) is True


def test_event_matches_strings_and_event_dicts():
    spec = {"event": "handoff_done"}
    s = scene()
    assert evaluate(spec, s, s, {}, ["handoff_done"]) is True
    assert evaluate(spec, s, s, {}, [{"event": "handoff_done"}]) is True
    assert evaluate(spec, s, s, {}, ["handoff_started"]) is False
    assert evaluate(spec, s, s, {}, None) is False


def test_nested_combinators_return_real_bools():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    tea_in_tray = {"in_zone": {"object": "tea_tin", "zone": "tray"}}
    spec = {"all": [{"any": [IN_TRAY, {"event": "never"}]}, {"not": tea_in_tray}]}
    result = evaluate(spec, before, after, {}, [])
    assert result is True and type(result) is bool
    assert evaluate({"not": spec}, before, after, {}, []) is False


# --- the implicit obstacle rule ---


def test_obstacle_compared_by_position_regardless_of_seen():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    after["obstacles"]["cereal_box"]["seen"] = False
    assert evaluate(IN_TRAY, before, after, {}, []) is True


def test_obstacle_move_limit_is_two_centimeters_inclusive():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    after["obstacles"]["cereal_box"]["x"] += OBSTACLE_MAX_MOVE
    assert evaluate(IN_TRAY, before, after, {}, []) is True
    after["obstacles"]["cereal_box"]["x"] += 0.001
    assert evaluate(IN_TRAY, before, after, {}, []) is False


def test_obstacle_missing_from_after_fails():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    del after["obstacles"]["cereal_box"]
    assert evaluate(IN_TRAY, before, after, {}, []) is False


def test_moved_obstacle_fails_even_a_negated_predicate():
    before = scene()
    after = copy.deepcopy(before)
    after["obstacles"]["cereal_box"]["y"] -= 0.05
    assert evaluate({"not": IN_TRAY}, before, before, {}, []) is True
    assert evaluate({"not": IN_TRAY}, before, after, {}, []) is False


def test_evaluate_does_not_modify_its_inputs():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    spec = copy.deepcopy(IN_TRAY)
    snapshots = (copy.deepcopy(before), copy.deepcopy(after), copy.deepcopy(spec))
    evaluate(spec, before, after, {}, [])
    assert (before, after, spec) == snapshots


# --- "$name" substitution in the zone field (S0 hardening) ---


def test_zone_param_substitution():
    spec = {"in_zone": {"object": "juice_box", "zone": "$zone"}}
    before = scene()
    after = put(before, "juice_box", *TRAY)
    assert evaluate(spec, before, after, {"zone": "tray"}, []) is True
    assert evaluate(spec, before, before, {"zone": "tray"}, []) is False
    both = {"in_zone": {"object": "$object", "zone": "$zone"}}
    assert evaluate(both, before, after, {"object": "juice_box", "zone": "tray"}, []) is True
    assert evaluate(both, before, after, {"object": "tea_tin", "zone": "tray"}, []) is False


def test_missing_zone_arg_raises_like_a_missing_object_arg():
    before = scene()
    after = put(before, "juice_box", *TRAY)
    with pytest.raises(ValueError) as zone_error:
        evaluate({"in_zone": {"object": "juice_box", "zone": "$zone"}}, before, after, {}, [])
    with pytest.raises(ValueError) as object_error:
        evaluate({"in_zone": {"object": "$object", "zone": "tray"}}, before, after, {}, [])
    assert "args has no 'zone'" in str(zone_error.value)
    assert "args has no 'object'" in str(object_error.value)


@pytest.mark.parametrize(
    "args",
    [None, {}, {"zone": ""}, {"zone": 3}, {"zone": "sink"}, {"zone": "handoff"}, {"zone": "*"}],
)
def test_bad_zone_arguments_raise_value_error(args):
    before = scene()
    after = put(before, "juice_box", *TRAY)
    with pytest.raises(ValueError):
        evaluate({"in_zone": {"object": "juice_box", "zone": "$zone"}}, before, after, args, [])


def test_bad_zone_raises_even_when_the_reference_expands_to_no_object():
    before = scene()
    for item in before["objects"].values():
        item["seen"] = False
    with pytest.raises(ValueError):
        evaluate({"in_zone": {"object": "*", "zone": "sink"}}, before, before, {}, [])
    with pytest.raises(ValueError):
        evaluate({"in_zone": {"object": "*", "zone": "$zone"}}, before, before, {}, [])
    with pytest.raises(ValueError):
        already_true({"in_zone": {"object": "*", "zone": "handoff"}}, before, {})


# --- obstacles named explicitly (S0 hardening) ---


def test_explicit_obstacle_name_is_found_in_the_obstacle_table():
    s = scene()
    assert evaluate({"not_moved": {"object": "cereal_box", "max": 0.02}}, s, s, {}, []) is True
    assert evaluate({"moved": {"object": "cereal_box", "min": 0.05}}, s, s, {}, []) is False
    near = {"near": {"object": "cereal_box", "x": 0.24, "y": 0.11, "tol": 0.01}}
    assert evaluate(near, s, s, {}, []) is True
    assert evaluate({"in_zone": {"object": "cereal_box", "zone": "tray"}}, s, s, {}, []) is False
    assert resolve_object("cereal_box", s, {}) == ["cereal_box"]
    assert resolve_object("$object", s, {"object": "cereal_box"}) == ["cereal_box"]


def test_explicit_obstacle_moved_within_the_implicit_limit():
    before = scene()
    after = copy.deepcopy(before)
    after["obstacles"]["cereal_box"]["x"] += 0.01  # allowed by the implicit rule (max 0.02)
    assert evaluate({"moved": {"object": "cereal_box", "min": 0.005}}, before, after, {}, []) is True
    assert evaluate({"not_moved": {"object": "cereal_box", "max": 0.005}}, before, after, {}, []) is False
    assert evaluate({"not_moved": {"object": "cereal_box", "max": 0.02}}, before, after, {}, []) is True


def test_explicit_obstacle_follows_the_seen_rules_of_objects():
    before = scene()
    unseen = copy.deepcopy(before)
    unseen["obstacles"]["cereal_box"]["seen"] = False
    still = {"not_moved": {"object": "cereal_box", "max": 0.02}}
    assert evaluate(still, before, unseen, {}, []) is False  # unseen in after
    assert evaluate(still, unseen, before, {}, []) is False  # unseen in before
    near = {"near": {"object": "cereal_box", "x": 0.24, "y": 0.11, "tol": 0.01}}
    assert evaluate(near, before, unseen, {}, []) is False
    # unseen in after reads as "taken", exactly like an object; the implicit rule still passes (same pose)
    assert evaluate({"moved": {"object": "cereal_box", "min": 0.05}}, before, unseen, {}, []) is True


def test_wildcard_still_never_expands_to_an_obstacle():
    before = scene()
    away = copy.deepcopy(before)
    for item in away["objects"].values():
        item["x"] += 0.10
    # every object moved and the obstacle did not: "*" holds only because it leaves cereal_box out
    assert evaluate({"moved": {"object": "*", "min": 0.05}}, before, away, {}, []) is True
    assert evaluate({"moved": {"object": "$object", "min": 0.05}}, before, away, {"object": "*"}, []) is True
    assert "cereal_box" not in resolve_object("*", before, {})


def test_object_table_wins_over_obstacle_table_for_a_shared_name():
    before = scene()
    before["obstacles"]["juice_box"] = dict(before["obstacles"]["cereal_box"])  # same name in both tables
    after = put(before, "juice_box", *TRAY)  # moves only the objects entry
    assert evaluate(IN_TRAY, before, after, {}, []) is True
    assert evaluate({"not_moved": {"object": "juice_box", "max": 0.02}}, before, after, {}, []) is False


def test_unknown_name_is_false_for_every_object_predicate():
    s = scene()
    assert evaluate({"in_zone": {"object": "banana", "zone": "tray"}}, s, s, {}, []) is False
    assert evaluate({"near": {"object": "banana", "x": 0.0, "y": 0.0, "tol": 1.0}}, s, s, {}, []) is False
    assert evaluate({"not_moved": {"object": "banana", "max": 1.0}}, s, s, {}, []) is False
    assert evaluate({"moved": {"object": "banana", "min": 0.0}}, s, s, {}, []) is False


# --- non-finite coordinates fail closed (S0 hardening round 2) ---

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    "before_x,after_x",
    [(0.24, NAN), (NAN, 0.24), (NAN, NAN), (INF, INF), (0.24, INF), (INF, 0.24)],
)
def test_implicit_obstacle_rule_fails_closed_on_non_finite_coordinates(before_x, after_x):
    before = scene()
    before["obstacles"]["cereal_box"]["x"] = before_x
    after = put(before, "juice_box", *TRAY)
    after["obstacles"]["cereal_box"]["x"] = after_x
    assert (
        evaluate(IN_TRAY, before, after, {}, []) is False
    )  # NaN > limit is False: must not count as unmoved


def test_implicit_obstacle_rule_still_passes_on_finite_coordinates():
    before = scene()
    assert evaluate(IN_TRAY, before, put(before, "juice_box", *TRAY), {}, []) is True


@pytest.mark.parametrize(
    "spec",
    [
        IN_TRAY,
        {"near": {"object": "juice_box", "x": 0.30, "y": 0.22, "tol": 1.0}},
        {"moved": {"object": "juice_box", "min": 0.0}},
        {"not_moved": {"object": "juice_box", "max": 1.0}},
    ],
)
def test_object_predicates_fail_closed_on_nan(spec):
    before = scene()
    after = put(before, "juice_box", NAN, TRAY[1])
    assert evaluate(spec, before, after, {}, []) is False
    assert evaluate(spec, after, after, {}, []) is False


# --- literal pins (final hardening): numbers and rules the mutation harness found unpinned ---


def move_obstacle(s, name, dx=0.0, dy=0.0):
    s = copy.deepcopy(s)
    s["obstacles"][name]["x"] += dx
    s["obstacles"][name]["y"] += dy
    return s


def test_obstacle_move_limit_is_the_literal_two_centimeters():
    assert OBSTACLE_MAX_MOVE == 0.02
    before = scene()
    done = put(before, "juice_box", *TRAY)
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dx=0.02), {}, []) is True
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dy=-0.02), {}, []) is True
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dx=0.021), {}, []) is False
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dy=0.021), {}, []) is False
    # a tenth of a millimeter over the limit already fails: the guard absorbs float noise only
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dx=0.0201), {}, []) is False
    # a distance, not a per-axis box: 0.015 on each axis is 0.0212 away
    assert evaluate(IN_TRAY, before, move_obstacle(done, "cereal_box", dx=0.015, dy=0.015), {}, []) is False
    # the implicit rule guards every spec, an event-only one included
    s = scene()
    event = {"event": "handoff_done"}
    assert evaluate(event, s, move_obstacle(s, "cereal_box", dx=0.02), {}, ["handoff_done"]) is True
    assert evaluate(event, s, move_obstacle(s, "cereal_box", dx=0.021), {}, ["handoff_done"]) is False


def test_near_boundary_is_inclusive_at_the_literal_tolerance():
    before = scene()  # tea_tin at (0.15, 0.03)
    near = {"near": {"object": "tea_tin", "x": 0.20, "y": 0.00, "tol": 0.03}}
    assert evaluate(near, before, put(before, "tea_tin", 0.20, 0.03), {}, []) is True  # exactly tol away
    assert evaluate(near, before, put(before, "tea_tin", 0.23, 0.00), {}, []) is True
    assert evaluate(near, before, put(before, "tea_tin", 0.20, 0.0301), {}, []) is False
    assert evaluate(near, before, put(before, "tea_tin", 0.2301, 0.00), {}, []) is False
    assert evaluate(near, before, put(before, "tea_tin", 0.2213, 0.0213), {}, []) is False  # 0.0301 diagonal
    # the guard absorbs float noise only: one micrometer past the tolerance already fails
    assert evaluate(near, before, put(before, "tea_tin", 0.20, 0.030001), {}, []) is False
    assert evaluate(near, before, put(before, "tea_tin", 0.20, 0.0), {}, []) is True


def test_moved_boundary_is_inclusive_at_the_literal_minimum():
    before = scene()  # tea_tin at (0.15, 0.03)
    spec = {"moved": {"object": "tea_tin", "min": 0.05}}
    assert evaluate(spec, before, put(before, "tea_tin", 0.20, 0.03), {}, []) is True  # exactly min
    assert evaluate(spec, before, put(before, "tea_tin", 0.15, 0.08), {}, []) is True
    assert evaluate(spec, before, put(before, "tea_tin", 0.1999, 0.03), {}, []) is False
    assert evaluate(spec, before, put(before, "tea_tin", 0.15, 0.0799), {}, []) is False
    assert evaluate(spec, before, put(before, "tea_tin", 0.15, 0.03), {}, []) is False  # did not move at all
    assert evaluate({"moved": {"object": "tea_tin", "min": 0.0}}, before, before, {}, []) is True


def test_not_moved_boundary_is_inclusive_at_the_literal_maximum():
    before = scene()  # juice_box at (0.24, -0.04)
    spec = {"not_moved": {"object": "juice_box", "max": 0.02}}
    assert evaluate(spec, before, put(before, "juice_box", 0.26, -0.04), {}, []) is True  # exactly max
    assert evaluate(spec, before, put(before, "juice_box", 0.24, -0.06), {}, []) is True
    assert evaluate(spec, before, put(before, "juice_box", 0.2601, -0.04), {}, []) is False
    assert evaluate(spec, before, put(before, "juice_box", 0.24, -0.0601), {}, []) is False
    assert evaluate({"not_moved": {"object": "juice_box", "max": 0.0}}, before, before, {}, []) is True


@pytest.mark.parametrize(
    "events",
    [
        ["handoff"],  # a prefix of the name
        ["done"],  # a suffix
        ["handoff_done_twice"],  # the name as a prefix of an event
        ["pre_handoff_done"],  # as a suffix
        ["HANDOFF_DONE"],  # another case
        ["handoff_done "],  # trailing space
        [" handoff_done"],
        [{"event": "handoff_done_twice"}],
        [{"event": "handoff"}],
        [{"event": "HANDOFF_DONE"}],
        [{"event": "grip_close", "detail": "handoff_done"}],  # the name under another key of the dict
    ],
)
def test_event_matching_is_exact_not_substring(events):
    s = scene()
    assert evaluate({"event": "handoff_done"}, s, s, {}, events) is False


def test_event_matches_whole_names_anywhere_in_the_log():
    s = scene()
    log = ["grip_close", "handoff_done", "home"]
    assert evaluate({"event": "handoff_done"}, s, s, {}, log) is True
    assert evaluate({"event": "home"}, s, s, {}, log) is True
    assert evaluate({"event": "grip_close"}, s, s, {}, log) is True
    assert evaluate({"event": "grip"}, s, s, {}, log) is False
    assert evaluate({"event": "handoff_done"}, s, s, {}, ("grip_close", {"event": "handoff_done"})) is True
    assert evaluate({"event": "handoff_done"}, s, s, {}, []) is False


def test_already_true_uses_an_empty_event_log():
    s = scene()
    assert already_true({"event": "handoff_done"}, s, {}) is False
    assert already_true({"any": [{"event": "handoff_done"}, {"event": "home"}]}, s, {}) is False
    assert already_true({"not": {"event": "handoff_done"}}, s, {}) is True
    assert already_true({"not": {"event": "home"}}, s, None) is True


def test_already_true_compares_the_scene_with_itself():
    s = scene()
    assert already_true({"not_moved": {"object": "*", "max": 0.0}}, s, {}) is True
    assert already_true({"moved": {"object": "juice_box", "min": 0.000001}}, s, {}) is False
    assert already_true({"near": {"object": "tea_tin", "x": 0.15, "y": 0.03, "tol": 0.0}}, s, None) is True
    # an obstacle anywhere is "where it was": the implicit rule never blocks a done check
    shifted = move_obstacle(s, "cereal_box", dx=0.5)
    assert already_true(IN_TRAY, put(shifted, "juice_box", *TRAY), {}) is True
    snapshot = copy.deepcopy(s)
    already_true(IN_TRAY, s, {})
    assert s == snapshot
