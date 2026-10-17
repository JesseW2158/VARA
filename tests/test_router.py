"""agents.router: the 20 fixture phrases in fake mode, the keyword fallback and the scene checks (C2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from agents import llm, router
from sim.success import already_true

ROOT = Path(__file__).resolve().parents[1]
PHRASES = json.loads((ROOT / "tests" / "fixtures" / "router_phrases.json").read_text(encoding="utf-8"))
SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
TASK_SPEC = jsonschema.Draft202012Validator(
    json.loads((ROOT / "schemas" / "task_spec.json").read_text(encoding="utf-8"))
)
SKILLS = router.EVAL_SKILLS
PLACE_IN_TRAY = {
    "name": "place_in_tray",
    "description": "Put the named object into the tray.",
    "params": {"object": "object"},
}
HAND_OVER_PILLS = {"action": "run", "skill": "hand_over", "args": {"object": "pill_bottle"}}
GOOD_MORNING_STEPS = [
    {"skill": "hand_over", "args": {"object": "pill_bottle"}},
    {"skill": "hand_over", "args": {"object": "juice_box"}},
]


@pytest.fixture(autouse=True)
def fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.setattr(llm, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def no_dotenv(monkeypatch):
    """`main()` loads `.env` like every CLI; a developer's `.env` must never leak keys or flags into the
    test process, so the loader is replaced and its calls counted."""
    calls: list[dict] = []

    def load_dotenv(*args, **kwargs):
        calls.append(dict(kwargs))
        return False

    monkeypatch.setattr("dotenv.load_dotenv", load_dotenv)
    return calls


def _learn(spec):
    return lambda *args, **kwargs: {"action": "learn", "spec": copy.deepcopy(spec)}


def _reply(value):
    return lambda *args, **kwargs: copy.deepcopy(value)


def _raise(exc):
    def down(*args, **kwargs):
        raise exc

    return down


# --- the fixture phrases ------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", PHRASES, ids=[p["text"] for p in PHRASES])
def test_fixture_phrase_routes_as_expected(entry):
    result = router.route(entry["text"], SKILLS, SCENE)
    router.VALIDATOR.validate(result)
    assert router.compare(entry, result) is None, router.compare(entry, result)


def test_fixture_file_shape():
    assert len(PHRASES) == 20
    assert sum(1 for p in PHRASES if p["action"] == "queue") == 5
    texts = [p["text"] for p in PHRASES]
    assert len(set(texts)) == 20
    for text in (
        "Hand me my pills",
        "Clear the table. Put everything in the tray.",
        "Fold my laundry",
        "When I say good morning, hand me my pill bottle, then my juice box",
        "Good morning",
        "put the juice box in the tray",
        "clear the table",
        "stop",
    ):
        assert text in texts, text
    by_text = {p["text"]: p for p in PHRASES}
    juice = by_text["put the juice box in the tray"]
    assert juice["action"] == "learn" and juice["spec_name"] == "juice_box_to_tray"
    assert juice["success"] == {"in_zone": {"object": "juice_box", "zone": "tray"}} and juice["params"] == {}
    clear = by_text["clear the table"]
    assert clear["spec_name"] == "clear_table"
    assert clear["success"] == {"all": [{"in_zone": {"object": "*", "zone": "tray"}}]}
    assert by_text["hand me my pills".title()[0] + "and me my pills"]["skill"] == "hand_over"
    assert (
        by_text["When I say good morning, hand me my pill bottle, then my juice box"]["name"]
        == "good_morning"
    )
    assert by_text["Good morning"]["skill"] == "good_morning"
    assert by_text["stop"]["action"] == "stop"


def test_every_fixture_learn_spec_is_valid_and_not_already_true():
    learned = 0
    for entry in PHRASES:
        if entry["action"] != "learn":
            continue
        result = router.route(entry["text"], SKILLS, SCENE)
        TASK_SPEC.validate(result["spec"])
        assert result["spec"]["params"] == {}
        assert already_true(result["spec"], SCENE, {}) is False
        learned += 1
    assert learned >= 5


def test_eval_cli_in_fake_mode(capsys, no_dotenv):
    assert router.main(["--eval", "--fake"]) == 0
    out = capsys.readouterr().out
    assert "20/20 phrases routed as expected" in out
    assert "FAIL" not in out
    assert len(no_dotenv) == 1  # the CLI loads .env exactly once (and the test never let it read one)


# --- the keyword router -------------------------------------------------------------------------------


def test_keyword_route_demo_phrases():
    skills = [*SKILLS, PLACE_IN_TRAY]
    assert router.keyword_route("Hand me my pills", skills) == HAND_OVER_PILLS
    assert router.keyword_route("Clear the table. Put everything in the tray.", skills) == {
        "action": "learn",
        "spec": router.CLEAR_TABLE_SPEC,
    }
    assert router.keyword_route("Fold my laundry", skills)["action"] == "queue"
    assert router.keyword_route(
        "When I say good morning, hand me my pill bottle, then my juice box", skills
    ) == {
        "action": "define_routine",
        "name": "good_morning",
        "steps": GOOD_MORNING_STEPS,
    }
    assert router.keyword_route("Good morning", skills) == {
        "action": "run",
        "skill": "good_morning",
        "args": {},
    }
    assert router.keyword_route("stop", skills) == {"action": "stop"}
    assert router.keyword_route("Stop right now!", skills) == {"action": "stop"}
    assert router.keyword_route("put the juice box in the tray", skills) == {
        "action": "run",
        "skill": "place_in_tray",
        "args": {"object": "juice_box"},
    }


def test_keyword_route_without_the_tray_skill_learns():
    result = router.keyword_route("put the juice box in the tray", SKILLS)
    assert result["action"] == "learn"
    assert result["spec"]["name"] == "juice_box_to_tray"
    assert result["spec"]["success"] == {"in_zone": {"object": "juice_box", "zone": "tray"}}
    TASK_SPEC.validate(result["spec"])
    clear = router.keyword_route(
        "clear the table", [*SKILLS, {"name": "clear_table", "description": "", "params": {}}]
    )
    assert clear == {"action": "run", "skill": "clear_table", "args": {}}


def test_keyword_route_unknown_phrase_is_none():
    assert router.keyword_route("what time is it", SKILLS) is None
    assert router.keyword_route("", SKILLS) is None
    assert router.keyword_route("hand me the remote", SKILLS) is None  # no known object


def test_keyword_route_is_not_over_eager():
    """Regression: an object word plus "tray" became a learn, and "stop" anywhere became stop."""
    skills = [*SKILLS, PLACE_IN_TRAY]
    for text in (
        "keep the juice out of the tray",
        "take the juice box out of the tray",
        "do not put the juice in the tray",
        "leave the pill bottle in the tray",
        "is the juice box in the tray",
    ):
        assert router.keyword_route(text, skills) is None, text
    for text in (
        "put the juice box in the tray",
        "get the juice in the tray",
        "move the pills into the tray",
    ):
        result = router.keyword_route(text, skills)
        assert result and result["action"] == "run" and result["skill"] == "place_in_tray", text
    for text in (
        "put the juice box next to the bus stop sign today",
        "move the juice box to the stop line on the tray please",
    ):
        result = router.keyword_route(text, skills)
        assert result is None or result["action"] != "stop", text
    for text in ("stop", "Stop right now!", "please stop what you are doing right now", "robot, stop moving"):
        assert router.keyword_route(text, skills) == {"action": "stop"}, text


def test_keyword_route_guards_every_rule():
    """Negations, compound sentences and questions get None from every rule, not only the tray rule.

    Regression: "don't hand me the pills" ran hand_over, "put everything in the tray except the tea tin"
    learned clear_table (the tea tin included), and "hand me the pills, then put the juice box in the
    tray" ran hand_over and dropped the second half. A phrase table cannot split a sentence; the model
    (or the queue) decides those.
    """
    assert router.COMPOUND == {"except", "then", "after", "before", "but"}
    skills = [*SKILLS, PLACE_IN_TRAY, {"name": "clear_table", "description": "", "params": {}}]
    for text in (
        "don't hand me the pills",
        "never hand me the tea tin",
        "hand me the pills but not the juice",
        "hand me the pills, then put the juice box in the tray",
        "hand me the pills after you clear the table",
        "before you clear the table, hand me the pills",
        "do not clear the table",
        "clear the table but leave the cereal box",
        "clear the table except for the tea tin",
        "put everything in the tray except the tea tin",
        "put the juice box in the tray, then hand me the pills",
        "don't fold my laundry",
        "fold my laundry, then hand me the pills",
        "is the juice box in the tray",
        "where are my pills",
        "when I say good morning, don't hand me the pills",
        "when I say tidy up, put everything in the tray except the tea tin",
    ):
        assert router.keyword_route(text, skills) is None, text
    # the same requests without the guard words still route
    assert router.keyword_route("hand me the pills", skills) == HAND_OVER_PILLS
    run_clear = {"action": "run", "skill": "clear_table", "args": {}}
    assert router.keyword_route("clear the table", skills) == run_clear
    assert router.keyword_route("put everything in the tray", skills) == run_clear
    assert router.keyword_route("fold my laundry", skills)["action"] == "queue"
    # inside a routine, "then" separates the steps
    phrase = "When I say good morning, hand me my pill bottle, then my juice box"
    assert router.keyword_route(phrase, skills) == {
        "action": "define_routine",
        "name": "good_morning",
        "steps": GOOD_MORNING_STEPS,
    }
    # stop is the exception: the safety action is never argued away by what follows it
    for text in ("stop, but first put the juice box down", "please stop, then go home"):
        assert router.keyword_route(text, skills) == {"action": "stop"}, text


def test_keyword_route_every_output_validates():
    for entry in PHRASES:
        result = router.keyword_route(entry["text"], [*SKILLS, PLACE_IN_TRAY])
        if result is not None:
            router.VALIDATOR.validate(result)


def test_mentioned_objects_in_order():
    assert router.mentioned_objects("hand me my pill bottle, then my juice box") == [
        "pill_bottle",
        "juice_box",
    ]
    assert router.mentioned_objects("the tea tin and the pills") == ["tea_tin", "pill_bottle"]
    assert router.mentioned_objects("nothing here") == []


# --- the scene and skill checks after the model reply -------------------------------------------------


def test_already_true_spec_becomes_queue(monkeypatch):
    spec = {
        "name": "keep_tea_tin",
        "description": "Leave the tea tin where it is.",
        "params": {},
        "success": {"not_moved": {"object": "tea_tin", "max": 0.02}},
        "english": "The tea tin stays put.",
    }
    monkeypatch.setattr(router, "chat", _learn(spec))
    result = router.route("leave the tea tin alone", SKILLS, SCENE)
    assert result == {"action": "queue", "task": "leave the tea tin alone", "reason": router.ALREADY_DONE}


def test_value_error_from_already_true_becomes_queue(monkeypatch):
    spec = router.CLEAR_TABLE_SPEC
    monkeypatch.setattr(router, "chat", _learn(spec))

    def boom(*args, **kwargs):
        raise ValueError("unknown zone")

    monkeypatch.setattr(router, "already_true", boom)
    result = router.route("clear the table", SKILLS, SCENE)
    assert result == {"action": "queue", "task": "clear the table", "reason": router.NO_SUCCESS_TEST}


def test_dollar_param_in_learn_spec_becomes_queue(monkeypatch):
    spec = {
        "name": "to_tray",
        "description": "Put an object in the tray.",
        "params": {},
        "success": {"in_zone": {"object": "$object", "zone": "tray"}},
        "english": "The object ends in the tray.",
    }
    monkeypatch.setattr(router, "chat", _learn(spec))
    result = router.route("put it in the tray", SKILLS, SCENE)
    assert result["action"] == "queue" and result["reason"] == router.NO_SUCCESS_TEST
    # with the matching parameter declared, the params rule speaks first
    monkeypatch.setattr(router, "chat", _learn(dict(spec, params={"object": "object"})))
    result = router.route("put it in the tray", SKILLS, SCENE)
    assert result["action"] == "queue" and result["reason"] == router.LEARN_HAS_PARAMS


def test_learn_spec_with_params_becomes_queue(monkeypatch):
    """A learn spec carries no args (the router output for `learn` has none), so a parameter could never
    be filled: refused with a spoken reason even when every predicate is concrete."""
    for params in ({"object": "object"}, {"zone": "zone"}, {"count": "number"}):
        monkeypatch.setattr(router, "chat", _learn(dict(router.CLEAR_TABLE_SPEC, params=params)))
        result = router.route("clear the table", SKILLS, SCENE)
        expected = {"action": "queue", "task": "clear the table", "reason": router.LEARN_HAS_PARAMS}
        assert result == expected, params
    monkeypatch.setattr(router, "chat", _learn(router.CLEAR_TABLE_SPEC))
    assert router.route("clear the table", SKILLS, SCENE)["action"] == "learn"


def test_star_with_nothing_seen_becomes_queue(monkeypatch):
    """`"*"` expands to the objects the camera sees. With none, `already_true` would call a clear-table
    spec vacuously done ("That already looks done.", a regression); the user hears what is wrong instead."""
    monkeypatch.setattr(router, "chat", _learn(router.CLEAR_TABLE_SPEC))
    expected = {"action": "queue", "task": "clear the table", "reason": router.NOTHING_SEEN}
    unseen = copy.deepcopy(SCENE)
    for item in unseen["objects"].values():
        item["seen"] = False
    assert router.route("clear the table", SKILLS, unseen) == expected
    unseen["objects"] = {}
    assert router.route("clear the table", SKILLS, unseen) == expected
    assert router.route("clear the table", SKILLS, SCENE)["action"] == "learn"


def test_handoff_zone_in_in_zone_becomes_queue(monkeypatch):
    spec = {
        "name": "juice_to_hand",
        "description": "Put the juice box at the hand-off spot.",
        "params": {},
        "success": {"in_zone": {"object": "juice_box", "zone": "handoff"}},
        "english": "The juice box is at the hand-off spot.",
    }
    monkeypatch.setattr(router, "chat", _learn(spec))
    result = router.route("move the juice to the handoff spot", SKILLS, SCENE)
    assert result["action"] == "queue"


def test_unknown_object_or_obstacle_in_spec_becomes_queue(monkeypatch):
    cup = dict(router.CLEAR_TABLE_SPEC, success={"in_zone": {"object": "cup", "zone": "tray"}})
    monkeypatch.setattr(router, "chat", _learn(cup))
    result = router.route("put the cup in the tray", SKILLS, SCENE)
    assert result == {
        "action": "queue",
        "task": "put the cup in the tray",
        "reason": "I don't see a cup on the table.",
    }
    cereal = dict(router.CLEAR_TABLE_SPEC, success={"in_zone": {"object": "cereal_box", "zone": "tray"}})
    monkeypatch.setattr(router, "chat", _learn(cereal))
    result = router.route("put the cereal box in the tray", SKILLS, SCENE)
    assert result["action"] == "queue" and "cereal box" in result["reason"]


def test_obstacle_in_any_predicate_becomes_queue(monkeypatch):
    """An obstacle never appears in a learn predicate, `not_moved` included.

    Regression: `not_moved cereal_box` passed the router. The evaluator pins every obstacle anyway, and
    the camera check (`vision.verify.missing_objects`) looks names up among the objects only, so the
    spec would have read as "unseen after the run: cereal_box" on the real table every time.
    """
    pinned = {"not_moved": {"object": "cereal_box", "max": 0.02}}
    predicates = [
        {"in_zone": {"object": "cereal_box", "zone": "tray"}},
        {"near": {"object": "cereal_box", "x": 0.2, "y": 0.0, "tol": 0.03}},
        {"moved": {"object": "cereal_box", "min": 0.05}},
        pinned,
        {"all": [{"in_zone": {"object": "*", "zone": "tray"}}, pinned]},
        {"any": [{"in_zone": {"object": "juice_box", "zone": "tray"}}, {"not": pinned}]},
    ]
    for success in predicates:
        monkeypatch.setattr(router, "chat", _learn(dict(router.CLEAR_TABLE_SPEC, success=success)))
        result = router.route("clear the table but leave the cereal", SKILLS, SCENE)
        assert result["action"] == "queue" and "cereal box" in result["reason"], success
    monkeypatch.setattr(router, "chat", _learn(dict(router.CLEAR_TABLE_SPEC, success=pinned)))
    assert router.route("leave the cereal", SKILLS, SCENE)["reason"] == (
        "The cereal box always stays where it is; I can't put it in a task."
    )
    # pinning an object (not an obstacle) is still a fine learn spec
    tea = {
        "all": [
            {"in_zone": {"object": "juice_box", "zone": "tray"}},
            {"not_moved": {"object": "tea_tin", "max": 0.02}},
        ]
    }
    monkeypatch.setattr(router, "chat", _learn(dict(router.CLEAR_TABLE_SPEC, success=tea)))
    assert router.route("put the juice in the tray, leave the tea", SKILLS, SCENE)["action"] == "learn"


def test_run_with_unknown_skill_becomes_queue(monkeypatch):
    monkeypatch.setattr(router, "chat", _reply({"action": "run", "skill": "dance", "args": {}}))
    result = router.route("dance for me", SKILLS, SCENE)
    assert result == {
        "action": "queue",
        "task": "dance for me",
        "reason": "I don't have a skill called dance.",
    }


def test_run_with_missing_or_unknown_object_becomes_queue(monkeypatch):
    monkeypatch.setattr(router, "chat", _reply({"action": "run", "skill": "hand_over", "args": {}}))
    assert router.route("hand it over", SKILLS, SCENE)["action"] == "queue"
    monkeypatch.setattr(
        router, "chat", _reply({"action": "run", "skill": "hand_over", "args": {"object": "cup"}})
    )
    result = router.route("hand me the cup", SKILLS, SCENE)
    assert result == {
        "action": "queue",
        "task": "hand me the cup",
        "reason": "I don't see a cup on the table.",
    }


def test_unseen_object_becomes_queue(monkeypatch):
    """Regression: a run arg or a learn spec naming an object the camera does not currently see passed,
    although the arm cannot pick it and the spec could never be true."""
    unseen = copy.deepcopy(SCENE)
    unseen["objects"]["juice_box"]["seen"] = False
    not_seen = "I don't see a juice box on the table."
    monkeypatch.setattr(
        router, "chat", _reply({"action": "run", "skill": "hand_over", "args": {"object": "juice_box"}})
    )
    assert router.route("pass me the juice box", SKILLS, unseen)["reason"] == not_seen
    assert router.route("pass me the juice box", SKILLS, SCENE)["action"] == "run"
    steps = [{"skill": "hand_over", "args": {"object": "juice_box"}}]
    monkeypatch.setattr(router, "chat", _reply({"action": "define_routine", "name": "juice", "steps": steps}))
    assert router.route("when I say juice, hand me the juice box", SKILLS, unseen)["reason"] == not_seen
    spec = router.keyword_route("put the juice box in the tray", SKILLS)["spec"]
    monkeypatch.setattr(router, "chat", _learn(spec))
    assert router.route("put the juice box in the tray", SKILLS, unseen)["reason"] == not_seen
    assert router.route("put the juice box in the tray", SKILLS, SCENE)["action"] == "learn"
    pinned = dict(spec, success={"not_moved": {"object": "juice_box", "max": 0.02}})
    monkeypatch.setattr(router, "chat", _learn(pinned))
    assert router.route("leave the juice box", SKILLS, unseen)["reason"] == not_seen


def test_star_or_non_string_object_arg_becomes_queue(monkeypatch):
    """Regression: `hand_over` with object "*" passed the checks and failed later in `pick("*")`."""
    for value in ("*", 5, None, ["juice_box"]):
        monkeypatch.setattr(
            router, "chat", _reply({"action": "run", "skill": "hand_over", "args": {"object": value}})
        )
        result = router.route("hand me everything", SKILLS, SCENE)
        assert result == {
            "action": "queue",
            "task": "hand me everything",
            "reason": "I need to know the object for hand over.",
        }, value


def test_routine_with_unknown_skill_becomes_queue(monkeypatch):
    reply = {"action": "define_routine", "name": "tea_time", "steps": [{"skill": "pour_tea", "args": {}}]}
    monkeypatch.setattr(router, "chat", _reply(reply))
    result = router.route("when I say tea time, pour tea", SKILLS, SCENE)
    assert result["action"] == "queue" and "pour tea" in result["reason"]


def test_invalid_model_reply_becomes_queue(monkeypatch):
    monkeypatch.setattr(router, "chat", _reply({"action": "learn", "spec": {"name": "x"}}))
    result = router.route("something odd", SKILLS, SCENE)
    assert result["action"] == "queue"


@pytest.mark.parametrize(
    "failure",
    [
        llm.ModelCallFailed("APIConnectionError: Connection error."),
        llm.ModelCallFailed("gave up after 3 retries: RateLimitError", 429),
        llm.InvalidReply("model reply rejected twice: schema violation at (root)"),
    ],
    ids=["connection", "429", "invalid-reply"],
)
def test_failed_model_call_falls_back_to_keywords_outside_fake_mode(monkeypatch, failure):
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setattr(router, "chat", _raise(failure))
    assert router.route("Hand me my pills", SKILLS, SCENE) == HAND_OVER_PILLS
    assert router.route("Good morning", SKILLS, SCENE) == {
        "action": "run",
        "skill": "good_morning",
        "args": {},
    }
    result = router.route("make me a sandwich", SKILLS, SCENE)
    assert result == {"action": "queue", "task": "make me a sandwich", "reason": router.MODEL_UNREACHABLE}


@pytest.mark.parametrize("exc_type", [ValueError, TypeError, RuntimeError, KeyError])
def test_other_errors_from_chat_propagate_outside_fake_mode(monkeypatch, exc_type):
    """Regression: `except Exception` turned a ValueError or TypeError from `chat` (a bug in the call,
    not an outage) into the keyword fallback, which hid the bug behind a demo phrase."""
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setattr(router, "chat", _raise(exc_type("bad call")))
    with pytest.raises(exc_type, match="bad call"):
        router.route("Hand me my pills", SKILLS, SCENE)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("bad fixture"), llm.ModelCallFailed("bad fixture", 503), llm.InvalidReply("bad fixture")],
)
def test_chat_failure_in_fake_mode_raises(monkeypatch, failure):
    monkeypatch.setattr(router, "chat", _raise(failure))
    with pytest.raises(type(failure), match="bad fixture"):
        router.route("Hand me my pills", SKILLS, SCENE)


def test_stop_never_waits_for_the_model(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("the model was called for stop")

    monkeypatch.setattr(router, "chat", never)
    for text in ("stop", "Stop right now!", "please stop", "STOP."):
        assert router.route(text, SKILLS, SCENE) == {"action": "stop"}
    assert router.is_stop_phrase("put the juice box next to the bus stop sign today") is False


def test_empty_request_is_queued():
    result = router.route("   ", SKILLS, SCENE)
    assert result["action"] == "queue"
    router.VALIDATOR.validate(result)


def test_route_writes_prompt_and_replies_for_a_task(fake_env):
    result = router.route("Hand me my pills", SKILLS, SCENE, task_id="t-0007")
    assert result == HAND_OVER_PILLS
    run_dir = fake_env / "t-0007"
    prompt = (run_dir / "prompts" / "router.md").read_text(encoding="utf-8")
    assert "## system" in prompt and "Hand me my pills" in prompt and "hand_over" in prompt
    assert json.loads((run_dir / "replies" / "router.json").read_text(encoding="utf-8")) == HAND_OVER_PILLS
    assert json.loads((run_dir / "router.json").read_text(encoding="utf-8")) == HAND_OVER_PILLS
    assert (run_dir / "llm.jsonl").is_file()


@pytest.mark.parametrize("task_id", ["../escape", "C:\\x", "a/b", "t 1"])
def test_bad_task_id_raises_in_every_mode(monkeypatch, fake_env, task_id):
    """Regression: a bad task_id was only a warning, then crashed in fake mode and was swallowed into the
    keyword fallback in real mode. Now both modes raise before any call or file."""
    calls: list[str] = []

    def never(*args, **kwargs):
        calls.append("called")
        return {"action": "stop"}

    monkeypatch.setattr(router, "chat", never)
    for fake_mode in (True, False):
        if fake_mode:
            monkeypatch.setenv("FAKE_LLM", "1")
        else:
            monkeypatch.delenv("FAKE_LLM", raising=False)
        for text in ("Hand me my pills", "stop", ""):
            with pytest.raises(ValueError, match="plain identifier"):
                router.route(text, SKILLS, SCENE, task_id=task_id)
    assert calls == [] and not fake_env.exists()


def test_route_timeout_is_a_budget_for_the_whole_call(monkeypatch, tmp_path):
    """Regression: with persistent 503s the router made 4 attempts x timeout + 7 s of backoff before the
    keyword fallback; `timeout` now bounds the model call, retries included."""
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    now = {"t": 100.0}
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    class Error(Exception):
        status_code = 503

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        now["t"] += 1.0  # every request "takes" a second
        raise Error("service unavailable")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "clock", lambda: now["t"])
    monkeypatch.setattr(llm, "sleep", sleep)
    monkeypatch.setattr(llm, "client_factory", lambda **kwargs: client)
    assert router.route("Hand me my pills", SKILLS, SCENE, timeout=5) == HAND_OVER_PILLS  # the fallback
    assert now["t"] - 100.0 <= 5.0
    assert [c["timeout"] for c in calls] == [5.0, 3.0] and sleeps == [1.0]


def test_prompt_teaches_the_vocabulary_and_the_rules():
    text = (router.PROMPT_PATH).read_text(encoding="utf-8")
    for needle in (
        "in_zone",
        "near",
        "moved",
        "not_moved",
        "event",
        '"params": {}',
        "pill_bottle",
        "handoff_done",
        "clear_table",
        "juice_box_to_tray",
        "hand_over",
        "return `queue`",
        "Never name an obstacle",
        "only with the zone `tray`",
    ):
        assert needle in text, needle
    messages = router.build_messages("Hand me my pills", SKILLS, SCENE)
    assert messages[0]["role"] == "system" and messages[0]["content"] == text
    user = messages[1]["content"]
    assert "Hand me my pills" in user and "cereal_box" in user and "tray" in user and "hand_over" in user


def test_compare_ignores_a_single_item_all_wrapper():
    entry = {"text": "x", "action": "learn", "success": {"in_zone": {"object": "juice_box", "zone": "tray"}}}
    wrapped = {
        "action": "learn",
        "spec": {"success": {"all": [{"in_zone": {"object": "juice_box", "zone": "tray"}}]}},
    }
    assert router.compare(entry, wrapped) is None
    assert router.compare(entry, {"action": "queue", "reason": "r"}) is not None
