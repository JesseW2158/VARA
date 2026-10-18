"""Router: one model call turns a request into `run`, `learn`, `define_routine`, `queue` or `stop` (C2).

`route(text, skills, scene, *, task_id=None, timeout=15)` builds the prompt from `agents/prompts/router.md`
(the predicate vocabulary, three worked specs, the "queue when the predicates cannot express success" rule
and the spoken-name mapping), calls `agents.llm.chat` with `schemas/router.json` as the reply schema, and
then checks the reply against the skill list and the scene:

- `run` or `define_routine` naming a skill that does not exist, an object that is not on the table or not
  currently seen, or `"*"` for an object parameter, becomes `queue`;
- a `learn` spec becomes `queue` when its `params` is not `{}` (a learn spec carries no args, so a
  parameter could never be filled), when a predicate names an unknown or unseen object, an unknown zone
  or an obstacle (in any predicate, `not_moved` included: the evaluator already pins every obstacle, and
  `vision.verify.missing_objects` looks names up among the objects only, so a pinned obstacle would read
  as "unseen after the run" on the real table every time), when it uses a `$param`, or when `"*"`
  expands to no seen object ("I don't see anything on the table."); a spec that
  `sim.success.already_true` finds true on the scene becomes `queue` with the reason "That already looks
  done."; a `ValueError` from `already_true` (a spec the evaluator cannot run) becomes `queue` with
  "I could not write a success test for that.";
- "stop" in a few words never waits for the model.

`timeout` is the budget of the model call (`agents.llm.chat` keeps every retry inside it), so `route`
returns within about `timeout` seconds plus the keyword fallback. A `task_id` that is not a plain
identifier raises `ValueError` before anything happens, in every mode.

Outside fake mode, a failed model call (`agents.llm.ModelCallFailed`: the API is down, rate-limited or
out of budget; `agents.llm.InvalidReply`: two replies that did not fit the schema) falls back to
`keyword_route(text, skills)`, the offline insurance for the demo phrases, and then to `queue`. A
`ValueError` or `TypeError` from `chat` is a bug in the call, not an outage, and propagates. In fake mode
(`FAKE_LLM=1`) replies come from `tests/fixtures/llm/` and every failure raises, so a bad fixture is loud.

`python -m agents.router --eval [--fake]` routes every phrase of `tests/fixtures/router_phrases.json` against
`config/fake_scene.json` and the skills `hand_over` and `good_morning` (deliberately without
`place_in_tray`, so "put the juice box in the tray" is a `learn`, as the end-to-end fake expects) and prints
the pass count.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from agents import llm
from agents.llm import chat, normalize, router_model
from sim.success import already_true

__all__ = [
    "ALREADY_DONE",
    "CLEAR_TABLE_SPEC",
    "COMPOUND",
    "EVAL_SKILLS",
    "LEARN_HAS_PARAMS",
    "MODEL_UNREACHABLE",
    "NOTHING_SEEN",
    "NO_SUCCESS_TEST",
    "SCHEMA",
    "build_messages",
    "compare",
    "evaluate_phrases",
    "is_stop_phrase",
    "keyword_route",
    "mentioned_objects",
    "route",
]

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "agents" / "prompts" / "router.md"
OBJECTS_CONFIG = ROOT / "config" / "objects.yaml"  # only for the `called` names in the router context
SCHEMA_PATH = ROOT / "schemas" / "router.json"
PHRASES_PATH = ROOT / "tests" / "fixtures" / "router_phrases.json"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

ALREADY_DONE = "That already looks done."
NO_SUCCESS_TEST = "I could not write a success test for that."
MODEL_UNREACHABLE = "I could not reach the language model to understand that."
NOTHING_SEEN = "I don't see anything on the table."
LEARN_HAS_PARAMS = "I can only learn a task about the specific things on the table."

CLEAR_TABLE_SPEC: dict[str, Any] = {
    "name": "clear_table",
    "description": "Put every object on the table into the tray.",
    "params": {},
    "success": {"all": [{"in_zone": {"object": "*", "zone": "tray"}}]},
    "english": "Every object ends inside the tray.",
}

# Spoken words -> scene object names, for the keyword router (the model does this mapping itself).
OBJECT_WORDS: dict[str, str] = {
    "pill bottle": "pill_bottle",
    "pills": "pill_bottle",
    "pill": "pill_bottle",
    "medicine": "pill_bottle",
    "medication": "pill_bottle",
    "meds": "pill_bottle",
    "bottle": "pill_bottle",
    "juice box": "juice_box",
    "juice": "juice_box",
    "drink": "juice_box",
    "tea tin": "tea_tin",
    "tea": "tea_tin",
    "tin": "tea_tin",
    # the 2026-10-18 props keep the slot names: the power bank is juice_box, the box is tea_tin
    "power bank": "juice_box",
    "battery": "juice_box",
    "charger": "juice_box",
    "bandage box": "tea_tin",
    "band aid box": "tea_tin",
    "t10": "tea_tin",
    "t 10": "tea_tin",
}
HAND_VERBS = ("hand", "give", "pass", "bring", "fetch", "get")
# keyword_route: words that may lead a "stop" ("please stop what you are doing"); words that negate a
# request ("don't hand me", "keep the juice out of the tray"); words that join two requests into one
# sentence ("hand me the pills, then put the juice in the tray"), which a phrase table cannot split; the
# openers of a question; and the verbs that ask for a placement.
STOP_LEADERS = frozenset({"please", "ok", "okay", "hey", "robot", "arm", "now", "just", "can", "you"})
NEGATIONS = frozenset({"not", "dont", "never", "out", "away", "off", "except", "without", "keep", "leave"})
COMPOUND = frozenset({"except", "then", "after", "before", "but"})
ROUTINE_COMPOUND = COMPOUND - {"then"}  # inside "when I say X, ...", "then" separates the steps
QUESTION_WORDS = frozenset(
    {"is", "are", "was", "were", "does", "did", "has", "have", "where", "what", "which"}
)
PLACE_VERBS = frozenset(
    {"put", "place", "move", "drop", "set", "stick", "load", "get", "bring", "carry", "return", "take"}
)
TRAY_PHRASES = (
    " in the tray",
    " into the tray",
    " in tray",
    " to the tray",
    " on the tray",
    " inside the tray",
)

# The skill list `--eval` routes against (see the module docstring for why place_in_tray is absent).
EVAL_SKILLS: list[dict[str, Any]] = [
    {
        "name": "hand_over",
        "description": "Pick up an object and hand it to the user.",
        "params": {"object": "object"},
    },
    {
        "name": "good_morning",
        "description": "Routine: hand over the pill bottle, then the juice box.",
        "params": {},
    },
]

logger = logging.getLogger(__name__)


# --- the router ---------------------------------------------------------------------------------------


def route(
    text: str,
    skills: Sequence[Mapping[str, Any]],
    scene: Mapping[str, Any],
    *,
    task_id: str | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    """Route one request. Always returns a dict valid against schemas/router.json.

    Raises ValueError for a `task_id` that is not a plain identifier (`llm.run_dir` decides)."""
    run_dir = llm.run_dir(task_id) if task_id else None
    text = str(text or "").strip()
    skills = [dict(s) for s in skills]
    if not text:
        return _finish(_queue("(empty request)", "I did not catch that."))
    if is_stop_phrase(text):
        return _finish({"action": "stop"})
    messages = build_messages(text, skills, scene)
    if run_dir is not None:
        _write(run_dir / "prompts" / "router.md", _render(messages))
    try:
        reply = chat(
            messages,
            router_model(),
            schema=SCHEMA,
            timeout=timeout,
            purpose="router",
            key=text,
            task_id=task_id,
        )
    except (llm.ModelCallFailed, llm.InvalidReply) as exc:
        # An outage, a rate limit, a spent budget or two unusable replies: the offline table takes over.
        # Anything else (a ValueError or TypeError is a bug in the call, not the network) propagates.
        if llm.is_fake():
            raise
        logger.warning("router model call failed (%s: %s); using the keyword router", type(exc).__name__, exc)
        reply = keyword_route(text, skills) or _queue(text, MODEL_UNREACHABLE)
    if run_dir is not None:
        _write(run_dir / "replies" / "router.json", json.dumps(reply, indent=2))
    result = _check(dict(reply), text, skills, scene)
    if run_dir is not None:
        _write(run_dir / "router.json", json.dumps(result, indent=2))
    return _finish(result)


def _object_details(scene: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per object and obstacle in the scene: what it is and what people call it (config/objects.yaml
    `called`, the slot names are fixed) and its table position in meters (x out from the arm, y to its
    left), so "the box", a misheard "T10" or "on the right" can be routed."""
    called = _called_names()
    out: dict[str, dict[str, Any]] = {}
    for group in ("objects", "obstacles"):
        for name, item in (scene.get(group) or {}).items():
            entry: dict[str, Any] = {"kind": group[:-1], "called": called.get(name, [])}
            if isinstance(item, Mapping):
                try:
                    entry["x"], entry["y"] = round(float(item["x"]), 2), round(float(item["y"]), 2)
                except (KeyError, TypeError, ValueError):
                    pass
            out[str(name)] = entry
    return out


def _called_names() -> dict[str, list[str]]:
    """config/objects.yaml's `called` lists by object name ({} when the file or the field is missing)."""
    try:
        raw = yaml.safe_load(OBJECTS_CONFIG.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    out: dict[str, list[str]] = {}
    for entry in raw.values() if isinstance(raw, dict) else []:
        if isinstance(entry, dict) and isinstance(entry.get("called"), list):
            out[str(entry.get("name"))] = [str(v) for v in entry["called"]]
    return out


def build_messages(
    text: str, skills: Sequence[Mapping[str, Any]], scene: Mapping[str, Any]
) -> list[dict[str, str]]:
    """The system prompt from agents/prompts/router.md plus the request context as the user message."""
    system = PROMPT_PATH.read_text(encoding="utf-8")
    objects = scene.get("objects") or {}
    seen = [name for name, item in objects.items() if item.get("seen", True)]
    unseen = [name for name, item in objects.items() if not item.get("seen", True)]
    context = {
        "saved_skills": [
            {"name": s.get("name"), "description": s.get("description", ""), "params": s.get("params") or {}}
            for s in skills
        ],
        "scene": {
            "objects": seen,
            "objects_not_currently_seen": unseen,
            "obstacles": list((scene.get("obstacles") or {}).keys()),
            "zones": list((scene.get("zones") or {}).keys()),
            "object_details": _object_details(scene),
        },
        "request": text,
    }
    user = (
        "Request context:\n```json\n"
        + json.dumps(context, indent=2)
        + "\n```\nRoute the request. Reply with exactly one JSON object and nothing else."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def is_stop_phrase(text: str) -> bool:
    """A short request containing the word "stop" ("stop", "Stop right now!", "please stop")."""
    words = normalize(text).split("_")
    return 0 < len(words) <= 4 and "stop" in words


def _check(
    reply: dict[str, Any], text: str, skills: list[dict[str, Any]], scene: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the scene and skill checks that the model cannot be trusted with."""
    try:
        VALIDATOR.validate(reply)
    except jsonschema.ValidationError as exc:
        logger.warning("router reply rejected by the schema: %s", exc.message)
        return _queue(text, NO_SUCCESS_TEST)
    by_name = {s.get("name"): s for s in skills}
    action = reply["action"]
    if action == "stop":
        return {"action": "stop"}
    if action == "queue":
        return _queue(reply.get("task") or text, reply.get("reason") or "I cannot do that yet.")
    if action == "run":
        problem = _skill_call_problem(reply["skill"], reply.get("args") or {}, by_name, scene)
        if problem:
            return _queue(text, problem)
        return {"action": "run", "skill": reply["skill"], "args": dict(reply.get("args") or {})}
    if action == "define_routine":
        for step in reply["steps"]:
            problem = _skill_call_problem(step["skill"], step.get("args") or {}, by_name, scene)
            if problem:
                return _queue(text, problem)
        return {"action": "define_routine", "name": reply["name"], "steps": copy.deepcopy(reply["steps"])}
    spec = copy.deepcopy(reply["spec"])
    problem = _spec_problem(spec, scene)
    if problem:
        return _queue(text, problem)
    try:
        done = already_true(spec, scene, {})
    except ValueError as exc:
        logger.warning("spec %r cannot be evaluated: %s", spec.get("name"), exc)
        return _queue(text, NO_SUCCESS_TEST)
    if done:
        return _queue(text, ALREADY_DONE)
    return {"action": "learn", "spec": spec}


def _skill_call_problem(
    name: str, args: Mapping[str, Any], by_name: Mapping[str, Mapping[str, Any]], scene: Mapping[str, Any]
) -> str | None:
    skill = by_name.get(name)
    if skill is None:
        return f"I don't have a skill called {_words(name)}."
    for param, kind in (skill.get("params") or {}).items():
        if param not in args:
            return f"I need to know the {param} for {_words(name)}."
        value = args[param]
        if kind == "object":
            if not isinstance(value, str) or value == "*":  # a skill picks one named object
                return f"I need to know the {param} for {_words(name)}."
            if not _is_seen_object(scene, value):
                return _missing_object(scene, value)
        if kind == "zone" and value not in (scene.get("zones") or {}):
            return f"I don't know a place called {_words(str(value))}."
    return None


def _spec_problem(spec: Mapping[str, Any], scene: Mapping[str, Any]) -> str | None:
    """A queue reason when the spec could never be learned or checked on this scene, or None.

    Refused: `params` other than `{}` (a learn spec carries no args, so nothing could ever fill a
    parameter), a `$param`, an unknown or unseen object, an obstacle in any predicate (see
    `_obstacle_reason`), `"*"` when the camera sees no object at all, an unknown zone, and `in_zone` on a
    zone without a radius.
    """
    if spec.get("params") != {}:
        return LEARN_HAS_PARAMS
    zones = scene.get("zones") or {}
    obstacles = scene.get("obstacles") or {}
    for key, body in _predicates(spec["success"]):
        if key == "event":
            continue
        ref = body.get("object")
        if not isinstance(ref, str) or not ref:
            return NO_SUCCESS_TEST
        if ref.startswith("$"):
            return NO_SUCCESS_TEST  # learn specs carry no args: a $param can never be filled
        if ref == "*":
            if not _seen_objects(scene):
                return NOTHING_SEEN  # "*" expands to the seen objects: an empty table has nothing to move
        elif not _is_seen_object(scene, ref):
            if ref in obstacles:
                return _obstacle_reason(key, ref)
            return _missing_object(scene, ref)  # unknown, or known but not seen by the camera
        if key == "in_zone":
            zone = body.get("zone")
            if not isinstance(zone, str) or zone.startswith("$"):
                return NO_SUCCESS_TEST
            if zone not in zones:
                return f"I don't know a place called {_words(zone)}."
            if "r" not in zones[zone]:
                return "I can only check whether something is in the tray."
    return None


def _predicates(node: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Every leaf predicate as (key, body), walking all/any/not."""
    if not isinstance(node, Mapping) or len(node) != 1:
        return
    ((key, body),) = node.items()
    if key in ("all", "any"):
        for child in body if isinstance(body, list) else []:
            yield from _predicates(child)
    elif key == "not":
        yield from _predicates(body)
    elif key == "event":
        yield key, {"event": body}
    elif isinstance(body, Mapping):
        yield key, body


def _is_seen_object(scene: Mapping[str, Any], name: Any) -> bool:
    """A scene object the camera currently sees (an unseen one cannot be picked or checked)."""
    objects = scene.get("objects") or {}
    if not isinstance(name, str) or name not in objects:
        return False
    item = objects[name]
    return isinstance(item, Mapping) and bool(item.get("seen", False))


def _seen_objects(scene: Mapping[str, Any]) -> list[str]:
    """The objects the camera currently sees: what `"*"` expands to (`sim.success.resolve_object`)."""
    objects = scene.get("objects") or {}
    return [name for name, item in objects.items() if isinstance(item, Mapping) and item.get("seen", False)]


def _obstacle_reason(key: str, name: str) -> str:
    """Why an obstacle may not appear in a learn predicate, `not_moved` included: the evaluator already
    requires every obstacle to stay put, and the camera check (`vision.verify.missing_objects`) looks
    names up among the objects only, so a pinned obstacle would read as "unseen after the run" on the
    real table every time."""
    if key == "not_moved":
        return f"The {_words(name)} always stays where it is; I can't put it in a task."
    return f"I'm not allowed to move the {_words(name)}."


def _missing_object(scene: Mapping[str, Any], name: Any) -> str:
    return f"I don't see a {_words(str(name))} on the table."


def _words(name: str) -> str:
    return name.replace("_", " ")


def _queue(task: str, reason: str) -> dict[str, Any]:
    return {"action": "queue", "task": task or "(empty request)", "reason": reason}


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    VALIDATOR.validate(result)
    return result


def _render(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(f"## {m['role']}\n\n{m['content']}" for m in messages) + "\n"


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write %s: %s", path, exc)


# --- the keyword router (offline insurance) ----------------------------------------------------------


def keyword_route(text: str, skills: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Fixed actions for the demo phrases when the model is unreachable; None for anything else.

    "stop" (a few words, or a sentence that opens with it) -> stop; "fold my laundry" -> queue; "when I
    say X, hand me A, then B" -> define_routine X; "<saved routine name>" -> run it; "hand me my pills" ->
    run hand_over pill_bottle; "clear the table" or "everything in the tray" -> run clear_table when saved,
    else the clear_table learn spec; "put the juice box in the tray" (a placing verb or "in the tray") ->
    run place_in_tray when saved, else a learn spec `<object>_to_tray`.

    Every rule but stop is guarded (`_guarded`): a negation ("don't hand me the pills", "keep the juice
    out of the tray"), a compound sentence ("hand me the pills, then put the juice box in the tray",
    "clear the table but leave the cereal box", "everything except the tea tin") or a question ("is the
    juice box in the tray") gets None, because a phrase table cannot split a sentence and acting on half
    of one is worse than queueing it. Inside "when I say X, ...", "then" separates the steps and is no
    compound. Stop is exempt: the safety action fires whatever surrounds it, as `is_stop_phrase` does
    in `route()`.
    """
    names = {s.get("name"): s for s in skills}
    plain = normalize(text).replace("_", " ")
    words = plain.split()
    if not words:
        return None
    if _asks_to_stop(words):
        return {"action": "stop"}
    routine = _parse_routine(text)
    if routine is not None:
        return routine
    if _guarded(words):
        return None
    if "laundry" in words or ("fold" in words and ("clothes" in words or "shirt" in words)):
        return _queue(text, "needs cloth handling")
    for name, skill in names.items():
        if isinstance(name, str) and not (skill.get("params") or {}) and plain == _words(name):
            return {"action": "run", "skill": name, "args": {}}
    objects = mentioned_objects(text)
    if any(verb in words for verb in HAND_VERBS) and "tray" not in words and objects:
        if "hand_over" in names:
            return {"action": "run", "skill": "hand_over", "args": {"object": objects[0]}}
        return None
    everything = ("everything" in words or "all" in words) and "tray" in words
    if ("clear" in words and "table" in words) or everything:
        if "clear_table" in names:
            return {"action": "run", "skill": "clear_table", "args": {}}
        return {"action": "learn", "spec": copy.deepcopy(CLEAR_TABLE_SPEC)}
    if "tray" in words and objects and _asks_to_place(words, plain):
        target = objects[0]
        if "place_in_tray" in names:
            return {"action": "run", "skill": "place_in_tray", "args": {"object": target}}
        return {"action": "learn", "spec": _to_tray_spec(target)}
    return None


def _asks_to_stop(words: Sequence[str]) -> bool:
    """ "stop" within four words, or as the first word after politeness ("please stop what you are doing");
    not "put it next to the bus stop sign"."""
    if 0 < len(words) <= 4 and "stop" in words:
        return True
    leading = next((w for w in words if w not in STOP_LEADERS), None)
    return leading == "stop"


def _guarded(words: Sequence[str], compound: frozenset[str] = COMPOUND) -> bool:
    """True when no keyword rule may fire on `words`: a negation ("don't", "keep ... out", "leave"), a
    compound sentence (a word of `compound`) or a question opener ("is the juice box in the tray")."""
    return bool(NEGATIONS.intersection(words) or compound.intersection(words) or words[0] in QUESTION_WORDS)


def _asks_to_place(words: Sequence[str], plain: str) -> bool:
    """A placing verb or an "in the tray" phrase (the negation and question guards ran already)."""
    padded = f" {plain} "
    return bool(PLACE_VERBS.intersection(words)) or any(phrase in padded for phrase in TRAY_PHRASES)


def mentioned_objects(text: str) -> list[str]:
    """Scene object names mentioned in `text`, in order of first mention."""
    plain = " " + normalize(text).replace("_", " ") + " "
    found: dict[str, int] = {}
    for phrase, name in OBJECT_WORDS.items():
        position = plain.find(" " + phrase + " ")
        if position >= 0 and (name not in found or position < found[name]):
            found[name] = position
    return sorted(found, key=lambda name: found[name])


def _parse_routine(text: str) -> dict[str, Any] | None:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    match = re.match(r"(?:ok |please )?when i say (.+?)\s*[,;:]\s*(.+)$", lowered)
    if match is None:
        verbs = r"(?:then |please )?(?:hand|give|pass|bring|put) .+"
        match = re.match(rf"(?:ok |please )?when i say (.+?)\s+({verbs})$", lowered)
    if match is None:
        return None
    name = normalize(match.group(1))
    rest = match.group(2)
    if not SKILL_NAME_RE.match(name):
        return None
    if _guarded(normalize(rest).split("_"), ROUTINE_COMPOUND):
        return None  # "when I say X, don't ..." or "... except ...": the model decides
    objects = mentioned_objects(rest)
    if not objects:
        return None
    skill = "place_in_tray" if "tray" in normalize(rest).split("_") else "hand_over"
    steps = [{"skill": skill, "args": {"object": obj}} for obj in objects]
    return {"action": "define_routine", "name": name, "steps": steps}


def _to_tray_spec(target: str) -> dict[str, Any]:
    return {
        "name": f"{target}_to_tray",
        "description": f"Put the {_words(target)} into the tray.",
        "params": {},
        "success": {"in_zone": {"object": target, "zone": "tray"}},
        "english": f"The {_words(target)} ends inside the tray.",
    }


# --- --eval ------------------------------------------------------------------------------------------


def compare(entry: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    """None when `result` matches the fixture entry, else what differs.

    Entry keys: `action`, and optionally `skill`, `args`, `name`, `steps` (compared as is), `spec_name`,
    `success`, `params` (compared inside `spec`; a single-item `all` wrapper is ignored). `text` and
    `note` are documentation; a queue `reason` is never compared.
    """
    if result.get("action") != entry["action"]:
        return f"expected {entry['action']}, got {result.get('action')!r} ({result.get('reason', '')})"
    spec = result.get("spec") or {}
    for key, expected in entry.items():
        if key in ("text", "action", "note", "reason"):
            continue
        if key == "spec_name":
            actual: Any = spec.get("name")
        elif key in ("success", "params"):
            actual = spec.get(key)
        else:
            actual = result.get(key)
        if key == "success":
            if _unwrap(actual) != _unwrap(expected):
                return f"success: expected {json.dumps(expected)}, got {json.dumps(actual)}"
        elif actual != expected:
            return f"{key}: expected {expected!r}, got {actual!r}"
    return None


def _unwrap(predicate: Any) -> Any:
    while isinstance(predicate, Mapping) and set(predicate) == {"all"} and len(predicate["all"]) == 1:
        predicate = predicate["all"][0]
    return predicate


def evaluate_phrases(
    phrases: Sequence[Mapping[str, Any]], skills: Sequence[Mapping[str, Any]], scene: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], dict[str, Any], str | None]]:
    """Route every fixture phrase and compare: (entry, result, problem or None)."""
    out = []
    for entry in phrases:
        try:
            result = route(entry["text"], skills, scene)
            problem = compare(entry, result)
        except Exception as exc:  # the eval reports instead of stopping
            result, problem = {"action": "error"}, f"{type(exc).__name__}: {exc}"
        out.append((entry, result, problem))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agents.router --eval [--fake]`: route the 20 fixture phrases and print the pass count."""
    parser = argparse.ArgumentParser(prog="python -m agents.router", description="request router")
    parser.add_argument("--eval", action="store_true", help="route every phrase in the fixture file")
    parser.add_argument("--fake", action="store_true", help="answer from the fixtures (FAKE_LLM=1)")
    parser.add_argument("--phrases", default=str(PHRASES_PATH), help="phrase fixture file")
    parser.add_argument("--scene", default=str(FAKE_SCENE_PATH), help="scene JSON to route against")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    if args.fake:
        os.environ["FAKE_LLM"] = "1"
    if not args.eval:
        parser.print_help()
        return 2
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    phrases = json.loads(Path(args.phrases).read_text(encoding="utf-8"))
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    passed = 0
    for entry, result, problem in evaluate_phrases(phrases, EVAL_SKILLS, scene):
        if problem is None:
            passed += 1
            print(f"ok    {entry['text']!r} -> {result.get('action')}")
        else:
            print(f"FAIL  {entry['text']!r} -> {result.get('action')}: {problem}")
    print(f"{passed}/{len(phrases)} phrases routed as expected")
    return 0 if passed == len(phrases) else 1


if __name__ == "__main__":
    sys.exit(main())
