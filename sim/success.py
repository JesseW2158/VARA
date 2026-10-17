"""Success evaluation for task specs (docs/interfaces.md, "Task spec").

`evaluate(spec, before, after, args, events)` scores every simulated trial inside the sandboxes and checks
the camera snapshot after the real run. It is our code, never AI-written, and depends only on the standard
library so it runs wherever the sandbox bundle goes.

Semantics (mirrored by tests/contract/test_success.py and tests/test_success_extra.py):
- `spec` is a full task spec (its `success` predicate is used) or a bare predicate;
- `in_zone`, `near`, `moved`, `not_moved` take an object reference: a name, `"*"` for every object seen in
  `before` (never obstacles), or `"$name"` for `args["name"]`; a multi-object reference means every object;
- a name is looked up in a snapshot's `objects` first and then in its `obstacles`, so a spec may pin one
  obstacle explicitly (`{"not_moved": {"object": "cereal_box", "max": 0.02}}` holds on an unchanged scene);
  the wildcard still never expands to an obstacle;
- the `zone` of `in_zone` takes the same `"$name"` substitution as an object reference (a `$name` missing
  from `args` raises the same ValueError) and is resolved before any object is checked, so an unknown zone
  or a zone without a radius raises even when the object reference expands to nothing;
- `in_zone`, `near` and `not_moved` are false when the object is unseen (or absent) in `after`, and
  `not_moved` also needs it seen in `before`; `moved` is true when it is unseen in `after`;
- `event` is true when the run's event log contains that event;
- `all`, `any` and `not` combine predicates;
- implicitly, every obstacle in `before` must have moved at most 0.02 m, compared by position regardless
  of `seen` (an unseen obstacle keeps its last pose);
- comparisons are inclusive with a 1e-9 tolerance, so a boundary case is not decided by float noise, and
  a NaN coordinate (or inf in both snapshots, whose difference is NaN) fails every comparison, the
  implicit obstacle rule included: a snapshot with a broken number can never make a spec pass.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["OBSTACLE_MAX_MOVE", "PREDICATES", "already_true", "evaluate", "resolve_object"]

OBSTACLE_MAX_MOVE = 0.02
PREDICATES = frozenset({"all", "any", "not", "in_zone", "near", "moved", "not_moved", "event"})
_OBJECT_PREDICATES = frozenset({"in_zone", "near", "moved", "not_moved"})
_EPS = 1e-9


def evaluate(
    spec: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    args: Mapping[str, Any] | None,
    events: Iterable[Any] | None,
) -> bool:
    """Return True when `after` satisfies the spec relative to `before`, as a real bool.

    Raises ValueError for a malformed predicate (unknown key, not exactly one key, a `$param` missing
    from `args` for an object or a zone, or an unknown zone). Schema validation happens upstream; this
    only guards the walk.
    """
    predicate = spec["success"] if isinstance(spec, Mapping) and "success" in spec else spec
    arguments = args if args is not None else {}
    event_list = list(events) if events is not None else []
    if not _obstacles_unmoved(before, after):
        return False
    return bool(_eval(predicate, before, after, arguments, event_list))


def already_true(spec: Mapping[str, Any], scene: Mapping[str, Any], args: Mapping[str, Any] | None) -> bool:
    """True when the spec already holds on `scene` with nothing moved and no events (then the task is
    rejected with "That already looks done")."""
    return evaluate(spec, scene, scene, args, [])


def resolve_object(ref: Any, before: Mapping[str, Any], args: Mapping[str, Any] | None) -> list[str]:
    """Expand an object reference into object names.

    `"*"` is every object seen in `before` (never an obstacle); `"$name"` is `args["name"]` (which may
    itself be `"*"`); anything else is a single name, which the predicates look up in `objects` first and
    then in `obstacles`. The result keeps the scene's object order.
    """
    name = _substitute(ref, args, "object")
    if name == "*":
        objects = before.get("objects") or {}
        return [found for found, item in objects.items() if item.get("seen", False)]
    return [name]


def _substitute(ref: Any, args: Mapping[str, Any] | None, kind: str) -> str:
    """Return `ref`, or `args["name"]` when `ref` is `"$name"`; either way a non-empty string."""
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{kind} reference must be a non-empty string, got {ref!r}")
    if not ref.startswith("$"):
        return ref
    key = ref[1:]
    if args is None or key not in args:
        raise ValueError(f"spec refers to {ref!r} but args has no {key!r}")
    value = args[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"args[{key!r}] must name a {kind}, got {value!r}")
    return value


def _eval(
    predicate: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    args: Mapping[str, Any],
    events: list[Any],
) -> bool:
    if not isinstance(predicate, Mapping) or len(predicate) != 1:
        raise ValueError(f"a predicate is an object with exactly one key, got {predicate!r}")
    ((key, body),) = predicate.items()
    if key == "all":
        return all(_eval(p, before, after, args, events) for p in _predicate_list(key, body))
    if key == "any":
        return any(_eval(p, before, after, args, events) for p in _predicate_list(key, body))
    if key == "not":
        return not _eval(body, before, after, args, events)
    if key == "event":
        return _has_event(events, body)
    if key in _OBJECT_PREDICATES:
        if not isinstance(body, Mapping):
            raise ValueError(f"{key} needs an object, got {body!r}")
        names = resolve_object(body["object"], before, args)
        if key == "in_zone":
            zone = _zone(body["zone"], before, after, args)  # before the names: a bad zone always raises
            return all(_in_zone(name, zone, after) for name in names)
        if key == "near":
            return all(_near(name, body, after) for name in names)
        if key == "moved":
            return all(_moved(name, body, before, after) for name in names)
        return all(_not_moved(name, body, before, after) for name in names)
    raise ValueError(f"unknown predicate {key!r}; known: {', '.join(sorted(PREDICATES))}")


def _predicate_list(key: str, body: Any) -> list[Any]:
    if not isinstance(body, (list, tuple)):
        raise ValueError(f"{key} needs a list of predicates, got {body!r}")
    return list(body)


def _has_event(events: list[Any], name: Any) -> bool:
    for event in events:
        if event == name:
            return True
        if isinstance(event, Mapping) and (event.get("event") == name or event.get("name") == name):
            return True
    return False


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _find(scene: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """The entry for `name`: in `objects` first, then in `obstacles`; None when it is in neither."""
    for table in ("objects", "obstacles"):
        item = (scene.get(table) or {}).get(name)
        if item is not None:
            return item
    return None


def _seen(scene: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """`_find`, but only when the entry is seen in that snapshot."""
    item = _find(scene, name)
    if item is None or not item.get("seen", False):
        return None
    return item


def _zone(
    ref: Any, before: Mapping[str, Any], after: Mapping[str, Any], args: Mapping[str, Any]
) -> Mapping[str, Any]:
    name = _substitute(ref, args, "zone")
    for scene in (after, before):
        zones = scene.get("zones") or {}
        if name in zones:
            zone = zones[name]
            if "r" not in zone:
                raise ValueError(f"zone {name!r} has no radius; in_zone needs a round zone")
            return zone
    raise ValueError(f"unknown zone {name!r}")


def _in_zone(name: str, zone: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    item = _seen(after, name)
    if item is None:
        return False
    return _distance(item, zone) <= float(zone["r"]) + _EPS


def _near(name: str, body: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    item = _seen(after, name)
    if item is None:
        return False
    distance = math.hypot(float(item["x"]) - float(body["x"]), float(item["y"]) - float(body["y"]))
    return distance <= float(body["tol"]) + _EPS


def _moved(name: str, body: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    start = _find(before, name)
    if start is None:
        return False  # never in the scene: nothing we can say moved
    end = _seen(after, name)
    if end is None:
        return True  # no longer seen: the user took it
    return _distance(start, end) + _EPS >= float(body["min"])


def _not_moved(
    name: str, body: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    start = _seen(before, name)
    end = _seen(after, name)
    if start is None or end is None:
        return False
    return _distance(start, end) <= float(body["max"]) + _EPS


def _obstacles_unmoved(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Every obstacle in `before` is still (within 0.02 m) where it was, whatever its `seen` flag.

    Written as `not (distance <= limit)` rather than `distance > limit` so a NaN distance (a NaN
    coordinate, or inf in both snapshots) fails closed, exactly like the object predicates.
    """
    after_obstacles = after.get("obstacles") or {}
    for name, start in (before.get("obstacles") or {}).items():
        end = after_obstacles.get(name)
        if end is None or not (_distance(start, end) <= OBSTACLE_MAX_MOVE + _EPS):
            return False
    return True
