"""I2, arm meets camera: hover 5 cm above every seen object marker and measure the gap.

The script reads the scene from `vision.scene.get_tracker()` (the median of 10 frames), plans a move for
every seen object marker with `arm.ik.solve` to 5 cm above the marker center (the pitch ladder relaxes the
straight-down request when the arm cannot reach it), builds a trajectory from `home_q` in steps of at most
2 degrees per tick (home -> a hover point 8 cm higher -> the target, and back the same way), validates both
legs with `arm.replay.validate`, replays them with `arm.replay.replay`, and reports the horizontal gap
between the tip and the marker center against the 1.5 cm limit. Results go to runs/touch_markers/results.json.

Modes (exactly one):
  --fake   dry run: FAKE_ROBOT and FAKE_CAMERA are forced, no prompts; the gap is `arm.ik.fk_tip` of the
           reached joints against the marker center. `--chain` runs the REAL SceneTracker on the FakeCamera
           instead of the FakeSceneTracker, so the detection error is part of the gap.
  --real   the real arm: a confirmation prompt, a 3 s countdown, Esc freezes the arm (arm.driver.EscWatcher)
           and `driver.connect()` happens only here; an Esc that lands inside connect() aborts before any
           move. Refused with FAKE, FAKE_ROBOT or FAKE_CAMERA set to 1, or when `vision.scene.get_tracker()`
           is a fake: a plan from config/fake_scene.json must never reach the real arm. For every marker the
           arm holds the hover pose while you measure the gap with a ruler and type it in cm (blank keeps
           the fk estimate). Both legs replay with the running SceneTracker's `latest()` as the hand
           source, so arm.replay's wristband rules (freeze within 10 cm of the band, 0.1 m/s within 15 cm)
           cover the person holding the ruler.

I2 lays five LOOSE object-size markers flat on the table, away from the objects: the spare cards that
scripts/make_markers.py prints (ids 13-17 by default, "spare, I2 check"). Pass `--loose`: instead of the
tracker's scene, which knows only the ids of config/objects.yaml, the script detects every object-size id
10-19 in 10 frames of the overhead camera (`vision.markers.detect`; an id seen in at least half of the
frames keeps the median of its corners) and maps each center through the saved config/homography.npy with
`vision.calibrate.pixel_to_table` at height 0: a loose marker lies on the table, so there is no height
correction. That is wrong for a marker glued on top of an object: it maps to a table point inside the
object's footprint, and the hover 5 cm above the table would drive the tip into the object (2 to 5.5 cm
into the demo objects). So every id that config/objects.yaml assigns (10-12 now, `object_card_ids`) is
LEFT OUT by default, and the ids seen and left out are printed with the reason (`leave_out_object_cards`).
`--object-cards-flat` includes them: it is the operator's word that those cards lie flat on the table
(before they are glued, with the objects taken away). The targets are named `marker_<id>` and the hover
point is 5 cm above the table; at least five must remain (`--count N`, 1 to 10) or nothing moves (exit 1).
A `--count` above the number of ids that may be targets is a usage error (exit 2), as is, with --loose, an
`--objects` name other than `marker_<id>` of ids 10-19 or, without --object-cards-flat, one of a left-out
id; a named marker that is not among the targets moves nothing (exit 1). Planning, validation, replay, the
prompts, `--limit`, `--out` and the `--real` gating are those of the normal run.
In --real the frames are the running SceneTracker's (`wait_for_frame`, then `last_frame`), the same
tracker that feeds the wristband rules; a tracker whose camera is a fake is refused like a fake tracker
(exit 2, before any prompt), and the saved homography must exist (B5). In --fake the I2 table is
synthesized from the fake scene (`loose_fake_scene`): its objects keep standing with their markers glued on
top (seen and left out, as on the real table) and five spare cards lie flat (13-17: the first ids from 13
that config/objects.yaml leaves free, as make_markers prints them); with --object-cards-flat the objects are
taken away and their cards laid flat where they stood, next to spares 13 and 14. Plain --fake takes every
marker where the overhead camera would see it at table height 0 (`fake_detections`); --fake --chain renders
the table with a FakeCamera and runs the real detection on its frames, mapped through
`vision.calibrate.synthetic_homography` (the FakeCamera's own camera model; config/homography.npy is the
physical camera's once B5 has run), so the detection error is part of the gap. results.json then also holds
`count`, `object_cards_flat`, `left_out` (`[{marker, name}]`) and `loose_detections` (`[{marker, frames}]`
for every id 10-19 detected, null when synthesized).
Without --loose, markers are assumed glued on top of their objects, as the scene reports them.
If the real run misses 1.5 cm, fix config/joint_map.yaml (A6) or the homography (B5) before anything else.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ABOVE_M = 0.05
HOVER_M = 0.08
MAX_Z = 0.28
LIMIT_CM = 1.5
STEP_RAD = math.radians(2.0)
DEFAULT_OUT = ROOT / "runs" / "touch_markers"
FAKE_FLAGS = ("FAKE", "FAKE_ROBOT", "FAKE_CAMERA")  # any of them at "1" refuses --real
LOOSE_IDS = tuple(range(10, 20))  # object-size markers (vision.markers: ids 10-19, 3 cm)
LOOSE_COUNT = 5  # I2 lays five loose markers; --count changes it
LOOSE_FRAMES = 10  # frames per detection, as the tracker's snapshot
# Spare cards: make_markers prints the first five of these that config/objects.yaml leaves free.
SPARE_IDS = tuple(range(13, 20))
# Where the --fake I2 table lays its spare cards, (x, y): spots the arm reaches 5 cm above the table, at
# least 9 cm from the fake scene's objects (which keep standing there unless --object-cards-flat).
LOOSE_SPOTS = ((0.20, 0.10), (0.27, 0.05), (0.30, -0.14), (0.22, 0.17), (0.14, 0.09))


def load_home_q() -> list[float]:
    cfg = yaml.safe_load((ROOT / "config" / "scene.yaml").read_text(encoding="utf-8"))
    return [float(v) for v in cfg["home_q"]]


def marker_center(
    item: dict[str, Any], *, loose: bool, c: Sequence[float], hc: float
) -> tuple[float, float, float]:
    """World center of the marker: on top of the object, (x, y, h), or (loose) on the table, (x, y, 0).

    A loose card (h = 0, from `loose_targets` or `fake_detections`) comes back unchanged; an item on top of
    an object (h > 0, as the tracker reports it) has the tracker's height correction undone, which is the
    table point where the homography at height 0 puts its marker (what `fake_detections` uses).
    """
    x, y, h = float(item["x"]), float(item["y"]), float(item["h"])
    if not loose:
        return x, y, h
    k = 1.0 - h / hc  # the tracker applied p = c + (p_table - c) * k for a marker that was really at z = 0
    return c[0] + (x - c[0]) / k, c[1] + (y - c[1]) / k, 0.0


def _loose_item(marker_id: int, x: float, y: float, yaw: float) -> dict[str, Any]:
    """A loose 3 cm card lying on the table at (x, y): an item for `plan()` and the FakeCamera (h = 0)."""
    from vision.markers import MARKER_SIDE_M

    side = MARKER_SIDE_M["object"]
    return {
        "marker": int(marker_id),
        "x": float(x),
        "y": float(y),
        "yaw": float(yaw),
        "h": 0.0,
        "size": [side, side],
        "seen": True,
    }


def object_card_ids(objects: Mapping[int, Mapping[str, Any]] | None = None) -> dict[int, str]:
    """`{id: name}` of the object-size ids (10-19) that config/objects.yaml (or `objects`) assigns, whatever
    the kind: their cards are glued on top of things that stand on the table, so `--loose` leaves them out
    unless --object-cards-flat (`leave_out_object_cards`)."""
    from vision.markers import load_objects

    table = objects if objects is not None else load_objects()
    return {int(i): str(entry["name"]) for i, entry in sorted(table.items()) if int(i) in LOOSE_IDS}


def leave_out_object_cards(
    targets: Sequence[tuple[str, dict[str, Any]]], assigned: Mapping[int, str], *, object_cards_flat: bool
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """`(kept, left_out)` of the detected loose `targets`: without `object_cards_flat`, every one whose id
    `assigned` (`object_card_ids`) holds is left out. Such a card is glued on top of an object; the
    homography at table height 0 maps it a little outward from under the lens, still inside the object's
    footprint, and a hover planned from the table there would put the tip inside the object.
    `object_cards_flat` is the operator's word that those cards lie flat on the table: all are kept."""
    if object_cards_flat:
        return list(targets), []
    kept = [(name, item) for name, item in targets if int(item["marker"]) not in assigned]
    left_out = [(name, item) for name, item in targets if int(item["marker"]) in assigned]
    return kept, left_out


def loose_fake_scene(
    scene: Mapping[str, Any],
    count: int = LOOSE_COUNT,
    *,
    object_cards_flat: bool = False,
    assigned: Collection[int] | None = None,
) -> dict[str, Any]:
    """The I2 table of a --fake run. By default `scene`'s objects keep standing with their markers glued on
    top (copied as the scene reports them, h > 0: what `--loose` sees and leaves out) and `count` spare
    cards lie flat on the table; with `object_cards_flat` the objects are taken away and their cards laid
    flat where they stood, then spare cards until `count` cards lie flat. A spare card takes the next of
    SPARE_IDS that neither `assigned` (default: `object_card_ids()`) nor the scene's objects use, as
    make_markers prints them, and the next spot of LOOSE_SPOTS; cards lying flat have h = 0 and are named
    `marker_<id>`. No obstacles, the hand unseen, the zones kept. An input for the FakeCamera and
    `fake_detections`, not a scene to publish (schemas/scene.json wants h > 0)."""
    used = set(object_card_ids() if assigned is None else assigned)
    objects: dict[str, dict[str, Any]] = {}
    flat = 0
    for name, item in (scene.get("objects") or {}).items():
        marker_id = int(item["marker"])
        used.add(marker_id)
        if object_cards_flat:
            card = _loose_item(marker_id, item["x"], item["y"], item.get("yaw", 0.0))
            objects[f"marker_{marker_id}"] = card
            flat += 1
        else:
            objects[str(name)] = copy.deepcopy(dict(item))
    spares = [marker_id for marker_id in SPARE_IDS if marker_id not in used]
    for marker_id, (x, y) in zip(spares, LOOSE_SPOTS, strict=False):
        if flat >= count:
            break
        objects[f"marker_{marker_id}"] = _loose_item(marker_id, x, y, 0.0)
        flat += 1
    hand = dict(scene.get("hand") or {"marker": 40, "x": 0.0, "y": 0.0, "z": 0.03})
    hand["seen"] = False
    return {
        "t": float(scene.get("t", 0.0)),
        "objects": objects,
        "obstacles": {},
        "hand": hand,
        "zones": copy.deepcopy(scene.get("zones") or {}),
    }


def fake_detections(
    table: Mapping[str, Any], c: Sequence[float], hc: float
) -> list[tuple[str, dict[str, Any]]]:
    """What `loose_targets` finds on the --fake I2 `table` (`loose_fake_scene`) without a camera: every seen
    object-size marker (ids 10-19) at the table point where the overhead camera sees it at height 0
    (`marker_center(..., loose=True)`: a card lying flat stays put, a marker glued on top of an object lands
    a little farther out from under the lens, as through the homography), named `marker_<id>`, by id."""
    found: dict[int, dict[str, Any]] = {}
    for item in (table.get("objects") or {}).values():
        marker_id = int(item["marker"])
        if marker_id not in LOOSE_IDS or not item.get("seen", True):
            continue
        x, y, _ = marker_center(item, loose=True, c=c, hc=hc)
        found[marker_id] = _loose_item(marker_id, x, y, float(item.get("yaw", 0.0)))
    return [(f"marker_{marker_id}", found[marker_id]) for marker_id in sorted(found)]


def collect_frames(next_frame: Callable[[], np.ndarray | None], n: int = LOOSE_FRAMES) -> list[np.ndarray]:
    """`n` frames from `next_frame`, which returns None for a missed grab (not counted toward n).

    TimeoutError when more than half of the grabs in the window miss, the rule of `SceneTracker.snapshot`:
    a dead camera is an error, never an empty table with nothing to touch.
    """
    count = max(1, int(n))
    frames: list[np.ndarray] = []
    missed = 0
    while len(frames) < count:
        frame = next_frame()
        if frame is None:
            missed += 1
            if 2 * missed > count:
                raise TimeoutError(
                    f"camera delivered no frame for {missed} of {missed + len(frames)} grabs while looking "
                    f"for loose markers ({count} frames wanted)"
                )
            continue
        frames.append(frame)
    return frames


def tracker_frames(tracker: Any, timeout: float = 2.0) -> Callable[[], np.ndarray | None]:
    """A `next_frame` for `collect_frames` on a SceneTracker, without opening its camera a second time.

    While its thread runs: the frame behind the next publication (`wait_for_frame`, then `last_frame`); a
    publication without a new frame (a missed grab) or `timeout` s of silence is a miss. Otherwise one
    synchronous `step()`, a miss when it delivered no frame.
    """

    def next_frame() -> np.ndarray | None:
        before = tracker.frames
        if tracker.running:
            if not tracker.wait_for_frame(timeout=timeout):
                return None
        else:
            tracker.step()
        if tracker.frames == before:
            return None
        return tracker.last_frame()

    return next_frame


def loose_targets(
    frames: Sequence[np.ndarray], h: np.ndarray, *, ids: Sequence[int] = LOOSE_IDS
) -> tuple[list[tuple[str, dict[str, Any]]], dict[int, int]]:
    """The loose markers in `frames`: `[(name, item)]` for `plan()` and `{id: frames it was seen in}`.

    `vision.markers.detect` runs on every frame; an id of `ids` seen in at least half of the frames keeps
    the median of its corners (the tracker's snapshot rule: a card that flickers is not a target), whose
    center maps through `h` with `vision.calibrate.pixel_to_table` to the table point under the marker at
    height 0 (a loose marker lies on the table: no height correction). The yaw is the direction corner 0 ->
    corner 1 on the table. Items are named `marker_<id>`, sorted by id, with `frames` added.
    """
    from vision.calibrate import pixel_to_table
    from vision.markers import detect

    if not frames:
        raise ValueError("no frames to look for loose markers in")
    wanted = {int(i) for i in ids}
    seen: dict[int, list[np.ndarray]] = {}
    for frame in frames:
        for marker_id, corners in detect(frame):
            if marker_id in wanted:
                seen.setdefault(marker_id, []).append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
    counts = {marker_id: len(readings) for marker_id, readings in sorted(seen.items())}
    targets: list[tuple[str, dict[str, Any]]] = []
    for marker_id, readings in sorted(seen.items()):
        if 2 * len(readings) < len(frames):
            continue
        pts = np.median(np.stack(readings), axis=0)
        x, y = pixel_to_table(h, *pts.mean(axis=0))
        p0 = pixel_to_table(h, pts[0, 0], pts[0, 1])
        p1 = pixel_to_table(h, pts[1, 0], pts[1, 1])
        item = _loose_item(marker_id, x, y, math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        item["frames"] = len(readings)
        targets.append((f"marker_{marker_id}", item))
    return targets, counts


def trajectory(name: str, waypoints: Sequence[Sequence[float]]) -> dict[str, Any]:
    return {
        "version": 1,
        "skill": "touch_markers",
        "args": {"object": name},
        "dt": 1 / 30,
        "joints": list(JOINTS),
        "segments": [{"kind": "path", "q": [[round(float(v), 6) for v in q] for q in waypoints]}],
    }


def plan(
    name: str,
    item: dict[str, Any],
    home_q: Sequence[float],
    *,
    above: float,
    loose: bool,
    c: Sequence[float],
    hc: float,
) -> dict[str, Any]:
    """IK, waypoints and validation for one marker; `problems` is empty when both legs may be replayed."""
    from arm.ik import fk, solve
    from arm.replay import bridge, load_box, load_limits, validate

    mx, my, mz = marker_center(item, loose=loose, c=c, hc=hc)
    target = (mx, my, mz + above)
    hover = (mx, my, min(MAX_Z, target[2] + HOVER_M))
    result: dict[str, Any] = {
        "name": name,
        "marker": int(item["marker"]),
        "marker_center": [mx, my, mz],
        "target": list(target),
        "problems": [],
    }
    q_hover, ok_hover = solve(hover, -math.pi / 2, q0=list(home_q))
    q_goal, ok = solve(target, -math.pi / 2, q0=q_hover if ok_hover else list(home_q))
    if not ok:
        result["problems"].append(f"ik: {target} is unreachable at every ladder pitch")
        return result
    via = [q_hover] if ok_hover else []
    go: list[list[float]] = [list(home_q)]
    for q_from, q_to in zip([list(home_q), *via], [*via, q_goal], strict=True):
        go.extend(bridge(q_from, q_to, STEP_RAD))
    back: list[list[float]] = [list(q_goal)]
    for q_from, q_to in zip([q_goal, *reversed(via)], [*reversed(via), list(home_q)], strict=True):
        back.extend(bridge(q_from, q_to, STEP_RAD))
    result["traj_go"] = trajectory(name, go)
    result["traj_back"] = trajectory(name, back)
    limits, box = load_limits(), load_box()
    tip_of = _fk_tip()
    for leg, traj in (("go", result["traj_go"]), ("back", result["traj_back"])):
        result["problems"].extend(f"{leg}: {p}" for p in validate(traj, limits, tip_of, box))
    tip, pitch = fk(q_goal)
    result.update(q=[float(v) for v in q_goal], pitch_deg=math.degrees(pitch), planned_tip=list(tip))
    return result


def _fk_tip() -> Any:
    from arm.ik import fk_tip

    return fk_tip


def stream_fake(driver: Any, traj: dict[str, Any]) -> bool:
    """Send every waypoint through `driver.command` (FakeDriver only; a stand-in for replay())."""
    if not getattr(driver, "is_fake", False):
        raise RuntimeError("stream_fake is for the FakeDriver only")
    gripper = driver.read()[1]
    for segment in traj["segments"]:
        if segment["kind"] != "path":
            continue
        for q in segment["q"]:
            if not driver.command(q, gripper):
                return False
    return True


def run_trajectory(
    driver: Any, traj: dict[str, Any], *, fake: bool, scene: Callable[[], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """`arm.replay.replay` on the driver with `scene` (the tracker's `latest`) as the hand source, so the
    wristband checks run; in --fake mode a placeholder replay() falls back to streaming."""
    from arm.replay import replay

    try:
        return replay(traj, driver=driver, realtime=not driver.is_fake, scene=scene)
    except NotImplementedError as exc:
        if not (fake and getattr(driver, "is_fake", False)):
            raise
        print(f"note: arm.replay.replay is still a placeholder ({exc}); streaming on the FakeDriver")
        ok = stream_fake(driver, traj)
        return {
            "ok": ok,
            "reason": None if ok else "stopped",
            "events": [],
            "note": "streamed on the FakeDriver",
        }


def ask_gap(name: str, fallback_cm: float) -> float:
    try:
        text = input(f"{name}: measured gap tip -> marker center in cm (blank = {fallback_cm:.2f}): ").strip()
    except EOFError:
        return fallback_cm
    if not text:
        return fallback_cm
    try:
        return float(text.replace(",", "."))
    except ValueError:
        print("not a number; keeping the fk estimate")
        return fallback_cm


def _countdown(seconds: int, watcher: Any) -> bool:
    for n in range(seconds, 0, -1):
        print(f"Moving the REAL arm in {n}... keep clear (Esc aborts)", flush=True)
        for _ in range(10):
            time.sleep(0.1)
            if watcher.pressed:
                print("Aborted before connecting.")
                return False
    return True


def _scene_and_hand(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Callable[[], dict[str, Any]] | None]:
    """The scene the markers are planned from (the median of 10 frames) and the hand source for replay.

    --fake --chain: a fresh SceneTracker on the FakeCamera and no hand source (that tracker is not
    running and the fake scene shows no hand). Otherwise the process-wide `vision.scene.get_tracker()`:
    its `snapshot(10)` and its `latest`, so that in --real the running SceneTracker feeds arm.replay's
    wristband rules (freeze within 10 cm of the band, 0.1 m/s within 15 cm). RuntimeError when --real
    gets a fake tracker (FAKE_CAMERA=1 from .env, or a fake installed in-process): a plan from
    config/fake_scene.json must never reach the real arm, and a fake shows no real hand.
    """
    if args.fake and args.chain:
        from vision.scene import fake_chain_tracker

        return fake_chain_tracker().snapshot(10), None
    tracker = _checked_tracker(args)
    return tracker.snapshot(10), tracker.latest


def _checked_tracker(args: argparse.Namespace) -> Any:
    """The process-wide `vision.scene.get_tracker()`; RuntimeError when --real gets a fake (FAKE_CAMERA=1
    from .env, or a fake installed in-process): a plan from config/fake_scene.json must never reach the
    real arm, and a fake shows no real hand."""
    from vision.scene import get_tracker

    tracker = get_tracker()
    if args.real and getattr(tracker, "is_fake", True) is not False:
        raise RuntimeError(
            f"--real but vision.scene.get_tracker() is a {type(tracker).__name__}, a fake: refusing (the "
            "plan would come from config/fake_scene.json and the wristband check would see no real hand; "
            "unset FAKE_CAMERA/FAKE or use --fake)"
        )
    return tracker


def _loose_targets_and_hand(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[int, int] | None, Callable[[], dict[str, Any]] | None]:
    """Every object-size marker found on the table (`loose_targets`: main() then leaves out the ids that
    config/objects.yaml assigns, `leave_out_object_cards`), the frames each id was seen in (None when the
    table was synthesized, so nothing was detected) and the hand source for replay.

    --fake: `fake_detections` of `loose_fake_scene` of the process-wide tracker's snapshot (the
    FakeSceneTracker, FAKE_CAMERA being forced), the hand source its `latest` (which shows no hand).
    --fake --chain: the same table rendered by a FakeCamera, 10 frames detected and mapped through
    `synthetic_homography`, the FakeCamera's own camera model (config/homography.npy is the physical
    camera's once B5 has run, and would misplace every synthetic marker), no hand source.
    Both lay the object cards flat only with `args.object_cards_flat`; otherwise the objects keep standing.
    --real: 10 frames of the running `vision.scene.get_tracker()` mapped through the saved homography
    (FileNotFoundError until B5 writes it) and its `latest` as the hand source; a fake tracker is refused
    exactly as without --loose, and so (RuntimeError, fail-closed) is a tracker whose `camera` is not
    marked `is_fake = False`: its frames would be synthetic.
    """
    if args.fake and args.chain:
        from vision.calibrate import load_scene_cfg, synthetic_homography
        from vision.cameras import FakeCamera, load_fake_scene
        from vision.markers import marker_table

        cfg = load_scene_cfg()
        table = loose_fake_scene(load_fake_scene(), object_cards_flat=args.object_cards_flat)
        camera = FakeCamera(table, scene_cfg=cfg, objects=marker_table())
        frames = collect_frames(camera.read, LOOSE_FRAMES)
        targets, counts = loose_targets(frames, synthetic_homography(cfg))
        return targets, counts, None
    tracker = _checked_tracker(args)
    if args.fake:
        from vision.calibrate import camera_pose

        c, hc = camera_pose()
        table = loose_fake_scene(tracker.snapshot(LOOSE_FRAMES), object_cards_flat=args.object_cards_flat)
        return fake_detections(table, c, hc), None, tracker.latest
    camera = getattr(tracker, "camera", None)
    if getattr(camera, "is_fake", True) is not False:
        raise RuntimeError(
            f"--real --loose but the tracker's camera is a {type(camera).__name__}, not the overhead "
            "camera: refusing (the loose markers would come from synthetic frames; use --fake)"
        )
    from vision.calibrate import load_homography

    h = load_homography()
    targets, counts = loose_targets(collect_frames(tracker_frames(tracker), LOOSE_FRAMES), h)
    return targets, counts, tracker.latest


def _count_arg(text: str) -> int:
    """argparse type for --count: an integer from 1 to the number of object-size ids (10)."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if not 1 <= value <= len(LOOSE_IDS):
        raise argparse.ArgumentTypeError(
            f"must be an integer from 1 to {len(LOOSE_IDS)} (the object-size ids "
            f"{LOOSE_IDS[0]}-{LOOSE_IDS[-1]}), got {text!r}"
        )
    return value


def _positive(text: str) -> float:
    """argparse type for --above (m) and --limit (cm): a finite number > 0 (a hover height of 0 or less
    would drive the tip into the object)."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="I2: hover above every seen object marker and measure.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="dry run on the FakeDriver and fake scene")
    mode.add_argument("--real", action="store_true", help="move the REAL arm (confirmation, countdown, Esc)")
    parser.add_argument("--chain", action="store_true", help="--fake: real SceneTracker on the FakeCamera")
    parser.add_argument(
        "--loose",
        action="store_true",
        help="I2: touch the loose object-size markers (ids 10-19) the camera sees lying on the table",
    )
    parser.add_argument(
        "--count",
        type=_count_arg,
        default=None,
        help=f"--loose: loose markers the camera must see before anything moves (1-{len(LOOSE_IDS)}, "
        f"default {LOOSE_COUNT})",
    )
    parser.add_argument(
        "--object-cards-flat",
        action="store_true",
        help="--loose: also target the ids config/objects.yaml assigns (left out by default: those cards "
        "are glued on objects standing on the table). Only when those cards lie FLAT on the table and the "
        "objects are taken away",
    )
    parser.add_argument(
        "--above", type=_positive, default=ABOVE_M, help="hover height above the marker (m, > 0)"
    )
    parser.add_argument("--limit", type=_positive, default=LIMIT_CM, help="pass limit on the gap (cm, > 0)")
    parser.add_argument(
        "--objects", nargs="*", default=None, help="only these object names (marker_<id> with --loose)"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="results directory")
    args = parser.parse_args(argv)
    if args.count is not None and not args.loose:
        parser.error("--count needs --loose")
    if args.object_cards_flat and not args.loose:
        parser.error("--object-cards-flat needs --loose")
    count = args.count if args.count is not None else LOOSE_COUNT
    wanted = set(args.objects) if args.objects else None
    assigned: dict[int, str] = {}
    if args.loose:
        assigned = {} if args.object_cards_flat else object_card_ids()
        possible = len(LOOSE_IDS) - len(assigned)
        if count > possible:
            parser.error(
                f"--count {count} but only {possible} object-size ids may be targets (config/objects.yaml "
                f"assigns {' '.join(str(i) for i in assigned)}, left out without --object-cards-flat)"
            )
    if args.loose and wanted:
        bad = sorted(wanted - {f"marker_{i}" for i in LOOSE_IDS})
        if bad:
            parser.error(
                f"--loose names its targets marker_<id> for ids {LOOSE_IDS[0]}-{LOOSE_IDS[-1]}; "
                f"--objects got {' '.join(bad)}"
            )
        glued = sorted(name for name in wanted if int(name.split("_", 1)[1]) in assigned)
        if glued:
            parser.error(
                f"--objects {' '.join(glued)}: config/objects.yaml assigns those ids, so their cards are "
                "glued on objects and left out; pass --object-cards-flat only if they lie flat on the table "
                "with the objects taken away"
            )

    if args.fake:
        os.environ["FAKE_ROBOT"] = "1"
        if not args.chain:
            os.environ["FAKE_CAMERA"] = "1"
    else:
        from dotenv import load_dotenv

        load_dotenv()
        fakes = [flag for flag in FAKE_FLAGS if os.environ.get(flag) == "1"]
        if fakes:
            print(
                f"--real with {', '.join(fakes)}=1 set: refusing (the plan would come from the fake scene or "
                "go to the fake arm; unset the fakes or use --fake)",
                file=sys.stderr,
            )
            return 2

    from arm.driver import EscWatcher, get_driver
    from arm.ik import fk_tip
    from vision.calibrate import camera_pose

    counts: dict[int, int] | None = None
    left_out: list[tuple[str, dict[str, Any]]] = []
    try:
        if args.loose:
            targets, counts, hand_source = _loose_targets_and_hand(args)
        else:
            scene, hand_source = _scene_and_hand(args)
            targets = [(name, item) for name, item in scene.get("objects", {}).items() if item.get("seen")]
    except (RuntimeError, TimeoutError, FileNotFoundError, ValueError) as exc:
        print(f"cannot read the scene: {exc}; nothing moved", file=sys.stderr)
        return 2
    if args.loose:
        targets, left_out = leave_out_object_cards(
            targets, assigned, object_cards_flat=args.object_cards_flat
        )
        for _, item in left_out:
            marker_id = int(item["marker"])
            print(
                f"left out: marker {marker_id} ({assigned[marker_id]} in config/objects.yaml): its card is "
                "glued on an object standing on the table, and a hover planned from the table there would "
                "drive the tip into it (--object-cards-flat only if it lies flat with the object removed)"
            )
        if args.object_cards_flat:
            print(
                "--object-cards-flat: every object-size card seen is a target; the objects must be OFF the "
                "table and their cards lying flat"
            )
        ids = [item["marker"] for _, item in targets]
        print(
            f"loose markers on the table: {ids if ids else 'none'} ({len(ids)} of ids "
            f"{LOOSE_IDS[0]}-{LOOSE_IDS[-1]} seen, {count} needed)"
        )
        dropped = {int(item["marker"]) for _, item in left_out}
        for marker_id, n in (counts or {}).items():
            if marker_id not in ids and marker_id not in dropped:
                print(f"note: marker {marker_id} seen in only {n} of {LOOSE_FRAMES} frames: not a target")
        if len(targets) < count:
            print(
                f"only {len(targets)} loose markers seen; need {count} (--count); nothing moved",
                file=sys.stderr,
            )
            return 1
        unseen = sorted(wanted - {name for name, _ in targets}) if wanted else []
        if unseen:
            print(
                f"{' '.join(unseen)} named with --objects but not among the loose markers seen; "
                "nothing moved",
                file=sys.stderr,
            )
            return 1
    objects = [(name, item) for name, item in targets if wanted is None or name in wanted]
    if not objects:
        print("no seen object markers in the scene; nothing to do", file=sys.stderr)
        return 1
    c, hc = camera_pose()
    home_q = load_home_q()
    plans = [
        plan(name, item, home_q, above=args.above, loose=args.loose, c=c, hc=hc) for name, item in objects
    ]
    for p in plans:
        status = "; ".join(p["problems"]) if p["problems"] else f"ok, pitch {p['pitch_deg']:+.0f} deg"
        print(
            f"plan {p['name']:12s} marker {p['marker']} target "
            f"{tuple(round(v, 3) for v in p['target'])}: {status}"
        )

    driver = get_driver()
    watcher = None
    connected_at: float | None = None
    results: list[dict[str, Any]] = []
    aborted = False
    if args.real:
        answer = input(
            "Is the arm clear, the table set and the 12 V plug within reach? Type yes to continue: "
        )
        if answer.strip().lower() != "yes":
            print("aborted; nothing moved")
            return 1
        watcher = EscWatcher(callback=driver.freeze)
        watcher.start()
        if not _countdown(3, watcher):
            watcher.stop()
            return 1
        try:
            driver.connect()
        except BaseException:
            watcher.stop()
            raise
        connected_at = time.monotonic()  # Esc presses before this are connect-phase (A13)
        # An Esc that lands inside connect() (port open, goal sync, configure) must not be lost: the
        # driver may end connect() with `stopped` cleared, so the watcher's flag counts too.
        if watcher.pressed and not driver.stopped:
            driver.freeze()
        if driver.stopped:
            print("Esc during connect: the arm is FROZEN with torque on; nothing moved", file=sys.stderr)
            aborted = True
    try:
        for p in plans:
            if aborted:
                break
            entry: dict[str, Any] = {
                k: p[k] for k in ("name", "marker", "marker_center", "target", "problems")
            }
            if p["problems"]:
                entry.update(ok=False, gap_cm=None)
                results.append(entry)
                continue
            entry.update(q=p["q"], pitch_deg=p["pitch_deg"], planned_tip=p["planned_tip"])
            go = run_trajectory(driver, p["traj_go"], fake=args.fake, scene=hand_source)
            entry["replay_go"] = {k: go.get(k) for k in ("ok", "reason", "note")}
            if not go.get("ok"):
                entry.update(ok=False, gap_cm=None)
                results.append(entry)
                print(f"{p['name']}: replay failed ({go.get('reason')}); stopping here")
                aborted = True
                break
            q_now, _ = driver.read()
            tip = fk_tip(q_now)
            mx, my, mz = p["marker_center"]
            gap_fk = math.hypot(tip[0] - mx, tip[1] - my) * 100.0
            dz_cm = (tip[2] - (mz + args.above)) * 100.0
            gap = ask_gap(p["name"], gap_fk) if args.real else gap_fk
            entry.update(
                reached_tip=list(tip), gap_fk_cm=gap_fk, dz_cm=dz_cm, gap_cm=gap, ok=gap <= args.limit
            )
            print(f"{p['name']:12s} gap {gap:.2f} cm (fk {gap_fk:.2f} cm, height error {dz_cm:+.2f} cm)")
            back = run_trajectory(driver, p["traj_back"], fake=args.fake, scene=hand_source)
            entry["replay_back"] = {k: back.get(k) for k in ("ok", "reason", "note")}
            results.append(entry)
            if not back.get("ok"):
                print(f"{p['name']}: return leg failed ({back.get('reason')}); stopping here")
                aborted = True
                break
    finally:
        if args.real:
            if watcher is not None:
                from arm.driver import report_esc_latency

                report_esc_latency(
                    watcher,
                    driver,
                    source="touch_markers",
                    connected_at=connected_at,
                    extra={"argv": list(sys.argv[1:] if argv is None else argv)},
                )
            if driver.stopped:
                print("Arm is FROZEN with torque on (Esc/STOP). Resume from the orchestrator or power-cycle.")
            driver.disconnect()  # torque stays on
        if watcher is not None:
            watcher.stop()

    gaps = [r["gap_cm"] for r in results if r.get("gap_cm") is not None]
    worst = max(gaps) if gaps else None
    passed = bool(gaps) and not aborted and all(r["ok"] for r in results)
    summary = {
        "t": time.time(),
        "mode": "fake" if args.fake else "real",
        "chain": bool(args.chain),
        "loose": bool(args.loose),
        "above_m": args.above,
        "limit_cm": args.limit,
        "worst_cm": worst,
        "passed": passed,
        "aborted": aborted,
        "markers": results,
    }
    if args.loose:
        summary["count"] = count
        summary["object_cards_flat"] = bool(args.object_cards_flat)
        summary["left_out"] = [
            {"marker": int(item["marker"]), "name": assigned[int(item["marker"])]} for _, item in left_out
        ]
        summary["loose_detections"] = (
            None if counts is None else [{"marker": m, "frames": n} for m, n in sorted(counts.items())]
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if worst is None:
        print(f"no gap measured; results in {out / 'results.json'}")
        return 1
    verdict = "PASS" if passed else "FAIL"
    print(
        f"worst gap {worst:.2f} cm against the {args.limit:.2f} cm limit over {len(gaps)} markers: {verdict}"
    )
    print(f"results in {out / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
