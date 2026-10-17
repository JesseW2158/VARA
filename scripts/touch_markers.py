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

I2 lays LOOSE markers flat on the table: pass `--loose` so the objects.yaml height correction is undone and
the hover point is 5 cm above the table. Without it, markers are assumed glued on top of their objects.
If the real run misses 1.5 cm, fix config/joint_map.yaml (A6) or the homography (B5) before anything else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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


def load_home_q() -> list[float]:
    cfg = yaml.safe_load((ROOT / "config" / "scene.yaml").read_text(encoding="utf-8"))
    return [float(v) for v in cfg["home_q"]]


def marker_center(
    item: dict[str, Any], *, loose: bool, c: Sequence[float], hc: float
) -> tuple[float, float, float]:
    """World center of the marker: on top of the object, or (loose) back on the table."""
    x, y, h = float(item["x"]), float(item["y"]), float(item["h"])
    if not loose:
        return x, y, h
    k = 1.0 - h / hc  # the tracker applied p = c + (p_table - c) * k for a marker that was really at z = 0
    return c[0] + (x - c[0]) / k, c[1] + (y - c[1]) / k, 0.0


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
    from vision.scene import get_tracker

    tracker = get_tracker()
    if args.real and getattr(tracker, "is_fake", True) is not False:
        raise RuntimeError(
            f"--real but vision.scene.get_tracker() is a {type(tracker).__name__}, a fake: refusing (the "
            "plan would come from config/fake_scene.json and the wristband check would see no real hand; "
            "unset FAKE_CAMERA/FAKE or use --fake)"
        )
    return tracker.snapshot(10), tracker.latest


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
    parser.add_argument("--loose", action="store_true", help="markers lie flat on the table (I2)")
    parser.add_argument(
        "--above", type=_positive, default=ABOVE_M, help="hover height above the marker (m, > 0)"
    )
    parser.add_argument("--limit", type=_positive, default=LIMIT_CM, help="pass limit on the gap (cm, > 0)")
    parser.add_argument("--objects", nargs="*", default=None, help="only these object names")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="results directory")
    args = parser.parse_args(argv)

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

    try:
        scene, hand_source = _scene_and_hand(args)
    except (RuntimeError, TimeoutError, FileNotFoundError, ValueError) as exc:
        print(f"cannot read the scene: {exc}; nothing moved", file=sys.stderr)
        return 2
    wanted = set(args.objects) if args.objects else None
    objects = [
        (name, item)
        for name, item in scene.get("objects", {}).items()
        if item.get("seen") and (wanted is None or name in wanted)
    ]
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
