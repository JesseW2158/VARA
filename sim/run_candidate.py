"""Score one candidate program or saved skill on the MuJoCo sim (task C4; brief sections 4, 6 and 8.4).

    python -m sim.run_candidate (--candidate cand.py --task task.json | --skill skill.json --args '{...}')
        --scene scene.json --trials 5 --seed 7 --out DIR [--index N] [--skills-dir DIR] [--timeout 60]

This process is the security boundary for AI-written code: candidates, saved skills and routines run only
here, inside a Daytona sandbox or the orchestrator's clean local subprocess (environment PATH, PYTHONPATH,
SYSTEMROOT only), never in the orchestrator. Three layers: `sim.candidate_check.check_source` filters the
source (run again here, whatever the orchestrator checked), `sim.restricted.load_entry` executes it in a
trimmed namespace (restricted `__import__` serving only `math`, `print` to stderr), and every trial runs in
ONE child process (multiprocessing spawn, same interpreter) that the parent kills on the wall clock.

Parent (this process): reads the inputs (strict JSON through `loads_strict`: NaN and Infinity are refused;
the candidate file must be UTF-8), validates the scene (schemas/scene.json), the task spec or skill file
and `--args`, runs `check_source` (a rejected candidate gets `sim.scoring.rejected_score`, exit 0), starts
the child, enforces `--timeout` seconds of wall clock per trial (terminate, then kill; the pid is logged
and the kill confirmed), summarizes with `sim.scoring.summarize`, writes `out/score.json` and
`out/trials.json` (the per-trial list of brief 8.4), and prints the score line (schemas/score.json) as the
LAST and only line of stdout. Everything else goes to stderr. Exit 0 whenever a score line was printed,
even for a failed candidate; non-zero only when the runner itself refuses or breaks (bad arguments or
inputs, including a scene the simulator rejects because it names an object the model lacks: 2; the child
died, hung at start-up or broke the protocol: 1).

Child: redirects stdout to stderr (the Python object and the file descriptor, so MuJoCo's own messages land
there too), builds one SimRobot and loads the candidate once, then for each trial: `perturb_scene`,
`reset(scene)`, `set_skill(name, args)`, `run_entry(fn, args)` under `SimTimeLimit` with every exception
caught as the trial's error (BaseException included: a SystemExit or KeyboardInterrupt raised by the
candidate is its failure, not the runner's; error texts are clipped to `sim.scoring.MAX_TEXT_CHARS` and
an exception whose str() itself fails is reported by type), sim time past the limit counted as an error
even when the candidate swallowed `SimTimeLimit`, success from
`sim.success.evaluate(spec, scene_before, robot.final_scene(), args, robot.events)` (the candidate's return
value is logged, not scored), and for trial 1 `out/traj.json` (robot.trajectory() with the skill and args
set, validated against schemas/trajectory.json) and `out/final_scene.json` (schemas/scene.json). One JSON
line per trial streams back through a pipe. A second pipe is a dead-man's switch: when the parent dies,
the child's watchdog thread sees end-of-file and exits, so a killed runner leaves no orphan burning CPU.

Trials after a timeout do not run; the score counts the requested `--trials` and `fail_reason` is
"trial k: timeout". `--index` defaults to the N in `cand_N.py`, else 0. `--skill` loads a skill file
(schemas/skill.json) and scores `skill["code"]` against `skill["spec"]` with `--args`; `--candidate` scores a
source file against a task spec (schemas/task_spec.json), also with `--args` (default `{}`). `--skills-dir`
is where `robot.run_skill` (routines) finds saved skills; the default is `skills/` in the repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import pathlib
import re
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from jsonschema import Draft202012Validator

from sim import scoring
from sim.candidate_check import check_source

__all__ = ["DEFAULT_TIMEOUT", "candidate_index", "loads_strict", "main"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"

DEFAULT_TRIALS = 5
DEFAULT_SEED = 7
DEFAULT_TIMEOUT = 60.0  # s of wall clock per trial
STARTUP_TIMEOUT = 120.0  # s for the child to import MuJoCo, load the model and the candidate
POLL_INTERVAL = 0.25  # s between checks that the child is still alive while waiting for a trial
KILL_GRACE = 5.0  # s to wait after terminate() and again after kill()
EXIT_RUNNER_BROKE = 1
EXIT_USAGE = 2

TRAJ_FILE = "traj.json"
FINAL_SCENE_FILE = "final_scene.json"
SCORE_FILE = "score.json"
TRIALS_FILE = "trials.json"

log = logging.getLogger("sim.run_candidate")


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


_SCENE_VALIDATOR = _validator("scene.json")
_TASK_VALIDATOR = _validator("task_spec.json")
_SKILL_VALIDATOR = _validator("skill.json")
_SCORE_VALIDATOR = _validator("score.json")
_TRAJ_VALIDATOR = _validator("trajectory.json")


class UsageError(Exception):
    """Bad arguments or unreadable, invalid inputs: the runner refuses to start (exit 2)."""


class RunnerError(Exception):
    """The runner itself broke: the child died, hung at start-up or spoke out of protocol (exit 1)."""


# ---------------------------------------------------------------------------- arguments and inputs


def candidate_index(path: str | os.PathLike[str] | None) -> int:
    """The N of a `cand_N.py` file name, else 0 (the `--index` default)."""
    if path is None:
        return 0
    match = re.search(r"cand_(\d+)\.py$", pathlib.Path(path).name)
    return int(match.group(1)) if match else 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sim.run_candidate",
        description="Score a candidate program or a saved skill over randomized MuJoCo trials.",
    )
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--candidate", help="candidate source file (cand_N.py)")
    what.add_argument("--skill", help="saved skill file (schemas/skill.json)")
    p.add_argument("--task", help="task spec file (schemas/task_spec.json), with --candidate")
    p.add_argument("--args", default="{}", help="JSON object of arguments for run(robot, args)")
    p.add_argument("--scene", required=True, help="scene file (schemas/scene.json)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of trials (default 5)")
    p.add_argument("--seed", default=str(DEFAULT_SEED), help="perturbation seed (default 7)")
    p.add_argument("--out", required=True, help="output directory (traj, final_scene, score, trials)")
    p.add_argument("--index", type=int, default=None, help="candidate index (default: the N of cand_N.py)")
    p.add_argument("--skills-dir", default=None, help="directory robot.run_skill reads skills from")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="wall-clock seconds per trial")
    return p


def _no_json_constant(token: str) -> Any:
    raise ValueError(f"{token} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def loads_strict(text: str) -> Any:
    """`json.loads` that refuses NaN, Infinity and overflowing floats: Python's parser accepts them, JSON
    does not, and a NaN coordinate would reach MuJoCo (which then writes MUJOCO_LOG.TXT into the cwd)."""
    return json.loads(text, parse_constant=_no_json_constant, parse_float=_finite_float)


def _read_json(path: str, what: str) -> Any:
    try:
        return loads_strict(pathlib.Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"cannot read {what} {path}: {exc}") from exc
    except ValueError as exc:
        raise UsageError(f"{what} {path} is not JSON: {exc}") from exc


def _check(validator: Draft202012Validator, doc: Any, what: str) -> None:
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "(root)"
        raise UsageError(f"{what} is invalid at {where}: {first.message}")


def load_job(ns: argparse.Namespace) -> dict:
    """The validated inputs as one plain dict (it is pickled to the child): source, spec, args, skill name,
    scene, skills_dir, index, label. Raises UsageError."""
    if ns.trials < 1:
        raise UsageError("--trials must be 1 or more")
    if not (ns.timeout > 0):
        raise UsageError("--timeout must be positive")
    try:
        args = loads_strict(ns.args)
    except ValueError as exc:
        raise UsageError(f"--args is not JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise UsageError("--args must be a JSON object")
    scene = _read_json(ns.scene, "scene")
    _check(_SCENE_VALIDATOR, scene, "scene")
    if ns.candidate is not None:
        if ns.task is None:
            raise UsageError("--candidate needs --task")
        try:
            source = pathlib.Path(ns.candidate).read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"cannot read candidate {ns.candidate}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise UsageError(f"candidate {ns.candidate} is not UTF-8: {exc}") from exc
        spec = _read_json(ns.task, "task spec")
        _check(_TASK_VALIDATOR, spec, "task spec")
        name = spec["name"]
        label = f"candidate {pathlib.Path(ns.candidate).name}"
        index = ns.index if ns.index is not None else candidate_index(ns.candidate)
    else:
        if ns.task is not None:
            raise UsageError("--task goes with --candidate; --skill carries its own spec")
        skill = _read_json(ns.skill, "skill")
        _check(_SKILL_VALIDATOR, skill, "skill")
        source, spec, name = skill["code"], skill["spec"], skill["name"]
        label = f"skill {name}"
        index = ns.index if ns.index is not None else 0
    if index < 0:
        raise UsageError("--index must be 0 or more")
    return {
        "source": source,
        "spec": spec,
        "args": args,
        "skill": name,
        "scene": scene,
        "skills_dir": ns.skills_dir,
        "index": int(index),
        "label": label,
    }


# ---------------------------------------------------------------------------- the child process


def _silence_stdout() -> None:
    """Route everything the child (or MuJoCo) writes to stdout into stderr: the score line is the parent's."""
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    try:
        os.dup2(sys.stderr.fileno(), 1)
    except (OSError, ValueError, AttributeError):
        pass
    sys.stdout = sys.stderr


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _send(conn: Any, message: dict) -> None:
    conn.send(json.dumps(message))


def _watch_parent(reader: Any) -> None:
    """Dead-man's switch: the parent never writes on this pipe, so end-of-file means it is gone."""
    try:
        reader.recv()
    except (EOFError, OSError):
        pass
    except Exception:  # noqa: BLE001 - whatever happened, the parent is not there to report to
        pass
    os._exit(3)


def _describe(exc: BaseException) -> str:
    """`Type: message`, clipped to `scoring.MAX_TEXT_CHARS`. The message is replaced by a note when str()
    itself fails (a candidate can raise ValueError(d) with d nested deeper than the recursion limit)."""
    name = type(exc).__name__
    try:
        text = str(exc)
    except Exception as err:  # noqa: BLE001 - whatever str() raised, the trial still gets an error text
        text = f"<message not printable: {type(err).__name__}>"
    return scoring.clip(f"{name}: {text}" if text else name)


def _run_trial(robot: Any, fn: Any, job: dict, k: int, seed: str, out_dir: pathlib.Path) -> dict:
    """One trial inside the child: reset, run the candidate, score, write trial 1's files."""
    from arm.sim_robot import SIM_TIME_LIMIT, SimTimeLimit
    from sim.success import evaluate

    scene = scoring.perturb_scene(job["scene"], k, seed)
    error: str | None = None
    returned: bool | None = None
    success = False
    t0 = time.perf_counter()
    try:
        robot.reset(scene)
        robot.set_skill(job["skill"], job["args"])
        returned = robot.run_entry(fn, job["args"])
    except SimTimeLimit as exc:
        error = _describe(exc)
    except BaseException as exc:  # the candidate may raise anything, SystemExit and KeyboardInterrupt too
        error = _describe(exc)
    wall = time.perf_counter() - t0
    stats = robot.stats
    if error is None and float(stats.get("sim_s") or 0.0) > SIM_TIME_LIMIT:
        error = f"SimTimeLimit: simulated time passed {SIM_TIME_LIMIT:.0f} s (swallowed by the candidate)"
    if error is None:
        try:
            success = bool(evaluate(job["spec"], scene, robot.final_scene(), job["args"], robot.events))
        except Exception as exc:
            error = f"evaluate: {_describe(exc)}"
    traj_written = False
    if k == 1:
        try:
            traj = robot.trajectory()
            _TRAJ_VALIDATOR.validate(traj)
            final = robot.final_scene()
            _SCENE_VALIDATOR.validate(final)
            (out_dir / TRAJ_FILE).write_text(json.dumps(traj), encoding="utf-8")
            (out_dir / FINAL_SCENE_FILE).write_text(json.dumps(final, indent=2), encoding="utf-8")
            traj_written = True
        except Exception as exc:
            if error is None:
                error = f"trajectory: {_describe(exc)}"
                success = False
            _say(f"trial {k}: could not write the trajectory: {_describe(exc)}")
    result = scoring.trial_result(k, success, stats, error)
    _say(
        f"trial {k}: success={success} returned={returned} sim_s={result['sim_s']} "
        f"collisions={result['collisions']} limit_hits={result['limit_hits']} "
        f"ik_failures={result['ik_failures']} first_collision={result['first_collision']!r} "
        f"error={error!r} wall={wall:.1f} s"
    )
    return {
        "kind": "trial",
        "result": result,
        "returned": returned,
        "wall_s": round(wall, 3),
        "traj": traj_written,
    }


def _child_main(conn: Any, watch: Any, job: dict, out: str, trials: int, seed: str) -> None:
    """Entry point of the spawned child (see the module docstring)."""
    _silence_stdout()
    threading.Thread(target=_watch_parent, args=(watch,), name="watch-parent", daemon=True).start()
    out_dir = pathlib.Path(out)
    try:
        from arm.sim_robot import SimRobot
        from sim.restricted import load_entry

        skills_dir = pathlib.Path(job["skills_dir"]) if job.get("skills_dir") else None
        robot = SimRobot(job["scene"], skills_dir=skills_dir)
    except ValueError as exc:
        # SimRobot.reset raises ValueError for a scene naming objects or obstacles the model lacks: an
        # input the parent cannot check without loading MuJoCo, so it becomes a usage error (exit 2)
        message = f"the simulator rejects the inputs: {_describe(exc)}"
        _send(conn, {"kind": "fatal", "input": True, "message": message})
        return
    except Exception as exc:
        _send(conn, {"kind": "fatal", "message": f"could not build the simulator: {_describe(exc)}"})
        return
    try:
        fn = load_entry(job["source"], "run", filename=f"<{job['label']}>")
    except Exception as exc:
        # check_source passed in the parent, so this is `run` not callable after exec or a disagreement
        _send(conn, {"kind": "ready"})
        result = scoring.trial_result(1, False, None, _describe(exc))
        _say(f"trial 1: the candidate could not be loaded: {result['error']}")
        _send(conn, {"kind": "trial", "result": result, "returned": None, "wall_s": 0.0, "traj": False})
        _send(conn, {"kind": "end", "reason": "candidate not loadable"})
        return
    _send(conn, {"kind": "ready"})
    try:
        for k in range(1, trials + 1):
            _send(conn, _run_trial(robot, fn, job, k, seed, out_dir))
    except KeyboardInterrupt:
        return
    except Exception as exc:
        _send(conn, {"kind": "fatal", "message": f"trial loop broke: {_describe(exc)}"})
        return
    _send(conn, {"kind": "end", "reason": "all trials ran"})


# ---------------------------------------------------------------------------- the parent side


class _ChildExited(Exception):
    """The child closed the pipe or died without a message."""


def _await(conn: Any, proc: Any, timeout: float) -> dict | None:
    """The next message from the child, or None when `timeout` seconds pass without one."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        if conn.poll(min(remaining, POLL_INTERVAL)):
            try:
                raw = conn.recv()
            except (EOFError, OSError, ValueError) as exc:
                raise _ChildExited() from exc
            try:
                message = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise RunnerError(f"child sent something that is not JSON: {raw!r}") from exc
            if not isinstance(message, dict) or "kind" not in message:
                raise RunnerError(f"child sent a message without a kind: {message!r}")
            return message
        if not proc.is_alive() and not conn.poll(0):
            raise _ChildExited()


def _kill(proc: Any, why: str) -> bool:
    """Terminate, then kill; True when the child is confirmed gone."""
    if proc.is_alive():
        proc.terminate()
        proc.join(KILL_GRACE)
    if proc.is_alive():
        proc.kill()
        proc.join(KILL_GRACE)
    gone = not proc.is_alive()
    if gone:
        log.info("child pid %s killed (%s); exit code %s; no orphan", proc.pid, why, proc.exitcode)
    else:
        log.error("child pid %s survived terminate and kill (%s): an orphan is running", proc.pid, why)
    return gone


def run_trials(job: dict, out_dir: pathlib.Path, trials: int, seed: str, timeout: float) -> tuple[list, bool]:
    """Run the trials in one spawned child. Returns (trial results in order, trial 1 wrote traj.json).
    Raises RunnerError when the child died, hung at start-up or broke the protocol, UsageError when the
    simulator rejected the scene (an object or obstacle the model does not have)."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    watch_reader, watch_writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_main,
        args=(child_conn, watch_reader, job, str(out_dir), trials, seed),
        name="trials",
        daemon=True,
    )
    results: list[dict] = []
    traj_written = False
    proc.start()
    child_conn.close()
    watch_reader.close()
    log.info(
        "child pid %s started for %s: %d trial(s), seed %s, %.1f s per trial",
        proc.pid,
        job["label"],
        trials,
        seed,
        timeout,
    )
    try:
        try:
            first = _await(parent_conn, proc, STARTUP_TIMEOUT)
        except _ChildExited as exc:
            raise RunnerError(
                f"child pid {proc.pid} exited with code {proc.exitcode} before it was ready"
            ) from exc
        if first is None:
            _kill(proc, "did not start")
            raise RunnerError(f"child pid {proc.pid} did not start within {STARTUP_TIMEOUT:.0f} s")
        if first["kind"] == "fatal":
            if first.get("input"):
                raise UsageError(str(first.get("message")))
            raise RunnerError(str(first.get("message")))
        if first["kind"] != "ready":
            raise RunnerError(f"child sent {first['kind']!r} before it was ready")
        for k in range(1, trials + 1):
            started = time.monotonic()
            try:
                message = _await(parent_conn, proc, timeout)
            except _ChildExited as exc:
                raise RunnerError(
                    f"child pid {proc.pid} exited with code {proc.exitcode} during trial {k}"
                ) from exc
            if message is None:
                elapsed = time.monotonic() - started
                log.warning(
                    "trial %d: timeout after %.1f s of wall clock (limit %.1f s)", k, elapsed, timeout
                )
                _kill(proc, f"trial {k} timeout")
                results.append(scoring.trial_result(k, False, None, scoring.TIMEOUT))
                break
            if message["kind"] == "fatal":
                raise RunnerError(str(message.get("message")))
            if message["kind"] == "end":
                log.info("child ended early: %s", message.get("reason"))
                break
            if message["kind"] != "trial":
                raise RunnerError(f"child sent {message['kind']!r} instead of trial {k}")
            result = message["result"]
            if int(result.get("trial", -1)) != k:
                raise RunnerError(f"child sent trial {result.get('trial')!r} instead of trial {k}")
            if k == 1:
                traj_written = bool(message.get("traj"))
            results.append(scoring.trial_result(k, result["success"], result, result.get("error")))
        proc.join(KILL_GRACE)
        if proc.is_alive():
            _kill(proc, "did not exit after the trials")
    finally:
        if proc.is_alive():
            _kill(proc, "runner stopping")
        parent_conn.close()
        watch_writer.close()
    return results, traj_written


def _write_json(path: pathlib.Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    ns = _parser().parse_args(argv)
    try:
        job = load_job(ns)
    except UsageError as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    out_dir = pathlib.Path(ns.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in (TRAJ_FILE, FINAL_SCENE_FILE, SCORE_FILE, TRIALS_FILE):
            (out_dir / stale).unlink(missing_ok=True)
    except OSError as exc:
        log.error("cannot prepare %s: %s", out_dir, exc)
        return EXIT_USAGE
    problems = check_source(job["source"])
    if problems:
        log.warning("%s rejected: %s", job["label"], "; ".join(problems))
        score = scoring.rejected_score(job["index"], ns.trials, problems[0])
        results: list[dict] = []
    else:
        try:
            results, traj_written = run_trials(job, out_dir, ns.trials, str(ns.seed), float(ns.timeout))
        except UsageError as exc:
            log.error("%s", exc)
            return EXIT_USAGE
        except RunnerError as exc:
            log.error("runner failed: %s", exc)
            return EXIT_RUNNER_BROKE
        traj_path = (out_dir / TRAJ_FILE).as_posix() if traj_written else None
        score = scoring.summarize(job["index"], results, traj_path, trials=ns.trials)
    errors = list(_SCORE_VALIDATOR.iter_errors(score))
    if errors:
        log.error("runner produced an invalid score line: %s", errors[0].message)
        return EXIT_RUNNER_BROKE
    try:
        _write_json(out_dir / TRIALS_FILE, results)
        _write_json(out_dir / SCORE_FILE, score)
    except OSError as exc:
        log.error("cannot write %s: %s", out_dir, exc)
        return EXIT_RUNNER_BROKE
    log.info(
        "candidate %d: passed=%s successes=%d/%d fail_reason=%r",
        score["candidate"],
        score["passed"],
        score["successes"],
        score["trials"],
        score["fail_reason"],
    )
    sys.stderr.flush()
    print(json.dumps(score), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
