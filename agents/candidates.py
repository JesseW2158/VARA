"""Candidate programs: the source check, generation, repair and routines (C3, C7, C10).

`check_source` lives in `sim.candidate_check` because the sandbox bundle (`arm/`, `sim/`, `schemas/`,
`config/`) has no `agents/`; this module re-exports it so `from agents.candidates import check_source`
works on the laptop. Nothing here executes AI-written code: candidates are text until `sim.run_candidate`
runs them in a sandbox or a clean subprocess.

- `HINTS`: the six strategy hints from docs/tasks/C-agents.md, verbatim; candidate i follows hint i.
- `generate(spec, scene, skills, *, task_dir, n=6, timeout=45, demo_mode=False, on_rejected=None)`: six
  parallel `chat(..., purpose="write_skill", key=f"{slug}:{i}")` calls on `agents/prompts/write_skill.md`
  (the Robot API and its notes verbatim from docs/interfaces.md, the candidate rules and the trap list, the
  spec in English and JSON, the scene JSON, the two starter skills as worked examples, one hint, and the
  closing line "Return one Python code block and nothing else."). Each reply goes through `extract_code`
  and `check_source`; a rejected candidate is regenerated once with its problems appended (key
  `f"{slug}:{i}:retry"`), unless `demo_mode` is set or the budget is spent. `timeout` is the budget in
  seconds for the whole call (45 s normal, 30 s demo); a candidate that is not back in time comes back with
  the problem "generation timed out" and no source. Files under `task_dir`: `prompts/write_skill_<i>.md`
  (and `_retry`), `replies/write_skill_<i>.md`, `cand_<i>.py`, `cand_<i>.problems.json`.
  `on_rejected(candidate)` is called (from a worker thread) for every attempt-1 rejection.
- `repair(spec, scene, skills, failures, *, task_dir, timeout)`: the same with `agents/prompts/repair.md`
  appended ("fix these failures" with the three best failures: code, score line, fail_reason), purpose
  `repair`, files under `task_dir/repair/` and `prompts/repair_<i>.md`.
- `routine_candidate(steps)`: a `run(robot, args)` that calls `robot.run_skill(step["skill"],
  step["args"])` for each step and returns False at the first False; it passes `check_source`.
- `extract_code(text)`: the first fenced block holding `def run`/`def reward` (or the first block, or an
  unclosed fence's remainder, or the whole text when there is no fence); `extract_code_blocks(text)` returns
  every closed fenced block. Fences are three or more backticks or tildes at the start of a line, closed by
  the same run at the start of a line, so a ``` inside a string literal does not end the block; CRLF, lone
  CR and a BOM are normalized first.
- `generate` and `repair` raise `ValueError` for a `task_id` that is not a plain identifier, before any
  file is written.
- `python -m agents.candidates --eval "<text>"`: route the text, generate, and print how many candidates
  pass `check_source` on the first try (with FAKE_LLM=1 everything comes from the fixtures).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import llm
from agents.llm import chat, codegen_model
from sim.candidate_check import check_source

__all__ = [
    "CLOSING_LINE",
    "Candidate",
    "HINTS",
    "STARTER_EXAMPLES",
    "check_source",
    "extract_code",
    "extract_code_blocks",
    "generate",
    "render_messages",
    "repair",
    "repair_messages",
    "routine_candidate",
    "write_skill_messages",
]

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "agents" / "prompts"
STARTER_DIR = ROOT / "skills" / "_starter"
SKILLS_DIR = ROOT / "skills"
FAKE_SCENE_PATH = ROOT / "config" / "fake_scene.json"

CLOSING_LINE = "Return one Python code block and nothing else."
SYSTEM_PROMPT = (
    "You write Python programs that drive a home-assistant robot arm through the simulated Robot API the "
    "user message describes. Follow its rules exactly. Reply with exactly one Python code block and "
    "nothing else: no explanation before or after it."
)
GRACE_S = 2.0  # how long generate() waits past its budget for replies already on the wire
MIN_RETRY_S = 5.0  # no regeneration when less than this is left of the budget
SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

HINTS: tuple[str, ...] = (
    "Nearest object first, default parameters.",
    "Farthest object first.",
    "Set `transit_z` to the tallest obstacle plus 0.08 m on every pick and place.",
    "Route around obstacles: rise, move out to x = 0.32 m, then sideways.",
    "Slow down to 0.08 m/s and use pitch -π/4 for targets beyond 0.30 m from the base.",
    "Fill tray slots far side first, and retry a failed grasp once.",
)

# The two starter skills (docs/tasks/A-arm.md A12) as worked examples, used when skills/_starter/ is empty.
STARTER_EXAMPLES: dict[str, dict[str, Any]] = {
    "hand_over": {
        "spec": {
            "name": "hand_over",
            "description": "Pick up an object and hand it to the user.",
            "params": {"object": "object"},
            "success": {"all": [{"moved": {"object": "$object", "min": 0.05}}, {"event": "handoff_done"}]},
            "english": "The object leaves its spot and the user takes it from the gripper.",
        },
        "code": (
            '"""hand_over: pick up the named object and carry it to the user\'s hand."""\n'
            "import math\n"
            "\n"
            "\n"
            "def run(robot, args):\n"
            '    name = args["object"]\n'
            "    scene = robot.scene()\n"
            '    if name not in scene["objects"] or not scene["objects"][name]["seen"]:\n'
            "        return False\n"
            '    tallest = max([o["h"] for o in scene["obstacles"].values()] + [0.0])\n'
            "    if not robot.pick(name, transit_z=tallest + 0.08):\n"
            "        return False\n"
            "    return robot.carry_to_hand()\n"
        ),
    },
    "place_in_tray": {
        "spec": {
            "name": "place_in_tray",
            "description": "Put the named object into the tray.",
            "params": {"object": "object"},
            "success": {"in_zone": {"object": "$object", "zone": "tray"}},
            "english": "The object ends inside the tray.",
        },
        "code": (
            '"""place_in_tray: put the named object into the tray, rising above every obstacle."""\n'
            "import math\n"
            "\n"
            "\n"
            "def run(robot, args):\n"
            '    name = args["object"]\n'
            "    scene = robot.scene()\n"
            '    if name not in scene["objects"] or not scene["objects"][name]["seen"]:\n'
            "        return False\n"
            '    transit = max([o["h"] for o in scene["obstacles"].values()] + [0.0]) + 0.08\n'
            "    if not robot.pick(name, transit_z=transit):\n"
            "        return False\n"
            '    if not robot.place_in_zone("tray", transit_z=transit):\n'
            "        return False\n"
            "    robot.home()\n"
            "    return True\n"
        ),
    },
}

_OPEN_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")  # a fence opens (or closes) at the start of a line
_ENTRY_RE = re.compile(r"^[ \t]*def (run|reward)\(", re.MULTILINE)

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One generated program. `problems` empty means `check_source` accepted it."""

    index: int
    hint: str
    source: str
    problems: list[str] = field(default_factory=list)
    attempt: int = 1
    path: Path | None = None

    @property
    def accepted(self) -> bool:
        return not self.problems


@dataclass
class _Job:
    index: int
    hint: str
    messages: list[dict[str, str]]
    purpose: str


# --- prompts -----------------------------------------------------------------------------------------


def write_skill_messages(
    spec: Mapping[str, Any], scene: Mapping[str, Any], skills: Sequence[Mapping[str, Any]], hint: str
) -> list[dict[str, str]]:
    """The system and user messages for one candidate, rendered from agents/prompts/write_skill.md."""
    template = (PROMPTS_DIR / "write_skill.md").read_text(encoding="utf-8")
    user = (
        template.replace("{{TASK_ENGLISH}}", _task_english(spec))
        .replace("{{TASK_JSON}}", json.dumps(spec, indent=2))
        .replace("{{SCENE_JSON}}", json.dumps(scene, indent=2))
        .replace("{{SKILLS}}", _skills_text(skills))
        .replace("{{EXAMPLES}}", _examples_text())
        .replace("{{HINT}}", hint.strip())
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user.rstrip("\n") + "\n"},
    ]


def repair_messages(
    spec: Mapping[str, Any],
    scene: Mapping[str, Any],
    skills: Sequence[Mapping[str, Any]],
    hint: str,
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """The write-skill prompt followed by agents/prompts/repair.md with the failures filled in."""
    base = write_skill_messages(spec, scene, skills, hint)
    body = base[1]["content"].rstrip("\n")
    if body.endswith(CLOSING_LINE):
        body = body[: -len(CLOSING_LINE)].rstrip("\n")
    template = (PROMPTS_DIR / "repair.md").read_text(encoding="utf-8")
    addendum = template.replace("{{FAILURES}}", _failures_text(failures)).replace("{{HINT}}", hint.strip())
    return [base[0], {"role": "user", "content": body + "\n\n" + addendum.rstrip("\n") + "\n"}]


def render_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Markdown rendering of a message list, for the prompt files under runs/<task_id>/prompts/."""
    return "\n\n".join(f"## {m['role']}\n\n{m['content']}" for m in messages) + "\n"


def _task_english(spec: Mapping[str, Any]) -> str:
    description = str(spec.get("description", "")).strip()
    english = str(spec.get("english", "")).strip()
    return f"{description}\n\nSuccess, in words: {english}".strip()


def _skills_text(skills: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for skill in skills:
        params = skill.get("params") or {}
        signature = ", ".join(f"{k}: {v}" for k, v in params.items()) or "no args"
        lines.append(f"- `{skill.get('name')}` ({signature}): {skill.get('description', '')}".rstrip())
    return "\n".join(lines) if lines else "(none saved yet)"


def _examples_text() -> str:
    parts = []
    for name, fallback in STARTER_EXAMPLES.items():
        spec, code = fallback["spec"], fallback["code"]
        path = STARTER_DIR / f"{name}.json"
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                spec, code = saved["spec"], saved["code"]
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("starter skill %s unreadable (%s); using the built-in example", path, exc)
        parts.append(
            f"### `{name}`: {spec.get('description', '')}\n\n"
            f"Spec:\n\n```json\n{json.dumps(spec, indent=2)}\n```\n\n"
            f"Program:\n\n```python\n{code.rstrip()}\n```"
        )
    return "\n\n".join(parts)


def _failures_text(failures: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for n, raw in enumerate(failures, 1):
        failure = _normalize_failure(raw)
        label = f"Failed program {n}"
        if failure["index"] is not None:
            label += f" (candidate {failure['index']})"
        if failure["hint"]:
            label += f", strategy: {failure['hint']}"
        parts.append(
            f"### {label}\n\n```python\n{failure['code'].rstrip()}\n```\n\n"
            f"Score line: `{failure['score']}`\n\nFailure: {failure['fail_reason']}"
        )
    return "\n\n".join(parts) if parts else "(no failure details were recorded)"


def _normalize_failure(raw: Mapping[str, Any]) -> dict[str, Any]:
    score = raw.get("score")
    if isinstance(score, Mapping):
        score_text = json.dumps(score, sort_keys=True)
        fail_reason = raw.get("fail_reason") or score.get("fail_reason") or score.get("error")
    else:
        score_text = str(score) if score is not None else "(no score)"
        fail_reason = raw.get("fail_reason")
    index = raw.get("index", raw.get("candidate"))
    return {
        "code": str(raw.get("code") or raw.get("source") or "(no source)"),
        "score": score_text,
        "fail_reason": str(fail_reason or "unknown"),
        "index": index if isinstance(index, int) else None,
        "hint": str(raw.get("hint") or ""),
    }


# --- generation --------------------------------------------------------------------------------------


def generate(
    spec: Mapping[str, Any],
    scene: Mapping[str, Any],
    skills: Sequence[Mapping[str, Any]],
    *,
    task_dir: str | Path,
    n: int = 6,
    timeout: float = 45,
    demo_mode: bool = False,
    on_rejected: Callable[[Candidate], None] | None = None,
    task_id: str | None = None,
) -> list[Candidate]:
    """Write `n` candidates in parallel (see the module docstring). Returns them in index order."""
    task_dir = Path(task_dir)
    jobs = []
    for i in range(1, int(n) + 1):
        hint = HINTS[(i - 1) % len(HINTS)]
        jobs.append(_Job(i, hint, write_skill_messages(spec, scene, skills, hint), "write_skill"))
    return _run_jobs(jobs, str(spec["name"]), task_dir, task_dir, timeout, demo_mode, on_rejected, task_id)


def repair(
    spec: Mapping[str, Any],
    scene: Mapping[str, Any],
    skills: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    task_dir: str | Path,
    timeout: float,
    n: int = 6,
    on_rejected: Callable[[Candidate], None] | None = None,
    task_id: str | None = None,
) -> list[Candidate]:
    """The repair round: six new candidates written against the three best failures (C7, normal mode)."""
    task_dir = Path(task_dir)
    best = list(failures)[:3]
    jobs = []
    for i in range(1, int(n) + 1):
        hint = HINTS[(i - 1) % len(HINTS)]
        jobs.append(_Job(i, hint, repair_messages(spec, scene, skills, hint, best), "repair"))
    out_dir = task_dir / "repair"
    return _run_jobs(jobs, str(spec["name"]), task_dir, out_dir, timeout, False, on_rejected, task_id)


def _run_jobs(
    jobs: list[_Job],
    slug: str,
    task_dir: Path,
    out_dir: Path,
    timeout: float,
    demo_mode: bool,
    on_rejected: Callable[[Candidate], None] | None,
    task_id: str | None,
) -> list[Candidate]:
    if task_id:
        llm.run_dir(task_id)  # ValueError for an id that is not a plain identifier, before any file exists
    prompts_dir, replies_dir = task_dir / "prompts", task_dir / "replies"
    out_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        _write(prompts_dir / f"{job.purpose}_{job.index}.md", render_messages(job.messages))
    task_id = task_id or _task_id_for(task_dir)
    deadline = time.monotonic() + float(timeout)
    cancelled = threading.Event()
    settings = {
        "slug": slug,
        "out_dir": out_dir,
        "prompts_dir": prompts_dir,
        "replies_dir": replies_dir,
        "deadline": deadline,
        "cancelled": cancelled,
        "demo_mode": demo_mode,
        "on_rejected": on_rejected,
        "task_id": task_id,
    }
    results: list[Candidate | None] = [None] * len(jobs)
    pool = ThreadPoolExecutor(max_workers=max(1, len(jobs)), thread_name_prefix="candidates")
    try:
        futures = [pool.submit(_generate_one, job, settings) for job in jobs]
        for k, future in enumerate(futures):
            remaining = deadline + GRACE_S - time.monotonic()
            try:
                results[k] = future.result(timeout=max(0.0, remaining))
            except FutureTimeout:
                results[k] = None
    finally:
        cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)
    out: list[Candidate] = []
    for k, job in enumerate(jobs):
        candidate = results[k]
        if candidate is None:
            problem = f"generation timed out after {float(timeout):.0f} s"
            candidate = Candidate(job.index, job.hint, "", [problem], 1, out_dir / f"cand_{job.index}.py")
            _save(candidate, out_dir)
            logger.warning("candidate %d (%s): %s", job.index, job.purpose, problem)
        out.append(candidate)
    return out


def _generate_one(job: _Job, settings: dict[str, Any]) -> Candidate:
    key = f"{settings['slug']}:{job.index}"
    candidate, reply = _attempt(job, job.messages, key, 1, settings)
    cancelled: threading.Event = settings["cancelled"]
    if candidate.problems and not cancelled.is_set():
        _notify(settings["on_rejected"], candidate)
        remaining = settings["deadline"] - time.monotonic()
        if not settings["demo_mode"] and remaining >= MIN_RETRY_S:
            retry_messages = [
                *job.messages,
                {"role": "assistant", "content": reply},
                {"role": "user", "content": _rejection_prompt(candidate.problems)},
            ]
            retry_prompt = settings["prompts_dir"] / f"{job.purpose}_{job.index}_retry.md"
            _write(retry_prompt, render_messages(retry_messages))
            candidate, _reply = _attempt(job, retry_messages, f"{key}:retry", 2, settings)
    return candidate


def _attempt(
    job: _Job, messages: list[dict[str, str]], key: str, attempt: int, settings: dict[str, Any]
) -> tuple[Candidate, str]:
    out_dir: Path = settings["out_dir"]
    remaining = settings["deadline"] - time.monotonic()
    reply = ""
    if remaining <= 0:
        source, problems = "", ["generation timed out before the model was asked"]
    else:
        try:
            reply = str(
                chat(
                    messages,
                    codegen_model(),
                    timeout=remaining,
                    purpose=job.purpose,
                    key=key,
                    task_id=settings["task_id"],
                )
            )
        except Exception as exc:
            source, problems = "", [f"generation failed: {type(exc).__name__}: {exc}"]
        else:
            source = extract_code(reply)
            problems = check_source(source)
    path = out_dir / f"cand_{job.index}.py"
    candidate = Candidate(job.index, job.hint, source, list(problems), attempt, path)
    cancelled: threading.Event = settings["cancelled"]
    if not cancelled.is_set():
        suffix = "" if attempt == 1 else "_retry"
        if reply:
            _write(settings["replies_dir"] / f"{job.purpose}_{job.index}{suffix}.md", reply)
        _save(candidate, out_dir)
        if problems:
            logger.info("candidate %d attempt %d rejected: %s", job.index, attempt, problems[0])
    return candidate, reply


def _rejection_prompt(problems: Sequence[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "The static check rejected your program:\n"
        f"{listed}\n\n"
        "Fix every problem and return the whole corrected program again as one Python code block and "
        "nothing else."
    )


def _notify(callback: Callable[[Candidate], None] | None, candidate: Candidate) -> None:
    if callback is None:
        return
    try:
        callback(candidate)
    except Exception as exc:  # a dashboard callback must not kill a worker
        logger.warning("on_rejected callback failed: %s: %s", type(exc).__name__, exc)


def _save(candidate: Candidate, out_dir: Path) -> None:
    _write(out_dir / f"cand_{candidate.index}.py", candidate.source)
    _write(out_dir / f"cand_{candidate.index}.problems.json", json.dumps(candidate.problems, indent=2) + "\n")


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write %s: %s", path, exc)


def _task_id_for(task_dir: Path) -> str | None:
    """`task_dir.name` when `task_dir` is `runs/<task_id>` (so llm.jsonl lands beside the candidates)."""
    try:
        if task_dir.resolve().parent != llm.RUNS.resolve():
            return None
        llm.run_dir(task_dir.name)
    except (OSError, ValueError):
        return None
    return task_dir.name


# --- helpers for the orchestrator ---------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[str]:
    """Every closed fenced code block in `text` (with or without a language tag), fence markers removed."""
    return _scan_fences(_clean_reply(text))[0]


def extract_code(text: str) -> str:
    """The program in a model reply: the fenced block holding `def run`/`def reward` (closed or not), else
    the first closed block, else an unclosed fence's remainder, else the whole text."""
    text = _clean_reply(text)
    blocks, unclosed = _scan_fences(text)
    for block in blocks:
        if _ENTRY_RE.search(block):
            return block
    if unclosed is not None and _ENTRY_RE.search(unclosed):
        return unclosed
    if blocks:
        return blocks[0]
    if unclosed is not None:
        return unclosed
    return text.strip("\n") + "\n"


def _clean_reply(text: Any) -> str:
    """Model text as `str` with Unix line endings and no byte-order mark."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return text[1:] if text.startswith("﻿") else text


def _scan_fences(text: str) -> tuple[list[str], str | None]:
    """(closed blocks, the remainder of an unclosed fence or None), walking the lines once.

    A fence is three or more backticks or tildes at the start of a line (an info string may follow); the
    block ends at the next line that starts with the same run. A closing fence glued to the last code line
    of an unclosed block is dropped."""
    lines = text.split("\n")
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        opening = _OPEN_FENCE_RE.match(lines[i])
        if opening is None:
            i += 1
            continue
        fence = opening.group(1)
        j = i + 1
        while j < len(lines) and not lines[j].lstrip(" \t").startswith(fence):
            j += 1
        body = lines[i + 1 : j]
        if j < len(lines):
            blocks.append(_block_text(body))
            i = j + 1
            continue
        if body and body[-1].rstrip().endswith(fence):
            body[-1] = body[-1].rstrip()[: -len(fence)]
        return blocks, _block_text(body)
    return blocks, None


def _block_text(body: Sequence[str]) -> str:
    return "\n".join(body).strip("\n") + "\n"


def routine_candidate(steps: Sequence[Mapping[str, Any]]) -> str:
    """A candidate that runs saved skills in order through `robot.run_skill` (C10).

    Raises ValueError for a step without a plain skill name or JSON args, or when the result does not pass
    `check_source` (it always should; the check is a guard).
    """
    cleaned = []
    for step in steps:
        if not isinstance(step, Mapping):
            raise ValueError(f"a routine step is an object with skill and args, got {step!r}")
        skill = step.get("skill")
        if not isinstance(skill, str) or not SKILL_NAME_RE.match(skill):
            raise ValueError(f"invalid skill name in routine step: {skill!r}")
        args = step.get("args", {})
        if not isinstance(args, Mapping):
            raise ValueError(f"routine step args must be an object, got {args!r}")
        try:
            args = json.loads(json.dumps(dict(args), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"routine step args are not plain JSON: {exc}") from None
        cleaned.append({"skill": skill, "args": args})
    if not cleaned:
        raise ValueError("a routine needs at least one step")
    literal = repr(cleaned)
    if ast.literal_eval(literal) != cleaned:
        raise ValueError("routine steps do not round-trip as a Python literal")
    source = (
        '"""Routine: run the saved skills below in order; stop at the first one that fails."""\n'
        "\n"
        f"STEPS = {literal}\n"
        "\n"
        "\n"
        "def run(robot, args):\n"
        "    for step in STEPS:\n"
        '        if not robot.run_skill(step["skill"], step["args"]):\n'
        "            return False\n"
        "    return True\n"
    )
    problems = check_source(source)
    if problems:
        raise ValueError("routine candidate rejected: " + "; ".join(problems))
    return source


# --- CLI ---------------------------------------------------------------------------------------------


def _skill_summaries() -> list[dict[str, Any]]:
    """`{"name", "description", "params"}` for every skill under skills/ and skills/_starter/."""
    out: dict[str, dict[str, Any]] = {}
    for folder in (SKILLS_DIR, STARTER_DIR):
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                skill = json.loads(path.read_text(encoding="utf-8"))
                spec = skill.get("spec") or {}
                name = skill.get("name") or spec.get("name") or path.stem
                summary = {
                    "name": name,
                    "description": spec.get("description", ""),
                    "params": spec.get("params") or {},
                }
                out.setdefault(name, summary)
            except (OSError, ValueError, AttributeError) as exc:
                logger.warning("skipping unreadable skill %s: %s", path, exc)
    return list(out.values())


def _utf8_stdout() -> None:
    """The hints contain a π; a cp1252 console would otherwise crash the CLI on printing them."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agents.candidates --eval "<text>"`: route, generate and count the accepted candidates."""
    parser = argparse.ArgumentParser(prog="python -m agents.candidates", description="candidate generation")
    parser.add_argument("--eval", metavar="TEXT", help="route this request and generate its candidates")
    parser.add_argument("--task-dir", default=None, help="where to write (default runs/eval-<spec name>)")
    parser.add_argument("--demo", action="store_true", help="DEMO_MODE budgets: no regeneration, 30 s")
    parser.add_argument("--timeout", type=float, default=None, help="generation budget in seconds")
    args = parser.parse_args(argv)
    if not args.eval:
        parser.print_help()
        return 2
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    _utf8_stdout()
    from agents.router import route

    demo_mode = args.demo or os.environ.get("DEMO_MODE") == "1"
    timeout = args.timeout if args.timeout is not None else (30.0 if demo_mode else 45.0)
    skills = _skill_summaries()
    scene = json.loads(FAKE_SCENE_PATH.read_text(encoding="utf-8"))
    result = route(args.eval, skills, scene)
    if result.get("action") != "learn":
        print(f"not a learn request: {json.dumps(result)}")
        return 1
    spec = result["spec"]
    task_dir = Path(args.task_dir) if args.task_dir else llm.RUNS / f"eval-{spec['name']}"
    started = time.monotonic()
    candidates = generate(spec, scene, skills, task_dir=task_dir, timeout=timeout, demo_mode=demo_mode)
    elapsed = time.monotonic() - started
    first_try = sum(1 for c in candidates if c.attempt == 1 and c.accepted)
    final = sum(1 for c in candidates if c.accepted)
    for c in candidates:
        state = "ok" if c.accepted else c.problems[0]
        lines = len(c.source.splitlines())
        print(f"cand_{c.index} (attempt {c.attempt}, {lines} lines): {state}\n    hint: {c.hint}")
    print(
        f"{first_try} of {len(candidates)} passed check_source on the first try; "
        f"{final} of {len(candidates)} after regeneration; {elapsed:.1f} s; files in {task_dir}"
    )
    return 0 if first_try >= min(5, len(candidates)) else 1


if __name__ == "__main__":
    sys.exit(main())
