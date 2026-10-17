"""agents.candidates: generation in fake mode, the prompt contents, extract_code, routines, fixtures (C3)."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from agents import candidates, llm
from agents.candidates import (
    HINTS,
    Candidate,
    check_source,
    extract_code,
    extract_code_blocks,
    generate,
    repair,
    routine_candidate,
)
from sim import restricted

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
LLM_FIXTURES = FIXTURES / "llm"
SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
SPEC = json.loads((LLM_FIXTURES / "router" / "learn_juice_box_to_tray.json").read_text(encoding="utf-8"))[
    "spec"
]
CLEAR_TABLE = (FIXTURES / "candidate_clear_table.py").read_text(encoding="utf-8")
SKILLS = [
    {
        "name": "hand_over",
        "description": "Pick up an object and hand it to the user.",
        "params": {"object": "object"},
    },
    {
        "name": "place_in_tray",
        "description": "Put the named object into the tray.",
        "params": {"object": "object"},
    },
]
BAD_CANDIDATE = "import os\n\ndef run(robot, args):\n    return True\n"
CANDIDATE_FILES = sorted(LLM_FIXTURES.glob("candidates/*/cand_*.py")) + sorted(
    LLM_FIXTURES.glob("repair/*/cand_*.py")
)
REWARD_FILES = sorted(LLM_FIXTURES.glob("rewards/variant_*.py"))


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


def _bad_fixtures(tmp_path, monkeypatch, slug="juice_box_to_tray"):
    fixtures = tmp_path / "fx"
    folder = fixtures / "candidates" / slug
    folder.mkdir(parents=True)
    (folder / "cand_1.py").write_text(BAD_CANDIDATE, encoding="utf-8")
    monkeypatch.setattr(llm, "FIXTURES", fixtures)
    return fixtures


class StubRobot:
    def __init__(self, results=None):
        self.calls: list[tuple[str, dict]] = []
        self.results = list(results or [])

    def run_skill(self, name, args):
        self.calls.append((name, dict(args)))
        return self.results.pop(0) if self.results else True


# --- HINTS and the prompt ----------------------------------------------------------------------------


def test_hints_are_verbatim_from_the_task_doc():
    text = (ROOT / "docs" / "tasks" / "C-agents.md").read_text(encoding="utf-8")
    section = text.split("## C3")[1].split("## C4")[0]
    items = re.findall(r"^\d+\. (.+)$", section, re.MULTILINE)
    assert tuple(items) == HINTS
    assert len(HINTS) == 6


def test_write_skill_prompt_holds_the_robot_api_and_rules_verbatim():
    doc = (ROOT / "docs" / "interfaces.md").read_text(encoding="utf-8")
    template = (candidates.PROMPTS_DIR / "write_skill.md").read_text(encoding="utf-8")
    start = doc.index("```python\nclass Robot:")
    end = doc.index("```", start + 10)
    api = doc[start + len("```python\n") : end]
    assert api in template
    notes = doc[end + 3 :].split("\n\n")[1]
    note_lines = [line for line in notes.splitlines() if line.startswith("- ")]
    assert len(note_lines) == 5
    for line in note_lines:
        assert line in template, line
    rules = doc.split("It rejects a file that:\n\n")[1].split("\n\n")[0]
    rule_lines = [line for line in rules.splitlines() if line.startswith("- ")]
    assert len(rule_lines) == 7
    for line in rule_lines:
        assert line in template, line
    for trap in (
        "__main__",
        "`_`",
        "from typing import",
        "args: object",
        "BaseException",
        "200 lines",
        "finally",
    ):
        assert trap in template, trap
    assert template.rstrip().endswith(candidates.CLOSING_LINE)


def test_write_skill_messages_fill_every_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(candidates, "STARTER_DIR", tmp_path / "no_starter")  # the built-in examples
    messages = candidates.write_skill_messages(SPEC, SCENE, SKILLS, HINTS[2])
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "{{" not in user
    assert json.dumps(SPEC, indent=2) in user
    assert json.dumps(SCENE, indent=2) in user
    assert SPEC["english"] in user and SPEC["description"] in user
    assert HINTS[2] in user
    assert "`hand_over`" in user and "`place_in_tray`" in user
    assert "def run(robot, args):" in user  # the worked examples
    assert "robot.carry_to_hand()" in user and 'robot.place_in_zone("tray", transit_z=transit)' in user
    assert user.rstrip().endswith(candidates.CLOSING_LINE)
    for example in candidates.STARTER_EXAMPLES.values():
        assert check_source(example["code"]) == []


def _example_programs(user: str) -> list[str]:
    section = user.split("## Worked examples")[1].split("## Strategy")[0]
    return re.findall(r"Program:\n\n```python\n(.*?)\n```", section, re.DOTALL)


def test_prompt_is_complete_whatever_the_starter_dir_holds():
    """Regression: the prompt test must not depend on whether skills/_starter/ is populated (it is
    written by Lane A at its own time); whatever is there is what the model sees, and it must be valid."""
    user = candidates.write_skill_messages(SPEC, SCENE, SKILLS, HINTS[0])[1]["content"]
    assert "{{" not in user and user.rstrip().endswith(candidates.CLOSING_LINE)
    programs = _example_programs(user)
    assert len(programs) == 2
    for program in programs:
        assert "def run(robot, args):" in program
        assert check_source(program + "\n") == []
    for name, example in candidates.STARTER_EXAMPLES.items():
        assert f"### `{name}`" in user
        path = candidates.STARTER_DIR / f"{name}.json"
        expected = json.loads(path.read_text(encoding="utf-8"))["code"] if path.is_file() else example["code"]
        assert expected.rstrip() in programs, name


def test_examples_come_from_starter_files_when_present(monkeypatch, tmp_path):
    starter = tmp_path / "_starter"
    starter.mkdir()
    skill = {
        "name": "hand_over",
        "spec": candidates.STARTER_EXAMPLES["hand_over"]["spec"],
        "code": "def run(robot, args):\n    return robot.carry_to_hand()\n",
    }
    (starter / "hand_over.json").write_text(json.dumps(skill), encoding="utf-8")
    monkeypatch.setattr(candidates, "STARTER_DIR", starter)
    user = candidates.write_skill_messages(SPEC, SCENE, [], HINTS[0])[1]["content"]
    assert "def run(robot, args):\n    return robot.carry_to_hand()" in user
    assert candidates.STARTER_EXAMPLES["place_in_tray"]["code"].rstrip() in user  # the missing one falls back


# --- generate ----------------------------------------------------------------------------------------


def test_generate_in_fake_mode_returns_six_candidates_and_writes_files(tmp_path):
    task_dir = tmp_path / "t"
    result = generate(SPEC, SCENE, SKILLS, task_dir=task_dir)
    assert [c.index for c in result] == [1, 2, 3, 4, 5, 6]
    assert [c.hint for c in result] == list(HINTS)
    assert all(isinstance(c, Candidate) and c.accepted and c.attempt == 1 for c in result)
    for c in result:
        fixture = (LLM_FIXTURES / "candidates" / "juice_box_to_tray" / f"cand_{c.index}.py").read_text(
            encoding="utf-8"
        )
        assert c.source == fixture
        assert c.path == task_dir / f"cand_{c.index}.py"
        assert (task_dir / f"cand_{c.index}.py").read_text(encoding="utf-8") == fixture
        assert json.loads((task_dir / f"cand_{c.index}.problems.json").read_text(encoding="utf-8")) == []
        prompt = (task_dir / "prompts" / f"write_skill_{c.index}.md").read_text(encoding="utf-8")
        assert c.hint in prompt and "## system" in prompt and "## user" in prompt
        assert prompt.rstrip().endswith(candidates.CLOSING_LINE)
        reply = (task_dir / "replies" / f"write_skill_{c.index}.md").read_text(encoding="utf-8")
        assert reply.startswith("```python")
    assert not (task_dir / "prompts" / "write_skill_1_retry.md").exists()


def test_generate_logs_llm_calls_when_task_dir_is_a_run_dir(fake_env):
    task_dir = fake_env / "t-0042"
    generate(SPEC, SCENE, SKILLS, task_dir=task_dir, n=3)
    lines = (task_dir / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["purpose"] == "write_skill" for line in lines)


def test_rejected_fixture_triggers_one_regeneration(tmp_path, monkeypatch):
    _bad_fixtures(tmp_path, monkeypatch)
    rejected: list[Candidate] = []
    task_dir = tmp_path / "t"
    result = generate(SPEC, SCENE, SKILLS, task_dir=task_dir, n=2, on_rejected=rejected.append)
    first, second = result
    assert first.index == 1 and first.attempt == 2 and first.accepted
    assert first.source == CLEAR_TABLE  # the retry key has no fixture, so the fake falls back
    assert second.index == 2 and second.attempt == 1 and second.accepted
    assert [c.index for c in rejected] == [1]
    assert rejected[0].attempt == 1 and rejected[0].source == BAD_CANDIDATE
    assert any("import" in p for p in rejected[0].problems)
    retry_prompt = (task_dir / "prompts" / "write_skill_1_retry.md").read_text(encoding="utf-8")
    assert "rejected your program" in retry_prompt and rejected[0].problems[0] in retry_prompt
    assert BAD_CANDIDATE.strip() in retry_prompt  # the rejected program travels back as the assistant turn
    assert (task_dir / "replies" / "write_skill_1_retry.md").is_file()
    assert (task_dir / "cand_1.py").read_text(encoding="utf-8") == CLEAR_TABLE
    assert json.loads((task_dir / "cand_1.problems.json").read_text(encoding="utf-8")) == []


def test_demo_mode_skips_the_regeneration(tmp_path, monkeypatch):
    _bad_fixtures(tmp_path, monkeypatch)
    rejected: list[Candidate] = []
    task_dir = tmp_path / "t"
    result = generate(
        SPEC, SCENE, SKILLS, task_dir=task_dir, n=2, demo_mode=True, on_rejected=rejected.append
    )
    assert result[0].attempt == 1 and not result[0].accepted and result[0].source == BAD_CANDIDATE
    assert result[1].accepted
    assert [c.index for c in rejected] == [1]
    assert not (task_dir / "prompts" / "write_skill_1_retry.md").exists()
    assert json.loads((task_dir / "cand_1.problems.json").read_text(encoding="utf-8")) == result[0].problems


def test_model_failure_becomes_a_rejected_candidate(tmp_path, monkeypatch):
    def down(*args, **kwargs):
        raise llm.ModelCallFailed("gave up after 3 retries: 503")

    monkeypatch.setattr(candidates, "chat", down)
    result = generate(SPEC, SCENE, SKILLS, task_dir=tmp_path / "t", n=2, demo_mode=True)
    assert all(not c.accepted and c.source == "" for c in result)
    assert all(c.problems[0].startswith("generation failed: ModelCallFailed") for c in result)


def test_generate_gives_up_at_the_budget(tmp_path, monkeypatch):
    release = threading.Event()

    def slow(*args, **kwargs):
        release.wait(5.0)
        return "```python\ndef run(robot, args):\n    return True\n```"

    monkeypatch.setattr(candidates, "chat", slow)
    monkeypatch.setattr(candidates, "GRACE_S", 0.1)
    started = time.monotonic()
    try:
        result = generate(SPEC, SCENE, SKILLS, task_dir=tmp_path / "t", n=2, timeout=0.3)
    finally:
        release.set()
    assert time.monotonic() - started < 3.0
    assert all(not c.accepted and "timed out" in c.problems[0] for c in result)
    assert (
        json.loads((tmp_path / "t" / "cand_1.problems.json").read_text(encoding="utf-8"))
        == result[0].problems
    )


def test_on_rejected_errors_do_not_break_generation(tmp_path, monkeypatch):
    _bad_fixtures(tmp_path, monkeypatch)

    def explode(candidate):
        raise RuntimeError("dashboard down")

    result = generate(SPEC, SCENE, SKILLS, task_dir=tmp_path / "t", n=1, on_rejected=explode)
    assert result[0].accepted and result[0].attempt == 2


@pytest.mark.parametrize("task_id", ["../escape", "C:\\x", "a/b", "t 1"])
def test_bad_task_id_is_rejected_before_any_file_is_written(tmp_path, task_id):
    task_dir = tmp_path / "t"
    with pytest.raises(ValueError, match="plain identifier"):
        generate(SPEC, SCENE, SKILLS, task_dir=task_dir, n=1, task_id=task_id)
    with pytest.raises(ValueError, match="plain identifier"):
        repair(SPEC, SCENE, SKILLS, [], task_dir=task_dir, timeout=5, n=1, task_id=task_id)
    assert not task_dir.exists()


# --- repair ------------------------------------------------------------------------------------------


def test_repair_uses_the_repair_prompt_and_fixtures(tmp_path):
    cand_1 = (LLM_FIXTURES / "candidates" / "juice_box_to_tray" / "cand_1.py").read_text(encoding="utf-8")
    score = {
        "candidate": 1,
        "passed": False,
        "successes": 0,
        "trials": 5,
        "fail_reason": "trial 1: gripper hit cereal_box",
    }
    failures = [
        {
            "code": cand_1,
            "score": score,
            "fail_reason": "trial 1: gripper hit cereal_box",
            "index": 1,
            "hint": HINTS[0],
        },
        {
            "source": "def run(robot, args):\n    return False\n",
            "score": {"fail_reason": "trial 1: spec not satisfied"},
        },
        {
            "code": "def run(robot, args):\n    return True\n",
            "score": "{}",
            "fail_reason": "trial 2: timeout",
        },
        {"code": "ignored fourth failure", "fail_reason": "ignored"},
    ]
    task_dir = tmp_path / "t"
    result = repair(SPEC, SCENE, SKILLS, failures, task_dir=task_dir, timeout=10)
    assert [c.index for c in result] == [1, 2, 3, 4, 5, 6]
    assert all(c.accepted for c in result)
    for c in result:
        fixture = (LLM_FIXTURES / "repair" / "juice_box_to_tray" / f"cand_{c.index}.py").read_text(
            encoding="utf-8"
        )
        assert c.source == fixture
        assert "transit_z=" in c.source
        assert c.path == task_dir / "repair" / f"cand_{c.index}.py"
        assert (task_dir / "repair" / f"cand_{c.index}.py").read_text(encoding="utf-8") == fixture
    prompt = (task_dir / "prompts" / "repair_1.md").read_text(encoding="utf-8")
    assert "Fix these failures" in prompt
    assert cand_1.strip() in prompt and "gripper hit cereal_box" in prompt and "spec not satisfied" in prompt
    assert "ignored fourth failure" not in prompt
    assert prompt.count(candidates.CLOSING_LINE) == 1 and prompt.rstrip().endswith(candidates.CLOSING_LINE)
    assert "class Robot:" in prompt  # the API travels with the repair prompt


# --- extract_code ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "```python\ndef run(robot, args):\n    return True\n```",
            "def run(robot, args):\n    return True\n",
        ),
        ("```\ndef run(robot, args):\n    return True\n```\n", "def run(robot, args):\n    return True\n"),
        (
            "Here you go:\n```py\nimport math\n\ndef run(robot, args):\n    return True\n```\nEnjoy!",
            "import math\n\ndef run(robot, args):\n    return True\n",
        ),
        ("def run(robot, args):\n    return True", "def run(robot, args):\n    return True\n"),
        (
            "```python\r\ndef run(robot, args):\r\n    return True\r\n```",
            "def run(robot, args):\n    return True\n",
        ),
        ("```python\ndef run(robot, args):\n    return True\n", "def run(robot, args):\n    return True\n"),
        (
            'Use this:\n```json\n{"a": 1}\n```\nthen\n```python\ndef run(robot, args):\n    return True\n```',
            "def run(robot, args):\n    return True\n",
        ),
        ("```text\nnot code\n```\n```\nalso not\n```", "not code\n"),
        # lone-CR line endings (old Mac style) are normalized like CRLF
        (
            "```python\rdef run(robot, args):\r    return True\r```",
            "def run(robot, args):\n    return True\n",
        ),
        # tilde fences are Markdown too
        (
            "~~~python\ndef run(robot, args):\n    return True\n~~~\n",
            "def run(robot, args):\n    return True\n",
        ),
        # a triple backtick inside a string literal does not close the block
        (
            "```python\nimport math\n\nS = '```'\n\ndef run(robot, args):\n    return S\n```",
            "import math\n\nS = '```'\n\ndef run(robot, args):\n    return S\n",
        ),
        # four-backtick fences, and text after the closing fence on the same line
        (
            "````python\ndef run(robot, args):\n    return True\n````",
            "def run(robot, args):\n    return True\n",
        ),
        (
            "```python\ndef run(robot, args):\n    return True\n```  done",
            "def run(robot, args):\n    return True\n",
        ),
        # an indented fence
        (
            "  ```python\n  def run(robot, args):\n      return True\n  ```",
            "  def run(robot, args):\n      return True\n",
        ),
        # a byte-order mark, and bytes instead of str
        (
            "﻿```python\ndef run(robot, args):\n    return True\n```",
            "def run(robot, args):\n    return True\n",
        ),
        (
            b"```python\ndef run(robot, args):\n    return True\n```",
            "def run(robot, args):\n    return True\n",
        ),
        # the closing fence glued to the last line of code
        ("```python\ndef run(robot, args):\n    return True```", "def run(robot, args):\n    return True\n"),
        # the entry lives in an unclosed block after a closed non-code block
        (
            '```json\n{"a": 1}\n```\n```python\ndef run(robot, args):\n    return True\n',
            "def run(robot, args):\n    return True\n",
        ),
        ("```python\n```", "\n"),
    ],
)
def test_extract_code_variants(text, expected):
    assert extract_code(text) == expected


def test_extract_code_blocks():
    text = "a\n```python\nx = 1\n```\nb\n```\ny = 2\n```"
    assert extract_code_blocks(text) == ["x = 1\n", "y = 2\n"]
    assert extract_code_blocks("no fences") == []
    assert extract_code_blocks("```python\nx = 1\n") == []  # unclosed blocks are not blocks
    assert extract_code_blocks("~~~\nx = 1\n~~~\n```py\ny = 2\n```") == ["x = 1\n", "y = 2\n"]
    assert extract_code_blocks("```python\nS = '```'\nx = 1\n```") == ["S = '```'\nx = 1\n"]


# --- routines ----------------------------------------------------------------------------------------


def test_routine_candidate_runs_skills_in_order_and_stops_at_the_first_failure():
    steps = [
        {"skill": "hand_over", "args": {"object": "pill_bottle"}},
        {"skill": "hand_over", "args": {"object": "juice_box"}},
        {"skill": "place_in_tray", "args": {"object": "tea_tin"}},
    ]
    source = routine_candidate(steps)
    assert check_source(source) == []
    assert "robot.run_skill(" in source
    run = restricted.load_entry(source)
    robot = StubRobot()
    assert run(robot, {}) is True
    assert robot.calls == [(s["skill"], s["args"]) for s in steps]
    failing = StubRobot(results=[True, False, True])
    assert restricted.load_entry(source)(failing, {}) is False
    assert failing.calls == [(s["skill"], s["args"]) for s in steps[:2]]


def test_routine_candidate_rejects_bad_steps():
    with pytest.raises(ValueError):
        routine_candidate([])
    with pytest.raises(ValueError):
        routine_candidate([{"skill": "Not A Name", "args": {}}])
    with pytest.raises(ValueError):
        routine_candidate([{"skill": "hand_over", "args": "pill_bottle"}])
    with pytest.raises(ValueError):
        routine_candidate([{"skill": "hand_over", "args": {"object": float("inf")}}])
    source = routine_candidate([{"skill": "hand_over", "args": {"object": 'it\'s "quoted"', "n": 2}}])
    assert check_source(source) == []
    robot = StubRobot()
    restricted.load_entry(source)(robot, {})
    assert robot.calls == [("hand_over", {"object": 'it\'s "quoted"', "n": 2})]


# --- every fixture passes check_source --------------------------------------------------------------


def test_fixture_sets_are_complete():
    for slug in ("juice_box_to_tray", "clear_table"):
        names = sorted(p.name for p in (LLM_FIXTURES / "candidates" / slug).glob("cand_*.py"))
        assert names == [f"cand_{i}.py" for i in range(1, 7)], slug
    names = sorted(p.name for p in (LLM_FIXTURES / "repair" / "juice_box_to_tray").glob("cand_*.py"))
    assert names == [f"cand_{i}.py" for i in range(1, 7)]
    assert [p.name for p in REWARD_FILES] == [f"variant_{k}.py" for k in range(1, 5)]
    juice = LLM_FIXTURES / "candidates" / "juice_box_to_tray"
    for index in (3, 4, 6):
        assert "transit_z=" in (juice / f"cand_{index}.py").read_text(encoding="utf-8")
    assert "transit_z" not in (juice / "cand_1.py").read_text(encoding="utf-8")
    for path in (LLM_FIXTURES / "repair" / "juice_box_to_tray").glob("cand_*.py"):
        assert "transit_z=" in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize(
    "path", CANDIDATE_FILES, ids=[f"{p.parent.parent.name}/{p.parent.name}/{p.name}" for p in CANDIDATE_FILES]
)
def test_fixture_candidate_passes_check_source(path):
    assert check_source(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", REWARD_FILES, ids=[p.name for p in REWARD_FILES])
def test_fixture_reward_passes_check_source_and_computes(path):
    source = path.read_text(encoding="utf-8")
    assert check_source(source, entry="reward") == []
    assert "numpy" not in source
    reward = restricted.load_entry(source, entry="reward")
    obs = [0.0] * 16
    value = reward(
        obs, [0.1, -0.2, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0], {"dist": 0.12, "success": False, "step": 7}
    )
    assert isinstance(value, float)
    better = reward(obs, [0.0] * 4, [0.0] * 4, {"dist": 0.01, "success": True, "step": 7, "prev_dist": 0.02})
    assert better > value


# --- the CLI -----------------------------------------------------------------------------------------


def test_eval_cli_in_fake_mode(tmp_path, capsys, no_dotenv):
    code = candidates.main(["--eval", "put the juice box in the tray", "--task-dir", str(tmp_path / "eval")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "6 of 6 passed check_source on the first try" in out
    assert (tmp_path / "eval" / "cand_6.py").is_file()
    assert len(no_dotenv) == 1  # the CLI loads .env exactly once (and the test never let it read one)


def test_eval_cli_refuses_a_non_learn_request(tmp_path, capsys, no_dotenv):
    code = candidates.main(["--eval", "Fold my laundry", "--task-dir", str(tmp_path / "eval")])
    assert code == 1
    assert "not a learn request" in capsys.readouterr().out
    assert len(no_dotenv) == 1
