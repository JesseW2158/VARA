"""scripts/verify.py (I1): the step checks, the interpreter and host lookup, the record, and the CLI.

Every step runner here is canned: no pytest, e2e or ruff subprocess runs, except two tiny `sys.executable
-c` commands that exercise the real streaming runner and its timeout."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify

PYTEST_CLEAN = (
    ".................                                                        [100%]\n17 passed in 1.07s\n"
)
PYTEST_SKIPPED = (
    ".sx                                                                      [100%]\n"
    "=========================== short test summary info ===========================\n"
    "SKIPPED [1] tests/test_probe.py:8: probe skip\n"
    "1 passed, 1 skipped, 1 xfailed in 0.72s\n"
)


def e2e_log(last: str = "saved", summary: str = "juice_box_to_tray", code_line: bool = True) -> str:
    stages = [
        ("routing", '"put the juice box in the tray"'),
        ("spec_ready", "juice_box_to_tray: The juice box ends inside the tray."),
        ("writing_code", "candidate 1: Nearest object first, default parameters."),
        ("candidate_scored", "candidate 1 PASS 5/5 [local]"),
        ("selected", "candidate 3 (Set `transit_z` to the tallest obstacle plus 0.08 m)"),
        ("validating", "4 segments"),
        ("executing", "4 segments"),
        ("verifying", "The juice box ends inside the tray."),
        (last, summary),
    ]
    lines = [
        "skills library: a copy in C:/x/runs/e2e_fake/20261017-000000/skills (--in-place uses skills/)",
        "task t-0001: 'put the juice box in the tray' (mode fake, runs in C:/x/runs/t-0001)",
        *(f"{stage:<19} {text}" for stage, text in stages),
    ]
    if code_line:
        lines.append("saved skill: C:/x/runs/e2e_fake/20261017-000000/skills/juice_box_to_tray.json")
    return "\n".join(lines) + "\n"


def canned(outputs: dict[str, tuple[int | None, str]]):
    """A step runner answering from `outputs` (by step name) and recording every call."""
    calls: list[tuple[str, list[str], Path, dict[str, str], float, Path]] = []

    def runner(command, cwd, env, timeout_s, log_path):
        if any("e2e_fake" in part for part in command):
            name = "e2e"
        elif "pytest" in command:
            name = "pytest"
        else:
            name = "ruff"
        calls.append((name, list(command), Path(cwd), dict(env), timeout_s, Path(log_path)))
        if name not in outputs:
            raise AssertionError(f"step {name} must not run")
        code, output = outputs[name]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output, encoding="utf-8")
        return code, output

    runner.calls = calls
    return runner


# --- the checks --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output,expected",
    [
        (PYTEST_CLEAN, {"passed": 17}),
        (PYTEST_SKIPPED, {"passed": 1, "skipped": 1, "xfailed": 1}),
        ("==== 428 passed, 2 warnings in 312.45s (0:05:12) ====\n", {"passed": 428, "warnings": 2}),
        ("2 failed, 10 passed, 1 error in 2.00s\n", {"failed": 2, "passed": 10, "error": 1}),
        ("no tests ran in 0.01s\n", {}),
        ("collected 3 items\n...\n", None),
        ("", None),
        # PY_COLORS=1 or FORCE_COLOR colours the line even through a pipe
        ("\x1b[32m\x1b[1m17 passed\x1b[0m\x1b[32m in 1.07s\x1b[0m\n", {"passed": 17}),
        (
            "\x1b[32m3 passed\x1b[0m, \x1b[33m1 skipped\x1b[0m\x1b[33m in 0.50s\x1b[0m\n",
            {"passed": 3, "skipped": 1},
        ),
    ],
)
def test_pytest_counts_parses_the_summary_line(output, expected):
    assert verify.pytest_counts(output) == expected


@pytest.mark.parametrize(
    "exit_code,output,passed,fragment",
    [
        (0, PYTEST_CLEAN, True, "17 passed, nothing skipped"),
        (0, PYTEST_SKIPPED, False, "1 skipped, 1 xfailed"),
        (0, "3 passed, 1 xpassed in 1.0s\n", False, "1 xpassed"),
        (1, "2 failed, 10 passed in 2.00s\n", False, "2 failed"),
        (1, "1 passed, 1 error in 2.00s\n", False, "1 error"),
        (5, "no tests ran in 0.01s\n", False, "no tests passed"),
        (1, "3 passed in 1.0s\n", False, "exit 1"),
        (0, "collected 3 items\n", False, "no pytest summary line"),
        (0, "3 passed, 2 warnings in 1.0s\n", True, "3 passed"),
    ],
)
def test_check_pytest_fails_on_anything_skipped_or_failed(exit_code, output, passed, fragment):
    ok, reason = verify.check_pytest(exit_code, output)
    assert ok is passed and fragment in reason


def test_e2e_stages_reads_the_stage_lines_only():
    stages = verify.e2e_stages(e2e_log())
    assert stages[:2] == ["routing", "spec_ready"] and stages[-1] == "saved"
    assert "task" not in stages and "skills" not in stages
    assert verify.e2e_stages("saved skill: C:/x/juice_box_to_tray.json\n") == []
    assert verify.e2e_stages(f"{'saved':<19} \n") == ["saved"]  # an empty summary still counts


@pytest.mark.parametrize(
    "exit_code,output,passed,fragment",
    [
        (0, e2e_log(), True, "last stage saved"),
        (0, e2e_log("done", "ok", code_line=False), False, "last stage 'done'"),
        (1, e2e_log("queued", "I could not write a success test", code_line=False), False, "'queued'"),
        (1, e2e_log(), False, "exit 1"),
        (
            0,
            "task t-0001: 'x' (mode fake)\ntimed out after 240 s (state busy, last stage None)\n",
            False,
            "no stage lines",
        ),
    ],
)
def test_check_e2e_requires_exit_0_and_a_final_saved_stage(exit_code, output, passed, fragment):
    ok, reason = verify.check_e2e(exit_code, output)
    assert ok is passed and fragment in reason


def test_check_exit():
    assert verify.check_exit(0, "") == (True, "exit 0")
    assert verify.check_exit(1, "x.py:1:1: F401\n") == (False, "exit 1")


# --- interpreter, host, paths ------------------------------------------------------------------------


def test_find_interpreter_prefers_the_venv_next_to_the_root(tmp_path):
    assert verify.find_interpreter(tmp_path) == (sys.executable, False)
    exe = tmp_path / ".venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert verify.find_interpreter(tmp_path) == (str(exe), True)


def test_interpreter_version_reports_the_real_one_and_none_for_a_dud(tmp_path):
    assert verify.interpreter_version(sys.executable) == sys.version.split()[0]
    dud = tmp_path / "python.exe"
    dud.write_bytes(b"")
    assert verify.interpreter_version(str(dud)) is None


def test_host_info_carries_the_identity_fields(tmp_path, monkeypatch):
    answers = {("rev-parse", "--short", "HEAD"): "abc1234", ("status", "--porcelain"): " M arm/driver.py"}
    monkeypatch.setattr(verify, "git_output", lambda root, *args: answers[args])
    host = verify.host_info(tmp_path)
    assert set(host) == {"hostname", "os", "python", "git_head", "git_dirty", "utc"}
    assert host["hostname"] and host["os"] and host["python"] == sys.version.split()[0]
    assert host["git_head"] == "abc1234" and host["git_dirty"] is True
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", host["utc"])
    monkeypatch.setattr(verify, "git_output", lambda root, *args: "" if args[0] == "status" else "abc1234")
    assert verify.host_info(tmp_path)["git_dirty"] is False
    monkeypatch.setattr(verify, "git_output", lambda root, *args: None)
    host = verify.host_info(tmp_path)
    assert host["git_head"] is None and host["git_dirty"] is None


def test_git_output_never_raises(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 128, "", "fatal: not a git repository")

    monkeypatch.setattr(verify.subprocess, "run", failing)
    assert verify.git_output(tmp_path, "rev-parse", "--short", "HEAD") is None

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(verify.subprocess, "run", missing)
    assert verify.git_output(tmp_path, "status", "--porcelain") is None


def test_record_paths_are_never_reused(tmp_path):
    first, logs = verify.new_record_paths(tmp_path, "my host!", "20261017-000000")
    assert first == tmp_path / "my_host-20261017-000000.json" and logs == tmp_path / "my_host-20261017-000000"
    first.write_text("{}", encoding="utf-8")
    second, logs2 = verify.new_record_paths(tmp_path, "my host!", "20261017-000000")
    assert second.name == "my_host-20261017-000000-2.json" and logs2.name == "my_host-20261017-000000-2"
    logs2.mkdir()
    third, _ = verify.new_record_paths(tmp_path, "my host!", "20261017-000000")
    assert third.name == "my_host-20261017-000000-3.json"
    assert verify.new_record_paths(tmp_path, "", "s")[0].name == "host-s.json"


def test_runs_dir_follows_vara_runs_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VARA_RUNS_DIR", raising=False)
    assert verify.runs_dir_for(tmp_path) == tmp_path / "runs" / "verify"
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert verify.runs_dir_for(tmp_path) == tmp_path / "elsewhere" / "verify"


def test_plan_steps_commands_and_selection():
    steps = verify.plan_steps("py", pytest_args="tests/test_x.py -k 'a or b'")
    assert [s["name"] for s in steps] == ["pytest", "e2e", "ruff"]
    assert steps[0]["command"] == [
        "py",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-rs",
        "tests/test_x.py",
        "-k",
        "a or b",
    ]
    assert steps[1]["command"] == ["py", "scripts/e2e_fake.py", "put the juice box in the tray"]
    assert steps[2]["command"] == ["py", "-m", "ruff", "check", "."]
    assert not any(s["skipped"] for s in steps)
    steps = verify.plan_steps("py", skip_e2e=True, only=["e2e", "ruff"])
    assert [(s["skipped"], s["why"]) for s in steps] == [
        (True, "not in --only"),
        (True, "--skip-e2e"),
        (False, None),
    ]
    with pytest.raises(ValueError, match="unknown step"):
        verify.plan_steps("py", only=["lint"])


# --- the record --------------------------------------------------------------------------------------


def test_full_run_writes_a_passing_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "verify-secret-never-recorded")
    runner = canned({"pytest": (0, PYTEST_CLEAN), "e2e": (0, e2e_log()), "ruff": (0, "All checks passed!\n")})
    runs = tmp_path / "runs" / "verify"
    record, path = verify.verify(root=tmp_path, runs_dir=runs, runner=runner, echo=False)
    assert path.parent == runs and path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert record["passed"] is True and record["complete"] is True
    assert [s["name"] for s in record["steps"]] == ["pytest", "e2e", "ruff"]
    assert all(s["passed"] is True and s["skipped"] is False and s["exit_code"] == 0 for s in record["steps"])
    assert record["steps"][0]["reason"] == "17 passed, nothing skipped"
    assert record["steps"][1]["reason"] == "last stage saved"
    assert record["steps"][2]["output_tail"] == "All checks passed!"
    assert record["host"]["hostname"] and record["host"]["python"] == sys.version.split()[0]
    assert record["interpreter"] == {
        "path": sys.executable,
        "is_venv": False,
        "python": sys.version.split()[0],
    }
    assert record["root"] == str(tmp_path) and record["kind"] == "vara.verify"
    assert "verify-secret-never-recorded" not in path.read_text(encoding="utf-8")
    # every step ran with the resolved interpreter, from the root, within the default timeout
    assert [c[0] for c in runner.calls] == ["pytest", "e2e", "ruff"]
    assert all(c[1][0] == sys.executable and c[2] == tmp_path for c in runner.calls)
    assert all(c[4] == verify.DEFAULT_STEP_TIMEOUT_S for c in runner.calls)
    assert all(c[3]["PYTHONIOENCODING"] == "utf-8" for c in runner.calls)
    assert runner.calls[0][1][1:] == ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"]
    # the full logs sit in the folder named like the record
    logs = path.with_suffix("")
    assert record["log_dir"] == str(logs)
    assert [Path(s["log"]) for s in record["steps"]] == [
        logs / "pytest.log",
        logs / "e2e.log",
        logs / "ruff.log",
    ]
    assert (logs / "pytest.log").read_text(encoding="utf-8") == PYTEST_CLEAN
    out = capsys.readouterr().out
    assert "[1/3] pytest: PASS" in out and "verify: PASS: 3 of 3 steps passed, 0 skipped" in out
    assert "WARNING: no .venv" in out  # tmp_path has none; the real repo does


def test_a_skipped_test_fails_the_pytest_step_and_the_run(tmp_path):
    runner = canned({"pytest": (0, PYTEST_SKIPPED), "e2e": (0, e2e_log()), "ruff": (0, "")})
    record, _ = verify.verify(root=tmp_path, runs_dir=tmp_path / "v", runner=runner, echo=False)
    assert record["passed"] is False
    step = record["steps"][0]
    assert step["passed"] is False and step["exit_code"] == 0
    assert step["reason"] == "summary counts 1 skipped, 1 xfailed"
    assert record["steps"][1]["passed"] is True and record["steps"][2]["passed"] is True


def test_an_e2e_run_that_ends_elsewhere_fails(tmp_path):
    runner = canned(
        {"pytest": (0, PYTEST_CLEAN), "e2e": (0, e2e_log("done", "ok", code_line=False)), "ruff": (0, "")}
    )
    record, _ = verify.verify(root=tmp_path, runs_dir=tmp_path / "v", runner=runner, echo=False)
    assert record["passed"] is False and record["steps"][1]["reason"] == "last stage 'done', expected 'saved'"


def test_fast_rerun_is_recorded_as_incomplete(tmp_path):
    runner = canned({"pytest": (0, PYTEST_CLEAN), "ruff": (0, "")})
    record, _ = verify.verify(
        root=tmp_path,
        runs_dir=tmp_path / "v",
        skip_e2e=True,
        pytest_args="tests/test_setup_check.py -x",
        runner=runner,
        echo=False,
    )
    assert record["passed"] is True and record["complete"] is False
    assert record["args"] == {
        "skip_e2e": True,
        "pytest_args": "tests/test_setup_check.py -x",
        "only": None,
        "step_timeout_s": verify.DEFAULT_STEP_TIMEOUT_S,
    }
    e2e = record["steps"][1]
    assert (
        e2e["skipped"] is True and e2e["passed"] is None and e2e["log"] is None and e2e["why"] == "--skip-e2e"
    )
    assert [c[0] for c in runner.calls] == ["pytest", "ruff"]
    assert runner.calls[0][1][-2:] == ["tests/test_setup_check.py", "-x"]


def test_only_runs_the_named_steps(tmp_path, capsys):
    runner = canned({"ruff": (1, "x.py:1:1: F401 `os` imported but unused\nFound 1 error.\n")})
    record, _ = verify.verify(
        root=tmp_path, runs_dir=tmp_path / "v", only=["ruff"], runner=runner, echo=False
    )
    assert [c[0] for c in runner.calls] == ["ruff"]
    assert record["passed"] is False and record["complete"] is False
    assert record["steps"][2]["reason"] == "exit 1" and "Found 1 error." in record["steps"][2]["output_tail"]
    assert "verify: FAIL: 0 of 1 steps passed, 2 skipped; not a full run" in capsys.readouterr().out


def test_nothing_run_is_a_failure(tmp_path):
    record, _ = verify.verify(
        root=tmp_path, runs_dir=tmp_path / "v", only=["e2e"], skip_e2e=True, runner=canned({}), echo=False
    )
    assert record["passed"] is False and all(s["skipped"] for s in record["steps"])


def test_a_step_that_cannot_start_fails_and_the_record_is_still_written(tmp_path, capsys):
    calls = []

    def runner(command, cwd, env, timeout_s, log_path):
        calls.append(list(command))
        if "pytest" in command:
            raise FileNotFoundError(2, "No such file or directory", command[0])
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("All checks passed!\n", encoding="utf-8")
        return 0, "All checks passed!\n"

    record, path = verify.verify(
        root=tmp_path, runs_dir=tmp_path / "v", only=["pytest", "ruff"], runner=runner, echo=False
    )
    assert path.is_file() and json.loads(path.read_text(encoding="utf-8")) == record
    step = record["steps"][0]
    assert step["passed"] is False and step["exit_code"] is None and step["log"] is None
    assert step["reason"].startswith("could not run the step: FileNotFoundError")
    assert record["steps"][2]["passed"] is True  # the next step still ran
    assert len(calls) == 2 and record["passed"] is False
    assert "could not run the step" in capsys.readouterr().out


def test_a_killed_step_fails_with_a_timeout_reason(tmp_path):
    runner = canned({"pytest": (None, "....\n"), "e2e": (0, e2e_log()), "ruff": (0, "")})
    record, _ = verify.verify(
        root=tmp_path, runs_dir=tmp_path / "v", step_timeout=7, runner=runner, echo=False
    )
    step = record["steps"][0]
    assert step["passed"] is False and step["exit_code"] is None
    assert step["reason"] == "timed out after 7 s and was killed"
    assert runner.calls[0][4] == 7


def test_two_runs_in_the_same_second_keep_both_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verify.time,
        "strftime",
        lambda fmt, *args: "20261017-120000" if fmt.startswith("%Y%m%d") else "2026-10-17T12:00:00Z",
    )
    runner = canned({"ruff": (0, "")})
    _, first = verify.verify(root=tmp_path, runs_dir=tmp_path / "v", only=["ruff"], runner=runner, echo=False)
    _, second = verify.verify(
        root=tmp_path, runs_dir=tmp_path / "v", only=["ruff"], runner=runner, echo=False
    )
    assert first != second and first.is_file() and second.is_file()
    assert second.name.endswith("-20261017-120000-2.json")


def test_tail_keeps_the_last_lines():
    text = "\n".join(str(i) for i in range(100))
    assert verify.tail(text, lines=3) == "97\n98\n99"
    assert verify.tail("x" * 50, chars=10) == "x" * 10


# --- the real runner ---------------------------------------------------------------------------------


def test_run_command_streams_to_the_log_and_reports_the_exit_code(tmp_path):
    log = tmp_path / "logs" / "x.log"
    program = "import sys; print('hello'); print('err', file=sys.stderr); sys.exit(3)"
    code, out = verify.run_command(
        [sys.executable, "-c", program], tmp_path, dict(os.environ), 60, log, echo=False
    )
    assert code == 3
    assert "hello\n" in out and "err\n" in out and "\r" not in out
    assert log.read_text(encoding="utf-8") == out


def test_run_command_kills_a_step_at_the_timeout(tmp_path):
    program = "import time; print('start', flush=True); time.sleep(30)"
    code, out = verify.run_command(
        [sys.executable, "-c", program], tmp_path, dict(os.environ), 1.0, tmp_path / "slow.log", echo=False
    )
    assert code is None and out.startswith("start")


# --- the CLI -----------------------------------------------------------------------------------------


def test_cli_help_and_misuse(capsys):
    with pytest.raises(SystemExit) as error:
        verify.main(["--help"])
    assert error.value.code == 0
    out = capsys.readouterr().out
    assert "--skip-e2e" in out and "--pytest-args" in out and "--only" in out
    for argv in (
        ["--only", "lint"],
        ["--step-timeout", "0"],
        ["--step-timeout", "inf"],
        ["--step-timeout", "nan"],
        ["--step-timeout", "soon"],
        ["--pytest-args", "'unclosed"],
        ["--bogus"],
    ):
        with pytest.raises(SystemExit) as error:
            verify.main(argv)
        assert error.value.code == 2, argv


def test_cli_writes_the_record_under_vara_runs_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    calls = []

    def fake_run(command, cwd, env, timeout_s, log_path, *, echo=True):
        calls.append((list(command), echo))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("All checks passed!\n", encoding="utf-8")
        return 0, "All checks passed!\n"

    monkeypatch.setattr(verify, "run_command", fake_run)
    assert verify.main(["--only", "ruff", "--quiet"]) == 0
    out = capsys.readouterr().out
    match = re.search(r"^verify record: (.+)$", out, re.MULTILINE)
    assert match, out
    assert out.splitlines()[-1] == match.group(0)  # the record path is the last line
    path = Path(match.group(1))
    assert path.parent == tmp_path / "runs" / "verify" and path.is_file()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["passed"] is True and record["complete"] is False and record["root"] == str(verify.ROOT)
    assert calls == [([record["interpreter"]["path"], "-m", "ruff", "check", "."], False)]
    assert record["log_dir"] == str(path.with_suffix("")) and Path(record["steps"][2]["log"]).is_file()
    assert f"verify logs: {record['log_dir']}" in out.splitlines()

    def failing(command, cwd, env, timeout_s, log_path, *, echo=True):
        return 1, "Found 1 error.\n"

    monkeypatch.setattr(verify, "run_command", failing)
    assert verify.main(["--only", "ruff", "--quiet"]) == 1
