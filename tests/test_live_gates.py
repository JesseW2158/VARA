"""C1, C2, C3, C5, C6, C8 and A14 live gates: scripts/live_gates.py plans, refuses, parses and records.

`subprocess.run` is replaced by a fake that answers each step with canned output, so nothing here reaches
xAI, Daytona, the arm or a camera; `.env` is never read (load_dotenv is stubbed) and both keys are dummy
values that the tests then look for in every written file. The voice probe runs in-process with
`httpx.post` stubbed, exactly as tests/test_orchestrator.py does.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import live_gates as lg

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY_XAI = "xai-DUMMY-live-gates-key-0123456789abcdef"
DUMMY_DAYTONA = "dtn-DUMMY-live-gates-key-0123456789abcdef"
PHRASES = len(json.loads((ROOT / "tests" / "fixtures" / "router_phrases.json").read_text(encoding="utf-8")))
SCORE = {
    "candidate": 1,
    "passed": True,
    "successes": 5,
    "trials": 5,
    "nominal_success": True,
    "collisions": 0,
    "limit_hits": 0,
    "ik_failures": 0,
    "mean_sim_s": 12.3,
    "fail_reason": None,
    "traj": None,
    "error": None,
}
PING_OK = "router (grok-4.3): 'ok' in 900 ms\ncodegen (grok-code-fast-1): 'ok' in 1500 ms\n"
CANDIDATES_OK = (
    "6 of 6 passed check_source on the first try; 6 of 6 after regeneration; 41.2 s; files in somewhere\n"
)


def bench_out(*seconds: float, runner: str = "daytona", remaining: str = "0") -> str:
    lines = ["pool: SandboxPool", "start: 7 sandboxes live in 30.1 s"]
    lines += [
        f"task t{i}: 6 candidates x 5 trials, 6 passed, runner {runner}, {s:.1f} s"
        for i, s in enumerate(seconds, 1)
    ]
    return "\n".join([*lines, f"remaining: {remaining}"]) + "\n"


@pytest.fixture
def env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Dummy keys, no fake flags, runs under tmp_path, and `.env` never read."""
    monkeypatch.setattr(lg, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv(lg.XAI, DUMMY_XAI)
    monkeypatch.setenv(lg.DAYTONA, DUMMY_DAYTONA)
    for flag in lg.FAKE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


class Children:
    """A stand-in for `subprocess.run`: classifies each child by its argv and answers with canned output.

    `listings` are the `remaining (pool): N` counts the inventories report, one per --list call in order;
    `overrides` maps a kind to (exit code, stdout, stderr) or to an exception to raise."""

    def __init__(self, listings: list[int] | None = None, **overrides: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.listings = list(listings if listings is not None else [0, 0, 0])
        self.overrides = overrides

    @staticmethod
    def kind(argv: list[str]) -> str:
        args = argv[1:]
        if "--voice-probe" in args:
            return "voice"
        if args[0].endswith("daytona_smoke.py"):
            return "smoke"
        module = args[1]
        if module == "agents.sandbox_pool":
            return "cleanup" if "--cleanup" in args else "list" if "--list" in args else "bench"
        if module == "agents.candidates":
            return "candidates-help" if "--help" in args else "candidates"
        return {
            "agents.llm": "ping",
            "agents.router": "router",
            "agents.orchestrator": "bench-run",
        }[module]

    def kinds(self) -> list[str]:
        return [call["kind"] for call in self.calls]

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        assert argv[0] == lg.PYTHON
        kind = self.kind(argv)
        env = kwargs["env"]
        self.calls.append({"kind": kind, "argv": list(argv), "env": dict(env), "timeout": kwargs["timeout"]})
        override = self.overrides.get(kind.replace("-", "_"))
        if isinstance(override, BaseException):
            raise override
        if override is not None:
            code, out, err = override
        else:
            code, out, err = self.answer(kind, argv, env)
        return subprocess.CompletedProcess(argv, code, out, err)

    def answer(self, kind: str, argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
        if kind == "candidates-help":
            return 0, "usage: python -m agents.candidates [--skills {all,starter,none}]\n", ""
        if kind == "ping":  # a child that echoes its key: the logs must still not hold it
            return 0, PING_OK, f"debug: Authorization: Bearer {env[lg.XAI]}\n"
        if kind == "router":
            return 0, f"ok    'hand me my pills' -> run\n{PHRASES}/{PHRASES} phrases routed as expected\n", ""
        if kind == "candidates":
            return 0, CANDIDATES_OK, ""
        if kind in ("list", "cleanup"):
            remaining = self.listings.pop(0) if kind == "list" else 0
            out_dir = argv[argv.index("--out") + 1]
            text = "deleted sb-9 (vara=pool): ok\n" if kind == "cleanup" else ""
            text += f"counts: vara=pool {remaining}, vara=rl 0, vara=rl-check 0\n"
            text += f"remaining (pool): {remaining}; written {out_dir}/inventory-x.json\n"
            return (0 if remaining == 0 else 1), text, ""
        if kind == "smoke":
            out_dir = pathlib.Path(argv[argv.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "runner": "daytona",
                "seconds_create": 20.0,
                "seconds_score": 15.0,
                "seconds_total": 41.5,
                "sandbox_deleted": True,
                "remaining_sandboxes": 0,
            }
            (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            err = "scored in 15.0 s by runner 'daytona'\nsandbox deleted; total 41.5 s\n"
            err += f"debug: api_key={env[lg.DAYTONA]}\n"  # a child that echoes its key: scrubbed
            return 0, json.dumps(SCORE) + "\n", err
        if kind == "bench":
            return 0, bench_out(21.0, 22.5, 19.8), ""
        if kind == "bench-run":
            summary = pathlib.Path(env["VARA_RUNS_DIR"]) / "bench" / "run-1" / "summary.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps({"runs": [{"seconds": 2.0 + i / 10} for i in range(5)]}))
            out = (
                "bench-run 'hand me my pills' x5: router=xAI robot=fake camera=fake sandbox=fake mode=demo "
                "(target 5 s)\nmedian 2.2 s (5 of 5 runs reached executing; target 5 s): PASS\n"
                f"wrote {summary}\n"
            )
            return 0, out, ""
        if kind == "voice":
            data = {
                "status": 200,
                "model": "grok-voice-latest",
                "expires_at": 1792209900,
                "token_length": 40,
                "fake_llm": False,
                "fake_token": False,
                "detail": None,
            }
            return 0, json.dumps(data) + "\n", ""
        raise AssertionError(kind)


def files_under(folder: pathlib.Path) -> list[pathlib.Path]:
    return [path for path in folder.rglob("*") if path.is_file()]


def no_key_anywhere(folder: pathlib.Path) -> None:
    for path in files_under(folder):
        data = path.read_bytes()
        assert DUMMY_XAI.encode() not in data, path
        assert DUMMY_DAYTONA.encode() not in data, path


# ------------------------------------------------------------------------------------------- CLI and refusal


def test_help_lists_the_ten_steps(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        lg.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert len(lg.STEP_NAMES) == 10
    for name in lg.STEP_NAMES:
        assert name in text
    for flag in ("--dry-run", "--only", "--skip", "--continue", "--timeout", "--out"):
        assert flag in text
    assert "smoke 1, bench 7, never together" in text


@pytest.mark.parametrize(
    "argv",
    [["--only", "nope"], ["--only", "smoke", "--skip", "inventory-after-smoke"], ["--timeout", "ping=0"]],
)
def test_misuse_exits_2(env: pathlib.Path, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Children())
    with pytest.raises(SystemExit) as exc:
        lg.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["FAKE", "FAKE_LLM", "FAKE_SANDBOX"])
def test_a_fake_flag_refuses_and_writes_nothing(
    env: pathlib.Path, flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_child(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused run started a child")

    monkeypatch.setattr(subprocess, "run", no_child)
    monkeypatch.setenv(flag, "1")
    daytona_loaded = "daytona" in sys.modules
    out = env / "live"
    assert lg.main(["--out", str(out)]) == 2
    assert not out.exists() and not (env / "runs").exists()
    assert ("daytona" in sys.modules) == daytona_loaded  # no Daytona client was imported
    err = capsys.readouterr().err
    assert f"{flag} is 1" in err and "nothing was run" in err


def test_a_missing_key_refuses_with_its_name_only(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subprocess, "run", Children())
    monkeypatch.delenv(lg.DAYTONA)
    assert lg.main(["--out", str(env / "live")]) == 2
    captured = capsys.readouterr()
    assert "DAYTONA_API_KEY not set" in captured.err
    assert DUMMY_XAI not in captured.out + captured.err
    assert not (env / "live").exists()
    monkeypatch.delenv(lg.XAI)
    assert str(lg.refusal(env / "x")).startswith("XAI_API_KEY and DAYTONA_API_KEY not set")


def test_a_used_out_folder_is_refused(env: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Children())
    out = env / "used"
    out.mkdir()
    (out / "old.txt").write_text("x")
    assert lg.main(["--out", str(out)]) == 2
    assert [p.name for p in out.iterdir()] == ["old.txt"]


def test_dry_run_prints_the_plan_and_key_presence_only(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children()
    monkeypatch.setattr(subprocess, "run", children)
    out = env / "live"
    assert lg.main(["--dry-run", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert children.kinds() == ["candidates-help"]  # only the --skills probe; no gate ran
    assert not out.exists()
    assert "plan: 10 step(s)" in text and "ready: run without --dry-run" in text
    assert "keys present: XAI_API_KEY=True, DAYTONA_API_KEY=True" in text
    assert "sandbox counts: smoke 1, bench 7" in text and "--skills none" in text
    assert DUMMY_XAI not in text and DUMMY_DAYTONA not in text
    monkeypatch.setenv("FAKE_LLM", "1")
    assert lg.main(["--dry-run", "--out", str(out)]) == 2
    assert "a real run would refuse (exit 2): FAKE_LLM is 1" in capsys.readouterr().out
    assert not out.exists()


# ------------------------------------------------------------------------------------------------ selection


def test_select_steps_keeps_plan_order_and_brings_the_inventories() -> None:
    assert lg.select_steps(None, None) == list(lg.STEP_NAMES)
    assert lg.select_steps("smoke", None) == ["inventory-before", "smoke", "inventory-after-smoke"]
    assert lg.select_steps("bench,ping", None) == [
        "ping",
        "inventory-before",
        "bench",
        "inventory-after-bench",
    ]
    assert lg.select_steps(None, None, skip_daytona=True) == [
        "ping",
        "router",
        "candidates",
        "bench-run",
        "voice",
    ]
    assert lg.select_steps(None, "bench-run,voice", skip_xai=True) == list(lg.DAYTONA_STEPS)
    with pytest.raises(ValueError, match="do not skip inventory-before"):
        lg.select_steps("bench", "inventory-before")
    with pytest.raises(ValueError, match="no step selected"):
        lg.select_steps("ping", None, skip_xai=True)
    with pytest.raises(ValueError, match="unknown step"):
        lg.select_steps(None, "pong")


def test_parse_timeouts() -> None:
    assert lg.parse_timeouts(["bench=2400", " ping = 30"]) == {"bench": 2400.0, "ping": 30.0}
    for bad in (["bench"], ["nope=3"], ["ping=-1"], ["ping=x"]):
        with pytest.raises(ValueError):
            lg.parse_timeouts(bad)


def test_build_plan_argv_keys_and_the_skills_flag(tmp_path: pathlib.Path) -> None:
    plan = {step.name: step for step in lg.build_plan(tmp_path, skills_flag=True)}
    assert list(plan) == list(lg.STEP_NAMES)
    assert plan["candidates"].argv[-2:] == ("--skills", "none")
    assert plan["smoke"].argv[1:] == ("--out", str(tmp_path / "daytona-smoke"), "--max-seconds", "60")
    assert plan["bench"].argv == ("-m", "agents.sandbox_pool", "--bench", "--out", str(tmp_path / "bench"))
    assert "--list" in plan["inventory-before"].argv and plan["inventory-before"].cleanup_argv is None
    assert plan["inventory-after-bench"].follows == "bench"
    assert "--cleanup" in plan["inventory-after-bench"].cleanup_argv
    assert {plan[name].needs for name in lg.DAYTONA_STEPS} == {lg.DAYTONA}
    assert {plan[name].needs for name in lg.XAI_STEPS} == {lg.XAI}
    assert plan["voice"].env["FAKE_ROBOT"] == plan["voice"].env["FAKE_SANDBOX"] == "1"
    assert plan["bench-run"].env["FAKE_LLM"] == "0" and "--live-router" in plan["bench-run"].argv
    without = {step.name: step for step in lg.build_plan(tmp_path, skills_flag=False)}
    assert "--skills" not in without["candidates"].argv
    assert "no --skills flag" in without["candidates"].note


# -------------------------------------------------------------------------------------------------- parsers


def test_parse_ping() -> None:
    assert lg.parse_ping(0, PING_OK, "", pathlib.Path("."))["pass"] is True
    slow = PING_OK.replace("1500 ms", "15001 ms")
    parsed = lg.parse_ping(0, slow, "", pathlib.Path("."))
    assert parsed["pass"] is False and parsed["over_limit"] == ["codegen"]
    assert (
        lg.parse_ping(0, PING_OK.splitlines()[0], "", pathlib.Path("."))["pass"] is False
    )  # codegen missing
    assert lg.parse_ping(1, PING_OK, "", pathlib.Path("."))["pass"] is False
    failed = "router (grok-4.3): 'ok' in 900 ms\ncodegen (grok-code-fast-1): FAILED HTTPError: 401\n"
    assert lg.parse_ping(1, failed, "", pathlib.Path("."))["failed"] == [failed.splitlines()[1]]


def test_parse_router() -> None:
    ok = lg.parse_router(0, f"{PHRASES}/{PHRASES} phrases routed as expected\n", "", pathlib.Path("."))
    assert ok["pass"] is True and ok["routed"] == ok["expected"] == PHRASES
    out = f"FAIL  'x' -> run: wrong skill\n{PHRASES - 1}/{PHRASES} phrases routed as expected\n"
    bad = lg.parse_router(1, out, "", pathlib.Path("."))
    assert bad["pass"] is False and bad["routed"] == PHRASES - 1 and bad["failures"] == [out.splitlines()[0]]
    assert lg.parse_router(0, "3/3 phrases routed as expected", "", pathlib.Path("."))["pass"] is False


def test_parse_candidates(tmp_path: pathlib.Path) -> None:
    ok = lg.parse_candidates(0, CANDIDATES_OK, "", tmp_path)
    assert ok["pass"] is True and (ok["first_try"], ok["candidates"], ok["seconds"]) == (6, 6, 41.2)
    four = CANDIDATES_OK.replace("6 of 6 passed", "4 of 6 passed")
    assert lg.parse_candidates(0, four, "", tmp_path)["pass"] is False
    run = lg.parse_candidates(1, 'not a learn request: {"action": "run"}\n', "", tmp_path)
    assert run["pass"] is False and "--skills none" in run["reason"]


def test_parse_inventory() -> None:
    zero = lg.parse_inventory(
        0, "remaining (pool): 0; written C:/x/inventory-1.json\n", "", pathlib.Path(".")
    )
    assert zero == {
        "pass": True,
        "remaining": 0,
        "scope": "pool",
        "inventory": "C:/x/inventory-1.json",
        "refused": False,
    }
    two = lg.parse_inventory(1, "remaining (pool): 2; written x.json\n", "", pathlib.Path("."))
    assert two["pass"] is False and two["remaining"] == 2
    refused = lg.parse_inventory(2, "", "DAYTONA_API_KEY is not set", pathlib.Path("."))
    assert refused["pass"] is False and refused["refused"] is True


def test_parse_smoke(tmp_path: pathlib.Path) -> None:
    folder = tmp_path / "daytona-smoke"
    folder.mkdir()
    result = {"runner": "daytona", "seconds_total": 41.5, "sandbox_deleted": True, "remaining_sandboxes": 0}
    (folder / "result.json").write_text(json.dumps(result))
    score = json.dumps(SCORE) + "\n"
    ok = lg.parse_smoke(0, score, "", tmp_path)
    assert ok["pass"] is True and ok["seconds_total"] == 41.5 and ok["score"] == SCORE
    (folder / "result.json").write_text(json.dumps({**result, "seconds_total": 60.4}))
    assert lg.parse_smoke(0, score, "", tmp_path)["pass"] is False
    (folder / "result.json").write_text(json.dumps({**result, "runner": "local"}))
    assert lg.parse_smoke(0, score, "", tmp_path)["pass"] is False
    (folder / "result.json").unlink()  # no result.json: the stderr lines still give the numbers
    err = "scored in 15.0 s by runner 'daytona'\nsandbox deleted; total 44.0 s\n"
    assert lg.parse_smoke(0, score, err, tmp_path)["seconds_total"] == 44.0
    broken = lg.parse_smoke(0, json.dumps({**SCORE, "successes": "five"}), err, tmp_path)
    assert broken["pass"] is False and "successes" in broken["score_problem"]


def test_parse_bench() -> None:
    ok = lg.parse_bench(0, bench_out(21.0, 22.5, 19.8), "", pathlib.Path("."))
    assert ok["pass"] is True and ok["sandboxes_live"] == 7 and ok["remaining"] == 0
    assert [t["seconds"] for t in ok["tasks"]] == [21.0, 22.5, 19.8]
    slow = lg.parse_bench(0, bench_out(21.0, 40.2, 19.8), "", pathlib.Path("."))
    assert slow["pass"] is False and slow["over_limit"] == ["t2"]
    local = lg.parse_bench(0, bench_out(21.0, 22.5, 19.8, runner="local"), "", pathlib.Path("."))
    assert local["pass"] is False and local["local_fallback"] == ["t1", "t2", "t3"]
    assert lg.parse_bench(0, bench_out(21.0, 22.5), "", pathlib.Path("."))["pass"] is False
    unknown = lg.parse_bench(0, bench_out(1.0, 2.0, 3.0, remaining="unknown"), "", pathlib.Path("."))
    assert unknown["remaining"] is None


def test_parse_bench_run(tmp_path: pathlib.Path) -> None:
    head = "bench-run 'hand me my pills' x5: router=xAI robot=fake camera=fake sandbox=fake mode=demo\n"
    line = "median 2.2 s (5 of 5 runs reached executing; target 5 s): PASS\n"
    assert lg.parse_bench_run(0, head + line, "", tmp_path)["median_s"] == 2.2
    assert lg.parse_bench_run(0, head + line, "", tmp_path)["pass"] is True
    fixtures = head.replace("router=xAI", "router=fixtures")
    assert lg.parse_bench_run(0, fixtures + line, "", tmp_path)["pass"] is False
    slow = "median 5.3 s (5 of 5 runs reached executing; target 5 s): FAIL\n"
    assert lg.parse_bench_run(1, head + slow, "", tmp_path)["pass"] is False
    none = "no median (0 of 5 runs reached executing; target 5 s): FAIL\n"
    assert lg.parse_bench_run(1, head + none, "", tmp_path)["median_s"] is None


def test_parse_voice() -> None:
    data = {"status": 200, "model": "m", "expires_at": 1, "token_length": 40, "fake_llm": False}
    assert lg.parse_voice(0, json.dumps({**data, "fake_token": False}), "", pathlib.Path("."))["pass"]
    fake = lg.parse_voice(
        1, json.dumps({**data, "fake_llm": True, "fake_token": True}), "", pathlib.Path(".")
    )
    assert fake["pass"] is False
    assert lg.parse_voice(1, "", "Traceback", pathlib.Path("."))["pass"] is False


def test_scrub_and_child_env(env: pathlib.Path) -> None:
    assert lg.scrub(f"a {DUMMY_XAI} b {DUMMY_DAYTONA}") == "a [XAI_API_KEY] b [DAYTONA_API_KEY]"
    xai_child = lg.child_env(lg.XAI, {"FAKE_ROBOT": "1"})
    assert xai_child[lg.XAI] == DUMMY_XAI and xai_child[lg.DAYTONA] == "" and xai_child["FAKE_ROBOT"] == "1"
    neither = lg.child_env(None)
    assert neither[lg.XAI] == neither[lg.DAYTONA] == ""


# ---------------------------------------------------------------------------------------------- whole runs


def test_a_passing_run_writes_one_entry_per_step_and_no_key(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    children = Children()
    monkeypatch.setattr(subprocess, "run", children)
    assert lg.main([]) == 0  # default --out: <VARA_RUNS_DIR>/live-gates/<UTC stamp>/
    (out,) = (env / "runs" / "live-gates").iterdir()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "finished" and summary["passed"] is True
    assert [s["step"] for s in summary["steps"]] == list(lg.STEP_NAMES)
    assert all(s["pass"] is True for s in summary["steps"])
    assert summary["keys_present"] == {lg.XAI: True, lg.DAYTONA: True}
    assert summary["sandbox_caps"]["smoke"] == 1 and summary["sandbox_caps"]["bench"] == 7
    assert "never together" in summary["sandbox_caps"]["rule"]
    assert summary["failed_steps"] == [] and summary["skipped_steps"] == []
    for name in lg.STEP_NAMES:
        record = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        assert record["pass"] is True and "stdout" not in record and "env" not in record
        assert (out / f"{name}.stdout.log").exists() and (out / f"{name}.stderr.log").exists()
    assert children.kinds() == [
        "candidates-help",
        "ping",
        "router",
        "candidates",
        "list",
        "smoke",
        "list",
        "bench",
        "list",
        "bench-run",
        "voice",
    ]
    for call in children.calls[1:]:  # each child holds only the key its step needs
        needs = (
            lg.XAI if call["kind"] in ("ping", "router", "candidates", "bench-run", "voice") else lg.DAYTONA
        )
        other = lg.DAYTONA if needs == lg.XAI else lg.XAI
        assert call["env"][needs] != "" and call["env"][other] == ""
    candidates = next(c for c in children.calls if c["kind"] == "candidates")
    assert candidates["argv"][-2:] == ["--skills", "none"]
    assert next(c for c in children.calls if c["kind"] == "bench")["timeout"] == lg.DEFAULT_TIMEOUTS["bench"]
    assert "[XAI_API_KEY]" in (out / "ping.stderr.log").read_text(encoding="utf-8")
    no_key_anywhere(out)
    assert DUMMY_XAI not in capsys.readouterr().out
    smoke = json.loads((out / "smoke.json").read_text(encoding="utf-8"))["parsed"]
    assert smoke["seconds_total"] == 41.5 and smoke["runner"] == "daytona"
    bench_run = json.loads((out / "bench-run.json").read_text(encoding="utf-8"))["parsed"]
    assert bench_run["seconds"] == [2.0, 2.1, 2.2, 2.3, 2.4]


def test_a_failed_smoke_still_lists_cleans_up_and_stops(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = Children(listings=[0, 1], smoke=(1, "", "the score came from the local fallback\n"))
    monkeypatch.setattr(subprocess, "run", children)
    out = env / "live"
    assert lg.main(["--out", str(out), "--timeout", "smoke=90"]) == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_steps"] == ["smoke", "inventory-after-smoke"]
    assert summary["skipped_steps"] == ["bench", "inventory-after-bench", "bench-run", "voice"]
    assert len(summary["steps"]) == 10 and summary["clean_start"] is True
    assert children.kinds()[-3:] == ["smoke", "list", "cleanup"]
    assert next(c for c in children.calls if c["kind"] == "smoke")["timeout"] == 90.0
    cleanup = children.calls[-1]
    assert cleanup["argv"][-7:-4] == ["--cleanup", "--label", "pool"] and cleanup["env"][lg.DAYTONA]
    after = json.loads((out / "inventory-after-smoke.json").read_text(encoding="utf-8"))
    assert after["parsed"]["remaining"] == 1 and after["cleanup"]["ran"] is True
    assert after["cleanup"]["remaining"] == 0
    no_key_anywhere(out)


def test_a_dirty_account_creates_nothing_even_with_continue(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = Children(listings=[3])
    monkeypatch.setattr(subprocess, "run", children)
    out = env / "live"
    assert lg.main(["--out", str(out), "--skip-xai", "--continue"]) == 1
    assert children.kinds() == ["list"]  # no smoke, no bench, no cleanup of someone else's sandboxes
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["clean_start"] is False and summary["failed_steps"] == ["inventory-before"]
    reasons = {s["step"]: s.get("reason") for s in summary["steps"]}
    assert "inventory-before found" in reasons["smoke"] and "inventory-before found" in reasons["bench"]
    assert reasons["inventory-after-smoke"] == "smoke did not run"


def test_an_unclean_start_never_cleans_up(env: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = env / "live"
    plan = [s for s in lg.build_plan(out, skills_flag=True) if s.name in ("smoke", "inventory-after-smoke")]
    children = Children(listings=[2])
    monkeypatch.setattr(subprocess, "run", children)
    summary = lg.run_gates(plan, out)  # no inventory-before in this plan: the start is unknown
    assert children.kinds() == ["smoke", "list"]
    assert summary["steps"][1]["cleanup"] == {"ran": False, "reason": lg.CLEANUP_REFUSED}


def test_a_timeout_kills_the_bench_and_the_inventory_still_runs(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeout = subprocess.TimeoutExpired(["python"], 5, output=b"pool: SandboxPool\n", stderr=None)
    children = Children(bench=timeout)
    monkeypatch.setattr(subprocess, "run", children)
    out = env / "live"
    assert lg.main(["--out", str(out), "--only", "bench"]) == 1
    assert children.kinds() == ["list", "bench", "list"]
    bench = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    assert bench["timed_out"] is True and bench["exit_code"] is None and bench["pass"] is False
    assert "killed after the" in (out / "bench.stderr.log").read_text(encoding="utf-8")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_steps"] == ["bench"]
    assert [s["pass"] for s in summary["steps"]] == [True, False, True]


def test_ctrl_c_runs_only_the_inventory_of_the_step_that_ran(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = Children(smoke=KeyboardInterrupt())
    monkeypatch.setattr(subprocess, "run", children)
    out = env / "live"
    assert lg.main(["--out", str(out), "--only", "smoke,bench", "--continue"]) == 1
    assert children.kinds() == ["list", "smoke", "list"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "interrupted" and summary["passed"] is False
    reasons = {s["step"]: s.get("reason") for s in summary["steps"]}
    assert (
        reasons["bench"] == "interrupted (Ctrl+C)" and reasons["inventory-after-bench"] == "bench did not run"
    )


# ---------------------------------------------------------------------------------------------- voice probe


def test_voice_probe_prints_length_and_expiry_never_the_token(
    env: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agents import orchestrator, sandbox_pool
    from arm import driver as driver_mod
    from vision import scene as scene_mod

    token = "ek-SECRET-short-lived-token-value-42"
    skills = env / "skills"
    shutil.copytree(ROOT / "skills" / "_starter", skills / "_starter")
    monkeypatch.setenv("VARA_SKILLS_DIR", str(skills))
    for flag in ("FAKE_ROBOT", "FAKE_CAMERA", "FAKE_SANDBOX"):
        monkeypatch.setenv(flag, "1")  # the probe forces them to 1 as well; monkeypatch restores them
    calls: list[str] = []

    def post(url: str, **kwargs: Any) -> Any:
        calls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {"value": token, "expires_at": 1792209900})

    monkeypatch.setattr(orchestrator.httpx, "post", post)
    driver_mod.reset_driver()
    scene_mod.reset_tracker()
    sandbox_pool.reset_pool()
    try:
        monkeypatch.setattr(orchestrator, "orch", orchestrator.Orchestrator())
        code = lg.main(["--voice-probe"])
    finally:
        driver_mod.reset_driver()
        scene_mod.reset_tracker()
        sandbox_pool.reset_pool()
    captured = capsys.readouterr()
    assert code == 0 and len(calls) == 1
    data = json.loads(captured.out.strip().splitlines()[-1])
    assert data == {
        "status": 200,
        "model": "grok-voice-latest",
        "expires_at": 1792209900,
        "token_length": len(token),
        "fake_llm": False,
        "fake_token": False,
        "detail": None,
    }
    assert token not in captured.out + captured.err and DUMMY_XAI not in captured.out + captured.err
    assert lg.parse_voice(code, captured.out, captured.err, env)["pass"] is True
