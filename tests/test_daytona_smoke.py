"""C5: scripts/daytona_smoke.py refuses without the live API, records the result and applies the 60 s
verdict. A fake pool stands in for `SandboxPool`; nothing here reaches Daytona, and `.env` is not read."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from scripts import daytona_smoke as smoke

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
}


class FakePool:
    """What the smoke test needs from `SandboxPool`: start, sandbox_ids, score, shutdown, client."""

    runner = "daytona"
    live = 1

    def __init__(self, snapshot: str) -> None:
        self.snapshot = snapshot
        self.started: tuple | None = None
        self.shut = False
        self.client = SimpleNamespace()

    def start(self, n_score: int = 6, n_plan: int = 1) -> int:
        self.started = (n_score, n_plan)
        return self.live

    @property
    def sandbox_ids(self) -> list[str]:
        return [] if self.shut or not self.live else ["sb-1"]

    def score(self, task_dir, candidates, task_path, scene_path, skills_dir, *, trials, seed, timeout):
        assert candidates == [(1, task_dir / "cand_1.py")]
        return [{"candidate": 1, "runner": self.runner, "score": SCORE, "trial_results": [], "out_dir": "x"}]

    def shutdown(self) -> None:
        self.shut = True


class LocalFallbackPool(FakePool):
    runner = "local"


class NoSandboxPool(FakePool):
    live = 0


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No .env, no fake flags, a dummy key, runs under tmp, the inventory canned as 'all gone'."""
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    for flag in ("FAKE", "FAKE_SANDBOX"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        smoke.sp,
        "inventory_after_shutdown",
        lambda client, ids, timeout=30.0: {
            "ours_gone": True,
            "ours_active": [],
            "remaining": 0,
            "sandboxes": [],
        },
    )
    return tmp_path


def test_smoke_refuses_fake_flags_a_missing_key_and_a_used_out_dir(offline, monkeypatch, capsys):
    monkeypatch.setenv("FAKE", "1")
    assert smoke.main(["--out", str(offline / "a")]) == 2
    assert "needs the live Daytona API" in capsys.readouterr().err and not (offline / "a").exists()
    monkeypatch.delenv("FAKE")
    monkeypatch.setenv("FAKE_SANDBOX", "1")
    assert smoke.main(["--out", str(offline / "a")]) == 2 and not (offline / "a").exists()
    monkeypatch.delenv("FAKE_SANDBOX")
    monkeypatch.delenv("DAYTONA_API_KEY")
    assert smoke.main(["--out", str(offline / "b")]) == 2
    assert "DAYTONA_API_KEY is not set" in capsys.readouterr().err and not (offline / "b").exists()
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    (offline / "c").mkdir()
    (offline / "c" / "result.json").write_text("{}", encoding="utf-8")
    assert smoke.main(["--out", str(offline / "c")]) == 2
    assert "already holds result.json" in capsys.readouterr().err
    assert not (offline / "c" / "cand_1.py").exists()


def read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_smoke_write_result_records_the_c5_verdict_and_never_overwrites(tmp_path):
    scored = {
        "snapshot": "arm-sim-v1",
        "runner": "daytona",
        "score": SCORE,
        "seconds_create": 21.0,
        "seconds_score": 19.2,
        "seconds_total": 47.3,
        "sandbox_deleted": True,
        "remaining_sandboxes": 0,
    }
    data = read(smoke.write_result(tmp_path / "r", scored))
    for key in ("snapshot", "runner", "score", "seconds_create", "seconds_score", "seconds_total"):
        assert data[key] == scored[key], key
    assert set(smoke.RESULT_KEYS) <= set(data)
    assert data["under_60"] is True and data["under_max"] is True and data["max_seconds"] == 60.0
    assert data["passed"] is True and data["sandbox_deleted"] is True and data["remaining_sandboxes"] == 0
    with pytest.raises(FileExistsError):
        smoke.write_result(tmp_path / "r", {})
    slow = read(smoke.write_result(tmp_path / "s", {**scored, "seconds_total": 60.5}))
    assert slow["under_60"] is False and slow["under_max"] is False and slow["passed"] is False
    custom = read(smoke.write_result(tmp_path / "t", {**scored, "seconds_total": 80.0, "max_seconds": 90.0}))
    assert custom["under_60"] is False and custom["under_max"] is True and custom["passed"] is True
    empty = read(smoke.write_result(tmp_path / "u" / "deeper", {}))  # parents are created
    assert empty["under_60"] is False and empty["passed"] is False and empty["runner"] is None


GOOD = {"runner": "daytona", "score": SCORE, "seconds_total": 47.3, "max_seconds": 60.0}


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"sandbox_deleted": True}, True),
        ({"sandbox_deleted": None}, True),  # an unknowable listing does not fail the smoke test
        ({"seconds_total": 61.0}, False),
        ({"runner": "local"}, False),
        ({"sandbox_deleted": False}, False),
        ({"error": "boom"}, False),
        ({"score": None}, False),
        ({"runner": None, "seconds_total": None}, False),
    ],
)
def test_smoke_passed_rule(change, expected):
    assert smoke.smoke_passed({**GOOD, **change}) is expected
    assert smoke.under_limit(None, 60.0) is False and smoke.under_limit(60.0, 60.0) is True


@pytest.mark.parametrize(
    "argv", [["--trials", "0"], ["--max-seconds", "-1"], ["--timeout", "0"], ["--bogus"]]
)
def test_smoke_misuse_exits_2(argv, offline):
    with pytest.raises(SystemExit) as exc:
        smoke.main(argv)
    assert exc.value.code == 2


def test_smoke_default_out_dir_is_a_new_stamped_folder_under_runs(offline):
    out = smoke.default_out_dir()
    assert out.parent == offline / "runs" / "daytona-smoke"
    assert re.fullmatch(r"\d{8}-\d{6}", out.name)


def test_smoke_main_with_a_fake_pool_keeps_the_evidence_and_prints_the_score_last(
    offline, monkeypatch, capsys
):
    pools: list[FakePool] = []

    def make(snapshot: str) -> FakePool:
        pools.append(FakePool(snapshot))
        return pools[-1]

    monkeypatch.setattr(smoke, "SandboxPool", make)
    out = offline / "smoke"
    assert smoke.main(["--out", str(out), "--trials", "2"]) == 0
    assert pools[0].started == (1, 0) and pools[0].shut is True  # one scoring sandbox, deleted
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip().splitlines()[-1]) == SCORE
    assert "sandbox deleted; total" in captured.err and "scored in" in captured.err
    assert "result:" in captured.err
    data = read(out / "result.json")
    assert data["runner"] == "daytona" and data["exit_code"] == 0 and data["passed"] is True
    assert data["under_60"] is True and data["under_max"] is True
    assert data["sandbox_id"] == "sb-1" and data["sandbox_deleted"] is True
    assert data["remaining_sandboxes"] == 0 and data["trials"] == 2 and data["error"] is None
    assert data["seconds_create"] is not None and data["seconds_score"] is not None
    assert data["seconds_total"] >= data["seconds_create"]
    for name in ("cand_1.py", "task.json", "scene.json"):
        assert (out / name).is_file(), name


def test_smoke_main_without_out_keeps_a_new_stamped_folder_under_runs(offline, monkeypatch):
    monkeypatch.setattr(smoke, "SandboxPool", FakePool)
    assert smoke.main([]) == 0
    assert smoke.main([]) == 0  # a second run never reuses the first folder
    results = sorted((offline / "runs" / "daytona-smoke").glob("*/result.json"))
    assert len(results) == 2
    assert all(re.fullmatch(r"\d{8}-\d{6}(-\d+)?", path.parent.name) for path in results)


def test_smoke_main_fails_on_local_fallback_no_sandbox_over_time_or_a_straggler(offline, monkeypatch):
    monkeypatch.setattr(smoke, "SandboxPool", LocalFallbackPool)
    assert smoke.main(["--out", str(offline / "local")]) == 1
    data = json.loads((offline / "local" / "result.json").read_text(encoding="utf-8"))
    assert data["runner"] == "local" and data["exit_code"] == 1

    monkeypatch.setattr(smoke, "SandboxPool", NoSandboxPool)
    assert smoke.main(["--out", str(offline / "none")]) == 1
    data = json.loads((offline / "none" / "result.json").read_text(encoding="utf-8"))
    assert data["runner"] is None and "could not be created" in data["error"]

    monkeypatch.setattr(smoke, "SandboxPool", FakePool)
    ticks = iter(range(0, 1000, 25))  # every reading of the run's clock is 25 s after the previous one
    monkeypatch.setattr(smoke, "_clock", lambda: float(next(ticks)))
    assert smoke.main(["--out", str(offline / "slow")]) == 1
    data = json.loads((offline / "slow" / "result.json").read_text(encoding="utf-8"))
    assert data["runner"] == "daytona" and data["seconds_total"] > 60.0
    assert data["under_60"] is False and data["under_max"] is False and data["exit_code"] == 1
    assert smoke.main(["--out", str(offline / "slow-but-allowed"), "--max-seconds", "500"]) == 0
    data = json.loads((offline / "slow-but-allowed" / "result.json").read_text(encoding="utf-8"))
    assert data["under_60"] is False and data["under_max"] is True  # under_60 stays the C5 verdict
    monkeypatch.setattr(smoke, "_clock", smoke.time.perf_counter)

    monkeypatch.setattr(
        smoke.sp,
        "inventory_after_shutdown",
        lambda client, ids, timeout=30.0: {
            "ours_gone": False,
            "ours_active": list(ids),
            "remaining": 1,
            "sandboxes": [],
        },
    )
    assert smoke.main(["--out", str(offline / "straggler")]) == 1
    data = json.loads((offline / "straggler" / "result.json").read_text(encoding="utf-8"))
    assert data["sandbox_deleted"] is False and data["remaining_sandboxes"] == 1

    monkeypatch.setattr(smoke.sp, "inventory_after_shutdown", lambda client, ids, timeout=30.0: None)
    assert smoke.main(["--out", str(offline / "unknown")]) == 0  # an unknowable listing is not a failure
    data = json.loads((offline / "unknown" / "result.json").read_text(encoding="utf-8"))
    assert data["sandbox_deleted"] is None and data["remaining_sandboxes"] is None


def test_smoke_main_records_an_exception_and_still_shuts_the_pool_down(offline, monkeypatch):
    class BrokenPool(FakePool):
        def score(self, *args, **kwargs):
            raise RuntimeError("transport broke")

    pools: list[BrokenPool] = []

    def make(snapshot: str) -> BrokenPool:
        pool = BrokenPool(snapshot)
        pools.append(pool)
        return pool

    monkeypatch.setattr(smoke, "SandboxPool", make)
    assert smoke.main(["--out", str(offline / "broken")]) == 1
    assert pools[0].shut is True
    data = json.loads((offline / "broken" / "result.json").read_text(encoding="utf-8"))
    assert "transport broke" in data["error"] and data["exit_code"] == 1


def test_smoke_main_records_ctrl_c_and_still_checks_the_sandbox_the_signal_handler_deleted(
    offline, monkeypatch
):
    seen: list[list[str]] = []

    class InterruptedPool(FakePool):
        def score(self, *args, **kwargs):
            self.shut = True  # the pool's SIGINT handler deleted the sandboxes and emptied the pool
            raise KeyboardInterrupt

    def after(client, ids, timeout=30.0):
        seen.append(list(ids))
        return {"ours_gone": True, "ours_active": [], "remaining": 0, "sandboxes": []}

    monkeypatch.setattr(smoke, "SandboxPool", InterruptedPool)
    monkeypatch.setattr(smoke.sp, "inventory_after_shutdown", after)
    with pytest.raises(KeyboardInterrupt):
        smoke.main(["--out", str(offline / "interrupted")])
    data = read(offline / "interrupted" / "result.json")
    assert data["error"] == "interrupted (Ctrl+C)" and data["exit_code"] == 1
    assert seen == [["sb-1"]] and data["sandbox_id"] == "sb-1" and data["sandbox_deleted"] is True
