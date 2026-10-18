"""`--record NAME` on `python -m arm.replay` (the A9/A12 4-of-5 evidence): one JSON line per --fake/--real
attempt in runs/calibration/replays-NAME.jsonl (VARA_RUNS_DIR), the running tally `NAME: k of n ok`, the
safe-stem check, --record without a mode, a missed grasp counted as not ok, the preflight that refuses a
folder that cannot be created or a corrupt file before anything moves (`prepare_record`), and a record
that cannot be written after the replay failing loud. FakeDriver and stubs only."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import pytest

from arm import driver as driver_mod
from arm import ik, real_robot, replay
from arm.driver import FakeDriver

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180
HOME = [float(v) for v in ik.home_q()]


def traj_file(tmp_path: Path, *segments: dict) -> Path:
    near = list(HOME)
    near[0] += 0.5 * DEG
    t = {
        "version": 1,
        "skill": "record_test",
        "args": {"object": "juice_box"},
        "dt": 1 / 30,
        "joints": list(replay.JOINTS),
        "segments": [{"kind": "path", "q": [HOME, near]}, *segments],
    }
    f = tmp_path / "traj.json"
    f.write_text(json.dumps(t), encoding="utf-8")
    return f


def json_line(out: str) -> dict:
    return json.loads(next(line for line in out.splitlines() if line.startswith("{")))


def lines(path: Path) -> list[dict]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


@pytest.fixture
def runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setenv("VARA_RUNS_DIR", str(runs))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))
    return runs


# -- helpers -------------------------------------------------------------------------------------------------


def test_record_attempt_appends_and_tallies(tmp_path):
    path, ok, total = replay.record_attempt("alpha", {"ok": True, "mode": "fake"}, runs_dir=tmp_path)
    assert path == tmp_path / "calibration" / "replays-alpha.jsonl" and (ok, total) == (1, 1)
    missed = {"ok": False, "reason": "grasp_missed"}
    path2, ok, total = replay.record_attempt("alpha", missed, runs_dir=tmp_path)
    assert path2 == path and (ok, total) == (1, 2)
    entries = lines(path)
    assert [e["label"] for e in entries] == ["alpha", "alpha"] and entries[1]["reason"] == "grasp_missed"
    assert replay.tally(path) == (1, 2)
    assert replay.tally(tmp_path / "missing.jsonl") == (0, 0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n{not json\n")
    with pytest.raises(ValueError, match="line 4"):
        replay.tally(path)
    path.write_text('{"ok": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="boolean 'ok'"):
        replay.tally(path)
    for bad in ({}, {"ok": "yes"}, {"ok": 1}, "ok", None):
        with pytest.raises(ValueError, match="boolean 'ok'"):
            replay.record_attempt("beta", bad, runs_dir=tmp_path)
    assert not (tmp_path / "calibration" / "replays-beta.jsonl").exists()


def test_record_attempt_terminates_a_hand_edited_last_line_and_writes_lf(tmp_path):
    path = tmp_path / "calibration" / "replays-edit.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"label": "edit", "ok": true}')  # valid, but its newline was lost in an edit
    assert replay.record_attempt("edit", {"ok": False, "reason": "stopped"}, runs_dir=tmp_path)[1:] == (1, 2)
    data = path.read_bytes()
    assert data.count(b"\n") == 2 and b"\r" not in data and data.endswith(b"\n")
    assert [e["ok"] for e in lines(path)] == [True, False]


def test_record_names_are_safe_file_stems():
    for good in ("place_in_tray", "A9-run_1", "x", "a" * 64):
        assert replay.checked_record_name(good) == good
    for bad in ("", "../x", "a b", "x.json", "a/b", "a\\b", ".hidden", "-lead", "a" * 65, None, 7):
        with pytest.raises(ValueError, match="--record NAME"):
            replay.checked_record_name(bad)


def test_record_path_honours_vara_runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert replay.record_path("demo") == tmp_path / "elsewhere" / "calibration" / "replays-demo.jsonl"
    monkeypatch.delenv("VARA_RUNS_DIR")
    assert replay.record_path("demo") == ROOT / "runs" / "calibration" / "replays-demo.jsonl"
    assert replay.record_path("demo", tmp_path) == tmp_path / "calibration" / "replays-demo.jsonl"
    with pytest.raises(ValueError):
        replay.record_path("../demo", tmp_path)


# -- CLI -----------------------------------------------------------------------------------------------------


def test_cli_records_fake_attempts_and_prints_the_running_tally(runs, tmp_path, capsys):
    f = traj_file(tmp_path)
    before = datetime.now().astimezone()
    rc = replay.main(["--fake", "--traj", str(f), "--record", "demo"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json_line(out)
    assert "demo: 1 of 1 ok" in out
    path = runs / "calibration" / "replays-demo.jsonl"
    assert path.exists() and str(path) in out
    (entry,) = lines(path)
    assert entry["label"] == "demo" and entry["mode"] == "fake" and entry["ok"] is True
    assert entry["reason"] is None and entry["events"] == len(result["events"])
    assert entry["traj"] == str(f.resolve()) and entry["skill"] == "record_test"
    assert entry["traj_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert 0.0 <= entry["elapsed_s"] < 60.0
    stamped = datetime.fromisoformat(entry["t"])
    assert stamped.tzinfo is not None and abs((stamped - before).total_seconds()) < 120
    rc = replay.main(["--fake", "--traj", str(f), "--record", "demo"])
    out = capsys.readouterr().out
    assert rc == 0 and "demo: 2 of 2 ok" in out
    assert len(lines(path)) == 2  # appended, never overwritten
    # the JSON result line still precedes the tally line
    assert out.index("{") < out.index("demo: 2 of 2 ok")


def test_cli_counts_a_missed_grasp_as_not_ok(runs, monkeypatch, tmp_path, capsys):
    close = {"kind": "grip", "action": "close", "object": "juice_box", "expect_held": True}
    f = traj_file(tmp_path, close)
    original = driver_mod.FakeDriver

    def empty_jaws():
        d = original()
        d.gripper_reading_override = 0.0  # nothing between the jaws
        return d

    monkeypatch.setattr(driver_mod, "FakeDriver", empty_jaws)
    rc = replay.main(["--fake", "--traj", str(f), "--record", "grasp"])
    out = capsys.readouterr().out
    assert rc == 1 and json_line(out)["reason"] == "grasp_missed"
    assert "grasp: 0 of 1 ok" in out
    monkeypatch.setattr(driver_mod, "FakeDriver", original)
    rc = replay.main(["--fake", "--traj", str(f), "--record", "grasp"])
    out = capsys.readouterr().out
    assert rc == 0 and "grasp: 1 of 2 ok" in out
    entries = lines(runs / "calibration" / "replays-grasp.jsonl")
    assert [e["ok"] for e in entries] == [False, True] and entries[0]["reason"] == "grasp_missed"
    assert replay.tally(runs / "calibration" / "replays-grasp.jsonl") == (1, 2)


def test_record_needs_a_mode_and_a_safe_name(runs, tmp_path, capsys):
    f = traj_file(tmp_path)
    with pytest.raises(SystemExit) as exc:
        replay.main(["--traj", str(f), "--record", "demo"])
    assert exc.value.code == 2 and "--record needs --fake or --real" in capsys.readouterr().err
    for bad in ("../demo", "a b", "demo.jsonl"):
        with pytest.raises(SystemExit) as exc:
            replay.main(["--fake", "--traj", str(f), "--record", bad])
        assert exc.value.code == 2 and "--record NAME" in capsys.readouterr().err
    assert not runs.exists()  # nothing was written or replayed


class _StubWatcher:
    def __init__(self, callback=None, **kwargs) -> None:
        self.callback = callback
        self.pressed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _StubDriver(FakeDriver):
    is_fake = False

    def __init__(self, *, port=None, follower_id=None) -> None:
        super().__init__()
        self.connected = False
        self.torque = False

    def connect(self) -> None:
        with self.lock:
            self.connected = True
            self.torque = True
            self.stopped = False
            self.last_command = None

    def _sleep(self, seconds: float) -> None:
        pass


def test_cli_records_real_attempts_with_mode_real(runs, monkeypatch, tmp_path, capsys):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(driver_mod, "Driver", lambda **kw: _StubDriver(**kw))
    monkeypatch.setattr(driver_mod, "EscWatcher", _StubWatcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: True)
    monkeypatch.setattr(replay.time, "sleep", lambda s: None)
    f = traj_file(tmp_path)
    rc = replay.main(["--real", "--traj", str(f), "--record", "real_run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "real_run: 1 of 1 ok" in out
    (entry,) = lines(runs / "calibration" / "replays-real_run.jsonl")
    assert entry["mode"] == "real" and entry["ok"] is True
    # a refused confirmation is not an attempt: nothing is recorded
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: False)
    assert replay.main(["--real", "--traj", str(f), "--record", "real_run"]) == 1
    assert len(lines(runs / "calibration" / "replays-real_run.jsonl")) == 1
    real_robot.clear_cancel()


def test_prepare_record_creates_the_folder_and_writes_nothing(tmp_path):
    path, ok, total = replay.prepare_record("fresh", runs_dir=tmp_path)
    assert path == tmp_path / "calibration" / "replays-fresh.jsonl" and (ok, total) == (0, 0)
    assert path.parent.is_dir() and not path.exists()
    replay.record_attempt("fresh", {"ok": True}, runs_dir=tmp_path)
    replay.record_attempt("fresh", {"ok": False, "reason": "stopped"}, runs_dir=tmp_path)
    assert replay.prepare_record("fresh", runs_dir=tmp_path) == (path, 1, 2)
    assert len(lines(path)) == 2  # the preflight never writes a line
    (tmp_path / "calibration" / "replays-folder.jsonl").mkdir()
    with pytest.raises(ValueError, match="not a file"):
        replay.prepare_record("folder", runs_dir=tmp_path)
    with pytest.raises(ValueError, match="--record NAME"):
        replay.prepare_record("../escape", runs_dir=tmp_path)


def test_cli_announces_the_attempt_before_the_replay(runs, tmp_path, capsys):
    f = traj_file(tmp_path)
    assert replay.main(["--fake", "--traj", str(f), "--record", "count"]) == 0
    out = capsys.readouterr().out
    path = runs / "calibration" / "replays-count.jsonl"
    assert f"recording attempt 1 in {path} (0 ok so far)" in out
    assert replay.main(["--fake", "--traj", str(f), "--record", "count"]) == 0
    out = capsys.readouterr().out
    announce = out.index(f"recording attempt 2 in {path} (1 ok so far)")
    assert announce < out.index("{") < out.index("count: 2 of 2 ok")


def test_record_folder_that_cannot_be_created_refuses_before_the_replay(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "runs"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("VARA_RUNS_DIR", str(blocker))
    f = traj_file(tmp_path)
    rc = replay.main(["--fake", "--traj", str(f), "--record", "demo"])
    captured = capsys.readouterr()
    assert rc == 2 and "cannot use --record demo" in captured.err and "nothing moved" in captured.err
    assert not any(line.startswith("{") for line in captured.out.splitlines())  # no replay ran
    assert blocker.read_text(encoding="utf-8") == "not a directory"


def test_corrupt_record_file_refuses_before_the_replay(runs, monkeypatch, tmp_path, capsys):
    path = runs / "calibration" / "replays-demo.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"label": "demo", "ok": true}\n{"label": "demo", "ok": tru', encoding="utf-8")
    before = path.read_bytes()

    def no_replay(*args, **kwargs):
        raise AssertionError("replayed although the evidence file is corrupt")

    monkeypatch.setattr(replay, "replay", no_replay)
    f = traj_file(tmp_path)
    rc = replay.main(["--fake", "--traj", str(f), "--record", "demo"])
    captured = capsys.readouterr()
    assert rc == 2 and "line 2: not JSON" in captured.err and "nothing moved" in captured.err
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="line 2"):  # the helper never appends to it either
        replay.record_attempt("demo", {"ok": True})
    assert path.read_bytes() == before


def test_record_that_cannot_be_written_after_the_replay_fails_loud(runs, monkeypatch, tmp_path, capsys):
    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(replay, "record_attempt", disk_full)
    f = traj_file(tmp_path)
    rc = replay.main(["--fake", "--traj", str(f), "--record", "demo"])
    captured = capsys.readouterr()
    assert json_line(captured.out)["ok"]  # the replay itself ran fine
    assert rc == 1 and "cannot record the attempt: disk full" in captured.err
