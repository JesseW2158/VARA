"""`python -m vision.verify` (B7's live check) on FakeSceneTracker only: the before/after round trip on
tests/fixtures/verify pairs, the never-overwrite rules, take_after's retry, and the fake refusals. No camera,
no arm, no .env."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest

from vision import verify as verify_mod
from vision.scene import FakeSceneTracker
from vision.verify import (
    latest_session,
    load_spec,
    new_session_dir,
    parse_args_json,
    parse_events_json,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "verify"
SKILL_SPEC = ROOT / "skills" / "_starter" / "place_in_tray.json"
TASK_SPEC = ROOT / "tests" / "fixtures" / "task_spec_clear_table.json"
FROZEN = datetime(2026, 10, 17, 21, 55, 2, tzinfo=timezone.utc)
STAMP = "20261017-215502"  # FROZEN as a session folder stamp


class _FrozenDatetime(datetime):
    """`datetime` whose `now()` is always FROZEN, patched over vision.verify's: every session folder a test
    creates gets the same UTC stamp, so the same-second `-<n>` suffix is exercised however slow (or loaded)
    the machine is."""

    @classmethod
    def now(cls, tz=None):
        return FROZEN.astimezone(tz) if tz is not None else FROZEN.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No device, no .env, FAKE flags cleared, runs/ under tmp_path."""

    def boom(*args, **kwargs):
        raise AssertionError(f"cv2.VideoCapture{args} must never be called in tests")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for flag in ("FAKE", "FAKE_CAMERA", "FAKE_ROBOT"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VARA_SKILLS_DIR", str(tmp_path / "skills"))


def fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _write(path: Path, data) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return str(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(scene: dict, reference: dict) -> bool:
    """Scene equality ignoring `t` (FakeSceneTracker stamps every snapshot with the current time)."""
    return {k: v for k, v in scene.items() if k != "t"} == {k: v for k, v in reference.items() if k != "t"}


class _Stub:
    """A tracker whose snapshots come from a list (the last one repeats); `is_fake` as given."""

    def __init__(self, scenes, *, is_fake=True, raise_timeout=False):
        self.scenes = [copy.deepcopy(s) for s in scenes]
        self.is_fake = is_fake
        self.raise_timeout = raise_timeout
        self.calls = 0
        self.closed = False

    def snapshot(self, n=10):
        self.calls += 1
        if self.raise_timeout:
            raise TimeoutError("camera delivered no frame")
        return copy.deepcopy(self.scenes[min(self.calls, len(self.scenes)) - 1])

    def close(self):
        self.closed = True


# -- the round trip on fixture pairs ------------------------------------------------------------------------


def test_before_then_after_pass_round_trip(tmp_path, capsys):
    fx = fixture("pass_juice_box_in_tray")
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC before: 3 of 3 objects seen" in out and str(session / "before.json") in out
    assert _same(_load(session / "before.json"), fx["before"])
    meta = _load(session / "session.json")
    assert meta["mode"] == "synthetic" and meta["fake"] is True and meta["frames"] == 10
    assert (
        meta["tracker"] == "FakeSceneTracker"
        and meta["fake_scene"] == before_file
        and "SYNTHETIC" in meta["note"]
    )
    assert sorted(p.name for p in session.iterdir()) == ["before.json", "session.json"]

    argv = [
        "--fake",
        "--fake-scene",
        after_file,
        "--after",
        "--spec",
        spec_file,
        "--wait",
        "0",
        "--out",
        str(session),
    ]
    assert verify_mod.main(argv) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC verdict: ok=True missing=[] reason=spec satisfied" in out
    assert _same(_load(session / "after.json"), fx["after"])
    record = _load(session / "verify.json")
    assert record["ok"] is True and record["missing"] == [] and record["spec"] == fx["spec"]
    result = _load(session / "result.json")
    assert result["ok"] is True and result["mode"] == "synthetic" and result["fake"] is True
    assert result["spec_name"] == "juice_box_to_tray" and result["args"] == {} and result["events"] == []
    # >=: the Windows wall clock ticks every few ms, so the two stamps can be equal
    assert result["before_at"] == meta["before_at"] and result["after_at"] >= result["before_at"]
    assert result["files"]["after"] == str(session / "after.json") and "SYNTHETIC" in result["note"]
    assert sorted(p.name for p in session.iterdir()) == [
        "after.json",
        "before.json",
        "result.json",
        "session.json",
        "verify.json",
    ]


@pytest.mark.parametrize(
    "name", ["fail_juice_box_unseen_after", "fail_obstacle_nudged_3cm", "pass_not_moved_juice_box"]
)
def test_fixture_verdicts_match_through_the_cli(tmp_path, name):
    fx = fixture(name)
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
    argv = [
        "--fake",
        "--fake-scene",
        after_file,
        "--after",
        "--spec",
        spec_file,
        "--args",
        json.dumps(fx["args"]),
    ]
    argv += ["--wait", "0", "--retry", "0.4", "--out", str(session)]
    assert verify_mod.main(argv) == (0 if fx["expected"] else 1)
    result = _load(session / "result.json")
    assert result["ok"] is fx["expected"] and result["retry_s"] == 0.4 and result["wait_s"] == 0.0
    if name == "fail_juice_box_unseen_after":
        assert result["missing"] == ["juice_box"] and "unseen after the run: juice_box" in result["reason"]
    if name == "fail_obstacle_nudged_3cm":
        assert "cereal_box" in result["reason"]


def test_events_feed_event_predicates(tmp_path):
    fx = fixture("pass_hand_over_with_event")
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    common = [
        "--fake",
        "--fake-scene",
        after_file,
        "--after",
        "--spec",
        spec_file,
        "--args",
        json.dumps(fx["args"]),
    ]
    common += ["--wait", "0", "--retry", "0"]
    for session, events, expected in (
        (tmp_path / "with", '["handoff_done"]', 0),
        (tmp_path / "without", "[]", 1),
    ):
        assert (
            verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
        )
        assert verify_mod.main([*common, "--events", events, "--out", str(session)]) == expected
        result = _load(session / "result.json")
        assert result["missing"] == ["pill_bottle"] and result["events"] == json.loads(events)


def test_spec_from_a_skill_file_with_args(tmp_path):
    fx = fixture("pass_juice_box_in_tray")
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
    argv = ["--fake", "--fake-scene", after_file, "--after", "--spec", str(SKILL_SPEC), "--wait", "0"]
    assert verify_mod.main([*argv, "--args", '{"object": "juice_box"}', "--out", str(session)]) == 0
    result = _load(session / "result.json")
    assert result["spec_name"] == "place_in_tray" and result["args"] == {"object": "juice_box"}
    assert _load(session / "verify.json")["spec"]["params"] == {"object": "object"}


def test_after_uses_the_newest_unverified_session_under_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_mod, "datetime", _FrozenDatetime)  # both --before runs in the same second
    fx = fixture("pass_juice_box_in_tray")
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before"]) == 0
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before"]) == 0
    base = tmp_path / "runs" / "verify-check"
    sessions = sorted(p for p in base.iterdir() if p.is_dir())
    # a --fake session is labelled in its folder name; the second one in the same second gets -1
    assert [p.name for p in sessions] == [f"{STAMP}-synthetic", f"{STAMP}-synthetic-1"]
    assert latest_session() == sessions[1]
    argv = ["--fake", "--fake-scene", after_file, "--after", "--spec", spec_file, "--wait", "0"]
    assert verify_mod.main(argv) == 0
    assert (sessions[1] / "verify.json").exists() and not (sessions[0] / "verify.json").exists()
    assert latest_session() == sessions[0]
    assert verify_mod.main(argv) == 0
    assert (sessions[0] / "verify.json").exists() and latest_session() is None
    assert verify_mod.main(argv) == 2  # no unverified session left


def test_combined_before_and_after_prompts_between_the_snapshots(tmp_path, monkeypatch, capsys):
    fx = fixture("pass_not_moved_juice_box")
    scene_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        raise EOFError  # no terminal: EOF counts as Enter

    monkeypatch.setattr("builtins.input", fake_input)
    session = tmp_path / "session"
    argv = ["--fake", "--fake-scene", scene_file, "--before", "--after", "--spec", spec_file, "--wait", "0"]
    assert verify_mod.main([*argv, "--out", str(session)]) == 0
    assert prompts == ["Move the objects by hand to the end state, then press Enter: "]
    assert _same(_load(session / "before.json"), fx["after"]) and _load(session / "result.json")["ok"] is True
    out = capsys.readouterr().out
    assert "SYNTHETIC before:" in out and "SYNTHETIC verdict: ok=True" in out


def test_take_after_retry_semantics_are_kept(tmp_path, monkeypatch):
    fx = fixture("pass_juice_box_in_tray")
    unseen = copy.deepcopy(fx["after"])
    unseen["objects"]["juice_box"]["seen"] = False
    stub = _Stub([fx["before"], unseen, unseen, fx["after"]])
    monkeypatch.setattr(verify_mod, "build_tracker", lambda fake, fake_scene=None: stub)
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--before", "--out", str(session)]) == 0
    assert stub.calls == 1 and stub.closed
    argv = ["--fake", "--after", "--spec", spec_file, "--wait", "0", "--retry", "1.0", "--out", str(session)]
    assert verify_mod.main(argv) == 0
    assert stub.calls == 4  # the first after plus two retries at 0.2 s while juice_box was unseen
    assert _same(_load(session / "after.json"), fx["after"])


# -- never overwritten --------------------------------------------------------------------------------------


def test_a_session_is_never_overwritten(tmp_path):
    fx = fixture("pass_juice_box_in_tray")
    before_file = _write(tmp_path / "before.json", fx["before"])
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
    assert verify_mod.main(["--fake", "--fake-scene", after_file, "--before", "--out", str(session)]) == 2
    assert _same(_load(session / "before.json"), fx["before"])
    argv = [
        "--fake",
        "--fake-scene",
        after_file,
        "--after",
        "--spec",
        spec_file,
        "--wait",
        "0",
        "--out",
        str(session),
    ]
    assert verify_mod.main(argv) == 0
    stamps = {p.name: p.stat().st_mtime_ns for p in session.iterdir()}
    assert verify_mod.main(argv) == 2
    assert {p.name: p.stat().st_mtime_ns for p in session.iterdir()} == stamps


def test_after_without_a_session_is_refused(tmp_path):
    fx = fixture("pass_juice_box_in_tray")
    after_file = _write(tmp_path / "after.json", fx["after"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    argv = ["--fake", "--fake-scene", after_file, "--after", "--spec", spec_file, "--wait", "0"]
    assert verify_mod.main(argv) == 2
    assert verify_mod.main([*argv, "--out", str(tmp_path / "nowhere")]) == 2
    assert not (tmp_path / "nowhere").exists()


# -- fakes are refused unless --fake ------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["FAKE", "FAKE_CAMERA"])
def test_fake_flags_without_fake_are_refused_before_any_snapshot(tmp_path, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")

    def never(*args, **kwargs):
        raise AssertionError("no tracker may be built")

    monkeypatch.setattr(verify_mod, "build_tracker", never)
    assert verify_mod.main(["--before"]) == 2
    assert verify_mod.main(["--before", "--out", str(tmp_path / "s")]) == 2
    assert not (tmp_path / "runs").exists() and not (tmp_path / "s").exists()


def test_an_in_process_fake_tracker_is_refused_without_fake(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_mod, "build_tracker", lambda fake, fake_scene=None: FakeSceneTracker())
    session = tmp_path / "s"
    assert verify_mod.main(["--before", "--out", str(session)]) == 2
    assert not session.exists()


def test_synthetic_before_never_pairs_with_a_physical_after(tmp_path, monkeypatch):
    fx = fixture("pass_juice_box_in_tray")
    before_file = _write(tmp_path / "before.json", fx["before"])
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    assert verify_mod.main(["--fake", "--fake-scene", before_file, "--before", "--out", str(session)]) == 0
    physical = _Stub([fx["after"]], is_fake=False)
    monkeypatch.setattr(verify_mod, "build_tracker", lambda fake, fake_scene=None: physical)
    argv = ["--after", "--spec", spec_file, "--wait", "0", "--out", str(session)]
    assert verify_mod.main(argv) == 2
    assert physical.calls == 0 and not (session / "after.json").exists()
    # and the reverse: a physical before, then --fake
    other = tmp_path / "other"
    assert verify_mod.main(["--before", "--out", str(other)]) == 0
    assert _load(other / "session.json")["mode"] == "physical"
    after_file = _write(tmp_path / "after.json", fx["after"])
    assert verify_mod.main(["--fake", "--fake-scene", after_file, *argv[:-1], str(other)]) == 2


def test_a_dead_camera_is_an_error_not_a_verdict(tmp_path, monkeypatch):
    fx = fixture("pass_juice_box_in_tray")
    spec_file = _write(tmp_path / "spec.json", fx["spec"])
    session = tmp_path / "session"
    _write(session / "before.json", fx["before"])
    dead = _Stub([fx["after"]], raise_timeout=True)
    monkeypatch.setattr(verify_mod, "build_tracker", lambda fake, fake_scene=None: dead)
    argv = ["--fake", "--after", "--spec", spec_file, "--wait", "0", "--retry", "0.4", "--out", str(session)]
    assert verify_mod.main(argv) == 2
    assert (
        dead.calls == 3 and not (session / "after.json").exists() and not (session / "verify.json").exists()
    )
    assert verify_mod.main(["--fake", "--before", "--out", str(tmp_path / "fresh")]) == 2
    assert not (tmp_path / "fresh" / "before.json").exists()


# -- argument checks -----------------------------------------------------------------------------------------


def test_bad_input_exits_2(tmp_path, capsys):
    fx = fixture("pass_juice_box_in_tray")
    after_file = _write(tmp_path / "after.json", fx["after"])
    session = tmp_path / "session"
    _write(session / "before.json", fx["before"])
    base = ["--fake", "--fake-scene", after_file, "--after", "--out", str(session), "--wait", "0"]
    assert verify_mod.main([*base, "--spec", str(SKILL_SPEC)]) == 2  # params missing from --args
    assert verify_mod.main([*base, "--spec", str(SKILL_SPEC), "--args", "[1]"]) == 2
    assert verify_mod.main([*base, "--spec", str(SKILL_SPEC), "--args", "{bad"]) == 2
    assert (
        verify_mod.main([*base, "--spec", str(SKILL_SPEC), "--args", '{"object": "x"}', "--events", "{}"])
        == 2
    )
    bad_spec = _write(tmp_path / "bad_spec.json", {"name": "x", "success": {"nope": {}}})
    assert verify_mod.main([*base, "--spec", bad_spec]) == 2
    assert verify_mod.main([*base, "--spec", str(tmp_path / "missing.json")]) == 2
    assert not (session / "after.json").exists()
    for argv in (
        ["--after"],
        ["--before", "--spec", str(SKILL_SPEC)],
        ["--before", "--fake-scene", after_file],
        ["--fake", "--before", "--frames", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            verify_mod.main(argv)
        assert exc.value.code == 2
    assert verify_mod.main([]) == 2
    assert "--before" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        verify_mod.main(["--help"])
    assert exc.value.code == 0


def test_load_spec_and_parsers():
    spec = load_spec(SKILL_SPEC)
    assert spec["name"] == "place_in_tray" and "code" not in spec
    assert load_spec(TASK_SPEC)["name"] == "clear_table"
    assert parse_args_json('{"object": "juice_box"}', spec) == {"object": "juice_box"}
    with pytest.raises(ValueError, match="missing the spec's params: object"):
        parse_args_json("{}", spec)
    with pytest.raises(ValueError, match="JSON object"):
        parse_args_json("3", spec)
    assert parse_events_json('["handoff_done"]') == ["handoff_done"]
    with pytest.raises(ValueError, match="JSON list"):
        parse_events_json('"handoff_done"')
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_events_json("[")


def test_session_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_mod, "datetime", _FrozenDatetime)
    base = tmp_path / "vc"
    first = new_session_dir(base)
    second = new_session_dir(base)
    assert first.is_dir() and second.is_dir() and [first.name, second.name] == [STAMP, f"{STAMP}-1"]
    assert new_session_dir(base, synthetic=True).name == f"{STAMP}-synthetic"
    assert latest_session(base) is None  # no before.json yet
    (first / "before.json").write_text("{}", encoding="utf-8")
    assert latest_session(base) == first
    (second / "before.json").write_text("{}", encoding="utf-8")
    assert latest_session(base) == second
    (second / "verify.json").write_text("{}", encoding="utf-8")
    assert latest_session(base) == first
    assert latest_session(tmp_path / "absent") is None
