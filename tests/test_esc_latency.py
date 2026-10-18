"""A13 Esc-latency evidence across the `--real` entry points (W2b).

`arm.driver.report_esc_latency` (the shared end-of-session step of arm.driver, arm.replay, arm.real_robot,
scripts/touch_markers.py and the orchestrator's console watcher), `python -m arm.driver
--esc-latency-report` and the orchestrator's dashboard STOP record. No hardware: FakeDriver, stand-in
watchers, records under tmp_path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agents import orchestrator
from arm import driver
from arm.driver import FakeDriver


@pytest.fixture(autouse=True)
def runs_dir(monkeypatch, tmp_path) -> Path:
    """Every default record path points under tmp_path, never at the repo's evidence file."""
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs" / "calibration" / "esc-latency.jsonl"


class Watcher:
    """EscWatcher stand-in with the two attributes the helpers read."""

    def __init__(self, pressed_at: float | None = None):
        self.pressed = pressed_at is not None
        self.pressed_at = pressed_at


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_records(path: Path, records: list[dict | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in records)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# -- report_esc_latency: the shared step ------------------------------------------------------------------


def test_report_esc_latency_records_nothing_without_a_press(runs_dir, capsys):
    d = FakeDriver()
    assert driver.report_esc_latency(Watcher(), d, source="x", connected_at=1.0) is None
    assert not runs_dir.exists() and capsys.readouterr().out == ""


def test_report_esc_latency_session_press_prints_and_records(runs_dir, capsys):
    d = FakeDriver()
    connected_at = time.monotonic()
    w = Watcher(pressed_at=connected_at + 0.01)
    d.freeze()
    d.frozen_at = w.pressed_at + 0.042
    written = driver.report_esc_latency(
        w, d, source="arm.replay", connected_at=connected_at, extra={"argv": ["--real"]}
    )
    assert written == runs_dir
    out = capsys.readouterr().out
    assert "Esc -> freeze command: 42 ms" in out and "(phase session)" in out
    (record,) = lines(runs_dir)
    assert record["source"] == "arm.replay" and record["phase"] == "session"
    assert record["latency_ms"] == pytest.approx(42.0) and record["extra"] == {"argv": ["--real"]}


def test_report_esc_latency_connect_press_is_a_diagnostic(runs_dir, capsys):
    d = FakeDriver()
    w = Watcher(pressed_at=time.monotonic())
    d.frozen_at = w.pressed_at + 0.6
    driver.report_esc_latency(w, d, source="arm.real_robot", connected_at=w.pressed_at + 0.6)
    out = capsys.readouterr().out
    assert "Esc -> freeze command" not in out and "connect phase" in out
    assert lines(runs_dir)[0]["phase"] == "connect"
    # never connected at all: connect phase too
    driver.report_esc_latency(w, d, source="arm.real_robot", connected_at=None)
    assert lines(runs_dir)[1]["phase"] == "connect"


def test_report_esc_latency_never_raises_on_a_failed_write(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    w = Watcher(pressed_at=time.monotonic())
    got = driver.report_esc_latency(w, FakeDriver(), source="x", connected_at=0.0, path=blocker / "esc.jsonl")
    assert got is None and "Esc latency not recorded" in capsys.readouterr().err


# -- --esc-latency-report ---------------------------------------------------------------------------------


def rec(source: str, latency: float | None, phase: str | None = "session") -> dict:
    record = {"t": "2026-10-17T20:00:00.000-04:00", "source": source, "latency_ms": latency, "extra": None}
    if phase is not None:
        record["phase"] = phase
    return record


def test_esc_latency_report_passes_when_every_session_record_is_within_200_ms(runs_dir, capsys):
    write_records(
        runs_dir,
        [rec("arm.replay", 35.0), rec("arm.real_robot", 180.0), rec("arm.driver --jog", 609.0, "connect")],
    )
    assert driver.esc_latency_report() == 0
    out = capsys.readouterr().out
    assert "2 session record(s), max 180 ms; A13 bar 200 ms: PASS" in out
    assert "connect-phase records (Esc while connecting, nothing moving; not judged): 1" in out
    assert "609 ms" in out  # listed, not judged


def test_esc_latency_report_fails_over_the_bar_or_without_a_freeze(runs_dir, capsys):
    write_records(runs_dir, [rec("arm.replay", 35.0), rec("touch_markers", 230.0)])
    assert driver.esc_latency_report() == 1
    assert "FAIL (1 over the bar or never frozen)" in capsys.readouterr().out
    write_records(runs_dir, [rec("arm.replay", 35.0), rec("arm.replay", None)])
    assert driver.esc_latency_report() == 1
    assert "no freeze" in capsys.readouterr().out


def test_esc_latency_report_treats_a_record_without_phase_as_session(runs_dir):
    write_records(runs_dir, [rec("arm.driver --jog", 250.0, phase=None)])
    assert driver.esc_latency_report() == 1


@pytest.mark.parametrize(
    "text",
    [
        None,  # missing
        "",  # empty
        "{not json\n",
        '["a list"]\n',
        json.dumps(rec("x", 10.0, phase="motion")) + "\n",
        '{"t": "?", "source": 3, "latency_ms": 1.0, "phase": "session"}\n',
        '{"source": "x", "phase": "session", "latency_ms": NaN}\n',
        '{"source": "x", "phase": "session", "latency_ms": true}\n',
        json.dumps(rec("x", 2000.0, "connect")) + "\n",  # no session record: nothing to judge
    ],
)
def test_esc_latency_report_exit_2_without_usable_records(runs_dir, text):
    if text is not None:
        runs_dir.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.write_text(text, encoding="utf-8")
    assert driver.esc_latency_report() == 2


def test_esc_latency_report_cli(tmp_path, capsys):
    path = write_records(tmp_path / "esc.jsonl", [rec("arm.replay", 50.0)])
    assert driver.main(["--esc-latency-report", "--file", str(path)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert driver.main(["--esc-latency-report", "--file", str(tmp_path / "missing.jsonl")]) == 2
    for argv in (
        ["--file", str(path), "--jog", "--fake"],  # --file belongs to the report
        ["--esc-latency-report", "--real"],  # the report touches no arm
        ["--esc-latency-report", "--jog"],  # one action per run
    ):
        with pytest.raises(SystemExit) as exc:
            driver.main(argv)
        assert exc.value.code == 2


# -- the orchestrator -------------------------------------------------------------------------------------


def wait_for(path: Path, timeout: float = 2.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return lines(path)
        time.sleep(0.01)
    raise AssertionError(f"{path} was not written")


def test_dashboard_stop_on_the_real_arm_is_recorded_after_the_freeze(runs_dir):
    d = FakeDriver()
    d.is_fake = False  # stands in for the connected real arm
    reply = {"source": "button", "latency_ms": 0.4, "press_to_freeze_ms": 37.0, "task_id": "t-0001"}
    orchestrator._record_dashboard_stop(d, reply)
    (record,) = wait_for(runs_dir)
    assert record["source"] == "dashboard-button" and record["phase"] == "session"
    assert record["latency_ms"] == 37.0  # key press to freeze, the number A13 judges
    assert record["extra"] == {"server_latency_ms": 0.4, "task_id": "t-0001"}


def test_dashboard_stop_records_nothing_for_the_fake_or_a_disconnected_arm(runs_dir):
    reply = {"source": "esc", "latency_ms": 0.1, "press_to_freeze_ms": 20.0, "task_id": None}
    orchestrator._record_dashboard_stop(FakeDriver(), reply)  # the fake arm
    disconnected = FakeDriver()
    disconnected.is_fake = False
    disconnected.disconnect()
    orchestrator._record_dashboard_stop(disconnected, reply)
    orchestrator._record_dashboard_stop(None, reply)
    time.sleep(0.1)
    assert not runs_dir.exists()


def test_console_esc_records_only_on_the_real_arm_after_the_connect_starts(monkeypatch, runs_dir):
    """`python -m agents.orchestrator --real`: the console watcher's callback stops first, then records
    with source orchestrator-console; nothing is recorded for an Esc during the countdown."""
    import uvicorn

    from arm import real_robot

    d = FakeDriver()
    d.is_fake = False
    d.disconnect()
    orch = orchestrator.Orchestrator()
    made: list = []

    class StubWatcher:
        def __init__(self, callback):
            self.callback, self.pressed, self.pressed_at = callback, False, None
            made.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def press(self):
            self.pressed, self.pressed_at = True, time.monotonic()
            self.callback()

    def serve(*args, **kwargs):
        time.sleep(0.05)  # past the connect stamp (Windows' monotonic clock ticks every ~15.6 ms)
        made[-1].press()  # Esc while serving: a session press

    monkeypatch.setattr(orchestrator, "orch", orch)
    monkeypatch.setattr(orchestrator.driver_mod, "get_driver", lambda: d)
    monkeypatch.setattr(orchestrator.driver_mod, "EscWatcher", StubWatcher)
    monkeypatch.setattr(real_robot, "confirm_arm_clear", lambda seconds=3, watcher=None: True)
    monkeypatch.setattr(real_robot, "cancelled", lambda: False)
    monkeypatch.setattr(uvicorn, "run", serve)
    assert orchestrator.main(["--real"]) == 0
    (record,) = lines(runs_dir)
    assert record["source"] == "orchestrator-console" and record["phase"] == "session"
    assert record["latency_ms"] is not None and record["extra"] == {"argv": ["--real"]}
    assert d.stopped and orch.state == "stopped"

    # an Esc during the countdown aborts the startup and leaves no record
    runs_dir.unlink()
    d2 = FakeDriver()
    d2.is_fake = False
    d2.disconnect()
    orch2 = orchestrator.Orchestrator()
    monkeypatch.setattr(orchestrator, "orch", orch2)
    monkeypatch.setattr(orchestrator.driver_mod, "get_driver", lambda: d2)

    def countdown(seconds=3, watcher=None):
        watcher.press()
        return False

    monkeypatch.setattr(real_robot, "confirm_arm_clear", countdown)
    assert orchestrator.main(["--real"]) == 1
    assert not runs_dir.exists() and not d2.connected
