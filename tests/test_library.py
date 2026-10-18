"""agents.library: seeding, validation, atomic saves, reset/backup, routines and the CLI (C9, C10).

Every test works on a tmp_path copy of skills/_starter so the repo's skills/ stays untouched.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from agents import library as lib
from agents.candidates import check_source, routine_candidate
from agents.library import Library, LibraryError, MissingArgument, UnknownSkill, make_skill, routine_spec
from sim.success import already_true, evaluate

ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = ROOT / "skills" / "_starter"
FIXTURES = ROOT / "tests" / "fixtures"
SKILL_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "skill.json").read_text(encoding="utf-8"))
)
TASK_SPEC_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "task_spec.json").read_text(encoding="utf-8"))
)
CLEAR_TABLE = json.loads((FIXTURES / "skill_clear_table.json").read_text(encoding="utf-8"))
SCENE = json.loads((ROOT / "config" / "fake_scene.json").read_text(encoding="utf-8"))
STARTERS = ["hand_over", "place_in_tray"]
ISO_WITH_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")

# A spec that uses "$object" and "$zone" inside nested all/any/not and in every object predicate body.
NESTED_SPEC = {
    "name": "nested_demo",
    "description": "Nested demo.",
    "params": {"object": "object", "zone": "zone"},
    "success": {
        "all": [
            {"in_zone": {"object": "$object", "zone": "$zone"}},
            {
                "any": [
                    {"near": {"object": "$object", "x": 0.2, "y": 0.0, "tol": 0.03}},
                    {"moved": {"object": "$object", "min": 0.05}},
                ]
            },
            {"not": {"not_moved": {"object": "$object", "max": 0.02}}},
            {"not_moved": {"object": "cereal_box", "max": 0.02}},
            {"event": "handoff_done"},
        ]
    },
    "english": "Nested demo passes.",
}


def learned(name: str = "clear_table", **overrides: Any) -> dict[str, Any]:
    """A copy of the clear_table fixture skill renamed to `name`, with top-level overrides applied."""
    skill = json.loads(json.dumps(CLEAR_TABLE))
    skill["name"] = name
    skill["spec"]["name"] = name
    skill.update(overrides)
    return skill


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def live_files(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("*.json"))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A copy of skills/_starter under tmp_path, an empty _backup and no live skills."""
    root = tmp_path / "skills"
    shutil.copytree(STARTER_DIR, root / "_starter")
    (root / "_backup").mkdir()
    return root


@pytest.fixture
def library(root: Path, tmp_path: Path) -> Library:
    return Library(root, archive_root=tmp_path / "runs" / "archive")


@pytest.fixture
def seeded(library: Library) -> Library:
    library.ensure_seeded()
    return library


# --- the repo's starters and the defaults --------------------------------------------------------------


def test_repo_starters_are_valid_trusted_skills():
    assert sorted(p.stem for p in STARTER_DIR.glob("*.json")) == STARTERS
    for name in STARTERS:
        skill = json.loads((STARTER_DIR / f"{name}.json").read_text(encoding="utf-8"))
        SKILL_VALIDATOR.validate(skill)
        assert skill["name"] == name and skill["trusted"] is True
        assert check_source(skill["code"]) == []
        assert ISO_WITH_OFFSET.match(skill["created"])


def test_default_folders_point_into_the_repo():
    default = Library()
    assert default.root == ROOT / "skills"
    assert default.starter_dir == ROOT / "skills" / "_starter"
    assert default.backup_dir == ROOT / "skills" / "_backup"
    assert default.archive_root == ROOT / "runs" / "archive"
    custom = Library(ROOT / "x", starter_dir=ROOT / "s", backup_dir=ROOT / "b", archive_root=ROOT / "a")
    assert (custom.starter_dir, custom.backup_dir, custom.archive_root) == (
        ROOT / "s",
        ROOT / "b",
        ROOT / "a",
    )


# --- seeding, reading ------------------------------------------------------------------------------------


def test_fresh_root_has_no_live_skills(library: Library):
    assert library.all() == []
    assert library.summaries() == []
    assert library.get("hand_over") is None
    assert library.starter_names() == STARTERS


def test_ensure_seeded_copies_the_starters_once(library: Library, root: Path):
    assert library.ensure_seeded() == STARTERS
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    for name in STARTERS:
        live, starter = root / f"{name}.json", STARTER_DIR / f"{name}.json"
        assert json.loads(live.read_text(encoding="utf-8")) == json.loads(starter.read_text(encoding="utf-8"))
        raw = live.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"}\n") and raw.startswith(b'{\n  "name": ')
    assert library.ensure_seeded() == []  # already seeded: nothing copied, nothing touched
    assert not list(root.glob("*.tmp"))


def test_ensure_seeded_leaves_a_non_empty_library_alone(library: Library, root: Path):
    write_json(root / "clear_table.json", learned())
    assert library.ensure_seeded() == []
    assert live_files(root) == ["clear_table.json"]


def test_all_get_and_summaries(seeded: Library):
    skills = seeded.all()
    assert [s["name"] for s in skills] == STARTERS
    for skill in skills:
        SKILL_VALIDATOR.validate(skill)
        assert skill["trusted"] is True
    assert seeded.get("hand_over")["spec"]["params"] == {"object": "object"}
    assert seeded.get("missing") is None
    assert seeded.get("../hand_over") is None
    assert seeded.get("_starter") is None
    assert seeded.get("") is None
    assert seeded.summaries() == [
        {"name": s["name"], "description": s["spec"]["description"], "params": s["spec"]["params"]}
        for s in skills
    ]
    assert seeded.path("hand_over") == seeded.root / "hand_over.json"
    with pytest.raises(LibraryError):
        seeded.path("_starter")


def test_all_sorts_by_created_then_name(seeded: Library):
    seeded.save(learned("older", created="2025-01-01T00:00:00+00:00"))
    seeded.save(learned("newer", created="2030-01-01T00:00:00-04:00"))
    seeded.save(learned("odd_stamp", created="sometime later"))  # unparsable: sorted last
    assert [s["name"] for s in seeded.all()] == ["older", *STARTERS, "newer", "odd_stamp"]


# --- save -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Bad-Name", "_hidden", "x", "1abc", "", "a" * 42, "hand over", "clear_table\n", "hand_over\n", None, 7],
)
def test_save_refuses_bad_names(seeded: Library, root: Path, name: Any):
    skill = learned()
    skill["name"] = name
    if isinstance(name, str):
        skill["spec"]["name"] = name
    with pytest.raises(LibraryError):
        seeded.save(skill)
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    assert not list(root.glob("*.tmp"))


def test_trailing_newline_names_are_refused_before_the_filesystem(seeded: Library, root: Path):
    """`re.match` (and jsonschema's `pattern`) accept "clear_table\\n" because `$` matches before a final
    newline; the library refuses it with LibraryError instead of failing inside mkstemp (Windows, OSError
    22) or writing "clear_table\\n.json" (POSIX)."""
    bad = "clear_table\n"
    assert lib.NAME_RE.match(bad) is not None and lib._valid_name(bad) is False
    skill = learned(bad)
    with pytest.raises(LibraryError):
        seeded.save(skill)
    with pytest.raises(LibraryError):
        seeded.path(bad)
    with pytest.raises(LibraryError):
        make_skill(skill["spec"], "x", None, None)
    with pytest.raises(LibraryError):
        routine_spec(bad, [{"skill": "hand_over", "args": {"object": "pill_bottle"}}], seeded)
    with pytest.raises(UnknownSkill):
        routine_spec("morning", [{"skill": "hand_over\n", "args": {"object": "pill_bottle"}}], seeded)
    assert seeded.get(bad) is None and seeded.delete(bad) is False
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    assert not list(root.glob(".*"))


def test_save_refuses_schema_violations(seeded: Library, root: Path):
    breakers = [
        lambda s: s.pop("code"),
        lambda s: s.__setitem__("code", ""),
        lambda s: s.__setitem__("trusted", "yes"),
        lambda s: s.__setitem__("extra", 1),
        lambda s: s["sim"].__setitem__("trials", -1),
        lambda s: s["real"].pop("runs"),
        lambda s: s["spec"].__setitem__("success", {"nope": 1}),
        lambda s: s["spec"].__setitem__("params", {"object": "thing"}),
        lambda s: s.__setitem__("created", "short"),
    ]
    for breaker in breakers:
        skill = learned()
        breaker(skill)
        with pytest.raises(LibraryError):
            seeded.save(skill)
    with pytest.raises(LibraryError):
        seeded.save("not a mapping")  # type: ignore[arg-type]
    assert seeded.get("clear_table") is None
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]


def test_untrusted_skill_never_shadows_a_trusted_one(seeded: Library, root: Path):
    before = (root / "hand_over.json").read_bytes()
    with pytest.raises(LibraryError, match="hand_over"):
        seeded.save(learned("hand_over", trusted=False))
    assert (root / "hand_over.json").read_bytes() == before

    # trusted -> trusted is allowed: the orchestrator bumps a starter's real counts after a run
    skill = seeded.get("hand_over")
    skill["real"] = {"runs": 1, "successes": 1}
    assert seeded.save(skill) == root / "hand_over.json"
    assert seeded.get("hand_over")["real"] == {"runs": 1, "successes": 1}

    # a trusted non-starter stays protected too, and an untrusted skill may replace an untrusted one
    seeded.save(learned("rehearsal", trusted=True))
    with pytest.raises(LibraryError):
        seeded.save(learned("rehearsal", trusted=False))
    seeded.save(learned("clear_table"))
    updated = learned("clear_table")
    updated["real"] = {"runs": 2, "successes": 1}
    seeded.save(updated)
    assert seeded.get("clear_table")["real"]["runs"] == 2

    # a trusted skill may still replace a starter whose live file is broken
    (root / "place_in_tray.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(LibraryError):
        seeded.save(learned("place_in_tray"))
    fixed = json.loads((STARTER_DIR / "place_in_tray.json").read_text(encoding="utf-8"))
    seeded.save(fixed)
    assert seeded.get("place_in_tray")["trusted"] is True


def test_save_writes_canonical_json(seeded: Library):
    skill = learned("unicode_one")
    skill["spec"]["description"] = "Rotate by π/4, then place."
    path = seeded.save(skill)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n") and "π".encode() in raw
    assert raw.decode("utf-8").startswith('{\n  "name": "unicode_one",\n  "trusted": false,\n')
    assert json.loads(raw) == skill
    assert list(json.loads(raw)) == ["name", "trusted", "spec", "code", "sim", "real", "created"]


def test_failed_dump_or_replace_leaves_the_old_file_intact(seeded: Library, root: Path, monkeypatch):
    path = seeded.save(learned())
    before = path.read_bytes()
    changed = learned()
    changed["real"]["runs"] = 9

    def bad_dump(data):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(lib, "_dump", bad_dump)
    with pytest.raises(RuntimeError):
        seeded.save(changed)
    assert path.read_bytes() == before
    monkeypatch.undo()

    def bad_replace(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(lib, "_replace", bad_replace)
    with pytest.raises(OSError):
        seeded.save(changed)
    assert path.read_bytes() == before
    monkeypatch.undo()

    assert not list(root.glob("*.tmp")) and not list(root.glob(".*"))
    seeded.save(changed)  # the writer works again once the fault is gone
    assert seeded.get("clear_table")["real"]["runs"] == 9


def test_save_survives_a_transient_permission_error_on_replace(seeded: Library, monkeypatch):
    real_replace = lib.os.replace
    calls: list[int] = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("another reader holds the file")
        real_replace(src, dst)

    monkeypatch.setattr(lib.os, "replace", flaky)
    monkeypatch.setattr(lib.time, "sleep", lambda s: None)
    seeded.save(learned())
    assert len(calls) == 3
    assert seeded.get("clear_table")["name"] == "clear_table"


def test_save_fails_loud_when_the_directory_refuses_writes(seeded: Library, root: Path, monkeypatch):
    """A root whose ACL denies file creation made `tempfile.mkstemp` retry for minutes on Windows; the
    library's own temp-file loop raises the PermissionError at once and leaves nothing behind."""
    path = seeded.save(learned())
    before = path.read_bytes()
    real_open = lib.os.open
    attempts: list[str] = []

    def denied(name, flags, *args, **kwargs):
        if str(name).startswith(str(root)):
            attempts.append(str(name))
            raise PermissionError(13, "Access is denied")
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(lib.os, "open", denied)
    started = time.perf_counter()
    with pytest.raises(PermissionError):
        seeded.save(learned())
    with pytest.raises(PermissionError):
        Library(root / "fresh", starter_dir=root / "_starter").ensure_seeded()
    assert time.perf_counter() - started < 2.0
    assert len(attempts) == 2  # one attempt per write: no spin
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert not list(root.glob("*.tmp")) and not list((root / "fresh").glob("*"))

    # a name collision is retried a bounded number of times with fresh random names, then reported
    def taken(name, flags, *args, **kwargs):
        if str(name).startswith(str(root)):
            attempts.append(str(name))
            raise FileExistsError(17, "exists")
        return real_open(name, flags, *args, **kwargs)

    attempts.clear()
    monkeypatch.setattr(lib.os, "open", taken)
    with pytest.raises(FileExistsError):
        seeded.save(learned())
    assert len(attempts) == lib._TEMP_TRIES == len(set(attempts))
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert not list(root.glob("*.tmp"))


def test_reset_demo_retries_a_transient_permission_error_on_move(
    seeded: Library, tmp_path: Path, monkeypatch
):
    """A sandbox upload or dashboard poll holding a learned file open for a moment must not fail the reset."""
    seeded.save(learned("clear_table"))
    real_move = lib.shutil.move
    calls: list[int] = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("another process still reads the file")
        return real_move(src, dst)

    monkeypatch.setattr(lib.shutil, "move", flaky)
    monkeypatch.setattr(lib.time, "sleep", lambda s: None)
    assert seeded.reset_demo() == STARTERS
    assert len(calls) == 3
    archives = list((tmp_path / "runs" / "archive").iterdir())
    assert len(archives) == 1 and [p.name for p in archives[0].iterdir()] == ["clear_table.json"]
    assert seeded.get("clear_table") is None

    # a file that stays locked surfaces the error after the bounded retries; nothing is lost
    seeded.save(learned("stuck"))
    calls.clear()

    def locked(src, dst):
        calls.append(1)
        raise PermissionError("still open")

    monkeypatch.setattr(lib.shutil, "move", locked)
    with pytest.raises(PermissionError):
        seeded.reset_demo()
    assert len(calls) == lib._REPLACE_TRIES
    assert seeded.get("stuck")["name"] == "stuck"


# --- delete, reset_demo, load_backup -------------------------------------------------------------------


def test_delete_never_removes_starters(seeded: Library, root: Path):
    seeded.save(learned())
    assert seeded.delete("clear_table") is True
    assert not (root / "clear_table.json").exists()
    assert seeded.delete("clear_table") is False
    assert seeded.delete("hand_over") is False
    assert seeded.delete("../hand_over") is False
    assert seeded.delete("_starter") is False
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    assert (root / "_starter" / "hand_over.json").is_file()


def test_reset_demo_archives_learned_skills_and_restores_the_starters(seeded: Library, root: Path, tmp_path):
    archive_root = tmp_path / "runs" / "archive"
    seeded.save(learned("clear_table"))
    seeded.save(learned("good_morning"))
    (root / "_junk.json").write_text("{not a skill", encoding="utf-8")
    used = seeded.get("hand_over")
    used["real"] = {"runs": 3, "successes": 2}
    seeded.save(used)

    assert seeded.reset_demo() == STARTERS
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    assert seeded.get("hand_over")["real"] == {"runs": 0, "successes": 0}
    archives = sorted(archive_root.iterdir())
    assert len(archives) == 1 and re.fullmatch(r"\d{8}-\d{6}", archives[0].name)
    assert sorted(p.name for p in archives[0].iterdir()) == [
        "_junk.json",
        "clear_table.json",
        "good_morning.json",
    ]
    archived = json.loads((archives[0] / "clear_table.json").read_text(encoding="utf-8"))
    assert archived["name"] == "clear_table"

    # nothing learned: no new archive directory; learned again: its own directory, even within a second
    assert seeded.reset_demo() == STARTERS
    assert len(list(archive_root.iterdir())) == 1
    seeded.save(learned("again"))
    assert seeded.reset_demo() == STARTERS
    assert len(list(archive_root.iterdir())) == 2
    assert seeded.get("again") is None


def test_reset_demo_seeds_an_empty_library(library: Library, root: Path, tmp_path):
    assert library.reset_demo() == STARTERS
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    assert not (tmp_path / "runs" / "archive").exists()


def test_load_backup_loads_untrusted_and_skips_invalid_files(seeded: Library, root: Path, caplog):
    backup = root / "_backup"
    write_json(backup / "rehearsal.json", learned("rehearsal"))
    write_json(backup / "claims_trust.json", learned("claims_trust", trusted=True))
    (backup / "broken.json").write_text("{not json", encoding="utf-8")
    bad = learned("bad")
    del bad["code"]
    write_json(backup / "bad.json", bad)
    write_json(backup / "mismatch.json", learned("other_name"))
    write_json(backup / "hand_over.json", learned("hand_over"))  # an untrusted shadow of a starter
    tray_shadow = json.loads((STARTER_DIR / "place_in_tray.json").read_text(encoding="utf-8"))
    tray_shadow["code"] = "def run(robot, args):\n    return True\n"
    write_json(backup / "place_in_tray.json", tray_shadow)  # a starter copy claiming trusted: true
    (backup / "notes.txt").write_text("not a skill", encoding="utf-8")
    starters_before = [(root / f"{name}.json").read_bytes() for name in STARTERS]

    with caplog.at_level(logging.WARNING, logger="agents.library"):
        assert seeded.load_backup() == ["claims_trust", "rehearsal"]
    assert seeded.get("rehearsal")["spec"]["name"] == "rehearsal"
    # trusted marks the hand-written starters only: a backup claiming it loads untrusted, on disk too
    assert seeded.get("claims_trust")["trusted"] is False
    assert json.loads((root / "claims_trust.json").read_text(encoding="utf-8"))["trusted"] is False
    assert [(root / f"{name}.json").read_bytes() for name in STARTERS] == starters_before
    assert seeded.get("bad") is None and seeded.get("other_name") is None and seeded.get("mismatch") is None
    assert live_files(root) == [
        "claims_trust.json",
        "hand_over.json",
        "place_in_tray.json",
        "rehearsal.json",
    ]
    warned = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    for fragment in (
        "broken.json",
        "bad.json",
        "mismatch.json",
        "hand_over.json",
        "place_in_tray.json",
        "claims_trust.json",
    ):
        assert any(fragment in message for message in warned), fragment

    # loading again overwrites the live copies and reports the same names; a backup-loaded skill is an
    # ordinary learned skill, so an untrusted re-save (a re-learn of that name) and delete work
    live = seeded.get("rehearsal")
    live["real"] = {"runs": 4, "successes": 4}
    seeded.save(live)
    seeded.save(learned("claims_trust"))
    assert seeded.load_backup() == ["claims_trust", "rehearsal"]
    assert seeded.get("rehearsal")["real"] == {"runs": 1, "successes": 1}
    assert seeded.delete("claims_trust") is True
    assert seeded.reset_demo() == STARTERS


def test_load_backup_with_no_backup_folder(seeded: Library, root: Path):
    shutil.rmtree(root / "_backup")
    assert seeded.load_backup() == []


# --- malformed live files and concurrency --------------------------------------------------------------


def test_malformed_live_files_are_skipped_not_fatal(seeded: Library, root: Path, caplog):
    (root / "broken.json").write_text('{"name": ', encoding="utf-8")
    write_json(root / "wrong.json", {"name": "wrong", "trusted": False})
    write_json(root / "renamed.json", learned("something_else"))
    (root / "notes.txt").write_text("not a skill", encoding="utf-8")
    (root / "folder.json").mkdir()
    with caplog.at_level(logging.WARNING, logger="agents.library"):
        assert [s["name"] for s in seeded.all()] == STARTERS
        assert seeded.get("broken") is None
        assert seeded.get("wrong") is None
        assert seeded.get("renamed") is None
        assert [s["name"] for s in seeded.summaries()] == STARTERS
    warned = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    for fragment in ("broken.json", "wrong.json", "renamed.json"):
        assert any(fragment in message for message in warned), fragment

    # an unchanged malformed file is reported once, not on every dashboard poll
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agents.library"):
        seeded.all()
        seeded.get("broken")
    assert not [r for r in caplog.records if "broken.json" in r.message]

    # repairing the file makes it live again
    write_json(root / "broken.json", learned("broken"))
    assert seeded.get("broken")["name"] == "broken"
    assert "broken" in [s["name"] for s in seeded.all()]


def test_concurrent_saves_from_eight_threads(seeded: Library, root: Path):
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            for _round in range(5):
                seeded.save(learned(f"skill_{i}"))
                assert seeded.get(f"skill_{i}")["name"] == f"skill_{i}"
                seeded.all()
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,), name=f"saver-{i}") for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert not any(t.is_alive() for t in threads)
    names = [s["name"] for s in seeded.all()]
    assert sorted(n for n in names if n.startswith("skill_")) == [f"skill_{i}" for i in range(8)]
    for i in range(8):
        SKILL_VALIDATOR.validate(json.loads((root / f"skill_{i}.json").read_text(encoding="utf-8")))
    assert not list(root.glob("*.tmp"))


def test_lock_serializes_file_operations_under_contention(seeded: Library, root: Path, monkeypatch):
    """Pins `Library.lock`: six threads saving the SAME name, one resetting/loading/deleting and one
    reading never overlap inside the file helpers. With a no-op lock the peak overlap exceeds one and
    reset_demo starts raising on files that vanish under it."""
    write_json(root / "_backup" / "rehearsal.json", learned("rehearsal"))
    inside = 0
    peak = 0
    guard = threading.Lock()

    def tracked(original):
        def wrapper(self, *args):
            nonlocal inside, peak
            with guard:
                inside += 1
                peak = max(peak, inside)
            try:
                time.sleep(0)  # yield the GIL: without the lock another thread walks in here
                return original(self, *args)
            finally:
                with guard:
                    inside -= 1

        return wrapper

    monkeypatch.setattr(Library, "_write", tracked(Library._write))
    monkeypatch.setattr(Library, "_load", tracked(Library._load))
    errors: list[BaseException] = []
    resets: list[list[str]] = []
    barrier = threading.Barrier(8)

    def saver(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            for n in range(5):
                seeded.save(learned("shared", created=f"2030-01-01T00:00:{n:02d}+00:00"))
                seeded.save(learned(f"own_{i}"))
                got = seeded.get("shared")  # None only when the resetter archived it in between
                assert got is None or got["name"] == "shared"
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(exc)

    def resetter() -> None:
        try:
            barrier.wait(timeout=10)
            for _round in range(5):
                resets.append(seeded.reset_demo())
                seeded.load_backup()
                seeded.delete("rehearsal")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait(timeout=10)
            for _round in range(12):
                for skill in seeded.all():
                    SKILL_VALIDATOR.validate(skill)
                seeded.summaries()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=saver, args=(i,), name=f"same-name-saver-{i}") for i in range(6)]
    threads.append(threading.Thread(target=resetter, name="resetter"))
    threads.append(threading.Thread(target=reader, name="reader"))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert peak == 1, f"{peak} threads were inside the file helpers at once"
    assert len(resets) == 5 and all(names == STARTERS for names in resets)
    for path in root.glob("*.json"):
        SKILL_VALIDATOR.validate(json.loads(path.read_text(encoding="utf-8")))
    assert not list(root.glob("*.tmp"))
    assert seeded.reset_demo() == STARTERS


# --- make_skill and routine_spec -----------------------------------------------------------------------


def test_make_skill_output_validates(seeded: Library):
    spec = CLEAR_TABLE["spec"]
    score = {
        "candidate": 3,
        "passed": True,
        "successes": 4,
        "trials": 5,
        "nominal_success": True,
        "collisions": 0,
        "limit_hits": 0,
        "ik_failures": 0,
        "mean_sim_s": 21.4,
        "fail_reason": None,
        "traj": "out/traj.json",
        "error": None,
    }
    skill = make_skill(spec, CLEAR_TABLE["code"], score, {"runs": 1, "successes": 1})
    SKILL_VALIDATOR.validate(skill)
    assert skill["name"] == "clear_table" and skill["trusted"] is False
    assert skill["sim"] == {"successes": 4, "trials": 5}
    assert skill["real"] == {"runs": 1, "successes": 1}
    assert skill["spec"] == spec and skill["spec"] is not spec
    assert skill["code"] == CLEAR_TABLE["code"]
    assert ISO_WITH_OFFSET.match(skill["created"])
    stamp = datetime.fromisoformat(skill["created"])
    assert stamp.utcoffset() is not None
    assert abs((datetime.now().astimezone() - stamp).total_seconds()) < 120
    assert seeded.save(skill) == seeded.root / "clear_table.json"

    trusted = Library.make_skill(spec, "def run(robot, args):\n    return True\n", None, None, trusted=True)
    assert trusted["trusted"] is True
    assert trusted["sim"] == {"successes": 0, "trials": 0} and trusted["real"] == {"runs": 0, "successes": 0}
    assert make_skill(spec, "x", {"successes": 2.0, "trials": 5}, None)["sim"] == {
        "successes": 2,
        "trials": 5,
    }

    for bad_call in (
        lambda: make_skill(spec, "", score, None),
        lambda: make_skill(spec, None, score, None),
        lambda: make_skill(spec, "x", {"successes": 1}, None),
        lambda: make_skill(spec, "x", {"successes": True, "trials": 1}, None),
        lambda: make_skill(spec, "x", {"successes": -1, "trials": 1}, None),
        lambda: make_skill(spec, "x", {"successes": 1.5, "trials": 1}, None),
        lambda: make_skill(spec, "x", 5, None),
        lambda: make_skill({**spec, "name": "_private"}, "x", None, None),
        lambda: make_skill({**spec, "english": ""}, "x", None, None),
        lambda: make_skill("not a spec", "x", None, None),
    ):
        with pytest.raises(LibraryError):
            bad_call()


def test_routine_spec_substitutes_args_everywhere(seeded: Library):
    seeded.save(learned("nested_demo", spec=NESTED_SPEC))
    steps = [
        {"skill": "hand_over", "args": {"object": "pill_bottle"}},
        {"skill": "nested_demo", "args": {"object": "juice_box", "zone": "tray", "unused": "ignored"}},
        {"skill": "place_in_tray", "args": {"object": "tea_tin"}},
    ]
    spec = routine_spec("good_morning", steps, seeded)
    TASK_SPEC_VALIDATOR.validate(spec)
    assert spec["name"] == "good_morning"
    assert spec["params"] == {}
    # the brief's literal formula: "Routine: " + "; then ".join(descriptions); english joined the same way
    step_specs = [seeded.get(step["skill"])["spec"] for step in steps]
    assert spec["description"] == "Routine: " + "; then ".join(s["description"] for s in step_specs)
    assert spec["english"] == "; then ".join(s["english"] for s in step_specs)
    assert spec["description"] == (
        "Routine: Pick up an object and hand it to the user.; then Nested demo.; "
        "then Put an object into the tray."
    )
    assert spec["english"] == (
        "The object leaves its spot and the user takes it from the gripper.; then Nested demo passes.; "
        "then The object ends inside the tray."
    )
    assert spec["success"] == {
        "all": [
            {"all": [{"moved": {"object": "pill_bottle", "min": 0.05}}, {"event": "handoff_done"}]},
            {
                "all": [
                    {"in_zone": {"object": "juice_box", "zone": "tray"}},
                    {
                        "any": [
                            {"near": {"object": "juice_box", "x": 0.2, "y": 0.0, "tol": 0.03}},
                            {"moved": {"object": "juice_box", "min": 0.05}},
                        ]
                    },
                    {"not": {"not_moved": {"object": "juice_box", "max": 0.02}}},
                    {"not_moved": {"object": "cereal_box", "max": 0.02}},
                    {"event": "handoff_done"},
                ]
            },
            {"in_zone": {"object": "tea_tin", "zone": "tray"}},
        ]
    }
    assert "$" not in json.dumps(spec)
    assert NESTED_SPEC["success"]["all"][0]["in_zone"]["object"] == "$object"  # the saved spec is untouched

    # the same spec from the method, and from a plain dict of skills
    assert seeded.routine_spec("good_morning", steps) == spec
    by_name = {s["name"]: s for s in seeded.all()}
    assert routine_spec("good_morning", steps, by_name) == spec

    # sim.success evaluates it without args, like any learned spec
    assert already_true(spec, SCENE, {}) is False
    after = json.loads(json.dumps(SCENE))
    after["objects"]["pill_bottle"]["seen"] = False
    # two of the scene's own tray slots, so the check follows the tray when the table is re-taped
    # (the literal (0.30, 0.22) and (0.28, 0.20) were inside the kit tray only)
    (jx, jy), (tx, ty) = SCENE["zones"]["tray"]["slots"][:2]
    after["objects"]["juice_box"].update({"x": jx, "y": jy})
    after["objects"]["tea_tin"].update({"x": tx, "y": ty})
    assert evaluate(spec, SCENE, after, {}, ["handoff_done"]) is True
    assert evaluate(spec, SCENE, after, {}, []) is False


def test_routine_spec_pairs_with_the_routine_candidate(seeded: Library):
    steps = json.loads((FIXTURES / "router_define_routine.json").read_text(encoding="utf-8"))["steps"]
    spec = routine_spec("good_morning", steps, seeded)
    TASK_SPEC_VALIDATOR.validate(spec)
    assert spec["success"] == {
        "all": [
            {"all": [{"moved": {"object": "pill_bottle", "min": 0.05}}, {"event": "handoff_done"}]},
            {"all": [{"moved": {"object": "juice_box", "min": 0.05}}, {"event": "handoff_done"}]},
        ]
    }
    code = routine_candidate(steps)
    assert check_source(code) == []
    assert 'robot.run_skill(step["skill"], step["args"])' in code
    skill = make_skill(spec, code, {"successes": 5, "trials": 5}, {"runs": 1, "successes": 1})
    path = seeded.save(skill)
    assert path == seeded.root / "good_morning.json"
    assert seeded.get("good_morning")["spec"] == spec
    assert {"name": "good_morning", "description": spec["description"], "params": {}} in seeded.summaries()
    # a single-step routine is still an `all` with one predicate
    single = routine_spec("just_pills", steps[:1], seeded)
    TASK_SPEC_VALIDATOR.validate(single)
    assert single["success"] == {"all": [spec["success"]["all"][0]]}
    assert single["description"] == "Routine: Pick up an object and hand it to the user."


def test_routine_spec_errors(seeded: Library):
    ok = {"skill": "hand_over", "args": {"object": "pill_bottle"}}
    with pytest.raises(UnknownSkill, match="fold_laundry") as info:
        routine_spec("morning", [ok, {"skill": "fold_laundry", "args": {}}], seeded)
    assert isinstance(info.value, KeyError) and "step 2" in str(info.value)
    with pytest.raises(KeyError, match="_starter"):
        routine_spec("morning", [{"skill": "_starter", "args": {}}], seeded)
    with pytest.raises(MissingArgument, match="object") as info:
        routine_spec("morning", [{"skill": "hand_over", "args": {}}], seeded)
    assert isinstance(info.value, KeyError)
    with pytest.raises(KeyError):
        routine_spec("morning", [{"skill": "hand_over"}], seeded)  # no args at all
    with pytest.raises(KeyError):
        routine_spec("morning", [{"skill": "hand_over", "args": {"objekt": "pill_bottle"}}], seeded)
    for bad_name in ("Bad Name", "_morning", "m", "", None):
        with pytest.raises(LibraryError):
            routine_spec(bad_name, [ok], seeded)
    with pytest.raises(LibraryError):
        routine_spec("morning", [], seeded)
    with pytest.raises(LibraryError):
        routine_spec("morning", "hand_over", seeded)
    with pytest.raises(LibraryError):
        routine_spec("morning", ["hand_over"], seeded)
    with pytest.raises(LibraryError):
        routine_spec("morning", [{"skill": "hand_over", "args": ["pill_bottle"]}], seeded)
    with pytest.raises(LibraryError):  # a number where the predicate needs an object name
        routine_spec("morning", [{"skill": "hand_over", "args": {"object": 7}}], seeded)
    assert seeded.get("morning") is None


# --- CLI --------------------------------------------------------------------------------------------------


def test_cli_list_reset_and_backup(root: Path, tmp_path: Path, capsys, monkeypatch):
    dotenv_calls: list[int] = []
    monkeypatch.setattr(lib, "load_dotenv", lambda *args, **kwargs: dotenv_calls.append(1))
    repo_archive = ROOT / "runs" / "archive"
    repo_archive_before = sorted(repo_archive.iterdir()) if repo_archive.is_dir() else []
    assert lib.main([]) == 2
    capsys.readouterr()
    assert dotenv_calls == []  # help only: nothing loaded

    assert lib.main(["--root", str(root), "--list"]) == 0
    out = capsys.readouterr().out
    assert dotenv_calls == [1]
    assert "seeded the starters: hand_over, place_in_tray" in out
    assert "hand_over" in out and "place_in_tray" in out and "yes" in out
    assert "2 skill(s) in" in out
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]

    library = Library(root)
    library.save(learned("clear_table"))
    write_json(root / "_backup" / "rehearsal.json", learned("rehearsal"))
    (root / "_backup" / "broken.json").write_text("{", encoding="utf-8")

    # with --root the archive lands beside that library, never in the repo's runs/archive
    assert lib.main(["--root", str(root), "--reset-demo"]) == 0
    assert capsys.readouterr().out.strip() == "reset demo, live skills: hand_over, place_in_tray"
    assert live_files(root) == ["hand_over.json", "place_in_tray.json"]
    archives = list((root / "_archive").iterdir())
    assert len(archives) == 1 and [p.name for p in archives[0].iterdir()] == ["clear_table.json"]
    assert (sorted(repo_archive.iterdir()) if repo_archive.is_dir() else []) == repo_archive_before

    # --archive-root overrides that default
    library.save(learned("again"))
    elsewhere = tmp_path / "elsewhere"
    assert lib.main(["--root", str(root), "--archive-root", str(elsewhere), "--reset-demo"]) == 0
    capsys.readouterr()
    assert [p.name for p in next(iter(elsewhere.iterdir())).iterdir()] == ["again.json"]
    assert len(list((root / "_archive").iterdir())) == 1

    assert lib.main(["--root", str(root), "--load-backup", "--list"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("loaded from backup: rehearsal\n")
    assert "seeded" not in out
    lines = out.strip().splitlines()
    assert lines[1].split() == ["name", "trusted", "real", "runs/ok", "sim", "ok/trials", "created"]
    assert lines[2].split()[:2] == ["hand_over", "yes"]
    assert lines[4].split()[:2] == ["rehearsal", "no"]
    assert lines[4].split()[2:4] == ["1/1", "5/5"]
    assert lines[-1] == f"3 skill(s) in {root}"
    assert dotenv_calls == [1, 1, 1, 1]  # once per real CLI run, through the patched hook


def test_format_listing_without_skills(tmp_path: Path):
    text = lib.format_listing([], tmp_path)
    assert text.splitlines()[0].split() == [
        "name",
        "trusted",
        "real",
        "runs/ok",
        "sim",
        "ok/trials",
        "created",
    ]
    assert text.splitlines()[-1] == f"0 skill(s) in {tmp_path}"


# --- save_backup and provenance ---------------------------------------------------------------------------


def provenance(root: Path) -> list[dict[str, Any]]:
    path = root / "_backup" / lib.PROVENANCE_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_save_backup_copies_resets_real_counts_and_records_provenance(seeded: Library, root: Path, tmp_path):
    source = tmp_path / "archive" / "20261017-194857" / "clear_table.json"
    write_json(source, learned("clear_table"))  # real 1/1: what the orchestrator writes after a fake replay
    path = seeded.save_backup(source, note="fake learn, fixture replies, sim 5/5, not a rehearsal")
    assert path == root / "_backup" / "clear_table.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    SKILL_VALIDATOR.validate(saved)
    expected = learned("clear_table")
    expected["real"] = {"runs": 0, "successes": 0}
    assert saved == expected  # trusted false and sim 5/5 kept, spec and code untouched, real reset
    assert list(saved) == ["name", "trusted", "spec", "code", "sim", "real", "created"]
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert not list((root / "_backup").glob("*.tmp"))

    lines = provenance(root)
    assert len(lines) == 1
    line = lines[0]
    assert list(line) == ["t", "name", "source", "note", "git_head", "keep_real", "sim", "real"]
    assert ISO_WITH_OFFSET.match(line["t"])
    assert line["name"] == "clear_table"
    assert line["note"] == "fake learn, fixture replies, sim 5/5, not a rehearsal"
    assert line["source"] == str(source.resolve())  # outside the repo: absolute
    assert line["keep_real"] is False
    assert line["sim"] == {"successes": 5, "trials": 5} and line["real"] == {"runs": 0, "successes": 0}
    assert line["git_head"] is None or re.fullmatch(r"[0-9a-f]{7,64}", line["git_head"])
    assert b"\r" not in (root / "_backup" / lib.PROVENANCE_NAME).read_bytes()

    # the backup is what Load backup restores after a reset: untrusted, zero real runs
    seeded.save(learned("clear_table"))
    assert seeded.reset_demo() == STARTERS
    assert seeded.load_backup() == ["clear_table"]
    restored = seeded.get("clear_table")
    assert restored["trusted"] is False and restored["real"] == {"runs": 0, "successes": 0}
    assert restored["sim"] == {"successes": 5, "trials": 5}


def test_save_backup_force_keep_real_name_and_repo_relative_source(seeded: Library, root: Path, tmp_path):
    source = tmp_path / "clear_table.json"
    write_json(source, learned("clear_table"))
    first = seeded.save_backup(source, note="first")
    before = first.read_bytes()
    with pytest.raises(FileExistsError, match="force"):
        seeded.save_backup(source, note="second")
    assert first.read_bytes() == before and len(provenance(root)) == 1

    # force replaces it, keep_real keeps the counts of a real rehearsal, and a second line is appended
    path = seeded.save_backup(source, note="real rehearsal, real 1/1", keep_real=True, force=True)
    assert path == first
    assert json.loads(path.read_text(encoding="utf-8"))["real"] == {"runs": 1, "successes": 1}
    lines = provenance(root)
    assert len(lines) == 2
    assert lines[1]["keep_real"] is True and lines[1]["real"] == {"runs": 1, "successes": 1}
    assert lines[1]["note"] == "real rehearsal, real 1/1"

    # name renames the file, the skill and its spec; a source inside the repo is recorded repo-relative
    renamed = seeded.save_backup(FIXTURES / "skill_clear_table.json", note="fixture", name="clear_table_two")
    assert renamed == root / "_backup" / "clear_table_two.json"
    skill = json.loads(renamed.read_text(encoding="utf-8"))
    SKILL_VALIDATOR.validate(skill)
    assert skill["name"] == "clear_table_two" == skill["spec"]["name"]
    assert skill["code"] == CLEAR_TABLE["code"]
    assert provenance(root)[2]["source"] == "tests/fixtures/skill_clear_table.json"
    assert provenance(root)[2]["name"] == "clear_table_two"
    assert seeded.load_backup() == ["clear_table", "clear_table_two"]
    assert seeded.get("clear_table_two")["spec"]["name"] == "clear_table_two"


def test_save_backup_refusals_leave_nothing_behind(seeded: Library, root: Path, tmp_path: Path):
    counter = iter(range(1, 100))

    def file_with(data: Any) -> Path:
        path = tmp_path / "src" / f"case_{next(counter)}.json"  # one file per case
        if isinstance(data, str):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
        else:
            write_json(path, data)
        return path

    no_code = learned("no_code")
    del no_code["code"]
    cases: list[tuple[Path, dict[str, Any], str]] = [
        (file_with(learned("rehearsal", trusted=True)), {}, "trusted"),
        (file_with(learned("rehearsal", trusted=True)), {"keep_real": True}, "trusted"),
        (file_with(learned("hand_over")), {}, "starter"),
        (file_with(learned("clear_table")), {"name": "place_in_tray"}, "starter"),
        (file_with(learned("clear_table")), {"name": "Bad Name"}, "invalid skill name"),
        (file_with(learned("clear_table")), {"name": "_hidden"}, "invalid skill name"),
        (file_with(learned("clear_table\n")), {}, "invalid skill name"),
        (file_with(no_code), {}, "skill invalid"),
        (file_with("{broken"), {}, "cannot read"),
        (tmp_path / "src" / "missing.json", {}, "no skill file"),
        (file_with(learned("clear_table")), {"note": ""}, "note"),
        (file_with(learned("clear_table")), {"note": "   "}, "note"),
    ]
    for source, kwargs, message in cases:
        with pytest.raises(LibraryError, match=message):
            seeded.save_backup(source, **{"note": "x", **kwargs})
    assert list((root / "_backup").iterdir()) == []
    assert provenance(root) == []


def test_git_head_is_a_hash_or_none(monkeypatch):
    head = lib.git_head()
    assert head is None or re.fullmatch(r"[0-9a-f]{7,64}", head)

    def missing(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(lib.subprocess, "run", missing)
    assert lib.git_head() is None

    class NotARepo:
        returncode = 128
        stdout = "fatal: not a git repository\n"

    monkeypatch.setattr(lib.subprocess, "run", lambda *args, **kwargs: NotARepo())
    assert lib.git_head() is None


def test_library_from_env_layouts(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VARA_SKILLS_DIR", raising=False)
    monkeypatch.delenv("VARA_RUNS_DIR", raising=False)
    default = lib.library_from_env()
    assert (default.root, default.starter_dir, default.backup_dir, default.archive_root) == (
        ROOT / "skills",
        ROOT / "skills" / "_starter",
        ROOT / "skills" / "_backup",
        ROOT / "runs" / "archive",
    )
    # --root: every folder under it, so a scratch library never touches the repo
    scratch = lib.library_from_env(tmp_path / "lib")
    assert (scratch.root, scratch.starter_dir, scratch.backup_dir, scratch.archive_root) == (
        tmp_path / "lib",
        tmp_path / "lib" / "_starter",
        tmp_path / "lib" / "_backup",
        tmp_path / "lib" / "_archive",
    )
    custom = lib.library_from_env(tmp_path / "lib", backup_dir=tmp_path / "b", archive_root=tmp_path / "a")
    assert (custom.backup_dir, custom.archive_root) == (tmp_path / "b", tmp_path / "a")

    # the orchestrator's layout: an isolated VARA_SKILLS_DIR without _backup/_starter uses the repo's
    isolated = tmp_path / "demo" / "skills"
    isolated.mkdir(parents=True)
    monkeypatch.setenv("VARA_SKILLS_DIR", str(isolated))
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "demo" / "runs"))
    env_library = lib.library_from_env()
    assert env_library.root == isolated
    assert env_library.starter_dir == ROOT / "skills" / "_starter"
    assert env_library.backup_dir == ROOT / "skills" / "_backup"
    assert env_library.archive_root == tmp_path / "demo" / "runs" / "archive"
    # what --save-backup uses: no fallback, so an isolated library never writes into the repo's backups
    writer = lib.library_from_env(backup_fallback=False)
    assert writer.backup_dir == isolated / "_backup" and writer.starter_dir == ROOT / "skills" / "_starter"
    (isolated / "_backup").mkdir()
    (isolated / "_starter").mkdir()
    own = lib.library_from_env()
    assert own.backup_dir == isolated / "_backup" and own.starter_dir == isolated / "_starter"
    assert lib.library_from_env(backup_dir=tmp_path / "elsewhere").backup_dir == tmp_path / "elsewhere"
    assert lib.library_from_env(archive_root=tmp_path / "arch").archive_root == tmp_path / "arch"


def test_cli_save_backup(root: Path, tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(lib, "load_dotenv", lambda *args, **kwargs: None)
    source = tmp_path / "learned" / "clear_table.json"
    write_json(source, learned("clear_table"))
    assert lib.main(["--root", str(root), "--save-backup", str(source), "--note", "fake learn"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"backed up: clear_table -> {root / '_backup' / 'clear_table.json'}"
    assert lines[1] == f"provenance: {root / '_backup' / lib.PROVENANCE_NAME}"
    saved = json.loads((root / "_backup" / "clear_table.json").read_text(encoding="utf-8"))
    assert saved["trusted"] is False and saved["real"] == {"runs": 0, "successes": 0}
    assert provenance(root)[0]["note"] == "fake learn"

    # a refusal exits 1 with the reason on stderr
    assert lib.main(["--root", str(root), "--save-backup", str(source), "--note", "again"]) == 1
    assert "exists" in capsys.readouterr().err
    starter = STARTER_DIR / "hand_over.json"
    assert lib.main(["--root", str(root), "--save-backup", str(starter), "--note", "x"]) == 1
    assert "trusted" in capsys.readouterr().err
    assert lib.main(["--root", str(root), "--save-backup", str(tmp_path / "nope.json"), "--note", "x"]) == 1
    assert "no skill file" in capsys.readouterr().err
    assert len(provenance(root)) == 1

    # misuse exits 2: no note, a blank note, backup flags without --save-backup
    for argv in (
        ["--root", str(root), "--save-backup", str(source)],
        ["--root", str(root), "--save-backup", str(source), "--note", " "],
        ["--root", str(root), "--list", "--note", "x"],
        ["--root", str(root), "--list", "--force"],
        ["--root", str(root), "--list", "--keep-real"],
        ["--root", str(root), "--list", "--name", "x"],
    ):
        with pytest.raises(SystemExit) as info:
            lib.main(argv)
        assert info.value.code == 2, argv
        capsys.readouterr()

    # --force, --keep-real, --name and --backup-dir; --load-backup honours --backup-dir too
    elsewhere = tmp_path / "elsewhere"
    argv = ["--root", str(root), "--backup-dir", str(elsewhere), "--save-backup", str(source)]
    argv += ["--note", "real rehearsal", "--keep-real", "--name", "clear_table_real"]
    assert lib.main(argv) == 0
    capsys.readouterr()
    saved = json.loads((elsewhere / "clear_table_real.json").read_text(encoding="utf-8"))
    assert saved["name"] == "clear_table_real" and saved["real"] == {"runs": 1, "successes": 1}
    assert (elsewhere / lib.PROVENANCE_NAME).is_file()
    assert lib.main(["--root", str(root), "--save-backup", str(source), "--note", "replaced", "--force"]) == 0
    capsys.readouterr()
    assert [line["note"] for line in provenance(root)] == ["fake learn", "replaced"]
    assert lib.main(["--root", str(root), "--backup-dir", str(elsewhere), "--load-backup", "--list"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("loaded from backup: clear_table_real\n")
    assert live_files(root) == ["clear_table_real.json"]
    assert out.strip().splitlines()[2].split()[:4] == ["clear_table_real", "no", "1/1", "5/5"]


def test_save_backup_by_live_skill_name(seeded: Library, root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a bare name resolves against the library, not the working directory
    seeded.save(learned("clear_table"))
    path = seeded.save_backup("clear_table", note="fake learn")
    assert path == root / "_backup" / "clear_table.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["real"] == {"runs": 0, "successes": 0} and saved["code"] == CLEAR_TABLE["code"]
    assert provenance(root)[0]["source"] == str((root / "clear_table.json").resolve())

    # an existing file of that name wins over the live skill
    write_json(tmp_path / "juice_box_to_tray", learned("juice_box_to_tray"))
    seeded.save(learned("juice_box_to_tray", sim={"successes": 4, "trials": 5}))
    seeded.save_backup("juice_box_to_tray", note="from the working directory")
    assert provenance(root)[1]["source"] == str((tmp_path / "juice_box_to_tray").resolve())
    assert provenance(root)[1]["sim"] == {"successes": 5, "trials": 5}

    # the live starter is trusted; a name with no live skill and a non-name say what is missing
    with pytest.raises(LibraryError, match="trusted"):
        seeded.save_backup("hand_over", note="x")
    with pytest.raises(LibraryError, match="no live skill 'tea_tin_to_tray' in"):
        seeded.save_backup("tea_tin_to_tray", note="x")
    with pytest.raises(LibraryError, match="^no skill file at Not A Name$"):
        seeded.save_backup("Not A Name", note="x")
    assert len(provenance(root)) == 2


def test_save_backup_rolls_back_when_the_provenance_line_fails(
    seeded: Library, root: Path, tmp_path: Path, monkeypatch
):
    source = tmp_path / "clear_table.json"
    write_json(source, learned("clear_table"))
    real_append = lib._append_line

    def full_disk(path: Path, data: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(lib, "_append_line", full_disk)
    with pytest.raises(OSError, match="No space"):
        seeded.save_backup(source, note="first")
    assert list((root / "_backup").iterdir()) == []  # no backup without its provenance line

    monkeypatch.setattr(lib, "_append_line", real_append)
    first = seeded.save_backup(source, note="first")
    before = first.read_bytes()
    monkeypatch.setattr(lib, "_append_line", full_disk)
    with pytest.raises(OSError, match="No space"):
        seeded.save_backup(source, note="second", keep_real=True, force=True)
    assert first.read_bytes() == before  # the replaced backup is restored byte for byte
    assert sorted(p.name for p in (root / "_backup").iterdir()) == ["clear_table.json", lib.PROVENANCE_NAME]
    assert [line["note"] for line in provenance(root)] == ["first"]


def test_cli_save_backup_by_name(root: Path, tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(lib, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    shutil.copy(STARTER_DIR / "hand_over.json", root / "hand_over.json")
    write_json(root / "clear_table.json", learned("clear_table"))
    assert lib.main(["--root", str(root), "--save-backup", "clear_table", "--note", "fake learn"]) == 0
    assert capsys.readouterr().out.startswith(f"backed up: clear_table -> {root / '_backup'}")
    assert lib.main(["--root", str(root), "--save-backup", "hand_over", "--note", "x"]) == 1
    assert "trusted" in capsys.readouterr().err
    assert lib.main(["--root", str(root), "--save-backup", "juice_box_to_tray", "--note", "x"]) == 1
    assert "no live skill 'juice_box_to_tray'" in capsys.readouterr().err
    assert [line["name"] for line in provenance(root)] == ["clear_table"]


def test_cli_save_backup_stays_under_vara_skills_dir(tmp_path: Path, capsys, monkeypatch):
    """With VARA_SKILLS_DIR isolating the library and no --root, --load-backup still reads the repo's
    skills/_backup while the isolated folder has none, but --save-backup writes only to
    <VARA_SKILLS_DIR>/_backup; --backup-dir is the explicit way to write anywhere else."""
    monkeypatch.setattr(lib, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(lib, "git_head", lambda: None)
    repo = tmp_path / "repo"  # stands in for the repo, so even a regression cannot write into the real one
    monkeypatch.setattr(lib, "ROOT", repo)
    shutil.copytree(STARTER_DIR, repo / "skills" / "_starter")
    write_json(repo / "skills" / "_backup" / "repo_backup.json", learned("repo_backup"))
    isolated = tmp_path / "demo" / "skills"
    isolated.mkdir(parents=True)
    monkeypatch.setenv("VARA_SKILLS_DIR", str(isolated))
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "demo" / "runs"))
    source = tmp_path / "learned" / "clear_table.json"
    write_json(source, learned("clear_table"))

    assert lib.main(["--load-backup"]) == 0  # reading falls back to the repo's backups
    assert capsys.readouterr().out.startswith("loaded from backup: repo_backup")
    assert live_files(isolated) == ["repo_backup.json"]

    assert lib.main(["--save-backup", str(source), "--note", "fake learn"]) == 0
    target = isolated / "_backup" / "clear_table.json"
    assert capsys.readouterr().out.startswith(f"backed up: clear_table -> {target}")
    written = sorted(p.name for p in (isolated / "_backup").iterdir())
    assert written == ["clear_table.json", lib.PROVENANCE_NAME]
    assert sorted(p.name for p in (repo / "skills" / "_backup").iterdir()) == ["repo_backup.json"]
    assert lib.main(["--load-backup"]) == 0  # the isolated folder now exists, so it is the one read
    assert capsys.readouterr().out.startswith("loaded from backup: clear_table\n")

    explicit = ["--backup-dir", str(repo / "skills" / "_backup"), "--save-backup", str(source)]
    assert lib.main([*explicit, "--note", "into the repo on purpose"]) == 0
    capsys.readouterr()
    assert (repo / "skills" / "_backup" / "clear_table.json").is_file()
    assert (repo / "skills" / "_backup" / lib.PROVENANCE_NAME).is_file()


def test_repo_backup_folder_is_declared():
    """The checked-in skills/_backup: untrusted, valid skills, each with a provenance line saying what it
    is; a backup that did not keep its real counts shows zero real runs."""
    folder = ROOT / "skills" / "_backup"
    assert (folder / ".gitkeep").is_file()  # the folder exists in every clone
    record = folder / lib.PROVENANCE_NAME
    lines = (
        [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line.strip()]
        if record.is_file()
        else []
    )
    for path in sorted(folder.glob("*.json")):
        skill = json.loads(path.read_text(encoding="utf-8"))
        SKILL_VALIDATOR.validate(skill)
        assert skill["name"] == path.stem and skill["trusted"] is False
        assert path.stem not in STARTERS
        mine = [line for line in lines if line["name"] == path.stem]
        assert mine, f"{path.name} has no line in {record.name}"
        last = mine[-1]
        assert last["note"].strip() and ISO_WITH_OFFSET.match(last["t"])
        assert last["sim"] == skill["sim"] and last["real"] == skill["real"]
        if not last["keep_real"]:
            assert skill["real"] == {"runs": 0, "successes": 0}
