"""The skill library: one `skills/<name>.json` per saved skill, behind one class (C9, C10).

Live skills are the `*.json` files directly under `skills/`. `skills/_starter/` holds the two hand-written,
trusted skills (`hand_over`, `place_in_tray`) and `skills/_backup/` the rehearsal skills the operator loads
when a judge group should see a fuller library. Every file that crosses this module is validated against
`schemas/skill.json` (loaded once at import). A malformed live file is skipped with a logged warning, never
raised, so the dashboard keeps rendering; the warning repeats only when the file changes. Every file
operation runs under one re-entrant lock, because the orchestrator calls the library from request handlers
and from its worker thread.

- `Library(root=None, *, starter_dir=None, backup_dir=None, archive_root=None)`: `root` defaults to
  `skills/`, the others to `root/_starter`, `root/_backup` and `runs/archive`.
- `ensure_seeded()`: copies the starters in when `root` holds no `*.json` (called at orchestrator start,
  so a fresh clone works); returns the names copied.
- `all()` (validated, sorted by `created` then name), `get(name)`, `summaries()` (`{"name",
  "description", "params"}` for the router), `starter_names()`, `path(name)`.
- `save(skill) -> Path`: validates first, refusing a name that starts with `_` or is not exactly the
  schema pattern (`fullmatch`: a trailing newline is refused although the schema's own `pattern` lets it
  through), and an untrusted skill that would replace a trusted one (a learned skill never shadows
  `hand_over` or `place_in_tray`); then writes `root/<name>.json` atomically: an exclusive temp file
  beside it (at most `_TEMP_TRIES` names, so a directory that refuses writes raises `PermissionError`
  at once instead of spinning inside `tempfile.mkstemp`), fsync, `os.replace`, UTF-8, LF, 2-space JSON
  with a trailing newline.
- `delete(name) -> bool`: removes a live skill; never a starter.
- `reset_demo() -> list[str]`: moves every non-starter live file to `runs/archive/<UTC YYYYmmdd-HHMMSS>/`
  (retrying briefly while a reader holds a file open), copies the starters in (overwriting) and returns
  the live names sorted: exactly the two starters.
- `load_backup() -> list[str]`: copies every valid `_backup/*.json` into `root` as an untrusted skill
  (`trusted: true` marks the hand-written starters only, so a backup claiming it is loaded untrusted with
  a warning); an invalid file or a backup named like a trusted live skill is skipped with a warning.
  Returns the names copied.
- `make_skill(spec, code, sim, real, trusted=False) -> dict`: a validated skill with `created` in ISO 8601
  carrying the local UTC offset; `sim` and `real` may be any mapping holding the counts (a score dict
  works), or None for zeros.
- `routine_spec(name, steps, library) -> dict`: the task spec of a routine: `success` is `all` of the
  step skills' success predicates with every `"$x"` string replaced by the step's `args["x"]`, `params`
  `{}`, `description` `"Routine: " + "; then ".join(step descriptions)` and `english` the step
  sentences joined with `"; then "` (the brief's literal formula; the parts are not reworded).
  Raises `UnknownSkill` or `MissingArgument` (both `KeyError`) and `LibraryError` (a `ValueError`) for a
  bad name, empty steps or a result that does not validate against `schemas/task_spec.json`. `library`
  is anything with `get(name) -> dict | None`, a `Library` or a plain dict of skills. Both helpers also
  exist on the class (`Library.make_skill(...)`, `library.routine_spec(name, steps)`).
- `python -m agents.library --list | --reset-demo | --load-backup [--root DIR] [--archive-root DIR]`:
  loads `.env` by convention and reads nothing from it; no network. `--list` seeds the starters first
  when the library is empty, as the orchestrator does at startup. With `--root` the archive defaults to
  `<root>/_archive`, so a scratch library never archives into the repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import jsonschema
from dotenv import load_dotenv
from jsonschema.exceptions import best_match

__all__ = [
    "Library",
    "LibraryError",
    "MissingArgument",
    "NAME_RE",
    "ROOT",
    "ROUTINE_PREFIX",
    "SKILL_SCHEMA",
    "SKILL_VALIDATOR",
    "STEP_JOINER",
    "TASK_SPEC_SCHEMA",
    "TASK_SPEC_VALIDATOR",
    "UnknownSkill",
    "format_listing",
    "main",
    "make_skill",
    "routine_spec",
    "validate_skill",
    "validate_task_spec",
]

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = ROOT / "schemas"
SKILL_SCHEMA: dict[str, Any] = json.loads((SCHEMAS_DIR / "skill.json").read_text(encoding="utf-8"))
TASK_SPEC_SCHEMA: dict[str, Any] = json.loads((SCHEMAS_DIR / "task_spec.json").read_text(encoding="utf-8"))
SKILL_VALIDATOR = jsonschema.Draft202012Validator(SKILL_SCHEMA)
TASK_SPEC_VALIDATOR = jsonschema.Draft202012Validator(TASK_SPEC_SCHEMA)
NAME_RE = re.compile(SKILL_SCHEMA["properties"]["name"]["pattern"])
ROUTINE_PREFIX = "Routine: "
STEP_JOINER = "; then "
ARCHIVE_STAMP = "%Y%m%d-%H%M%S"
_REPLACE_TRIES = 5  # replace/move retries on Windows while a reader briefly holds a file open
_TEMP_TRIES = 8  # temp-file name attempts; bounded so a write-denied directory fails loud (_open_temp)
_TEMP_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOINHERIT", 0)
)

logger = logging.getLogger(__name__)


class LibraryError(ValueError):
    """A skill the library refuses: invalid against its schema, a bad name, or an untrusted skill that
    would replace a trusted one."""


class UnknownSkill(KeyError):
    """A routine step names a skill that is not in the library."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class MissingArgument(KeyError):
    """A step skill's success predicate refers to `$x` but the step's args carry no `x`."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class SkillSource(Protocol):
    """What `routine_spec` needs from a library: `get(name)` returning the skill dict or None."""

    def get(self, name: str, /) -> Mapping[str, Any] | None: ...


# --- validation and formatting -------------------------------------------------------------------------


def validate_skill(skill: Any) -> None:
    """Raise `LibraryError` unless `skill` matches schemas/skill.json."""
    _validate(SKILL_VALIDATOR, skill, "skill")


def validate_task_spec(spec: Any) -> None:
    """Raise `LibraryError` unless `spec` matches schemas/task_spec.json."""
    _validate(TASK_SPEC_VALIDATOR, spec, "task spec")


def _validate(validator: jsonschema.Draft202012Validator, data: Any, what: str) -> None:
    error = best_match(validator.iter_errors(data))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise LibraryError(f"{what} invalid at {where}: {error.message}")


def _valid_name(name: Any) -> bool:
    # fullmatch: `re.match` and jsonschema's `pattern` accept "name\n" (`$` matches before a final newline)
    return isinstance(name, str) and not name.startswith("_") and NAME_RE.fullmatch(name) is not None


def _check_name(name: Any) -> str:
    if not _valid_name(name):
        raise LibraryError(
            f"invalid skill name {name!r}: it must match {NAME_RE.pattern} and may not start with '_'"
        )
    return name


def _dump(data: Mapping[str, Any]) -> str:
    """The on-disk form: 2-space JSON, non-ASCII kept as is, no NaN, one trailing newline."""
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _retry(operation: Callable[[], object]) -> None:
    """Run `operation`, retrying briefly on `PermissionError`: Windows refuses to replace or move a file
    while another handle (a dashboard poll, a sandbox upload) holds it open for a few milliseconds."""
    for attempt in range(_REPLACE_TRIES):
        try:
            operation()
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _replace(src: Path, dst: Path) -> None:
    """`os.replace` with retries (see `_retry`)."""
    _retry(lambda: os.replace(src, dst))


def _move(src: Path, dst: Path) -> None:
    """`shutil.move` with retries (see `_retry`); unlike `os.replace` it also crosses volumes."""
    _retry(lambda: shutil.move(str(src), str(dst)))


def _open_temp(directory: Path, stem: str) -> tuple[int, Path]:
    """Create `.<stem>-<8 hex>.tmp` exclusively in `directory`; returns the descriptor and the path.

    Bounded on purpose: `tempfile.mkstemp` reads a `PermissionError` on Windows as a name collision with
    a directory and retries up to `TMP_MAX` (2**31) times, so a root whose ACL denies writes made
    `save()` and `ensure_seeded()` spin for minutes. Here the error surfaces at once.
    """
    for _ in range(_TEMP_TRIES):
        tmp = directory / f".{stem}-{os.urandom(4).hex()}.tmp"
        try:
            return os.open(tmp, _TEMP_FLAGS, 0o600), tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"no free temp file name in {directory} after {_TEMP_TRIES} tries")


def _now_iso() -> str:
    """ISO 8601 to the second with the local UTC offset, like `2026-10-17T08:29:40-04:00`."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _created_key(skill: Mapping[str, Any]) -> tuple[float, str]:
    """Sort key for `all()`: the `created` instant (unparsable last), then the name."""
    name = str(skill.get("name", ""))
    try:
        stamp = datetime.fromisoformat(str(skill.get("created", "")))
    except ValueError:
        return (float("inf"), name)
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    return (stamp.timestamp(), name)


def _counts(data: Any, keys: tuple[str, str], what: str) -> dict[str, int]:
    """`{key: int}` for the two count keys of a `sim`/`real` block; None means zeros."""
    if data is None:
        return dict.fromkeys(keys, 0)
    if not isinstance(data, Mapping):
        raise LibraryError(f"{what} must be a mapping with {keys[0]!r} and {keys[1]!r}, got {data!r}")
    out: dict[str, int] = {}
    for key in keys:
        if key not in data:
            raise LibraryError(f"{what} lacks {key!r}")
        value = data[key]
        integral = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        if isinstance(value, bool) or not integral or value < 0:
            raise LibraryError(f"{what}[{key!r}] must be a non-negative integer, got {value!r}")
        out[key] = int(value)
    return out


def _json_copy(data: Mapping[str, Any], what: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(data), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise LibraryError(f"{what} is not plain JSON: {exc}") from None


# --- skill and routine construction ---------------------------------------------------------------------


def make_skill(
    spec: Mapping[str, Any],
    code: str,
    sim: Mapping[str, Any] | None,
    real: Mapping[str, Any] | None,
    trusted: bool = False,
) -> dict[str, Any]:
    """A skill dict for `Library.save`, validated against schemas/skill.json.

    `sim` needs `successes` and `trials`, `real` needs `runs` and `successes`; extra keys (a whole score
    dict) are ignored and None means zeros. `created` is now, ISO 8601 with the local UTC offset.
    """
    if not isinstance(spec, Mapping):
        raise LibraryError(f"spec must be a task spec mapping, got {spec!r}")
    skill = {
        "name": _check_name(spec.get("name")),
        "trusted": bool(trusted),
        "spec": _json_copy(spec, "spec"),
        "code": code,
        "sim": _counts(sim, ("successes", "trials"), "sim"),
        "real": _counts(real, ("runs", "successes"), "real"),
        "created": _now_iso(),
    }
    validate_skill(skill)
    return skill


def routine_spec(
    name: str, steps: Sequence[Mapping[str, Any]], library: SkillSource | Mapping[str, Any]
) -> dict[str, Any]:
    """The task spec of a routine (see the module docstring); validates against schemas/task_spec.json."""
    routine_name = _check_name(name)
    step_list = list(steps) if isinstance(steps, Sequence) and not isinstance(steps, str) else None
    if not step_list:
        raise LibraryError("a routine needs a list of at least one step")
    predicates: list[Any] = []
    descriptions: list[str] = []
    sentences: list[str] = []
    for index, step in enumerate(step_list, 1):
        if not isinstance(step, Mapping) or not isinstance(step.get("skill"), str):
            raise LibraryError(f"routine step {index} must be an object with a skill name, got {step!r}")
        skill_name = step["skill"]
        args = step.get("args")
        args = {} if args is None else args
        if not isinstance(args, Mapping):
            raise LibraryError(f"routine step {index} ({skill_name}) args must be an object, got {args!r}")
        skill = library.get(skill_name) if _valid_name(skill_name) else None
        if skill is None:
            raise UnknownSkill(f"routine step {index} uses an unknown skill {skill_name!r}")
        spec = skill["spec"]
        where = f"routine step {index} ({skill_name})"
        predicates.append(_substitute(spec["success"], args, where))
        descriptions.append(str(spec["description"]))
        sentences.append(str(spec["english"]))
    result = {
        "name": routine_name,
        "description": ROUTINE_PREFIX + STEP_JOINER.join(descriptions),
        "params": {},
        "success": {"all": predicates},
        "english": STEP_JOINER.join(sentences),
    }
    validate_task_spec(result)
    return result


def _substitute(node: Any, args: Mapping[str, Any], where: str) -> Any:
    """A deep copy of a predicate tree with every string `"$x"` replaced by `args["x"]`."""
    if isinstance(node, str):
        if not node.startswith("$"):
            return node
        key = node[1:]
        if key not in args:
            raise MissingArgument(f"{where} refers to {node!r} but its args carry no {key!r}")
        return args[key]
    if isinstance(node, Mapping):
        return {str(key): _substitute(value, args, where) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_substitute(value, args, where) for value in node]
    return node


# --- the library --------------------------------------------------------------------------------------------


class Library:
    """The live skills under `root` plus the starter, backup and archive folders (module docstring)."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        starter_dir: str | Path | None = None,
        backup_dir: str | Path | None = None,
        archive_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else ROOT / "skills"
        self.starter_dir = Path(starter_dir) if starter_dir is not None else self.root / "_starter"
        self.backup_dir = Path(backup_dir) if backup_dir is not None else self.root / "_backup"
        self.archive_root = Path(archive_root) if archive_root is not None else ROOT / "runs" / "archive"
        self.lock = threading.RLock()
        self._warned: dict[Path, int] = {}  # malformed file -> mtime_ns already reported

    def __repr__(self) -> str:
        return f"Library({str(self.root)!r})"

    # --- reading ---

    def path(self, name: str) -> Path:
        """`root/<name>.json` for a valid name (no existence check); `LibraryError` otherwise."""
        return self.root / f"{_check_name(name)}.json"

    def starter_names(self) -> list[str]:
        """The names of the skills in `starter_dir`, sorted."""
        return sorted(path.stem for path in _json_files(self.starter_dir))

    def all(self) -> list[dict[str, Any]]:
        """Every valid live skill, sorted by `created` then name; malformed files are skipped (warning)."""
        with self.lock:
            skills = [skill for path in _json_files(self.root) if (skill := self._load(path)) is not None]
        skills.sort(key=_created_key)
        return skills

    def get(self, name: str) -> dict[str, Any] | None:
        """The live skill `name`, or None when it is missing, malformed or the name is invalid."""
        if not _valid_name(name):
            return None
        with self.lock:
            path = self.root / f"{name}.json"
            return self._load(path) if path.is_file() else None

    def summaries(self) -> list[dict[str, Any]]:
        """`{"name", "description", "params"}` per live skill, what the router gets."""
        return [
            {
                "name": skill["name"],
                "description": skill["spec"]["description"],
                "params": dict(skill["spec"]["params"]),
            }
            for skill in self.all()
        ]

    # --- writing ---

    def ensure_seeded(self) -> list[str]:
        """Copy the starters in when `root` holds no `*.json` at all; returns the names copied."""
        with self.lock:
            if _json_files(self.root):
                return []
            return self._copy_starters()

    def save(self, skill: Mapping[str, Any]) -> Path:
        """Validate `skill` and write `root/<name>.json` atomically; returns that path.

        Raises `LibraryError` for an invalid skill, a name starting with `_` or outside the schema pattern,
        or an untrusted skill that would replace a trusted one (a starter or a trusted live skill).
        """
        if not isinstance(skill, Mapping):
            raise LibraryError(f"a skill is a mapping, got {skill!r}")
        name = _check_name(skill.get("name"))
        validate_skill(skill)
        path = self.root / f"{name}.json"
        with self.lock:
            self._check_shadow(name, bool(skill["trusted"]), path)
            self._write(path, skill)
        return path

    def delete(self, name: str) -> bool:
        """Remove the live skill `name`; False when it is absent, invalid or a starter."""
        if not _valid_name(name) or name in self.starter_names():
            return False
        path = self.root / f"{name}.json"
        with self.lock:
            if not path.is_file():
                return False
            path.unlink()
            self._warned.pop(path, None)
        return True

    def reset_demo(self) -> list[str]:
        """Archive every non-starter live file, restore the starters, return the live names sorted."""
        with self.lock:
            starters = set(self.starter_names())
            learned = [path for path in _json_files(self.root) if path.stem not in starters]
            if learned:
                archive = self._new_archive_dir()
                for path in learned:
                    _move(path, archive / path.name)
                    self._warned.pop(path, None)
                logger.info("reset demo: archived %d skill file(s) to %s", len(learned), archive)
            self._copy_starters()
            return sorted(skill["name"] for skill in self.all())

    def load_backup(self) -> list[str]:
        """Copy every valid `backup_dir/*.json` into `root` as an untrusted skill; returns the names copied.

        `trusted: true` belongs to the hand-written starters only (contract), so a backup claiming it is
        loaded untrusted with a warning, and one named like a starter or a trusted live skill is skipped.
        """
        copied: list[str] = []
        with self.lock:
            for source in _json_files(self.backup_dir):
                skill = self._load(source)
                if skill is None:
                    continue
                if skill["trusted"]:
                    logger.warning("backup %s claims trusted: true; loading it as untrusted", source)
                    skill = {**skill, "trusted": False}
                target = self.root / source.name
                try:
                    self._check_shadow(skill["name"], False, target)
                except LibraryError as exc:
                    logger.warning("skipping backup %s: %s", source, exc)
                    continue
                self._write(target, skill)
                copied.append(skill["name"])
        return copied

    # --- construction helpers, also usable as module functions ---

    @staticmethod
    def make_skill(
        spec: Mapping[str, Any],
        code: str,
        sim: Mapping[str, Any] | None,
        real: Mapping[str, Any] | None,
        trusted: bool = False,
    ) -> dict[str, Any]:
        """See the module function `make_skill`."""
        return make_skill(spec, code, sim, real, trusted=trusted)

    def routine_spec(
        self,
        name: str,
        steps: Sequence[Mapping[str, Any]],
        library: SkillSource | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """See the module function `routine_spec`; `library` defaults to this one."""
        return routine_spec(name, steps, self if library is None else library)

    # --- internals ---

    def _load(self, path: Path) -> dict[str, Any] | None:
        """Parse and validate one file; None (with one warning per file version) when it is malformed."""
        try:
            skill = json.loads(path.read_text(encoding="utf-8-sig"))
            validate_skill(skill)
            if skill["name"] != path.stem:
                raise LibraryError(f"file name {path.name} does not match the skill name {skill['name']!r}")
            _check_name(skill["name"])  # the schema pattern alone would let "name\n" through
        except (OSError, ValueError) as exc:  # JSONDecodeError, UnicodeDecodeError and LibraryError included
            self._warn(path, exc)
            return None
        self._warned.pop(path, None)
        return skill

    def _warn(self, path: Path, exc: Exception) -> None:
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = -1
        if self._warned.get(path) != stamp:
            self._warned[path] = stamp
            logger.warning("skipping malformed skill file %s: %s", path, exc)

    def _check_shadow(self, name: str, trusted: bool, path: Path) -> None:
        if trusted:
            return
        if name in self.starter_names():
            raise LibraryError(f"{name!r} is a trusted starter skill; a learned skill may not replace it")
        if path.is_file():
            existing = self._load(path)
            if existing is not None and existing.get("trusted"):
                raise LibraryError(f"{name!r} is a trusted skill; an untrusted skill may not replace it")

    def _copy_starters(self) -> list[str]:
        names: list[str] = []
        for source in _json_files(self.starter_dir):
            skill = self._load(source)
            if skill is None:
                continue
            self._write(self.root / source.name, skill)
            names.append(skill["name"])
        return names

    def _write(self, path: Path, data: Mapping[str, Any]) -> None:
        """Atomic write: dump first, then an exclusive temp file beside `path`, fsync, `os.replace`."""
        text = _dump(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _open_temp(path.parent, path.stem)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            _replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._warned.pop(path, None)

    def _new_archive_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime(ARCHIVE_STAMP)
        for n in range(1000):
            candidate = self.archive_root / (stamp if n == 0 else f"{stamp}-{n}")
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise OSError(f"could not create an archive directory under {self.archive_root}")


def _json_files(folder: Path) -> list[Path]:
    """The `*.json` files directly in `folder` (no subdirectories), sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.glob("*.json") if path.is_file())


# --- CLI ----------------------------------------------------------------------------------------------


def format_listing(skills: Sequence[Mapping[str, Any]], root: Path) -> str:
    """One row per skill: name, trusted, real runs/successes, sim successes/trials, created."""
    rows = [("name", "trusted", "real runs/ok", "sim ok/trials", "created")]
    for skill in skills:
        rows.append(
            (
                str(skill["name"]),
                "yes" if skill["trusted"] else "no",
                f"{skill['real']['runs']}/{skill['real']['successes']}",
                f"{skill['sim']['successes']}/{skill['sim']['trials']}",
                str(skill["created"]),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows
    ]
    lines.append(f"{len(skills)} skill(s) in {root}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agents.library --list | --reset-demo | --load-backup [--root DIR] [--archive-root DIR]`."""
    parser = argparse.ArgumentParser(prog="python -m agents.library", description="the saved-skill library")
    parser.add_argument("--root", default=None, help="skills directory (default: skills/ in the repo)")
    parser.add_argument(
        "--archive-root",
        default=None,
        help="archive dir for --reset-demo (default: runs/archive, or <root>/_archive with --root)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every live skill (seeding the two starters first when the library is empty)",
    )
    parser.add_argument(
        "--reset-demo", action="store_true", help="archive learned skills and restore exactly the starters"
    )
    parser.add_argument(
        "--load-backup", action="store_true", help="copy skills/_backup/*.json into the library"
    )
    args = parser.parse_args(argv)
    if not (args.list or args.reset_demo or args.load_backup):
        parser.print_help()
        return 2
    load_dotenv()  # the convention for CLIs; nothing here reads a key
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    archive_root = args.archive_root
    if archive_root is None and args.root is not None:
        archive_root = Path(args.root) / "_archive"  # a scratch library never archives into the repo
    library = Library(args.root, archive_root=archive_root)
    if args.reset_demo:
        names = library.reset_demo()
        print("reset demo, live skills: " + (", ".join(names) or "(none)"))
    if args.load_backup:
        names = library.load_backup()
        print("loaded from backup: " + (", ".join(names) or "(nothing valid to load)"))
    if args.list:
        seeded = library.ensure_seeded()
        if seeded:
            print("seeded the starters: " + ", ".join(seeded))
        print(format_listing(library.all(), library.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
