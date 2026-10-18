"""docs/runbook.md cannot rot: every repo command it shows exists and accepts every flag it uses.

Each command line that runs the venv's python on a repo module (`-m arm.driver`) or script
(`scripts/touch_markers.py`) is parsed; its `--help` is produced once per target in-process through the
target's `main(["--help"])` (dotenv patched out, so no .env is read) and every `--flag` on the line must
appear in it. Nothing is executed without `--help`. The runbook's relative links must point at files.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import re
import shlex
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "runbook.md"
PYTHON = re.compile(r"(?:\.\\\.venv\\Scripts\\python\.exe|\.venv/Scripts/python\.exe)\s+(.*)$")
EXTERNAL = {"mujoco.viewer", "pytest"}  # not ours: their flags are not checked


def commands() -> list[tuple[str, list[str]]]:
    """`(target, args)` for every repo command line in the runbook: target `-m pkg.mod` or a script path."""
    found = []
    for line in RUNBOOK.read_text(encoding="utf-8").splitlines():
        match = PYTHON.search(line.strip())
        if not match:
            continue
        args = shlex.split(match.group(1), posix=True)
        if args[0] == "-m":
            if args[1] in EXTERNAL:
                continue
            found.append((args[1], args[2:]))
        elif args[0].startswith("scripts/") and args[0].endswith(".py"):
            found.append((args[0], args[1:]))
        else:
            raise AssertionError(f"unrecognised runbook command: {line.strip()}")
    return found


@cache
def help_text(target: str) -> str:
    """The `--help` output of a repo module or script, produced in-process."""
    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        if target.endswith(".py"):
            name = f"runbook_{Path(target).stem}"
            spec = importlib.util.spec_from_file_location(name, ROOT / target)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # dataclasses look their module up while the script loads
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
            module.main(["--help"])
        assert exc.value.code in (0, None), f"{target} --help exited {exc.value.code}"
        return out.getvalue()
    finally:
        dotenv.load_dotenv = original


def test_the_runbook_shows_commands():
    targets = {target for target, _ in commands()}
    assert len(commands()) >= 25
    for needed in (
        "-m arm.driver",
        "scripts/touch_markers.py",
        "-m vision.calibrate",
        "scripts/live_gates.py",
    ):
        assert needed.removeprefix("-m ") in targets, needed


@pytest.mark.parametrize("target,args", commands(), ids=lambda v: v if isinstance(v, str) else " ".join(v))
def test_every_runbook_command_exists_with_its_flags(target, args):
    text = help_text(target)
    flags = {arg.split("=", 1)[0] for arg in args if arg.startswith("--")}
    missing = sorted(flag for flag in flags if flag not in text)
    assert not missing, f"{target}: flags not in --help: {missing}"


def test_the_runbook_links_point_at_files():
    text = RUNBOOK.read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if link.startswith("http"):
            continue
        assert (RUNBOOK.parent / link).resolve().exists(), link


def test_the_runbook_follows_the_prose_style():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "—" not in text  # no em-dashes
    assert "\r" not in text
