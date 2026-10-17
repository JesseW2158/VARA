"""The restricted namespace AI-written code runs in (candidates, saved skills, reward functions).

This is the second of the three layers around AI-written code: `sim.candidate_check.check_source` filters
the source, this module gives the accepted source a namespace with a trimmed `builtins` and only the `math`
module, and `sim.run_candidate` (a Daytona sandbox or a clean local subprocess) is the security boundary.
Nothing here is a sandbox on its own: never call `load_entry` in the orchestrator process.

`load_entry(src, entry="run")` runs `check_source`, compiles the source, executes it in a fresh namespace
and returns the entry function. A candidate's top-level `import math` calls `builtins.__import__`, so the
namespace carries a restricted `__import__` that serves `math` and nothing else. `print` goes to stderr so
the score line on stdout stays clean.
"""

from __future__ import annotations

import builtins
import math
import sys
from collections.abc import Callable
from typing import Any

from sim.candidate_check import check_source

__all__ = ["REMOVED_BUILTINS", "SAFE_BUILTINS", "load_entry", "namespace", "safe_builtins"]

# Names removed from the builtins a candidate sees. Most are also rejected statically by check_source;
# the rest (memoryview, classmethod, staticmethod, property) have no use inside a candidate.
REMOVED_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "help",
        "dir",
        "type",
        "object",
        "super",
        "memoryview",
        "classmethod",
        "staticmethod",
        "property",
        "__build_class__",
        "exit",
        "quit",
        "copyright",
        "credits",
        "license",
    }
)


def _import_math(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0):
    """The only import a candidate may perform: `import math`."""
    if name == "math" and level == 0 and not fromlist:
        return math
    raise ImportError(f"import of {name!r} is not allowed; only 'import math' is available")


def _print_to_stderr(*args: Any, **kwargs: Any) -> None:
    """`print` for candidates: stdout belongs to the score line, so candidate output goes to stderr."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def safe_builtins() -> dict[str, Any]:
    """A fresh copy of the trimmed builtins (callers may mutate their copy)."""
    allowed = {name: value for name, value in vars(builtins).items() if name not in REMOVED_BUILTINS}
    allowed["__import__"] = _import_math
    allowed["print"] = _print_to_stderr
    return allowed


SAFE_BUILTINS: dict[str, Any] = safe_builtins()


def namespace() -> dict[str, Any]:
    """A fresh globals dict for one candidate: trimmed builtins plus `math`."""
    return {"__builtins__": safe_builtins(), "__name__": "candidate", "math": math}


def load_entry(src: str, entry: str = "run", filename: str = "<candidate>") -> Callable[..., Any]:
    """Check, compile and execute `src`, then return its `entry` function.

    Raises `ValueError` listing the problems when `check_source` rejects the source, and re-raises
    whatever the candidate's top level raises. Only `sim.run_candidate` (sandbox or clean subprocess)
    and `SimRobot.run_skill` inside it may call this.
    """
    problems = check_source(src, entry=entry)
    if problems:
        raise ValueError("rejected: " + "; ".join(problems))
    code = compile(src, filename, "exec")
    ns = namespace()
    exec(code, ns)  # noqa: S102 - this is the one place AI code is executed, inside the runner process
    fn = ns.get(entry)
    if not callable(fn):
        raise ValueError(f"rejected: no callable {entry!r} after executing the source")
    return fn
