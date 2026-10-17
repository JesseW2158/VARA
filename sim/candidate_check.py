"""Static check of AI-written candidate programs (docs/interfaces.md, "Candidate program").

`check_source(src, entry="run")` returns every problem it finds as a non-empty string; an empty list
means the source is accepted. The orchestrator runs it before scoring and `sim.run_candidate` runs it
again inside the sandbox, so this module depends on the standard library only.

The check is a filter that keeps candidates inside the Robot API, not the security boundary: AI-written
code still runs only inside `sim.run_candidate` in a Daytona sandbox or a clean local subprocess.

Rules (mirrored by tests/contract/test_candidate_check.py and tests/test_candidate_check_extra.py):
- at most 200 lines, and the file must parse (a SyntaxError is reported as one problem);
- top level: a docstring, `import math` (only that exact form), `def`, and `NAME = <literal>`;
- the entry function exists at the top level with its exact arity: `run(robot, args)` or
  `reward(obs, action, prev_action, info)`;
- no import but `import math` anywhere;
- no identifier starting with `_`, and none equal to a forbidden name (`open`, `exec`, `eval`, ...,
  `type`, `object`, `super`, plus `mro`, because `Exception.mro()[1]` is BaseException). An identifier
  here is a name, an attribute, a parameter, a keyword argument, a function name, an `except ... as`
  name, a match-pattern capture, a match-class keyword attribute (`case C(x=1)` reads attribute `x`) or
  a PEP 695 type parameter (`def f[T](x)`);
- no class, global, nonlocal, async def/for/with, await, yield, yield from or with; an async comprehension
  (`[x async for x in ...]`, `(x async for x in ...)`) counts as async for;
- no bare `except:`, and the identifier `BaseException` nowhere at all: not in an except clause (alone or
  in a tuple), not aliased first (`e = BaseException`, `errs = (ValueError, BaseException)`), not as an
  attribute. The sim time limit is a BaseException and a candidate must not be able to catch it;
- no `return`, `break`, `continue` or `raise` that leaves a `finally` block: the first three discard the
  in-flight exception and `raise` replaces it, the time limit included.

A `finally` block or an `except` clause whose own code fails (`finally: args["missing"]`, `except robot:`)
replaces the in-flight exception too, which no static rule can see, so the runner must treat sim time
past the limit as an error on its own, whatever the candidate returned.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

__all__ = ["ENTRY_SIGNATURES", "FORBIDDEN_NAMES", "FORBIDDEN_NODES", "MAX_LINES", "check_source"]

MAX_LINES = 200

# The contract's list plus `mro`: `Exception.mro()[1]` would be BaseException by another spelling.
FORBIDDEN_NAMES = frozenset(
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
        "mro",
    }
)

# Entry function name -> the exact positional parameters it must take.
ENTRY_SIGNATURES: dict[str, tuple[str, ...]] = {
    "run": ("robot", "args"),
    "reward": ("obs", "action", "prev_action", "info"),
}

FORBIDDEN_NODES: dict[type, str] = {
    ast.ClassDef: "a class definition",
    ast.Global: "a global statement",
    ast.Nonlocal: "a nonlocal statement",
    ast.AsyncFunctionDef: "an async function",
    ast.Await: "await",
    ast.AsyncFor: "async for",
    ast.AsyncWith: "async with",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
    ast.With: "a with statement",
}

_ALLOWED_TOP_LEVEL = "a docstring, 'import math', 'def' and 'NAME = <literal>'"
_BASE_EXCEPTION = "BaseException"
# PEP 695 type parameters and `try`/`except*`; the names are missing on interpreters older than 3.12/3.11,
# where the syntax cannot be parsed anyway.
_TYPE_PARAM_NODES: tuple[type, ...] = tuple(
    getattr(ast, name) for name in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, name)
)
_TRY_NODES: tuple[type, ...] = tuple(getattr(ast, name) for name in ("Try", "TryStar") if hasattr(ast, name))
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)


def check_source(src: str, entry: str = "run") -> list[str]:
    """Return every problem with the candidate source; an empty list means accepted.

    `entry` names the required top-level function (`run` for skills, `reward` for Lane D's reward
    functions). Problems carry a line number where one exists and are sorted by line.
    """
    found: list[tuple[int, str]] = []  # (line or -1, message)

    line_count = len(src.splitlines())
    if line_count > MAX_LINES:
        found.append((-1, f"{line_count} lines; the limit is {MAX_LINES}"))

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        line = exc.lineno if exc.lineno is not None else -1
        prefix = f"line {line}: " if line > 0 else ""
        found.append((line, f"{prefix}syntax error: {exc.msg}"))
        return _finish(found)
    except (ValueError, RecursionError, MemoryError) as exc:
        found.append((-1, f"syntax error: {exc}"))
        return _finish(found)

    _check_entry(tree, entry, found)
    _check_top_level(tree, found)
    for node in ast.walk(tree):
        _check_node(node, found)
    return _finish(found)


def _finish(found: list[tuple[int, str]]) -> list[str]:
    found.sort(key=lambda item: item[0])  # stable: keeps discovery order within a line
    problems: list[str] = []
    for _line, message in found:
        if message not in problems:
            problems.append(message)
    return problems


def _signature(entry: str) -> str:
    params = ENTRY_SIGNATURES.get(entry)
    return f"{entry}({', '.join(params)})" if params else f"{entry}(...)"


def _check_entry(tree: ast.Module, entry: str, found: list[tuple[int, str]]) -> None:
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry]
    if not defs:
        found.append((-1, f"missing top-level entry function {_signature(entry)}"))
        return
    expected = ENTRY_SIGNATURES.get(entry)
    if expected is None:
        return
    for fn in defs:
        a = fn.args
        positional = list(a.posonlyargs) + list(a.args)
        if len(positional) != len(expected) or a.vararg or a.kwarg or a.kwonlyargs:
            actual = ", ".join(_format_params(a))
            wanted = ", ".join(expected)
            found.append(
                (fn.lineno, f"line {fn.lineno}: {entry}() must take exactly ({wanted}), not ({actual})")
            )


def _format_params(a: ast.arguments) -> list[str]:
    parts = [p.arg for p in a.posonlyargs]
    if a.posonlyargs:
        parts.append("/")
    parts += [p.arg for p in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return parts


def _is_import_math(node: ast.Import) -> bool:
    return len(node.names) == 1 and node.names[0].name == "math" and node.names[0].asname is None


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _check_top_level(tree: ast.Module, found: list[tuple[int, str]]) -> None:
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # the module docstring
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # the import rule in _check_node reports anything but 'import math'
        if isinstance(node, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in node.targets):
                found.append(
                    (node.lineno, f"line {node.lineno}: top-level assignment must be 'NAME = <literal>'")
                )
            elif not _is_literal(node.value):
                names = ", ".join(target.id for target in node.targets if isinstance(target, ast.Name))
                found.append(
                    (
                        node.lineno,
                        f"line {node.lineno}: top-level assignment to {names} must be a literal value",
                    )
                )
            continue
        if type(node) in FORBIDDEN_NODES:
            continue  # reported by _check_node
        kind = type(node).__name__
        found.append(
            (node.lineno, f"line {node.lineno}: top-level {kind} is not allowed; only {_ALLOWED_TOP_LEVEL}")
        )


def _line(node: ast.AST) -> int:
    """The node's line, or -1. A comprehension clause carries no position, so its target's line is used."""
    if isinstance(node, ast.comprehension):
        node = node.target
    return getattr(node, "lineno", -1)


def _check_node(node: ast.AST, found: list[tuple[int, str]]) -> None:
    line = _line(node)
    prefix = f"line {line}: " if line > 0 else ""

    description = FORBIDDEN_NODES.get(type(node))
    if description is not None:
        found.append((line, f"{prefix}{description} is not allowed"))
    elif isinstance(node, ast.comprehension) and node.is_async:
        # ast.parse accepts `[x async for x in y]` in a plain def and `(x async for x in y)` even compiles
        found.append((line, f"{prefix}async for is not allowed, not even inside a comprehension"))

    if isinstance(node, ast.ImportFrom):
        found.append((line, f"{prefix}only 'import math' is allowed, not 'from ... import'"))
    elif isinstance(node, ast.Import) and not _is_import_math(node):
        what = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
        found.append((line, f"{prefix}only 'import math' is allowed, not 'import {what}'"))

    # Every field of the AST that carries an identifier goes through the same checks.
    if isinstance(node, ast.Name):
        _check_identifier(node.id, "name", line, prefix, found)
    elif isinstance(node, ast.Attribute):
        _check_identifier(node.attr, "attribute", line, prefix, found)
    elif isinstance(node, ast.arg):
        _check_identifier(node.arg, "parameter", line, prefix, found)
    elif isinstance(node, ast.keyword) and node.arg is not None:
        _check_identifier(node.arg, "keyword argument", line, prefix, found)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        _check_identifier(node.name, "function name", line, prefix, found)
    elif isinstance(node, ast.ExceptHandler):
        if node.name is not None:
            _check_identifier(node.name, "name", line, prefix, found)
        if node.type is None:
            found.append((line, f"{prefix}bare 'except:' is not allowed; catch Exception instead"))
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
        _check_identifier(node.name, "name", line, prefix, found)
    elif isinstance(node, ast.MatchMapping) and node.rest is not None:
        _check_identifier(node.rest, "name", line, prefix, found)
    elif isinstance(node, ast.MatchClass):
        for attr in node.kwd_attrs:  # `case C(x=1)` reads attribute x of the subject
            _check_identifier(attr, "attribute", line, prefix, found)
    elif isinstance(node, _TYPE_PARAM_NODES):
        _check_identifier(node.name, "type parameter", line, prefix, found)

    if isinstance(node, _TRY_NODES):
        _check_finally(node, found)


def _check_identifier(name: str, kind: str, line: int, prefix: str, found: list[tuple[int, str]]) -> None:
    if name == _BASE_EXCEPTION:
        found.append(
            (
                line,
                f"{prefix}{kind} 'BaseException' is not allowed anywhere (the sim time limit is one); "
                "catch Exception instead",
            )
        )
    elif name in FORBIDDEN_NAMES:
        found.append((line, f"{prefix}{kind} '{name}' is not allowed"))
    elif name.startswith("_"):
        found.append((line, f"{prefix}{kind} '{name}' starts with an underscore"))


def _check_finally(node: ast.AST, found: list[tuple[int, str]]) -> None:
    for stmt in _leaving_finally(node.finalbody, in_loop=False):
        word = type(stmt).__name__.lower()
        effect = "replace" if isinstance(stmt, ast.Raise) else "swallow"
        found.append(
            (
                stmt.lineno,
                f"line {stmt.lineno}: '{word}' inside a 'finally' block is not allowed; it would {effect} "
                "the sim time limit",
            )
        )


def _leaving_finally(stmts: list[ast.stmt], in_loop: bool) -> Iterator[ast.stmt]:
    """Yield the statements in `stmts` that leave the enclosing `finally` block early.

    A `return` or `raise` always leaves it (a bare `raise` only re-raises, but nothing a candidate needs
    looks like that); `break` and `continue` only when no loop inside the block owns them. A nested
    function or class keeps its own control flow.
    """
    for stmt in stmts:
        leaves = isinstance(stmt, (ast.Return, ast.Raise))
        if leaves or (not in_loop and isinstance(stmt, (ast.Break, ast.Continue))):
            yield stmt
        elif isinstance(stmt, _SCOPE_NODES):
            continue
        elif isinstance(stmt, _LOOP_NODES):
            yield from _leaving_finally(stmt.body, True)
            yield from _leaving_finally(stmt.orelse, in_loop)  # a loop's else runs outside the loop
        else:
            for field in ("body", "orelse", "finalbody"):
                yield from _leaving_finally(getattr(stmt, field, []), in_loop)
            for handler in getattr(stmt, "handlers", []):
                yield from _leaving_finally(handler.body, in_loop)
            for case in getattr(stmt, "cases", []):
                yield from _leaving_finally(case.body, in_loop)
