"""sim.restricted: the namespace AI-written code gets inside the runner (no sandbox of its own)."""

from __future__ import annotations

import math

import pytest

from sim import restricted

GOOD = '''"""Doc."""
import math

LIMIT = 0.3

def helper(x):
    return math.sqrt(x) + LIMIT

def run(robot, args):
    print("hello from a candidate")
    return helper(4.0) > 2.0
'''


def test_load_entry_runs_import_math_and_returns_the_entry(capsys):
    run = restricted.load_entry(GOOD)
    assert run(None, {}) is True
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean for the score line
    assert "hello from a candidate" in captured.err


def test_rejected_source_raises_before_executing():
    with pytest.raises(ValueError, match="rejected"):
        restricted.load_entry("import os\n\ndef run(robot, args):\n    return True\n")


def test_only_math_can_be_imported():
    # `import math` inside a function is accepted by check_source and served by the restricted __import__
    assert restricted.load_entry("def run(robot, args):\n    import math\n    return math.pi > 3\n")(None, {})
    ns = restricted.namespace()
    assert ns["__builtins__"]["__import__"]("math") is math
    with pytest.raises(ImportError):
        ns["__builtins__"]["__import__"]("os")
    with pytest.raises(ImportError):
        ns["__builtins__"]["__import__"]("math", None, None, ("sqrt",), 0)


def test_dangerous_builtins_are_absent():
    ns = restricted.namespace()
    for name in ("open", "exec", "eval", "compile", "getattr", "type", "object", "__build_class__"):
        assert name not in ns["__builtins__"], name
    assert ns["__builtins__"]["len"] is len
    assert restricted.safe_builtins() is not restricted.SAFE_BUILTINS


def test_entry_must_be_callable_after_exec():
    src = "def run(robot, args):\n    return True\n\nrun = 3\n"
    assert restricted.check_source(src) == []  # `run = 3` is a literal assignment, so the static check passes
    with pytest.raises(ValueError, match="no callable"):
        restricted.load_entry(src)
    with pytest.raises(ValueError, match="rejected"):
        restricted.load_entry("def reward(obs, action, prev_action, info):\n    return 0.0\n", entry="run")


def test_reward_entry():
    reward = restricted.load_entry(
        "import math\n\ndef reward(obs, action, prev_action, info):\n    return -info['dist']\n",
        entry="reward",
    )
    assert reward([0.0] * 16, [0.0] * 4, [0.0] * 4, {"dist": 0.25}) == -0.25
