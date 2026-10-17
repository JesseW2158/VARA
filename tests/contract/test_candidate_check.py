"""Contract: agents.candidates.check_source(src, entry="run") -> list[str].

An empty list means the source is accepted. Every problem is a non-empty string.
Spec: docs/interfaces.md, "Candidate program". Don't edit without team agreement.
"""

import pathlib

import pytest

check_source = pytest.importorskip(
    "agents.candidates", reason="S0 implements agents/candidates.py"
).check_source

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
GOOD = (FIXTURES / "candidate_clear_table.py").read_text()


def test_contract_example_is_accepted():
    assert check_source(GOOD) == []


def test_docstring_constants_and_helpers_are_accepted():
    src = (
        '"""Put the juice box in the tray."""\n'
        "import math\n"
        "LIFT = 0.12\n"
        "SLOTS = [0, 1]\n\n"
        "def helper(robot):\n"
        "    robot.home()\n\n"
        "def run(robot, args):\n"
        "    if not robot.move_to(0.2, 0.0, LIFT, pitch=-math.pi / 2):\n"
        "        return False\n"
        "    helper(robot)\n"
        "    return True\n"
    )
    assert check_source(src) == []


def test_except_exception_is_accepted():
    src = (
        "def run(robot, args):\n"
        "    try:\n"
        "        robot.home()\n"
        "    except Exception:\n"
        "        return False\n"
        "    return True\n"
    )
    assert check_source(src) == []


def test_reward_entry_for_lane_d():
    src = (
        "import math\n\ndef reward(obs, action, prev_action, info):\n    return -math.hypot(obs[0], obs[1])\n"
    )
    assert check_source(src, entry="reward") == []
    assert check_source(src) != []  # no run()


RUN_OK = "def run(robot, args):\n    return True\n"
BAD = {
    "import_os": "import os\n\n" + RUN_OK,
    "import_math_as": "import math as m\nimport os.path\n\n" + RUN_OK,
    "from_import": "from math import pi\n\n" + RUN_OK,
    "open_call": "def run(robot, args):\n    open('x', 'w')\n    return True\n",
    "eval_call": "def run(robot, args):\n    return eval('1')\n",
    "exec_call": "def run(robot, args):\n    exec('x = 1')\n    return True\n",
    "compile_call": "def run(robot, args):\n    compile('1', 'f', 'eval')\n    return True\n",
    "dunder_import": "def run(robot, args):\n    return __import__('os')\n",
    "getattr_call": "def run(robot, args):\n    return getattr(robot, 'home')()\n",
    "setattr_call": "def run(robot, args):\n    setattr(robot, 'x', 1)\n    return True\n",
    "globals_call": "def run(robot, args):\n    return globals()\n",
    "vars_call": "def run(robot, args):\n    return vars(robot)\n",
    "type_call": "def run(robot, args):\n    return type(robot)\n",
    "dunder_attribute": "def run(robot, args):\n    return robot.__class__\n",
    "private_attribute": "def run(robot, args):\n    return robot._driver\n",
    "underscore_name": "def run(robot, args):\n    _x = 1\n    return True\n",
    "dunder_in_lambda": "def run(robot, args):\n    f = lambda: robot.__dict__\n    return True\n",
    "class_def": "class A:\n    pass\n\n" + RUN_OK,
    "global_stmt": "X = 1\n\ndef run(robot, args):\n    global X\n    return True\n",
    "nonlocal_stmt": "def run(robot, args):\n    x = 1\n    def f():\n        nonlocal x\n    return True\n",
    "with_stmt": "def run(robot, args):\n    with robot:\n        pass\n    return True\n",
    "yield_stmt": "def run(robot, args):\n    yield 1\n",
    "async_def": "async def run(robot, args):\n    return True\n",
    "bare_except": (
        "def run(robot, args):\n    try:\n        robot.home()\n    except:\n        pass\n    return True\n"
    ),
    "base_exception": (
        "def run(robot, args):\n    try:\n        robot.home()\n"
        "    except BaseException:\n        pass\n    return True\n"
    ),
    "missing_run": "import math\n\ndef go(robot, args):\n    return True\n",
    "top_level_call": "import math\nprint('hi')\n\n" + RUN_OK,
    "top_level_non_literal": "X = len('abc')\n\n" + RUN_OK,
    "syntax_error": "def run(robot, args)\n    return True\n",
    "too_long": "def run(robot, args):\n" + "    x = 1\n" * 200 + "    return True\n",
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_bad_source_is_rejected(case):
    problems = check_source(BAD[case])
    assert problems, f"{case} should be rejected"
    assert all(isinstance(p, str) and p.strip() for p in problems)
