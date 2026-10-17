"""Extra edge cases for check_source beyond tests/contract/test_candidate_check.py (S0)."""

import pytest

from sim.candidate_check import FORBIDDEN_NAMES, MAX_LINES, check_source

RUN_OK = "def run(robot, args):\n    return True\n"


def test_agents_candidates_re_exports_the_same_function():
    from agents.candidates import check_source as via_agents

    assert via_agents is check_source


# --- entry function and arity ---


@pytest.mark.parametrize(
    "src",
    [
        "def run(robot):\n    return True\n",
        "def run(robot, args, extra):\n    return True\n",
        "def run(*a):\n    return True\n",
        "def run(robot, args, **kw):\n    return True\n",
        "def run(robot, args, *, flag=False):\n    return True\n",
    ],
)
def test_run_arity_is_exact(src):
    problems = check_source(src)
    assert problems and any("must take exactly (robot, args)" in p for p in problems)


def test_defaults_and_positional_only_keep_the_arity():
    assert check_source("def run(robot, args=None):\n    return True\n") == []
    assert check_source("def run(robot, args, /):\n    return True\n") == []


def test_reward_arity_is_exact():
    assert check_source("def reward(obs, action):\n    return 0.0\n", entry="reward")
    good = "def reward(obs, action, prev_action, info):\n    return 0.0\n"
    assert check_source(good, entry="reward") == []


def test_nested_run_does_not_count_as_the_entry():
    src = (
        "def outer(robot, args):\n    def run(robot, args):\n        return True\n"
        "    return run(robot, args)\n"
    )
    assert any("missing top-level entry function run(robot, args)" in p for p in check_source(src))


def test_empty_source_is_rejected_only_for_the_missing_entry():
    assert check_source("") == ["missing top-level entry function run(robot, args)"]


# --- top-level statements ---


@pytest.mark.parametrize(
    "stmt",
    [
        "X = -1.5",
        "X = (1, 2)",
        "X = {'a': [1, 2], 'b': None}",
        "X = None",
        "X = 'text'",
        "A = B = 1",
        "X = 1 + 2j",
    ],
)
def test_literal_constants_are_accepted(stmt):
    assert check_source(f"{stmt}\n\n{RUN_OK}") == []


@pytest.mark.parametrize(
    "stmt",
    [
        "X = math.pi",
        "X = [i for i in range(3)]",
        "X: float = 1.0",
        "x, y = 1, 2",
        "X = 2 ** 10",
        "X = -'a'",
        "X = run",
        "X = {1, 2} | {3}",
        "X = (lambda: 1)()",
    ],
)
def test_non_literal_top_level_assignments_are_rejected(stmt):
    assert check_source(f"import math\n{stmt}\n\n{RUN_OK}")


def test_docstring_then_imports_then_constants_is_accepted():
    src = '"""Doc."""\nimport math\nimport math\nK = 2\n\n' + RUN_OK
    assert check_source(src) == []


def test_string_expression_after_the_first_statement_is_rejected():
    assert check_source('import math\n"""late docstring"""\n\n' + RUN_OK)


@pytest.mark.parametrize(
    "stmt",
    [
        "pass",
        "if True:\n    pass",
        "for i in range(2):\n    pass",
        "run(None, {})",
        "try:\n    pass\nexcept Exception:\n    pass",
        "assert True",
        "type Alias = int",
    ],
)
def test_other_top_level_statements_are_rejected(stmt):
    problems = check_source(f"{stmt}\n\n{RUN_OK}")
    assert problems and any("top-level" in p for p in problems)


# --- imports ---


def test_import_inside_a_function_is_rejected_unless_it_is_math():
    assert check_source("def run(robot, args):\n    import os\n    return True\n")
    assert check_source("def run(robot, args):\n    import math\n    return math.pi > 3\n") == []


def test_multi_name_import_is_rejected():
    assert check_source("import math, os\n\n" + RUN_OK)


# --- identifiers ---


@pytest.mark.parametrize(
    "src",
    [
        "def run(robot, args):\n    robot.pick('a', _fast=True)\n    return True\n",
        "def run(robot, args):\n    try:\n        robot.home()\n    except Exception as _e:\n"
        "        pass\n    return True\n",
        "def _helper(robot):\n    robot.home()\n\n" + RUN_OK,
        "def helper(_r):\n    return _r\n\n" + RUN_OK,
        "def run(robot, args):\n    for _ in range(2):\n        robot.home()\n    return True\n",
        "def run(robot, args):\n    return [x for x in args if x.__len__]\n",
    ],
)
def test_underscore_identifiers_are_rejected(src):
    problems = check_source(src)
    assert problems and all("underscore" in p for p in problems)


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NAMES))
def test_every_forbidden_name_is_rejected_as_name_attribute_and_definition(name):
    assert check_source(f"def run(robot, args):\n    {name}\n    return True\n")
    assert check_source(f"def run(robot, args):\n    return robot.{name}\n")
    assert check_source(f"def {name}(x):\n    return x\n\n" + RUN_OK)
    assert check_source(f"def helper({name}):\n    return 1\n\n" + RUN_OK)


def test_shadowing_a_forbidden_name_is_rejected():
    assert check_source("def run(robot, args):\n    open = 1\n    return True\n")


# --- except clauses ---


def test_base_exception_inside_a_tuple_is_rejected():
    src = (
        "def run(robot, args):\n    try:\n        robot.home()\n"
        "    except (ValueError, BaseException):\n        pass\n    return True\n"
    )
    assert any("BaseException" in p for p in check_source(src))


def test_specific_exception_tuples_and_finally_are_accepted():
    src = (
        "def run(robot, args):\n    try:\n        robot.home()\n    except (ValueError, TypeError):\n"
        "        return False\n    finally:\n        robot.open_gripper()\n    return True\n"
    )
    assert check_source(src) == []


# --- other nodes ---


@pytest.mark.parametrize(
    "src",
    [
        "async def helper():\n    pass\n\n" + RUN_OK,
        "def run(robot, args):\n    yield from []\n",
        "def run(robot, args):\n    with robot:\n        pass\n    return True\n",
        "def run(robot, args):\n    x = 1\n    def f():\n        nonlocal x\n        x = 2\n"
        "    return True\n",
    ],
)
def test_forbidden_nodes_are_rejected_wherever_they_appear(src):
    assert check_source(src)


def test_lambda_comprehension_fstring_match_and_walrus_are_accepted():
    src = (
        "import math\n\n"
        "def run(robot, args):\n"
        "    objs = robot.scene()['objects']\n"
        "    order = sorted(objs, key=lambda n: objs[n]['x'])\n"
        "    heights = {n: objs[n]['h'] for n in order}\n"
        "    label = f'{len(order)} objects at {math.pi:.2f}'\n"
        "    if (count := len(order)) == 0:\n"
        "        return False\n"
        "    match count:\n"
        "        case 1:\n"
        "            robot.home()\n"
        "        case _:\n"
        "            robot.home()\n"
        "    return bool(label) and heights is not None\n"
    )
    assert check_source(src) == []


# --- size, syntax and report format ---


def test_exactly_200_lines_is_accepted_and_201_is_rejected():
    body = "def run(robot, args):\n" + "    x = 1\n" * (MAX_LINES - 2) + "    return True\n"
    assert len(body.splitlines()) == MAX_LINES
    assert check_source(body) == []
    assert any("lines" in p for p in check_source(body + "# one more line\n"))


def test_syntax_error_is_exactly_one_problem_with_a_line_number():
    problems = check_source("def run(robot, args):\n    return (\n")
    assert len(problems) == 1
    assert problems[0].startswith("line ") and "syntax error" in problems[0]


def test_null_byte_is_a_syntax_error_not_a_crash():
    problems = check_source("def run(robot, args):\n    return True\x00\n")
    assert len(problems) == 1 and "syntax error" in problems[0]


def test_problems_carry_line_numbers_and_are_sorted_by_line():
    src = "import os\n\ndef run(robot, args):\n    x = eval('1')\n    return robot._x\n"
    problems = check_source(src)
    lines = [int(p.split(":")[0].split()[1]) for p in problems if p.startswith("line ")]
    assert lines == sorted(lines)
    assert lines[0] == 1 and 4 in lines and 5 in lines


def test_problems_are_deduplicated_non_empty_strings():
    problems = check_source("def run(robot, args):\n    return robot.__class__, robot.__class__\n")
    assert len(problems) == 1
    assert all(isinstance(p, str) and p.strip() for p in problems)


# --- BaseException anywhere (S0 hardening) ---


@pytest.mark.parametrize(
    "src",
    [
        # the alias evasion: the name never appears in the except clause itself
        "def run(robot, args):\n    e = BaseException\n    try:\n        robot.home()\n    except e:\n"
        "        pass\n    return True\n",
        "def run(robot, args):\n    errs = (ValueError, BaseException)\n    try:\n        robot.home()\n"
        "    except errs:\n        pass\n    return True\n",
        "def run(robot, args):\n    try:\n        pass\n    except (BaseException):\n        pass\n"
        "    return True\n",
        "def run(robot, args):\n    return robot.BaseException\n",
        "def run(robot, args):\n    f = lambda: BaseException\n    return f is not None\n",
        "def helper(BaseException):\n    return BaseException\n\n" + RUN_OK,
        "def run(robot, args):\n    try:\n        robot.home()\n    except Exception as BaseException:\n"
        "        pass\n    return True\n",
        "def run(robot, args):\n    match args:\n        case BaseException():\n            pass\n"
        "    return True\n",
        "def BaseException():\n    return 1\n\n" + RUN_OK,
        "def run(robot, args):\n    robot.pick('a', BaseException=1)\n    return True\n",
        "def run(robot, args):\n    raise BaseException('x')\n",
    ],
)
def test_base_exception_is_rejected_wherever_the_name_appears(src):
    problems = check_source(src)
    assert problems and any("'BaseException'" in p and "catch Exception" in p for p in problems)


def test_base_exception_problem_names_the_line():
    src = "def run(robot, args):\n    x = 1\n    e = BaseException\n    return x\n"
    assert any(p.startswith("line 3:") and "BaseException" in p for p in check_source(src))


def test_mro_cannot_spell_base_exception():
    src = (
        "def run(robot, args):\n    try:\n        robot.home()\n    except Exception.mro()[1]:\n"
        "        pass\n    return True\n"
    )
    assert "mro" in FORBIDDEN_NAMES
    assert any("'mro' is not allowed" in p for p in check_source(src))


def test_catching_exception_subclasses_is_still_accepted():
    src = (
        "def run(robot, args):\n    try:\n        robot.home()\n"
        "    except (ValueError, KeyError, RuntimeError) as err:\n        return bool(err)\n    return True\n"
    )
    assert check_source(src) == []


# --- control flow that leaves a finally block (S0 hardening) ---

FINALLY_RETURN = "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        return True\n"
FINALLY_NESTED_RETURN = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        if args:\n"
    "            return False\n    return True\n"
)
FINALLY_RETURN_IN_LOOP = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        for i in range(2):\n"
    "            return i\n    return True\n"
)
FINALLY_BREAK = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            break\n    return True\n"
)
FINALLY_CONTINUE = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            continue\n    return True\n"
)
# the raise evasion: the ValueError replaces the in-flight SimTimeLimit and the outer handler eats it
FINALLY_RAISE = (
    "def run(robot, args):\n    while True:\n        try:\n            try:\n                robot.home()\n"
    "            finally:\n                raise ValueError('x')\n"
    "        except ValueError:\n            pass\n"
)
FINALLY_RAISE_IN_HANDLER = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        try:\n"
    "            robot.open_gripper()\n        except Exception:\n            raise RuntimeError('cleanup')\n"
)
FINALLY_BARE_RAISE = "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        raise\n"


@pytest.mark.parametrize(
    "src,word",
    [
        (FINALLY_RETURN, "return"),
        (FINALLY_NESTED_RETURN, "return"),
        (FINALLY_RETURN_IN_LOOP, "return"),
        (FINALLY_BREAK, "break"),
        (FINALLY_CONTINUE, "continue"),
        (FINALLY_RAISE, "raise"),
        (FINALLY_RAISE_IN_HANDLER, "raise"),
        (FINALLY_BARE_RAISE, "raise"),
    ],
)
def test_control_flow_that_leaves_a_finally_block_is_rejected(src, word):
    problems = check_source(src)
    assert problems and any(f"'{word}'" in p and "finally" in p for p in problems)


def test_finally_blocks_that_only_clean_up_are_accepted():
    src = (
        "def run(robot, args):\n"
        "    for name in args:\n"
        "        try:\n"
        "            if not robot.pick(name):\n"
        "                continue\n"  # inside the try body, not the finally block
        "        finally:\n"
        "            robot.open_gripper()\n"
        "            for i in range(2):\n"
        "                if i:\n"
        "                    break\n"  # leaves the inner loop, not the finally block
        "            def note():\n"
        "                return 1\n"  # leaves the nested function
        "            note()\n"
        "    try:\n"
        "        robot.home()\n"
        "    except Exception:\n"
        "        return False\n"
        "    else:\n"
        "        return True\n"
        "    finally:\n"
        "        robot.open_gripper()\n"
    )
    assert check_source(src) == []


def test_raise_in_finally_problem_names_the_line_and_the_effect():
    problems = check_source(FINALLY_RAISE)
    assert len(problems) == 1
    assert problems[0].startswith("line 7: 'raise' inside a 'finally' block")
    assert "replace the sim time limit" in problems[0]


def test_raise_outside_a_finally_block_is_still_accepted():
    src = (
        "def run(robot, args):\n"
        "    try:\n"
        "        if not robot.pick(args['object']):\n"
        "            raise ValueError('pick failed')\n"  # in the try body
        "    except ValueError:\n"
        "        raise\n"  # in a handler
        "    finally:\n"
        "        robot.open_gripper()\n"
        "        def note():\n"
        "            raise RuntimeError('never called')\n"  # a nested function keeps its own control flow
        "    return True\n"
    )
    assert check_source(src) == []


# --- async comprehensions (S0 hardening round 2) ---


@pytest.mark.parametrize(
    "expr",
    [
        "[x async for x in robot]",
        "{x async for x in robot}",
        "{x: 1 async for x in robot}",
        "(x async for x in robot)",  # an async generator expression even compiles inside a plain def
        "[y for x in robot async for y in x]",
    ],
)
def test_async_comprehension_inside_a_plain_def_is_rejected(expr):
    # ast.parse accepts these; only the compiler (or the runner's load) would complain, too late to regenerate
    problems = check_source(f"def run(robot, args):\n    return {expr}\n")
    assert problems and any(p.startswith("line 2:") and "async for" in p for p in problems)


def test_plain_comprehensions_are_still_accepted():
    src = "def run(robot, args):\n    return [x for x in args], {x for x in args}, (x for x in args)\n"
    assert check_source(src) == []


# --- Python 3.12 identifier fields (S0 hardening) ---


@pytest.mark.parametrize(
    "src,expected",
    [
        ("def helper[_T](x):\n    return x\n\n" + RUN_OK, "type parameter '_T' starts with an underscore"),
        ("def helper[*_Ts](x):\n    return x\n\n" + RUN_OK, "underscore"),
        ("def helper[**_P](x):\n    return x\n\n" + RUN_OK, "underscore"),
        ("def helper[open](x):\n    return x\n\n" + RUN_OK, "type parameter 'open' is not allowed"),
        ("def run(robot, args):\n    type Pair[_T] = tuple[_T, _T]\n    return True\n", "underscore"),
        (
            "def run(robot, args):\n    match args:\n        case dict(_x=1):\n            pass\n"
            "    return True\n",
            "attribute '_x' starts with an underscore",
        ),
        (
            "def run(robot, args):\n    match args:\n        case dict(open=1):\n            pass\n"
            "    return True\n",
            "attribute 'open' is not allowed",
        ),
    ],
)
def test_type_parameters_and_match_class_keywords_go_through_the_name_checks(src, expected):
    problems = check_source(src)
    assert problems and any(expected in p for p in problems)


def test_clean_type_parameters_and_match_class_keywords_are_accepted():
    src = (
        "def first[T](items: list[T]) -> T:\n    return items[0]\n\n"
        "def run(robot, args):\n"
        "    match args:\n"
        "        case dict(x=1):\n"
        "            return first([True])\n"
        "        case {'object': name, **rest}:\n"
        "            return bool(name) and rest is not None\n"
        "    return True\n"
    )
    assert check_source(src) == []


# --- literal pins (final hardening): forms the mutation harness found unpinned ---


def test_import_math_with_an_alias_is_rejected_on_its_own():
    problems = check_source("import math as m\n\n" + RUN_OK)
    assert len(problems) == 1
    assert problems[0].startswith("line 1:") and "import math as m" in problems[0]
    inside = check_source("def run(robot, args):\n    import math as m\n    return m.pi > 3\n")
    assert inside and all(p.startswith("line 2:") and "import math as m" in p for p in inside)
    assert check_source("import math as math\n\n" + RUN_OK)  # not the exact form either


def test_a_class_nested_in_a_function_is_rejected():
    src = "def run(robot, args):\n    class Helper:\n        pass\n    return True\n"
    assert check_source(src) == ["line 2: a class definition is not allowed"]
    deep = (
        "def run(robot, args):\n    if args:\n        for i in range(2):\n"
        "            class Helper:\n                pass\n    return True\n"
    )
    assert any(p.startswith("line 4:") and "class definition" in p for p in check_source(deep))


@pytest.mark.parametrize(
    "case,expected",
    [
        ("case _x:\n            return True", "name '_x' starts with an underscore"),
        ("case [first, *_rest]:\n            return first", "name '_rest' starts with an underscore"),
        ("case {'object': n, **_rest}:\n            return n", "name '_rest' starts with an underscore"),
        ("case {'object': _name}:\n            return True", "name '_name' starts with an underscore"),
        ("case str() as _s:\n            return True", "name '_s' starts with an underscore"),
        ("case (_a, _b):\n            return True", "name '_b' starts with an underscore"),
        ("case open:\n            return True", "name 'open' is not allowed"),
    ],
)
def test_match_capture_names_go_through_the_identifier_checks(case, expected):
    src = f"def run(robot, args):\n    match args:\n        {case}\n    return True\n"
    problems = check_source(src)
    assert problems and any(p.startswith("line 3:") and expected in p for p in problems)


def test_clean_match_captures_and_the_wildcard_are_accepted():
    src = (
        "def run(robot, args):\n"
        "    match args:\n"
        "        case [first, *rest]:\n"
        "            return first\n"
        "        case {'object': name, **rest}:\n"
        "            return name\n"
        "        case str() as s:\n"
        "            return s\n"
        "        case _:\n"
        "            return True\n"
    )
    assert check_source(src) == []


FINALLY_MATCH_RETURN = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        match args:\n"
    "            case {'object': name}:\n                return name\n            case _:\n"
    "                pass\n    return True\n"
)
FINALLY_MATCH_BREAK = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            match name:\n                case 'x':\n                    break\n"
    "    return True\n"
)
FINALLY_MATCH_CONTINUE = FINALLY_MATCH_BREAK.replace("break", "continue")
FINALLY_LOOP_ELSE_RETURN = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        for i in range(2):\n"
    "            pass\n        else:\n            return True\n    return True\n"
)
# the `else` of a loop runs after the loop: its break/continue belong to the loop OUTSIDE the finally block
FINALLY_LOOP_ELSE_BREAK = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            for i in range(2):\n                pass\n            else:\n"
    "                break\n    return True\n"
)
FINALLY_WHILE_ELSE_CONTINUE = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            while False:\n                pass\n            else:\n"
    "                continue\n    return True\n"
)
FINALLY_EXCEPT_STAR_RETURN = (
    "def run(robot, args):\n    try:\n        robot.home()\n    except* ValueError:\n        pass\n"
    "    finally:\n        return True\n"
)
FINALLY_EXCEPT_STAR_HANDLER_RETURN = (
    "def run(robot, args):\n    try:\n        robot.home()\n    finally:\n        try:\n"
    "            robot.open_gripper()\n        except* ValueError:\n            return False\n"
    "    return True\n"
)
FINALLY_EXCEPT_STAR_HANDLER_BREAK = (
    "def run(robot, args):\n    for name in args:\n        try:\n            robot.pick(name)\n"
    "        finally:\n            try:\n                robot.open_gripper()\n"
    "            except* ValueError:\n                break\n    return True\n"
)


@pytest.mark.parametrize(
    "src,word,line",
    [
        (FINALLY_MATCH_RETURN, "return", 7),
        (FINALLY_MATCH_BREAK, "break", 8),
        (FINALLY_MATCH_CONTINUE, "continue", 8),
        (FINALLY_LOOP_ELSE_RETURN, "return", 8),
        (FINALLY_LOOP_ELSE_BREAK, "break", 9),
        (FINALLY_WHILE_ELSE_CONTINUE, "continue", 9),
        (FINALLY_EXCEPT_STAR_RETURN, "return", 7),
        (FINALLY_EXCEPT_STAR_HANDLER_RETURN, "return", 8),
        (FINALLY_EXCEPT_STAR_HANDLER_BREAK, "break", 9),
    ],
)
def test_leaving_a_finally_block_through_match_loop_else_or_except_star_is_rejected(src, word, line):
    problems = check_source(src)
    assert len(problems) == 1
    assert problems[0].startswith(f"line {line}: '{word}' inside a 'finally' block is not allowed")
    assert "swallow the sim time limit" in problems[0]


def test_match_loop_else_and_except_star_inside_finally_are_accepted_when_nothing_leaves():
    src = (
        "def run(robot, args):\n"
        "    try:\n"
        "        robot.home()\n"
        "    except* ValueError:\n"
        "        return False\n"  # a handler outside any finally block may return
        "    finally:\n"
        "        match args:\n"
        "            case {'object': name}:\n"
        "                robot.open_gripper()\n"
        "            case _:\n"
        "                pass\n"
        "        for i in range(2):\n"
        "            for j in range(2):\n"
        "                pass\n"
        "            else:\n"
        "                break\n"  # leaves the outer `for i`, which lives inside the finally block
        "        while False:\n"
        "            pass\n"
        "        else:\n"
        "            robot.open_gripper()\n"
        "        try:\n"
        "            robot.open_gripper()\n"
        "        except* ValueError:\n"
        "            pass\n"
        "    return True\n"
    )
    assert check_source(src) == []


def test_except_star_base_exception_is_rejected_too():
    src = (
        "def run(robot, args):\n    try:\n        robot.home()\n    except* BaseException:\n"
        "        pass\n    return True\n"
    )
    assert any("'BaseException'" in p and "catch Exception" in p for p in check_source(src))
