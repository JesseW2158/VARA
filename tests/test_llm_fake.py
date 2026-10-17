"""agents.llm: the fixture-backed fake, and the real path driven through a stubbed client (C1).

No test here touches the network: the fake answers from tests/fixtures/llm/, and the real path gets a stub
client through `llm.client_factory` with `llm.sleep` replaced.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from agents import llm

ROOT = Path(__file__).resolve().parents[1]
ROUTER_SCHEMA = json.loads((ROOT / "schemas" / "router.json").read_text(encoding="utf-8"))
CLEAR_TABLE = (ROOT / "tests" / "fixtures" / "candidate_clear_table.py").read_text(encoding="utf-8")
LLM_FIXTURES = ROOT / "tests" / "fixtures" / "llm"
MSGS = [{"role": "user", "content": "hello"}]
SECRET = "xai-test-key-0123456789-do-not-leak"
LOG_KEYS = {"t", "task_id", "model", "purpose", "prompt_tokens", "completion_tokens", "ms"}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.setattr(llm, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def real(monkeypatch, tmp_path):
    """Real path: no fake flag, a key in the environment, recorded sleeps, runs/ in tmp."""
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("ROUTER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("CODEGEN_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("XAI_API_KEY", SECRET)
    monkeypatch.setattr(llm, "RUNS", tmp_path / "runs")
    sleeps: list[float] = []
    monkeypatch.setattr(llm, "sleep", sleeps.append)
    return sleeps


class FakeClock:
    """A clock that only moves when the code sleeps (or a stub request "takes" time)."""

    def __init__(self, now: float = 1000.0):
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(real, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(llm, "clock", clock)
    monkeypatch.setattr(llm, "sleep", clock.sleep)
    return clock


@pytest.fixture
def no_dotenv(monkeypatch):
    """`main()` loads `.env` like every CLI; a developer's `.env` must never leak keys or flags into the
    test process, so the loader is replaced and its calls counted."""
    calls: list[dict] = []

    def load_dotenv(*args, **kwargs):
        calls.append(dict(kwargs))
        return False

    monkeypatch.setattr("dotenv.load_dotenv", load_dotenv)
    return calls


class StatusError(Exception):
    """Stands in for openai.APIStatusError: carries `status_code` like the real one."""

    def __init__(self, status_code: int, message: str = "boom"):
        super().__init__(message)
        self.status_code = status_code


def _response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    message = SimpleNamespace(content=text, refusal=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class StubClient:
    """`client.chat.completions.create(**kwargs)` replays a script of texts and exceptions; with a
    `clock` and `cost`, every request advances the fake clock by `cost` seconds."""

    def __init__(self, script, clock: FakeClock | None = None, cost: float = 0.0):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.clock, self.cost = clock, cost

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.clock is not None:
            self.clock.now += self.cost
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _response(item)


def install(monkeypatch, script, clock: FakeClock | None = None, cost: float = 0.0):
    client = StubClient(script, clock, cost)
    seen: dict = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return client

    monkeypatch.setattr(llm, "client_factory", factory)
    return client, seen


# --- fake path ---------------------------------------------------------------------------------------


def test_fake_router_known_phrase(fake):
    reply = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="Hand me my pills")
    assert reply == {"action": "run", "skill": "hand_over", "args": {"object": "pill_bottle"}}


def test_fake_router_normalizes_through_aliases(fake):
    assert (
        llm.normalize("  Clear the table. Put everything in the tray! ")
        == "clear_the_table_put_everything_in_the_tray"
    )
    assert llm.normalize("HAND me,  my pills!!") == "hand_me_my_pills"
    a = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="HAND me,  my pills!!")
    b = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="hand me my pills")
    assert a == b


def test_fake_router_every_fixture_is_valid(fake):
    validator = jsonschema.Draft202012Validator(ROUTER_SCHEMA)
    for path in sorted((LLM_FIXTURES / "router").glob("*.json")):
        validator.validate(json.loads(path.read_text(encoding="utf-8")))
    aliases = json.loads((LLM_FIXTURES / "aliases.json").read_text(encoding="utf-8"))
    for phrase, slug in aliases.items():
        assert llm.normalize(phrase) == phrase, phrase
        assert (LLM_FIXTURES / "router" / f"{slug}.json").is_file(), slug


def test_fake_router_unknown_phrase_queues(fake):
    text = "Juggle the tea tin"
    reply = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key=text)
    assert reply == {"action": "queue", "task": text, "reason": "no fixture for this phrase"}


def test_fake_router_queue_fixture_carries_the_users_words(fake):
    reply = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="FOLD my laundry!")
    assert reply["action"] == "queue"
    assert reply["task"] == "FOLD my laundry!"
    assert reply["reason"] == "needs cloth handling"
    assert llm.normalize("fold_my_laundry") == "fold_my_laundry"  # slugs normalize to themselves


def test_fake_router_uses_the_last_user_message_without_a_key(fake):
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "Hand me my pills"}]
    reply = llm.chat(msgs, "m", schema=ROUTER_SCHEMA, purpose="router")
    assert reply["action"] == "run"


def test_fake_candidate_is_fenced_fixture(fake):
    from agents.candidates import extract_code

    text = llm.chat(MSGS, "m", purpose="write_skill", key="juice_box_to_tray:3")
    assert isinstance(text, str) and text.startswith("```python\n") and text.rstrip().endswith("```")
    fixture = (LLM_FIXTURES / "candidates" / "juice_box_to_tray" / "cand_3.py").read_text(encoding="utf-8")
    assert extract_code(text) == fixture


def test_fake_candidate_unknown_key_falls_back_to_contract_example(fake):
    from agents.candidates import extract_code

    for key in ("no_such_task:1", "juice_box_to_tray:99", None, "garbage"):
        assert extract_code(llm.chat(MSGS, "m", purpose="write_skill", key=key)) == CLEAR_TABLE


def test_fake_candidate_retry_key(fake):
    from agents.candidates import extract_code

    assert (
        extract_code(llm.chat(MSGS, "m", purpose="write_skill", key="juice_box_to_tray:1:retry"))
        == CLEAR_TABLE
    )


def test_fake_repair_fixture_and_fallback(fake):
    from agents.candidates import extract_code

    fixture = (LLM_FIXTURES / "repair" / "juice_box_to_tray" / "cand_2.py").read_text(encoding="utf-8")
    assert extract_code(llm.chat(MSGS, "m", purpose="repair", key="juice_box_to_tray:2")) == fixture
    assert extract_code(llm.chat(MSGS, "m", purpose="repair", key="clear_table:2")) == CLEAR_TABLE


def test_fake_rewards(fake):
    from agents.candidates import extract_code, extract_code_blocks

    fixture = (LLM_FIXTURES / "rewards" / "variant_3.py").read_text(encoding="utf-8")
    assert extract_code(llm.chat(MSGS, "m", purpose="write_rewards", key="variant_3")) == fixture
    assert extract_code(llm.chat(MSGS, "m", purpose="write_rewards", key="3")) == fixture
    blocks = extract_code_blocks(llm.chat(MSGS, "m", purpose="write_rewards"))
    assert len(blocks) == 4
    assert all("def reward(obs, action, prev_action, info)" in b for b in blocks)


def test_fake_validates_the_fixture_against_the_schema(fake, monkeypatch, tmp_path):
    fixtures = tmp_path / "fx"
    (fixtures / "router").mkdir(parents=True)
    (fixtures / "router" / "dance.json").write_text('{"action": "dance"}', encoding="utf-8")
    (fixtures / "aliases.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(llm, "FIXTURES", fixtures)
    with pytest.raises(jsonschema.ValidationError):
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="dance")


def test_fake_never_imports_or_constructs_openai(fake, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # any `import openai` now raises ImportError

    def factory(**kwargs):
        raise AssertionError("client constructed in fake mode")

    monkeypatch.setattr(llm, "client_factory", factory)
    monkeypatch.setenv("XAI_API_KEY", SECRET)
    assert llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="stop") == {"action": "stop"}
    assert llm.chat(MSGS, "m", purpose="write_skill", key="clear_table:1")
    assert llm.chat(MSGS, "m") == "pong (FAKE_LLM)"


def test_fake_flag_alone_enables_the_fake(monkeypatch, tmp_path):
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    assert llm.is_fake()
    assert llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", key="stop") == {"action": "stop"}


def test_fake_writes_llm_jsonl(fake):
    llm.chat(MSGS, "fake-model", schema=ROUTER_SCHEMA, purpose="router", key="stop", task_id="t-0001")
    llm.chat(MSGS, "fake-model", purpose="write_skill", key="clear_table:1", task_id="t-0001")
    lines = (fake / "t-0001" / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert set(entry) == LOG_KEYS
        assert entry["task_id"] == "t-0001" and entry["model"] == "fake-model"
        assert isinstance(entry["prompt_tokens"], int) and isinstance(entry["completion_tokens"], int)
        assert entry["purpose"] in ("router", "write_skill")
    written = sorted(p.relative_to(fake).as_posix() for p in fake.rglob("*") if p.is_file())
    assert written == ["t-0001/llm.jsonl"]  # nothing else was written under runs/


def test_bad_arguments_are_rejected_before_any_call(fake):
    with pytest.raises(ValueError):
        llm.chat([], "m")
    with pytest.raises(ValueError):
        llm.chat("hello", "m")
    with pytest.raises(ValueError):
        llm.chat(MSGS, "")
    with pytest.raises(ValueError):
        llm.chat(MSGS, "m", task_id="../escape")
    with pytest.raises(ValueError):
        llm.run_dir("a/b")
    for timeout in (0, -1, float("inf"), float("nan"), "soon", None):
        with pytest.raises(ValueError, match="timeout"):
            llm.chat(MSGS, "m", timeout=timeout)


def test_model_names_from_env_with_defaults(monkeypatch, caplog):
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("CODEGEN_MODEL", raising=False)
    monkeypatch.setattr(llm, "_WARNED", set())  # the warning fires once per process
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        assert llm.router_model() == "grok-4.3"
        assert llm.codegen_model() == "grok-4.7"
    assert "ROUTER_MODEL" in caplog.text and "CODEGEN_MODEL" in caplog.text
    monkeypatch.setenv("ROUTER_MODEL", "grok-3-mini")
    monkeypatch.setenv("CODEGEN_MODEL", "grok-4-0709")
    assert (llm.router_model(), llm.codegen_model()) == ("grok-3-mini", "grok-4-0709")


def test_parse_json_object_variants():
    assert llm.parse_json_object('{"a": 1}') == {"a": 1}
    assert llm.parse_json_object('Sure!\n```json\n{"a": {"b": [1, 2]}}\n```\nDone.') == {"a": {"b": [1, 2]}}
    assert llm.parse_json_object('x {not json} then {"ok": true} tail') == {"ok": True}
    with pytest.raises(ValueError):
        llm.parse_json_object("no braces here")
    with pytest.raises(ValueError):
        llm.parse_json_object("[1, 2, 3]")


def test_parse_json_object_rejects_nan_and_infinity():
    """Python's json accepts NaN/Infinity, JSON does not; a spec with x = NaN could never be satisfied."""
    for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"a": {"x": NaN}}'):
        with pytest.raises(ValueError):
            llm.parse_json_object(text)
    assert llm.parse_json_object('{"x": NaN} {"x": 1}') == {"x": 1}  # the next valid object is taken
    assert llm.parse_json_object('{"a": "NaN", "b": 1e308}') == {"a": "NaN", "b": 1e308}


def test_real_nan_in_a_plain_text_reply_triggers_the_validation_retry(real, monkeypatch):
    spec = {
        "name": "near_nan",
        "description": "d",
        "params": {},
        "success": {"near": {"object": "juice_box", "x": "NAN", "y": 0, "tol": 0.1}},
        "english": "e",
    }
    nan_reply = json.dumps({"action": "learn", "spec": spec}).replace('"NAN"', "NaN")
    client, _seen = install(monkeypatch, [StatusError(400), nan_reply, '{"action": "stop"}'])
    assert llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router") == {"action": "stop"}
    assert len(client.calls) == 3  # the NaN spec never came back as a learn: it cost a validation retry
    assert "not acceptable" in client.calls[2]["messages"][-1]["content"]
    install(monkeypatch, [StatusError(400), nan_reply, nan_reply])
    with pytest.raises(llm.InvalidReply):
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")


# --- real path with a stubbed client -----------------------------------------------------------------


def test_real_json_schema_success(real, monkeypatch):
    client, seen = install(monkeypatch, ['{"action": "stop"}'])
    reply = llm.chat(MSGS, "grok-4.3", schema=ROUTER_SCHEMA, timeout=7, purpose="router", task_id="t-0002")
    assert reply == {"action": "stop"}
    assert seen == {"api_key": SECRET, "base_url": "https://api.x.ai/v1", "timeout": 7.0}
    call = client.calls[0]
    assert call["model"] == "grok-4.3" and call["messages"] == MSGS and call["timeout"] == 7.0
    assert call["temperature"] == 0.2
    assert call["reasoning_effort"] == "none" and call["max_completion_tokens"] == 2048
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "router"
    assert call["response_format"]["json_schema"]["schema"]["oneOf"] == ROUTER_SCHEMA["oneOf"]
    assert real == []
    line = json.loads((llm.RUNS / "t-0002" / "llm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(line) == LOG_KEYS
    assert (line["prompt_tokens"], line["completion_tokens"], line["purpose"]) == (10, 5, "router")


def test_real_text_reply_for_code(real, monkeypatch):
    client, _seen = install(monkeypatch, ["```python\ndef run(robot, args):\n    return True\n```"])
    reply = llm.chat(MSGS, "grok-4.7", purpose="write_skill")
    assert reply.startswith("```python")
    assert client.calls[0]["temperature"] == 0.7
    assert client.calls[0]["reasoning_effort"] == "low"
    assert client.calls[0]["max_completion_tokens"] == 8192
    assert "response_format" not in client.calls[0]


@pytest.mark.parametrize("purpose,temperature", [("repair", 0.7), ("write_rewards", 0.7), (None, 0.7)])
def test_real_temperatures(real, monkeypatch, purpose, temperature):
    client, _seen = install(monkeypatch, ["ok"])
    llm.chat(MSGS, "grok-4", purpose=purpose)
    assert client.calls[0]["temperature"] == temperature


@pytest.mark.parametrize(
    ("model", "purpose", "effort", "tokens"),
    [
        ("grok-4.3", "router", "none", 2048),
        ("grok-4.7", "router", "low", 2048),
        ("grok-4.3", "write_skill", "low", 8192),
        ("grok-4.7-latest", "repair", "low", 8192),
        ("grok-4.7", "write_rewards", "low", 16384),
        ("grok-4-fast", "router", None, 2048),
        ("custom-code-model", "write_skill", None, 8192),
    ],
)
def test_latency_options_respect_known_model_capabilities(real, monkeypatch, model, purpose, effort, tokens):
    client, _ = install(monkeypatch, ["ok"])
    assert llm.chat(MSGS, model, purpose=purpose) == "ok"
    request = client.calls[0]
    assert request["model"] == model
    assert request.get("reasoning_effort") == effort
    assert request["max_completion_tokens"] == tokens
    assert "max_tokens" not in request


@pytest.mark.parametrize("purpose", ["router", "write_skill", "repair", "write_rewards"])
def test_explicit_effort_overrides_are_forwarded_for_known_and_custom_models(real, monkeypatch, purpose):
    variable = "ROUTER_REASONING_EFFORT" if purpose == "router" else "CODEGEN_REASONING_EFFORT"
    monkeypatch.setenv(variable, " medium ")
    for model in ("grok-4.7", "custom-model"):
        client, _ = install(monkeypatch, ["ok"])
        llm.chat(MSGS, model, purpose=purpose)
        assert client.calls[0]["reasoning_effort"] == "medium"


@pytest.mark.parametrize("effort", ["none", "unbounded"])
def test_bad_effort_fails_before_constructing_client(real, monkeypatch, effort):
    monkeypatch.setenv("CODEGEN_REASONING_EFFORT", effort)
    client, seen = install(monkeypatch, ["unused"])
    with pytest.raises(ValueError, match="CODEGEN_REASONING_EFFORT"):
        llm.chat(MSGS, "grok-4.7", purpose="write_skill")
    assert not seen and not client.calls


def test_effort_and_output_caps_survive_schema_fallback_and_validation_retry(real, monkeypatch):
    client, _ = install(monkeypatch, [StatusError(400), '{"action":"dance"}', '{"action":"stop"}'])
    assert llm.chat(MSGS, "grok-4.3", schema=ROUTER_SCHEMA, purpose="router") == {"action": "stop"}
    assert len(client.calls) == 3
    assert all(call["reasoning_effort"] == "none" for call in client.calls)
    assert all(call["max_completion_tokens"] == 2048 for call in client.calls)
    assert "response_format" not in client.calls[-1]


def test_truncated_output_is_rejected_even_if_the_visible_prefix_parses(real, monkeypatch):
    client, _ = install(monkeypatch, [])
    response = _response('{"action":"stop"}')
    response.choices[0].finish_reason = "length"
    client.chat.completions.create = lambda **_: response
    with pytest.raises(llm.ModelCallFailed, match="truncated"):
        llm.chat(MSGS, "grok-4.3", schema=ROUTER_SCHEMA, purpose="router")


def test_client_construction_consumes_the_whole_call_budget(fake_clock, monkeypatch):
    client, _ = install(monkeypatch, ["ok"], clock=fake_clock)

    def slow_factory(**_):
        fake_clock.now += 2.0
        return client

    monkeypatch.setattr(llm, "client_factory", slow_factory)
    assert llm.chat(MSGS, "grok-4.7", purpose="write_skill", timeout=5) == "ok"
    assert client.calls[0]["timeout"] == 3.0


def test_real_4xx_falls_back_to_plain_text(real, monkeypatch, caplog):
    client, _seen = install(
        monkeypatch, [StatusError(400, "response_format unsupported"), 'Sure:\n{"action": "stop"}\nBye']
    )
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        reply = llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")
    assert reply == {"action": "stop"}
    assert "response_format" in client.calls[0] and "response_format" not in client.calls[1]
    assert real == []  # a 4xx is never retried with backoff
    assert "falling back to plain text" in caplog.text


def test_real_4xx_on_the_text_attempt_raises(real, monkeypatch):
    install(monkeypatch, [StatusError(400), StatusError(404, "no such model")])
    with pytest.raises(llm.ModelCallFailed) as info:
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")
    assert info.value.status == 404


def test_real_429_then_success(real, monkeypatch):
    client, _seen = install(monkeypatch, [StatusError(429), '{"action": "stop"}'])
    assert llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router") == {"action": "stop"}
    assert real == [1.0]
    assert len(client.calls) == 2
    assert "response_format" in client.calls[1]  # a 429 keeps the structured request


def test_real_persistent_429_gives_up_without_a_plain_text_attempt(real, monkeypatch):
    """Four 429s with a schema: the structured request is retried three times (1, 2, 4 s) and the call
    then fails with status 429. A rate limit is never mistaken for "structured output unsupported", so
    no plain-text attempt follows: that would be a fifth request against a limit that just said no."""
    client, _seen = install(monkeypatch, [StatusError(429, "rate limited")] * 4)
    with pytest.raises(llm.ModelCallFailed) as info:
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")
    assert info.value.status == 429
    assert "gave up after 3 retries" in str(info.value)
    assert len(client.calls) == 4
    assert real == [1.0, 2.0, 4.0]
    assert all("response_format" in call for call in client.calls)  # never the plain-text fallback


def test_real_5xx_backoff_then_success(real, monkeypatch):
    install(monkeypatch, [StatusError(500), StatusError(502), StatusError(503), "pong"])
    assert llm.chat(MSGS, "m") == "pong"
    assert real == [1.0, 2.0, 4.0]


def test_real_5xx_gives_up_after_three_retries(real, monkeypatch):
    client, _seen = install(monkeypatch, [StatusError(503)] * 4)
    with pytest.raises(llm.ModelCallFailed) as info:
        llm.chat(MSGS, "m")
    assert info.value.status == 503
    assert real == [1.0, 2.0, 4.0]
    assert len(client.calls) == 4


def test_real_connection_error_is_not_retried(real, monkeypatch):
    install(monkeypatch, [ConnectionError("no network")])
    with pytest.raises(llm.ModelCallFailed) as info:
        llm.chat(MSGS, "m")
    assert info.value.status is None
    assert real == []


def test_real_validation_retry_carries_the_error(real, monkeypatch, caplog):
    client, _seen = install(monkeypatch, ['{"action": "dance"}', '{"action": "stop"}'])
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        assert llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router") == {"action": "stop"}
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[:1] == MSGS
    assert retry_messages[-2] == {"role": "assistant", "content": '{"action": "dance"}'}
    assert retry_messages[-1]["role"] == "user"
    assert "not acceptable" in retry_messages[-1]["content"]
    assert "schema violation" in retry_messages[-1]["content"]
    assert real == []


def test_real_validation_fails_twice(real, monkeypatch):
    install(monkeypatch, ['{"action": "dance"}', "no json at all"])
    with pytest.raises(llm.InvalidReply):
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and everything chained to it through __cause__ and __context__."""
    out: list[BaseException] = []
    seen: set[int] = set()
    todo: list[BaseException | None] = [exc]
    while todo:
        item = todo.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        out.append(item)
        todo.extend([item.__cause__, item.__context__])
    return out


def test_real_key_never_appears_in_errors_or_logs(real, monkeypatch, caplog):
    # The structured attempt gets a 401, the plain-text fallback gets another: both carry the key.
    install(monkeypatch, [StatusError(401, f"bad key {SECRET} rejected")] * 2)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(llm.ModelCallFailed) as info:
            llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", task_id="t-0003")
    assert SECRET not in str(info.value)
    assert "[XAI_API_KEY]" in str(info.value)
    assert SECRET not in caplog.text
    assert _chain(info.value) == [info.value]  # the API error is not chained: its message holds the key
    log_dir = llm.RUNS / "t-0003"
    for path in log_dir.rglob("*") if log_dir.exists() else []:
        assert SECRET not in path.read_text(encoding="utf-8")
    # A reply that echoes the key and fails the schema twice: the chained ValidationError would hold it.
    install(monkeypatch, [json.dumps({"action": "dance", "task": SECRET})] * 2)
    with pytest.raises(llm.InvalidReply) as info:
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router")
    assert _chain(info.value) == [info.value]
    assert SECRET not in f"{info.value} {info.value.args!r}"


def test_real_missing_key_fails_before_any_call(real, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    client, seen = install(monkeypatch, ["unused"])
    with pytest.raises(llm.ModelCallFailed, match="XAI_API_KEY"):
        llm.chat(MSGS, "m")
    assert client.calls == [] and seen == {}


def test_real_client_factory_failure_is_wrapped_and_scrubbed(real, monkeypatch, caplog):
    """Regression: a factory that raises with the key in its message leaked it through the router's
    warning log and the candidates' problem lists; now it is a ModelCallFailed without the key."""

    def factory(**kwargs):
        raise RuntimeError(f"cannot build a client for key {SECRET}")

    monkeypatch.setattr(llm, "client_factory", factory)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(llm.ModelCallFailed) as info:
            llm.chat(MSGS, "m", purpose="write_skill")
    text = f"{info.value} {info.value!r} {info.value.args!r}"
    assert SECRET not in text and "[XAI_API_KEY]" in text and "RuntimeError" in text
    assert _chain(info.value) == [info.value]  # nothing chained carries the original message
    assert SECRET not in caplog.text


def test_real_budget_bounds_the_retries(fake_clock, monkeypatch):
    """`timeout` is the budget for the whole call: every attempt gets the time left, and a backoff that
    would leave less than MIN_ATTEMPT_S is not taken (the old code spent 4 x timeout + 7 s)."""
    client, _seen = install(monkeypatch, [StatusError(503)] * 4, clock=fake_clock, cost=1.0)
    started = fake_clock.now
    with pytest.raises(llm.ModelCallFailed) as info:
        llm.chat(MSGS, "m", timeout=5)
    assert info.value.status == 503 and "budget is spent" in str(info.value)
    # t=0 attempt (5 s left) -> 503 at t=1, sleep 1 -> attempt at t=2 (3 s left) -> 503 at t=3: the next
    # backoff of 2 s would leave 0 s, so no third attempt.
    assert [c["timeout"] for c in client.calls] == [5.0, 3.0]
    assert fake_clock.sleeps == [1.0]
    assert fake_clock.now - started <= 5.0


def test_real_budget_allows_the_full_backoff_when_there_is_time(fake_clock, monkeypatch):
    client, _seen = install(monkeypatch, [StatusError(500), StatusError(502), StatusError(503), "pong"])
    assert llm.chat(MSGS, "m", timeout=30) == "pong"
    assert fake_clock.sleeps == [1.0, 2.0, 4.0]
    assert [c["timeout"] for c in client.calls] == [30.0, 29.0, 27.0, 23.0]


def test_real_validation_retry_is_skipped_when_the_budget_is_spent(fake_clock, monkeypatch):
    client, _seen = install(monkeypatch, ['{"action": "dance"}', '{"action": "stop"}'], fake_clock, cost=4.8)
    with pytest.raises(llm.InvalidReply, match="budget is spent"):
        llm.chat(MSGS, "m", schema=ROUTER_SCHEMA, purpose="router", timeout=5)
    assert len(client.calls) == 1


def test_real_budget_spent_before_the_first_attempt(fake_clock, monkeypatch):
    client, _seen = install(monkeypatch, ["pong"])
    with pytest.raises(llm.ModelCallFailed, match="budget is spent"):
        llm.chat(MSGS, "m", timeout=0.1)
    assert client.calls == []


def test_real_empty_choices_is_a_failure(real, monkeypatch):
    client = StubClient([])
    client.chat.completions.create = lambda **kwargs: SimpleNamespace(choices=[], usage=None)
    monkeypatch.setattr(llm, "client_factory", lambda **kwargs: client)
    with pytest.raises(llm.ModelCallFailed, match="no choices"):
        llm.chat(MSGS, "m")


def test_ping_cli_in_fake_mode(fake, capsys, no_dotenv):
    assert llm.main(["--ping"]) == 0
    out = capsys.readouterr().out
    assert "router (" in out and "codegen (" in out and "pong (FAKE_LLM)" in out
    assert len(no_dotenv) == 1  # the CLI loads .env exactly once (and the test never let it read one)
