"""Model client: xAI's OpenAI-compatible chat API behind one function, plus a fixture-backed fake (C1).

`chat(messages, model, schema=None, timeout=30, *, purpose=None, key=None, task_id=None)` returns the
reply text, or a dict validated against `schema` when one is given.

Real path (no fake flag):
- `openai.OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)` and `chat.completions.create`, with
  temperature 0.2 for `purpose="router"` and 0.7 for `write_skill`, `repair` and `write_rewards`;
- with `schema`, the first attempt asks for `response_format={"type": "json_schema", ...}`; any 4xx other
  than 429 falls back to plain text, from which the first `{...}` block is parsed (`NaN` and `Infinity`
  are not JSON and never get through); the result is validated with jsonschema and, on failure, the model
  is asked once more with the error appended;
- `timeout` is the budget for the whole call, retries included: every HTTP attempt gets the time that is
  left, 429 and 5xx answers are retried up to 3 times with 1, 2 and 4 s of backoff while the budget allows
  it, and the validation retry is skipped when less than `MIN_ATTEMPT_S` remains (`sleep` and `clock` are
  module attributes, so tests replace them); connection errors are not retried, the router falls back to
  its keyword table instead;
- when `task_id` is given, one line {"t", "task_id", "model", "purpose", "prompt_tokens",
  "completion_tokens", "ms"} is appended to `runs/<task_id>/llm.jsonl` per completed call.

The API key is read from `os.environ` only inside the real path (`_make_client`); it is never logged,
printed, returned or put into an exception (error messages are scrubbed of it, including a failure of the
client factory itself). `client_factory` is a module attribute tests set to a stub:
`client_factory(api_key=..., base_url=..., timeout=...) -> client`.

Fake path (`FAKE_LLM=1` or `FAKE=1`): replies come from `tests/fixtures/llm/` and `openai` is never
imported. Layout:
- `aliases.json` maps a normalized phrase to a slug (`normalize`: lowercase, drop everything but letters,
  digits and spaces, collapse spaces, spaces -> `_`); an unmapped phrase is its own slug;
- `router/<slug>.json`: the router reply; an unknown phrase gives
  `{"action": "queue", "task": <text>, "reason": "no fixture for this phrase"}`;
- `candidates/<slug>/cand_<i>.py` (key `"<slug>:<i>"`, purpose `write_skill`) and
  `repair/<slug>/cand_<i>.py` (purpose `repair`, optional); the regeneration after a rejected candidate uses
  key `"<slug>:<i>:retry"` and `cand_<i>_retry.py`; any missing file falls back to
  `tests/fixtures/candidate_clear_table.py`. Sources come back inside a ```python fence so the real-path
  code extraction is exercised;
- `rewards/variant_<k>.py` (purpose `write_rewards`): key `"variant_<k>"` or `"<k>"` returns that file
  fenced; no key returns all variants, each in its own fence.
The fake still validates every reply against `schema`, so a bad fixture fails loudly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema

__all__ = [
    "BACKOFF_S",
    "DEFAULT_CODEGEN_MODEL",
    "DEFAULT_ROUTER_MODEL",
    "InvalidReply",
    "MIN_ATTEMPT_S",
    "ModelCallFailed",
    "PURPOSES",
    "TEMPERATURES",
    "XAI_BASE_URL",
    "chat",
    "client_factory",
    "clock",
    "codegen_model",
    "fence",
    "is_fake",
    "normalize",
    "parse_json_object",
    "router_model",
    "run_dir",
    "sleep",
]

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIXTURES = ROOT / "tests" / "fixtures" / "llm"
DEFAULT_CANDIDATE = ROOT / "tests" / "fixtures" / "candidate_clear_table.py"

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_ROUTER_MODEL = "grok-4-fast"
DEFAULT_CODEGEN_MODEL = "grok-4"
PURPOSES = ("router", "write_skill", "repair", "write_rewards")
TEMPERATURES = {"router": 0.2, "write_skill": 0.7, "repair": 0.7, "write_rewards": 0.7}
BACKOFF_S = (1.0, 2.0, 4.0)
MIN_ATTEMPT_S = 0.5  # no HTTP attempt, backoff or validation retry starts with less budget than this
RETRY_PROMPT = (
    "Your previous reply was not acceptable: {problem}\n"
    "Reply again with exactly one JSON object that follows the required schema, and nothing else."
)
_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_CANDIDATE_KEY_RE = re.compile(r"^(?P<slug>[A-Za-z0-9_-]+):(?P<index>\d+)(?::(?P<rest>[A-Za-z0-9_-]+))?$")

logger = logging.getLogger(__name__)

# Injection points for tests: a sleep replacement, the clock the budget is measured on, a client stub factory.
sleep: Callable[[float], None] = time.sleep
clock: Callable[[], float] = time.monotonic
client_factory: Callable[..., Any] | None = None

_LOG_LOCK = threading.Lock()
_VALIDATORS: dict[int, tuple[Any, jsonschema.Draft202012Validator]] = {}
_WARNED: set[str] = set()


class ModelCallFailed(RuntimeError):
    """The API call failed (after the retries, for 429 and 5xx). `status` is the HTTP status or None."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class InvalidReply(ValueError):
    """The model's reply could not be parsed or did not match the schema, even after one retry."""


def is_fake() -> bool:
    """True when `FAKE=1` or `FAKE_LLM=1` (checked at call time)."""
    return os.environ.get("FAKE") == "1" or os.environ.get("FAKE_LLM") == "1"


def router_model() -> str:
    """`ROUTER_MODEL` from the environment, or the default with a warning."""
    return _model_from_env("ROUTER_MODEL", DEFAULT_ROUTER_MODEL)


def codegen_model() -> str:
    """`CODEGEN_MODEL` from the environment, or the default with a warning."""
    return _model_from_env("CODEGEN_MODEL", DEFAULT_CODEGEN_MODEL)


def _model_from_env(var: str, default: str) -> str:
    value = os.environ.get(var, "").strip()
    if value:
        return value
    if var not in _WARNED:  # once per process, not once per call
        _WARNED.add(var)
        logger.warning("%s is not set; using the default %r (set it in .env)", var, default)
    return default


def run_dir(task_id: str) -> Path:
    """`runs/<task_id>` for a plain task id (ValueError otherwise); the directory is not created."""
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"task_id {task_id!r} is not a plain identifier")
    return RUNS / task_id


def normalize(text: str) -> str:
    """Lowercase, keep only letters, digits and spaces, collapse spaces, then spaces -> `_`."""
    lowered = re.sub(r"\s+", " ", str(text).lower())
    # `_` counts as a space so that a slug normalizes to itself (aliases.json keys are already normalized).
    kept = "".join(" " if ch == "_" else ch for ch in lowered if ch.isalnum() or ch in " _")
    return "_".join(kept.split())


def fence(source: str) -> str:
    """Wrap a source in a ```python fence, as the code-writing model does."""
    body = source.rstrip("\n")
    return f"```python\n{body}\n```"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


# Strict JSON: Python's decoder would otherwise accept NaN and Infinity, which jsonschema then passes as
# numbers, and a spec with a NaN coordinate can never be satisfied.
_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def parse_json_object(text: str) -> dict[str, Any]:
    """The first JSON object in `text` (the whole text, a fenced block or an embedded `{...}`).

    An object holding `NaN` or `Infinity` is skipped like any other invalid candidate."""
    for match in re.finditer(r"\{", text):
        try:
            value, _end = _DECODER.raw_decode(text, match.start())
        except ValueError:  # JSONDecodeError, or the NaN/Infinity rejection
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("the reply holds no JSON object")


def chat(
    messages: Sequence[Mapping[str, Any]],
    model: str,
    schema: Mapping[str, Any] | None = None,
    timeout: float = 30,
    *,
    purpose: str | None = None,
    key: str | None = None,
    task_id: str | None = None,
) -> str | dict[str, Any]:
    """One chat completion. Returns the text, or a dict validated against `schema` when given.

    `purpose` picks the temperature and, in fake mode, the fixture family; `key` selects the fixture (the
    phrase for the router, `"<slug>:<i>"` for candidates); `task_id` enables `runs/<task_id>/llm.jsonl`;
    `timeout` is the budget in seconds for the whole call, retries included.
    """
    msgs = _check_messages(messages)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if task_id is not None and not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"task_id {task_id!r} is not a plain identifier")
    budget = _budget(timeout)
    if is_fake():
        return _fake_chat(msgs, model, schema, purpose, key, task_id)
    return _real_chat(msgs, model, schema, budget, purpose, task_id)


def _budget(timeout: Any) -> float:
    try:
        budget = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}") from None
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
    return budget


def _check_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes, Mapping)) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    out: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping) or "role" not in item or "content" not in item:
            raise ValueError("every message needs a 'role' and a 'content'")
        out.append(dict(item))
    return out


# --- real path ---------------------------------------------------------------------------------------


def _real_chat(
    msgs: list[dict[str, Any]],
    model: str,
    schema: Mapping[str, Any] | None,
    timeout: float,
    purpose: str | None,
    task_id: str | None,
) -> str | dict[str, Any]:
    client = _make_client(timeout)
    deadline = clock() + timeout
    temperature = _temperature(purpose, schema)
    structured = schema is not None
    for validation_round in (1, 2):
        text, usage, ms, structured = _complete_with_fallback(
            client, model, msgs, temperature, schema if structured else None, timeout, deadline, purpose
        )
        _log(task_id, model, purpose, usage, ms)
        if schema is None:
            return text
        try:
            data = parse_json_object(text)
            _validator(schema).validate(data)
            return data
        except (ValueError, jsonschema.ValidationError) as exc:
            problem = _scrub(_describe(exc))
        # Raised outside the except block so the original (which quotes the reply) is not chained.
        if validation_round == 2:
            raise InvalidReply(f"model reply rejected twice: {problem}")
        if deadline - clock() < MIN_ATTEMPT_S:
            raise InvalidReply(f"model reply rejected and the {timeout:g} s budget is spent: {problem}")
        logger.warning("model reply rejected (%s); asking once more", problem)
        msgs = [
            *msgs,
            {"role": "assistant", "content": text},
            {"role": "user", "content": RETRY_PROMPT.format(problem=problem)},
        ]
    raise AssertionError("unreachable")


def _temperature(purpose: str | None, schema: Mapping[str, Any] | None) -> float:
    if purpose in TEMPERATURES:
        return TEMPERATURES[purpose]
    return 0.2 if schema is not None else 0.7


def _make_client(timeout: float) -> Any:
    """Build the client. The only place the key is read; it goes to the factory and nowhere else."""
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key.strip():
        raise ModelCallFailed("XAI_API_KEY is not set; put it in .env or run with FAKE_LLM=1")
    factory = client_factory or _openai_client
    try:
        return factory(api_key=api_key, base_url=XAI_BASE_URL, timeout=timeout)
    except Exception as exc:  # the factory's own message may quote the key: scrub it, chain nothing
        message = _scrub(f"could not build the model client: {type(exc).__name__}: {exc}")
    raise ModelCallFailed(message)


def _openai_client(*, api_key: str, base_url: str, timeout: float) -> Any:
    import openai  # imported lazily: never in fake mode

    return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def _response_format(schema: Mapping[str, Any], purpose: str | None) -> dict[str, Any]:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", purpose or "reply")[:64] or "reply"
    return {"type": "json_schema", "json_schema": {"name": name, "schema": dict(schema)}}


def _complete_with_fallback(
    client: Any,
    model: str,
    msgs: list[dict[str, Any]],
    temperature: float,
    schema: Mapping[str, Any] | None,
    timeout: float,
    deadline: float,
    purpose: str | None,
) -> tuple[str, tuple[int, int], float, bool]:
    """Structured output first; on a 4xx (other than 429) fall back to plain text."""
    if schema is not None:
        try:
            response_format = _response_format(schema, purpose)
            text, usage, ms = _complete(client, model, msgs, temperature, response_format, timeout, deadline)
            return text, usage, ms, True
        except ModelCallFailed as exc:
            if exc.status is None or not (400 <= exc.status < 500) or exc.status == 429:
                raise
            logger.warning("structured output refused (HTTP %s); falling back to plain text", exc.status)
    text, usage, ms = _complete(client, model, msgs, temperature, None, timeout, deadline)
    return text, usage, ms, False


def _complete(
    client: Any,
    model: str,
    msgs: list[dict[str, Any]],
    temperature: float,
    response_format: Mapping[str, Any] | None,
    timeout: float,
    deadline: float,
) -> tuple[str, tuple[int, int], float]:
    """One completion with up to 3 retries (1, 2, 4 s) on 429 and 5xx, all inside the budget that ends at
    `deadline` (on `clock`): each attempt gets the time left, and a retry whose backoff would leave less
    than `MIN_ATTEMPT_S` is not made. Raises ModelCallFailed."""
    kwargs: dict[str, Any] = {"model": model, "messages": msgs, "temperature": temperature}
    if response_format is not None:
        kwargs["response_format"] = dict(response_format)
    retries = len(BACKOFF_S)
    for attempt in range(retries + 1):
        remaining = deadline - clock()
        if remaining < MIN_ATTEMPT_S:
            raise ModelCallFailed(f"the {timeout:g} s budget is spent before attempt {attempt + 1}")
        kwargs["timeout"] = round(min(timeout, remaining), 2)
        started = clock()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # only its scrubbed text survives: the API's message may quote the key
            status = _status_of(exc)
            message = _scrub(f"{type(exc).__name__}: {exc}")
        else:
            ms = (clock() - started) * 1000.0
            return _content_of(response), _usage_of(response), ms
        retryable = status is not None and (status == 429 or status >= 500)
        if retryable and attempt < retries:
            delay = BACKOFF_S[attempt]
            if deadline - clock() - delay >= MIN_ATTEMPT_S:
                logger.warning(
                    "model call failed (HTTP %s); retry %d of %d in %.0f s",
                    status,
                    attempt + 1,
                    retries,
                    delay,
                )
                sleep(delay)
                continue
            message = f"gave up after {attempt + 1} attempts, the {timeout:g} s budget is spent: {message}"
        elif retryable:
            message = f"gave up after {retries} retries: {message}"
        raise ModelCallFailed(message, status)  # outside the except block: nothing chained
    raise AssertionError("unreachable")


def _status_of(exc: BaseException) -> int | None:
    """HTTP status of an API error (duck-typed: openai's errors and test stubs both carry it)."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _content_of(response: Any) -> str:
    choices = _field(response, "choices") or []
    if not choices:
        raise ModelCallFailed("the model returned no choices")
    message = _field(choices[0], "message")
    content = _field(message, "content") if message is not None else None
    if content is None:
        refusal = _field(message, "refusal") if message is not None else None
        if refusal:
            raise ModelCallFailed(f"the model refused: {_scrub(str(refusal))}")
        return ""
    return str(content)


def _usage_of(response: Any) -> tuple[int, int]:
    usage = _field(response, "usage")
    if usage is None:
        return 0, 0
    return _int(_field(usage, "prompt_tokens")), _int(_field(usage, "completion_tokens"))


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _describe(exc: BaseException) -> str:
    if isinstance(exc, jsonschema.ValidationError):
        where = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        return f"schema violation at {where}: {exc.message}"
    return str(exc)


def _scrub(message: str) -> str:
    """Remove the API key from any text that might be logged or raised."""
    key = os.environ.get("XAI_API_KEY", "")
    if key and key in message:
        return message.replace(key, "[XAI_API_KEY]")
    return message


def _validator(schema: Mapping[str, Any]) -> jsonschema.Draft202012Validator:
    entry = _VALIDATORS.get(id(schema))
    if entry is None or entry[0] is not schema:
        entry = (schema, jsonschema.Draft202012Validator(schema))
        _VALIDATORS[id(schema)] = entry
    return entry[1]


def _log(task_id: str | None, model: str, purpose: str | None, usage: tuple[int, int], ms: float) -> None:
    if not task_id:
        return
    line = {
        "t": time.time(),
        "task_id": task_id,
        "model": model,
        "purpose": purpose,
        "prompt_tokens": usage[0],
        "completion_tokens": usage[1],
        "ms": round(ms, 1),
    }
    path = RUNS / task_id / "llm.jsonl"
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")


# --- fake path ---------------------------------------------------------------------------------------


def _fake_chat(
    msgs: list[dict[str, Any]],
    model: str,
    schema: Mapping[str, Any] | None,
    purpose: str | None,
    key: str | None,
    task_id: str | None,
) -> str | dict[str, Any]:
    started = time.monotonic()
    if purpose == "router":
        phrase = key if key is not None else _last_user_text(msgs)
        text = json.dumps(_fake_router_reply(phrase))
    elif purpose in ("write_skill", "repair"):
        text = fence(_fake_candidate_source(purpose, key))
    elif purpose == "write_rewards":
        text = _fake_rewards_reply(key)
    elif schema is None:
        text = "pong (FAKE_LLM)"
    else:
        raise ValueError(f"FAKE_LLM has no fixtures for purpose {purpose!r} with a schema")
    prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)
    usage = (prompt_chars // 4, len(text) // 4)  # a rough token estimate: the fake has no tokenizer
    _log(task_id, model, purpose, usage, (time.monotonic() - started) * 1000.0)
    if schema is None:
        return text
    data = parse_json_object(text)
    _validator(schema).validate(data)  # a bad fixture fails loudly
    return data


def _last_user_text(msgs: list[dict[str, Any]]) -> str:
    for message in reversed(msgs):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(msgs[-1].get("content", ""))


def _fake_router_reply(phrase: str) -> dict[str, Any]:
    slug = normalize(phrase)
    aliases_path = FIXTURES / "aliases.json"
    if aliases_path.is_file():
        aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        slug = str(aliases.get(slug, slug))
    path = FIXTURES / "router" / f"{slug}.json" if slug else None
    if path is None or not path.is_file():
        return {"action": "queue", "task": phrase or "(empty)", "reason": "no fixture for this phrase"}
    reply = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(reply, dict) and reply.get("action") == "queue" and phrase:
        reply = {**reply, "task": phrase}  # the fixture's task text stands in for the user's words
    return reply


def _fake_candidate_source(purpose: str, key: str | None) -> str:
    folder = "repair" if purpose == "repair" else "candidates"
    match = _CANDIDATE_KEY_RE.match(key or "")
    if match:
        slug, index, rest = match.group("slug"), match.group("index"), match.group("rest")
        name = f"cand_{index}_retry.py" if rest == "retry" else f"cand_{index}.py"
        path = FIXTURES / folder / slug / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return DEFAULT_CANDIDATE.read_text(encoding="utf-8")


def _fake_rewards_reply(key: str | None) -> str:
    folder = FIXTURES / "rewards"
    if key:
        match = re.fullmatch(r"(?:variant_)?(\d+)", key.strip())
        if match:
            path = folder / f"variant_{match.group(1)}.py"
            if path.is_file():
                return fence(path.read_text(encoding="utf-8"))
    files = sorted(folder.glob("variant_*.py"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not files:
        raise FileNotFoundError(f"no reward fixtures under {folder}")
    parts = [f"Variant {i}:\n{fence(path.read_text(encoding='utf-8'))}" for i, path in enumerate(files, 1)]
    return "\n\n".join(parts)


# --- CLI ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agents.llm --ping`: one short reply from the router and the codegen model."""
    parser = argparse.ArgumentParser(prog="python -m agents.llm", description="xAI model client")
    parser.add_argument("--ping", action="store_true", help="ask both models for a one-word reply")
    args = parser.parse_args(argv)
    if not args.ping:
        parser.print_help()
        return 2
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    ok = True
    for label, model in (("router", router_model()), ("codegen", codegen_model())):
        started = time.monotonic()
        try:
            reply = chat([{"role": "user", "content": "Reply with the single word: pong"}], model, timeout=15)
        except Exception as exc:  # a ping reports every failure, scrubbed of the key
            print(f"{label} ({model}): FAILED {type(exc).__name__}: {_scrub(str(exc))}")
            ok = False
            continue
        ms = (time.monotonic() - started) * 1000.0
        print(f"{label} ({model}): {str(reply).strip()[:80]!r} in {ms:.0f} ms")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
