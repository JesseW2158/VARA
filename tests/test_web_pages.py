"""Regression tests for web/ (the dashboard and the voice page).

Static checks run everywhere. The behaviour checks run web/static/app.js and voice.js under node with the
small fake DOM in tests/web/ (start-up while a deferred script stalls, junk and 5 MB curves.json, Chart.js
arriving late, hung /stop and /state, 5000-character texts, trial rows and the "rejected:" label, non-integer
candidate indices, the DEMO countdown under clock skew, junk /state entries and a throwing render, T
forwarding to the voice iframe, the STOP latency measurement (the {source, t_client_ms} body, the console
line and the timeline entry in both orders of reply and event, an older reply shape, no performance.now, a
re-send, the voice page's numbers reaching the dashboard, pairing by source, latency_ms and
press_to_freeze_ms while other stops land in between, also when latency_ms is quantized to 0, the press
from the event's timeStamp, and a spoken stop timed from the push-to-talk release), the calibration result
with and without check_points, and the voice page's tool calls, task announcements, junk audio deltas and
forwarded keys); they are skipped when node is not installed. Nothing touches the network.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
INDEX = WEB / "index.html"
VOICE_HTML = WEB / "voice.html"
APP_JS = WEB / "static" / "app.js"
VOICE_JS = WEB / "static" / "voice.js"
STYLE = WEB / "static" / "style.css"
TEXT_FILES = sorted(p for p in WEB.rglob("*") if p.is_file() and p.suffix in {".html", ".js", ".css", ".md"})
VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
}
VOID.add("wbr")


def css_rules(css: str) -> dict[str, dict[str, str]]:
    """Selector -> declarations for every plain rule (rules inside @media blocks included)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        decls: dict[str, str] = {}
        for d in m.group(2).split(";"):
            if ":" in d:
                k, v = d.split(":", 1)
                decls[k.strip()] = v.strip()
        for sel in m.group(1).split(","):
            rules.setdefault(sel.strip(), {}).update(decls)
    return rules


def code_only(script: pathlib.Path) -> str:
    """The script without its // comments (comments may name the things the code must not use)."""
    return re.sub(r"//[^\n]*", "", script.read_text(encoding="utf-8"))


class _Checker(HTMLParser):
    """Balanced tags and unique ids, like the verifier's parse_check.py."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.ids: dict[str, int] = {}
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                self.problems.append(f"duplicate id {a['id']!r} at line {self.getpos()[0]}")
            self.ids[str(a["id"])] = self.getpos()[0]
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            self.problems.append(f"end tag for void element {tag!r} at line {self.getpos()[0]}")
        elif not self.stack or self.stack[-1][0] != tag:
            self.problems.append(f"unexpected end tag {tag!r} at line {self.getpos()[0]}")
        else:
            self.stack.pop()


@pytest.mark.parametrize("page", [INDEX, VOICE_HTML], ids=["index", "voice"])
def test_pages_parse_with_balanced_tags_and_unique_ids(page: pathlib.Path) -> None:
    c = _Checker()
    c.feed(page.read_text(encoding="utf-8"))
    c.close()
    problems = c.problems + [f"unclosed <{t}> from line {n}" for t, n in c.stack]
    assert not problems, problems
    assert len(c.ids) >= 9


@pytest.mark.parametrize("path", TEXT_FILES, ids=[str(p.relative_to(WEB)) for p in TEXT_FILES])
def test_pages_are_plain_ascii_and_hold_no_secrets(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    non_ascii = [i + 1 for i, line in enumerate(text.splitlines()) if any(ord(ch) > 127 for ch in line)]
    assert not non_ascii, f"non-ASCII characters on lines {non_ascii[:10]}"
    # The two key names from .env.example, the HTTP authorization scheme word and key-looking tokens.
    secret = r"xai_api_key|daytona_api_key|bearer |\bsk-[A-Za-z0-9]{10}|\bxai-[A-Za-z0-9]{10}"
    assert not re.search(secret, text, re.I)


@pytest.mark.parametrize("script", [APP_JS, VOICE_JS], ids=["app.js", "voice.js"])
def test_scripts_start_without_waiting_for_domcontentloaded(script: pathlib.Path) -> None:
    """DOMContentLoaded waits for deferred scripts (Chart.js from the CDN). A stalled CDN kept the old
    gate from ever running init(): no cards, no polls, and STOP/Esc dead for the whole stall."""
    code = code_only(script)
    assert "DOMContentLoaded" not in code
    assert "document.readyState" not in code  # ws.readyState in voice.js is a different thing
    tail = [line.strip() for line in code.splitlines() if line.strip()][-3:]
    assert "init();" in tail, tail


def test_chart_js_is_deferred_and_pinned_and_app_js_is_a_plain_script_after_the_markup() -> None:
    html = INDEX.read_text(encoding="utf-8")
    chart = re.search(r"<script[^>]*chart\.umd\.js[^>]*>", html)
    assert chart, "Chart.js tag missing"
    tag = chart.group(0)
    assert "/Chart.js/4.4.1/chart.umd.js" in tag
    assert 'integrity="sha384-' in tag and 'crossorigin="anonymous"' in tag
    assert re.search(r"\sdefer\b", tag), "Chart.js must not block the parser"
    app = re.search(r"<script[^>]*src=\"/static/app\.js\"[^>]*>", html)
    assert app, "app.js tag missing"
    assert not re.search(r"\s(defer|async)\b", app.group(0)), "app.js is a plain end-of-body script"
    assert app.start() > html.rfind('id="'), "every element with an id must precede the app.js tag"
    assert html.find("<script", app.end()) == -1, "app.js is the last script"


def test_text_boxes_wrap_unbroken_tokens_instead_of_widening_the_page() -> None:
    """A 600-character routing text without spaces widened the page to 7871 px at a 1366 px viewport."""
    rules = css_rules(STYLE.read_text(encoding="utf-8"))
    boxes = [
        ".headline",
        ".spec-english",
        ".card .hint",
        ".card .reason",
        ".timeline .body",
        ".queue-list li",
    ]
    for sel in boxes:
        assert rules.get(sel, {}).get("overflow-wrap") == "anywhere", sel
    assert "minmax(0, 1fr)" in rules[".timeline li"]["grid-template-columns"]
    inline = re.search(r"<style>(.*?)</style>", VOICE_HTML.read_text(encoding="utf-8"), re.S)
    assert inline, "voice.html inline style missing"
    voice_rules = css_rules(inline.group(1))
    assert voice_rules[".voice-log .text"]["overflow-wrap"] == "anywhere"


def test_card_status_never_wraps_and_the_local_badge_has_its_own_row() -> None:
    """At a 1366 px viewport the card status wrapped letter by letter beside the LOCAL badge (brief 8.7):
    the status keeps white-space: nowrap (an absurd one is clipped with an ellipsis, the full text sits in
    its title) and the badge lives in a .badges row under the header, which the node harness pins."""
    rules = css_rules(STYLE.read_text(encoding="utf-8"))
    status = rules[".card .status"]
    assert status["white-space"] == "nowrap"
    assert status["overflow"] == "hidden" and status["text-overflow"] == "ellipsis"
    assert rules[".card .badges"]["display"] == "flex"


def test_app_js_computes_chart_extremes_with_reduce_not_apply() -> None:
    """Math.max.apply(null, values) passes every value as a call argument: a 5 MB curves.json (400 000
    values) overflowed the call stack and lost the chart (brief 9.9)."""
    code = code_only(APP_JS)
    assert "Math.max.apply" not in code and "Math.min.apply" not in code
    draw = re.search(r"function drawChart\(.*?\n  }\n", code, re.S)
    assert draw and ".reduce(" in draw.group(0)


def test_voice_pending_tasks_have_no_prototype() -> None:
    """An event with task_id "__proto__" reached Object.prototype and "toString" passed as a pending task
    (brief 9.9): the map has no prototype and is read through an own-property check."""
    voice = VOICE_JS.read_text(encoding="utf-8")
    assert "var pendingTasks = Object.create(null);" in voice
    assert "Object.prototype.hasOwnProperty.call(pendingTasks, ev.task_id)" in voice


def test_stop_requests_carry_their_source_and_press_time() -> None:
    """Every STOP path (Esc, the STOP buttons, the voice `stop` tool) posts {source, t_client_ms} with the
    wall clock at the press (the server turns it into press_to_freeze_ms on the same laptop), measures the
    round trip on the monotonic clock from the key or click event's timeStamp, and logs one 'stop <source>:
    server <latency_ms> ms, round trip <ms> ms, press to freeze <ms> ms' line (an older server reports
    neither server number); a spoken stop adds the time from the push-to-talk release. The dashboard pairs
    a line with its own stop's timeline entry by source, latency_ms and press_to_freeze_ms (the server puts
    the same three in the reply and the event; latency_ms alone mostly reads 0 on a 15.6 ms server clock)."""
    for script in (APP_JS, VOICE_JS):
        code = code_only(script)
        assert "function pressStamp(ev)" in code and "ev.timeStamp" in code, script.name
        assert "{ source: wire, t_client_ms: press.wall }" in code, script.name
        assert '"server latency not reported"' in code, script.name
        assert '", round trip "' in code, script.name
        assert '", press to freeze "' in code, script.name
        assert '", talk release to response "' in code, script.name
        assert "performance.now()" in code, script.name
    app = code_only(APP_JS)
    assert 'stop("Esc", 0, pressStamp(ev))' in app and 'stop("button", 0, pressStamp(ev))' in app
    assert "stopKey(d.source, num(d.latency_ms), num(d.press_to_freeze_ms))" in app
    assert "stopKey(source, num(m.latency_ms), num(m.press_to_freeze_ms))" in app
    voice = code_only(VOICE_JS)
    assert 'doStop("Esc", pressStamp(ev))' in voice and 'doStop("button", pressStamp(ev))' in voice
    assert 'doStop("voice", null, releasedAt)' in voice
    assert "release_to_response_ms: r.ok ? sinceRelease : null" in voice


def test_calibration_result_has_two_lines_with_their_own_colours() -> None:
    """The independent check-points line (B5) sits under the marker line; each line carries its own ok/bad
    colour, and a missing or null check_points says so instead of hiding it."""
    rules = css_rules(STYLE.read_text(encoding="utf-8"))
    assert rules[".result .line"]["display"] == "block"
    assert rules[".result .line.ok"]["color"] == "var(--good)"
    assert rules[".result .line.bad"]["color"] == "var(--serious)"
    code = code_only(APP_JS)
    assert '"no independent check points recorded"' in code
    assert '"-point check: mean "' in code


def test_pages_behave_under_node() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed; the behaviour checks need it")
    proc = subprocess.run(
        [node, str(ROOT / "tests" / "web" / "app_checks.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0, report
    # The harness prints "FAIL <name>" at the start of a line; a check's name may quote the page's own
    # "(FAIL)" verdict text, so only a line-leading FAIL is a failed check.
    assert not re.search(r"^FAIL ", proc.stdout, re.M), report
    assert re.search(r"==== (\d+)/\1 checks passed ====", proc.stdout), report
    # The STOP latency and calibration scenarios ran (their check names carry these words).
    phrases = (
        "t_client_ms",
        "stop voice: server",
        "re-send 1",
        "check_points null",
        "talk release to response 1237 ms",
        "another tab's Esc",
        "latency_ms quantized to 0",
        "event's timeStamp",
    )
    for phrase in phrases:
        assert phrase in proc.stdout, phrase
