"""docs/gates.md keeps one row per gate in docs/schedule.md and enforces its own ticking rule."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "docs" / "schedule.md"
LEDGER = ROOT / "docs" / "gates.md"
GATE_LINE = re.compile(r"^- \[(?P<tick>[ xX])\] \*\*(?P<gate>.+?), (?:Fri|Sat|Sun) ")
STATUS = re.compile(r"^(open|partial|pass|fail|cut-[1-4])$")
COLUMNS = ["Gate", "Deadline", "Test", "Status", "Local timestamp", "Who", "Evidence path", "Notes"]


def _schedule_gates() -> list[tuple[str, bool]]:
    gates = []
    for line in SCHEDULE.read_text(encoding="utf-8").splitlines():
        match = GATE_LINE.match(line)
        if match:
            gates.append((match["gate"], match["tick"] != " "))
    return gates


def _ledger_rows() -> tuple[list[str], list[dict[str, str]]]:
    """The header and rows of the table under '## Gates' (stops at the next heading)."""
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Gates")
    table = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            table.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    header, rows = table[0], table[2:]
    return header, [dict(zip(header, row, strict=True)) for row in rows]


def _row_for(gate: str, rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [r for r in rows if r["Gate"] == gate or r["Gate"].startswith(f"{gate} (")]
    assert len(matches) == 1, f"{gate}: expected one ledger row, found {len(matches)}"
    return matches[0]


def test_every_schedule_gate_has_one_ledger_row_in_schedule_order() -> None:
    gates = [gate for gate, _ in _schedule_gates()]
    assert gates[:3] == ["S0", "I1", "M1a"] and gates[-1] == "M8" and len(gates) == 12
    header, rows = _ledger_rows()
    assert header == COLUMNS
    order = [rows.index(_row_for(gate, rows)) for gate in gates]
    assert order == sorted(order)
    assert any(r["Gate"] == "I2" for r in rows)


def test_statuses_use_the_ticking_rule_vocabulary_and_pass_rows_are_complete() -> None:
    _, rows = _ledger_rows()
    for row in rows:
        assert STATUS.match(row["Status"]), f"{row['Gate']}: status {row['Status']!r}"
        if row["Status"] == "pass":
            assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}", row["Local timestamp"]), row["Gate"]
            assert row["Who"] and "runs/" in row["Evidence path"], row["Gate"]
    assert _row_for("S0", rows)["Status"] == "pass"


def test_a_ticked_schedule_gate_has_a_pass_row() -> None:
    _, rows = _ledger_rows()
    for gate, ticked in _schedule_gates():
        if ticked:
            assert _row_for(gate, rows)["Status"] == "pass", f"{gate} is ticked in schedule.md"
