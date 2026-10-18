"""C5/C6 inventory: `agents.sandbox_pool` lists and deletes only vara-labelled sandboxes, waits for deletes
to land, enforces the cap against the whole account in `start()`, and its `--list`/`--cleanup` CLI writes
inventory JSON. A fake Daytona client stands in; nothing here touches the network, a key, or the repo's
runs/ (every file lands under tmp_path)."""

from __future__ import annotations

import json
import pathlib
import re
from types import SimpleNamespace

import pytest

from agents import sandbox_pool as sp


class Listed:
    """A sandbox as `Daytona.list()` yields it: id, name, state, labels, snapshot, created_at, delete()."""

    def __init__(
        self,
        id: str,
        labels: dict | None,
        *,
        state: str = "started",
        created_at: str = "2026-10-17T20:00:00Z",
        fail: bool = False,
    ) -> None:
        self.id = id
        self.name = f"name-{id}"
        self.labels = labels
        self.state = state
        self.snapshot = "arm-sim-v1"
        self.created_at = created_at
        self.fail = fail
        self.deletes: list[tuple] = []

    def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        self.deletes.append((timeout, wait))
        if self.fail:
            raise RuntimeError("delete broke")
        self.state = "destroyed"


class Box(Listed):
    """A sandbox `create()` returns: enough `fs`/`process` for the bundle install."""

    def __init__(self, number: int) -> None:
        super().__init__(f"sb-{number}", {"vara": "pool"})
        self.fs = SimpleNamespace(upload_file=lambda data, path: None)
        self.process = SimpleNamespace(
            exec=lambda command, cwd=None, timeout=None: SimpleNamespace(exit_code=0, result="")
        )


class FakeClient:
    """`list()` yields the listed sandboxes (optionally failing, or mutating per call); `create()` adds a
    Box."""

    def __init__(self, listed: list[Listed] | None = None) -> None:
        self.listed = list(listed or [])
        self.list_calls = 0
        self.list_error: Exception | None = None
        self.on_list = None  # callable(call_number) run before each listing
        self.create_calls: list = []

    def list(self, query=None):
        self.list_calls += 1
        if self.on_list is not None:
            self.on_list(self.list_calls)
        if self.list_error is not None:
            raise self.list_error
        yield from list(self.listed)

    def create(self, params, *, timeout: float = 60, **_: object) -> Box:
        self.create_calls.append(params)
        box = Box(len(self.create_calls))
        self.listed.append(box)
        return box


def account() -> list[Listed]:
    return [
        Listed("pool-a", {"vara": "pool"}, created_at="2026-10-17T20:00:01Z"),
        Listed("pool-b", {"vara": "pool"}, created_at="2026-10-17T20:00:00Z"),
        Listed("rl-c", {"vara": "rl", "variant": "v1"}),
        Listed("rl-check-d", {"vara": "rl-check"}),
        Listed("plain-e", None),
        Listed("other-f", {"team": "vara"}),
    ]


@pytest.fixture
def client(monkeypatch) -> FakeClient:
    fake = FakeClient(account())
    monkeypatch.setattr(sp, "_daytona_factory", lambda: fake)
    return fake


# ------------------------------------------------------------------------------------------ listing


def test_inventory_lists_only_vara_labelled_sandboxes_as_plain_records(client):
    records = sp.list_vara_sandboxes()  # default client from the factory, any vara value
    assert [r["id"] for r in records] == ["pool-b", "pool-a", "rl-c", "rl-check-d"]  # oldest first per label
    assert set(records[0]) == {"id", "name", "state", "labels", "snapshot", "created_at"}
    assert records[0]["labels"] == {"vara": "pool"} and records[0]["state"] == "started"
    assert [r["id"] for r in sp.list_vara_sandboxes(client, labels=("pool",))] == ["pool-b", "pool-a"]
    assert [r["id"] for r in sp.list_vara_sandboxes(client, labels="rl")] == ["rl-c"]
    assert sp.list_vara_sandboxes(client, labels=("nothing-like-this",)) == []
    with pytest.raises(ValueError):
        sp.list_vara_sandboxes(client, labels=())
    with pytest.raises(ValueError):
        sp.list_vara_sandboxes(client, labels=("",))


def test_inventory_sandbox_record_tolerates_enums_datetimes_and_missing_attributes():
    class State:
        value = "started"

    class When:
        def isoformat(self) -> str:
            return "2026-10-17T21:00:00+00:00"

    sandbox = SimpleNamespace(id="x", state=State(), labels={"vara": "pool"}, created_at=When())
    record = sp.sandbox_record(sandbox)
    assert record["state"] == "started" and record["created_at"] == "2026-10-17T21:00:00+00:00"
    assert record["name"] is None and record["snapshot"] is None
    assert sp.sandbox_record(object())["labels"] == {} and sp.sandbox_record(object())["id"] is None


def test_inventory_present_records_drop_destroyed_ones_and_active_records_also_destroying_ones():
    records = [
        {"id": "a", "state": "destroyed"},
        {"id": "b", "state": "destroying"},
        {"id": "c", "state": None},
    ]
    assert [r["id"] for r in sp.present_records(records)] == ["b", "c"]
    assert [r["id"] for r in sp.active_records(records)] == ["c"]


def test_inventory_label_constants_cover_the_pool_and_both_rl_values():
    assert sp.VARA_LABELS == ("pool", "rl", "rl-check")
    assert sp.LABEL_SCOPES == {"pool": ("pool",), "rl": ("rl", "rl-check"), "all": sp.VARA_LABELS}
    assert (sp.LABEL_KEY, sp.POOL_LABEL) == ("vara", "pool")  # what _sandbox_params puts on every sandbox


def test_inventory_report_counts_every_value_but_judges_only_the_scope():
    records = [
        {"id": "p", "state": "started", "labels": {"vara": "pool"}},
        {"id": "r", "state": "started", "labels": {"vara": "rl"}},
        {"id": "x", "state": "started", "labels": {"vara": "experiment"}},
        {"id": "g", "state": "destroyed", "labels": {"vara": "pool"}},
    ]
    report = sp.inventory_report(records, scope=("rl",))
    assert report["scope"] == ["rl"] and report["remaining"] == 1
    assert report["counts"] == {"pool": 1, "rl": 1, "rl-check": 0, "experiment": 1}
    assert [r["id"] for r in report["sandboxes"]] == ["p", "r", "x"] and report["deleted"] == []
    with pytest.raises(ValueError):
        sp.inventory_report(records, scope=())


# ------------------------------------------------------------------------------------------ deleting


def test_inventory_delete_touches_only_the_scope_and_keeps_going_after_a_failure(client):
    client.listed[0].fail = True  # pool-a
    deleted = sp.delete_vara_sandboxes(client)  # default scope: pool only
    assert [(r["id"], r["deleted"]) for r in deleted] == [("pool-b", True), ("pool-a", False)]
    assert "delete broke" in deleted[1]["error"] and deleted[0]["error"] is None
    assert client.listed[1].deletes == [(60.0, True)]  # waits for the destroy by default
    for box in client.listed[2:]:
        assert box.deletes == [], box.id  # rl, rl-check, unlabelled and other-label sandboxes untouched
    assert [r["id"] for r in sp.delete_vara_sandboxes(client, labels="rl", wait=False)] == ["rl-c"]
    assert client.listed[2].deletes == [(60.0, False)]


# ------------------------------------------------------------------------------------------ waiting


def test_inventory_wait_polls_only_while_a_sandbox_is_destroying(client):
    client.listed = [Listed("pool-z", {"vara": "pool"}, state="destroying")]

    def settle(call: int) -> None:
        if call >= 3:
            client.listed[0].state = "destroyed"

    client.on_list = settle
    assert sp.wait_for_sandboxes_gone(client, timeout=5.0, poll=0.001) == []
    assert client.list_calls == 3
    client.on_list = None
    client.listed = [Listed("pool-y", {"vara": "pool"}, state="started")]
    client.list_calls = 0
    remaining = sp.wait_for_sandboxes_gone(client, timeout=5.0, poll=0.001)
    assert [r["id"] for r in remaining] == ["pool-y"] and client.list_calls == 1  # live: no waiting
    assert sp.wait_for_sandboxes_gone(client, ids=["someone-else"], timeout=5.0, poll=0.001) == []
    client.listed[0].state = "destroying"
    stuck = sp.wait_for_sandboxes_gone(client, timeout=0.0, poll=0.001)
    assert [r["state"] for r in stuck] == ["destroying"]  # timeout: reported, not hidden


def test_inventory_after_shutdown_reports_ours_gone_and_the_remaining_count(client):
    client.listed = [
        Listed("ours", {"vara": "pool"}, state="destroyed"),
        Listed("theirs", {"vara": "pool"}),
        Listed("rl-c", {"vara": "rl"}),
    ]
    after = sp.inventory_after_shutdown(client, ["ours"], timeout=0.0)
    assert after["ours_gone"] is True and after["remaining"] == 1
    assert [r["id"] for r in after["sandboxes"]] == ["theirs"]
    client.listed[0].state = "started"
    assert sp.inventory_after_shutdown(client, ["ours"], timeout=0.0)["ours_gone"] is False
    client.list_error = RuntimeError("api down")
    assert sp.inventory_after_shutdown(client, ["ours"], timeout=0.0) is None


# ------------------------------------------------------------------------------------------ start() guard


@pytest.fixture
def quiet_pool(monkeypatch, tmp_path):
    """A SandboxPool on an empty bundle root with the process-wide exit handlers stubbed out."""
    monkeypatch.setattr(sp, "atexit", SimpleNamespace(register=lambda fn: None))
    monkeypatch.setattr(sp, "_handlers_installed", True)
    monkeypatch.setattr(sp, "_pools", [])
    (tmp_path / "bundle").mkdir()

    def make(client: FakeClient) -> sp.SandboxPool:
        monkeypatch.setattr(sp, "_daytona_factory", lambda: client)
        return sp.SandboxPool(bundle_root=tmp_path / "bundle")

    return make


def test_inventory_guard_refuses_when_the_account_already_holds_seven_pool_sandboxes(quiet_pool):
    client = FakeClient([Listed(f"old-{i}", {"vara": "pool"}) for i in range(7)])
    pool = quiet_pool(client)
    with pytest.raises(ValueError, match="cap"):
        pool.start()
    assert client.create_calls == [] and pool.sandboxes == []
    pool.shutdown()


def test_inventory_guard_counts_only_pool_sandboxes_and_not_destroyed_ones(quiet_pool):
    client = FakeClient(
        [Listed(f"old-{i}", {"vara": "pool"}) for i in range(5)]
        + [Listed("gone", {"vara": "pool"}, state="destroyed"), Listed("rl", {"vara": "rl"})]
    )
    pool = quiet_pool(client)
    with pytest.raises(ValueError):
        pool.start(n_score=2, n_plan=1)  # 5 + 3 > 7
    assert client.create_calls == []
    assert pool.start(n_score=2, n_plan=0) == 2  # 5 + 2 == 7: allowed
    assert len(client.create_calls) == 2
    pool.shutdown()


def test_inventory_guard_ignores_a_listing_failure_or_a_client_without_list(quiet_pool):
    client = FakeClient()
    client.list_error = RuntimeError("api hiccup")
    pool = quiet_pool(client)
    assert pool.start(n_score=2, n_plan=0) == 2
    pool.shutdown()
    bare = SimpleNamespace(create=client.create)  # the pre-existing fake shape: no list() at all
    pool = quiet_pool(bare)
    assert pool.start(n_score=1, n_plan=0) == 1
    pool.shutdown()


# ------------------------------------------------------------------------------------------ CLI


def test_inventory_cli_refuses_without_a_key_or_under_a_fake_flag_before_building_a_client(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(sp, "_daytona_factory", lambda: (_ for _ in ()).throw(AssertionError("no client")))
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_SANDBOX", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert sp.main(["--list"]) == 2
    err = capsys.readouterr().err
    assert "DAYTONA_API_KEY is not set (present: False)" in err and "nothing listed" in err
    monkeypatch.setenv("DAYTONA_API_KEY", "dummy-daytona-key-0123")
    monkeypatch.setenv("FAKE_SANDBOX", "1")
    assert sp.main(["--cleanup"]) == 2
    captured = capsys.readouterr()
    assert "FAKE or FAKE_SANDBOX is 1" in captured.err
    assert "dummy-daytona-key-0123" not in captured.out + captured.err  # only presence, never the value
    monkeypatch.delenv("FAKE_SANDBOX")
    monkeypatch.setenv("FAKE", "1")
    assert sp.main(["--list"]) == 2
    assert not (tmp_path / "runs").exists()  # a refusal writes nothing


@pytest.mark.parametrize(
    "argv",
    [
        ["--list", "--bench"],
        ["--cleanup", "--build-snapshot"],
        ["--list", "--label", "everything"],
        ["--list", "--wait", "-1"],
        ["--bench", "--trials", "0"],
    ],
)
def test_inventory_cli_misuse_exits_2(argv, monkeypatch):
    monkeypatch.setattr(sp, "_daytona_factory", lambda: (_ for _ in ()).throw(AssertionError("no client")))
    with pytest.raises(SystemExit) as exc:
        sp.main(argv)
    assert exc.value.code == 2


def _written(text: str) -> dict:
    """The JSON file the CLI names in its last line (`... written <path>`)."""
    match = re.search(r"; written (.+)$", text, re.M)
    assert match, text
    return json.loads(pathlib.Path(match.group(1).strip()).read_text(encoding="utf-8"))


def test_inventory_cli_lists_cleans_and_writes_json(client, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_SANDBOX", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    out = tmp_path / "sandboxes"
    assert sp.main(["--list", "--out", str(out), "--wait", "0"]) == 1  # two pool sandboxes remain
    text = capsys.readouterr().out
    assert "pool-a" in text and "pool-b" in text and "rl-c" in text  # the table shows every vara value
    assert "plain-e" not in text and "other-f" not in text  # never an unlabelled or foreign sandbox
    assert "counts: vara=pool 2, vara=rl 1, vara=rl-check 1" in text
    assert "remaining (pool): 2" in text
    assert len(list(out.glob("inventory-*.json"))) == 1
    report = _written(text)
    assert report["scope"] == ["pool"] and report["remaining"] == 2
    assert report["counts"] == {"pool": 2, "rl": 1, "rl-check": 1} and report["deleted"] == []
    assert {r["id"] for r in report["sandboxes"]} == {"pool-a", "pool-b", "rl-c", "rl-check-d"}

    assert sp.main(["--cleanup", "--out", str(out), "--wait", "0"]) == 0
    text = capsys.readouterr().out
    assert "deleted pool-a (vara=pool): ok" in text and "remaining (pool): 0" in text
    cleanup = _written(text)
    assert {r["id"] for r in cleanup["deleted"]} == {"pool-a", "pool-b"} and cleanup["remaining"] == 0
    assert all(r["deleted"] for r in cleanup["deleted"])
    assert len(list(out.glob("cleanup-*.json"))) == 1
    for box in client.listed[2:]:
        assert box.deletes == [], box.id  # rl, rl-check, unlabelled and foreign sandboxes untouched

    assert sp.main(["--list", "--out", str(out), "--wait", "0"]) == 0
    assert sp.main(["--list", "--label", "rl", "--out", str(out), "--wait", "0"]) == 1
    assert "remaining (rl/rl-check): 2" in capsys.readouterr().out
    assert sp.main(["--list", "--label", "all", "--out", str(out), "--wait", "0"]) == 1
    report = _written(capsys.readouterr().out)
    assert report["scope"] == ["pool", "rl", "rl-check"] and report["remaining"] == 2
    assert report["counts"] == {"pool": 0, "rl": 1, "rl-check": 1}
    assert len(list(out.glob("inventory-*.json"))) == 4  # a new file per call, never overwritten


def test_inventory_cli_cleanup_exits_1_when_a_delete_fails_or_the_listing_breaks(
    client, monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_SANDBOX", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    client.listed[0].fail = True  # pool-a cannot be deleted
    assert sp.main(["--cleanup", "--out", str(tmp_path / "s"), "--wait", "0"]) == 1
    text = capsys.readouterr().out
    assert "deleted pool-a (vara=pool): FAILED RuntimeError: delete broke" in text
    assert "remaining (pool): 1" in text
    client.list_error = RuntimeError("api down")
    assert sp.main(["--list", "--out", str(tmp_path / "s"), "--wait", "0"]) == 1
    assert "--list failed: RuntimeError: api down" in capsys.readouterr().err


def test_inventory_cli_uses_runs_dir_from_the_environment(client, monkeypatch, tmp_path):
    monkeypatch.delenv("FAKE", raising=False)
    monkeypatch.delenv("FAKE_SANDBOX", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "not-a-real-key")
    monkeypatch.setenv("VARA_RUNS_DIR", str(tmp_path / "runs"))
    client.listed = []
    assert sp.main(["--list", "--wait", "0"]) == 0
    assert list((tmp_path / "runs" / "sandboxes").glob("inventory-*.json"))
    assert sp.runs_dir() == pathlib.Path(tmp_path / "runs")
