"""``/api/schedules`` on the wire: the phone's read of what is armed.

WHY THIS FILE EXISTS. ``GET /api/schedules`` is the relay's half of the desktop
listings (``GET /v1/desktop/wakes`` / ``GET /v1/desktop/monitors``), and the
phone's Schedules surface is what consumes it — so these assertions are about
what THE PHONE IS TOLD, over the real app and the real indexes, not about the
stores (which have their own suites). The desktop listings' own boundaries are
covered by ``tests/unit/server/test_server_wakes_listing.py`` and
``..._monitors_listing.py``; where a cell matters on both surfaces it is
asserted here against the same shape, because "field for field the desktop's"
is only true while it is tested.

The two rules worth pinning here, because both are invisible in a green unit
suite and load-bearing on the client:

* ``GET /api/schedules`` is INDEX-BACKED: it must answer with no runtime
  running at all, because a schedule outlives the runtime it was armed from.
* The two EMPTY answers are different claims: a store with nothing armed is a
  statement about the user's machine, and a store this process could not read
  is a statement about the read. ``read_error`` is how the wire tells them
  apart — an unreadable index must never render as "nothing is armed".

Nothing here touches a real session: the config root is the isolated one
conftest installs, and every entry written is this test's own fixture.
"""

from __future__ import annotations

import json
import time
from typing import Any

from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir

SESSION_A = "aaa111222333"
SESSION_B = "bbb444555666"
FUTURE = int(time.time() * 1000) + 3_600_000
PAST = int(time.time() * 1000) - 60_000


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _body(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/schedules")
    assert response.status_code == 200, response.text
    return response.json()


def _session(session_id: str, *, transcript: bool = True) -> None:
    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": "m1",
                    "ts": 1.0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "the invoices workspace"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _wake_entry(
    session_id: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    cwd: str = "/work/here",
    **extra: object,
) -> None:
    entry: dict[str, Any] = {
        "schema": 1,
        "session_id": session_id,
        "cwd": cwd,
        "updated_at": 1_700_000_000_000,
        "schedules": rows if rows is not None else [_wake_row("w1")],
    }
    entry.update(extra)
    path = config_dir() / "wakes" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")


def _wake_row(wake_id: str = "w1", *, due: int | None = None, **extra: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": wake_id,
        "message": f"message {wake_id}",
        "next_due_at": FUTURE if due is None else due,
        "every_ms": 3_600_000,
        "until_at": None,
        "limit": None,
        "fired_count": 0,
        "created_at": 1,
    }
    row.update(extra)
    return row


def _patience_row(wake_id: str = "patience-1") -> dict[str, Any]:
    """A hidden patience wait, the shape ``WakeSchedule`` carries to the index."""
    return {
        "id": wake_id,
        "message": "",
        "next_due_at": FUTURE,
        "every_ms": None,
        "until_at": None,
        "limit": None,
        "fired_count": 0,
        "created_at": 1,
        "kind": "patience",
        "hidden": True,
    }


def _ask_timeout_row(wake_id: str = "ask-timeout-a1") -> dict[str, Any]:
    """A hidden queued-ask deadline (``asks.queue._arm_deadline_wake`` arms the
    real one, id ``ask-timeout-<ask_id>``): ``_patience_row``'s fixture shape,
    the queue's own kind."""
    return {
        "id": wake_id,
        "message": "ask a1 deadline",
        "next_due_at": FUTURE,
        "every_ms": None,
        "until_at": None,
        "limit": None,
        "fired_count": 0,
        "created_at": 1,
        "kind": "ask_timeout",
        "hidden": True,
    }


def _monitor_entry(
    session_id: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    cwd: str = "/work/here",
    **extra: object,
) -> None:
    entry: dict[str, Any] = {
        "schema": 1,
        "session_id": session_id,
        "cwd": cwd,
        "updated_at": 1_700_000_000_000,
        "monitors": rows if rows is not None else [_monitor_row("m1")],
    }
    entry.update(extra)
    path = config_dir() / "monitors" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")


def _monitor_row(
    monitor_id: str = "m1", *, due: int | None = None, **extra: object
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": monitor_id,
        "name": f"watch {monitor_id}",
        "tool": "bash",
        "arguments": {"command": "ls"},
        "every_ms": 60_000,
        "until_at": None,
        "description": "",
        "next_due_at": FUTURE if due is None else due,
        "last_check_at": 0,
        "checks": 0,
        "deliveries": 0,
        "consecutive_failures": 0,
        "disabled": False,
        "disabled_reason": "",
        "created_at": int(time.time() * 1000),
    }
    row.update(extra)
    return row


def test_the_gate_holds_and_login_unlocks_the_surface() -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    refused = client.get("/api/schedules")
    assert refused.status_code == 401
    assert refused.json()["error"] == "authentication required"

    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    assert set(_body(client)) == {"wakes", "monitors"}


def test_an_absent_store_is_an_empty_list_and_not_an_error() -> None:
    """No ``wakes/`` or ``monitors/`` directory is the ordinary state of a
    machine that never armed anything, and it must not read as a failed read.

    The wake listing also answers whether anything would FIRE
    (``supervisor``); the monitor listing deliberately has none — monitors
    never engage a cold session, so a supervisor-shaped field would advertise
    a watcher that does not exist."""
    body = _body(_client())

    assert body["wakes"]["entries"] == []
    assert body["wakes"]["total"] == 0
    assert body["wakes"]["truncated"] is False
    assert body["wakes"]["read_error"] is False
    assert isinstance(body["wakes"]["generated_at"], int)
    assert set(body["wakes"]["supervisor"]) >= {"supported", "running", "detail"}

    assert body["monitors"]["entries"] == []
    assert body["monitors"]["read_error"] is False
    assert "supervisor" not in body["monitors"]


def test_the_index_answers_without_a_runtime_and_names_each_row_s_own_session() -> None:
    """Answered with nothing running (this daemon has no records and no
    runtimes — only what the fixtures wrote), each row carrying the
    conversation to open and its own schedules/monitors beneath it."""
    _session(SESSION_A)
    _wake_entry(SESSION_A, rows=[_wake_row("w1", due=FUTURE + 60_000), _patience_row()])
    # No session directory: a ghost, still listed (the index outlives it).
    _wake_entry(SESSION_B, cwd="/tmp/bbb")
    _monitor_entry(SESSION_A)

    body = _body(_client())

    wakes = body["wakes"]
    # Soonest first; B's wake is due an hour before A's.
    assert [entry["session_id"] for entry in wakes["entries"]] == [SESSION_B, SESSION_A]
    assert wakes["total"] == 2 and wakes["truncated"] is False

    ghost, live = wakes["entries"]
    assert ghost["ghost"] is True and ghost["dormant"] is False
    # A nameless row is one the user cannot identify: the floor is the id and
    # the cwd's basename, never empty.
    assert ghost["name"] == "bbb44455 (bbb)"
    assert ghost["next_due_at"] == FUTURE

    assert live["ghost"] is False and live["dormant"] is False
    assert live["name"] == "the invoices workspace"
    assert live["origin"] == ""
    assert set(live) >= {
        "session_id",
        "name",
        "cwd",
        "origin",
        "updated_at",
        "dormant",
        "ghost",
        "next_due_at",
        "schedules",
    }
    # The patience wait is a mechanism, not a reminder the user set: it never
    # appears, and the mixed entry still carries its one visible row.
    assert [row["id"] for row in live["schedules"]] == ["w1"]
    row = live["schedules"][0]
    assert row["next_due_at"] == FUTURE + 60_000
    assert row["stale"] is False and row["overdue_s"] == 0.0
    assert set(row) >= {
        "id",
        "message",
        "next_due_at",
        "every_ms",
        "until_at",
        "limit",
        "fired_count",
        "overdue_s",
        "stale",
        "last_fired_at",
        "last_attempt_at",
    }

    monitors = body["monitors"]
    assert [entry["session_id"] for entry in monitors["entries"]] == [SESSION_A]
    assert monitors["entries"][0]["next_due_at"] == FUTURE
    watch = monitors["entries"][0]["monitors"][0]
    assert watch["id"] == "m1" and watch["state"] == "armed"
    assert watch["tool"] == "bash" and watch["arguments"] == {"command": "ls"}
    assert watch["health"] is None and watch["unavailable_since"] == 0
    assert watch["due_in_s"] > 3000
    assert set(watch) >= {
        "id",
        "name",
        "tool",
        "arguments",
        "description",
        "every_ms",
        "until_at",
        "notify",
        "sort_lines",
        "ignore",
        "cwd",
        "created_at",
        "next_due_at",
        "last_check_at",
        "checks",
        "deliveries",
        "consecutive_failures",
        "disabled",
        "disabled_reason",
        "due_in_s",
        "last_check_age_s",
        "state",
        "unavailable_since",
        "health",
    }


def test_a_parked_session_is_still_listed_and_marked_dormant() -> None:
    """Dormancy parks a wake; it never hides it. The desktop lists stopped
    (``stopped_at``) and Aida-held (``held_at``) entries by DEFAULT —
    ``include_dormant`` is true unless asked otherwise — and this route
    mirrors that default with no knob of its own, so a store holding only
    parked entries must not answer empty.

    B has no session directory: a parked entry is not a ghost either —
    nothing is gone, the session is deliberately not firing."""
    _wake_entry(SESSION_A, stopped_at=1_700_000_000_001)
    _wake_entry(SESSION_B, held_at=1_700_000_000_002, cwd="/tmp/bbb")

    body = _body(_client())

    wakes = body["wakes"]
    assert [entry["session_id"] for entry in wakes["entries"]] == [SESSION_A, SESSION_B]
    assert wakes["total"] == 2
    stopped, held = wakes["entries"]
    assert stopped["dormant"] is True and stopped["ghost"] is False
    assert [row["id"] for row in stopped["schedules"]] == ["w1"]
    assert held["dormant"] is True and held["ghost"] is False


def test_the_monitor_state_word_follows_the_cli_precedence() -> None:
    """Only ``armed`` was exercised before this cell. The wire's ``state`` word
    is the CLI's own precedence (``cli._monitor_state_word``): dormancy beats
    disabled (nothing is SUPPOSED to run, so a failure word would point at the
    wrong remedy); disabled beats the clock (a watch that does not tick must
    not read as merely late); expiry is the next fall. A late watch stays
    armed — the due time moves only on change events, so lateness between
    events is ordinary, not a state."""
    _monitor_entry(
        SESSION_A,
        rows=[
            _monitor_row("m1"),
            _monitor_row("m2", disabled=True, disabled_reason="5 failed"),
            _monitor_row("m3", until_at=PAST),
            _monitor_row("m4", due=PAST),
        ],
    )
    _monitor_entry(
        SESSION_B, stopped_at=1_700_000_000_001, rows=[_monitor_row("m5", disabled=True)]
    )

    body = _body(_client())

    entries = {entry["session_id"]: entry for entry in body["monitors"]["entries"]}
    rows = {row["id"]: row for entry in entries.values() for row in entry["monitors"]}
    assert {row_id: row["state"] for row_id, row in rows.items()} == {
        "m1": "armed",
        "m2": "disabled",
        "m3": "expired",
        "m4": "armed",
        "m5": "dormant",
    }
    # The word is not decoration on the flags: m5 carries its own disabled
    # flag, and the park must still be what a reader sees first.
    assert rows["m5"]["disabled"] is True
    assert rows["m2"]["disabled_reason"] == "5 failed"
    assert rows["m4"]["due_in_s"] < 0 and rows["m4"]["next_due_at"] == PAST
    assert rows["m1"]["due_in_s"] > 0
    assert entries[SESSION_B]["dormant"] is True and entries[SESSION_B]["ghost"] is False


def test_the_shared_wire_models_field_sets_are_pinned() -> None:
    """The mirror guarantee in ``local_operator/mobile/schedules.py``, made
    checkable. The rows on this wire are built as the desktop's own models, and
    this cell is what makes a declared field ADDED, RENAMED or REMOVED on those
    models fail here — construction alone cannot, because ``extra="allow"``
    absorbs all three quietly (a defaulted addition ships its default; a rename
    or a removal lands as an extra).

    A red here means the shared wire moved: mirror the change in
    ``schedules.py`` (and this file's cells) before a phone can read a default
    the desktop never serves."""
    from local_operator.server.models.desktop_monitors import (
        MonitorEntry,
        MonitorListing,
        MonitorRow,
    )
    from local_operator.server.models.desktop_wakes import (
        SupervisorInfo,
        WakeEntry,
        WakeListing,
        WakeScheduleRow,
    )

    assert set(WakeScheduleRow.model_fields) == {
        "id",
        "message",
        "next_due_at",
        "every_ms",
        "until_at",
        "limit",
        "fired_count",
        "overdue_s",
        "stale",
        "last_fired_at",
        "last_attempt_at",
    }
    assert set(WakeEntry.model_fields) == {
        "session_id",
        "name",
        "cwd",
        "origin",
        "updated_at",
        "dormant",
        "ghost",
        "next_due_at",
        "schedules",
    }
    assert set(SupervisorInfo.model_fields) == {"supported", "running", "detail", "verifiable"}
    assert set(WakeListing.model_fields) == {
        "entries",
        "generated_at",
        "total",
        "truncated",
        "supervisor",
        "read_error",
    }
    assert set(MonitorRow.model_fields) == {
        "id",
        "name",
        "tool",
        "arguments",
        "description",
        "every_ms",
        "until_at",
        "notify",
        "sort_lines",
        "ignore",
        "cwd",
        "created_at",
        "next_due_at",
        "last_check_at",
        "checks",
        "deliveries",
        "consecutive_failures",
        "disabled",
        "disabled_reason",
        "due_in_s",
        "last_check_age_s",
        "state",
        "unavailable_since",
        "health",
    }
    assert set(MonitorEntry.model_fields) == {
        "session_id",
        "name",
        "cwd",
        "origin",
        "updated_at",
        "dormant",
        "ghost",
        "next_due_at",
        "monitors",
    }
    assert set(MonitorListing.model_fields) == {
        "entries",
        "generated_at",
        "total",
        "truncated",
        "read_error",
    }


def test_a_patience_only_session_is_not_a_wake_carrying_session() -> None:
    """Internal timers are invisible on every human surface, so an entry whose
    only rows are hidden must not become an empty wake row."""
    _wake_entry(SESSION_A, rows=[_patience_row()])

    body = _body(_client())

    assert body["wakes"]["entries"] == []
    assert body["wakes"]["total"] == 0
    assert body["wakes"]["read_error"] is False


def test_an_ask_timeout_only_session_is_not_a_wake_carrying_session() -> None:
    """The queued ask's deadline rides the same engine (kind ``ask_timeout``,
    ``asks.queue._arm_deadline_wake``) and is invisible on human surfaces for
    the same reason a patience wait is — it is a mechanism, not a reminder.
    The filter must not over-cut either: a session beside a visible row keeps
    the visible one and loses only the timer."""
    _wake_entry(SESSION_A, rows=[_ask_timeout_row()])
    _wake_entry(SESSION_B, rows=[_wake_row("w2"), _ask_timeout_row()])

    body = _body(_client())

    wakes = body["wakes"]
    assert [entry["session_id"] for entry in wakes["entries"]] == [SESSION_B]
    assert [row["id"] for row in wakes["entries"][0]["schedules"]] == ["w2"]
    assert wakes["total"] == 1


def test_an_unreadable_wakes_store_reports_the_read_rather_than_an_empty_list() -> None:
    """The one case where "nothing is armed" would be a claim this process has
    not earned: the directory is there and cannot be listed. The other family
    still answers — one bad read must not speak for both stores."""
    _wake_entry(SESSION_A)
    _monitor_entry(SESSION_B)
    wakes = config_dir() / "wakes"
    wakes.chmod(0o000)
    try:
        body = _body(_client())
    finally:
        wakes.chmod(0o700)

    assert body["wakes"]["entries"] == []
    assert body["wakes"]["read_error"] is True
    assert body["monitors"]["read_error"] is False
    assert [entry["session_id"] for entry in body["monitors"]["entries"]] == [SESSION_B]


def test_an_unreadable_monitors_store_reports_the_read_rather_than_an_empty_list() -> None:
    _wake_entry(SESSION_A)
    _monitor_entry(SESSION_B)
    monitors = config_dir() / "monitors"
    monitors.chmod(0o000)
    try:
        body = _body(_client())
    finally:
        monitors.chmod(0o700)

    assert body["monitors"]["entries"] == []
    assert body["monitors"]["read_error"] is True
    assert body["wakes"]["read_error"] is False
    assert [entry["session_id"] for entry in body["wakes"]["entries"]] == [SESSION_A]
