"""Asks and inbox rows whose conversation is gone (``asks/orphans.py``) and the
answers route's doorless arms (``routes/desktop_sessions._with_doorless_ask_arms``).

THE REPORT THESE REPRODUCE (operator, 2026-11-24): a "Waiting on you" ask on the
Notifications page, titled from "Untitled conversation", that answered
"Requested session, profile, team or subscription not found" to Send, Decline
and Dismiss alike. Measured in an isolated store before the fix (PR evidence):

* a SUBAGENT's queued ask is listed by the inbox but the desktop pool's door
  refuses hidden origins, so all three verbs were 404 and the title was empty;
* an explicit delete of a conversation holding an open ask was REFUSED, with
  the misleading fallback sentence, because every queued ask arms a hidden
  ``ask_timeout`` wake row and "has an armed wake" fired first;
* a conversation whose directory left by any other path kept its inbox rows
  forever, and a client still holding its ask row got 404 on every verb.

Every cell runs on ``tmp_path`` with synthetic 12-hex ids; nothing reads the
operator's store.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.asks import orphans
from local_operator.asks import store as ask_store
from local_operator.notifications import items as inbox
from local_operator.notifications.store import (
    NETWORK_CREDENTIALS_SESSION_ID,
    NotificationStore,
)

SUB = "a1a1a1a1a1a1"
USER = "b2b2b2b2b2b2"
GONE = "c3c3c3c3c3c3"
ASK = "a-0001"


# ---------------------------------------------------------------------------
# Fixtures: the stores' own writers, never hand-built shapes
# ---------------------------------------------------------------------------


def _session(root: Path, session_id: str, *, origin: dict[str, Any] | None = None) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps({"type": "message", "payload": {"role": "user", "content": "hi"}}) + "\n",
        encoding="utf-8",
    )
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps(origin), encoding="utf-8")
    return directory


def _queue_ask(root: Path, session_id: str, *, ask_id: str = ASK, wake: bool = True) -> None:
    """A queued ask exactly as a runtime leaves it: log event, index entry, deadline row."""
    directory = root / "sessions" / session_id
    now = ask_store.now_ms()
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ask_id,
            "at": now,
            "created_at": now,
            "expires_at": now + 3_600_000,
            "timeout_s": 3600,
            "urgent": False,
            "questions": [
                {
                    "id": "q1",
                    "question": "Ship it?",
                    "options": [{"label": "yes", "description": ""}],
                    "multi": False,
                    "persist": False,
                }
            ],
        },
    )
    records = ask_store.fold(ask_store.read_events(directory), now)
    ask_store.write_entry(root, session_id, cwd=str(root), asks=records)
    if wake:
        from local_operator.wakes.store import write_entry

        write_entry(
            root,
            session_id,
            cwd=str(root),
            schedules=[
                {
                    "id": ask_store.timeout_row_id(ask_id),
                    "message": f"ask {ask_id} deadline",
                    "next_due_at": now + 3_600_000,
                    "created_at": now,
                    "kind": "ask_timeout",
                    "hidden": True,
                    "origin": "tool",
                }
            ],
        )


def _gone_index_entry(root: Path, session_id: str) -> None:
    """An index entry whose session directory is gone (what a client still shows)."""
    _session(root, session_id)
    _queue_ask(root, session_id, wake=False)
    import shutil

    shutil.rmtree(root / "sessions" / session_id)


def _note_rows(root: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(root / "notifications.db") as conn:
        return sorted(conn.execute("SELECT session_id, key FROM items").fetchall())


# ---------------------------------------------------------------------------
# Inbox facts: the title and the conversation_deleted flag
# ---------------------------------------------------------------------------


def test_a_subagent_row_is_titled_by_its_origin_label(tmp_path: Path) -> None:
    """Defect A's title half: "Untitled conversation" on a subagent's ask."""
    _session(tmp_path, SUB, origin={"origin": "subagent", "label": "Release window"})
    facts = inbox.SessionFacts(tmp_path)
    assert facts.get(SUB)["title"] == "Release window"
    assert facts.deleted(SUB) is False


def test_conversation_deleted_only_for_a_provably_gone_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sessions").mkdir()
    facts = inbox.SessionFacts(tmp_path)
    assert facts.deleted(GONE) is True
    # A pseudo-session and a machine-level row never had a directory.
    assert inbox.SessionFacts(tmp_path).deleted(NETWORK_CREDENTIALS_SESSION_ID) is False
    assert inbox.SessionFacts(tmp_path).deleted("") is False
    # Mid-hand-off is not deleted.
    (tmp_path / "sessions" / "handed-away" / GONE).mkdir(parents=True)
    assert inbox.SessionFacts(tmp_path).deleted(GONE) is False
    # Moved to a peer (in the cached listing) is not deleted.
    (tmp_path / "sessions" / "handed-away" / GONE).rmdir()
    monkeypatch.setattr(
        "local_operator.session.peer_rows.peer_session_row",
        lambda session_id, root=None: SimpleNamespace(id=session_id),
    )
    assert inbox.SessionFacts(tmp_path).deleted(GONE) is False


def test_notify_items_carry_the_flag_but_peer_rows_never_do(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()
    notes = NotificationStore(tmp_path)
    notes.notify(GONE, title="local update")
    # The peer row names a DIFFERENT gone id: a session with any mirrored row is
    # moved, not deleted (Q2, its own cell below), so sharing GONE here would
    # stop the local row from being flagged at all.
    other = "f6f6f6f6f6f6"
    notes.notify(other, title="peer update", key="remote-done:d_x:" + other)
    facts = inbox.SessionFacts(tmp_path)
    # A stub for the related-projects block: the real ``ProjectFacts`` starts the
    # process-wide registry refresher thread, which is not what this cell tests.
    projects: Any = SimpleNamespace(get=lambda session_id: [])
    from local_operator.notifications.store import _row_dict

    with sqlite3.connect(tmp_path / "notifications.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = [_row_dict(row) for row in conn.execute("SELECT * FROM items")]
    shaped = {row["key"]: inbox.notify_item(row, facts, projects) for row in rows}
    assert shaped[""]["conversation_deleted"] is True
    assert "conversation_deleted" not in shaped["remote-done:d_x:" + other]


def test_a_moved_conversation_is_neither_flagged_nor_swept(tmp_path: Path) -> None:
    """QA round 1, Q2: the LOCAL pre-move row of a conversation whose new holder
    mirrors ``remote-*`` rows back (live: ``ce6fcb93c28a``) was flagged
    ``conversation_deleted`` with a cold peer cache while the sweep treated the
    same session as moved. One predicate (``session_gone``) now answers both."""
    (tmp_path / "sessions").mkdir()
    moved = "d4d4d4d4d4d4"
    notes = NotificationStore(tmp_path)
    notes.notify(moved, title="from before the move")
    notes.notify(moved, title="peer", key="remote-done:d_x:" + moved)

    assert orphans.session_gone(tmp_path, moved) is False
    assert inbox.SessionFacts(tmp_path).deleted(moved) is False
    projects: Any = SimpleNamespace(get=lambda session_id: [])
    from local_operator.notifications.store import _row_dict

    with sqlite3.connect(tmp_path / "notifications.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = [_row_dict(row) for row in conn.execute("SELECT * FROM items")]
    facts = inbox.SessionFacts(tmp_path)
    assert not any("conversation_deleted" in inbox.notify_item(r, facts, projects) for r in rows)
    report = orphans.sweep(tmp_path)
    assert report.orphan_note_sessions == []
    assert len(_note_rows(tmp_path)) == 2
    # Control: the same local row with no mirror IS deleted, by both.
    notes.notify(GONE, title="orphan")
    assert inbox.SessionFacts(tmp_path).deleted(GONE) is True
    assert [r["session_id"] for r in orphans.sweep(tmp_path).orphan_note_sessions] == [GONE]


def test_a_deleted_conversations_ask_stays_listed_flagged_until_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-U1 (core half): the index reader unlinked the entry the moment the
    directory was missing, so the ask left the list and its detail 404'd before
    any surface could say "Conversation deleted". Now it is listed and detailed
    with the flag until the answers route settles it or the sweep removes it."""
    from local_operator.notifications.store import ask_ref

    monkeypatch.setattr(inbox, "ProjectFacts", lambda root: SimpleNamespace(get=lambda s: []))
    (tmp_path / "sessions").mkdir()
    _gone_index_entry(tmp_path, GONE)
    notes = NotificationStore(tmp_path)

    for _ in range(2):  # a second read must not have unlinked it either
        rows = inbox.open_asks(tmp_path)
        assert [(r["session_id"], r["ask_id"]) for r in rows] == [(GONE, ASK)]
    assert ask_store.read_index(tmp_path) == {}, "other readers still skip it"
    assert ask_store.entry_path(tmp_path, GONE).exists()
    projects: Any = SimpleNamespace(get=lambda s: [])
    listed = inbox.waiting_items(tmp_path, notes, inbox.SessionFacts(tmp_path), projects)
    assert listed[0]["conversation_deleted"] is True
    shown = inbox.detail(tmp_path, ask_ref(GONE, ASK))
    assert shown is not None and shown["item"]["conversation_deleted"] is True

    import asyncio

    asyncio.run(orphans.settle_deleted_ask(tmp_path, GONE, ASK))
    assert inbox.open_asks(tmp_path) == []
    assert inbox.detail(tmp_path, ask_ref(GONE, ASK)) is None


def test_a_missing_directory_that_is_not_gone_stays_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-off leaves no directory but is not a deletion: its ask is neither
    listed nor labelled, and its entry is not unlinked."""
    (tmp_path / "sessions").mkdir()
    _gone_index_entry(tmp_path, GONE)
    (tmp_path / "sessions" / "handed-away" / GONE).mkdir(parents=True)
    assert inbox.open_asks(tmp_path) == []
    assert ask_store.entry_path(tmp_path, GONE).exists()


def test_the_serve_sweeper_surfaces_its_settle_line(tmp_path: Path) -> None:
    """QA round 1, Q3: ``lop serve`` floors the root logger at WARNING, so the
    sweep's INFO "orphans: settled ..." never reached the serve log."""
    import logging

    root = logging.getLogger()
    saved_root, saved_own = root.level, orphans.logger.level
    try:
        root.setLevel(logging.WARNING)
        orphans.logger.setLevel(logging.NOTSET)
        assert not orphans.logger.isEnabledFor(logging.INFO)
        orphans.surface_settle_log()
        assert orphans.logger.isEnabledFor(logging.INFO)
        # Never lowers a DEBUG-configured process.
        orphans.logger.setLevel(logging.DEBUG)
        orphans.surface_settle_log()
        assert orphans.logger.level == logging.DEBUG
        orphans.logger.setLevel(logging.NOTSET)
        orphans.surface_settle_log()
        # The serve handler sits on the root (floored at WARNING) and has no
        # level of its own; the raised logger's record still reaches it.
        seen: list[str] = []
        sink = logging.Handler()
        sink.emit = lambda record: seen.append(record.getMessage())  # type: ignore[method-assign]
        root.addHandler(sink)
        try:
            (tmp_path / "sessions").mkdir()
            NotificationStore(tmp_path).notify(GONE, title="orphan")
            orphans.sweep(tmp_path)
        finally:
            root.removeHandler(sink)
        assert any(line.startswith("orphans: settled 1") for line in seen), seen
    finally:
        root.setLevel(saved_root)
        orphans.logger.setLevel(saved_own)


# ---------------------------------------------------------------------------
# The explicit delete cascade
# ---------------------------------------------------------------------------


def _marked(tmp_path: Path) -> None:
    from local_operator.session.cleanup import mark_store

    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    mark_store(tmp_path / "sessions")


def test_an_explicit_delete_withdraws_the_open_ask_and_removes_everything(
    tmp_path: Path,
) -> None:
    """Defect B: refused before ("could not be checked"), now settled and removed."""
    from local_operator.session.cleanup import delete_session

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER)
    NotificationStore(tmp_path).notify(USER, title="an update")
    withdrawn: list[list[dict[str, Any]]] = []
    real_remove = __import__("local_operator.session.cleanup", fromlist=["x"]).remove_session_dir

    def spy(target: Path, **kwargs: Any) -> bool:
        # The withdrawal must be in the log BEFORE the directory goes.
        withdrawn.append(ask_store.read_events(target))
        return real_remove(target, **kwargs)

    import local_operator.session.cleanup as cleanup

    original = cleanup.remove_session_dir
    cleanup.remove_session_dir = spy
    try:
        outcome = delete_session(tmp_path, USER, actor="test")
    finally:
        cleanup.remove_session_dir = original

    assert outcome.deleted is True, outcome.refusal
    assert not directory.exists()
    last = withdrawn[0][-1]
    assert last["kind"] == ask_store.EVENT_WITHDRAWN
    assert last["reason"] == orphans.WITHDRAW_REASON_DELETED
    assert ask_store.read_entry(tmp_path, USER) is None
    from local_operator.wakes.store import entry_path

    assert not entry_path(tmp_path, USER).exists()
    assert _note_rows(tmp_path) == []
    assert inbox.counts(tmp_path)["waiting_count"] == 0


def test_a_real_reminder_still_refuses_the_delete(tmp_path: Path) -> None:
    """Only the asks' own deadline rows are folded into the ask cascade."""
    from local_operator.session.cleanup import delete_session
    from local_operator.wakes.store import write_entry

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER, wake=False)
    write_entry(
        tmp_path,
        USER,
        cwd=str(tmp_path),
        schedules=[
            {
                "id": ask_store.timeout_row_id(ASK),
                "message": "d",
                "next_due_at": 4_102_444_800_000,
                "kind": "ask_timeout",
                "hidden": True,
            },
            {"id": "w1", "message": "check the parser fix", "next_due_at": 4_102_444_800_000},
        ],
    )
    outcome = delete_session(tmp_path, USER, actor="test")
    assert outcome.deleted is False
    assert "wake" in outcome.refusal
    assert directory.is_dir()
    assert ask_store.fold(ask_store.read_events(directory), ask_store.now_ms())[0]["status"] == (
        ask_store.STATUS_OPEN
    ), "a refused delete must not have withdrawn the ask"


def test_a_rehearsal_reports_the_delete_and_writes_nothing(tmp_path: Path) -> None:
    from local_operator.session.cleanup import delete_session

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER)
    before = (directory / ask_store.ASKS_LOG_NAME).read_bytes()
    outcome = delete_session(tmp_path, USER, actor="test", dry_run=True)
    assert outcome.refusal == ""
    assert directory.is_dir()
    assert (directory / ask_store.ASKS_LOG_NAME).read_bytes() == before


def test_the_automatic_policy_still_keeps_an_open_question(tmp_path: Path) -> None:
    """The guard is relaxed for an EXPLICIT delete only."""
    from local_operator.session.cleanup import _guard

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER, wake=False)
    assert _guard(directory, tmp_path, 0.0) == "has an open ask"


def _spool_mail(directory: Path) -> None:
    from local_operator.session.runtime.inbox import INBOX_NAME

    (directory / INBOX_NAME).write_text('{"from": "peer", "text": "unread"}\n')


def _arm_monitor(root: Path, session_id: str) -> None:
    from local_operator.monitors.store import entry_path

    path = entry_path(root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"monitors": [{"id": "m1", "command": "true"}]}))


@pytest.mark.parametrize("obstacle", ["mail", "monitor"])
def test_a_refused_delete_leaves_the_open_ask_and_its_deadline_untouched(
    tmp_path: Path, obstacle: str
) -> None:
    """Review round 1, R1/Q1: the asks were withdrawn and their deadline wake
    dropped BEFORE the guards behind the armed-wake probe ran; a delete then
    refused on unread mail (or an armed monitor) kept the conversation with its
    live question destroyed, and the rehearsal promised a delete that refused."""
    from local_operator.session.cleanup import delete_session
    from local_operator.wakes.store import entry_path as wake_path

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER)
    if obstacle == "mail":
        _spool_mail(directory)
        expected = "unread messages"
    else:
        _arm_monitor(tmp_path, USER)
        expected = "monitor"
    wake_before = wake_path(tmp_path, USER).read_bytes()
    log_before = (directory / ask_store.ASKS_LOG_NAME).read_bytes()

    rehearsal = delete_session(tmp_path, USER, actor="test", dry_run=True)
    real = delete_session(tmp_path, USER, actor="test")

    assert real.deleted is False
    assert expected in real.refusal.lower(), real.refusal
    assert rehearsal.refusal == real.refusal, "the rehearsal must give the same verdict"
    assert directory.is_dir()
    assert (directory / ask_store.ASKS_LOG_NAME).read_bytes() == log_before
    assert wake_path(tmp_path, USER).read_bytes() == wake_before
    entry = ask_store.read_entry(tmp_path, USER) or {}
    assert [row["status"] for row in entry["asks"]] == [ask_store.STATUS_OPEN]


def test_a_claim_taken_just_before_the_withdrawal_refuses_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, R3: the claim/lease is re-read right before the appends,
    so a runtime that engaged after the first guard read gets no withdrawals
    written behind it."""
    import local_operator.session.cleanup as cleanup

    _marked(tmp_path)
    directory = _session(tmp_path, USER)
    _queue_ask(tmp_path, USER)
    log_before = (directory / ask_store.ASKS_LOG_NAME).read_bytes()
    reads = {"n": 0}
    real_claimed = cleanup._claimed

    def claimed_late(target: Path, now: float) -> bool:
        # Every guard read before the withdrawal sees it free; the re-read
        # immediately before the appends sees the runtime that just engaged.
        reads["n"] += 1
        return reads["n"] >= 3 or real_claimed(target, now)

    monkeypatch.setattr(cleanup, "_claimed", claimed_late)
    outcome = cleanup.delete_session(tmp_path, USER, actor="test")
    assert reads["n"] == 3
    assert outcome.deleted is False
    assert (directory / ask_store.ASKS_LOG_NAME).read_bytes() == log_before


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_the_sweep_settles_orphans_and_spares_everything_reachable(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()
    _session(tmp_path, USER)
    notes = NotificationStore(tmp_path)
    notes.notify(USER, title="alive")
    notes.notify(GONE, title="orphan")
    notes.notify(NETWORK_CREDENTIALS_SESSION_ID, title="pseudo-session")
    moved = "d4d4d4d4d4d4"
    notes.notify(moved, title="from before the move")
    notes.notify(moved, title="peer", key="remote-done:d_x:" + moved)
    _gone_index_entry(tmp_path, "e5e5e5e5e5e5")

    preview = orphans.sweep(tmp_path, dry_run=True)
    assert [row["session_id"] for row in preview.orphan_note_sessions] == [GONE]
    assert [row["session_id"] for row in preview.orphan_ask_entries] == ["e5e5e5e5e5e5"]
    assert len(_note_rows(tmp_path)) == 5, "a dry run writes nothing"
    assert ask_store.entry_path(tmp_path, "e5e5e5e5e5e5").exists()

    report = orphans.sweep(tmp_path)
    assert report.settled == 2
    remaining = {sid for sid, _key in _note_rows(tmp_path)}
    assert remaining == {USER, NETWORK_CREDENTIALS_SESSION_ID, moved}
    assert not ask_store.entry_path(tmp_path, "e5e5e5e5e5e5").exists()


def test_the_sweep_is_bounded_and_defers_the_rest(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()
    notes = NotificationStore(tmp_path)
    for index in range(3):
        notes.notify(f"{index:012x}", title="orphan")
    first = orphans.sweep(tmp_path, max_settles=2)
    assert (first.settled, first.deferred) == (2, 1)
    second = orphans.sweep(tmp_path, max_settles=2)
    assert (second.settled, second.deferred) == (1, 0)
    assert _note_rows(tmp_path) == []


def test_the_incremental_pass_rereads_nothing_unchanged(tmp_path: Path) -> None:
    _session(tmp_path, USER)
    _queue_ask(tmp_path, USER, wake=False)
    orphans._MEMO.clear()
    assert orphans.sweep(tmp_path).ask_entries_read == 1
    assert orphans.sweep(tmp_path).ask_entries_read == 0
    # A conversation deleted out of band is still caught: existence is stat'ed
    # every pass, never memoised.
    import shutil

    shutil.rmtree(tmp_path / "sessions" / USER)
    report = orphans.sweep(tmp_path)
    assert [row["session_id"] for row in report.orphan_ask_entries] == [USER]


# ---------------------------------------------------------------------------
# The answers route's doorless arms
# ---------------------------------------------------------------------------


class _UnknownPool:
    """The pool door's verdict for an id it will not serve: ``KeyError``."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def session(self, session_id: str, **_: Any) -> Any:
        import contextlib

        pool = self

        @contextlib.asynccontextmanager
        async def door():
            pool.opened.append(session_id)
            raise KeyError("Unknown session")
            yield  # pragma: no cover

        return door()

    def assert_admitting(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def ask_respond(self, ask_id: str, answers: Any, *, by: str = "", **_: Any) -> str:
        self.calls.append(("respond", ask_id))
        return "answered"

    async def ask_decline(self, ask_id: str, *, by: str = "") -> str:
        self.calls.append(("decline", ask_id))
        return "declined"

    async def ask_revise(self, ask_id: str, answers: Any, *, by: str = "") -> str:
        self.calls.append(("revise", ask_id))
        return "revised"

    async def ask_dismiss(self, ask_id: str, *, by: str = "", reason: Any = None) -> str:
        self.calls.append(("dismiss", ask_id))
        return "dismissed"

    def close(self) -> None:
        self.closed = True


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions

    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    (tmp_path / "sessions").mkdir(exist_ok=True)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = _UnknownPool()
    app.include_router(desktop_sessions.router)
    client = _FakeClient()
    engaged: list[str] = []

    async def fake_engage(config_dir: Path, session_id: str, work: Any, **_: Any):
        engaged.append(session_id)
        return client, "admitted"

    monkeypatch.setattr("local_operator.mobile.attach_client.engage_session_client", fake_engage)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer token"},
    ) as http:
        yield SimpleNamespace(http=http, root=tmp_path, client=client, engaged=engaged)


def _url(session_id: str) -> str:
    return f"/v1/desktop/sessions/{session_id}/answers"


@pytest.mark.asyncio
async def test_a_subagent_ask_is_dismissed_cold(api) -> None:
    directory = _session(api.root, SUB, origin={"origin": "subagent", "label": "L"})
    _queue_ask(api.root, SUB, wake=False)
    result = await api.http.post(_url(SUB), json={"ask_id": ASK, "dismiss": True})
    assert result.status_code == 200, result.text
    assert result.json()["result"]["detail"] == "dismissed"
    assert ask_store.read_events(directory)[-1]["kind"] == ask_store.EVENT_DISMISSED
    assert api.engaged == [], "a cold dismissal boots nothing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,call,consent",
    [
        ({"decline": True}, "decline", None),
        ({"answers": {"q1": ["yes"]}}, "respond", False),
    ],
)
async def test_a_subagent_ask_is_answered_through_the_engage_seam(
    api, body: dict[str, Any], call: str, consent: Any
) -> None:
    _session(api.root, SUB, origin={"origin": "subagent", "label": "L"})
    _queue_ask(api.root, SUB, wake=False)
    result = await api.http.post(_url(SUB), json={"ask_id": ASK, **body})
    assert result.status_code == 200, result.text
    assert result.json()["result"]["consent_bearing"] is consent
    assert api.engaged == [SUB]
    assert api.client.calls == [(call, ASK)]
    assert api.client.closed is True


@pytest.mark.asyncio
async def test_a_subagent_without_that_indexed_ask_keeps_the_404(api) -> None:
    """The widening is per ASK: the door's hidden-origin refusal otherwise stands."""
    _session(api.root, SUB, origin={"origin": "subagent", "label": "L"})
    result = await api.http.post(_url(SUB), json={"ask_id": "a-nope", "dismiss": True})
    assert result.status_code == 404, result.text
    assert api.engaged == []


@pytest.mark.asyncio
async def test_a_deleted_owners_ask_dismisses_and_declines_locally(api) -> None:
    _gone_index_entry(api.root, GONE)
    for word, body in (("dismissed", {"dismiss": True}), ("declined", {"decline": True})):
        result = await api.http.post(_url(GONE), json={"ask_id": ASK, **body})
        assert result.status_code == 200, result.text
        assert result.json()["result"] == {"detail": word, "consent_bearing": None}
    assert ask_store.read_entry(api.root, GONE) is None
    assert api.engaged == []


@pytest.mark.asyncio
async def test_a_plain_answer_to_a_deleted_owner_is_refused_by_name(api) -> None:
    _gone_index_entry(api.root, GONE)
    result = await api.http.post(_url(GONE), json={"ask_id": ASK, "answers": {"q1": ["yes"]}})
    assert result.status_code == 409, result.text
    detail = result.json()["detail"]
    assert detail["code"] == "conversation_deleted"
    assert "deleted" in detail["text"]


@pytest.mark.asyncio
async def test_a_gate_answer_for_an_unknown_session_is_still_the_404(api) -> None:
    result = await api.http.post(
        _url(GONE),
        json={"epoch": "e", "request_id": "01234567-89ab-cdef-0123-456789abcdef", "approved": True},
    )
    assert result.status_code == 404, result.text
    assert result.json()["detail"]["code"] == "wire.errors.desktop.requested_not_found"
