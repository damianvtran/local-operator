"""End-to-end: dismissing a lapsed ask on a STOPPED session boots nothing.

Why this file exists (design ``docs/design/ask-nonblocking.md`` §2.2/§14, fork A).
The relay's cold arm for ``ask_dismiss`` used to ENGAGE the conversation — boot a
runtime, deliver the deadline notice the dismissal had not yet cancelled, and run
a paid turn — while every surface promised "dismiss — send no reply". Measured on
``origin/main`` by the architect's repro: a cold ``ask_dismiss`` on a lapsed ask
whose notice was never delivered produced ``message/ask_timeout`` then
``message/assistant`` and a live runtime child.

THIS file is the half a unit test cannot reach: a REAL relay daemon over a REAL
HTTP request, a REAL ``asks.jsonl`` and transcript on disk, and — for the reopen
half — a REAL runtime child on the mock provider. Two properties are asserted, and
each one is the opposite of a still:

* **The dismissal boots nothing.** No runtime is started, the transcript is
  byte-identical, and the dismissal IS durable (the log carries the row).
* **The notice is not erased; it waits.** A later reopen (a real engage) delivers
  the deadline row in its dismissed wording, so the agent is told it has no answer
  rather than never being told at all.

Isolation: the autouse ``headless_tui_env`` fixture points
``LOCAL_OPERATOR_CONFIG_DIR`` at a per-test directory and the root conftest
repoints ``HOME``, and any runtime child inherits both. Every process this file
starts is reaped by exact pid in the ``finally`` (``_reap``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.asks import store as ask_store
from local_operator.harness.types import Message
from local_operator.mobile.attach_client import (
    engage_session_client,
    find_runtime_record,
)
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.launch import WarmErrand
from local_operator.session.transcript import Transcript
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

SESSION = "noboot000001"
ASK_ID = "a-noboot000001"
#: The live cell's ask: it must not be the one the cold cell seeded, because its
#: whole point is an ask that came into being AFTER the runtime booted.
LIVE_ASK_ID = "a-nobootlive1"
QUESTION = "ship the no-boot dismissal?"

#: Generous: it covers a real process spawn and a real boot reconcile on a fleet
#: running dozens of suites at once (AGENTS.md). The watchdog turns a hang into a
#: named ``TimeoutError`` rather than a silent stall.
BOUND_S = 120.0

#: How long a dismissal gets to (wrongly) start a runtime before the frame is
#: read. A real engage publishes its discovery record within milliseconds and the
#: transcript row follows through a boot reconcile; the "boots nothing" claim is
#: worth this much patience rather than a same-tick assertion.
NO_BOOT_SETTLE_S = 5.0


def _seed(headless_tui_env: Path, *, ask_id: str = ASK_ID, lapsed: bool = True) -> Path:
    """A durable user session with one LAPSED queued ask and NO runtime.

    The transcript is seeded (not just the directory) because the relay's own gate
    is ``_durable_user_session_dir``: a conversation with no transcript is the
    "gone" refusal, which is a different cell. The ask's deadline is already past
    — that is the only state a dismissal is offered on — and inside the 7 d window,
    so its notice is still owed.
    """
    (headless_tui_env / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = headless_tui_env / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    asyncio.run(Transcript(directory).append_message(Message.user("hello", id="seed-user")))
    now = int(time.time() * 1000)
    row = {
        "ask_id": ask_id,
        "created_at": now - 3_600_000,
        "expires_at": now - 60_000 if lapsed else now + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": "open",
        "delivered": False,
        "questions": [
            {
                "id": "q1",
                "question": QUESTION,
                "options": [{"label": "yes", "description": ""}],
                "multi": False,
                "secret": False,
                "persist": False,
            }
        ],
    }
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ask_id,
            "at": now - 3_600_000,
            "created_at": now - 3_600_000,
            "expires_at": now - 60_000 if lapsed else now + 3_600_000,
            "timeout_s": 3600,
            "urgent": False,
            "tool_call_id": "",
            "questions": row["questions"],
        },
    )
    # The DERIVED index, as the last runtime to own this conversation left it: the
    # phone's row comes from here, and the cold dismissal must rewrite it.
    ask_store.write_entry(headless_tui_env, SESSION, cwd="/tmp/noboot", asks=[row])
    return directory


def _client() -> TestClient:
    """A logged-in client over the REAL relay app (its own isolated store)."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _transcript_rows(directory: Path) -> list[dict[str, object]]:
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _row_ids(directory: Path) -> list[str]:
    return [str(row.get("id")) for row in _transcript_rows(directory)]


def _wait_until(predicate, *, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def _reap() -> None:
    """Kill every runtime this test engaged, by exact pid.

    Scoped to the test's OWN config root: ``registry.scan()`` reads
    ``LOCAL_OPERATOR_CONFIG_DIR``, which the fixture re-homed, so a sibling
    session's runtimes are never touched.
    """
    for record, _state in registry.scan():
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)
    for record, _state in registry.scan():
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def test_cold_dismiss_boots_nothing_and_reopen_delivers(headless_tui_env: Path) -> None:
    """FORK A, the repro: no boot, no turn, no lost notice.

    Three assertions, in the order the defect appeared: (1) after the dismissal
    nothing is running and the transcript is byte-identical — on ``origin/main``
    this frame already carried ``ask-timeout`` AND a paid assistant turn; (2) the
    dismissal is durable all the same, so the phone's settled row is not a paint;
    (3) a later reopen delivers the owed notice in its dismissed wording.
    """
    directory = _seed(headless_tui_env)
    assert not (directory / ".session.pid").exists(), "the session must start STOPPED"
    before_ids = _row_ids(directory)
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    client = _client()
    try:
        with bounded(BOUND_S, "a cold dismiss boots nothing"):
            reply = client.post(
                f"/api/sessions/{SESSION}/command",
                json={"op": "ask_dismiss", "ask_id": ASK_ID},
            )
            assert reply.status_code == 200, reply.text
            assert reply.json() == {"ok": True, "detail": "dismissed"}

        # (1) NOTHING BOOTS: no owner appears and the transcript has not moved.
        # Both facts in ONE assertion so this run's failure states both, which is
        # what makes it a repro rather than a claim.
        time.sleep(NO_BOOT_SETTLE_S)
        after = (directory / "transcript.jsonl").read_text(encoding="utf-8")
        record, owner = find_runtime_record(config_dir(), SESSION)
        assert after == before and (record, owner) == (None, None), (
            "a cold dismissal must boot nothing: "
            f"runtime={owner!r}, transcript gained "
            f"{sorted(set(_row_ids(directory)) - set(before_ids))!r} "
            "(origin/main boots one and appends ask-timeout-<id> plus a paid turn)"
        )

        # (2) DURABLE ANYWAY: the log carries the row the fold reads, and the index
        # the phone reads has dropped the row from the open set.
        kinds = [event.get("kind") for event in ask_store.read_events(directory)]
        assert kinds == [ask_store.EVENT_QUEUED, ask_store.EVENT_DISMISSED]
        entry = ask_store.read_entry(headless_tui_env, SESSION)
        assert entry is not None
        assert entry["asks"][0]["status"] == ask_store.STATUS_DISMISSED
        # §17 case 2, and this PR's regression pin: a TIMED-OUT ask's dismissal
        # still removes the row. It is stamped at/after the deadline, so the
        # receipt rule does not keep it — putting a lapsed ask away still works.
        assert entry["asks"][0]["dismissed_reason"] == ask_store.DISMISS_REASON_DEFAULT
        assert ask_store.session_asks(headless_tui_env, SESSION) == []

        # (3) THE NOTICE WAITS, and a real reopen is what delivers it.
        async def reopen() -> str:
            engaged, detail = await engage_session_client(config_dir(), SESSION, WarmErrand())
            try:
                return detail
            finally:
                engaged.close()

        with bounded(BOUND_S, "a reopen delivers the owed dismissed notice"):
            detail = asyncio.run(reopen())
            row_id = ask_store.timeout_row_id(ASK_ID)
            _wait_until(
                lambda: row_id in _row_ids(directory),
                timeout_s=60,
                what="the owed deadline notice to reach the transcript",
            )
        assert detail
        row = next(row for row in _transcript_rows(directory) if row.get("id") == row_id)
        payload = json.dumps(row)
        assert "dismissed it without answering" in payload
        assert QUESTION in payload
        assert "Proceed without it" in payload
    finally:
        client.close()
        _reap()


def _append_lapsed_ask(directory: Path, ask_id: str) -> None:
    """One LAPSED ``queued`` row, appended from outside (any process may).

    The live cell needs an ask whose deadline has already passed while the
    runtime is up, which is the one state a boot reconcile cannot have delivered
    a notice for: the ask did not exist when the boot folded the log.
    """
    now = int(time.time() * 1000)
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ask_id,
            "at": now - 3_600_000,
            "created_at": now - 3_600_000,
            "expires_at": now - 60_000,
            "timeout_s": 3600,
            "urgent": False,
            "tool_call_id": "",
            "questions": [
                {
                    "id": "q1",
                    "question": QUESTION,
                    "options": [{"label": "yes", "description": ""}],
                    "multi": False,
                    "secret": False,
                    "persist": False,
                }
            ],
        },
    )


def _assistant_rows(directory: Path) -> list[dict[str, object]]:
    """The transcript's ASSISTANT turns — the rows a delivered notice buys."""
    found: list[dict[str, object]] = []
    for row in _transcript_rows(directory):
        payload = row.get("payload")
        if isinstance(payload, dict) and str(payload.get("role") or "") == "assistant":
            found.append(row)
    return found


async def _live_dismiss_and_wait(directory: Path) -> None:
    """Boot the session, put a lapsed ask in its log, dismiss it live, and wait.

    ONE event loop, deliberately: an ``AttachClient`` is bound to the loop it
    connected on, so dialling under one ``asyncio.run`` and calling an op under
    another is a closed transport (measured: ``ConnectionError: not attached``).
    The ask is appended AFTER the boot, which is the only honest way to have a
    lapsed ask whose notice is still undelivered while the session runs.
    """
    root = config_dir()
    at_client, _detail = await engage_session_client(root, SESSION, WarmErrand())
    try:
        _wait_until(
            lambda: find_runtime_record(root, SESSION)[1] is not None,
            timeout_s=60,
            what="the engaged runtime to claim the session",
        )
        _append_lapsed_ask(directory, LIVE_ASK_ID)
        row_id = ask_store.timeout_row_id(LIVE_ASK_ID)
        assert row_id not in _row_ids(directory), "the boot delivered a notice it had no ask for"
        assert await at_client.ask_dismiss(LIVE_ASK_ID, by="terminal") == "dismissed"
        _wait_until(
            lambda: row_id in _row_ids(directory),
            timeout_s=60,
            what="the live dismissal to deliver the owed notice",
        )
        _wait_until(
            lambda: bool(_assistant_rows(directory)),
            timeout_s=60,
            what="the turn the notice starts",
        )
    finally:
        at_client.close()


def test_live_dismiss_delivers_the_owed_notice_and_one_turn(headless_tui_env: Path) -> None:
    """Aida condition 1, ACCEPTED behaviour: on a LIVE session the dismissal
    delivers the notice it still owes and runs the one turn that notice is for.

    The live arm is deliberately UNCHANGED (decision D21: ``daemon.request`` →
    ``AskQueue.dismiss`` → ``_settled`` → ``_kick`` → reconcile), because a live
    runtime republishes its in-memory state and would miss an outside write. What
    this cell pins is that the reconcile it kicks still owes the notice.
    """
    directory = _seed(headless_tui_env, ask_id=LIVE_ASK_ID)
    # The lapsed ask arrives AFTER the boot, so load time has nothing to deliver
    # and the notice is genuinely undelivered when the dismissal lands.
    (directory / "asks.jsonl").write_text("", encoding="utf-8")
    client = _client()
    try:
        with bounded(BOUND_S, "a live dismissal delivers the owed notice"):
            asyncio.run(_live_dismiss_and_wait(directory))

        # The order the agent reads: the notice, THEN the turn it is for.
        row_id = ask_store.timeout_row_id(LIVE_ASK_ID)
        ids = _row_ids(directory)
        assert ids.index(row_id) > ids.index("seed-user")
        row = next(row for row in _transcript_rows(directory) if row.get("id") == row_id)
        assert "dismissed it without answering" in json.dumps(row)
        (assistant,) = _assistant_rows(directory)
        assert ids.index(str(assistant.get("id"))) > ids.index(row_id)
        assert any(
            event.get("kind") == ask_store.EVENT_DISMISSED
            for event in ask_store.read_events(directory)
        )
    finally:
        client.close()
        _reap()


def _wake_ids(directory: Path) -> list[str]:
    """The ids in the transcript's LATEST ``wake_schedules`` row."""
    latest: list[str] = []
    for row in _transcript_rows(directory):
        payload = row.get("payload")
        if isinstance(payload, dict) and payload.get("custom_type") == "wake_schedules":
            latest = [str(item.get("id")) for item in payload["details"]["schedules"]]
    return latest


def test_cold_open_ask_dismiss_boots_nothing_and_next_boot_delivers(
    headless_tui_env: Path,
) -> None:
    """D6 + D3 (design §15): an OPEN ask on a cold session is dismissible.

    (1) nothing boots and no row is added; (2) the deadline wake row is gone, so
    the supervisor cannot boot a runtime for it; (3) the dismissal is durable with
    its reason; (4) the next real boot delivers ``ask-response-<id>`` carrying the
    dismissal text — and NOT a deadline notice.
    """
    from local_operator.harness.wake import WakeSchedule

    directory = _seed(headless_tui_env, lapsed=False)
    wake_row = WakeSchedule(
        id=ask_store.timeout_row_id(ASK_ID),
        message="ask deadline",
        next_due_at=int(time.time() * 1000) + 3_600_000,
        created_at=int(time.time() * 1000),
        kind="ask_timeout",
        hidden=True,
    ).model_dump()
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "wake-entry-1",
                    "ts": time.time(),
                    "type": "custom",
                    "payload": {
                        "custom_type": "wake_schedules",
                        "details": {"schedules": [wake_row]},
                    },
                }
            )
            + "\n"
        )
    assert _wake_ids(directory) == [ask_store.timeout_row_id(ASK_ID)]
    before_ids = _row_ids(directory)
    client = _client()
    try:
        with bounded(BOUND_S, "a cold open-ask dismissal boots nothing"):
            reply = client.post(
                f"/api/sessions/{SESSION}/command",
                json={"op": "ask_dismiss", "ask_id": ASK_ID, "reason": "answered_in_chat"},
            )
            assert reply.status_code == 200, reply.text
            assert reply.json() == {"ok": True, "detail": "dismissed"}

        time.sleep(NO_BOOT_SETTLE_S)
        record, owner = find_runtime_record(config_dir(), SESSION)
        assert (record, owner) == (None, None), f"a cold dismissal booted {owner!r}"
        assert _wake_ids(directory) == [], "the deadline wake row must go with the dismissal"
        assert set(_row_ids(directory)) - set(before_ids) <= {
            str(row.get("id")) for row in _transcript_rows(directory) if _is_wake_row(row)
        }, "nothing but the wake-list rewrite may be appended"

        events = ask_store.read_events(directory)
        assert [event.get("kind") for event in events] == [
            ask_store.EVENT_QUEUED,
            ask_store.EVENT_DISMISSED,
        ]
        assert events[-1]["reason"] == "answered_in_chat"
        entry = ask_store.read_entry(headless_tui_env, SESSION)
        assert entry is not None
        assert entry["asks"][0]["status"] == ask_store.STATUS_DISMISSED
        # SA-D8 (design §17): the cold door is the row's only writer, so the WHEN
        # and WHY ride this index entry — and the reader a stopped conversation is
        # served through shows the settled receipt, stamped with the moment of the
        # tap, not with a later reconcile.
        assert entry["asks"][0]["dismissed_at"] == events[-1]["at"]
        assert entry["asks"][0]["dismissed_reason"] == "answered_in_chat"
        (shown,) = ask_store.session_asks(headless_tui_env, SESSION)
        assert shown["ask_id"] == ASK_ID
        assert shown["status"] == ask_store.STATUS_DISMISSED
        assert shown["dismissed_reason"] == "answered_in_chat"

        async def reopen() -> str:
            engaged, detail = await engage_session_client(config_dir(), SESSION, WarmErrand())
            try:
                return detail
            finally:
                engaged.close()

        with bounded(BOUND_S, "the next boot delivers the dismissal row"):
            asyncio.run(reopen())
            row_id = ask_store.response_row_id(ASK_ID)
            _wait_until(
                lambda: row_id in _row_ids(directory),
                timeout_s=60,
                what="the owed dismissal response row to reach the transcript",
            )
        row = next(row for row in _transcript_rows(directory) if row.get("id") == row_id)
        payload = json.dumps(row)
        assert "settled in the conversation" in payload
        assert '"status": "dismissed"' in payload
        assert ask_store.timeout_row_id(ASK_ID) not in _row_ids(directory)
    finally:
        client.close()
        _reap()


def _is_wake_row(row: dict[str, object]) -> bool:
    payload = row.get("payload")
    return isinstance(payload, dict) and payload.get("custom_type") == "wake_schedules"
