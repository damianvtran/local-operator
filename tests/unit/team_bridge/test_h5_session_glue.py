"""H5 — the REAL Session consume glue and concurrent settlement (SOURCE).

The session is the REAL constructed one (``make_session`` from the session
test suite: real Session, real Transcript, real ask queue, scripted external
model stream). Only the EXTENAL boundaries are faked: the decide transport
(patched at ``decide.decide_from_human_answer`` — the same seam the provenance
suite patches) and the native store acting as the bridge runtime's authority
(a REAL enrollment is committed through the real seam).

No internal door is substituted: the ASK response enters through
``Session._append_injected_message`` — the production consumption path that
re-resolves the row from the fold, commits it durably, closes the revision
window and is the ONLY place a decide may be dispatched from. The concurrent
arms overlap two consumption attempts with an ``asyncio.Event`` barrier; the
row-commit guard serializes them, and the settle count is asserted against the
REAL recorder.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import decide as decide_mod
from local_operator.team_bridge.asks_bridge import enqueue_bridge_request_ask
from local_operator.team_bridge.authority import scope_from_owner
from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeSignInRequired,
    BridgeUnreachable,
)
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _tasks_pending(session) -> bool:
    """The session's own background task set — settle evidence, never a sleep."""

    return bool(getattr(session, "_background_tasks", None))


def _decide_rows(tmp_path: Path) -> list[dict[str, Any]]:
    """Durable decide-ledger rows, read through the store's own connection.

    The settle predicate needs the LEDGER's own state (receipt/error/typed
    state), not just the transport effect: the row is the durable authority.
    """

    from local_operator.team_bridge.decide_intents import DecideIntents

    conn = DecideIntents(tmp_path)._connect()
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM decide_intents")]
    finally:
        conn.close()


def _decide_settled(tmp_path: Path, outcome: str) -> bool:
    """Whether ANY durable row reflects the outcome's OWN settled shape.

    success → state `sent` with a receipt; transient → still `pending` BUT a
    typed error recorded (initial `pending` alone is NEVER settled); signin →
    `signin` with its typed code; refused → `refused` with its code.
    """

    rows = _decide_rows(tmp_path)
    if outcome == "success":
        return any(str(r.get("state")) == "sent" and r.get("receipt_json") for r in rows)
    if outcome == "transient":
        return any(
            str(r.get("state")) == "pending" and str(r.get("error_code") or "") for r in rows
        )
    return any(str(r.get("state")) == outcome for r in rows)


pytestmark = pytest.mark.asyncio

MESSAGE_ID = "bmsg_glue"
OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="ep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=5,
)


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _session(tmp_path: Path):
    """A REAL Session + the native enrollment its bridge runtime stands on."""

    stream = ScriptedStream([[]])
    session = make_session(tmp_path, stream)

    # ``Session.ask_queue()`` is ``None`` until a front end installs the ask
    # hook (the queue is gated on it, session.py ``ask_queue``). The product's
    # raise path guards that ``None``; this fixture stands in for the attached
    # front end so the REAL queue exists. The hook itself is never consulted:
    # the queued arm enqueues without it.
    async def _never_answered(_questions):
        return None  # pragma: no cover

    session.set_ask_handler(_never_answered)
    Store(tmp_path).commit_enrollment(
        OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=session.session_id
    )
    session.bridge_runtime = SimpleNamespace(
        owner=OWNER,
        config_dir=tmp_path,
        session_id=session.session_id,
        tenant_id="tnt_1",
        client=object(),
        # The product keys the decide-intent row by ``BridgeRuntime.key()``
        # (session.py ``_dispatch_bridge_decisions``); a stub without it makes
        # the record step fail contained and NOTHING ever settles. Same
        # construction as ``BridgeRuntime.key()`` with no ``owner_key`` set.
        key=lambda: json.dumps([OWNER.canonical_base_url, OWNER.account_id, OWNER.tenant_id]),
    )
    return session


async def _answered_bridge_ask(session, tmp_path: Path):
    """One REAL bridge ask, answered by the human, re-resolved from the fold."""

    queue = session.ask_queue()
    scope = scope_from_owner(OWNER, session.session_id)
    fingerprint = hashlib.sha256(
        json.dumps(dict(scope), separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    outcome = enqueue_bridge_request_ask(
        queue,
        message_id=MESSAGE_ID,
        sender="dana@acme.example",
        excerpt="please approve",
        scope=scope,
        issuance_key="iss-glue-1",
        fingerprint=fingerprint,
    )
    assert outcome["ok"] is True
    ask_id = str(outcome["details"]["ask_id"])
    # TB-02 P1 (PR #2378): in the product the runtime's authority seam mints
    # this one-shot grant when it ADMITS the human's `bridge_decide` frame
    # (exercised by `test_approval_authority_seam.py`); this fixture builds the
    # answer directly, so it stands in for that admission — the dispatcher
    # refuses a decision carrying no grant, and these cells must keep testing
    # the consumption/settlement machinery, not the admission.
    from local_operator.harness.approval import mint_bridge_decide_grant

    mint_bridge_decide_grant(
        session_id=session.session_id, ask_id=ask_id, question_id=MESSAGE_ID, word="approve"
    )
    answered = queue.respond(ask_id, {f"bridge-{MESSAGE_ID}": ["approve"]}, by="human")
    assert answered.get("ok") is True
    record = [r for r in queue.records() if str(r.get("ask_id") or "") == ask_id][-1]
    return ask_id, queue.refresh_delivery_message(queue._response_message(record))


async def test_duplicate_consumption_and_reconcile_never_post_twice(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _session(tmp_path)
    ask_id, message = await _answered_bridge_ask(session, tmp_path)
    posted: list[str] = []

    async def fake_decide(client, decision, **kwargs):
        posted.append(str(getattr(decision, "word", "")))
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", fake_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    first = await session._append_injected_message(message)
    assert first is not None
    # The dispatch runs in a background task; settle on the transport ACTUALLY
    # having been entered (0 → 1 is non-trivial) before asserting counts.
    await wait_for(
        lambda: len(posted) == 1
        and not _tasks_pending(session)
        and _decide_settled(tmp_path, "success"),
        5.0,
    )  # consumed answer + dispatch done + durable receipt
    # The SAME durable row re-appended (boot reconcile / steering batch):
    # only the FIRST write is a new consumption — a re-append is not.
    second = await session._append_injected_message(message)
    assert second is not None
    assert len(posted) == 1
    await session.reconcile_asks()
    assert len(posted) == 1  # a terminal (sent) intent is never replayed
    await session.dispose()


async def test_a_transient_dispatch_settles_without_automatic_retries(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _session(tmp_path)
    ask_id, message = await _answered_bridge_ask(session, tmp_path)
    attempts: list[str] = []

    async def failing_decide(client, decision, **kwargs):
        attempts.append(str(getattr(decision, "word", "")))
        # The TYPED transport arm the production executor settles on: a bare
        # TimeoutError would bypass attempt_decide_intent's except arms and
        # never record a ledger row, so the durable settle predicate could
        # never be satisfied by this arm.
        raise BridgeUnreachable()

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", failing_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    try:
        await session._append_injected_message(message)
    except Exception:  # noqa: BLE001 — a failed dispatch is contained by design
        pass
    await wait_for(
        lambda: len(attempts) == 1
        and not _tasks_pending(session)
        and _decide_settled(tmp_path, "transient"),
        5.0,
    )
    # Re-append (the durable row already exists) and reconcile: NO automatic
    # POST at any boundary — only an explicit human Retry may replay.
    try:
        await session._append_injected_message(message)
    except Exception:  # noqa: BLE001
        pass
    await session.reconcile_asks()
    assert len(attempts) == 1
    await session.dispose()


async def test_transport_entry_is_not_settlement(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FALSE-PREDICATE CONTROL: entered ≠ settled.

    The external fake is held INSIDE the transport by an event: while it is
    entered-but-blocked no receipt exists and no count may move; only after
    the response is released does the settlement land. "Transport entered" is
    therefore strictly weaker than completion — asserted FALSE as a
    completion check at the discriminating moment, so a suite can never pass
    on entry alone.
    """

    session = await _session(tmp_path)
    ask_id, message = await _answered_bridge_ask(session, tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    posted: list[str] = []

    async def held_decide(client, decision, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), 5.0)
        posted.append(str(getattr(decision, "word", "")))
        return {"request": {"receipt": "orig-1"}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", held_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    dispatch = asyncio.create_task(session._append_injected_message(message))
    await asyncio.wait_for(entered.wait(), 5.0)
    # ENTERED, response BLOCKED: the transport-entry predicate is TRUE while
    # settlement has NOT happened — the discriminating slack.
    assert posted == []
    release.set()
    await dispatch
    await wait_for(
        lambda: len(posted) == 1
        and not _tasks_pending(session)
        and _decide_settled(tmp_path, "success"),
        5.0,
    )
    await session.dispose()


@pytest.mark.parametrize("outcome", ["success", "transient", "signin", "refused"])
async def test_every_settlement_state_duplicates_without_new_posts(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """The explicit state matrix: success, transient, signin, refused.

    Whichever arm the FIRST consumption took, the durable row re-appended at a
    later boundary and the actual reconcile must never POST again: only an
    explicit trusted-human Retry may replay (and it is not reachable here).
    """

    session = await _session(tmp_path)
    ask_id, message = await _answered_bridge_ask(session, tmp_path)
    attempts: list[str] = []

    async def behaving_decide(client, decision, **kwargs):
        attempts.append(str(getattr(decision, "word", "")))
        if outcome == "success":
            return {"request": {"receipt": "1"}}
        if outcome == "transient":
            raise BridgeUnreachable()
        if outcome == "signin":
            raise BridgeSignInRequired("sign in to Radient")
        raise BridgeRefusal("invalid_request", "the request is malformed")

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", behaving_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    try:
        await session._append_injected_message(message)
    except Exception:  # noqa: BLE001 — a failed dispatch is contained by design
        pass
    await wait_for(
        lambda: len(attempts) == 1
        and not _tasks_pending(session)
        and _decide_settled(tmp_path, outcome),
        5.0,
    )  # entry + dispatch done + durable typed state
    try:
        await session._append_injected_message(message)  # re-append: same row
    except Exception:  # noqa: BLE001
        pass
    await session.reconcile_asks()
    assert len(attempts) == 1  # every state: no automatic replay, ever
    # The LEDGER STATE that arm actually settled — read from the real DB, not
    # inferred from the HTTP outcome. The late re-append/reconcile above must
    # leave this exactly as the first consumption wrote it.
    import sqlite3

    from local_operator.team_bridge.decide_intents import decide_intents_db_path

    expected = {
        "success": "sent",
        "transient": "pending",
        "signin": "signin",
        "refused": "refused",
    }[outcome]
    conn = sqlite3.connect(decide_intents_db_path(tmp_path))
    try:
        states = [row[0] for row in conn.execute("SELECT state FROM decide_intents")]
    finally:
        conn.close()
    assert states == [expected]
    await session.dispose()


async def test_concurrent_consumption_settles_once_behind_a_barrier(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _session(tmp_path)
    ask_id, message = await _answered_bridge_ask(session, tmp_path)
    posted: list[str] = []
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def barred_decide(client, decision, **kwargs):
        posted.append(str(getattr(decision, "word", "")))
        arrived.set()
        await released.wait()
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", barred_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    first = asyncio.create_task(session._append_injected_message(message))
    await asyncio.wait_for(arrived.wait(), 2.0)
    # A SECOND attempt overlaps while the first is mid-flight (inside the
    # dispatch await): the row-commit guard must serialize the consumption.
    second = asyncio.create_task(session._append_injected_message(message))
    released.set()
    await asyncio.wait_for(asyncio.gather(first, second), 5.0)
    assert len(posted) == 1  # one consumption, one settle
    await session.reconcile_asks()
    assert len(posted) == 1
    await session.dispose()
