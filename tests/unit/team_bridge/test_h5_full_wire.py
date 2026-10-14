"""H5 — the FULL wire case: the ask can ONLY come from the RECEIVED frame.

Chain (LIVE): REAL ``mobile.peer_client.send_peer_message`` request assembly →
fake ONLY the socket (a fake ``asyncio.open_connection`` whose writer performs
the REAL ``RuntimeServer._dispatch("peer_message", frame)`` and replays its
receipt into the client's real ``_FrameReader`` loop) → REAL
``ServingSessionHandle.receive_peer_message`` → REAL ``Session.receive_peer_message``
with ``wake=True`` → the REAL wake turn's production persist
(``_append_injected_message``) calls the production hook (session.py:9592) on
the ACTUALLY persisted message. This file NEVER calls
``_peer_custom_message``/``_maybe_raise_bridge_ask``/``_live_ask``: nothing
here can create an ask if the real chain drops the locator.

EVERY wait is anchored to actual state — a durable TEAM_BRIDGE transcript row
written by the real turn plus the session's own background-task set
(``_spawn_background`` tracks every wake task; ``dispose`` walks the same set):
``_settled_after_receive`` is non-trivial for a fresh session because no such
row exists until the received frame is persisted. No sleep-based waits.

COLD adds the REAL ``launch._deliver`` errand consumer (launch.py:837) and the
REAL spool writer + ``_drain_spooled_peer_inbox_rows`` for a SECOND,
FULLY-BOUND producer message (its own delivery ref / body hash / display
digest / claim token / confirmed handoff phase, stamped through the REAL store
API — never a copied selector set).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.message_types import (
    TEAM_BRIDGE_MESSAGE_TYPE,
    display_context_digest,
)
from local_operator.mobile import peer_client as peer_client_mod
from local_operator.mobile.peer_client import send_peer_message
from local_operator.session.runtime.inbox import (
    SOURCE_PEER,
    InboxLine,
    append_inbox,
    peek_inbox,
)
from local_operator.session.runtime.server import RuntimeServer, _ack_frame_parts
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import read_entry_by_id
from local_operator.team_bridge import store as bridge_store
from local_operator.team_bridge.ask_issuance import AskIssuance
from local_operator.team_bridge.store import Store
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for
from tests.unit.team_bridge.test_h5_envelope_negatives import (
    _HINT_FIELDS,
    ADMISSION,
    DREF,
    MESSAGE_ID,
    POLICY_EPOCH,
    TEMPLATE,
    _meta,
    _owner,
    _runtime,
    _seed_store,
    _stamp,
)

pytestmark = pytest.mark.asyncio

DREF2 = "dref_" + "cd" * 16
MESSAGE_ID_2 = "bmsg_2"
#: Op sequence across fakes: proves the RE-SEND actually dispatched a NEW op
#: through the real chain (a count that is not already true before it).
_LAST_OPS: list[str] = []

#: The LOCAL transport ids: the receiver accepts ``peer-<32 lowercase hex>``
#: as a transcript entry id. The server bridge ids (``bmsg…``) live ONLY in
#: the bridge metadata — the two domains stay pinned apart, exactly as the
#: real receiver stamps them (a ``bmsg…`` id is not a valid local peer id).
LOCAL_ID = "peer-" + "ef" * 16
LOCAL_ID_2 = "peer-" + "99" * 16


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _FakeRead:
    """Feeds newline-framed bytes to the client's ``_FrameReader``."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self.ops: list[str] = []

    def queue(self, payload: bytes) -> None:
        self._chunks.append(payload)

    async def read(self, _n: int = 65536) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeWrite:
    """Captures the client's frames and answers from the REAL dispatch."""

    def __init__(self, read: _FakeRead, runtime: RuntimeServer) -> None:
        self._read = read
        self._runtime = runtime
        self._writes = 0

    def write(self, payload: bytes) -> None:
        self._writes += 1
        if self._writes < 2:
            return  # the auth key line: read as bytes, never replayed
        frame = json.loads(payload.decode("utf-8"))
        # The REAL server loop hands ``_dispatch`` the WHOLE decoded frame —
        # ``op`` and ``req`` included — and ``validate_control_frame`` refuses a
        # frame with no ``op``. Stripping them here made the fake stricter than
        # nothing the product does and failed every case before the hook ran.
        self._pending = dict(frame)
        self._op = str(frame.get("op") or "")
        self._req = int(frame.get("req") or 1)

    async def drain(self) -> None:
        if self._writes < 2:
            return
        # THE SOCKET'S ONLY FAKED JOB: run the REAL op and replay its receipt
        # (or the error frame the server loop would build from the exception).
        try:
            receipt = await self._runtime._dispatch(self._op, self._pending)
            # Build the frame with the server loop's OWN helper
            # (server.py `_ack_frame_parts`, called at the ack build in the
            # reader loop): the handle answers with an `AckDetail` whose
            # `delivery` rides `.fields`, NOT a `.delivery` attribute, so a
            # hand-rolled `getattr(receipt, "delivery")` read None and the
            # sender's PeerAck lost the classification (and `str(receipt)` was
            # the dataclass repr, not the receipt sentence).
            detail, extra = _ack_frame_parts(receipt)
            ack = {"req": self._req, "op": "ack", "detail": detail, **extra}
        except Exception as failure:  # noqa: BLE001 — mirror the loop's error frame
            ack = {"req": self._req, "op": "error", "message": str(failure)}
        self._read.ops.append(self._op)
        self._read.queue(json.dumps(ack).encode() + b"\n")

    def close(self) -> None:  # pragma: no cover — the client's finally
        return None

    async def wait_closed(self) -> None:  # pragma: no cover
        return None


def _wire(runtime: RuntimeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    global _LAST_OPS

    read = _FakeRead()
    _LAST_OPS = read.ops

    async def fake_open_connection(*_args, **_kwargs):
        return read, _FakeWrite(read, runtime)

    monkeypatch.setattr(peer_client_mod.asyncio, "open_connection", fake_open_connection)


async def _full_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real Session + serving handle + server + enrolled native store."""

    session = make_session(tmp_path, ScriptedStream([[], [], []]))

    # The ask queue only exists once a front end has installed the ask hook
    # (``Session.ask_queue`` returns ``None`` otherwise, and the bridge raise
    # path then leaves a durable ``no_queue`` pending instead of an ask). The
    # hook is never consulted by the queued arm.
    async def _never_answered(_questions):
        return None  # pragma: no cover

    session.set_ask_handler(_never_answered)
    store = _seed_store(tmp_path)  # enrollment + consent + claim + schema, via REAL reads
    _stamp(store)
    # The REAL ``authority_fence`` (the hook calls it; nothing patches it here)
    # reads ``team_bridge.enabled`` from the config on DISK and fails closed to
    # ``feature_off`` without it — no issuance row, no ask. Same real file
    # shape ``test_connector._wire_spine`` writes (``values`` of config.yml).
    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )
    runtime_stub = _runtime(tmp_path)

    async def _pinned_access():
        # The runtime's OWN pinned authority surface (S-8b): pytest fixtures
        # stand the real ``current_access`` in for the standing reader.
        return SimpleNamespace(credential_id=_owner().credential_id)

    runtime_stub.client = SimpleNamespace(current_access=_pinned_access)
    session.bridge_runtime = runtime_stub
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="tui")
    # The gate's own bit, through the production setter (server.py:4969): no
    # socket accept loop is booted here, and that is the ONLY fixture bit.
    server.set_record_started(True)
    _wire(server, monkeypatch)
    record = SimpleNamespace(control_port=0, control_key="k")
    return session, record


def _asks_for(session, message_id: str) -> list[dict]:
    """REAL queue rows from the BRIDGE builder for one message id."""

    rows = []
    for row in session.ask_queue().records():
        blob = json.dumps(row)
        if "ask_id" in row and "bridge-" in blob and message_id in blob:
            rows.append(row)
    return rows


def _bridge_row_exists(session) -> bool:
    """The durable evidence the wake turn ran: its persisted bridge row."""

    # By the SENDER-minted local id, not by custom type: a received peer
    # ``CustomMessage`` is appended as an ``ENTRY_MESSAGE`` row (``kind:
    # custom``), while the by-custom-type reader matches ``ENTRY_CUSTOM`` rows
    # only — so that probe is never true for this row and every wait built on
    # it times out regardless of what the product did.
    entry = read_entry_by_id(session.transcript.directory, LOCAL_ID)
    return (
        entry is not None
        and str((entry.payload or {}).get("custom_type") or "") == TEAM_BRIDGE_MESSAGE_TYPE
    )


def _tasks_pending(session) -> bool:
    """The session's own task set, the one ``_spawn_background`` maintains."""

    tasks = getattr(session, "_background_tasks", None)
    return bool(tasks)


async def _settled_after_receive(session, *, asks: int | None = None) -> None:
    """Await the ACTUAL durable row + the wake task settling — never a sleep.

    Non-trivial for a fresh session: no TEAM_BRIDGE row exists until the
    received frame is persisted by the real turn, so the predicate cannot be
    true before the receive path actually ran.
    """

    await wait_for(lambda: _bridge_row_exists(session) and not _tasks_pending(session), 5.0)
    if asks is not None:
        await wait_for(lambda: len(_asks_for(session, MESSAGE_ID)) == asks, 5.0)


def _seed_second_envelope(session) -> dict:
    """A FULLY-BOUND second producer message through the REAL store APIs.

    Own delivery ref, body hash, display digest (over its own hint fields) and
    claim token; its claims row and its confirmed handoff phase are written the
    same way the first message's were (SQL claim boundary + the real
    ``write_screened_request_delivery`` / ``mark_delivery_phase`` pair), so the
    receiver validates the SAME original-scope facts as message one — never
    copied selectors.
    """

    hints = dict(_HINT_FIELDS)
    hints["excerpt"] = "second request"
    digest = display_context_digest(hints)
    body = hashlib.sha256(b"second body").hexdigest()
    store = Store(session.bridge_runtime.config_dir)
    key = bridge_store._owner_key(_owner())
    own_envelope_digest = hashlib.sha256(b"bmsg-2-envelope").hexdigest()
    conn = sqlite3.connect(store._path)
    try:
        conn.execute(
            "INSERT INTO claims (owner_key, message_id, session_id, token, generation, "
            "expires_at, state) VALUES (?,?,?,?,?,?,?)",
            (
                key,
                MESSAGE_ID_2,
                "s1",
                "tok-2",
                _owner().enrollment_generation,
                time.time() + 300,
                "held",
            ),
        )
        # Message 2's OWN screening receipt, mirroring the first message's row
        # shape (same owner-scoped policy/admission) with ITS body hash and ITS
        # envelope digest: the native writer refuses a message with NO receipt
        # (no_receipt), and a copied receipt would be the wrong identity —
        # message 2 carries its own.
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                MESSAGE_ID_2,
                body,
                TEMPLATE,
                "prompt-v1",
                "model-a",
                "allow",
                json.dumps(["spam"]),
                own_envelope_digest,
                ADMISSION,
                POLICY_EPOCH,
                _owner().enrollment_generation,
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    store.write_screened_request_delivery(
        _owner(),
        delivery_ref=DREF2,
        message_id=MESSAGE_ID_2,
        designated_session="s1",
        display_digest=digest,
        body_sha256=body,
        claim_token="tok-2",
        request={
            "thread_id": "t-1",
            "message_kind": "request",
            "request_state": "pending",
            "request_expires_at": "",
            "request_ref": "",
        },
    )
    store.mark_delivery_phase(_owner(), DREF2, "handoff_started")
    return _meta(
        message_id=MESSAGE_ID_2,
        delivery_ref=DREF2,
        body_sha256=body,
        display_digest=digest,
        **hints,
    )


async def test_live_full_chain_ask_can_only_come_from_the_received_frame(
    tmp_path, isolated_config, monkeypatch
) -> None:
    session, record = await _full_stack(tmp_path, monkeypatch)
    meta = _meta()
    ack = await send_peer_message(
        record,
        text="[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        message_id=LOCAL_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta,
    )
    assert str(ack)
    # Settle = durable persisted row + wake task done; only THEN is the ask a
    # claim about the real chain rather than about test construction.
    await _settled_after_receive(session, asks=1)
    asks = _asks_for(session, MESSAGE_ID)
    assert len(asks) == 1
    ask_id = str(asks[0].get("ask_id"))
    serialized = json.dumps(asks[0])
    assert DREF in serialized and str(_owner().account_id) in serialized
    assert AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)["state"] == "raised"
    # H5-D2/A: capture the SESSION's own monotonic intake count BEFORE the
    # re-send. The re-send carries the SAME valid LOCAL peer id, so the
    # RECEIVER's duplicate branch is the expected production path — completion
    # must therefore come from a fresh duplicate ACK receipt (ops advancing),
    # with the arrival counter INTENTIONALLY unchanged; a counter move here
    # would mean the dedupe failed, not that the send completed.
    before_arrival = session._peer_arrival.count()
    # Re-send the SAME message through the same real chain: the durable
    # identity resolves to ONE ask.
    second = await send_peer_message(
        record,
        text="[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        message_id=LOCAL_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta,
    )
    # The TYPED duplicate proof rides the SAME frame the real server sends:
    # PeerAck is a str subclass whose `.delivery` carries the receiver's own
    # classification — assert the authentic fields, never a sentence.
    assert second.delivery is not None and second.delivery.get("duplicate") is True
    assert str(second.delivery.get("message_id") or "") == LOCAL_ID
    # The RE-SEND's completion is a NEW op on the same real chain — a
    # predicate FALSE before the re-send dispatched — plus the session's own
    # task set draining. The duplicate classification is `second.delivery` above; the
    # arrival counter MUST NOT move (no new turn/row/ASK).
    await wait_for(lambda: len(_LAST_OPS) >= 2 and not _tasks_pending(session), 5.0)
    assert session._peer_arrival.count() == before_arrival
    assert len(_asks_for(session, MESSAGE_ID)) == 1
    assert str(_asks_for(session, MESSAGE_ID)[0].get("ask_id")) == ask_id
    # SCENARIO B: a NEW valid LOCAL peer id carrying the SAME server bridge
    # identity (meta.message_id/delivery_ref unchanged) — the receiver accepts
    # a NEW durable incoming row and a real new turn, while the ASK/issuance
    # stays ONE (dedupe by the original bridge identity, never by the local
    # id, and the write-once local_row_id mapping keeps the FIRST row).
    before_b = session._peer_arrival.count()
    # Anchored FALSE before the send: this NEW local id has NO durable row
    # yet — after the send it must exist as its OWN intake row.
    wire_dir = session._transcript.directory
    from local_operator.session.transcript import read_entry_by_id as _read_row

    assert _read_row(wire_dir, LOCAL_ID_2) is None
    third = await send_peer_message(
        record,
        text="[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        message_id=LOCAL_ID_2,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta,
    )
    assert third.delivery is not None
    assert third.delivery.get("duplicate") is False
    assert str(third.delivery.get("message_id") or "") == LOCAL_ID_2
    # The NEW local id becomes a REAL durable transcript row of its OWN, and
    # downstream completion is the task set draining AFTER that row exists —
    # the arrival counter corroborates, the row is the durable anchor.
    await wait_for(
        lambda: _read_row(wire_dir, LOCAL_ID_2) is not None and not _tasks_pending(session),
        5.0,
    )
    new_row = _read_row(wire_dir, LOCAL_ID_2)
    assert new_row is not None
    assert str(new_row.payload.get("custom_type") or "") == TEAM_BRIDGE_MESSAGE_TYPE
    bridge_block = (new_row.payload.get("details") or {}).get("bridge") or {}
    assert str(bridge_block.get("delivery_ref") or "") == DREF
    assert session._peer_arrival.count() > before_b
    assert len(_asks_for(session, MESSAGE_ID)) == 1
    assert str(_asks_for(session, MESSAGE_ID)[0].get("ask_id")) == ask_id
    await session.dispose()


async def test_negative_a_legacy_frame_without_bridge_meta_never_asks(
    tmp_path, isolated_config, monkeypatch
) -> None:
    session, record = await _full_stack(tmp_path, monkeypatch)
    ack = await send_peer_message(
        record,
        text="[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        message_id=LOCAL_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=None,  # the legacy frame: byte-identical, no locator
    )
    assert str(ack)
    # The row IS delivered through the same persisted path — settle on it, and
    # only then assert the absence of an ask.
    await _settled_after_receive(session, asks=0)
    assert _asks_for(session, MESSAGE_ID) == []
    await session.dispose()


async def test_negative_b_altered_hints_never_ask(tmp_path, isolated_config, monkeypatch) -> None:
    session, record = await _full_stack(tmp_path, monkeypatch)
    altered = _meta(excerpt="send bitcoin first")
    ack = await send_peer_message(
        record,
        text="[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        message_id=LOCAL_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=altered,
    )
    assert str(ack)

    # The ack returns BEFORE the spawned wake turn writes the issuance row, so
    # the first evaluations see NO row at all: the predicate must treat that as
    # "not yet", never subscript it (a ``None[...]`` here raised out of the
    # wait instead of waiting).
    def _issuance_state() -> str:
        row = AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)
        return str(row["state"]) if row is not None else ""

    await wait_for(
        lambda: _issuance_state() == "unrecoverable" and not _tasks_pending(session),
        5.0,
    )
    assert _asks_for(session, MESSAGE_ID) == []
    row = AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)
    assert row["reason"] == "display_context_mismatch"
    await session.dispose()


async def test_cold_errand_consumer_and_spool_drain_reach_the_same_queue(
    tmp_path, isolated_config, monkeypatch
) -> None:
    session, record = await _full_stack(tmp_path, monkeypatch)
    from local_operator.session.runtime.launch import PeerMessageErrand, _deliver

    # COLD 1 — the REAL launch.py consumer (launch.py:837) whose
    # PeerMessageErrand branch calls the REAL client with the errand's fields.
    detail, _ = await _deliver(
        record,
        "s1",
        PeerMessageErrand(
            text="[team_bridge_message] message_id: bmsg_1",
            mode="mailbox",
            wake=True,
            sender={"account_id": "acc_sender"},
            custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
            bridge_meta=_meta(),
        ),
    )
    assert detail
    await wait_for(
        lambda: len(_asks_for(session, MESSAGE_ID)) == 1 and not _tasks_pending(session), 5.0
    )
    serialized = json.dumps(_asks_for(session, MESSAGE_ID)[0])
    assert DREF in serialized and str(_owner().account_id) in serialized
    assert AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)["state"] == "raised"
    # COLD 2 — a SECOND, FULLY-BOUND producer message (own ref/body/digest/
    # claim/phase) through the REAL spool writer and the REAL drain, which
    # drives the same wake turn: the hook runs on the persisted row only.
    meta2 = _seed_second_envelope(session)
    line = InboxLine(
        text="[team_bridge_message] message_id: bmsg_2",
        sender={"account_id": "acc_sender"},
        mode="mailbox",
        wake=True,
        source=SOURCE_PEER,
        message_id=LOCAL_ID_2,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta2,
    )
    assert append_inbox(session.transcript.directory, line) is True
    # THE DRAIN IS ONCE PER SESSION LIFETIME (session.py `_drain_spooled_peer_inbox_rows`
    # returns at its `_peer_inbox_drained` latch). COLD 1's wake turn already ran
    # `_run_turn_pipeline`, whose first-turn drain latched it on this very Session
    # — so a direct call here was a product-correct no-op and the row survived.
    # Anchor that diagnosis (latched now), then model what the contract describes
    # for a row written AFTER the first-turn drain: a fresh lifetime, which is
    # exactly the state `test_h5_wire_chain` drains from on a never-turned session.
    assert session._peer_inbox_drained is True
    session._peer_inbox_drained = False
    await session._drain_spooled_peer_inbox_rows()
    assert peek_inbox(session.transcript.directory) == []
    await wait_for(
        lambda: len(_asks_for(session, MESSAGE_ID_2)) == 1 and not _tasks_pending(session), 5.0
    )
    cold = _asks_for(session, MESSAGE_ID_2)
    assert len(cold) == 1
    serialized2 = json.dumps(cold[0])
    assert DREF2 in serialized2 and str(_owner().account_id) in serialized2
    assert AskIssuance(tmp_path).read(_owner(), MESSAGE_ID_2)["state"] == "raised"
    await session.dispose()


assert Session  # the real class is exercised via the real stack above
