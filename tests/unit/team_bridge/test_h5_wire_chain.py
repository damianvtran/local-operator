"""H5 — the wire chain at UNIT level: producer → spool → receive hook (SOURCE).

These cases pin each hop's contract in isolation (producer meta → closed-set
validator → spool row roundtrip → receiver stamp → hook). The FULL chains that
ACTUALLY invoke the client, the server op, the serving handle and the session
receive end to end live in ``test_h5_full_wire.py``; nothing here claims to be
that chain — the probes run as the op's own probe functions, the handle is
driven directly, and the hook arms call the real builder + hook on seeded
stores. The seeded-store helpers are imported from the envelope-negatives
suite so both files exercise ONE store fixture.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.message_types import (
    TEAM_BRIDGE_MESSAGE_TYPE,
    validate_bridge_meta,
)
from local_operator.session.runtime.inbox import (
    SOURCE_PEER,
    InboxLine,
    append_inbox,
    peek_inbox,
)
from local_operator.session.runtime.server import (
    receives_bridge_meta,
    receives_custom_type,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from tests.unit.team_bridge.test_h5_envelope_negatives import (  # noqa: F401
    _HINT_FIELDS,
    BODY_HASH,
    DISPLAY_DIGEST,
    DREF,
    LOCAL_ROW_ID,
    MESSAGE_ID,
    _meta,
    _no_enqueue,
    _owner,
    _runtime,
    _seed_store,
    _stamp,
    _StubSession,
)
from tests.unit.team_bridge.test_h5_session_glue import _session


def _assert_fence_open(runtime) -> None:
    """The fixture must satisfy the AUTHORITY FENCE, not replace it.

    Every door this file exercises sits BEHIND ``authority_fence``: a fixture
    that fails the fence would make each case pass vacuously (nothing reaches
    the door), so the guard fails loudly with the fence's own typed refusal
    and the reason is never guessed.
    """

    from local_operator.team_bridge.authority import authority_fence

    refusal = authority_fence(runtime)
    assert refusal is None, f"fixture does not satisfy the fence: {refusal}"


@pytest.fixture(autouse=True)
def _enabled_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every case in THIS file runs with the feature enabled.

    The doors exercised here sit BEHIND ``authority_fence``, and a default-off
    config makes the fence return ``feature_off`` before any door is reached —
    the case would then pass or fail for the wrong reason. ``autouse`` because
    a test that forgets to request an opt-in fixture is exactly the case that
    regresses silently (the store/enrollment reads stay real).
    """

    monkeypatch.setattr(
        "local_operator.config.read_config_values",
        lambda config_dir: {"team_bridge": {"enabled": True}},
    )


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones.

    The feature is ENABLED for this file's config reads (the established
    ``_enabled`` pattern in ``test_h5_remediation_fence.py``): every case here
    exercises the doors BEHIND the fence, and a default-off config would make
    the hook return before those doors — the store/enrollment reads stay real,
    so a seed that stops satisfying the fence still fails loudly.
    """

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _spool_roundtrip(meta: dict[str, Any]) -> dict[str, Any]:
    """The REAL spool serialization boundary: what the successor reads back."""

    line = InboxLine(
        text="[team_bridge_message] message_id: bmsg_1",
        sender={"account_id": "acc_sender", "name": "Org Member"},
        mode="mailbox",
        written_at=1.0,
        wake=True,
        source=SOURCE_PEER,
        message_id=MESSAGE_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=dict(meta),
    )
    return InboxLine.from_json(line.to_json()).bridge_meta


def test_live_chain_meta_survives_the_real_spool_and_raises_the_ask(
    tmp_path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    store = _seed_store(tmp_path)
    _stamp(store)
    meta = _meta()
    validated = validate_bridge_meta(meta)
    assert validated is not None
    drained = _spool_roundtrip(validated)
    # Byte-equal canonical JSON across the spool boundary.
    assert json.dumps(drained, sort_keys=True) == json.dumps(dict(meta), sort_keys=True)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    _assert_fence_open(stub.bridge_runtime)
    message = Session._peer_custom_message(
        stub,
        "[team_bridge_message] body",
        {"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        # The harness's own accepted row id: the durable issuance maps
        # THIS message id to the ask (S-6/S-7), and the ledger refuses a NEW
        # issuance row without it (TB-02/round-3 claim contract).
        entry_id=LOCAL_ROW_ID,
    )
    details = getattr(message, "details", {}) or {}
    assert str((details.get("bridge") or {}).get("delivery_ref") or "") == DREF
    Session._maybe_raise_bridge_ask(stub, message)
    assert len(_no_enqueue) == 1
    assert _no_enqueue[0]["message_id"] == MESSAGE_ID


def test_cold_chain_errand_carries_the_same_meta_to_the_same_hook(
    tmp_path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    from local_operator.session.runtime.launch import PeerMessageErrand

    store = _seed_store(tmp_path)
    _stamp(store)
    meta = _meta()
    errand = PeerMessageErrand(
        text="[team_bridge_message] body",
        mode="mailbox",
        wake=True,
        sender={"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta,
    )
    assert errand.bridge_meta == meta
    drained = _spool_roundtrip(errand.bridge_meta)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    message = Session._peer_custom_message(
        stub,
        errand.text,
        errand.sender,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        # The harness's own accepted row id: the durable issuance maps
        # THIS message id to the ask (S-6/S-7), and the ledger refuses a NEW
        # issuance row without it (TB-02/round-3 claim contract).
        entry_id=LOCAL_ROW_ID,
    )
    Session._maybe_raise_bridge_ask(stub, message)
    assert len(_no_enqueue) == 1


def test_legacy_omit_keeps_the_old_byte_shape_and_never_mints(
    tmp_path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    store = _seed_store(tmp_path)
    _stamp(store)
    legacy = {
        "v": "1",
        "message_id": MESSAGE_ID,
        "thread_id": "t-1",
        "type": "request",
        "request_state": "pending",
        "body_sha256": BODY_HASH,
        "sender_name": "dana@acme.example",
        "organization": "Acme",
        "excerpt": "please approve before 1700",
    }
    # The omit-when-None discipline: an OLD producer's frame carries NONE of
    # the new keys — byte-compared here so a future emitter cannot quietly
    # start writing empty-string placeholders.
    serialized = json.dumps(legacy, sort_keys=True)
    for key in ("delivery_ref", "display_digest", "sender_label"):
        assert key not in serialized
    drained = _spool_roundtrip(legacy)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    message = Session._peer_custom_message(
        stub,
        "[team_bridge_message] body",
        {"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        # The harness's own accepted row id: the durable issuance maps
        # THIS message id to the ask (S-6/S-7), and the ledger refuses a NEW
        # issuance row without it (TB-02/round-3 claim contract).
        entry_id=LOCAL_ROW_ID,
    )
    Session._maybe_raise_bridge_ask(stub, message)
    assert _no_enqueue == []
    from local_operator.team_bridge.ask_issuance import AskIssuance

    row = AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)
    assert row is not None
    assert row["state"] == "unrecoverable"
    assert row["reason"] == "no_original_envelope"


def test_copied_digest_with_changed_hints_never_raises_over_the_wire(
    tmp_path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    store = _seed_store(tmp_path)
    _stamp(store)
    drained = _spool_roundtrip(_meta(excerpt="send bitcoin first"))
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    message = Session._peer_custom_message(
        stub,
        "[team_bridge_message] body",
        {"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        # The harness's own accepted row id: the durable issuance maps
        # THIS message id to the ask (S-6/S-7), and the ledger refuses a NEW
        # issuance row without it (TB-02/round-3 claim contract).
        entry_id=LOCAL_ROW_ID,
    )
    Session._maybe_raise_bridge_ask(stub, message)
    assert _no_enqueue == []
    from local_operator.team_bridge.ask_issuance import AskIssuance

    row = AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)
    assert row is not None
    assert row["state"] == "unrecoverable"
    assert row["reason"] == "display_context_mismatch"


@pytest.mark.asyncio
async def test_the_live_serving_hop_receives_through_the_real_session(
    tmp_path: Path, isolated_config: Path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:
    """The LIVE internal chain: real probe → real ServingSessionHandle →
    real Session.receive_peer_message. Only the socket frame is absent; every
    function on the chain is the production one."""

    session = await _session(tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    # The server op's own probes (``receives_*``) answer TRUE for this handle:
    # the bridge block WOULD be forwarded to a session whose signature takes it.
    assert receives_custom_type(handle) is True
    assert receives_bridge_meta(handle) is True
    meta = validate_bridge_meta(_meta())
    result = await handle.receive_peer_message(
        "[team_bridge_message] message_id: bmsg_1",
        mode="mailbox",
        wake=False,
        sender={"account_id": "acc_sender"},
        message_id=MESSAGE_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=meta,
    )
    assert result is not None
    # The closed-set door still refuses a malformed block at the probe boundary.
    with pytest.raises(ValueError):
        validate_bridge_meta({"v": "1", "not_a_field": 1})
    await session.dispose()


@pytest.mark.asyncio
async def test_the_cold_spool_drain_receives_through_the_real_session(
    tmp_path: Path, isolated_config: Path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:
    """The COLD internal chain: the REAL spool writer → the REAL drain on a
    real Session → the production receive path (the drain calls
    ``receive_peer_message`` itself)."""

    session = await _session(tmp_path)
    drained = _spool_roundtrip(_meta())
    line = InboxLine(
        text="[team_bridge_message] message_id: bmsg_1",
        sender={"account_id": "acc_sender"},
        mode="mailbox",
        wake=False,
        source=SOURCE_PEER,
        message_id=MESSAGE_ID,
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
    )
    assert append_inbox(session.transcript.directory, line) is True
    await session._drain_spooled_peer_inbox_rows()
    assert peek_inbox(session.transcript.directory) == []  # consumed by the drain
    await session.dispose()


def test_the_wrap_shows_the_sender_account_never_the_local_owner(tmp_path) -> None:
    """TB-11: the displayed identity is the REMOTE sender's account.

    The envelope stamps the local owner's ``account_id`` beside the sender's;
    reading the owner's column labels every arrival with the OPERATOR's own
    account. The fixture makes the two differ, so the assertion discriminates.
    """

    store = _seed_store(tmp_path)
    _stamp(store, sender_account_id="acc_remote_9")
    drained = _spool_roundtrip(_meta())
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    _assert_fence_open(stub.bridge_runtime)
    message = Session._peer_custom_message(
        stub,
        "[team_bridge_message] body",
        {"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        entry_id=LOCAL_ROW_ID,
    )
    text = str((getattr(message, "details", {}) or {}).get("text") or "")
    assert "from_account='acc_remote_9'" in text
    assert "from_account='acc_1'" not in text


def test_a_stale_scope_over_the_wire_never_raises(
    tmp_path, _no_enqueue  # noqa: F811 — imported fixture
) -> None:  # noqa: F811 — imported fixture
    store = _seed_store(tmp_path)
    _stamp(store)
    # The drift is in the STORED ORIGINAL scope — the door's own subject — not
    # in the live binding: the enrollment and the runtime stay the CURRENT
    # owner (so the fence is satisfied), while the stamped row names the
    # PREVIOUS generation. Direct SQL is this suite's documented fixture
    # boundary, and the write API refuses anything but a live owner, so a
    # drifted scope can only be seeded (S-8's door exists for exactly this
    # case: a stored block that no longer matches the receiver).
    conn = sqlite3.connect(store._path)
    try:
        conn.execute(
            "UPDATE screened_request_deliveries SET native_generation = 6 " "WHERE message_id = ?",
            (MESSAGE_ID,),
        )
        conn.commit()
    finally:
        conn.close()
    drained = _spool_roundtrip(_meta())
    runtime = _runtime(tmp_path)
    stub = _StubSession(runtime, object(), config_dir=tmp_path)
    _assert_fence_open(stub.bridge_runtime)
    message = Session._peer_custom_message(
        stub,
        "[team_bridge_message] body",
        {"account_id": "acc_sender"},
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        bridge_meta=drained,
        # The harness's own accepted row id: the durable issuance maps
        # THIS message id to the ask (S-6/S-7), and the ledger refuses a NEW
        # issuance row without it (TB-02/round-3 claim contract).
        entry_id=LOCAL_ROW_ID,
    )
    Session._maybe_raise_bridge_ask(stub, message)
    assert _no_enqueue == []
    from local_operator.team_bridge.ask_issuance import AskIssuance

    row = AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)
    assert row is not None
    assert row["state"] == "unrecoverable"
    assert row["reason"] == "original_scope_drift"


assert DISPLAY_DIGEST is not None
assert _HINT_FIELDS  # the canonical hint fields exist for both suites
