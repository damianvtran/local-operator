"""Synthetic observer proof; no claim to reproduce a production failure cause."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from types import SimpleNamespace

import pytest

from local_operator.team_bridge.client import BridgeRefusal
from local_operator.team_bridge.connector import conversation, loop
from local_operator.team_bridge.store import StoreRefused
from tests.unit.team_bridge.test_conversation_delivery import (
    FakeServer,
    FakeWire,
    _enroll,
    _hold_rows,
    _owner,
    _ref,
    _reservation_rows,
    _wire_rig,
)

CANARY = "FAKE-PAYLOAD FAKE-TOKEN https://invalid.example/?secret=FAKE"


@pytest.mark.parametrize("phase", ["hub_admission", "routing", "reservation"])
def test_unexpected_failure_retains_hold_and_observes_only_structure(
    tmp_path, monkeypatch, caplog, phase
):
    owner = _owner("a", 1)
    root = tmp_path / "a"
    root.mkdir()
    store = _enroll(root, owner, "sess_a")
    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _ = _wire_rig(monkeypatch, server, {"a": (owner, root)})
    ref = _ref(server)
    store.persist_hold(owner, ref.encode(), "conversation_failed", retry_at=0)

    def fail(*args, **kwargs):
        raise RuntimeError(CANARY)

    async def fail_async(*args, **kwargs):
        fail()

    if phase == "hub_admission":
        monkeypatch.setattr(FakeWire, "conversation_admit", fail_async)
    elif phase == "routing":
        monkeypatch.setattr(conversation, "plan_route", fail)
    else:
        monkeypatch.setattr(conversation, "_record_body_reservation", fail)
    caplog.set_level(logging.WARNING, logger=conversation.__name__)
    out = asyncio.run(
        loop.retry_due_holds(
            owner,
            config_dir=root,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
    )
    assert out["outcomes"][0]["action"] == "hold"
    holds = _hold_rows(root)
    assert len(holds) == 1 and holds[0]["retry_at"] > 0
    assert holds[0]["hold_code"] == "conversation_failed"
    assert (
        asyncio.run(
            loop.retry_due_holds(
                owner,
                config_dir=root,
                access=SimpleNamespace(credential_id=owner.credential_id),
                auth_store=None,
            )
        )["outcomes"]
        == []
    )
    assert _reservation_rows(root) == []
    assert transcripts["sess_a"] == []
    records = [r for r in caplog.records if "code=conversation_failed" in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert f"phase={phase}" in record.getMessage(), "diagnostic phase missing"
    assert "failure_class=runtime_error" in record.getMessage()
    assert "http_status=unknown" in record.getMessage()
    assert (
        f"row_ref={hashlib.sha256(ref.encode().encode('utf-8')).hexdigest()}" in record.getMessage()
    )
    assert CANARY not in caplog.text, "private exception leaked"
    assert record.exc_info is None and record.stack_info is None


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, "400"),
        (401, "401"),
        (403, "403"),
        (404, "404"),
        (408, "408"),
        (409, "409"),
        (422, "422"),
        (429, "429"),
        (500, "500"),
        (502, "502"),
        (503, "503"),
        (504, "504"),
        (True, "unknown"),
        (False, "unknown"),
        ("503", "unknown"),
        (CANARY, "unknown"),
        (599, "unknown"),
        (None, "unknown"),
    ],
)
def test_refusal_observation_is_closed(caplog, status, expected):
    failure = BridgeRefusal(CANARY, CANARY, http_status=status)
    caplog.set_level(logging.WARNING, logger=conversation.__name__)
    conversation._log_conversation_failure(failure, CANARY, CANARY)
    record = caplog.records[-1]
    assert record.getMessage() == (
        "bridge: conversation delivery held code=conversation_failed "
        f"phase=unknown failure_class=bridge_refusal http_status={expected} "
        f"row_ref={hashlib.sha256(CANARY.encode()).hexdigest()}"
    )
    assert CANARY not in caplog.text
    assert record.exc_info is None and record.stack_info is None


def test_unknown_exception_never_reads_dynamic_fields(caplog):
    def forbidden(*args):
        raise AssertionError("exception text/property must not be inspected")

    hostile = type(
        CANARY,
        (RuntimeError,),
        {
            "__str__": forbidden,
            "__repr__": forbidden,
            "http_status": property(forbidden),
            "code": property(forbidden),
        },
    )
    caplog.set_level(logging.WARNING, logger=conversation.__name__)
    conversation._log_conversation_failure(hostile(), "routing", CANARY)
    assert "failure_class=unknown http_status=unknown" in caplog.text
    assert CANARY not in caplog.text
    assert caplog.records[-1].exc_info is None


@pytest.mark.parametrize("metaclass_kind", ["unhashable", "raising_hash", "raising_equality"])
def test_hostile_exception_metaclass_preserves_retry_hold(
    tmp_path, monkeypatch, caplog, metaclass_kind
):
    accesses = []

    def forbidden(name):
        def access(*args, **kwargs):
            accesses.append(name)
            raise AssertionError("diagnostics must not invoke exception hooks")

        return access

    # A colliding hash makes dictionary lookup reach equality too. Identity-only
    # classification must invoke neither hook, including on an unhashable type.
    class HostileMeta(type):
        __hash__ = (
            None
            if metaclass_kind == "unhashable"
            else (
                forbidden("hash")
                if metaclass_kind == "raising_hash"
                else lambda cls: accesses.append("hash") or hash(RuntimeError)
            )
        )
        __eq__ = forbidden("equality")

    class HostileFailure(RuntimeError, metaclass=HostileMeta):
        __str__ = forbidden("str")
        __repr__ = forbidden("repr")
        http_status = property(forbidden("http_status"))
        code = property(forbidden("code"))

    owner = _owner("a", 1)
    root = tmp_path / "a"
    root.mkdir()
    store = _enroll(root, owner, "sess_a")
    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _ = _wire_rig(monkeypatch, server, {"a": (owner, root)})
    ref = _ref(server)
    store.persist_hold(owner, ref.encode(), "conversation_failed", retry_at=0)

    def fail(*args, **kwargs):
        raise HostileFailure(CANARY)

    monkeypatch.setattr(conversation, "plan_route", fail)
    caplog.set_level(logging.WARNING, logger=conversation.__name__)
    outcome = asyncio.run(
        loop.retry_due_holds(
            owner,
            config_dir=root,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
    )
    assert outcome["outcomes"][0]["action"] == "hold"
    holds = _hold_rows(root)
    assert len(holds) == 1 and holds[0]["hold_code"] == "conversation_failed"
    assert holds[0]["retry_at"] > 0
    assert _reservation_rows(root) == [] and transcripts["sess_a"] == []
    assert accesses == []
    record = caplog.records[-1]
    assert record.args == (
        "routing",
        "unknown",
        "unknown",
        hashlib.sha256(ref.encode().encode("utf-8")).hexdigest(),
    )
    assert record.exc_info is None and record.stack_info is None
    assert CANARY not in caplog.text and ref.encode() not in caplog.text


@pytest.mark.parametrize("kind", ["refusal", "store", "cancel"])
def test_failure_protocol_is_not_reclassified(tmp_path, monkeypatch, caplog, kind):
    owner = _owner("a", 1)
    root = tmp_path / "a"
    root.mkdir()
    store = _enroll(root, owner, "sess_a")
    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    transcripts, _ = _wire_rig(monkeypatch, server, {"a": (owner, root)})
    ref = _ref(server)
    store.persist_hold(owner, ref.encode(), "conversation_failed", retry_at=0)
    failure = {
        "refusal": BridgeRefusal(CANARY, CANARY, http_status=503),
        "store": StoreRefused("state_unreadable", CANARY),
        "cancel": asyncio.CancelledError(CANARY),
    }[kind]

    async def fail(*args, **kwargs):
        raise failure

    caplog.set_level(logging.WARNING, logger=conversation.__name__)
    with monkeypatch.context() as patch:
        patch.setattr(FakeWire, "conversation_admit", fail)
        pending = loop.retry_due_holds(
            owner,
            config_dir=root,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
        if kind == "cancel":
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(pending)
        else:
            assert asyncio.run(pending)["outcomes"][0]["action"] == "hold"
    hold = _hold_rows(root)[0]
    assert hold["hold_code"] == ("state_unreadable" if kind == "store" else "conversation_failed")
    assert _reservation_rows(root) == [] and transcripts["sess_a"] == []
    assert CANARY not in caplog.text
    if kind == "refusal":
        assert "phase=hub_admission failure_class=bridge_refusal http_status=503" in caplog.text
    else:
        assert "code=conversation_failed" not in caplog.text
    if kind == "cancel":
        # Cancellation did not become terminal or keep a claim: the unchanged
        # due row replays through the real loop once the synthetic fault ends.
        result = asyncio.run(
            loop.retry_due_holds(
                owner,
                config_dir=root,
                access=SimpleNamespace(credential_id=owner.credential_id),
                auth_store=None,
            )
        )
        assert result["outcomes"][0]["action"] == "delivered"
        assert _hold_rows(root) == []
        assert len(_reservation_rows(root)) == 1 and len(transcripts["sess_a"]) == 1


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("drop_phase", "diagnostic phase missing"),
        ("leak_exception", "private exception leaked"),
    ],
)
def test_diagnostic_contract_detects_observer_mutants(
    tmp_path, monkeypatch, caplog, mutation, expected
):
    original = conversation._log_conversation_failure

    def mutant(failure, phase, reference):
        original(failure, "unknown" if mutation == "drop_phase" else phase, reference)
        if mutation == "leak_exception":
            conversation.logger.warning("UNSAFE %s", failure)

    # Mutate only a fixture seam in memory; never mutate repository source.
    monkeypatch.setattr(conversation, "_log_conversation_failure", mutant)
    with pytest.raises(AssertionError, match=expected):
        test_unexpected_failure_retains_hold_and_observes_only_structure(
            tmp_path, monkeypatch, caplog, "routing"
        )
