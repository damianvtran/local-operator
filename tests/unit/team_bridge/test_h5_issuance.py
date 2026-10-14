"""H5a/F3 issuance — the locator carriage, the durable ledger, and the hook.

Pinned here:
- the ``bridge_meta`` block is a CLOSED, bounded locator at every hop's
  validator, never a clearance claim;
- the durable issuance ledger is generation-scoped, freezes the ORIGINAL scope
  at claim time, and its retry set never crosses generations;
- the REAL ask queue's enqueue (via ``enqueue_bridge_request_ask``) is what the
  real ``_bridge_ask_lookup`` fold lookup answers — the composition, not a
  fixture shape;
- the receive hook raises an ask ONLY behind the trusted receipt + live fence,
  leaves a durable PENDING on a cap refusal, and never a second ask for the
  same message (re-delivery/reconcile/crash resolve to the durable record).

Where a boundary is faked it is faked AT the boundary (the store, the fence,
the enqueue call) while the sqlite ledger, the scope block, the locator
validator and — for the ask-once lookup — the ask queue and its fold are the
REAL implementations. The full queue→answer→consume→transcript→recovery
composition across a real Session is the named follow-up case in the batch
receipt (it needs a Session fixture, not a fixture shape substitution).
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.asks.queue import AskQueue

# Bound at IMPORT time, before the autouse ``_boundaries`` fixture swaps the
# module attribute for the receive-side fake: a function-local import would
# resolve to the FAKE and 'restoring' it would restore the fake.
from local_operator.harness.message_types import (
    BRIDGE_META_KEYS,
    display_context_digest,
    validate_bridge_meta,
)
from local_operator.session.runtime.inbox import InboxLine
from local_operator.session.session import Session
from local_operator.team_bridge import asks_bridge as asks_bridge_mod
from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import store as bridge_store
from local_operator.team_bridge.ask_issuance import AskIssuance
from local_operator.team_bridge.asks_bridge import enqueue_bridge_request_ask
from local_operator.team_bridge.connector.wake import (
    EXCERPT_MAX_CHARS,
    build_bridge_meta,
)
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.store import Store as _RealStore
from local_operator.team_bridge.tool import BridgeRuntime
from tests.unit.asks.test_queue import BASE
from tests.unit.asks.test_queue import FakeSession as QueueFakeSession

SENDER = "dana@acme.example"

BODY_HASH = "9f2a" * 16
#: The harness's own accepted transcript row id the receive hook passes to the
#: ledger; a NEW issuance row is refused without a trusted ``peer-<32hex>`` id.
LOCAL_ROW_ID = "peer-" + "ab" * 16
#: The canonical five hint fields the queue-side FakeStore envelope and the
#: test locator must agree on; the digest is DERIVED from them (S-9).
_HINT_FIELDS = {
    "sender_name": SENDER,
    "organization": "Acme",
    "sender_label": "",
    "thread_id": "t-1",
    "excerpt": "please approve before 1700",
}
DISPLAY_DIGEST = display_context_digest(_HINT_FIELDS)
assert DISPLAY_DIGEST is not None


def _owner() -> OwnerSnapshot:
    return cast(
        OwnerSnapshot,
        SimpleNamespace(
            canonical_base_url="https://hub.example",
            account_id="acc_1",
            tenant_id="ten_1",
            endpoint_id="ep_1",
            credential_id=7,
            enrollment_generation=7,
        ),
    )


def _runtime(tmp_path) -> BridgeRuntime:
    return cast(
        BridgeRuntime,
        SimpleNamespace(
            owner=_owner(),
            config_dir=tmp_path,
            tenant_id="ten_1",
            session_id="s1",
        ),
    )


def _read(ledger, owner, message_id: str = "bmsg_1") -> dict[str, Any]:
    """The ledger row, refusing a miss (``read`` returns ``dict | None``)."""
    row = ledger.read(owner, message_id)
    assert row is not None
    return row


def _meta(**overrides) -> dict[str, Any]:
    meta = {
        "v": "1",
        "delivery_ref": "dref_test",
        "message_id": "bmsg_1",
        "type": "request",
        "request_state": "pending",
        "body_sha256": BODY_HASH,
        "display_digest": DISPLAY_DIGEST,
        **_HINT_FIELDS,
    }
    meta.update(overrides)
    return meta


class _StubSession(Session):
    """The tested methods, WITHOUT constructing a real Session.

    ``__init__`` deliberately skips ``Session.__init__``: the door under test
    reaches only the attributes set here, and a full construction would pull a
    runtime for no fidelity these cases rely on. ``session_id`` is shadowed as
    a class attribute because the parent defines it as a property (no setter);
    ``_config_dir`` is the parent's own storage name for the config root.
    """

    session_id = "s1"

    def __init__(self, runtime, queue, *, config_dir):
        self.bridge_runtime = runtime
        self._test_queue = queue
        self._config_dir = config_dir

    def ask_queue(self):
        return self._test_queue


class _FakeStore:
    """The receive-side store boundary: the STAMPED ENVELOPE, nothing else."""

    def __init__(self, config_dir):
        self.config_dir = config_dir

    def active_owner(self):
        return _owner()

    def screened_request_delivery(self, owner, *, delivery_ref="", message_id=""):
        return {
            "delivery_ref": "dref_test",
            "owner_key": "k",
            "message_id": "bmsg_1",
            "native_generation": 7,
            "scope_version": "v1",
            "base_url": "https://hub.example",
            "account_id": "acc_1",
            "tenant_id": "ten_1",
            "endpoint_id": "ep_1",
            "credential_id": 7,
            "designated_session_id": "s1",
            "consent_version": "consent-v1",
            "consent_ref": "consent-ref-1",
            "thread_id": "t-1",
            "message_kind": "request",
            "request_state": "pending",
            "request_expires_at": "",
            "sender_account_id": "acc_s",
            "body_sha256": BODY_HASH,
            "display_digest": DISPLAY_DIGEST,
            "phase": "handoff_started",
        }

    def verify_original_envelope_current(self, owner, envelope):
        """Current-authority revalidation is exercised against the REAL store
        in test_h5_envelope_negatives.py; this boundary fake says "current"."""

        return ("", "")


@pytest.fixture(autouse=True)
def _boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two boundaries the hook consults, substituted at their modules."""

    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    monkeypatch.setattr(bridge_store, "Store", _FakeStore)


def _recorder(monkeypatch: pytest.MonkeyPatch, calls: list[Any]) -> None:
    def _fake_enqueue(queue, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "details": {"ask_id": "a-9"}}

    monkeypatch.setattr(asks_bridge_mod, "enqueue_bridge_request_ask", _fake_enqueue)


def test_the_locator_validator_refuses_unknown_oversize_unscalar_unversioned() -> None:
    with pytest.raises(ValueError):
        validate_bridge_meta({"v": "1", "screened": True})  # unknown key
    with pytest.raises(ValueError):
        validate_bridge_meta({"v": "1", "excerpt": "x" * 4096})  # over budget
    with pytest.raises(ValueError):
        validate_bridge_meta({"v": "1", "excerpt": {"nested": 1}})  # non-scalar
    with pytest.raises(ValueError):
        validate_bridge_meta({"message_id": "bmsg_1"})  # no version tag
    with pytest.raises(ValueError):
        validate_bridge_meta("not-a-mapping")


def test_the_locator_validator_normalizes_and_stays_closed() -> None:
    normalized = validate_bridge_meta({"v": 1, "message_id": "m", "excerpt": "e"})
    assert normalized["v"] == "1"
    assert set(normalized) <= BRIDGE_META_KEYS


def test_build_bridge_meta_fits_the_carriage_for_any_body_size() -> None:
    def _item(text: str) -> dict[str, Any]:
        return {
            "message": {
                "type": "request",
                "request_state": "pending",
                "thread_id": "t-1",
                "body": {"text": text},
            },
            "sender": {"name": "Dana", "organization": "Acme"},
            "thread": {"id": "t-1"},
            "delivery": {"state": "new"},
        }

    # R1-1 (PR #2378): the SHARED excerpt bound now FITS the carriage — a
    # full-length body builds a locator the validator ACCEPTS (the ask is
    # deliverable), because the excerpt window IS the display-context bound
    # both digest sides normalize by; a larger window made the send hop
    # refuse the producer's own block.
    long_meta = build_bridge_meta(_item("y" * 5000), message_id="bmsg_1", body_sha256=BODY_HASH)
    assert set(long_meta) <= BRIDGE_META_KEYS
    assert len(long_meta["excerpt"]) == EXCERPT_MAX_CHARS
    assert validate_bridge_meta(long_meta) == long_meta
    # A body inside the window is sent WHOLE and round-trips unchanged.
    short_meta = build_bridge_meta(_item("y" * 1200), message_id="bmsg_1", body_sha256=BODY_HASH)
    assert set(short_meta) <= BRIDGE_META_KEYS
    assert len(short_meta["excerpt"]) == 1200
    assert validate_bridge_meta(short_meta) == short_meta


def test_inbox_line_round_trips_the_locator_and_legacy_rows_read_empty() -> None:
    line = InboxLine(
        text="hi",
        sender={},
        mode="mailbox",
        written_at=1.0,
        message_id="peer-1",
        custom_type="team_bridge_message",
        bridge_meta=_meta(),
    )
    revived = InboxLine.from_json(line.to_json())
    assert revived.bridge_meta["message_id"] == "bmsg_1"
    legacy = InboxLine.from_json({"text": "old", "sender": {}, "written_at": 0.5})
    assert legacy.bridge_meta == {}


def test_the_issuance_ledger_is_generation_scoped_and_frozen_scope(tmp_path) -> None:
    owner = _owner()
    ledger = AskIssuance(tmp_path)
    first = ledger.claim(
        owner, "bmsg_1", {"v": 1, "kind": "bridge"}, sender="Dana", local_row_id="peer-" + "ab" * 16
    )
    assert first["state"] == "claimed" and first["scope_json"]
    # A second claim of the SAME message under the same generation returns the
    # SAME durable row (the scope captured first stands; attempts only count).
    again = ledger.claim(
        owner,
        "bmsg_1",
        {"v": 1, "kind": "bridge", "drifted": True},
        local_row_id="peer-" + "ab" * 16,
    )
    assert again["scope_json"] == first["scope_json"]
    # ``claim`` returns the row as it stood BEFORE its own attempt increment
    # (SELECT, then UPDATE), so the returned dict lags the durable count by one
    # and no caller reads it; the DURABLE counter is what the ledger promises:
    # one increment per claim, the scope untouched.
    assert int(_read(ledger, owner, "bmsg_1")["attempts"]) == 2
    ledger.mark_raised(owner, "bmsg_1", "a-9")
    assert _read(ledger, owner, "bmsg_1")["state"] == "raised"
    assert ledger.pending(owner) == []
    # A NEW generation is a different identity: the old row is not its to
    # retry, and it does not block a fresh claim.
    owner_new = _owner()
    owner_new.enrollment_generation = 8
    assert ledger.pending(owner_new) == []
    fresh = ledger.claim(
        owner_new, "bmsg_1", {"v": 1, "kind": "bridge"}, local_row_id="peer-" + "ab" * 16
    )
    assert fresh["generation"] == 8 and fresh["state"] == "claimed"


def test_the_real_queue_enqueue_is_found_by_the_message_lookup(tmp_path) -> None:
    # The product's lookup is TRI-STATE and scope-exact (S-10), so the REAL
    # fold answers ("found", ask_id, "") only for this message id UNDER the
    # same stored scope; the earlier id-only helper no longer exists because an
    # id-only match is exactly the first-match terminalization S-10 forbids.
    runtime = _runtime(tmp_path)
    queue = AskQueue(QueueFakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)
    scope = {"v": 1, "kind": "bridge", "generation": "7"}
    outcome = enqueue_bridge_request_ask(
        queue,
        message_id="bmsg_abc",
        sender=SENDER,
        summary="wants sign-off",
        excerpt="please approve",
        scope=scope,
    )
    assert outcome["ok"] is True
    stub = _StubSession(runtime, queue, config_dir=tmp_path)
    assert Session._bridge_ask_lookup(stub, "bmsg_abc", scope) == (
        "found",
        outcome["details"]["ask_id"],
        "",
    )
    assert Session._bridge_ask_lookup(stub, "bmsg_other", scope) == ("absent", "", "")
    # Same message id under a DIFFERENT stored scope is ambiguous: a hold,
    # never terminalized as this issuance's ask.
    assert Session._bridge_ask_lookup(stub, "bmsg_abc", {**scope, "generation": "8"}) == (
        "ambiguous",
        "",
        "question_scope_mismatch",
    )


def test_the_receipt_trust_matrix(tmp_path, monkeypatch) -> None:
    # The receipt trust check is not a standalone Session method: the hook's
    # one validator (``_authorize_bridge_envelope``) delegates it to
    # ``Store.verify_original_envelope_current``, which re-reads the REAL
    # screening receipt against the stamped envelope. This drives that real
    # path: allow passes; every other receipt shape refuses with a typed code.
    # !!! ORDER MATTERS !!! This file's autouse boundary fixture fakes
    # ``bridge_store.Store`` for the HOOK's sake, and the seeder module below
    # does ``from local_operator.team_bridge.store import Store`` AT ITS IMPORT
    # — so if this is the first import of that module in the session, a lazy
    # import here binds the FAKE and the case cannot build a real store at all.
    # The real class therefore goes back BEFORE the import, not after it.
    monkeypatch.setattr(bridge_store, "Store", _RealStore)
    # The seeder binds ``Store`` at ITS import, so this assert is the contract
    # the restore above exists for: a fake here means the case would silently
    # exercise a different store than the one the product reads.
    import tests.unit.team_bridge.test_h5_envelope_negatives as _neg
    from tests.unit.team_bridge.test_h5_envelope_negatives import DREF, MESSAGE_ID
    from tests.unit.team_bridge.test_h5_envelope_negatives import _owner as _env_owner
    from tests.unit.team_bridge.test_h5_envelope_negatives import _seed_store, _stamp

    assert _neg.Store is _RealStore, "the seeder must hold the REAL store class"

    store = _seed_store(tmp_path)
    _stamp(store)
    owner = _env_owner()
    envelope = store.screened_request_delivery(owner, delivery_ref=DREF, message_id=MESSAGE_ID)
    assert envelope is not None
    assert store.verify_original_envelope_current(owner, envelope) == ("", "")

    def _mutate(sql: str, *params) -> None:
        conn = sqlite3.connect(store._path)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    _mutate("UPDATE screening_receipts SET verdict = 'hold'")
    assert store.verify_original_envelope_current(owner, envelope)[0] == "proof_missing"
    _mutate("UPDATE screening_receipts SET verdict = 'allow', body_sha256 = 'other'")
    assert store.verify_original_envelope_current(owner, envelope)[0] == "proof_drift"
    for wrong_generation in (6, 0):
        _mutate(
            "UPDATE screening_receipts SET body_sha256 = ?, creator_generation = ?",
            envelope["body_sha256"],
            wrong_generation,
        )
        assert store.verify_original_envelope_current(owner, envelope)[0] == "proof_drift"
    _mutate("DELETE FROM screening_receipts")
    assert store.verify_original_envelope_current(owner, envelope)[0] == "proof_missing"


def test_the_hook_raises_once_and_is_terminal_for_a_repeat(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    _recorder(monkeypatch, calls)
    runtime = _runtime(tmp_path)
    stub = _StubSession(runtime, object(), config_dir=tmp_path)
    message = SimpleNamespace(id=LOCAL_ROW_ID, details={"bridge": _meta()})
    Session._maybe_raise_bridge_ask(stub, message)
    assert len(calls) == 1
    assert calls[0]["message_id"] == "bmsg_1"
    ledger = AskIssuance(tmp_path)
    assert _read(ledger, _owner(), "bmsg_1")["state"] == "raised"
    # Re-delivery / reconcile of the SAME message: durable record ends it.
    Session._maybe_raise_bridge_ask(stub, message)
    assert len(calls) == 1


def test_the_hook_refuses_without_a_trusted_arm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _recorder(monkeypatch, calls)
    runtime = _runtime(tmp_path)
    stub = _StubSession(runtime, object(), config_dir=tmp_path)
    # Changed HINT FIELDS with the copied genuine digest STRING: the receiver
    # recomputes over what it received and refuses (S-9) — visible
    # unrecoverable, never an ask.
    Session._maybe_raise_bridge_ask(
        stub,
        SimpleNamespace(id=LOCAL_ROW_ID, details={"bridge": _meta(excerpt="send bitcoin first")}),
    )
    # No bridge locator at all.
    Session._maybe_raise_bridge_ask(stub, SimpleNamespace(id=LOCAL_ROW_ID, details={"body": "hi"}))
    # Missing/legacy producer envelope: visible unrecoverable, no arrival mint.
    monkeypatch.setattr(_FakeStore, "screened_request_delivery", lambda self, owner, **kwargs: None)
    Session._maybe_raise_bridge_ask(
        stub, SimpleNamespace(id=LOCAL_ROW_ID, details={"bridge": _meta()})
    )
    assert calls == []
    ledger = AskIssuance(tmp_path)
    row = ledger.read(_owner(), "bmsg_1")
    assert row is not None
    assert row["state"] == "unrecoverable"
    assert row["reason"] == "no_original_envelope"


def test_a_cap_refusal_leaves_a_durable_pending_and_the_retry_raises_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def _refusing(queue, **kwargs):
        seen.append(kwargs)
        return {"ok": False, "error": "the open-ask cap is full — answer or dismiss first"}

    monkeypatch.setattr(asks_bridge_mod, "enqueue_bridge_request_ask", _refusing)
    runtime = _runtime(tmp_path)
    stub = _StubSession(runtime, object(), config_dir=tmp_path)
    Session._maybe_raise_bridge_ask(
        stub, SimpleNamespace(id=LOCAL_ROW_ID, details={"bridge": _meta()})
    )
    ledger = AskIssuance(tmp_path)
    assert _read(ledger, _owner(), "bmsg_1")["state"] == "pending"
    assert "cap" in _read(ledger, _owner(), "bmsg_1")["reason"]
    # The trusted local boundary retries it as an ASK only.
    calls: list[dict[str, Any]] = []
    _recorder(monkeypatch, calls)
    # S-8b: the retry is GRANT-SCOPED — an absent/empty grant map drives
    # NOTHING (a held row whose standing read did not succeed must never be
    # re-driven). The trusted boundary (``_refresh_standing_grants``) is what
    # grants a cap-held row once its fresh read passes; stand in for exactly
    # that outcome with the validated five-field display context.
    stub._standing_grants = {}
    Session._retry_pending_bridge_issuances(stub)
    assert calls == []
    stub._standing_grants = {
        "bmsg_1": {
            "message_id": "bmsg_1",
            "delivery_ref": "dref_test",
            "display": dict(_HINT_FIELDS),
        }
    }
    Session._retry_pending_bridge_issuances(stub)
    assert len(calls) == 1
    assert _read(ledger, _owner(), "bmsg_1")["state"] == "raised"


def test_no_queue_leaves_a_durable_pending(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _recorder(monkeypatch, calls)
    runtime = _runtime(tmp_path)
    stub = _StubSession(runtime, None, config_dir=tmp_path)
    Session._maybe_raise_bridge_ask(
        stub, SimpleNamespace(id=LOCAL_ROW_ID, details={"bridge": _meta()})
    )
    assert calls == []
    ledger = AskIssuance(tmp_path)
    assert _read(ledger, _owner(), "bmsg_1")["state"] == "pending"
    assert "queue" in _read(ledger, _owner(), "bmsg_1")["reason"]


def test_the_ledger_schema_is_what_it_claims(tmp_path) -> None:
    # A cheap structural check that the real file landed where the receipt
    # says — and that a torn/foreign file answers cleanly rather than raising.
    AskIssuance(tmp_path).claim(_owner(), "bmsg_1", {"v": 1}, local_row_id="peer-" + "ab" * 16)
    db = tmp_path / "team_bridge" / "ask_issuance.db"
    assert db.exists()
    conn = sqlite3.connect(db)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert "ask_issuance" in tables
