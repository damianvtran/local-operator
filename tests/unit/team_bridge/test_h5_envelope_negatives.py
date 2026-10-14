"""H5 F3 envelope negatives — the REAL native store + the REAL validator.

Every case here runs against a REAL ``Store`` (schema created by the store's
own ``_init``; rows seeded through direct SQL so the WRITE path's own
validations are exercised), the REAL ``write_screened_request_delivery`` /
``screened_request_delivery`` readers, the REAL ``scope_matches_runtime`` and
the REAL ``AskIssuance`` ledger. Only the wire edges — the enqueue call and
(one arm) the owner derivation — are substituted at their module attributes.

Pinned refusals (Root mandate, cases 1–5; SEC S-8/S-9/S-10):
- a stamped OLD endpoint/credential/designation/native-generation refuses with
  ZERO enqueue and a visible unrecoverable record (no rebind, no backfill);
- a forged ``meta.type==request`` on a stored NOTE refuses — the STORED kind
  decides;
- a display-digest mismatch refuses (hints are usable only after it matches);
- a missing/legacy envelope is visible-unrecoverable, never an arrival mint;
- an expired request refuses; an unreadable/ambiguous ask lookup HOLDS, while
  a found-exact ask terminalizes the issuance WITHOUT a second enqueue.
"""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from local_operator.team_bridge import asks_bridge as asks_bridge_mod
from local_operator.team_bridge import store as bridge_store
# Bound at IMPORT time, before the autouse ``_no_enqueue`` fixture swaps the
# module attribute for a recorder: the two tests that SEED the real ask log
# must write through the real builder, and a function-local ``from ... import``
# would resolve to the recorder instead (the seed then lands in the call list,
# the log stays empty, and the product under test correctly mints a new ask).
from local_operator.team_bridge.asks_bridge import (
    enqueue_bridge_request_ask as _real_enqueue_bridge_request_ask,
)
from local_operator.harness.message_types import display_context_digest
from local_operator.team_bridge.ask_issuance import (
    STATE_HELD,
    STATE_RAISED,
    STATE_UNRECOVERABLE,
    AskIssuance,
)
from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.session.session import Session

BODY_HASH = "9f2a" * 16
TEMPLATE = "aa11" * 16
ENVELOPE_DIGEST = "bb22" * 16
ADMISSION = "rev-1"
POLICY_EPOCH = 3
#: The canonical five hint fields — the SAME set the producer hashes. The
#: stored digest is RECOMPUTED from these on the receive side (a transported
#: digest string alone authorizes nothing, S-9), so the tests must be able to
#: change a FIELD and keep the string.
_HINT_FIELDS = {
    "sender_name": "dana@acme.example",
    "organization": "Acme",
    "sender_label": "",
    "thread_id": "t-1",
    "excerpt": "please approve before 1700",
}
DISPLAY_DIGEST = display_context_digest(_HINT_FIELDS)
assert DISPLAY_DIGEST is not None
MESSAGE_ID = "bmsg_1"
DREF = "dref_" + "ab" * 16
#: The harness's own accepted transcript row id (``peer-<32 lowercase hex>``);
#: the ledger refuses to commit a NEW issuance row without a trusted one.
LOCAL_ROW_ID = "peer-" + "ab" * 16


def _owner(**overrides) -> SimpleNamespace:
    values = {
        "canonical_base_url": "https://hub.example",
        "account_id": "acc_1",
        "tenant_id": "ten_1",
        "endpoint_id": "ep_1",
        "credential_id": 7,
        "enrollment_generation": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _seed_store(tmp_path, *, kind: str = "request", expires: str = "") -> Store:
    """A REAL store whose schema came from ``_init``; rows seeded by SQL.

    Direct SQL is the fixture boundary here — every READ the validator runs is
    the store's own, so a column/semantics drift cannot hide behind a fake.
    """

    store = Store(tmp_path)
    store.active_owner()  # force schema creation + stamp via the real _init
    owner = _owner()
    assert store._path.exists()
    conn = sqlite3.connect(store._path)
    try:
        conn.row_factory = sqlite3.Row
        key = bridge_store._owner_key(owner)
        conn.execute(
            "INSERT INTO enrollments (owner_key, credential_id, account_id, tenant_id, "
            "endpoint_id, canonical_base_url, enrollment_generation, consent_version, "
            "consent_ref, designated_session, state) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                owner.credential_id,
                owner.account_id,
                owner.tenant_id,
                owner.endpoint_id,
                owner.canonical_base_url,
                owner.enrollment_generation,
                "consent-v1",
                "consent-ref-1",
                "s1",
                "committed",
            ),
        )
        conn.execute(
            "INSERT INTO claims (owner_key, message_id, session_id, token, generation, "
            "expires_at, state) VALUES (?,?,?,?,?,?,?)",
            (key, MESSAGE_ID, "s1", "tok-1", 7, time.time() + 300, "held"),
        )
        conn.execute(
            "INSERT INTO screening_policy (owner_key, version, template_sha256, "
            "default_model, revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (key, "v1", TEMPLATE, "model-a", ADMISSION, POLICY_EPOCH, time.time()),
        )
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                MESSAGE_ID,
                BODY_HASH,
                TEMPLATE,
                "prompt-v1",
                "model-a",
                "allow",
                json.dumps(["spam"]),
                ENVELOPE_DIGEST,
                ADMISSION,
                POLICY_EPOCH,
                7,
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return store


def _seed_note_envelope(tmp_path) -> None:
    """A stored envelope whose kind is NOT request (the forged-locator arm)."""

    store = _seed_store(tmp_path)
    conn = sqlite3.connect(store._path)
    try:
        row = conn.execute("SELECT * FROM screened_request_deliveries").fetchone()
        assert row is None  # nothing stamped yet — seed the hostile row directly
        conn.execute(
            "INSERT INTO screened_request_deliveries (delivery_ref, owner_key, message_id, "
            "native_generation, scope_version, base_url, account_id, tenant_id, endpoint_id, "
            "credential_id, designated_session_id, consent_version, consent_ref, thread_id, "
            "message_kind, request_state, request_expires_at, request_ref, sender_account_id, "
            "sender_endpoint_id, recipient_account_id, recipient_endpoint_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories_digest, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, claim_hash, claim_session_id, "
            "claim_generation, claim_expires_at, display_digest, fingerprint, phase, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                DREF,
                bridge_store._owner_key(_owner()),
                MESSAGE_ID,
                7,
                "v1",
                "https://hub.example",
                "acc_1",
                "ten_1",
                "ep_1",
                7,
                "s1",
                "consent-v1",
                "consent-ref-1",
                "t-1",
                "note",  # THE HOSTILE KIND — the locator will claim "request"
                "pending",
                "",
                "",
                "acc_1",
                "ep_1",
                "acc_2",
                "ep_2",
                BODY_HASH,
                TEMPLATE,
                "prompt-v1",
                "model-a",
                "allow",
                "dd44" * 16,
                ENVELOPE_DIGEST,
                ADMISSION,
                POLICY_EPOCH,
                7,
                "ee55" * 16,
                "s1",
                7,
                time.time() + 300,
                DISPLAY_DIGEST,
                "ff66" * 16,
                "handoff_started",
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _stamp(store: Store, *, advance_phase: bool = True, sender_account_id: str = "acc_1") -> dict:
    """The REAL write path (+ the confirmed-handoff phase), for the seeds.

    ``sender_account_id`` is the REMOTE sender's account — distinct from the
    local owner's ``account_id`` — so a case can prove the displayed identity
    comes from the sender and never from the owner (TB-11).
    """

    envelope = store.write_screened_request_delivery(
        _owner(),
        delivery_ref=DREF,
        message_id=MESSAGE_ID,
        designated_session="s1",
        display_digest=DISPLAY_DIGEST,
        body_sha256=BODY_HASH,
        claim_token="tok-1",
        request={
            "thread_id": "t-1",
            "message_kind": "request",
            "request_state": "pending",
            "request_expires_at": "",
            "request_ref": "",
            "sender_account_id": sender_account_id,
            "sender_endpoint_id": "ep_1",
            "recipient_account_id": "acc_2",
            "recipient_endpoint_id": "ep_2",
        },
    )
    if advance_phase:
        store.mark_delivery_phase(_owner(), str(envelope["delivery_ref"]), "handoff_started")
    return envelope


class _StubSession(Session):
    """The validator methods, WITHOUT constructing a real Session."""

    session_id = "s1"

    def __init__(self, runtime, queue, *, config_dir):
        self.bridge_runtime = runtime
        self._test_queue = queue
        self._config_dir = config_dir

    def ask_queue(self):
        return self._test_queue


def _runtime(config_dir, owner=None) -> SimpleNamespace:
    return SimpleNamespace(
        owner=owner or _owner(),
        config_dir=config_dir,
        tenant_id="ten_1",
        session_id="s1",
    )


@pytest.fixture(autouse=True)
def _no_enqueue(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _recorder(queue, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "details": {"ask_id": "a-1"}}

    monkeypatch.setattr(asks_bridge_mod, "enqueue_bridge_request_ask", _recorder)
    return calls


def _meta(**overrides) -> dict:
    meta = {
        "v": "1",
        "delivery_ref": DREF,
        "message_id": MESSAGE_ID,
        "type": "request",
        "request_state": "pending",
        "body_sha256": BODY_HASH,
        "display_digest": DISPLAY_DIGEST,
        **_HINT_FIELDS,
    }
    meta.update(overrides)
    return meta


def test_the_real_store_write_is_write_once_and_idempotent(tmp_path) -> None:
    store = _seed_store(tmp_path)
    first = _stamp(store, advance_phase=False)
    assert first["delivery_ref"] == DREF
    assert first["phase"] == "prepared"
    assert first["claim_hash"] != "tok-1"  # the bearer token is NOT stored
    second = _stamp(store, advance_phase=False)
    assert second["delivery_ref"] == first["delivery_ref"]  # idempotent, no second row
    store.mark_delivery_phase(_owner(), DREF, "handoff_started")
    store.mark_delivery_phase(_owner(), DREF, "delivered")
    with pytest.raises(StoreRefused):
        store.mark_delivery_phase(_owner(), DREF, "handoff_started")  # monotonic
    # A retry that no longer HOLDS the token it acquired is not this
    # handoff's identity: refused rather than adopted (S-8).
    with pytest.raises(StoreRefused) as refused:
        store.write_screened_request_delivery(
            _owner(),
            delivery_ref=DREF,
            message_id=MESSAGE_ID,
            designated_session="s1",
            display_digest=DISPLAY_DIGEST,
            body_sha256=BODY_HASH,
            claim_token="tok-successor",
            request={
                "thread_id": "t-1",
                "message_kind": "request",
                "request_state": "pending",
            },
        )
    assert refused.value.code == "claim_not_mine"


@pytest.mark.parametrize(
    "mutation, expected",
    [
        # An endpoint move changes the namespace key itself: the envelope is
        # not THIS owner's to read (no cross-namespace adoption), visibly.
        ({"endpoint_id": "ep_B"}, "no_original_envelope"),
        ({"credential_id": 8}, "original_scope_drift"),
        ({"enrollment_generation": 8}, "original_scope_drift"),
    ],
)
def test_rebound_owner_scope_refuses_with_zero_enqueue(
    tmp_path, mutation, expected, _no_enqueue
) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    bound = _owner(**mutation)
    stub = _StubSession(_runtime(tmp_path, bound), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, bound, _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    # The refusal lands under the mutated owner's OWN namespace (endpoint moves
    # the key; credential/generation stay inside it) — never a rebind.
    row = AskIssuance(tmp_path).read(bound, MESSAGE_ID)
    assert row["state"] == STATE_UNRECOVERABLE
    assert row["reason"] == expected


def test_rebound_designation_refuses_with_zero_enqueue(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    # The receiver is a DIFFERENT session than the stamped designation: the
    # envelope's exact original scope cannot match it. The stub mirrors the
    # receiver's own session id, which is what scope_matches_runtime reads.
    runtime = _runtime(tmp_path)
    runtime.session_id = "s2"
    stub = _StubSession(runtime, object(), config_dir=tmp_path)
    stub.session_id = "s2"
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    assert AskIssuance(tmp_path).read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE


def test_a_prepared_or_fenced_envelope_never_raises_an_ask(tmp_path, _no_enqueue) -> None:
    # PREPARED: the stamp exists but no CONFIRMED handoff ever advanced it.
    store = _seed_store(tmp_path)
    _stamp(store, advance_phase=False)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "handoff_not_authorized"
    # FENCED: terminal for the handoff — an opaque-ref replay cannot reopen it.
    store2 = _seed_store(tmp_path / "fenced")
    _stamp(store2)
    store2.mark_delivery_phase(_owner(), DREF, "fenced")
    stub2 = _StubSession(_runtime(tmp_path / "fenced"), object(), config_dir=tmp_path / "fenced")
    Session._authorize_bridge_envelope(
        stub2, stub2.bridge_runtime, store2, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger2 = AskIssuance(tmp_path / "fenced")
    assert ledger2.read(_owner(), MESSAGE_ID)["reason"] == "handoff_not_authorized"


def test_a_successor_claim_refuses_the_stored_binding(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    # A DIFFERENT claim now holds the message: the stored original claim
    # identity is no longer the current one — refuse, never adopt.
    conn = sqlite3.connect(store._path)
    try:
        conn.execute(
            "UPDATE claims SET token = ?, session_id = ? WHERE message_id = ?",
            ("tok-successor", "s9", MESSAGE_ID),
        )
        conn.commit()
    finally:
        conn.close()
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "claim_successor"


def test_two_exact_asks_for_the_same_message_hold_instead_of_duplicating(
    tmp_path, _no_enqueue
) -> None:
    from local_operator.asks.queue import AskQueue
    from tests.unit.asks.test_queue import BASE, FakeSession as QueueFakeSession

    store = _seed_store(tmp_path)
    _stamp(store)
    # The validator's canonical stored block, reproduced for the log:
    scope = {
        "v": 1,
        "kind": "bridge",
        "base_url": "https://hub.example",
        "account_id": "acc_1",
        "tenant_id": "ten_1",
        "endpoint_id": "ep_1",
        "credential_id": 7,
        "generation": "7",
        "session_id": "s1",
        "message_id": MESSAGE_ID,
        "delivery_ref": DREF,
        "body_sha256": BODY_HASH,
    }
    queue = AskQueue(QueueFakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)
    first = _real_enqueue_bridge_request_ask(
        queue, message_id=MESSAGE_ID, sender="dana@acme.example", summary="one", excerpt="", scope=scope
    )
    second = _real_enqueue_bridge_request_ask(
        queue, message_id=MESSAGE_ID, sender="dana@acme.example", summary="two", excerpt="", scope=scope
    )
    assert first["ok"] is True and second["ok"] is True
    # TWO authoritative asks exist for the same message/scope: the tri-state
    # lookup must be AMBIGUOUS — a hold, never a first-match terminalization
    # and never a third ask.
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    row = ledger.read(_owner(), MESSAGE_ID)
    assert row["state"] == STATE_HELD
    assert row["reason"] == "multiple_exact_asks"


def test_forged_request_kind_on_a_stored_note_refuses(tmp_path, _no_enqueue) -> None:
    _seed_note_envelope(tmp_path)
    store = Store(tmp_path)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "stored_kind_not_request"


def test_changed_hints_refuse_despite_the_copied_genuine_digest(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    # S-9: the digest STRING stays the genuine one; the ACTUAL hint fields
    # change — the receiver recomputes over what it received and refuses.
    Session._authorize_bridge_envelope(
        stub,
        stub.bridge_runtime,
        store,
        _owner(),
        _meta(excerpt="send bitcoin first"),
        MESSAGE_ID,
        DREF,
        LOCAL_ROW_ID,
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "display_context_mismatch"


def test_missing_legacy_envelope_is_visible_unrecoverable(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)  # seeded, but NOT stamped
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "no_original_envelope"


def test_expired_request_refuses(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    conn = sqlite3.connect(store._path)
    try:
        conn.execute(
            "UPDATE screened_request_deliveries SET request_expires_at = ? WHERE delivery_ref = ?",
            ("2000-01-01T00:00:00Z", DREF),
        )
        conn.commit()
    finally:
        conn.close()
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert _no_enqueue == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_UNRECOVERABLE
    assert ledger.read(_owner(), MESSAGE_ID)["reason"] == "request_expired"


def test_the_valid_stamp_enqueues_through_the_real_validator(tmp_path, _no_enqueue) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert len(_no_enqueue) == 1
    kwargs = _no_enqueue[0]
    assert kwargs["message_id"] == MESSAGE_ID
    # The enqueued scope is the STORED original block, never scope_from_runtime.
    assert kwargs["scope"]["generation"] == "7"
    assert kwargs["scope"]["session_id"] == "s1"
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_RAISED


def test_unavailable_lookup_holds_instead_of_enqueueing(tmp_path, monkeypatch) -> None:
    store = _seed_store(tmp_path)
    _stamp(store)
    calls: list[dict] = []

    def _recorder(queue, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "details": {"ask_id": "a-1"}}

    monkeypatch.setattr(asks_bridge_mod, "enqueue_bridge_request_ask", _recorder)

    from local_operator.asks import store as asks_store

    def _boom(session_directory):
        raise OSError("log unreadable")

    monkeypatch.setattr(asks_store, "read_events", _boom)
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert calls == []
    ledger = AskIssuance(tmp_path)
    assert ledger.read(_owner(), MESSAGE_ID)["state"] == STATE_HELD


def test_found_exact_ask_terminalizes_without_a_second_enqueue(
    tmp_path, monkeypatch
) -> None:
    from local_operator.asks.queue import AskQueue
    from local_operator.team_bridge.authority import scope_from_owner
    from tests.unit.asks.test_queue import BASE, FakeSession as QueueFakeSession

    store = _seed_store(tmp_path)
    _stamp(store)
    # Seed the REAL ask log with the SAME message id + the SAME stored scope —
    # then patch the enqueue recorder to prove it is NOT called again.
    queue = AskQueue(QueueFakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)
    outcome = _real_enqueue_bridge_request_ask(
        queue,
        message_id=MESSAGE_ID,
        sender="dana@acme.example",
        summary="t-1",
        excerpt="",
        # The SAME block the validator derives from the stored envelope: the
        # owner/session fields PLUS the keyed identity (message id, opaque
        # ref, screened body hash — ``_authorize_bridge_envelope`` builds the
        # scope with them, and ``_bridge_ask_lookup`` compares whole-block
        # fingerprints). A seed carrying only the owner fields is a DIFFERENT
        # scope, which the product correctly reads as ambiguous and holds.
        scope={
            **scope_from_owner(_owner(), "s1"),
            "message_id": MESSAGE_ID,
            "delivery_ref": DREF,
            "body_sha256": BODY_HASH,
        },
    )
    assert outcome["ok"] is True
    calls: list[dict] = []

    def _recorder(queue_, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "details": {"ask_id": "a-2"}}

    monkeypatch.setattr(asks_bridge_mod, "enqueue_bridge_request_ask", _recorder)
    from local_operator.asks import store as asks_store

    real_read = asks_store.read_events
    stub = _StubSession(_runtime(tmp_path), object(), config_dir=tmp_path)
    # The real fold reads the REAL log — no patch needed for the found arm.
    assert real_read is not None
    Session._authorize_bridge_envelope(
        stub, stub.bridge_runtime, store, _owner(), _meta(), MESSAGE_ID, DREF, LOCAL_ROW_ID
    )
    assert calls == []  # found-exact ⇒ terminalized, never a second ask
    ledger = AskIssuance(tmp_path)
    row = ledger.read(_owner(), MESSAGE_ID)
    assert row["state"] == STATE_RAISED
    assert row["ask_id"] == str(outcome["details"]["ask_id"])
