"""H5 S-10 — the KEYED-IDEMPOTENT bridge enqueue (SOURCE; real queue + log).

The contract pinned here: the durable ask id derives from the preallocated
issuance identity (never caller/model/UI-chosen); same key + same fingerprint
returns the SAME ask id — including AFTER the ask was consumed — so a crash
retry or a live+cold duplicate cannot mint a second ask; a mismatched
fingerprint refuses; and the generic ``enqueue`` API is unchanged for every
other caller. The keyed path uses the store's own interprocess lock on the ask
log for its lookup+append critical section.
"""

from __future__ import annotations

import hashlib
import json

from local_operator.asks.queue import AskQueue
from local_operator.team_bridge.asks_bridge import bridge_request_question
from tests.unit.asks.test_queue import BASE
from tests.unit.asks.test_queue import FakeSession as QueueFakeSession

MESSAGE_ID = "bmsg_1"
SCOPE = {
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
    "delivery_ref": "dref_x",
    "body_sha256": "9f2a" * 16,
}
FP = hashlib.sha256(
    json.dumps(dict(SCOPE), separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()


def _queue(tmp_path) -> AskQueue:
    return AskQueue(QueueFakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)


def _question() -> dict:
    return bridge_request_question(
        message_id=MESSAGE_ID, sender="dana@acme.example", summary="t-1", excerpt="", scope=SCOPE
    )


def test_same_key_and_fingerprint_return_the_same_ask(tmp_path) -> None:
    queue = _queue(tmp_path)
    first = queue.enqueue_idempotent(issuance_key="iss-1", fingerprint=FP, questions=[_question()])
    assert first["ok"] is True
    ask_id = str(first["details"]["ask_id"])
    second = queue.enqueue_idempotent(issuance_key="iss-1", fingerprint=FP, questions=[_question()])
    assert second["ok"] is True
    assert str(second["details"]["ask_id"]) == ask_id
    assert second["details"]["status"] == "existing"


def test_the_same_key_after_consumption_still_returns_the_same_ask(tmp_path) -> None:
    queue = _queue(tmp_path)
    first = queue.enqueue_idempotent(issuance_key="iss-1", fingerprint=FP, questions=[_question()])
    ask_id = str(first["details"]["ask_id"])
    answered = queue.respond(ask_id, {f"bridge-{MESSAGE_ID}": ["approve"]}, by="human")
    assert answered.get("ok") is True
    again = queue.enqueue_idempotent(issuance_key="iss-1", fingerprint=FP, questions=[_question()])
    assert again["ok"] is True
    assert str(again["details"]["ask_id"]) == ask_id


def test_a_mismatched_fingerprint_refuses(tmp_path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue_idempotent(issuance_key="iss-1", fingerprint=FP, questions=[_question()])
    other = hashlib.sha256(b"a different original block").hexdigest()
    refused = queue.enqueue_idempotent(
        issuance_key="iss-1", fingerprint=other, questions=[_question()]
    )
    assert refused["ok"] is False
    assert "different request" in refused["error"]


def test_the_generic_enqueue_api_is_unchanged(tmp_path) -> None:
    queue = _queue(tmp_path)
    generic = queue.enqueue([_question()], None)
    assert generic["ok"] is True
    keyed = queue.enqueue_idempotent(issuance_key="iss-2", fingerprint=FP, questions=[_question()])
    assert keyed["ok"] is True
    assert str(keyed["details"]["ask_id"]) != str(generic["details"]["ask_id"])


def test_an_empty_timeout_is_a_bounds_error_not_a_silent_default(tmp_path) -> None:
    # ``None`` (omitted) is the policy default; ``""`` is an invalid value and
    # is refused with the shared bounds sentence, same as the generic enqueue.
    queue = _queue(tmp_path)
    refused = queue.enqueue_idempotent(
        issuance_key="iss-3", fingerprint=FP, questions=[_question()], timeout=""
    )
    assert refused["ok"] is False
    assert "timeout must be between" in refused["error"]
    assert queue.records() == []


def test_the_derived_id_is_in_the_store_namespace_and_deterministic(tmp_path) -> None:
    # R1-3 (PR #2378): the derived ask id must parse as the store's own
    # ``a-<hex>`` shape (``parse_mirror_request_id``), or the legacy mirror
    # card cannot answer a bridge ask; determinism keeps retries idempotent.
    from local_operator.asks import store as asks_store

    queue = _queue(tmp_path)
    first = queue.enqueue_idempotent(issuance_key="iss-ns", fingerprint=FP, questions=[_question()])
    assert first["ok"] is True
    ask_id = str(first["details"]["ask_id"])
    assert ask_id.startswith("a-")
    mirrored = f"{ask_id}{asks_store.MIRROR_REQUEST_SEPARATOR}1"
    assert asks_store.parse_mirror_request_id(mirrored) == (ask_id, 1)
    same = queue.enqueue_idempotent(issuance_key="iss-ns", fingerprint=FP, questions=[_question()])
    assert str(same["details"]["ask_id"]) == ask_id


def test_bridge_scope_digest_is_one_shared_serializer() -> None:
    # R1-5 (PR #2378): the queue's recompute and the issuance ledger's digest
    # are the SAME function — a divergence is a test failure, not a silent
    # "scope mismatch" refusal on an honest retry.
    from local_operator.asks.queue import _bridge_fingerprint_from_row
    from local_operator.team_bridge.ask_issuance import bridge_scope_digest

    assert bridge_scope_digest(SCOPE) == FP
    assert _bridge_fingerprint_from_row({"questions": [{"bridge": SCOPE}]}) == FP
    assert bridge_scope_digest(SCOPE) == _bridge_fingerprint_from_row(
        {"questions": [{"bridge": SCOPE}]}
    )
