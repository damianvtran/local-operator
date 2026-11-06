"""Sender provenance: the sender a message is CHARGED to is the one dispatch saw.

The #3199 residual, carried forward from the superseded ``fix/3199-sender-budget``.
Four properties, each one a way an unverified sender could buy allowance:

* the ledger column that makes a sender VERIFIED (``sender_kind``) and the check
  that reads it — an account with no stated class is unknown provenance, never
  an attributed charge;
* the dispatched job's provenance equals the registration's, refused BEFORE the
  physical charge (``reserve_screening_dispatch``);
* unattributed spend fails closed: it counts against every sender's allowance
  and sees all spend itself, so losing provenance can never buy a fresh twenty;
* an input refusal is isolated from the outage breaker — the provider rejected
  this submission, which says nothing about route health.

Not covered here: the tier/account ceilings (``test_batch_screening``) and the
screening POLICY, which this change does not touch.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import screening
from local_operator.team_bridge.connector.conversation import _allocation_sender
from local_operator.team_bridge.model_boundary import AdmissionJob, _scrub_wire_failure
from local_operator.team_bridge.settings import ScreeningBudget
from local_operator.team_bridge.store import StoreRefused, state_db_path
from tests.unit.team_bridge.test_screening import (  # noqa: F401 - autouse fixture
    Policy,
    Runner,
    _claim_turn,
    _item,
    _owner,
    _run,
    _setup,
)

#: Small, NESTED set (``ScreeningBudget`` refuses a non-nested one): the sender
#: fence is what these tests state, so it is the one held small.
SMALL = ScreeningBudget(account=10, ambient=8, unaddressed=9, sender=2)


def _provenance(sender: str = "", kind: str = "") -> dict[str, str]:
    """The two proof fields, always present (empty means unknown)."""

    return {"sender_account_id": sender, "sender_kind": kind}


def _proof(owner, *, sender: str = "", kind: str = "") -> dict[str, Any]:
    """A registered input proof, with provenance when the caller states it."""

    return {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": "b" * 64,
        "template": "ab" * 32,
        "version": "bridge-screen-v1",
        "provider": "openai",
        "model": "qa-default",
        "approved": ["qa-default"],
        "envelope": "e" * 64,
        "request": "r" * 64,
        "display": "d" * 64,
        "revision": "rev-1",
        "epoch": 1,
        "message": "bmsg_1",
        "tenant": owner.tenant_id,
        "logical": "rev-1|" + "e" * 64,
        **_provenance(sender, kind),
    }


def _job(owner, attempt: str, turn: str, *, sender: str = "", kind: str = "") -> AdmissionJob:
    return AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256="b" * 64,
        template_sha256="ab" * 32,
        template_version="bridge-screen-v1",
        provider="openai",
        model="qa-default",
        approved=("qa-default",),
        envelope_digest="e" * 64,
        request_digest="r" * 64,
        display_digest="d" * 64,
        revision="rev-1",
        epoch=1,
        turn=turn,
        sender_account_id=sender,
        sender_kind=kind,
    )


def _http_rows(tmp_path: Path) -> int:
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM screening_attempts WHERE kind = 'http'"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def _logical_rows(tmp_path: Path) -> list[tuple[str | None, str | None]]:
    """Every logical ledger row's ``(sender_account_id, sender_kind)``."""

    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT sender_account_id, sender_kind FROM screening_attempts "
                "WHERE kind = 'logical' ORDER BY rowid"
            )
        ]
    finally:
        conn.close()


# ------------------------- read-time provenance checks -------------------------


def test_a_job_whose_sender_changed_after_registration_is_refused(tmp_path: Path) -> None:
    """The charge follows the sender the REGISTRATION pinned, not the job's word."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint="rev-1|" + "e" * 64,
        input_proof=_proof(owner, sender="acc_ada", kind="human"),
        sender_account_id="acc_ada",
        sender_kind="human",
    )
    job = _job(owner, attempt, turn, sender="acc_eve", kind="agent")
    # The FIRST bind is the earliest native boundary, so the refusal lands
    # there rather than at the dispatch — no physical charge either way.
    with pytest.raises(StoreRefused) as refused:
        store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    assert refused.value.code == "attempt_mismatch"
    assert _http_rows(tmp_path) == 0  # refused BEFORE any physical charge


def test_a_job_with_no_provenance_cannot_reach_an_attributed_registration(
    tmp_path: Path,
) -> None:
    """Both directions: a registration cannot silently LOSE its attribution."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint="rev-1|" + "e" * 64,
        input_proof=_proof(owner, sender="acc_ada", kind="human"),
        sender_account_id="acc_ada",
        sender_kind="human",
    )
    job = _job(owner, attempt, turn)
    with pytest.raises(StoreRefused) as refused:
        store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    assert refused.value.code == "attempt_mismatch"
    assert _http_rows(tmp_path) == 0


def test_a_later_dispatch_with_a_different_sender_is_refused(tmp_path: Path) -> None:
    """The check runs again before every actual send (retry, resend, fallback)."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint="rev-1|" + "e" * 64,
        input_proof=_proof(owner, sender="acc_ada", kind="human"),
        sender_account_id="acc_ada",
        sender_kind="human",
    )
    bound = _job(owner, attempt, turn, sender="acc_ada", kind="human")
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=bound.digest, job=bound)
    altered = _job(owner, attempt, turn, sender="acc_eve", kind="agent")
    with pytest.raises(StoreRefused) as refused:
        store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=altered, turn=turn)
    # The pair rides the canonical digest, so a changed sender is refused by the
    # digest guard before the field comparison is reached: the same refusal, one
    # step earlier. The field comparison (``_input_proof_mismatch``) is the
    # defence for a registration mutated behind a still-matching digest, which
    # ``test_a_registered_proof_with_a_half_stated_pair_is_damage`` covers.
    assert refused.value.code == "unknown_attempt"
    assert _http_rows(tmp_path) == 0
    # The UNALTERED job still dispatches: the refusal is the change, not the row.
    store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=bound, turn=turn)
    assert _http_rows(tmp_path) == 1


def test_a_registered_proof_with_a_half_stated_pair_is_damage(tmp_path: Path) -> None:
    """The retained record is re-validated at every native boundary."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint="rev-1|" + "e" * 64,
        input_proof=_proof(owner, sender="acc_ada", kind="human"),
        sender_account_id="acc_ada",
        sender_kind="human",
    )
    job = _job(owner, attempt, turn, sender="acc_ada", kind="human")
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    # Damage the RETAINED registration the way a bad migration would: a class
    # with no account. The digest prefix stays, so only the field check can see
    # it — which is the point of validating the decoded record.
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        raw = conn.execute(
            "SELECT fingerprint FROM screening_attempts WHERE attempt_id = ?", (attempt,)
        ).fetchone()[0]
        digest, _, proof_raw = raw.partition("|")
        registered = json.loads(proof_raw[len("proof:") :])
        registered["sender_account_id"] = ""
        registered["sender_kind"] = "human"
        conn.execute(
            "UPDATE screening_attempts SET fingerprint = ? WHERE attempt_id = ?",
            (f"{digest}|proof:" + json.dumps(registered, sort_keys=True), attempt),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as refused:
        store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=job, turn=turn)
    assert refused.value.code == "proof_corrupt"


def test_a_pre_provenance_proof_still_dispatches(tmp_path: Path) -> None:
    """An upgrade mid-flight: a proof without the pair is not damage."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    legacy = {
        key: value
        for key, value in _proof(owner).items()
        if key not in ("sender_account_id", "sender_kind")
    }
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint="rev-1|" + "e" * 64,
        input_proof=legacy,
    )
    job = _job(owner, attempt, turn)
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=job, turn=turn)
    assert _http_rows(tmp_path) == 1


# ----------------------- where the identity may come from -----------------------


def test_the_account_is_the_attribution_and_the_class_only_rides_it() -> None:
    """A hub-stated account is the sender; a class on its own names nobody.

    Round 1 S-1/S-2: gating the charge on the class DROPPED the inbox lane's
    verified account, and those unattributed rows then exhausted every named
    sender. Inventing the class is the other failure (M1), so it is carried
    when stated and never inferred.
    """

    assert _allocation_sender({"sender_account_id": "acc_ada"}) == {
        "account_id": "acc_ada",
        "kind": "",
        "display_label": "",
    }
    assert (
        _allocation_sender({"sender_account_id": "acc_ada", "sender_kind": "human"})["kind"]
        == "human"
    )
    # A class with no account names nobody.
    assert _allocation_sender({"sender_kind": "human"})["account_id"] == ""
    # Present-but-null is the server saying nothing, not a malformed message
    # (round 1 N1).
    assert _allocation_sender({"sender_account_id": None, "sender_kind": None})["account_id"] == ""
    with pytest.raises(ValueError):
        _allocation_sender({"sender_account_id": "acc_ada", "sender_kind": "admin"})
    # The connector's cap is the envelope's, so a long account holds on the
    # ENVELOPE rule rather than passing here and failing one layer down
    # (round 1 S-5).
    with pytest.raises(ValueError):
        _allocation_sender({"sender_account_id": "a" * 201})
    # An OLD server states nothing at all: unknown provenance, never refused.
    assert _allocation_sender({})["account_id"] == ""


def test_the_envelope_takes_provenance_only_from_the_stated_participant() -> None:
    item = _item()
    item["message"]["sender"] = {"account_id": "acc_ada", "kind": "human"}
    envelope, problem = screening.validate_envelope(_owner(), item)
    assert problem is None and envelope is not None
    assert (envelope.sender_account, envelope.sender_kind) == ("acc_ada", "human")
    # A kind with no account is unknown, not a sender with a defaulted class.
    item["message"]["sender"] = {"kind": "human"}
    envelope, problem = screening.validate_envelope(_owner(), item)
    assert problem is None and envelope is not None
    assert (envelope.sender_account, envelope.sender_kind) == ("", "")
    # An ACCOUNT with no stated class IS the sender — the class only rides it
    # (round 1 S-1/S-2). This is the inbox lane's shape.
    item["message"]["sender"] = {"account_id": "acc_ada"}
    envelope, problem = screening.validate_envelope(_owner(), item)
    assert problem is None and envelope is not None
    assert (envelope.sender_account, envelope.sender_kind) == ("acc_ada", "")
    # An endpoint is NOT a class: an endpoint-only sender is unknown
    # provenance, never a guessed "agent" (round 1 M1).
    item["message"]["sender"] = {"account_id": "acc_ada", "endpoint_id": "bep_remote"}
    envelope, problem = screening.validate_envelope(_owner(), item)
    assert problem is None and envelope is not None
    assert (envelope.sender_account, envelope.sender_kind) == ("acc_ada", "")
    # An OUT-of-class kind refuses the envelope outright.
    item["message"]["sender"] = {"account_id": "acc_ada", "kind": "admin"}
    assert screening.validate_envelope(_owner(), item) == (None, "envelope_malformed")
    # The PAIR lane (no participant block at all) keeps its stored account, so
    # its charge is attributed and no digest moves (round 1 S-2).
    pair = _item()
    envelope, problem = screening.validate_envelope(_owner(), pair)
    assert problem is None and envelope is not None
    assert (envelope.sender_account, envelope.sender_kind) == ("acc_sender1", "")


# --------------------------- fail-closed accounting ---------------------------


def test_unattributed_rows_do_not_deplete_a_named_sender(tmp_path: Path) -> None:
    """Somebody else's unattributed work must not fence a named sender.

    Round 1 S-1, measured end to end by security: the inbox lane states an
    account and no class, so its rows were unattributed and the two
    ``COALESCE`` arms spent EVERY named sender's allowance with them — three
    ordinary inbox screens then held a tier-3 person DM ``budget_exhausted``
    while the account and tier ceilings still had room. Crossing senders is the
    property, so it is asserted across senders: A's unattributed traffic must
    not push B out of ``sender_exhausted``.
    """

    store, owner = _setup(
        tmp_path,
        screening_budget=ScreeningBudget(account=10, ambient=8, unaddressed=9, sender=2),
    )
    # TWO unattributed units is all the unattributed CALLER gets (it sees every
    # row, so the sender ceiling bounds it directly); the third is refused.
    for index in range(2):
        store.reserve_screening_attempt(owner, message_id=f"inbox{index}", fingerprint="f")
    with pytest.raises(StoreRefused) as unattributed:
        store.reserve_screening_attempt(owner, message_id="inbox_more", fingerprint="f")
    assert unattributed.value.code == "sender_exhausted"
    # B has spent nothing of its own, so B still reserves — twice, up to B's
    # own share, and no further. Under the round-1 rule B was refused its FIRST
    # unit here.
    for index in range(2):
        store.reserve_screening_attempt(
            owner,
            message_id=f"b_{index}",
            fingerprint="f",
            sender_account_id="acc_b",
            sender_kind="human",
        )
    with pytest.raises(StoreRefused) as refused:
        store.reserve_screening_attempt(
            owner,
            message_id="b_third",
            fingerprint="f",
            sender_account_id="acc_b",
            sender_kind="human",
        )
    # B's OWN two units are what fenced B, not the six inbox rows.
    assert refused.value.code == "sender_exhausted"


def test_unattributed_work_sees_all_spend_and_gets_no_fresh_allowance(
    tmp_path: Path,
) -> None:
    """Losing provenance must not buy a fresh twenty."""

    store, owner = _setup(tmp_path, screening_budget=SMALL)
    for index, sender in enumerate(("acc_a", "acc_b")):
        store.reserve_screening_attempt(
            owner,
            message_id=f"m{index}",
            fingerprint="f",
            sender_account_id=sender,
            sender_kind="agent",
        )
    with pytest.raises(StoreRefused) as refused:
        store.reserve_screening_attempt(owner, message_id="m-old", fingerprint="f")
    assert refused.value.code == "sender_exhausted"


def test_a_verified_account_without_a_stated_class_is_charged_to_that_sender(
    tmp_path: Path,
) -> None:
    """A pre-upgrade row (account, NULL kind) belongs to its sender, not to all.

    Round 1 S-1 spent these against every sender; the account IS the attribution
    (S-2), so one of them fences that account's share and nobody else's.
    """

    store, owner = _setup(
        tmp_path,
        screening_budget=ScreeningBudget(account=10, ambient=8, unaddressed=9, sender=1),
    )
    store.reserve_screening_attempt(
        owner, message_id="legacy", fingerprint="f", sender_account_id="acc_ada"
    )
    with pytest.raises(StoreRefused) as refused:
        store.reserve_screening_attempt(
            owner,
            message_id="new",
            fingerprint="f",
            sender_account_id="acc_ada",
            sender_kind="agent",
        )
    assert refused.value.code == "sender_exhausted"
    # A DIFFERENT sender is untouched by those rows.
    store.reserve_screening_attempt(
        owner,
        message_id="other",
        fingerprint="f",
        sender_account_id="acc_eve",
        sender_kind="agent",
    )


def test_the_pair_lanes_verified_account_reaches_the_ledger(tmp_path: Path) -> None:
    """The inbox/pair lane states an account and no class; it must be charged.

    Round 1 S-2, measured: the envelope held ``acc_sender1`` and the row was
    written ``('logical', None, None, ...)`` — a hub-verified identity validated
    and then dropped one line later, which is what made S-1 reachable on the
    live lane.
    """

    store, owner = _setup(tmp_path)
    out = _run(owner, tmp_path, Runner())
    assert out.disposition == "deliver"
    assert _logical_rows(tmp_path) == [("acc_sender1", None)]


# ------------------------ input refusal vs outage breaker ------------------------


def test_an_input_refusal_holds_without_opening_the_breaker(tmp_path) -> None:
    """A provider rejecting THIS submission is not an unhealthy route."""

    from local_operator.providers.failover import ProviderError

    refusal = ProviderError(
        400,
        'error: {"code":"data_inspection_failed","message":"Input text data may contain '
        'inappropriate content"}',
    )
    store, owner = _setup(tmp_path)
    for index in range(3):
        out = _run(
            owner,
            tmp_path,
            Runner(error=refusal),
            item=_item(f"probe {index}", message_id=f"bmsg_probe{index}"),
        )
        assert out.disposition == "hold"
        assert out.hold_code == "provider_error"
    assert store.screening_breaker_state(owner) is None
    # The next message dispatches instead of being parked behind a breaker.
    runner = Runner()
    out = _run(owner, tmp_path, runner, item=_item("hi", message_id="bmsg_next"))
    assert out.disposition == "deliver" and len(runner.calls) == 1


def test_a_route_failure_still_opens_the_breaker(tmp_path) -> None:
    """The isolation is narrow: ordinary provider failures keep the old path."""

    from local_operator.providers.failover import ProviderError

    store, owner = _setup(tmp_path)
    for index in range(3):
        out = _run(
            owner,
            tmp_path,
            Runner(error=ProviderError(503, "upstream unavailable")),
            item=_item(f"route {index}", message_id=f"bmsg_route{index}"),
        )
        assert out.hold_code == "provider_error"
    state = store.screening_breaker_state(owner)
    assert state is not None and state["failures"] == 3 and state["open_until"] > time.time()


def test_the_refusal_flag_survives_a_second_scrub() -> None:
    """Round 1 M3: once the prose is gone the predicate has nothing to read.

    Idempotent scrubbing is what keeps the class across a re-scrub, so a
    refusal cannot turn into an outage advance by being handled twice.
    """

    from local_operator.providers.failover import ProviderError

    once = _scrub_wire_failure(ProviderError(400, '{"code":"data_inspection_failed"}'))
    twice = _scrub_wire_failure(once)
    assert getattr(once, "tb_input_refusal", False) is True
    assert getattr(twice, "tb_input_refusal", False) is True
    ordinary = _scrub_wire_failure(_scrub_wire_failure(ProviderError(400, "bad image")))
    assert getattr(ordinary, "tb_input_refusal", False) is False


def test_the_typed_flag_is_decided_before_the_wire_text_is_scrubbed() -> None:
    """Only a boolean survives the scrub; the refusal prose never does."""

    from local_operator.providers.failover import ProviderError

    refusal = _scrub_wire_failure(ProviderError(400, '{"code":"data_inspection_failed"}'))
    assert getattr(refusal, "tb_input_refusal", False) is True
    assert "data_inspection_failed" not in str(refusal)
    ordinary = _scrub_wire_failure(ProviderError(400, "image could not be processed"))
    assert getattr(ordinary, "tb_input_refusal", False) is False
