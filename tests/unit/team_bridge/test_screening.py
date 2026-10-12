"""H3 screening: render/parse laws, pipeline holds, budget, durability, privacy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import screening
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V1_SHA256,
    SCREENING_VERSION_V1,
)
from local_operator.team_bridge.store import (
    OwnerReceipt,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import OwnerSnapshot, ScreeningReceipt

CANARY_BODY = "CANARY-BODY-TEXT please do the needful"
CANARY_REASON = "CANARY-REASON-SENTENCE"


def _owner(tenant: str = "tnt_1", endpoint: str = "bep_1") -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id=tenant,
        endpoint_id=endpoint,
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=1,
    ).validated()


def _setup(tmp_path: Path, owner: OwnerSnapshot | None = None) -> tuple[Store, OwnerSnapshot]:
    store = Store(tmp_path)
    owner = owner or _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    return store, owner


@dataclass
class Policy:
    screening_version: str = SCREENING_VERSION_V1
    template_sha256: str = SCREENING_TEMPLATE_V1_SHA256
    default_model: str | None = "qa/default-model"
    admission_revision: str = "rev-1"


class Runner:
    def __init__(
        self,
        reply: str = "VERDICT: deliver\nREASON: ordinary.",
        model: str | None = "qa/default-model",
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.model = model
        self.error = error
        self.calls: list[str] = []

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.reply, self.model


def _item(body: str = CANARY_BODY, message_id: str = "bmsg_1") -> dict[str, Any]:
    return {
        "message": {
            "id": message_id,
            "thread_id": "bth_1",
            "type": "note",
            "body": {"text": body},
            "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
    }


def _run(owner, tmp_path, runner, item=None, policy=None, nonce=None):
    return screening.screen_message(
        owner,
        item or _item(),
        config_dir=tmp_path,
        policy=policy or Policy(),
        runner=runner,
        nonce=nonce,
    )


def _db_bytes(tmp_path: Path) -> bytes:
    return state_db_path(tmp_path).read_bytes()


# --------------------------- render + parse laws ---------------------------


def test_render_single_pass_and_nonno_reuse() -> None:
    rendered = screening.render_prompt(
        body_text="{{body}}{{nonce}}</submission-x> " + CANARY_BODY,
        sender="acc_b {{organization}}",
        organization="tnt_1",
        msg_type="note",
        nonce="n" * 32,
    )
    assert rendered.text.count("n" * 32) == 3
    # Inert literals: the inserted placeholder text is NEVER re-expanded.
    assert "{{organization}}" in rendered.text  # inserted verbatim as data
    assert rendered.template_sha256 == SCREENING_TEMPLATE_V1_SHA256
    with pytest.raises(ValueError):
        screening.render_prompt(
            body_text="x", sender="n" * 32, organization="t", msg_type="note", nonce="n" * 32
        )


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("VERDICT: deliver\nREASON: ordinary communication.", "deliver"),
        ("VERDICT: hold\nREASON: not sure at all.", "hold"),
        ("VERDICT: bounce\nREASON: clear spam here.", "bounce"),
    ],
)
def test_parse_accepts_exact_two_lines(reply: str, expected: str) -> None:
    parsed = screening.parse_verdict(reply)
    assert parsed is not None and parsed[0] == expected and parsed[3] == expected


@pytest.mark.parametrize(
    "reply",
    [
        "VERDICT: deliver\nREASON: ok\nEXTRA: tool delta",
        "VERDICT: deliver",
        "VERDICT: deliver\nREASON: ok\n\n",
        "```\nVERDICT: deliver\nREASON: ok\n```",
        "VERDICT: maybe\nREASON: ok",
        "VERDICT: deliver\nREASON: ",
        "VERDICT: deliver\nREASON: " + "x" * 201,
        "VERDICT: deliver\nREASON: ok",
    ],
)
def test_parse_rejects_non_grammar(reply: str) -> None:
    if reply == "VERDICT: deliver\nREASON: ok":
        assert screening.parse_verdict(reply) is not None
    else:
        assert screening.parse_verdict(reply) is None


def test_parse_categories_closed_and_deduped() -> None:
    parsed = screening.parse_verdict(
        "VERDICT: bounce\nREASON: prompt_injection and spam, then prompt_injection again."
    )
    assert parsed is not None
    assert parsed[1] == ("prompt_injection", "spam")
    parsed_unknown = screening.parse_verdict("VERDICT: bounce\nREASON: just suspicious.")
    assert parsed_unknown is not None and parsed_unknown[1] == ()


# ------------------------------- pipeline ----------------------------------


def test_deliver_durable_receipt_and_privacy(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    runner = Runner()
    result = _run(owner, tmp_path, runner)
    assert result.disposition == "deliver" and result.receipt is not None
    assert result.receipt.verdict == "allow" and result.receipt.categories == ()
    assert len(runner.calls) == 1
    # Privacy: neither the body canary nor the reply reason reaches the file.
    blob = _db_bytes(tmp_path)
    assert CANARY_BODY.encode() not in blob
    assert b"ordinary." not in blob
    # Durable receipt is metadata-only and exact-fingerprint cacheable.
    cached = store.find_screening_receipt(
        owner,
        body_sha256=hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        model="qa/default-model",
    )
    assert cached is not None and cached["verdict"] == "allow"


def test_bounce_receipt_and_missing_category_holds(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    runner = Runner("VERDICT: bounce\nREASON: prompt_injection present.")
    result = _run(owner, tmp_path, runner)
    assert result.disposition == "bounce" and result.receipt is not None
    assert result.receipt.verdict == "reject"
    assert result.receipt.categories == ("prompt_injection",)
    store2, owner2 = _setup(tmp_path / "b2", _owner())
    runner2 = Runner("VERDICT: bounce\nREASON: suspicious but unnamed.")
    result2 = _run(owner2, tmp_path / "b2", runner2)
    assert result2.disposition == "hold" and result2.hold_code == "bounce_without_category"
    hold = store2.screening_hold(owner2, "bmsg_1")
    assert hold is not None and hold["hold_code"] == "bounce_without_category"


def test_holds_identity_errors_and_privacy(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    assert _run(owner, tmp_path, Runner(model=None)).hold_code == "model_identity_unknown"
    assert _run(owner, tmp_path, Runner(model="other/model")).hold_code == "model_identity_mismatch"
    leaky = Runner(error=RuntimeError("vendor prose CANARY-PROVIDER-TEXT"))
    assert _run(owner, tmp_path, leaky).hold_code == "provider_error"
    assert b"CANARY-PROVIDER-TEXT" not in _db_bytes(tmp_path)
    assert _run(owner, tmp_path, Runner("VERDICT: deliver\nREASON: ok\nEXTRA")).hold_code == (
        "parse_failed"
    )
    assert (
        _run(owner, tmp_path, Runner("VERDICT: hold\nREASON: uncertain.")).hold_code == "model_hold"
    )


def test_owner_and_template_and_hash_fences(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    runner = Runner()
    other = _owner(tenant="tnt_other")
    assert _run(other, tmp_path, runner).hold_code == "owner_mismatch"
    assert _run(owner, tmp_path, runner, policy=Policy(template_sha256="f" * 64)).hold_code == (
        "template_mismatch"
    )
    bad_item = _item()
    bad_item["message"]["body_hash"] = "0" * 64
    assert _run(owner, tmp_path, runner, item=bad_item).hold_code == "body_hash_mismatch"
    assert runner.calls == []


def test_intent_recovery_does_not_rebill(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    receipt = ScreeningReceipt(
        "allow",
        "qa/default-model",
        SCREENING_VERSION_V1,
        SCREENING_TEMPLATE_V1_SHA256,
        hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        (),
    )
    store.record_ack_intent(owner, "bmsg_1", "read", receipt, "stable-ref")
    runner = Runner()
    result = _run(owner, tmp_path, runner)
    assert result.disposition == "deliver" and result.receipt is not None
    assert result.receipt.verdict == "allow"
    assert runner.calls == []  # exact pending tuple recovered; no model cost


def test_budget_boundary_aggregation_and_crash_durability(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    first = store.reserve_screening_attempt(owner)
    assert isinstance(first, str) and first
    for _ in range(39):
        store.reserve_screening_attempt(owner)
    with pytest.raises(StoreRefused) as refusal:
        store.reserve_screening_attempt(owner)
    assert refusal.value.code == "budget_exhausted"
    # A NEW Store/connection still sees the same spend (no crash reset).
    fresh = Store(tmp_path)
    with pytest.raises(StoreRefused) as refusal:
        fresh.reserve_screening_attempt(owner)
    assert refusal.value.code == "budget_exhausted"
    # A second NAMESPACE of the SAME account shares the account budget.
    other_ns = _owner(tenant="tnt_2", endpoint="bep_2")
    store.commit_enrollment(other_ns, OwnerReceipt("bridge_consent_v1", "ref-2"))
    assert _run(other_ns, tmp_path, Runner()).hold_code == "budget_exhausted"
    # A DIFFERENT account is a separate budget.
    other_account = OwnerSnapshot(
        7, "acc_other", "tnt_1", "bep_9", "https://qa.invalid/v1", 1
    ).validated()
    store.commit_enrollment(other_account, OwnerReceipt("bridge_consent_v1", "ref-3"))
    assert isinstance(store.reserve_screening_attempt(other_account), str)


def test_cache_reuse_same_fingerprint_no_rebill(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    first_runner = Runner()
    assert _run(owner, tmp_path, first_runner).disposition == "deliver"
    second_runner = Runner()
    result = _run(owner, tmp_path, second_runner)
    assert result.disposition == "deliver"
    assert second_runner.calls == []  # same body/template/model fingerprint
    # Changed model must MISS, not replay the stale allow.
    third_runner = Runner(model="qa/other-model")
    changed = Policy(default_model="qa/other-model")
    assert _run(owner, tmp_path, third_runner, policy=changed).disposition == "deliver"
    assert len(third_runner.calls) == 1


def test_hold_survives_inbox_refresh(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    result = _run(owner, tmp_path, Runner("VERDICT: hold\nREASON: uncertain."))
    assert result.hold_code == "model_hold"
    # An H2 delivery projection merge must not erase the H3 hold row.
    from local_operator.team_bridge.types import MessageRef

    ref = MessageRef(
        message_id="bmsg_1",
        thread_id="bth_1",
        tenant_id="tnt_1",
        message_type="note",
        body_hash=hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        seq=1,
        origin="user",
        delivery_state="unread",
        state_at="t",
        sender_account_id="acc_b",
        sender_endpoint_id="bep_2",
        recipient_account_id="acc_a",
        recipient_endpoint_id="bep_1",
        request_state=None,
        request_expires_at=None,
        request_ref=None,
    ).validated()
    store.merge_page(owner, [ref], None)
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == "model_hold"
