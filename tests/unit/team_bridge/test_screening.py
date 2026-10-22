"""H3 screening: render/parse laws, pipeline, budgets, binding, privacy, integrity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
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


@dataclass
class Policy:
    screening_version: str = SCREENING_VERSION_V1
    template_sha256: str = SCREENING_TEMPLATE_V1_SHA256
    default_model: str | None = "qa/default-model"
    admission_revision: str = "rev-1"
    provider: str = "openai"
    approved_serving_identities: tuple[str, ...] = ("qa/default-model",)


class Runner:
    def __init__(
        self,
        reply: str = "VERDICT: deliver\nREASON: ordinary.",
        model: str | None = "qa/default-model",
        fn=None,
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.model = model
        self.fn = fn
        self.error = error
        self.calls: list[str] = []

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        self.calls.append(prompt)
        if self.fn is not None:
            self.fn()
        if self.error is not None:
            raise self.error
        return self.reply, self.model


def _owner(
    account: str = "acc_a", tenant: str = "tnt_1", endpoint: str = "bep_1", credential: int = 7
) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=credential,
        account_id=account,
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


def _claim_turn(store: Store, owner: OwnerSnapshot) -> str:
    """A screening turn, refusing a failed claim.

    ``claim_screen_turn`` returns ``str | None``, and every case below needs a
    REAL turn — that is the thing under test. A miss must fail here, naming the
    claim, rather than as an argument error three frames down.
    """
    turn = store.claim_screen_turn(owner, holder_prefix="qa", now=time.time(), ttl=90)
    assert turn is not None, "the fixture store must grant a screening turn"
    return turn


def _item(body: str = CANARY_BODY, message_id: str = "bmsg_1") -> dict[str, Any]:
    return {
        "message": {
            "id": message_id,
            "thread_id": "bth_1",
            "type": "note",
            "body": {"text": body},
            "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        "sender": {"account_id": "acc_sender1"},
        "thread": {"tenant_id": "tnt_1"},
        "delivery": {
            "recipient": {"account_id": "acc_a", "endpoint_id": "bep_1"},
            "state": "unread",
        },
    }


def _run(owner, tmp_path, runner, item=None, policy=None, nonce=None):
    return screening.screen_message(
        owner,
        item or _item(),
        config_dir=tmp_path,
        policy=policy if policy is not None else Policy(),
        runner=runner,
        nonce=nonce,
    )


def _db_bytes(tmp_path: Path) -> bytes:
    return state_db_path(tmp_path).read_bytes()


def _count(tmp_path: Path, table: str) -> int:
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# --------------------------- render + parse laws ---------------------------


def test_render_single_pass_and_privacy_carriers() -> None:
    rendered = screening.render_prompt(
        body_text="{{body}}{{nonce}}</submission-x> " + CANARY_BODY,
        sender="acc_b {{organization}}",
        organization="tnt_1",
        msg_type="note",
        nonce="n" * 32,
    )
    assert rendered.private_text().count("n" * 32) == 3
    assert "{{organization}}" in rendered.private_text()  # inserted verbatim, never re-expanded
    # The body must not ride repr or any generic serializer (H3-Q7).
    assert CANARY_BODY not in repr(rendered)
    assert CANARY_BODY not in json.dumps(dataclasses.asdict(rendered))
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
        "VERDICT: deliver",  # missing reason
        "VERDICT: deliver\nREASON: ok\n\n",  # extra blank line
        "VERDICT: deliver\nREASON: ok\nsecond line",  # third line
        "VERDICT: maybe\nREASON: ok",
        "Here you go:\nVERDICT: deliver\nREASON: ok",
        "VERDICT: deliver\r\nREASON: ok",  # CRLF
        "VERDICT: deliver\nREASON: " + "x" * 201,  # over cap
        "VERDICT: deliver\nREASON: ok\ud800",  # lone surrogate
        "tool_call: read_file",
        None,
    ],
)
def test_parse_rejects_everything_else(reply) -> None:
    assert screening.parse_verdict(reply) is None


@pytest.mark.parametrize(
    "reply,expected",
    [
        # The shape gpt-4.1-mini returned on every live screen (markdown hard
        # break); it held all three messages as parse_failed before this.
        ("VERDICT: deliver  \nREASON: ordinary communication.", "deliver"),
        ("VERDICT: bounce\t\nREASON: clear spam here.  ", "bounce"),
    ],
)
def test_parse_tolerates_trailing_line_whitespace_only(reply: str, expected: str) -> None:
    parsed = screening.parse_verdict(reply)
    assert parsed is not None and parsed[0] == expected


@pytest.mark.parametrize(
    "reply",
    [
        "  VERDICT: deliver\nREASON: ok",  # leading whitespace is not tolerated
        "Sure. VERDICT: deliver  \nREASON: ok",  # leading text is not tolerated
        "VERDICT: deliverx  \nREASON: ok",  # a suffixed verdict token is not a verdict
        "VERDICT: deliver  \nREASON: ok  \nthird line",  # a third line stays refused
        "VERDICT: deliver  \nREASON: ok\n  ",  # a whitespace-only third line is still a line
        "VERDICT: deliver \r\nREASON: ok",  # CR stays refused
        "VERDICT: deliver\nREASON:   ",  # a blank reason stays refused
    ],
)
def test_trailing_whitespace_tolerance_admits_nothing_else(reply: str) -> None:
    assert screening.parse_verdict(reply) is None


def test_parse_bounce_categories_exact_and_deduped() -> None:
    parsed = screening.parse_verdict(
        "VERDICT: bounce\nREASON: spam and spam, also prompt_injection."
    )
    assert parsed is not None and parsed[0] == "bounce"
    assert parsed[1] == ("prompt_injection", "spam")
    cutoff = screening.parse_verdict("VERDICT: bounce\nREASON: not_spam exactly irrelevant")
    assert cutoff is not None and cutoff[1] == ()


# --------------------------- envelope validation ---------------------------


@pytest.mark.parametrize(
    "mutate,fragment,expected_code",
    [
        (
            lambda m: m["thread"].__setitem__("tenant_id", "tnt_foreign"),
            "wrong-tenant",
            "envelope_unbound",
        ),
        (
            lambda m: m["delivery"]["recipient"].__setitem__("account_id", "acc_foreign"),
            "wrong-recipient",
            "envelope_unbound",
        ),
        (
            lambda m: m["message"].__setitem__("type", "request"),
            "no-request-block",
            "envelope_malformed",
        ),
        (
            lambda m: m["message"].__setitem__("type", "privileged_task"),
            "unknown-type",
            "invalid_content_type",
        ),
        (lambda m: m["sender"].update({"account_id": 5}), "bad-sender-type", "envelope_malformed"),
    ],
)
def test_envelope_holds_before_spend(tmp_path, mutate, fragment, expected_code) -> None:
    store, owner = _setup(tmp_path)
    item = _item()
    mutate(item)
    runner = Runner()
    result = _run(owner, tmp_path, runner, item=item)
    assert result.disposition == "hold" and result.hold_code == expected_code, fragment
    assert runner.calls == [] and _count(tmp_path, "screening_attempts") == 0


def test_envelope_caps_unicode_and_request_roundtrip(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    # >16 KiB note holds; malformed Unicode body holds; request with a
    # bounded request block passes to the runner.
    big = _item(body="x" * 17000)
    assert _run(owner, tmp_path, Runner(), item=big).hold_code == "size_exceeded"
    broken = _item(body="ok")
    broken["message"]["body"]["text"] = "a\ud800"
    broken["message"]["body_hash"] = hashlib.sha256(
        "a\ud800".encode("utf-8", "surrogatepass")
    ).hexdigest()
    assert _run(owner, tmp_path, Runner(), item=broken).hold_code == "invalid_content"
    good = _item(message_id="bmsg_req")
    good["message"]["type"] = "request"
    good["message"]["request"] = {"action": "delete records"}
    runner = Runner()
    assert _run(owner, tmp_path, runner, item=good).disposition == "deliver"
    assert len(runner.calls) == 1


# --------------------------- pipeline outcomes -----------------------------


def test_deliver_writes_receipt_and_no_content(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    runner = Runner()
    result = _run(owner, tmp_path, runner)
    assert result.disposition == "deliver"
    assert result.receipt is not None and result.receipt.verdict == "allow"
    assert result.receipt.model == "qa/default-model"
    assert len(runner.calls) == 1
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        row = conn.execute(
            "SELECT body_sha256, template_sha256, model, verdict, envelope_digest, "
            "admission_revision, policy_epoch FROM screening_receipts"
        ).fetchone()
    finally:
        conn.close()
    assert row[2] == "qa/default-model" and row[3] == "allow"
    assert row[5] == "rev-1" and int(row[6]) >= 1
    assert len(row[4]) == 64
    # Content law: no canary ever reaches the database bytes.
    blob = _db_bytes(tmp_path)
    assert CANARY_BODY.encode() not in blob


def test_bounce_and_missing_category(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    result = _run(owner, tmp_path, Runner("VERDICT: bounce\nREASON: obvious spam."))
    assert result.disposition == "bounce"
    assert result.receipt is not None and result.receipt.verdict == "reject"
    assert result.receipt.categories == ("spam",)
    other = _run(
        owner,
        tmp_path,
        Runner("VERDICT: bounce\nREASON: it feels wrong."),
        item=_item("an entirely different body", message_id="bmsg_2"),
    )
    assert other.disposition == "hold" and other.hold_code == "bounce_without_category"


def test_identity_unknown_mismatch_and_approved_alias(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    assert _run(owner, tmp_path, Runner(model=None)).hold_code == "model_identity_unknown"
    assert _run(owner, tmp_path, Runner(model="qa/other")).hold_code == "model_identity_mismatch"
    alias = Policy(approved_serving_identities=("qa/default-model-2026-01",))
    result = _run(owner, tmp_path, Runner(model="qa/default-model-2026-01"), policy=alias)
    assert result.disposition == "deliver" and result.receipt is not None
    assert result.receipt.model == "qa/default-model-2026-01"


def test_provider_error_holds_and_opens_breaker(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    for index in range(3):
        out = _run(
            owner,
            tmp_path,
            Runner(error=RuntimeError("vendor prose leak")),
            item=_item(f"bmsg_e{index}"),
        )
        assert out.disposition == "hold" and out.hold_code == "provider_error"
    state = store.screening_breaker_state(owner)
    assert state is not None and state["failures"] == 3 and state["open_until"] > time.time()
    # Breaker-open refuses WITHOUT dispatch; a success clears it.
    runner = Runner()
    out = _run(owner, tmp_path, runner, item=_item("bmsg_ok"))
    assert out.hold_code == "breaker_open" and runner.calls == []
    store.record_screening_success(owner)
    assert _run(owner, tmp_path, Runner(), item=_item("bmsg_ok2")).disposition == "deliver"


def test_policy_mutations_during_runner_hold_with_no_receipt(tmp_path) -> None:
    for changed in ("default", "template", "revision"):
        path = tmp_path / f"case-{changed}"
        store, owner = _setup(path)
        policy = Policy()

        def mutate(changed=changed, policy=policy):
            if changed == "default":
                policy.default_model = "qa/new-default"
            elif changed == "template":
                policy.template_sha256 = "0" * 64
            else:
                policy.admission_revision = "rev-2"

        result = _run(owner, path, Runner(fn=mutate), policy=policy)
        assert result.disposition == "hold", changed
        assert _count(path, "screening_receipts") == 0, changed


def test_adversarial_runner_cannot_switch_captured_default(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    policy = Policy()

    def swap(prompt, *, default_model):
        policy.default_model = "qa/new-default"
        policy.admission_revision = "rev-2"
        return "VERDICT: deliver\nREASON: ordinary.", "qa/new-default"

    result = screening.screen_message(
        owner, _item(), config_dir=tmp_path, policy=policy, runner=swap
    )
    assert result.disposition == "hold"
    assert _count(tmp_path, "screening_receipts") == 0


def test_owner_invalidation_during_runner_is_sanitized_hold(tmp_path) -> None:
    store, owner = _setup(tmp_path)

    def invalidate():
        Store(tmp_path).invalidate(owner)

    result = screening.screen_message(
        owner, _item(), config_dir=tmp_path, policy=Policy(), runner=Runner(fn=invalidate)
    )
    assert result.disposition == "hold" and result.hold_code == "owner_changed"
    assert _count(tmp_path, "screening_receipts") == 0


def test_pending_intent_never_admits(tmp_path) -> None:
    for default in (None, "qa/new-default", "qa/default-model"):
        path = tmp_path / f"pending-{default}"
        store, owner = _setup(path)
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
        policy = Policy(default_model=default)
        result = _run(owner, path, runner, policy=policy)
        assert result.disposition == "hold" and result.hold_code == "ack_recovery", default
        assert runner.calls == []  # no model cost; reconcile via the ACK path


def test_cache_fingerprint_covers_envelope_and_revision(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    first = Runner()
    assert _run(owner, tmp_path, first).disposition == "deliver"
    # Exact tuple reuses without another logical charge.
    second = Runner()
    reuse = _run(owner, tmp_path, second, item=_item(message_id="bmsg_second"))
    assert reuse.disposition == "deliver" and second.calls == []
    negative = Runner("VERDICT: hold\nREASON: context differs.")
    # Changed sender misses the cache: the negative runner RUNS and holds.
    changed = _item("bmsg_changed")
    changed["sender"]["account_id"] = "acc_attacker"
    out = _run(owner, tmp_path, negative, item=changed)
    assert out.disposition == "hold" and len(negative.calls) == 1
    # A fresh policy REVISION must come from the trusted publisher first
    # (a consumer cannot introduce authority); once published, the same-body
    # cache misses under the new revision and re-screens.
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-2",
        expected_epoch=1,
    )
    revised = Policy(admission_revision="rev-2")
    negative2 = Runner("VERDICT: hold\nREASON: context differs.")
    out = _run(owner, tmp_path, negative2, item=_item("bmsg_changed2"), policy=revised)
    assert out.disposition == "hold" and len(negative2.calls) == 1
    # A consumer presenting an UNPUBLISHED revision holds before any dispatch.
    stale = Policy(admission_revision="rev-3")
    never = Runner()
    out = _run(owner, tmp_path, never, item=_item("bmsg_changed3"), policy=stale)
    assert out.disposition == "hold" and out.hold_code == "policy_stale" and never.calls == []


def test_cached_reject_replays_without_rebill(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    first = Runner("VERDICT: bounce\nREASON: obvious spam.")
    assert _run(owner, tmp_path, first, item=_item("bmsg_b")).disposition == "bounce"
    spent = _count(tmp_path, "screening_attempts")
    second = Runner()
    out = _run(owner, tmp_path, second, item=_item("bmsg_b"))
    assert out.disposition == "bounce" and second.calls == []
    assert _count(tmp_path, "screening_attempts") == spent


# --------------------------- store integrity + binding ---------------------


def test_missing_v3_table_refuses_instead_of_resetting_spend(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    for _ in range(40):
        store.reserve_screening_attempt(owner, message_id="bmsg_1", fingerprint="rev-1|env")
    conn = sqlite3.connect(state_db_path(tmp_path))
    conn.execute("DROP TABLE screening_attempts")
    conn.commit()
    conn.close()
    runner = Runner()
    result = _run(owner, tmp_path, runner)
    assert result.disposition == "hold", "damaged store must refuse, never re-create empty"
    assert result.hold_code in ("schema_damaged", "retry_later")
    assert runner.calls == [] and _count(tmp_path, "screening_receipts") == 0


def test_v2_store_migrates_additively_preserving_owner_and_intent(tmp_path) -> None:
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
    conn = sqlite3.connect(state_db_path(tmp_path))
    for table in (
        "screening_attempts",
        "screening_receipts",
        "screening_holds",
        "screening_policy",
        "screening_breaker",
    ):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    migrated = Store(tmp_path)
    live = migrated.active_owner(credential_id=7)
    assert live == owner  # owner/generation preserved through the migration
    intent = migrated.find_pending_intent(owner, "bmsg_1", "read")
    assert intent is not None and intent["client_ref"] == "stable-ref"
    version = sqlite3.connect(state_db_path(tmp_path))
    try:
        stamp = version.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        version.close()
    assert stamp == "6"


def test_attempt_binding_and_write_once(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    attempt = store.reserve_screening_attempt(owner, message_id="bmsg_1", fingerprint="rev-1|env")
    receipt = ScreeningReceipt(
        "allow",
        "qa/default-model",
        SCREENING_VERSION_V1,
        SCREENING_TEMPLATE_V1_SHA256,
        hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        (),
    )
    epoch = store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
    )
    store.finish_screening(
        owner,
        "bmsg_1",
        attempt_id=attempt,
        fingerprint="rev-1|env",
        expected_epoch=epoch,
        receipt=receipt,
        envelope_digest="e" * 64,
    )
    # Exact replay is idempotent (crash recovery may repeat it).
    store.finish_screening(
        owner,
        "bmsg_1",
        attempt_id=attempt,
        fingerprint="rev-1|env",
        expected_epoch=epoch,
        receipt=receipt,
        envelope_digest="e" * 64,
    )
    # A second message cannot ride the same reserved ID.
    with pytest.raises(StoreRefused) as refusal:
        store.finish_screening(
            owner,
            "bmsg_other",
            attempt_id=attempt,
            fingerprint="rev-1|env",
            expected_epoch=epoch,
            hold_code="model_hold",
        )
    assert refusal.value.code in ("attempt_mismatch", "attempt_bound")
    # An allow cannot be overwritten by a reject on the same message.
    reject = ScreeningReceipt(
        "reject",
        "qa/default-model",
        SCREENING_VERSION_V1,
        SCREENING_TEMPLATE_V1_SHA256,
        hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        ("spam",),
    )
    with pytest.raises(StoreRefused) as refusal:
        store.finish_screening(
            owner,
            "bmsg_1",
            attempt_id=attempt,
            fingerprint="rev-1|env",
            expected_epoch=epoch,
            receipt=reject,
            envelope_digest="e" * 64,
        )
    assert refusal.value.code == "write_once_conflict"
    # An unreserved attempt id cannot fabricate an outcome.
    with pytest.raises(StoreRefused) as refusal:
        store.finish_screening(
            owner,
            "bmsg_1",
            attempt_id="not-reserved",
            fingerprint="rev-1|env",
            expected_epoch=epoch,
            hold_code="model_hold",
        )
    assert refusal.value.code == "unknown_attempt"


def test_policy_epoch_fences_commit(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    epoch = store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
    )
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/new-default",
        revision="rev-2",
    )
    attempt = store.reserve_screening_attempt(owner, message_id="bmsg_1", fingerprint="rev-1|x")
    with pytest.raises(StoreRefused) as refusal:
        store.finish_screening(
            owner,
            "bmsg_1",
            attempt_id=attempt,
            fingerprint="rev-1|x",
            expected_epoch=epoch,
            hold_code="model_hold",
        )
    assert refusal.value.code == "policy_stale"


# --------------------------- budgets + holds -------------------------------


def test_budget_boundaries_aggregation_and_kinds(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    for _ in range(40):
        store.reserve_screening_attempt(owner, message_id="m", fingerprint="f")
    with pytest.raises(StoreRefused) as refusal:
        store.reserve_screening_attempt(owner, message_id="m", fingerprint="f")
    assert refusal.value.code == "budget_exhausted"
    # The HTTP bucket is separate and independently bounded.
    for _ in range(40):
        store.reserve_http_attempt(owner, message_id="m")
    with pytest.raises(StoreRefused):
        store.reserve_http_attempt(owner, message_id="m")
    # A second namespace of the SAME account shares both buckets.
    other = _owner(tenant="tnt_2", endpoint="bep_2")
    store.commit_enrollment(other, OwnerReceipt("bridge_consent_v1", "ref-2"))
    with pytest.raises(StoreRefused) as refusal:
        store.reserve_http_attempt(other, message_id="m")
    assert refusal.value.code == "budget_exhausted"
    # A different account is separate.
    third = _owner(account="acc_other", endpoint="bep_9")
    store.commit_enrollment(third, OwnerReceipt("bridge_consent_v1", "ref-3"))
    assert store.reserve_screening_attempt(third, message_id="m", fingerprint="f")


def test_no_dispatch_calls_do_not_burn_budget(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    for _ in range(40):
        out = screening.screen_message(
            owner, _item(), config_dir=tmp_path, policy=Policy(), runner=None
        )
        assert out.disposition == "hold"
    assert _count(tmp_path, "screening_attempts") == 0, "known no-dispatch must not spend"
    runner = Runner()
    out = _run(owner, tmp_path, runner)
    assert out.disposition == "deliver" and len(runner.calls) == 1


def test_early_holds_are_durable(tmp_path) -> None:
    cases = ("no-default", "template-mismatch", "budget-exhausted")
    for kind in cases:
        path = tmp_path / f"early-{kind}"
        store, owner = _setup(path)
        policy = Policy()
        if kind == "no-default":
            policy.default_model = None
        elif kind == "template-mismatch":
            policy.template_sha256 = "0" * 64
        else:
            for _ in range(40):
                store.reserve_screening_attempt(owner, message_id="m", fingerprint="f")
        result = _run(owner, path, Runner(), policy=policy)
        assert result.disposition == "hold", kind
        assert store.screening_hold(owner, "bmsg_1") is not None, kind


def test_hold_survives_inbox_refresh(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    out = _run(owner, tmp_path, Runner("VERDICT: hold\nREASON: uncertain."))
    assert out.hold_code == "model_hold"
    from local_operator.team_bridge.types import MessageRef

    legacy_ref = MessageRef(
        message_id="bmsg_1",
        thread_id="bth_1",
        tenant_id="tnt_1",
        message_type="note",
        body_hash=hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        seq=1,
        origin="user",
        delivery_state="unread",
        state_at="t",
        sender_account_id="acc_sender1",
        sender_endpoint_id="bep_2",
        recipient_account_id="acc_a",
        recipient_endpoint_id="bep_1",
        request_state=None,
        request_expires_at=None,
        request_ref=None,
    ).validated()
    store.merge_page(owner, [legacy_ref], None)
    hold = store.screening_hold(owner, "bmsg_1")
    assert hold is not None and hold["hold_code"] == "model_hold"


def test_v2_store_migrates_additively_preserving_owner_and_tuple(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    before_owner = store.active_owner()
    receipt = ScreeningReceipt(
        "allow",
        "qa/default-model",
        SCREENING_VERSION_V1,
        SCREENING_TEMPLATE_V1_SHA256,
        hashlib.sha256(CANARY_BODY.encode()).hexdigest(),
        (),
    )
    store.record_ack_intent(owner, "bmsg_1", "read", receipt, "legacy-stable-ref")
    # Reshape the store to the H2 V2 layout: core tables only, stamp 2.
    conn = sqlite3.connect(state_db_path(tmp_path))
    for table in (
        "screening_attempts",
        "screening_receipts",
        "screening_holds",
        "screening_policy",
        "screening_breaker",
    ):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    # Reopening MIGRATES additively: owner + immutable tuple survive untouched,
    # and the legacy receipt (no new fingerprint fields) stays UNKNOWN to the
    # cache — never mistaken for a current admission.
    fresh = Store(tmp_path)
    resolved = fresh.active_owner()
    assert resolved == before_owner
    legacy = fresh.find_pending_intent(owner, "bmsg_1", "read")
    assert legacy is not None and legacy["client_ref"] == "legacy-stable-ref"
    assert (
        fresh.find_screening_receipt(
            owner,
            body_sha256=receipt.body_sha256,
            template_sha256=SCREENING_TEMPLATE_V1_SHA256,
            model="qa/default-model",
            admission_revision="rev-1",
            envelope_digest="e" * 64,
        )
        is None
    )
    conn = sqlite3.connect(state_db_path(tmp_path))
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone() == (
        "6",
    )


def test_allow_receipt_with_categories_is_corrupt_not_cached(tmp_path) -> None:
    """R6 static-cache adjunct: an allow verdict carrying rejection
    categories is DAMAGED metadata — consume refuses (proof_corrupt), no
    silent coalescing to [] and no delivery on corrupted proof."""

    import sqlite3 as _sqlite3

    from local_operator.team_bridge.store import StoreRefused, _owner_key

    store, owner = _setup(tmp_path)
    epoch = store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
    )
    conn = _sqlite3.connect(state_db_path(tmp_path))
    try:
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _owner_key(owner),
                "bmsg_c",
                "b" * 64,
                SCREENING_TEMPLATE_V1_SHA256,
                SCREENING_VERSION_V1,
                "qa/default-model",
                "allow",
                '["prompt_injection"]',
                "e" * 64,
                "rev-1",
                epoch,
                owner.enrollment_generation,
                0.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as err:
        store.consume_screening_receipt(
            owner,
            body_sha256="b" * 64,
            template_sha256=SCREENING_TEMPLATE_V1_SHA256,
            model="qa/default-model",
            admission_revision="rev-1",
            envelope_digest="e" * 64,
        )
    assert err.value.code == "proof_corrupt"


def test_loop_lock_table_is_loop_owned_not_weak_key_retained(tmp_path) -> None:
    """R6/Q15: the lock table lives ON the loop — the weak-key dictionary
    path (whose value strongly references its own key, so a finished loop
    can never be collected) is not taken for ordinary loops."""

    import asyncio as _asyncio

    from local_operator.team_bridge import screening as _screening

    before = len(_screening._ACCOUNT_LOCKS)

    async def use() -> None:
        _screening._account_lock(tmp_path, _owner())
        loop = _asyncio.get_running_loop()
        assert getattr(loop, "_team_bridge_account_locks", None) is not None

    _asyncio.run(use())
    # No weak-key entry was created: nothing retains the finished loop.
    assert len(_screening._ACCOUNT_LOCKS) == before


def test_empty_categories_sql_text_is_corrupt_not_cached(tmp_path) -> None:
    """R7/Q2: matched proof whose categories column is EMPTY SQL TEXT is
    damaged — consume refuses (proof_corrupt), never coalesces it to []."""

    import sqlite3 as _sqlite3

    from local_operator.team_bridge.store import StoreRefused, _owner_key

    store, owner = _setup(tmp_path)
    epoch = store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
    )
    conn = _sqlite3.connect(state_db_path(tmp_path))
    try:
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _owner_key(owner),
                "bmsg_e",
                "b" * 64,
                SCREENING_TEMPLATE_V1_SHA256,
                SCREENING_VERSION_V1,
                "qa/default-model",
                "allow",
                "",
                "e" * 64,
                "rev-1",
                epoch,
                owner.enrollment_generation,
                0.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as err:
        store.consume_screening_receipt(
            owner,
            body_sha256="b" * 64,
            template_sha256=SCREENING_TEMPLATE_V1_SHA256,
            model="qa/default-model",
            admission_revision="rev-1",
            envelope_digest="e" * 64,
        )
    assert err.value.code == "proof_corrupt"


def test_unverified_status_mappings_are_never_authority(tmp_path) -> None:
    """R9/Q3-4: a caller-supplied mapping — even one carrying the exact
    public owner-frame digest — cannot certify its provenance (QA forged
    such a negative into a DELIVER), and a minted capture whose owner frame,
    selector or freshness no longer holds is refused too. Nothing publishes."""

    import asyncio as _asyncio
    import time as _time

    from local_operator.team_bridge import scheduler as _scheduler
    from local_operator.team_bridge.scheduler import _CAPTURE_TOKEN, AcceptedCapture
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    (tmp_path / "config.yml").write_text("values:\n  hosting: openai\n  model_name: gpt-5.4\n")

    async def drive(capture):
        await _scheduler.run_screening_cycle(
            config_dir=tmp_path,
            owner=owner,
            items=[],
            auth_store=None,
            accepted_capture=capture,
        )

    forged = {
        "screening_version": SCREENING_VERSION_V1,
        "template_sha256": SCREENING_TEMPLATE_V1_SHA256,
        "owner_frame": _scheduler._owner_frame_digest(owner),
    }
    with pytest.raises(StoreRefused) as err:
        _asyncio.run(drive(forged))
    assert err.value.code == "policy_stale"

    wrong_frame = AcceptedCapture(
        _CAPTURE_TOKEN,
        version=SCREENING_VERSION_V1,
        template_hash=SCREENING_TEMPLATE_V1_SHA256,
        owner_frame="f" * 64,
        captured_at=_time.time(),
    )
    with pytest.raises(StoreRefused) as err2:
        _asyncio.run(drive(wrong_frame))
    assert err2.value.code == "policy_stale"

    stale = AcceptedCapture(
        _CAPTURE_TOKEN,
        version=SCREENING_VERSION_V1,
        template_hash=SCREENING_TEMPLATE_V1_SHA256,
        owner_frame=_scheduler._owner_frame_digest(owner),
        captured_at=_time.time() - 3600.0,
    )
    with pytest.raises(StoreRefused) as err3:
        _asyncio.run(drive(stale))
    assert err3.value.code == "policy_stale"

    # The minting guard itself: no external token may construct a capture.
    with pytest.raises(ValueError):
        AcceptedCapture(
            object(),
            version=SCREENING_VERSION_V1,
            template_hash=SCREENING_TEMPLATE_V1_SHA256,
            owner_frame="x",
            captured_at=_time.time(),
        )
    assert store.current_screening_policy(owner) is None  # nothing published


def test_status_law_normalizer_positive_and_refusal_arms(tmp_path) -> None:
    """R9/Q3-1: the FROZEN server law — nested per-tenant endpoint, exactly
    one owner-tenant row, eligibility, active state, local-only template
    authority, bounded text acceptance and committed-consent currency — is
    enforced; every arm refuses typed and a lawful DTO normalizes."""

    import copy as _copy

    from local_operator.team_bridge.client import SCREENING_TEMPLATE_V1
    from local_operator.team_bridge.scheduler import _normalize_accepted_status
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    base = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "endpoint", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge_guidance_v1", "text": "Guidance body."},
        "screening": {"version": SCREENING_VERSION_V1, "template": SCREENING_TEMPLATE_V1},
        "consent": {"version": "bridge_consent_v1", "text": "Consent body."},
    }
    law = _normalize_accepted_status(base, owner, store)
    assert law.version == SCREENING_VERSION_V1
    assert law.template_hash == SCREENING_TEMPLATE_V1_SHA256

    def refuse(mutate) -> str:
        arm = _copy.deepcopy(base)
        mutate(arm)
        with pytest.raises(StoreRefused) as caught:
            _normalize_accepted_status(arm, owner, store)
        assert caught.value.code == "policy_stale"
        return str(caught.value)

    refuse(lambda a: a.__setitem__("api_version", "9"))
    refuse(lambda a: a.__setitem__("available", False))
    refuse(lambda a: a.__setitem__("tenants", []))
    refuse(lambda a: a["tenants"].append(_copy.deepcopy(a["tenants"][0])))
    refuse(lambda a: a["tenants"][0].__setitem__("eligible", False))
    refuse(lambda a: a["tenants"][0].__setitem__("endpoint", None))
    refuse(lambda a: a["tenants"][0]["endpoint"].__setitem__("id", "bep_other"))
    refuse(lambda a: a["tenants"][0]["endpoint"].__setitem__("state", "paused"))
    refuse(lambda a: a.pop("screening"))
    refuse(lambda a: a["screening"].__setitem__("template", "TEMPLATE"))
    refuse(lambda a: a["screening"].__setitem__("version", "bridge-screen-v2"))
    refuse(lambda a: a.pop("guidance"))
    refuse(lambda a: a["guidance"].__setitem__("text", "x" * 3000))
    refuse(lambda a: a.pop("consent"))
    # Served consent moved on while the committed receipt stayed behind: the
    # served version alone is NOT acceptance (currency cannot be established).
    message = refuse(lambda a: a["consent"].__setitem__("version", "bridge_consent_v2"))
    assert "currency" in message or "current" in message


def test_first_bind_refuses_mutated_job_fields(tmp_path) -> None:
    """R8/Q3: the input proof registered at reservation pins provider/model/
    template/tenant/approved/epoch; a DispatchJob whose field was mutated
    between reservation and bind refuses BEFORE any I/O."""

    import dataclasses as _dataclasses

    from local_operator.team_bridge.model_boundary import AdmissionJob
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": "b" * 64,
        "template": "t" * 64,
        "version": "bridge-screen-v1",
        "provider": "openai",
        "model": "qa/default-model",
        "approved": ["qa/default-model"],
        "envelope": "e" * 64,
        "request": "r" * 64,
        "display": "d" * 64,
        "revision": "rev-1",
        "epoch": 1,
        "message": "bmsg_1",
        "tenant": "tnt_1",
        "logical": "rev-1|b|e",
    }
    attempt = store.reserve_screening_attempt(
        owner, message_id="bmsg_1", fingerprint="rev-1|b|e", input_proof=proof
    )
    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256="b" * 64,
        template_sha256="t" * 64,
        template_version="bridge-screen-v1",
        provider="openai",
        model="qa/default-model",
        approved=("qa/default-model",),
        envelope_digest="e" * 64,
        request_digest="r" * 64,
        display_digest="d" * 64,
        revision="rev-1",
        epoch=1,
        turn="cap",
    )
    mutated = _dataclasses.replace(job, provider="anthropic")
    with pytest.raises(StoreRefused) as err:
        store.bind_attempt_fingerprint(
            owner, attempt_id=attempt, fingerprint=mutated.digest, job=mutated
        )
    assert err.value.code == "attempt_mismatch"
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)


def test_same_version_different_hash_publisher_refuses(tmp_path) -> None:
    """R8/Q3: a native publication of the SAME version with a DIFFERENT
    template hash is a typed refusal and the original row is retained."""

    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    epoch = store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="t" * 64,
        default_model="qa/default-model",
        revision="rev-1",
    )
    with pytest.raises(StoreRefused) as err:
        store.publish_screening_policy(
            owner,
            version="bridge-screen-v1",
            template_sha256="s" * 64,
            default_model="qa/default-model",
            revision="rev-2",
        )
    assert err.value.code == "template_hash_conflict"
    row = store.current_screening_policy(owner)
    assert row is not None and row["template_sha256"] == "t" * 64
    assert int(row["epoch"]) == epoch


def test_expired_turn_cannot_admit_receipt(tmp_path) -> None:
    """R8 static adjunct: a turn whose lease EXPIRED (no successor) must not
    admit a receipt from inside the commit transaction."""

    import sqlite3 as _sqlite3
    import time as _time

    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    turn = _claim_turn(store, owner)
    conn = _sqlite3.connect(state_db_path(tmp_path))
    try:
        conn.execute(
            "UPDATE screening_schedule SET lease_expires = ? WHERE owner_key = ?",
            (_time.time() - 5.0, store._account_scope(owner)),
        )
        conn.commit()
    finally:
        conn.close()
    attempt = store.reserve_screening_attempt(owner, message_id="bmsg_1", fingerprint=None)
    with pytest.raises(StoreRefused) as err:
        store.finish_screening(
            owner,
            "bmsg_1",
            attempt_id=attempt,
            fingerprint="unused",
            expected_epoch=1,
            hold_code="model_hold",
            turn=turn,
        )
    assert err.value.code == "turn_lost"


def test_owner_consent_receipt_reads_committed_rows_only(tmp_path) -> None:
    """R9/Q3-2: consent acceptance is the COMMITTED enrollment receipt read
    under the exact live owner; a missing row or a non-live generation is
    None, never reconstructed from served text."""

    import dataclasses as _dataclasses

    store, owner = _setup(tmp_path)
    receipt = store.owner_consent_receipt(owner)
    assert receipt == {"consent_version": "bridge_consent_v1", "consent_ref": "ref-1"}
    other = _dataclasses.replace(owner, enrollment_generation=9)
    assert store.owner_consent_receipt(other) is None


def test_unverified_status_callback_never_mints_or_advances_floors(tmp_path) -> None:
    """R11/2 (F1): a public class shape is NOT provenance. An injected
    callback returning a lawful raw DTO — or a forged wrapper — must refuse
    BEFORE normalization, floor advance, client creation or spend; this is
    the guard that the earlier publicly-constructible marker class lacked.
    """

    import asyncio as _asyncio

    from local_operator.team_bridge import scheduler as _scheduler
    from local_operator.team_bridge.client import SCREENING_TEMPLATE_V1
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    (tmp_path / "config.yml").write_text("values:\n  hosting: openai\n  model_name: gpt-5.4\n")
    lawful = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "e", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge-guidance-v1", "text": "g"},
        "screening": {"version": SCREENING_VERSION_V1, "template": SCREENING_TEMPLATE_V1},
        "consent": {"version": "bridge_consent_v1", "text": "c"},
    }

    async def drive():
        await _scheduler.run_screening_cycle(
            config_dir=tmp_path,
            owner=owner,
            items=[],
            auth_store=None,
            status_fetcher=lambda: _asyncio.sleep(0, result=lawful),
        )

    with pytest.raises(StoreRefused) as err:
        _asyncio.run(drive())
    assert err.value.code == "policy_stale"
    # No floor was advanced and no capture-like authority was persisted.
    assert store.text_floor(owner, "guidance") is None
    assert store.text_floor(owner, "consent") is None


def test_floors_are_atomic_account_scoped_and_durable(tmp_path) -> None:
    """R11/3 (F1): floors advance as ONE accepted set, are keyed by account+
    hub (not tenant), refuse cross-organization rollback, are refused
    atomically when any candidate is stale, and survive a Store re-open."""

    import dataclasses as _dataclasses

    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    assert store.advance_text_floors is not None
    store.advance_text_floors(
        owner,
        [("guidance", "bridge-guidance-v2", "a" * 64), ("consent", "bridge_consent_v2", "b" * 64)],
    )
    # a stale candidate in the SAME set refuses ALL writes (atomicity)
    with pytest.raises(StoreRefused):
        store.advance_text_floors(
            owner,
            [
                ("guidance", "bridge-guidance-v3", "c" * 64),
                ("consent", "bridge_consent_v1", "d" * 64),
            ],
        )
    assert store.text_floor(owner, "guidance") == ("bridge-guidance-v2", "a" * 64)
    assert store.text_floor(owner, "consent") == ("bridge_consent_v2", "b" * 64)
    # same-version different bytes refuses; identical replay passes
    with pytest.raises(StoreRefused):
        store.advance_text_floors(owner, [("guidance", "bridge-guidance-v2", "z" * 64)])
    store.advance_text_floors(owner, [("guidance", "bridge-guidance-v2", "a" * 64)])
    # cross-organization on the SAME account cannot roll back the account floor
    other = _dataclasses.replace(owner, tenant_id="tnt_2", endpoint_id="bep_2")
    store.commit_enrollment(other, _dataclasses.replace(OwnerReceipt("bridge_consent_v2", "ref-2")))
    with pytest.raises(StoreRefused):
        store.advance_text_floors(other, [("guidance", "bridge-guidance-v1", "q" * 64)])
    # durability across a fresh Store instance (restart) — the floor persists
    store2 = Store(tmp_path)
    assert store2.text_floor(owner, "guidance") == ("bridge-guidance-v2", "a" * 64)
    with pytest.raises(StoreRefused):
        store2.advance_text_floors(owner, [("guidance", "bridge-guidance-v1", "q" * 64)])


def test_finish_refuses_field_mutation_with_matching_revision(tmp_path) -> None:
    """R12/F1+F2: every retained field is re-derived inside the finish
    transaction. With the MATCHING admission revision (so the revision
    comparison cannot be what refuses), mutating any registered non-body
    field — model, provider, request, display, logical reference, or an
    approved-map edit that keeps membership — must refuse, commit no
    receipt and leave the spent counts exactly as charged. On the
    pre-fix revision this test observes DELIVER (receipt committed), so it
    discriminates the defect it names.
    """

    import sqlite3 as _sqlite3

    from local_operator.team_bridge.model_boundary import AdmissionJob
    from local_operator.team_bridge.screening import _identities_digest
    from local_operator.team_bridge.store import StoreRefused, state_db_path
    from local_operator.team_bridge.types import ScreeningReceipt

    store, owner = _setup(tmp_path)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    body = "b" * 64
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": body,
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
        "logical": (f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default'}))}"),
    }
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint=f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default'}))}",
        input_proof=proof,
    )
    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256=body,
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
    )
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=job, turn=turn)
    db = state_db_path(tmp_path)
    receipt = ScreeningReceipt(
        verdict="allow",
        model="qa-default",
        prompt_version="bridge-screen-v1",
        template_sha256="ab" * 32,
        body_sha256=body,
        categories=(),
    ).validated()
    for field, replacement, expected_code in (
        ("model", "foreign-served-model", "attempt_mismatch"),
        ("provider", "foreign-provider", "attempt_mismatch"),
        ("request", "f" * 64, "attempt_mismatch"),
        ("display", "9" * 64, "attempt_mismatch"),
        ("logical", "rev-1|f|e", "proof_corrupt"),
        ("approved", ["qa-default", "sneaked-in"], "attempt_mismatch"),
    ):
        with _sqlite3.connect(db) as conn:
            raw = conn.execute(
                "SELECT fingerprint FROM screening_attempts WHERE attempt_id = ? "
                "AND kind = 'logical'",
                (attempt,),
            ).fetchone()[0]
            if field == "approved":
                mutated = raw.replace(
                    '"approved":["qa-default"]', '"approved":["qa-default","sneaked-in"]'
                )
            else:
                old_value = '"%s":"%s"' % (field, proof[field])
                new_value = '"%s":%s' % (
                    field,
                    (
                        _sqlite3.Binary and repr(replacement)
                        if isinstance(replacement, list)
                        else '"%s"' % replacement
                    ),
                )
                if isinstance(replacement, list):
                    new_value = '"approved":["qa-default","sneaked-in"]'
                mutated = raw.replace(old_value, new_value)
            assert mutated != raw, f"arm {field} did not mutate the retained record"
            conn.execute(
                "UPDATE screening_attempts SET fingerprint = ? WHERE attempt_id = ? "
                "AND kind = 'logical'",
                (mutated, attempt),
            )
        with _sqlite3.connect(db) as conn:
            spent_before = conn.execute(
                "SELECT COUNT(*) FROM screening_attempts WHERE kind IN ('physical', 'http')"
            ).fetchone()[0]
        with pytest.raises(StoreRefused) as err:
            store.finish_screening(
                owner,
                "bmsg_1",
                attempt_id=attempt,
                fingerprint=job.digest,
                expected_epoch=1,
                receipt=receipt,
                envelope_digest="e" * 64,
                admission_revision="rev-1",
            )
        assert err.value.code == expected_code, (field, err.value.code)
        assert "re-derive" in str(err.value), (field, str(err.value))
        with _sqlite3.connect(db) as conn:
            receipts = conn.execute("SELECT COUNT(*) FROM screening_receipts").fetchone()[0]
            spent_after = conn.execute(
                "SELECT COUNT(*) FROM screening_attempts WHERE kind IN ('physical', 'http')"
            ).fetchone()[0]
        assert receipts == 0
        assert spent_after == spent_before  # spent counts stay exactly as charged
        # restore the exact record for the next arm
        with _sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE screening_attempts SET fingerprint = ? WHERE attempt_id = ? "
                "AND kind = 'logical'",
                (
                    "job:"
                    + job.digest
                    + "|proof:"
                    + __import__("json").dumps(proof, separators=(",", ":")),
                    attempt,
                ),
            )


def test_first_bind_refuses_zero_digest(tmp_path) -> None:
    """R10/B preserved: an all-zero digest on a real minted job refuses
    BEFORE the bind — no row mutation, no physical charge."""

    import sqlite3 as _sqlite3

    from local_operator.team_bridge.model_boundary import AdmissionJob
    from local_operator.team_bridge.store import StoreRefused, state_db_path
    from local_operator.team_bridge.types import OwnerSnapshot

    owner = OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )
    store, owner = _setup(tmp_path, owner)
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
        fingerprint="rev-1|b|e",
        input_proof={
            "owner": "1|acc_1|tnt_1|bep_1|https://api.radienthq.com/v1|1",
            "message": "bmsg_1",
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
            "tenant": "tnt_1",
            "logical": "rev-1|b|e",
        },
    )
    job = AdmissionJob(
        owner_generation=1,
        credential_id=1,
        account_id="acc_1",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        hub_url="https://api.radienthq.com/v1",
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
    )
    with pytest.raises(StoreRefused) as err:
        store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint="0" * 64, job=job)
    assert err.value.code == "attempt_mismatch"
    db = state_db_path(tmp_path)
    with _sqlite3.connect(db) as conn:
        raw = conn.execute(
            "SELECT fingerprint FROM screening_attempts WHERE attempt_id = ? "
            "AND kind = 'logical'",
            (attempt,),
        ).fetchone()[0]
        physical = conn.execute(
            "SELECT COUNT(*) FROM screening_attempts WHERE attempt_id = ? " "AND kind = 'physical'",
            (attempt,),
        ).fetchone()[0]
    assert not raw.startswith("job:")
    assert physical == 0


def test_bound_registration_corruption_refused_at_reservation(tmp_path) -> None:
    """R10/B preserved at the native boundary: a bound registration
    corrupted (null proof / changed body / extra key) refuses the PHYSICAL
    reservation — no charge, no physical row."""

    import json as _json
    import sqlite3 as _sqlite3

    from local_operator.team_bridge.model_boundary import AdmissionJob
    from local_operator.team_bridge.screening import _identities_digest
    from local_operator.team_bridge.store import StoreRefused, state_db_path

    store, owner = _setup(tmp_path)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    body = "b" * 64
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": body,
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
        "logical": (f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default'}))}"),
    }
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint=f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default', 'qa/alias'}))}",
        input_proof=proof,
    )
    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256=body,
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
    )
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    db = state_db_path(tmp_path)
    exact = "job:" + job.digest + "|proof:" + _json.dumps(proof, separators=(",", ":"))
    tampered = [
        "job:" + job.digest + "|proof:null",
        exact.replace('"body":"%s"' % body, '"body":"%s"' % ("c" * 64)),
        exact.replace('"revision":"rev-1"', '"revision":"rev-1","extra":"x"'),
    ]
    for raw in tampered:
        with _sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE screening_attempts SET fingerprint = ? WHERE attempt_id = ? "
                "AND kind = 'logical'",
                (raw, attempt),
            )
        with pytest.raises(StoreRefused):
            store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=job, turn=turn)
        with _sqlite3.connect(db) as conn:
            physical = conn.execute(
                "SELECT COUNT(*) FROM screening_attempts WHERE attempt_id = ? "
                "AND kind = 'physical'",
                (attempt,),
            ).fetchone()[0]
        assert physical == 0


def test_alias_served_model_still_commits(tmp_path) -> None:
    """F1 keeps the alias rule: a served identity resolved through the
    approved map (a legitimate alias, not the default string) still
    commits its receipt."""

    from local_operator.team_bridge.model_boundary import AdmissionJob
    from local_operator.team_bridge.screening import _identities_digest
    from local_operator.team_bridge.types import ScreeningReceipt

    store, owner = _setup(tmp_path)
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    turn = _claim_turn(store, owner)
    body = "b" * 64
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": body,
        "template": "ab" * 32,
        "version": "bridge-screen-v1",
        "provider": "openai",
        "model": "qa-default",
        "approved": ["qa-default", "qa/alias"],
        "envelope": "e" * 64,
        "request": "r" * 64,
        "display": "d" * 64,
        "revision": "rev-1",
        "epoch": 1,
        "message": "bmsg_1",
        "tenant": owner.tenant_id,
        "logical": (
            f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default', 'qa/alias'}))}"
        ),
    }
    attempt = store.reserve_screening_attempt(
        owner,
        message_id="bmsg_1",
        fingerprint=f"rev-1|{'e' * 64}|{_identities_digest(frozenset({'qa-default', 'qa/alias'}))}",
        input_proof=proof,
    )
    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256=body,
        template_sha256="ab" * 32,
        template_version="bridge-screen-v1",
        provider="openai",
        model="qa-default",
        approved=("qa-default", "qa/alias"),
        envelope_digest="e" * 64,
        request_digest="r" * 64,
        display_digest="d" * 64,
        revision="rev-1",
        epoch=1,
        turn=turn,
    )
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    store.reserve_screening_dispatch(owner, message_id="bmsg_1", job=job, turn=turn)
    receipt = ScreeningReceipt(
        verdict="allow",
        model="qa/alias",
        prompt_version="bridge-screen-v1",
        template_sha256="ab" * 32,
        body_sha256=body,
        categories=(),
    ).validated()
    store.finish_screening(
        owner,
        "bmsg_1",
        attempt_id=attempt,
        fingerprint=job.digest,
        expected_epoch=1,
        receipt=receipt,
        envelope_digest="e" * 64,
        admission_revision="rev-1",
    )
    committed = store.find_screening_receipt(
        owner,
        body_sha256=body,
        template_sha256="ab" * 32,
        model="qa/alias",
        admission_revision="rev-1",
        envelope_digest="e" * 64,
    )
    assert committed is not None


def test_legacy_floor_row_is_read_and_reconciled(tmp_path) -> None:
    """R12/O3: a pre-scope-change legacy floor row is still READ (the
    conservative fallback) and an advance migrates to the account-scope key
    without destroying the legacy row."""

    import sqlite3 as _sqlite3
    import time as _time

    from local_operator.team_bridge import store as _store_mod
    from local_operator.team_bridge.store import state_db_path

    store, owner = _setup(tmp_path)
    legacy_key = _store_mod._owner_key(owner)
    db = state_db_path(tmp_path)
    with _sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO text_floors "
            "(owner_key, hub, kind, version, sha256, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                legacy_key,
                owner.canonical_base_url,
                "guidance",
                "bridge-guidance-v1",
                "a" * 64,
                _time.time(),
            ),
        )
    assert store.text_floor(owner, "guidance") == ("bridge-guidance-v1", "a" * 64)
    store.advance_text_floors(owner, [("guidance", "bridge-guidance-v2", "b" * 64)])
    assert store.text_floor(owner, "guidance") == ("bridge-guidance-v2", "b" * 64)
    with _sqlite3.connect(db) as conn:
        legacy = conn.execute(
            "SELECT COUNT(*) FROM text_floors WHERE owner_key = ?", (legacy_key,)
        ).fetchone()[0]
    assert legacy == 1  # the legacy row is preserved, never deleted or repaired


def test_accepted_capture_is_strictly_immutable(tmp_path) -> None:
    """R12/Q2: no production copy affordance — the supported capture is
    immutable, and the default copy machinery can only write through the
    same guard (AttributeError), so nothing supported can refresh the age
    provenance."""

    import copy as _copy
    import time as _time

    from local_operator.team_bridge import scheduler as _scheduler

    capture = _scheduler.AcceptedCapture(
        _scheduler._CAPTURE_TOKEN,
        version="bridge-screen-v1",
        template_hash="ab" * 32,
        owner_frame="f" * 64,
        captured_at=_time.time(),
        credential_id=1,
        account_id="acc_1",
        hub="https://api.radienthq.com/v1",
        owner_generation=1,
        consent_version="bridge_consent_v1",
        consent_ref="ref-1",
        consent_served_version="bridge_consent_v1",
        guidance_version="bridge-guidance-v1",
        guidance_sha="a" * 64,
        consent_sha="b" * 64,
        source="status",
    )
    with pytest.raises(AttributeError):
        _copy.copy(capture)
    with pytest.raises(AttributeError):
        capture.captured_at = _time.time()


def test_normalizer_refuses_consent_currency_without_floor_writes(tmp_path) -> None:
    """R12/O3: the normalizer no longer writes floors — a currency
    mismatch refuses in the law path and leaves BOTH floors unset."""

    from local_operator.team_bridge import scheduler as _scheduler
    from local_operator.team_bridge.client import SCREENING_TEMPLATE_V1
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    raw = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "e", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge-guidance-v1", "text": "g"},
        "screening": {"version": "bridge-screen-v1", "template": SCREENING_TEMPLATE_V1},
        "consent": {"version": "bridge_consent_v2", "text": "c"},
    }
    with pytest.raises(StoreRefused):
        _scheduler._normalize_accepted_status(raw, owner, store)
    assert store.text_floor(owner, "guidance") is None
    assert store.text_floor(owner, "consent") is None


def test_live_identity_refuses_logout_and_same_row_switch(tmp_path) -> None:
    """R12/O3: the reuse-time live-identity read (the supported seam the
    cycle calls before any cached DELIVER) refuses a logout and a same-row
    account switch, on the actual native AuthStore."""

    import asyncio as _asyncio
    import dataclasses as _dataclasses
    import json as _json
    import sqlite3 as _sqlite3
    import time as _time

    from local_operator.providers.auth_store import AuthStore
    from local_operator.team_bridge import scheduler as _scheduler
    from local_operator.team_bridge.store import StoreRefused

    store, owner0 = _setup(tmp_path)
    auth = AuthStore(db_path=tmp_path / "auth.db")
    expires = int((_time.time() + 3600) * 1000)  # AuthStore stores epoch MILLIS
    row = auth.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "synthetic-access",
            "refresh": "synthetic-refresh",
            "expires": expires,
            "account_id": owner0.account_id,
            "org_id": owner0.tenant_id,
            "api_endpoint": "https://api.radienthq.com/v1",
            "email": "q@example.test",
        },
    )
    owner = _dataclasses.replace(
        owner0,
        credential_id=row.id,
        canonical_base_url="https://api.radienthq.com/v1",
    )
    _asyncio.run(_scheduler._verify_live_identity(owner, auth, tmp_path))
    with _sqlite3.connect(tmp_path / "auth.db") as conn:
        data = _json.loads(
            conn.execute("SELECT data FROM auth_credentials WHERE id = ?", (row.id,)).fetchone()[0]
        )
        data["account_id"] = "acc_somewhere_else"
        conn.execute(
            "UPDATE auth_credentials SET data = ? WHERE id = ?",
            (_json.dumps(data), row.id),
        )
    with pytest.raises(StoreRefused):
        _asyncio.run(_scheduler._verify_live_identity(owner, auth, tmp_path))
    auth.delete_credential(row.id)
    with pytest.raises(StoreRefused):
        _asyncio.run(_scheduler._verify_live_identity(owner, auth, tmp_path))
