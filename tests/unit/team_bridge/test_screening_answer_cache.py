"""A screener's MODEL-ANSWER hold is cached: a retry must not re-draw it.

The draw is not deterministic (no temperature 0 is sent for the current model
families), so re-screening a held body on the retry schedule used to be "the
first draw that says deliver wins" — a persistent false negative eventually
admitted, and the receipt commit deleted the hold rows that would have shown it.
These cases pin the fix at the behaviour level: ONE physical send, the SAME
hold, no receipt and no admission, while a ROUTE failure still retries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from local_operator.team_bridge import screening
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V1_SHA256,
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_VERSION_V1,
    SCREENING_VERSION_V2,
)
from local_operator.team_bridge.store import state_db_path
from tests.unit.team_bridge.test_screening import (
    Policy,
    Runner,
    _count,
    _expire_breaker,
    _item,
    _run,
    _setup,
)

HOLD_REPLY = "VERDICT: hold\nREASON: cannot tell who this is."
DELIVER_REPLY = "VERDICT: deliver\nREASON: ordinary."
# Categories are inferred from the REASON vocabulary (no CATEGORIES line), and a
# reply with a third line is a parse failure — so a real reject is exactly two.
BOUNCE_REPLY = "VERDICT: bounce\nREASON: obvious spam."


def _cached_answers(tmp_path: Path) -> list[tuple[str, str]]:
    """The cached (hold_code, ...) rows, or [] on a store that has no table.

    A missing table is honest evidence of "nothing was cached" for the red run
    against a tree without this feature, so the behavioural assertions above
    each case stay the ones that fail first.
    """

    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT hold_code, model FROM screening_answer_holds ORDER BY hold_code"
            ).fetchall()
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def test_two_cycles_over_a_held_body_send_once_and_the_answer_parks(tmp_path) -> None:
    """(a) ONE physical send; the retry re-reads the answer and STAYS parked.

    The consequence is deliberate and has to be read with the fix: an ordinary
    message the model answers ``hold`` on is parked until the policy is
    republished, not eventually admitted. That is the trade against a re-draw —
    the alternative was a verdict that depended on how many times the retry
    schedule happened to run.
    """

    _store, owner = _setup(tmp_path)
    first = Runner(HOLD_REPLY)
    out = _run(owner, tmp_path, first, item=_item("held body text", "bmsg_held"))
    assert out.disposition == "hold" and out.hold_code == "model_hold"
    assert len(first.calls) == 1
    spent = _count(tmp_path, "screening_attempts")

    # The retry schedule comes around. This runner would ADMIT — on a re-draw it
    # would deliver, which is exactly the defect: a different sample changes the
    # verdict for a body that already answered.
    second = Runner(DELIVER_REPLY)
    again = _run(owner, tmp_path, second, item=_item("held body text", "bmsg_held"))
    assert again.disposition == "hold" and again.hold_code == "model_hold"
    assert second.calls == []  # no second physical send
    assert again.receipt is None  # no admission, nothing to ack
    assert _count(tmp_path, "screening_receipts") == 0
    assert _count(tmp_path, "screening_attempts") == spent  # no second charge
    # The durable per-message hold is still what the receiver status reads (and
    # the replay keeps the class of the failure the message was parked on).
    hold = _store.screening_hold(owner, "bmsg_held")
    assert hold is not None and hold["hold_code"] == "model_hold"
    assert ("model_hold", "qa/default-model") in _cached_answers(tmp_path)


def test_parse_failed_is_an_answer_and_is_cached_too(tmp_path) -> None:
    """A reply that cannot be parsed is a MODEL answer, not a route failure."""

    _store, owner = _setup(tmp_path)
    first = Runner("not a verdict at all")
    out = _run(owner, tmp_path, first, item=_item("bmsg_parse"))
    assert out.hold_code == "parse_failed" and len(first.calls) == 1
    second = Runner(DELIVER_REPLY)
    again = _run(owner, tmp_path, second, item=_item("bmsg_parse"))
    assert again.hold_code == "parse_failed" and second.calls == []


def test_route_failure_is_not_cached_and_still_retries(tmp_path) -> None:
    """(b) A provider outage must recover: its hold is never cached."""

    _store, owner = _setup(tmp_path)
    route = Runner(error=RuntimeError("vendor prose that must not be kept"))
    out = _run(owner, tmp_path, route, item=_item("bmsg_route"))
    assert out.disposition == "hold" and out.hold_code == "provider_error"
    assert len(route.calls) == 1
    assert _cached_answers(tmp_path) == []  # the outage left NO answer behind

    # The retry reaches the model for real, and only THAT answer is cached.
    _expire_breaker(tmp_path)
    answered = Runner(HOLD_REPLY)
    second = _run(owner, tmp_path, answered, item=_item("bmsg_route"))
    assert second.hold_code == "model_hold" and len(answered.calls) == 1
    assert ("model_hold", "qa/default-model") in _cached_answers(tmp_path)

    third = _run(owner, tmp_path, Runner(DELIVER_REPLY), item=_item("bmsg_route"))
    assert third.hold_code == "model_hold"
    assert _cached_answers(tmp_path) == [("model_hold", "qa/default-model")]


def test_republish_to_a_new_template_re_draws(tmp_path) -> None:
    """(c) A new template_sha256 is a NEW question: it must be asked again."""

    store, owner = _setup(tmp_path)
    first = Runner(HOLD_REPLY)
    assert _run(owner, tmp_path, first, item=_item("bmsg_re")).hold_code == "model_hold"
    assert len(first.calls) == 1

    # The publisher republishes the SAME body under the v2 template and a new
    # admission revision — the cache key carries both, so the answer cannot
    # outlive the policy that was asked of.
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        default_model="qa/default-model",
        revision="rev-2",
        expected_epoch=1,
    )
    republished = Policy(
        screening_version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        admission_revision="rev-2",
    )
    second = Runner(HOLD_REPLY)
    out = _run(owner, tmp_path, second, item=_item("bmsg_re"), policy=republished)
    assert out.hold_code == "model_hold"
    assert len(second.calls) == 1  # a SECOND send: the republish re-draws


def test_bounce_still_commits_a_reject_receipt_and_is_not_cached(tmp_path) -> None:
    """(d) A bounce is terminal and belongs in the receipt, not the answer cache."""

    _store, owner = _setup(tmp_path)
    first = Runner(BOUNCE_REPLY)
    out = _run(owner, tmp_path, first, item=_item("bmsg_bounce"))
    assert out.disposition == "bounce" and out.receipt is not None
    assert out.receipt.verdict == "reject"
    assert _count(tmp_path, "screening_receipts") == 1
    assert _cached_answers(tmp_path) == []  # a reject is not an answer hold

    # The verdict cache still replays the reject without another charge.
    second = Runner()
    again = _run(owner, tmp_path, second, item=_item("bmsg_bounce"))
    assert again.disposition == "bounce" and second.calls == []


def test_the_cache_key_is_content_equality_only(tmp_path) -> None:
    """(e) Key equality on digests: identity and content cannot be steered."""

    _store, owner = _setup(tmp_path)
    assert _run(owner, tmp_path, Runner(HOLD_REPLY), item=_item("body-k", "bmsg_k")).hold_code == (
        "model_hold"
    )

    # Same body, DIFFERENT message id: still a hit (the key is the content, not
    # the message), so a resend under a fresh id inherits the answer.
    same_body = Runner(DELIVER_REPLY)
    assert (
        _run(owner, tmp_path, same_body, item=_item("body-k", "bmsg_k_copy")).hold_code
        == "model_hold"
    )
    assert same_body.calls == []

    # The SAME message id carrying DIFFERENT bytes must MISS: an id cannot be
    # made to inherit an answer about content nobody screened.
    other_body = Runner(DELIVER_REPLY)
    out = _run(owner, tmp_path, other_body, item=_item("a different body", "bmsg_k"))
    assert len(other_body.calls) == 1  # screened on its own content
    assert out.disposition == "deliver"

    # The row holds digests and a closed code — never the body or a message id
    # that a sender could have chosen.
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(screening_answer_holds)").fetchall()
        }
    finally:
        conn.close()
    assert "message_id" not in columns and "body" not in columns
    assert "body_sha256" in columns and "routing" not in "".join(sorted(columns))
    assert screening._ANSWER_HOLD_CACHEABLE == frozenset(
        {
            "model_hold",
            "model_refused",
            "reply_truncated",
            "parse_failed",
            # A categoryless bounce is the model refusing the message too: it
            # explained instead of naming a category.
            "bounce_without_category",
        }
    )


def test_answer_cache_ignores_a_store_without_the_publishing_policy(tmp_path) -> None:
    """Fail-open: an unverifiable cache row is a MISS, never a replayed answer."""

    store, owner = _setup(tmp_path)
    assert _run(owner, tmp_path, Runner(HOLD_REPLY), item=_item("bmsg_p")).hold_code == (
        "model_hold"
    )
    # A row whose epoch no longer matches the published policy is unverifiable.
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        conn.execute("UPDATE screening_answer_holds SET policy_epoch = policy_epoch + 7")
        conn.commit()
    finally:
        conn.close()
    again = Runner(DELIVER_REPLY)
    out = _run(owner, tmp_path, again, item=_item("bmsg_p"))
    assert len(again.calls) == 1  # the stale row did not replay an answer
    assert out.disposition == "deliver"
    assert SCREENING_VERSION_V1 and SCREENING_TEMPLATE_V1_SHA256  # key material is the policy's


# --------------------------------------------------------------------------
# R1: a cached answer whose authority moved must be REPLACEABLE, never a silent
# cache disable. Each trigger below left the old row in place under write-once,
# so the read missed and the body re-drew on every cycle forever.
# --------------------------------------------------------------------------


def test_policy_rollback_replaces_the_cached_answer(tmp_path) -> None:
    """A rollback is a new epoch with old content: the row must be rewritten."""

    store, owner = _setup(tmp_path)
    body, mid = "rollback body", "bmsg_rb"
    assert _run(owner, tmp_path, Runner(HOLD_REPLY), item=_item(body, mid)).hold_code == (
        "model_hold"
    )

    # The publisher moves forward, then rolls BACK to the original content: a new
    # epoch serving the same template/model/revision the first answer was drawn
    # under.
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-2",
    )
    assert _run(
        owner,
        tmp_path,
        Runner(HOLD_REPLY),
        item=_item(body, mid),
        policy=Policy(admission_revision="rev-2"),
    ).hold_code == ("model_hold")
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
    )

    back = Runner(HOLD_REPLY)
    out = _run(owner, tmp_path, back, item=_item(body, mid))
    assert out.hold_code == "model_hold"
    assert len(back.calls) == 1  # the rolled-back epoch re-drew ONCE...
    after = Runner(DELIVER_REPLY)
    again = _run(owner, tmp_path, after, item=_item(body, mid))
    assert again.hold_code == "model_hold"
    assert after.calls == []  # ...and the replacement row serves the next pass


def test_approved_model_edit_epoch_bump_replaces_the_cached_answer(tmp_path) -> None:
    """An epoch bump with UNCHANGED content must not strand the old row."""

    store, owner = _setup(tmp_path)
    body, mid = "approved list body", "bmsg_ap"
    assert _run(owner, tmp_path, Runner(HOLD_REPLY), item=_item(body, mid)).hold_code == (
        "model_hold"
    )

    # ONLY the approved serving-model list changed: same version, template,
    # default model and revision — a real publication, so a real new epoch.
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        default_model="qa/default-model",
        revision="rev-1",
        approved_serving=[("openai", "qa/default-model"), ("openai", "qa/second")],
    )
    widened = Policy(approved_serving_identities=("qa/default-model", "qa/second"))
    redraw = Runner(HOLD_REPLY)
    assert _run(owner, tmp_path, redraw, item=_item(body, mid), policy=widened).hold_code == (
        "model_hold"
    )
    assert len(redraw.calls) == 1
    after = Runner(DELIVER_REPLY)
    assert _run(owner, tmp_path, after, item=_item(body, mid), policy=widened).hold_code == (
        "model_hold"
    )
    assert after.calls == []  # the row was replaced under the new epoch


def test_re_enrollment_replaces_the_cached_answer(tmp_path) -> None:
    """A new creator generation cannot serve the previous generation's row."""

    from local_operator.team_bridge.store import OwnerReceipt
    from local_operator.team_bridge.types import OwnerSnapshot

    store, owner = _setup(tmp_path)
    body, mid = "re-enrolled body", "bmsg_re"
    assert _run(owner, tmp_path, Runner(HOLD_REPLY), item=_item(body, mid)).hold_code == (
        "model_hold"
    )

    renewed = OwnerSnapshot(
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        canonical_base_url=owner.canonical_base_url,
        enrollment_generation=2,
    ).validated()
    store.commit_enrollment(renewed, OwnerReceipt("bridge_consent_v1", "ref-2"))

    redraw = Runner(HOLD_REPLY)
    assert _run(renewed, tmp_path, redraw, item=_item(body, mid)).hold_code == "model_hold"
    assert len(redraw.calls) == 1
    after = Runner(DELIVER_REPLY)
    assert _run(renewed, tmp_path, after, item=_item(body, mid)).hold_code == "model_hold"
    assert after.calls == []  # the previous generation's row is gone, not blocking


# --------------------------------------------------------------------------
# R2: only the model the key NAMES may be cached
# --------------------------------------------------------------------------


def test_a_fallback_hops_answer_is_never_cached_under_the_primary(tmp_path) -> None:
    """A route failure must not become a permanent answer via the fallback."""

    _store, owner = _setup(tmp_path)
    body, mid = "fallback body", "bmsg_fb"
    policy = Policy(approved_serving_identities=("qa/default-model", "qa/fallback-model"))
    # The primary was unreachable, so the fallback hop produced this answer.
    fallback = Runner(HOLD_REPLY, model="qa/fallback-model")
    out = _run(owner, tmp_path, fallback, item=_item(body, mid), policy=policy)
    assert out.hold_code == "model_hold" and len(fallback.calls) == 1
    assert _cached_answers(tmp_path) == []  # nothing remembered for the primary

    # The primary recovers. It must be ASKED again — a cached fallback answer
    # would have parked the message under an identity that never saw it.
    primary = Runner(DELIVER_REPLY)
    again = _run(owner, tmp_path, primary, item=_item(body, mid), policy=policy)
    assert len(primary.calls) == 1
    assert again.disposition == "deliver"


# --------------------------------------------------------------------------
# R3: a categoryless bounce is the model refusing, so it is an answer
# --------------------------------------------------------------------------


def test_categoryless_bounce_is_a_model_answer_and_is_cached(tmp_path) -> None:
    """The model said no without naming a category; the next draw must not admit."""

    _store, owner = _setup(tmp_path)
    body, mid = "categoryless body", "bmsg_cc"
    out = _run(
        owner, tmp_path, Runner("VERDICT: bounce\nREASON: it feels wrong."), item=_item(body, mid)
    )
    assert out.hold_code == "bounce_without_category"
    second = Runner(DELIVER_REPLY)
    again = _run(owner, tmp_path, second, item=_item(body, mid))
    assert again.hold_code == "bounce_without_category"
    assert second.calls == []  # no re-draw, so no chance to deliver
    assert _count(tmp_path, "screening_receipts") == 0


# --------------------------------------------------------------------------
# R4: a code fix clears a code artefact, without a republish
# --------------------------------------------------------------------------


def test_engine_version_bump_clears_a_cached_code_artefact(tmp_path, monkeypatch) -> None:
    """A parser/ceiling fix must invalidate the artefact it produced."""

    _store, owner = _setup(tmp_path)
    body, mid = "artefact body", "bmsg_eng"
    assert _run(owner, tmp_path, Runner("VERDICT: deliver  "), item=_item(body, mid)).hold_code == (
        "parse_failed"
    )
    assert _run(owner, tmp_path, Runner(DELIVER_REPLY), item=_item(body, mid)).hold_code == (
        "parse_failed"
    )
    # The ENGINE is fixed (a new stamp), and the old artefact must be re-drawn.
    monkeypatch.setattr(screening, "SCREENING_ENGINE_VERSION", "bridge-screen-engine-2")
    redraw = Runner(HOLD_REPLY)
    out = _run(owner, tmp_path, redraw, item=_item(body, mid))
    assert len(redraw.calls) == 1
    assert out.hold_code == "model_hold"  # the NEW answer, not the old artefact
    after = Runner(DELIVER_REPLY)
    assert _run(owner, tmp_path, after, item=_item(body, mid)).hold_code == "model_hold"
    assert after.calls == []  # and the new answer is cached under the new stamp


# --------------------------------------------------------------------------
# The key binds the attachment set and the message identity, not just the body
# --------------------------------------------------------------------------


def test_a_changed_attachment_set_re_draws(tmp_path) -> None:
    """The envelope digest binds the images, so a new image set is a new screen."""

    from tests.unit.team_bridge.test_attachments import ImageRunner, _image_item, _png

    _store, owner = _setup(tmp_path)
    first = ImageRunner(HOLD_REPLY)
    out = _run(owner, tmp_path, first, item=_image_item("look", _png((1, 2, 3))))
    assert out.hold_code == "model_hold" and len(first.calls) == 1
    second = ImageRunner(DELIVER_REPLY)
    # Same body and message id, a DIFFERENT image: the cached answer is about a
    # different image set, so it must not be reused.
    out2 = _run(owner, tmp_path, second, item=_image_item("look", _png((4, 5, 6))))
    assert len(second.calls) == 1
    assert out2.disposition == "deliver"
