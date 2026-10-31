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
        {"model_hold", "model_refused", "reply_truncated", "parse_failed"}
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
