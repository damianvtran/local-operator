"""The ONE text function: receipts, responses, timeout notices, refusal copy.

The design's rule is that no surface re-derives Q&A from these strings (design
``docs/design/ask-nonblocking.md`` §2.3): the model's turn and every card show
the same words, so the wording itself is a contract and is pinned here rather
than left to whoever writes the next renderer.
"""

from __future__ import annotations

from typing import Any

from local_operator.asks import render, store


def _record(**overrides) -> dict[str, Any]:
    record = {
        "ask_id": "a-7f3",
        "created_at": 1_700_000_000_000,
        "expires_at": 1_700_003_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": store.STATUS_OPEN,
        "questions": [
            {
                "id": "q0",
                "question": "Which database?",
                "options": [{"label": "staging"}, {"label": "prod"}],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            }
        ],
    }
    record.update(overrides)
    return record


# --- the receipt ------------------------------------------------------------


def test_the_receipt_names_the_ask_and_where_it_is_showing():
    text = render.receipt_text(_record(), "terminal")
    assert "a-7f3" in text
    assert "showing on terminal" in text
    assert "1 question(s)" in text


def test_the_receipt_says_it_is_not_consent():
    """Risk 1: a model that reads a receipt as approval runs the very command the
    ask was meant to authorise, so the warning is in capitals and in both arms."""
    for reach in ("terminal", None):
        text = render.receipt_text(_record(), reach)
        assert "A RECEIPT IS NOT CONSENT" in text
        assert "do not run anything the answer was meant to authorise" in text


def test_the_unreachable_receipt_never_claims_a_notification():
    text = render.receipt_text(_record(), None).lower()
    assert "nobody is attached" in text
    for overclaim in ("has been notified", "was notified", "has been told", "they were told"):
        assert overclaim not in text


def test_the_receipt_says_the_answer_arrives_as_a_turn_and_when_it_expires():
    text = render.receipt_text(_record(), "terminal")
    assert "ask response" in text
    assert "timeout notice" in text


# --- the response -----------------------------------------------------------


def test_an_answer_reuses_the_blocking_paths_own_report():
    """The model must not be able to tell the two paths apart in what it is told."""
    from local_operator.tools.builtin import _ask_report

    record = _record(status=store.STATUS_ANSWERED, answers={"q0": ["staging"]})
    expected = _ask_report(render._question_models(record["questions"]), {"q0": ["staging"]})
    assert render.response_text(record) == expected


def test_a_decline_reports_the_unanswered_text_not_a_denial():
    from local_operator.tools.builtin import ASK_UNANSWERED_TEXT

    record = _record(status=store.STATUS_DECLINED)
    assert render.response_text(record) == ASK_UNANSWERED_TEXT


def test_a_declined_secret_reports_the_secret_variant():
    from local_operator.tools.builtin import ASK_SECRET_UNANSWERED_TEXT

    record = _record(
        status=store.STATUS_DECLINED,
        questions=[
            {
                "id": "API_KEY",
                "question": "Paste it",
                "secret": True,
                "options": [],
                "multi": False,
                "persist": False,
                "recommended": None,
            }
        ],
    )
    assert render.response_text(record) == ASK_SECRET_UNANSWERED_TEXT


def test_a_late_answer_leads_with_the_fact_that_the_agent_moved_on():
    record = _record(status=store.STATUS_LATE, answers={"q0": ["staging"]})
    text = render.response_text(record)
    assert text.startswith("You already proceeded")
    assert "reconsider only if the answer changes your work" in text


# --- the timeout notice -----------------------------------------------------


def test_the_notice_reports_the_wait_and_tells_the_model_to_proceed():
    text = render.timeout_text(_record(), now_ms=1_700_003_600_000)
    assert text.startswith("[Ask timed out]")
    assert "1h" in text
    assert "Proceed without it" in text
    assert "if they answer later you will be told" in text
    # Never "the user denied": nobody denied anything, the window closed.
    assert "denied" not in text


def test_an_urgent_notice_adds_the_resolve_it_now_clause():
    text = render.timeout_text(_record(urgent=True), now_ms=1_700_003_600_000)
    assert "This ask was urgent" in text
    assert "subagent" in text


def test_a_secret_notice_names_the_key_and_never_quotes_the_prompt():
    """The question text may itself describe the credential, so the notice must
    not repeat it (design §2.5)."""
    record = _record(
        questions=[
            {
                "id": "API_KEY",
                "question": "Paste the sk-live-abc123 key",
                "secret": True,
                "options": [],
                "multi": False,
                "persist": False,
                "recommended": None,
            }
        ]
    )
    text = render.timeout_text(record, now_ms=1_700_003_600_000)
    assert "API_KEY" in text
    assert "sk-live-abc123" not in text
    assert "was not provided" in text


def test_a_long_question_is_clipped_in_the_notice():
    record = _record(
        questions=[
            {
                "id": "q0",
                "question": "x" * 500,
                "options": [],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            }
        ]
    )
    text = render.timeout_text(record, now_ms=1_700_003_600_000)
    assert "x" * 500 not in text
    assert "…" in text


# --- refusal copy -----------------------------------------------------------


def test_an_unknown_ask_is_its_own_refusal():
    assert "not in this session's queue" in render.refusal_copy(None)


def test_an_open_ask_is_accepted():
    assert render.refusal_copy(_record()) == ""


def test_a_timed_out_ask_is_accepted_and_folds_to_late():
    assert render.refusal_copy(_record(status=store.STATUS_TIMED_OUT)) == ""


def test_an_expired_ask_names_the_remedy():
    assert "expired" in render.refusal_copy(_record(status=store.STATUS_EXPIRED))


def test_a_decline_says_you_already_declined_this():
    assert "already declined" in render.refusal_copy(_record(status=store.STATUS_DECLINED))


def test_a_second_answer_names_the_surface_that_won():
    record = _record(status=store.STATUS_ANSWERED, answered_by={"surface": "desktop"})
    assert render.refusal_copy(record) == "already answered by desktop."


def test_the_span_helper_reads_like_a_person_wrote_it():
    assert render._span(45) == "45s"
    assert render._span(90) == "1m"
    assert render._span(3700) == "1h"
    assert render._span(2 * 86400) == "2d"
    assert render._span(0) == "a moment"


# --- the withdrawn state's sentence, and the withdraw op's own copy ----------


def test_a_withdrawn_ask_refuses_with_the_chat_route():
    """DESIGN §12's sentence, verbatim: a withdrawn ask's answer box hides
    everywhere (it is not outstanding), so a stale tap must be told the ONE
    route left — say it in chat, where the agent can record it."""
    assert render.refusal_copy(_record(status=store.STATUS_WITHDRAWN)) == (
        "the agent withdrew this question — if you have an answer, send it as a chat message."
    )


def test_a_withdrawn_refusal_keeps_the_state_table_byte_for_byte():
    """The §10 op-vs-state split (§12): the new state's row joins the STATE
    table, and the op's own sentences live outside it. A state-table caller
    still reads the declined sentence unchanged — the withdraw op does not
    rewrite rows other surfaces show."""
    assert (
        render.refusal_copy(_record(status=store.STATUS_DECLINED)) == "you already declined this."
    )


def test_the_withdraw_refusal_speaks_to_the_model_in_every_settled_state():
    """The op-level sentences answer the MODEL, so the state table's user voice
    ("you already declined this") — which would be false about who declined —
    is replaced per state. One sentence each, truthful in both directions."""
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_DECLINED), "moot")
        == "the user already declined this ask — there is nothing to withdraw."
    )
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_DISMISSED), "moot")
        == "the user already dismissed this ask — there is nothing to withdraw."
    )
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_ANSWERED), "moot")
        == "this ask already has the user's answer — it cannot be withdrawn."
    )
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_LATE), "moot")
        == "this ask already has the user's answer — it cannot be withdrawn."
    )
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_WITHDRAWN), "moot")
        == "this ask was already withdrawn."
    )
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_EXPIRED), "moot")
        == "this ask expired — there is nothing to withdraw."
    )


def test_the_chat_refusal_says_whether_anything_was_recorded():
    """``answered_in_chat``'s refusals carry the fact the model needs first:
    NOTHING was recorded, so the user's words still live only in the chat."""
    assert (
        render.withdraw_refusal(_record(status=store.STATUS_ANSWERED), "answered_in_chat")
        == "this ask already has the user's answer — the chat message was not recorded."
    )
    assert "not recorded" in render.withdraw_refusal(
        _record(status=store.STATUS_DECLINED), "answered_in_chat"
    )
    assert "nothing was recorded" in render.withdraw_refusal(
        _record(status=store.STATUS_EXPIRED), "answered_in_chat"
    )


def test_an_unknown_ask_refuses_the_withdraw_in_the_shared_words():
    """An ask no fold knows is refused with the SAME sentence every surface
    gets (``refusal_copy(None)``): there is one answer to "which ask?" and it
    does not get a second wording because the asker asked."""
    assert render.withdraw_refusal(None, "moot") == render.refusal_copy(None)


def test_the_receipts_state_what_happens_next():
    moot = render.withdraw_receipt("a-7f3", "moot")
    assert "a-7f3" in moot and "nothing will be delivered" in moot
    chat = render.withdraw_receipt("a-7f3", "answered_in_chat")
    assert "a-7f3" in chat and "response arrives" in chat


def test_the_op_constants_are_the_sentences_the_queue_emits():
    assert render.WITHDRAW_BAD_REASON == "reason must be 'moot' or 'answered_in_chat'."
    assert "no answers" in render.WITHDRAW_MOOT_TAKES_NO_ANSWERS
    assert "secret" in render.WITHDRAW_SECRET_REFUSAL
    assert "card" in render.WITHDRAW_SECRET_REFUSAL
