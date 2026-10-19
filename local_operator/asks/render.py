"""The ONE place the ask queue's text is written.

WHY THIS MODULE EXISTS (design ``docs/design/ask-nonblocking.md`` §2.3/§2.5).
A queued ask produces three kinds of prose: the RECEIPT the model reads the
moment it asks, the RESPONSE text injected when an answer (or a decline)
settles the ask, and the TIMEOUT notice injected at the deadline — plus, since
§12, the agent-side settle's own receipt and its OP-level refusals (which
answer the MODEL, not the user; see below). Every surface
shows the same words — the model's turn, the transcript row, the desktop card,
the phone card — and the design's rule is that no surface re-derives Q&A from
the text: one function produces the string, the structured payload rides beside
it in ``CustomMessage.details``, and a surface that needs structure reads the
payload rather than parsing the sentence.

**Reuse, not a second report.** The answer text IS the tool's existing report
(``tools/builtin._ask_report``), and a secret answer is key-substituted by the
existing ``_report_secret_answers``. Those are imported LAZILY inside the two
functions that need them: ``tools/builtin.py`` is the largest module in the
tree and importing it from a text helper would put the whole tool layer on the
session-construction path for every session, whether or not it ever queues an
ask.

**A receipt is not consent.** The reachable/unreachable split is MEASURED, not
asserted (§2.1): ``serving._attached_surfaces`` proves a client is connected,
which is not the same as a human having been told. The receipt therefore claims
presentation and never notice-delivery, and it carries the capitalised warning
the design's risk 1 depends on — a model that reads a receipt as approval would
run the very command the ask was meant to authorise.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

#: What a queued ask reports when something is attached. The middle sentence is
#: the risk-1 mitigation and is deliberately in capitals, NOT because shouting is
#: good copy but because this line text is the one a model is most likely to skim
#: on its way to acting.
RECEIPT_REACHABLE = (
    "Ask {ask_id} queued ({n} question(s)); showing on {reach}. Continue with other "
    "work — their answer arrives as an **ask response** turn. A RECEIPT IS NOT "
    "CONSENT: do not run anything the answer was meant to authorise. If nothing "
    "arrives by {expires} you will get a timeout notice."
)

#: The same receipt when nothing is attached. It still says the ask is durable —
#: that is the whole point of the queue — and it never claims someone was told.
RECEIPT_UNREACHABLE = (
    "Ask {ask_id} queued ({n} question(s)); nobody is attached right now, so it will "
    "be shown when the user next opens this session. Continue with other work. A "
    "RECEIPT IS NOT CONSENT: do not run anything the answer was meant to authorise. "
    "If nothing arrives by {expires} you will get a timeout notice."
)

#: Appended to a response that arrived after the deadline (``late``). It is not a
#: rebuke and not an instruction to undo: the agent already proceeded, and the
#: only useful question is whether the answer changes anything.
LATE_LEAD = (
    "You already proceeded when this ask timed out at {deadline}; reconsider only if "
    "the answer changes your work.\n\n"
)

#: The timeout notice. "Proceed without it" is the actionable half — the point of
#: the deadline is that the session must not stall forever. It used to end "The ask
#: stays open for the user; if they answer later you will be told", which stopped
#: being true when decision D10 made a lapsed ask read-only on every surface: no
#: surface offers the late answer it promised. A late answer through chat is still
#: delivered (``LATE_LEAD`` frames it), so nothing here forbids one.
TIMEOUT_NOTICE = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "{questions}\n"
    "Proceed without it: use your recommended option or best judgment and state the "
    "assumption in your report."
)

#: Added when the ask was urgent (``timeout <= 15 min``). An urgent ask whose
#: deadline passed is not a case for "carry on and see": the model said this
#: needed an answer now, so the honest resolution is to get one from the expertise
#: it can reach without the operator.
TIMEOUT_URGENT = (
    "This ask was urgent. Do not wait — resolve it now: delegate the question to a "
    "`task` subagent with the relevant expertise and decide on its answer."
)

#: Added when the deadline passed while the session was STOPPED (design §2.2). The
#: notice is owed either way — the agent still has to know it has no answer — but
#: without this sentence the timing reads as the agent having been slow to notice,
#: and the user who stopped the session gets told the agent moved on as though
#: they had been watching.
TIMEOUT_LAPSED_WHILE_STOPPED = (
    "This ask lapsed while the session was stopped, so the notice could not be "
    "delivered when the deadline passed."
)

#: A secret ask's timeout NEVER quotes the prompt text (the question may itself
#: describe a credential) — it names the key and says the value did not arrive.
TIMEOUT_SECRET = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "The credential {keys} was not provided. Do not run anything that needed it; "
    "report what was left undone."
)

#: The deadline notice a DISMISSAL still owes (fork A, 2026-10-16, §14). A
#: dismissal is a VIEW action — the user putting the ask away — and it tells the
#: model nothing, so an ask dismissed before its notice was delivered still owes
#: one; the sentence is what keeps the row from reading as an ordinary deadline.
#: Everything else is byte-identical to :data:`TIMEOUT_NOTICE`, and deliberately:
#: "Proceed without it" is the actionable half, and a dismissal makes proceeding
#: MORE certain, not less — there is no answer coming at all now.
TIMEOUT_NOTICE_DISMISSED = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "{questions}\n"
    "The user dismissed it without answering.\n"
    "Proceed without it: use your recommended option or best judgment and state the "
    "assumption in your report."
)

#: The same row for a SECRET ask: names the key and nothing else (see
#: :data:`TIMEOUT_SECRET`), and keeps that variant's own instruction.
TIMEOUT_SECRET_DISMISSED = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "The credential {keys} was not provided. The user dismissed it without answering. "
    "Do not run anything that needed it; report what was left undone."
)

#: When a secret answer was stored but the session restarted before the response
#: was delivered, the value is gone (session-memory only, ``persist`` excepted).
#: This is the one path where the user answered and the agent still cannot
#: proceed, so it says so unmistakably (risk 3).
SECRET_VALUE_LOST = (
    "The credential was provided but the session restarted before it could be used — "
    "ask again if it is still needed."
)


def receipt_text(record: Mapping[str, Any], reach: str | None) -> str:
    """The tool result a queued ask returns.

    ``reach`` is the measured presentation surface (``"terminal"``, ``"phone"``,
    ``"desktop"``), or ``None`` when nothing is attached. It is a claim about a
    surface, never about a person: see the module docstring.
    """
    questions = record.get("questions") or []
    template = RECEIPT_REACHABLE if reach else RECEIPT_UNREACHABLE
    return template.format(
        ask_id=record.get("ask_id", ""),
        n=len(questions),
        reach=reach,
        expires=_stamp(record.get("expires_at")),
    )


def response_text(
    record: Mapping[str, Any],
    *,
    secret_lost: bool = False,
    image_counts: Mapping[str, int] | None = None,
    images_missing: int = 0,
) -> str:
    """The text for an ``ask_response`` row — answered, declined, or late.

    All three are the SAME custom type and the same id (design §2.3): a decline
    is a response-shaped fact, and reusing the type keeps one registration, one
    render branch and one entry kind — so a miss cannot silently drop one of
    them. Only ``text`` and ``status`` differ.

    ``secret_lost`` is the verify-on-delivery half of §2.4 (review round 1, MAJOR
    3): when the caller has established that a key this row announces is no longer
    in the session's store, :data:`SECRET_VALUE_LOST` LEADS the text. It leads
    rather than trails because the report below is written as a delivered answer,
    and a model that reads the bottom line into a plan has already gone wrong by
    the time it reaches a caveat — the same reason the refusal copy for a version
    skew leads with the refusal.
    """
    from local_operator.tools.builtin import (
        ASK_SECRET_UNANSWERED_TEXT,
        ASK_UNANSWERED_TEXT,
        _ask_report,
    )

    status = str(record.get("status") or "")
    questions = list(record.get("questions") or [])
    if status == "dismissed":
        # NEVER through ``_ask_report``: a dismissal has no answers, and the report
        # would render the absence as an empty answer (design §15).
        return dismissed_text(record)
    answers = {
        str(k): [str(v) for v in (vals or [])] for k, vals in (record.get("answers") or {}).items()
    }
    if status == "declined":
        text = (
            ASK_SECRET_UNANSWERED_TEXT
            if any(q.get("secret") for q in questions)
            else ASK_UNANSWERED_TEXT
        )
    else:
        # TB-05 (PR #2378): a bridge request's report must not frame the
        # sender's words as the human's — the flag is derived from the RAW
        # record carriage (a non-empty question_id), the same predicate the
        # queue's doors use.
        bridge_request = any(
            isinstance(question, Mapping) and str(question.get("question_id") or "")
            for question in questions
        )
        # ``image_counts``/``images_missing`` are passed through only when the
        # answer carried pictures; for every text-only ask the call is the one it
        # always was, so the text the model reads is unchanged.
        text = _ask_report(
            _question_models(questions),
            answers,
            image_counts=image_counts,
            images_missing=images_missing,
            bridge_request=bridge_request,
        )
    if secret_lost:
        # The key name is not repeated here and the QUESTION TEXT is not quoted:
        # a secret question's prompt may itself describe the credential, so this
        # branch says only what happened and what to do about it.
        text = f"{SECRET_VALUE_LOST}\n{text}"
    if status == "late":
        return LATE_LEAD.format(deadline=_stamp(record.get("expires_at"))) + text
    return text


#: What an OPEN-ask dismissal tells the agent (design §15, D1). One row per
#: ``reason``; the secret variant names no value and says where else the credential
#: may be found.
DISMISSED_ANSWERED_IN_CHAT = (
    "The user dismissed ask {ask_id}: they say it was settled in the conversation. "
    "Take the answer from what was said in chat and do not ask again; there is no "
    "need to call `ask_withdraw` for it (it would be refused). If the chat did not "
    "settle it for you, say so in one line instead of re-asking."
)
DISMISSED_MOOT = (
    "The user dismissed ask {ask_id} as no longer needed. Do not ask it again; carry "
    "on without it and say in one line what you assumed."
)
DISMISSED_SECRET = (
    "The user dismissed the credential ask {ask_id} without entering a value "
    "({reason_label}). Do not run anything that needed it unless the conversation "
    "says where the credential is, such as a stored secret."
)
#: The operator-facing label of each reason (design §3.2).
DISMISS_REASON_LABELS = {
    "answered_in_chat": "settled in chat",
    "moot": "no longer needed",
}


def dismissed_text(record: Mapping[str, Any]) -> str:
    """The ``ask_response`` text for an ask the operator dismissed before its deadline."""
    from local_operator.asks import store

    reason = store.normalize_dismiss_reason(record.get("dismissed_reason"))
    ask_id = str(record.get("ask_id") or "")
    if any(q.get("secret") for q in (record.get("questions") or ()) if isinstance(q, Mapping)):
        return DISMISSED_SECRET.format(ask_id=ask_id, reason_label=DISMISS_REASON_LABELS[reason])
    template = DISMISSED_ANSWERED_IN_CHAT if reason == "answered_in_chat" else DISMISSED_MOOT
    return template.format(ask_id=ask_id)


def _timeout_listing_line(question: Mapping[str, Any]) -> str:
    """One question as the timeout notice names it.

    A BRIDGE question's text carries another organization's words (framed by
    §11.1 in the ask itself). This notice is a plain sentence about OUR ask, so
    for a request it names the request and the house untrusted-data clause
    instead of repeating the sender's words outside their own framing — quoting
    them here would be exactly the un-framed re-serve §11.1 forbids.
    """

    if str(question.get("question_id") or ""):
        from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE

        return f"{question.get('id')} — {BRIDGE_UNTRUSTED_CLAUSE}"
    return _clip(str(question.get("question") or ""), 200)


def timeout_text(
    record: Mapping[str, Any],
    *,
    now_ms: int,
    lapsed_while_stopped: bool = False,
    dismissed: bool = False,
) -> str:
    """The text for an ``ask_timeout`` row (design §2.5).

    Delivered to the model as an injected user-role message with a bracketed
    header — the ``wake_prompt`` shape — and never as "the user denied": nobody
    denied anything, the window simply closed. A secret ask names the key and
    nothing else.

    ``lapsed_while_stopped`` appends :data:`TIMEOUT_LAPSED_WHILE_STOPPED`, which is
    the stop rule's copy half (§2.2): the row is delivered at reopen for a deadline
    that passed with nothing running, and the reason it is late is the user's own
    stop rather than the agent's inattention.

    ``dismissed`` selects the fork-A variants (the row a dismissal still owes): the
    same notice with one added sentence saying the user put the ask away. It is a
    FLAG rather than the caller swapping templates, so the two variants cannot
    drift from the notices they are variants of.
    """
    questions = list(record.get("questions") or [])
    expires_at = int(record.get("expires_at") or 0)
    waited = _span(max(0, now_ms - int(record.get("created_at") or expires_at)) / 1000.0)
    asked = _stamp(record.get("created_at"))
    if any(q.get("secret") for q in questions):
        keys = ", ".join(str(q.get("id")) for q in questions if q.get("secret")) or "requested"
        template = TIMEOUT_SECRET_DISMISSED if dismissed else TIMEOUT_SECRET
        text = template.format(
            ask_id=record.get("ask_id", ""), waited=waited, asked=asked, keys=keys
        )
    else:
        listing = "; ".join(_timeout_listing_line(q) for q in questions)
        template = TIMEOUT_NOTICE_DISMISSED if dismissed else TIMEOUT_NOTICE
        text = template.format(
            ask_id=record.get("ask_id", ""),
            waited=waited,
            asked=asked,
            questions=listing,
        )
    if record.get("urgent"):
        text = f"{text}\n{TIMEOUT_URGENT}"
    if lapsed_while_stopped:
        text = f"{text}\n{TIMEOUT_LAPSED_WHILE_STOPPED}"
    return text


#: THE DELIVERED REFUSAL (design §10, #1936). Deliberately NOT a row of
#: :func:`refusal_copy`'s state table: that table answers "why is this ask not
#: answerable here", and the state this sentence needs — "answered, and the
#: response row already exists" — is the same ``answered`` state a plain repeat
#: ``respond`` must keep reading as "already answered by <surface>". The revision
#: path is the only caller that may say it, so the revision path emits it (see
#: ``AskQueue.revise``): the sentence belongs to the OP, not to the state table.
REVISED_ALREADY_DELIVERED = "already delivered — send a new message"


#: The ``by`` words that name a ROLE rather than a surface (D4, round 14).
#: Every ordinary caller of ``AskQueue.respond``/``revise`` passes the SURFACE the
#: human answered from ("terminal" from the TUI, "remote" from the phone), and
#: the sentence below is in the user's voice — so naming the surface tells them
#: where the answer came from. The bridge decide DOOR has no surface to name: it
#: is reached over the runtime's authority op, which the terminal, the desktop
#: app and the phone can all drive, and it stamps "operator" (`session.py`).
#: That word appears nowhere else a member can see, so it renders as the
#: UNATTRIBUTED form — a true sentence rather than a third-person label about a
#: role they were never shown.
ROLE_BY_WORDS = frozenset({"operator"})


def refusal_copy(record: Mapping[str, Any] | None) -> str:
    """Why an answer was refused, in one sentence per state (design §2.2).

    Every surface shows these words, so a stale tap reads the same on the TUI,
    the desktop app and the phone. ``None`` (no such ask) is its own answer
    rather than an "already answered" that would be a lie about an id this
    process has never seen.
    """
    from local_operator.asks import store

    if record is None:
        return "that ask is not in this session's queue — it may belong to another session."
    status = str(record.get("status") or "")
    if status == store.STATUS_OPEN:
        return ""  # accepted
    if status == store.STATUS_TIMED_OUT:
        return ""  # accepted, folded to `late`
    if status == store.STATUS_LATE:
        return ""  # accepted again; still one response row
    if status == store.STATUS_EXPIRED:
        return "this ask expired 7 days ago — ask again if it is still needed."
    if status == store.STATUS_DECLINED:
        return "you already declined this."
    if status == store.STATUS_DISMISSED:
        # THE VIEW ACTION HAS ITS OWN WORD (UX round 1, U3 = QA Q3 = agent review
        # round 1, MINOR-4): a dismissal is a decision the user took, but "declined"
        # names a different act — the one where the agent was told to decide.
        return DISMISS_ALREADY_DISMISSED
    if status == store.STATUS_WITHDRAWN:
        # DESIGN §12's sentence, verbatim: a withdrawn ask is not answerable
        # from any surface (its box hides everywhere — ``withdrawn`` is not
        # outstanding), and a stale tap reads the ONE route left — say it in
        # chat, where the agent can record it with `ask_withdraw`.
        return (
            "the agent withdrew this question — if you have an answer, send it "
            "as a chat message."
        )
    if status == store.STATUS_ANSWERED:
        by = (record.get("answered_by") or {}).get("surface")
        if not by or by in ROLE_BY_WORDS:
            return "already answered."
        return f"already answered by {by}."
    return "this ask is no longer open."


#: The refusal an OPEN-ask dismissal gets while the agent is working (D6, design
#: §15). Dismissal is "hear at the next boot": it never starts, wakes or
#: injects into running work, so it is offered only on a cold or STOPPED session.
#: On a running one the operator's route is the conversation itself, where the
#: agent is already reading.
DISMISS_AGENT_RUNNING = (
    "the agent is still running — reply in chat to settle this ask, "
    "or stop the session first and dismiss it then."
)

#: A dismissal aimed at a LATE ask: ``refusal_copy`` accepts a late ask (it can be
#: answered again), but dismissing one is a different act and the answer is in.
DISMISS_ALREADY_ANSWERED = "this ask was already answered."

#: An unknown ``reason`` on a dismissal, refused before anything is written.
DISMISS_BAD_REASON = "reason must be 'answered_in_chat' or 'moot'."

#: A team-bridge request carries another agent's question; its reply goes to that
#: agent's channel, so it stays dismissible only once it has timed out (§3.5).
DISMISS_BRIDGE_OPEN = (
    "this request comes from another agent — it can be dismissed only after it has timed out."
)

#: What a DISMISSED ask answers when something is asked of it AGAIN (UX round 1,
#: U3 = QA round 1, Q3 = agent review round 1, MINOR-4). ``DISMISS_NOT_LAPSED``
#: says the ask "is still open", which is the one thing a dismissed ask is not —
#: a repeat tap on the row the user had struck out was told the opposite of what
#: they had just done. ONE constant read by :func:`refusal_copy` (an ANSWER or a
#: revision aimed at a dismissed ask) and :func:`dismiss_refusal` (the repeat tap)
#: on both doors, so a stale sheet and a live queue cannot word it differently.
DISMISS_ALREADY_DISMISSED = "you already dismissed this."


def dismiss_refusal(
    record: Mapping[str, Any] | None,
    *,
    agent_running: bool = False,
) -> str:
    """Why a dismissal was refused, or ``""`` when it is ACCEPTED.

    ``record`` is the FOLDED ask (``None`` when the log holds no such ask). The
    decision is a property of the log plus ONE runtime fact, which is what lets
    the cold door hold a runtime-less session to exactly the rule a live one
    applies — and why this is one function rather than two.

    ``agent_running`` is that fact (D6, design §15): True when a runtime is up and
    NOT stopped. The cold door has no runtime and passes False; the live door
    passes what the session reports. It only matters for an OPEN ask — a timed-out
    one is accepted on every door, as before.
    """
    from local_operator.asks import store

    if record is None:
        return refusal_copy(None)
    status = str(record.get("status") or "")
    if status == store.STATUS_TIMED_OUT:
        return ""
    if status == store.STATUS_OPEN:
        if any(
            str(question.get("question_id") or "")
            for question in (record.get("questions") or ())
            if isinstance(question, Mapping)
        ):
            return DISMISS_BRIDGE_OPEN
        return DISMISS_AGENT_RUNNING if agent_running else ""
    if status == store.STATUS_DISMISSED:
        # THE REPEAT TAP: a dismissed ask is not a live one, and this is the
        # sentence that says what the user did.
        return DISMISS_ALREADY_DISMISSED
    if status == store.STATUS_LATE:
        # ``refusal_copy`` accepts a late ask (it can be answered again); a
        # dismissal of one is a different act, and the answer is already in.
        return DISMISS_ALREADY_ANSWERED
    return refusal_copy(record)


#: The tool result an agent-side WITHDRAWAL returns (design §12). Written to the
#: MODEL — `ask_withdraw` is the asker's own op, unlike every other settle a
#: surface authors for the user — so it states what is now true and what will
#: (not) happen next: nothing more is delivered for a moot ask, and a chat
#: answer still gets its standard response row.
WITHDRAWN_RECEIPT = (
    "Ask {ask_id} withdrawn — no longer waiting on an answer and nothing will be "
    "delivered for it."
)
ANSWERED_IN_CHAT_RECEIPT = (
    "Ask {ask_id} answered from the user's chat message — their words are recorded "
    "as its answer; the response arrives as an ask response turn."
)


#: The reasons `ask_withdraw` refuses BEFORE any state is consulted, and the
#: one conditional reason rule (design §12): ``answers`` belongs to
#: ``answered_in_chat`` alone, and refusing the mismatch is what keeps a model
#: that meant to record the user's words from silently losing them to a
#: ``moot`` withdrawal.
WITHDRAW_BAD_REASON = "reason must be 'moot' or 'answered_in_chat'."
WITHDRAW_MOOT_TAKES_NO_ANSWERS = (
    "reason 'moot' takes no answers — to record the user's words use reason " "'answered_in_chat'."
)

#: `answered_in_chat`'s secret refusal (design §12): there is no masked-entry
#: hop from chat text and there must not be one, so the card remains the only
#: secret path. Nothing is written, and the sentence says so.
WITHDRAW_SECRET_REFUSAL = (
    "this ask has a secret question — only its card can answer it; nothing was recorded."
)


def withdraw_receipt(ask_id: str, reason: str) -> str:
    """The receipt a successful ``AskQueue.withdraw`` hands back (design §12)."""
    template = ANSWERED_IN_CHAT_RECEIPT if reason == "answered_in_chat" else WITHDRAWN_RECEIPT
    return template.format(ask_id=ask_id)


def withdraw_refusal(record: Mapping[str, Any] | None, reason: str) -> str:
    """Why ``AskQueue.withdraw`` refused, in the MODEL's voice (design §12).

    THE §10 OP-VS-STATE SPLIT, applied to a second op. :func:`refusal_copy`
    answers a SURFACE's question — "why is this ask not answerable here" — in
    the user's voice ("you already declined this"), and every surface shows it.
    ``withdraw`` answers the MODEL that tried to settle the ask, and telling it
    "you already declined this" would be false about who declined. So the
    op carries its own sentences, emitted by the op path exactly as §10's
    delivered sentence is; the state table keeps its rows byte-for-byte.

    One sentence per (reason, state): ``moot`` explains why nothing can be
    retracted, ``answered_in_chat`` says whether anything was recorded. Both
    are agent-facing only — no surface renders them.
    """
    from local_operator.asks import store

    if record is None:
        return refusal_copy(None)
    status = str(record.get("status") or "")
    chat = reason == "answered_in_chat"
    if status in (store.STATUS_ANSWERED, store.STATUS_LATE):
        return (
            "this ask already has the user's answer — the chat message was not recorded."
            if chat
            else "this ask already has the user's answer — it cannot be withdrawn."
        )
    if status == store.STATUS_DECLINED:
        return (
            "the user already declined this ask — the chat message was not recorded."
            if chat
            else "the user already declined this ask — there is nothing to withdraw."
        )
    if status == store.STATUS_DISMISSED:
        return (
            "the user already dismissed this ask — the chat message was not recorded."
            if chat
            else "the user already dismissed this ask — there is nothing to withdraw."
        )
    if status == store.STATUS_WITHDRAWN:
        return "this ask was already withdrawn."
    if status == store.STATUS_EXPIRED:
        return (
            "this ask expired — nothing was recorded; ask again if it is still needed."
            if chat
            else "this ask expired — there is nothing to withdraw."
        )
    # Not a state the op refuses on — callers only reach here for a settled ask,
    # and the fall-through keeps a future fold status from ever answering empty.
    return refusal_copy(record)


# ---------------------------------------------------------------------------
# Relays (#2520): a watcher re-raising another session's pending ask
# ---------------------------------------------------------------------------
#
# A relay's refusals answer the MODEL that tried to relay, like ``ask_withdraw``'s
# — no surface renders them — and each names the ONE reason, because a model that
# reads a generic failure retries. The distinction between them is load-bearing:
# "the original is settled" is information (stop relaying), "the ref is
# malformed" is a correction (write it as `<session_id>/<ask_id>`), and "the
# origin is a secret ask" is a security boundary (a credential must never be
# routable through a watcher's store), not a retryable state.

#: The ref is not exactly `<session_id>/<ask_id>`.
RELAY_BAD_FORM = (
    "a relay ref is exactly '<session_id>/<ask_id>' — the ask id its receipt or "
    "card names, and the session that owns it."
)

#: The origin session's log is missing, or holds no such ask.
RELAY_ORIGIN_UNKNOWN = (
    "that relay names no ask in the original session's log — the ask may have "
    "been settled and swept, or the id is wrong; relaying copies nothing."
)

#: The origin is already settled: there is nothing left to answer.
RELAY_ORIGIN_SETTLED = (
    "the original ask is already settled — there is nothing left to answer; do " "not relay it."
)

#: The named origin is itself a relay; a relay of a relay has no answerable end.
RELAY_OF_RELAY = (
    "the original ask is itself a relay — relay the ask it points at instead, "
    "one hop from the session that asked it."
)

#: SECURITY, not a guard: a secret question never leaves its own session's card.
RELAY_SECRET_ORIGIN = (
    "the original ask carries a secret question — a credential is answered only "
    "on its own session's card and is never relayed."
)

#: The relay names this session's own ask: answer it here.
RELAY_SELF = (
    "that ask belongs to this session — answer it here directly rather than "
    "relaying it to itself."
)

#: This session already holds a live relay for that origin ask (#2673, R1-4).
#: Its own asks already refuse a duplicate, and a second relay of one origin
#: would mint a second card for the same question — two rows that both settle
#: from one log, so the human would be asked twice.
RELAY_ALREADY_HELD = (
    "this session already relays that ask — answer it on the card it already "
    "shows, or withdraw that relay before relaying the ask again."
)

#: The per-watcher bound. ``{cap}`` is ``policy.RELAY_CAP`` at the call site so
#: the sentence and the constant cannot drift.
RELAY_CAP_REFUSAL = (
    "this session already relays {cap} open asks (the relay cap is {cap}); "
    "settle or withdraw one before relaying more."
)

#: A relay on a host with no queued engine: the pointer needs the log.
RELAY_NEEDS_QUEUE = (
    "a relay needs the queued-ask engine — this host answers questions inline, "
    "so there is no ask log to point at."
)

#: Withdrawing a relay with chat text: the answer belongs to the ORIGIN's log.
RELAY_WITHDRAW_ANSWERED_IN_CHAT = (
    "a relayed ask is answered on the session that asked it — answer it there; "
    "nothing was recorded here."
)

#: Revising through a relay: the revision must land on the ORIGIN's log, and
#: this session has no row to supersede.
RELAY_REVISE_UNSUPPORTED = (
    "a relayed ask cannot be revised from this session — revise the original "
    "ask where it was asked; nothing was recorded here."
)

#: The receipt a successful relay hands its model.
RELAY_RECEIPT = (
    "Relay {relay_id} created: ask {origin_ask} in session {origin_session} is "
    "now pointed at from this queue's list{reach_note}. Answering either copy "
    "settles both. A RECEIPT IS NOT CONSENT: do not run anything the original "
    "ask was meant to authorise. Continue with other work; when it settles you "
    "will be told in one line here."
)


def relay_receipt(
    relay_id: str,
    origin_session: str,
    origin_ask: str,
    *,
    reach: str | None,
) -> str:
    """The tool result a successful relay returns (#2520).

    The reachable/unreachable split is the queued ask's own (``receipt_text``):
    presentation is MEASURED, never claimed as notice-delivery.
    """
    reach_note = f"; showing on {reach}" if reach else ""
    return RELAY_RECEIPT.format(
        relay_id=relay_id,
        origin_ask=origin_ask,
        origin_session=origin_session,
        reach_note=reach_note,
    )


def relay_refused_by_origin(origin_session: str, origin_ask: str, copy: str) -> str:
    """The origin's refusal, labelled so the human knows which ask refused (#2520).

    ``copy`` is ``refusal_copy``'s (or the queue's) wording VERBATIM: the relay is
    a pointer, so an answer routed through it is decided by the ORIGIN's log and
    must read exactly as it would there. The lead only says where the decision
    came from, or a stale tap on the watcher's card looks like the watcher's own
    rule.
    """
    return f"the original ask {origin_ask} in session {origin_session} refused this: {copy}"


#: How a settled relay's one-line note names the settlement. ``answered`` and
#: friends name the SURFACE when the origin's log records one (the whole point:
#: the watcher learns where the answer came from, not merely that one exists).
RELAY_SETTLED_NOTE = "relayed {relay_id} from {origin} settled: {how}"


def relay_settled_note(record: Mapping[str, Any]) -> str:
    """The one-line note the WATCHER session is told when its relay settles.

    Written to the model of the session that RELAYED, not the one that asked —
    it says which ask the note is about and how it ended, because the watcher may
    hold several and the row arrives on its own turn (``ask-relay-<id>``).

    A status the note cannot name falls back to the bare status rather than an
    empty sentence; the sentence is the surface, so it must always say something
    true.
    """
    from local_operator.asks import store

    relay = record.get("relay") or {}
    relay_id = str(record.get("ask_id") or "")
    origin_session = str(relay.get("session_id") or "")
    origin_ask = str(relay.get("ask_id") or "")
    status = str(record.get("status") or "")
    if status in (store.STATUS_ANSWERED, store.STATUS_LATE, store.STATUS_DECLINED):
        surface = str((record.get("answered_by") or {}).get("surface") or "")
        how = f"{status} by {surface}" if surface else status
    elif status == store.STATUS_TIMED_OUT:
        how = "timed out"
    elif status == store.STATUS_EXPIRED and relay.get("origin_gone"):
        how = "the original ask is gone (its session log no longer holds it)"
    elif status == store.STATUS_EXPIRED:
        how = "expired"
    elif status == store.STATUS_WITHDRAWN:
        how = "withdrawn"
    else:
        how = status or "settled"
    origin = f"ask {origin_ask} in session {origin_session}" if origin_ask else "the original ask"
    return RELAY_SETTLED_NOTE.format(relay_id=relay_id, origin=origin, how=how)


def mirror_card(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The legacy single-slot card fields for the HEAD open ask (design §4).

    THE LEGACY MIRROR. The desktop app and the native mobile app ship on their
    own schedules, so for one release a new core must still present a queued ask
    the way the old clients can see and answer it: as today's per-question
    ``pending``/``pending_gate`` card, whose ``request_id`` is
    ``"<ask_id>.<qidx>"`` (``store.mirror_request_id``) and whose answer arrives
    as the old ``ask_answer`` op.

    ``rows`` are the ``PendingAsk`` dicts the fold produced; the HEAD is the
    OLDEST still-open ask, chosen here rather than taken from the caller's order
    because the two callers order differently (the wire view is open-first
    NEWEST-first; the queue's own ``open_records`` is log order) and a card that
    changed identity between publishers is a card the user answers twice.

    That choice is deliberately NOT "the list's first row" — the published list
    leads with the NEWEST open ask, while a mirrored card that jumped to each new
    arrival would move under a user's finger mid-tap. The eventual fix is to make
    the list lead with the oldest too (the design's own "head" wording); until
    then the divergence is named here and in §4's A2 addendum rather than
    discovered.

    Only an ``open`` ask is mirrored, deliberately: a ``timed_out`` ask is past
    its deadline, and painting the old "waiting for you" card for it would tell
    the user their answer still counts as one did before the deadline. Since D10
    a timed-out ask is read-only on every surface (the ``ask_timeout`` row now
    says the agent goes ahead without the answer); the store still folds a late
    chat answer as ``late``, but no surface offers one, so the mirror has nothing
    to show for it.

    ``None`` means "no ask to mirror": the caller publishes its own empty card
    exactly as it does today.
    """
    from local_operator.asks import store

    opens = [row for row in rows if str(row.get("status") or "") == store.STATUS_OPEN]
    if not opens:
        return None
    head = min(
        opens,
        key=lambda row: (int(row.get("created_at") or 0), str(row.get("ask_id") or "")),
    )
    questions = [dict(q) for q in (head.get("questions") or []) if isinstance(q, Mapping)]
    # ``answers`` is what the LOG holds; ``draft_question_ids`` is what the old
    # client has ALREADY TAPPED in this runtime (the incremental legacy path,
    # §4 addendum). Both mean "do not offer this question again", and the card
    # has to advance on a draft or the old client would be offered the question
    # it just answered until the ask settles.
    answered = {str(key) for key in (head.get("answers") or {})}
    answered |= {str(key) for key in (head.get("draft_question_ids") or ())}
    index = 0
    for position, question in enumerate(questions):
        if str(question.get("id") or "") not in answered:
            index = position
            break
    question = questions[index] if questions else {}
    return {
        "ask_id": str(head.get("ask_id") or ""),
        "request_id": store.mirror_request_id(str(head.get("ask_id") or ""), index),
        "question_index": index,
        "question_total": len(questions) or 1,
        "title": str(question.get("question") or "the agent is asking"),
        "options": [
            {
                "label": str(option.get("label") or ""),
                "description": str(option.get("description") or ""),
            }
            for option in (question.get("options") or [])
            if isinstance(option, Mapping)
        ],
        "secret": bool(question.get("secret")),
        "recommended": question.get("recommended"),
        "persist": bool(question.get("persist")),
    }


def apply_secret_answers(
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Sequence[str]],
    *,
    variables: Any,
    journal_credential: Any = None,
) -> dict[str, list[str]]:
    """Store secret answers and return them key-substituted.

    Delegates to ``tools/builtin._report_secret_answers`` — the SAME hop the
    blocking path used — because that function is the last place the raw value
    can be kept out of the model's context, and giving the queue a second
    implementation would be giving the secret two ways out. The context it
    expects is duck-typed (``.variables`` / ``.journal_credential``), so a few
    lines of shim is the whole adapter.

    **ONLY THE CELLS THE CALLER SUPPLIED come back** (PRE-EXISTING defect found en
    route in round 2, identical in ``respond_ask`` and fixed with the revision
    contract). The delegated hop answers one cell for EVERY question id — ``[]``
    for the ones the caller never supplied — because the blocking picker always
    returned the whole set. Handed straight to ``AskQueue`` that fabricated cell
    defeated the whole-ask COMPLETENESS check (§2.4): an omitted question was
    recorded as "no answer" instead of being refused, so the model could be told
    the user had skipped a question they were never shown. Filtering here rather
    than in the hop keeps the tool layer's contract for the blocking path, and
    covers every queued caller at once — ``respond_ask``, ``revise_ask`` and the
    legacy per-tap mirror, which reads its one cell back out by key.
    """
    from local_operator.tools.builtin import _report_secret_answers

    class _Ctx:
        pass

    shim = _Ctx()
    shim.variables = variables  # type: ignore[attr-defined]
    shim.journal_credential = journal_credential  # type: ignore[attr-defined]
    models = _question_models(list(questions))
    # The function duck-types its context (``.variables`` / ``.journal_credential``);
    # the cast is the adapter's whole purpose, so it is named rather than silenced.
    reported = _report_secret_answers(
        models, {k: list(v) for k, v in answers.items()}, cast(Any, shim)
    )
    # The caller's OWN key set, so an omitted question stays omitted and the
    # queue's completeness check can refuse it (see the docstring).
    return {qid: cell for qid, cell in reported.items() if qid in answers}


def _question_models(questions: Sequence[Any]) -> list[Any]:
    """The question dicts as ``AskQuestion`` models (walk in: dicts walk out).

    The log holds plain dicts (``asks/store.py`` is stdlib-only and must not
    know about pydantic), while ``_ask_report`` takes models — and its
    validators are what normalise a hand-written or older row, so the
    conversion is a validation step rather than a no-op.
    """
    from local_operator.harness.types import AskQuestion

    out: list[Any] = []
    for raw in questions:
        if isinstance(raw, AskQuestion):
            out.append(raw)
            continue
        try:
            out.append(AskQuestion.model_validate(dict(raw)))
        except Exception:  # noqa: BLE001 — a row the model rejects is shown as-is
            out.append(AskQuestion.model_construct(**dict(raw)))
    return out


def _stamp(value: Any) -> str:
    """An epoch-ms value as a short local time, or ``"an unknown time"``.

    Local time, not UTC, and no date: these strings are read by a model and a
    human in the same conversation, and "asked 14:02" is what a person who was
    away for an hour needs. The absolute date is on the transcript row.
    """
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return "an unknown time"
    if ms <= 0:
        return "an unknown time"
    import time as _time

    return _time.strftime("%H:%M", _time.localtime(ms / 1000.0))


def _span(seconds: float) -> str:
    """A duration the way a person says it (``45s``/``12m``/``3h``/``2d``)."""
    if seconds <= 0:
        return "a moment"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "LATE_LEAD",
    "RECEIPT_REACHABLE",
    "RECEIPT_UNREACHABLE",
    "SECRET_VALUE_LOST",
    "TIMEOUT_NOTICE",
    "TIMEOUT_SECRET",
    "TIMEOUT_URGENT",
    "apply_secret_answers",
    "mirror_card",
    "receipt_text",
    "refusal_copy",
    "relay_receipt",
    "relay_refused_by_origin",
    "relay_settled_note",
    "response_text",
    "timeout_text",
]
