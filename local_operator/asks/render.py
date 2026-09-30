"""The ONE place the ask queue's text is written.

WHY THIS MODULE EXISTS (design ``docs/design/ask-nonblocking.md`` §2.3/§2.5).
A queued ask produces three kinds of prose: the RECEIPT the model reads the
moment it asks, the RESPONSE text injected when an answer (or a decline)
settles the ask, and the TIMEOUT notice injected at the deadline. Every surface
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
#: the deadline is that the session must not stall forever — and the last sentence
#: is what makes a late answer safe to accept rather than a second decision the
#: model did not ask for.
TIMEOUT_NOTICE = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "{questions}\n"
    "Proceed without it: use your recommended option or best judgment and state the "
    "assumption in your report. The ask stays open for the user; if they answer later "
    "you will be told."
)

#: Added when the ask was urgent (``timeout <= 15 min``). An urgent ask whose
#: deadline passed is not a case for "carry on and see": the model said this
#: needed an answer now, so the honest resolution is to get one from the expertise
#: it can reach without the operator.
TIMEOUT_URGENT = (
    "This ask was urgent. Do not wait — resolve it now: delegate the question to a "
    "`task` subagent with the relevant expertise and decide on its answer."
)

#: A secret ask's timeout NEVER quotes the prompt text (the question may itself
#: describe a credential) — it names the key and says the value did not arrive.
TIMEOUT_SECRET = (
    "[Ask timed out] No reply to ask {ask_id} arrived within {waited} (asked {asked}). "
    "The credential {keys} was not provided. Do not run anything that needed it; "
    "report what was left undone. The ask stays open for the user; if they answer "
    "later you will be told."
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


def response_text(record: Mapping[str, Any]) -> str:
    """The text for an ``ask_response`` row — answered, declined, or late.

    All three are the SAME custom type and the same id (design §2.3): a decline
    is a response-shaped fact, and reusing the type keeps one registration, one
    render branch and one entry kind — so a miss cannot silently drop one of
    them. Only ``text`` and ``status`` differ.
    """
    from local_operator.tools.builtin import (
        ASK_SECRET_UNANSWERED_TEXT,
        ASK_UNANSWERED_TEXT,
        _ask_report,
    )

    status = str(record.get("status") or "")
    questions = list(record.get("questions") or [])
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
        text = _ask_report(_question_models(questions), answers)
    if status == "late":
        return LATE_LEAD.format(deadline=_stamp(record.get("expires_at"))) + text
    return text


def timeout_text(record: Mapping[str, Any], *, now_ms: int) -> str:
    """The text for an ``ask_timeout`` row (design §2.5).

    Delivered to the model as an injected user-role message with a bracketed
    header — the ``wake_prompt`` shape — and never as "the user denied": nobody
    denied anything, the window simply closed. A secret ask names the key and
    nothing else.
    """
    questions = list(record.get("questions") or [])
    expires_at = int(record.get("expires_at") or 0)
    waited = _span(max(0, now_ms - int(record.get("created_at") or expires_at)) / 1000.0)
    asked = _stamp(record.get("created_at"))
    if any(q.get("secret") for q in questions):
        keys = ", ".join(str(q.get("id")) for q in questions if q.get("secret")) or "requested"
        return TIMEOUT_SECRET.format(
            ask_id=record.get("ask_id", ""), waited=waited, asked=asked, keys=keys
        )
    listing = "; ".join(_clip(str(q.get("question") or ""), 200) for q in questions)
    text = TIMEOUT_NOTICE.format(
        ask_id=record.get("ask_id", ""),
        waited=waited,
        asked=asked,
        questions=listing,
    )
    if record.get("urgent"):
        text = f"{text}\n{TIMEOUT_URGENT}"
    return text


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
    if status in (store.STATUS_DECLINED, store.STATUS_DISMISSED):
        return "you already declined this."
    if status == store.STATUS_ANSWERED:
        by = (record.get("answered_by") or {}).get("surface")
        return f"already answered by {by}." if by else "already answered."
    return "this ask is no longer open."


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
    return _report_secret_answers(models, {k: list(v) for k, v in answers.items()}, cast(Any, shim))


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
    "receipt_text",
    "refusal_copy",
    "response_text",
    "timeout_text",
]
