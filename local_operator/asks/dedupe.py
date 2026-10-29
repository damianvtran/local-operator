"""Fold only a verbatim repeated question; uncertainty means deliver.

An ask already notifies. An agent repeating its exact question via ``notify``
should not create a second unread task. Similar subject matter is NOT proof of
a duplicate: the previous token-set and signal-list rules hid removed polarity
("Do not delete … until backup?" vs "Delete … now") and rebound polarity
("Delete x but not y?" vs "Delete y but not x"). No bag-of-words, stemming,
stopword, option-label or question-id matching remains here (review round 3 B1).

Both directions use the SAME ordered equality comparison to ONE actual
``AskQuestion.question`` sentence. Normalization strips surrounding whitespace
ONLY. Interior whitespace, punctuation (including question marks), case,
Unicode code points, identifiers, numbers, operators and word order remain
byte-distinct. Even removing a final question mark could turn a question into
an instruction or alter a code identifier, so we deliberately do not do it.
A nonempty notify body must equal that same question too; a new link, outcome,
constraint, or contradictory body makes the whole update deliver. High
priority always delivers. These intentional false negatives cost a click;
contradictory suppression loses information. This is not semantic equivalence.

The window is 10 minutes: verbatim repetition within a long turn is likely a
duplicate; a later repeat may be a fresh reminder. An OPEN ask in THIS session
owns attached context. A matching earlier unread update is marked read (via
``merged_into_ask``), never dismissed, so surfaces without inline context can
still find it in All. Lock/recheck safety stays in ``AskQueue.add_context``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from local_operator.notifications.store import MERGED_INTO_ASK_VIA

WINDOW_MS = 10 * 60 * 1000

#: The ``via`` stamped on an update folded into an ask — distinct from the
#: operator's own ``explicit`` and the render receipt's ``in_view``, so the row
#: says WHY it left Unread rather than claiming the operator read it. Owned by
#: the store that writes it.
MERGED_VIA = MERGED_INTO_ASK_VIA

#: The tool result a suppressed ``notify`` returns. Model-facing, so English.
#: Names the ask so the agent can tell which question absorbed it, and says
#: what to do if it really has news.
SUPPRESSED_TEXT = (
    "Not posted: open ask {ask_id} already says this and notifies the user, so this "
    "update's text was attached to it. If you have news the ask does not say, post "
    "it with that news in the title."
)


def normalized(text: str) -> str:
    """Remove only surrounding whitespace, never rewrite the sentence itself."""
    return text.strip()


def adds_nothing(title: str, body: str, question: str, *, priority: str = "normal") -> bool:
    """True only for an ordered, verbatim repetition of a real question."""
    if priority == "high":
        return False
    expected = normalized(question)
    if not expected or normalized(title) != expected:
        return False
    return not normalized(body) or normalized(body) == expected


def repeats_question(title: str, body: str, ask: Mapping[str, Any], *, priority: str) -> bool:
    """Compare separately to each real question; labels/ids confer no equivalence."""
    return any(
        adds_nothing(title, body, str(q.get("question") or ""), priority=priority)
        for q in (ask.get("questions") or ())
        if isinstance(q, Mapping)
    )


def matching_ask(
    records: Iterable[Mapping[str, Any]],
    *,
    title: str,
    body: str,
    now: int,
    priority: str = "normal",
) -> dict[str, Any] | None:
    """The newest OPEN, non-relay ask raised within the window that this update restates.

    A relay row is excluded: its questions are another session's, and folding an
    update into it would attach this session's text to someone else's ask.
    """
    best: dict[str, Any] | None = None
    for record in records:
        if record.get("status") != "open" or record.get("relay"):
            continue
        created = int(record.get("created_at") or 0)
        if now - created > WINDOW_MS:
            continue
        if not repeats_question(title, body, record, priority=priority):
            continue
        if best is None or created >= int(best.get("created_at") or 0):
            best = dict(record)
    return best


def matching_updates(
    rows: Iterable[Mapping[str, Any]], *, ask: Mapping[str, Any], now: int
) -> list[dict[str, Any]]:
    """Unread, undismissed updates touched within the window that only restate ``ask``."""
    found: list[dict[str, Any]] = []
    for row in rows:
        if row.get("read") or row.get("dismissed"):
            continue
        if now - int(row.get("updated_at") or 0) > WINDOW_MS:
            continue
        if repeats_question(
            str(row.get("title") or ""),
            str(row.get("body_md") or ""),
            ask,
            priority=str(row.get("priority") or "normal"),
        ):
            found.append(dict(row))
    return found
