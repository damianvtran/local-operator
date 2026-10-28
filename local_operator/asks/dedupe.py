"""One topic, one inbox entry: an open ask and a same-topic ``notify`` never both land.

WHY THIS EXISTS. Agents often raise an ``ask`` AND a ``notify`` update about the
same thing — "Decision needed: drop the staging table?" as a question, and
"Waiting on you: staging table" as an update. The operator then answers the
question and ALSO has to mark the update read, for one piece of information. An
ask already notifies (it is the highest-priority row the inbox holds, with its
own banner), so the update is pure duplication. The prompts say so (``notify``
and ``ask`` descriptions, ``system.md``), but prompts drift, so the rule is
also enforced here, in ONE place, by both doors:

* ``notify`` while a same-topic OPEN ask from the same session exists → the
  update is NOT posted; its text is folded into that ask as context
  (``asks.store.EVENT_CONTEXT``) and the tool says so.
* ``ask`` while a same-topic UNREAD update from the same session exists → the
  update is marked read and dismissed with ``via="merged_into_ask"`` (so it
  leaves Unread and All), and its text is folded into the new ask.

THE WINDOW (:data:`WINDOW_MS`, 10 minutes). The duplicate pattern is a single
turn's work: the agent asks, then writes its wrap-up update (or the reverse)
seconds to a few minutes apart. Ten minutes covers a long turn with tool calls
between the two; past it, a same-topic update is more likely news (a result
that arrived later) than a restatement, and must be delivered.

THE SIMILARITY (:func:`similarity`, threshold :data:`THRESHOLD`). Deterministic,
cheap, no model call: lower-cased alphanumeric tokens of 3+ characters minus a
short stopword list, compared as sets with Jaccard. Two readings are scored and
the HIGHER wins — the update's title alone, and its title+body — because a
title is usually the topic restated while a long body dilutes any set measure.
The threshold is CONSERVATIVE on purpose: a false merge hides a real FYI from
the operator (the costly error), while a missed merge only costs them one extra
click (today's behaviour). ``tests/unit/asks/test_dedupe.py`` pins one pair just
under and one just over it.

IMPORT-LIGHT: stdlib plus the stdlib-only notifications store (for the one
``via`` constant it owns); the session imports this on the notify and ask
doors, and nothing here may load the harness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from local_operator.notifications.store import MERGED_INTO_ASK_VIA

#: How far apart an ask and a same-topic update may be and still be one topic.
WINDOW_MS = 10 * 60 * 1000

#: Jaccard at or above this is "the same topic". See the module docstring for
#: why it is high rather than balanced.
THRESHOLD = 0.5

#: The ``via`` stamped on an update folded into an ask — distinct from the
#: operator's own ``explicit`` and the render receipt's ``in_view``, so the row
#: says WHY it left Unread rather than claiming the operator read it. Owned by
#: the store that writes it.
MERGED_VIA = MERGED_INTO_ASK_VIA

#: The tool result a suppressed ``notify`` returns. Model-facing, so English.
SUPPRESSED_TEXT = (
    "Not sent: your open ask already notifies the user; its text was added to that ask."
)

_TOKEN = re.compile(r"[a-z0-9]+")

#: Function words that carry no topic. Short on purpose: every word removed
#: here makes two unrelated texts look more alike, never less.
_STOPWORDS = frozenset(
    "the and for you your are was were this that with from have has had not but "
    "can will should would could did does into onto about what which when where "
    "who why how all any now then than them they its our out use".split()
)


def tokens(text: str) -> frozenset[str]:
    """The topic-bearing tokens of ``text``: lower-case, 3+ chars, no stopwords."""
    return frozenset(
        word
        for word in _TOKEN.findall(str(text or "").lower())
        if len(word) >= 3 and word not in _STOPWORDS
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def ask_text(record: Mapping[str, Any]) -> str:
    """The ask's topic text: every question's ``header`` and ``question``."""
    parts: list[str] = []
    for question in record.get("questions") or ():
        if isinstance(question, Mapping):
            parts.append(str(question.get("header") or ""))
            parts.append(str(question.get("question") or ""))
    return " ".join(parts)


def similarity(title: str, body: str, ask: str) -> float:
    """The higher of Jaccard(title, ask) and Jaccard(title+body, ask)."""
    target = tokens(ask)
    return max(
        _jaccard(tokens(title), target),
        _jaccard(tokens(f"{title} {body}"), target),
    )


def same_topic(title: str, body: str, ask: str) -> bool:
    return similarity(title, body, ask) >= THRESHOLD


def matching_ask(
    records: Iterable[Mapping[str, Any]], *, title: str, body: str, now: int
) -> dict[str, Any] | None:
    """The newest OPEN, non-relay ask raised within the window on the same topic.

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
        if not same_topic(title, body, ask_text(record)):
            continue
        if best is None or created >= int(best.get("created_at") or 0):
            best = dict(record)
    return best


def matching_updates(
    rows: Iterable[Mapping[str, Any]], *, ask: str, now: int
) -> list[dict[str, Any]]:
    """Unread, undismissed updates touched within the window on the ask's topic."""
    found: list[dict[str, Any]] = []
    for row in rows:
        if row.get("read") or row.get("dismissed"):
            continue
        if now - int(row.get("updated_at") or 0) > WINDOW_MS:
            continue
        if same_topic(str(row.get("title") or ""), str(row.get("body_md") or ""), ask):
            found.append(dict(row))
    return found
