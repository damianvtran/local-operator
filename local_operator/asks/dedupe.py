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

THE RULE: "THE UPDATE ADDS NOTHING THE ASK DOES NOT ALREADY SAY" (:func:`adds_nothing`).
Deterministic, cheap, no model call, and deliberately ASYMMETRIC. Round 1 used a
symmetric Jaccard overlap and it was wrong in the costly direction: "Core
production deploy failed" scored 0.75 against the ask "Deploy core to
production?" and a high-priority outage was swallowed. An overlap measure asks
"same subject?", and an outcome about the SAME subject (failed, released, a
metric) is exactly the update the operator must see. So the test is now a
subset test over content words:

* never for ``priority="high"`` — a banner-worthy update is always delivered;
* every content word of the update's title AND body must already appear in the
  ask's text (question sentences, option labels, question ids), after dropping
  stopwords and a short list of pure FRAMING words ("waiting", "decision",
  "needed" ...) that restate that a question exists;
* and at least :data:`MIN_SHARED` content words must be shared, so a near-empty
  update does not vacuously match.

One new word — ``failed``, ``released``, ``ctr``, a number — keeps the update.
No stemming, on purpose: stemming would make "released" equal the ask's
"release" and turn an outcome back into a restatement. Tokens of 2+ characters
are kept (``pr``, ``ci``, ``ip``) and every digit run counts, so "PR #3094: CI
failed" adds ``ci``/``failed`` and is delivered. The failure mode left is a
missed merge (today's behaviour: one extra click), never a hidden result.

WHERE THE TEXT GOES. A suppressed update is never written; its text rides the
ask as context. On the ask-after-update path the earlier update is marked READ
with ``via="merged_into_ask"`` but NOT dismissed: it stays in the inbox's All
view, so a surface that does not yet render an ask's context (the phone's
answer sheet, the TUI's narrow layout) cannot lose it.

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

#: The fewest content words an update must share with the ask before it can
#: count as a restatement (an update of one shared word is not evidence).
MIN_SHARED = 2

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

_TOKEN = re.compile(r"[a-z0-9]+")

#: Function words that carry no topic. Short on purpose: every word removed
#: here makes an update look more like a restatement, never less.
_STOPWORDS = frozenset(
    "the and for you your are was were this that with from have has had not but "
    "can will should would could did does into onto about what which when where "
    "who why how all any now then than them they its our out use is it to of on "
    "in at be do or an as by if so we me my no".split()
)

#: Words that only restate THAT a question is pending. Removing them is what
#: lets "Waiting on you: branch feat/x" count as adding nothing to "Delete
#: branch feat/x?". Never an outcome or state word.
_FRAMING = frozenset(
    "waiting wait question decision decide needed need needs input approval "
    "approve answer reply pending please choose choice call confirm".split()
)


def tokens(text: str) -> frozenset[str]:
    """Content tokens: lower-case, 2+ chars (any digit run counts), no stopwords."""
    return frozenset(
        word
        for word in _TOKEN.findall(str(text or "").lower())
        if (len(word) >= 2 or word.isdigit()) and word not in _STOPWORDS
    )


def ask_text(record: Mapping[str, Any]) -> str:
    """The ask's topic text, from the fields ``AskQuestion`` actually carries.

    The question sentence, every option's label, and the question ``id`` with
    its separators split (ids are model-written and often topic-bearing, e.g.
    ``rollback_decision``).
    """
    parts: list[str] = []
    for question in record.get("questions") or ():
        if not isinstance(question, Mapping):
            continue
        parts.append(str(question.get("question") or ""))
        parts.append(str(question.get("id") or "").replace("_", " ").replace("-", " "))
        for option in question.get("options") or ():
            if isinstance(option, Mapping):
                parts.append(str(option.get("label") or ""))
    return " ".join(parts)


def novel_words(title: str, body: str, ask: str) -> frozenset[str]:
    """The update's content words the ask does not already say."""
    return tokens(f"{title} {body}") - tokens(ask) - _FRAMING


def adds_nothing(title: str, body: str, ask: str, *, priority: str = "normal") -> bool:
    """Whether this update only restates the ask (see the module docstring)."""
    if priority == "high":
        return False
    shared = (tokens(f"{title} {body}") - _FRAMING) & tokens(ask)
    return len(shared) >= MIN_SHARED and not novel_words(title, body, ask)


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
        if not adds_nothing(title, body, ask_text(record), priority=priority):
            continue
        if best is None or created >= int(best.get("created_at") or 0):
            best = dict(record)
    return best


def matching_updates(
    rows: Iterable[Mapping[str, Any]], *, ask: str, now: int
) -> list[dict[str, Any]]:
    """Unread, undismissed updates touched within the window that only restate the ask."""
    found: list[dict[str, Any]] = []
    for row in rows:
        if row.get("read") or row.get("dismissed"):
            continue
        if now - int(row.get("updated_at") or 0) > WINDOW_MS:
            continue
        if adds_nothing(
            str(row.get("title") or ""),
            str(row.get("body_md") or ""),
            ask,
            priority=str(row.get("priority") or "normal"),
        ):
            found.append(dict(row))
    return found
