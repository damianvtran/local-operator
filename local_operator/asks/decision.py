"""Which verified decision a card answer makes — shared by every answer surface.

A question carrying a producer-stamped ``question_id`` (a Team Bridge request,
or a headless node's ``site:v1:`` browser approval) is decided ONLY through the
signed ``bridge_decide`` door; the plain ``ask_respond`` door refuses it. The
TUI and the desktop answer route both need to turn "the user picked Approve on
this card" into ``(ask_id, question_id, word)``, so the mapping lives here
rather than in either surface (it used to live in ``tui/app.py``, which the
server must not import).

It is also the SIGNER's last look at the card. Both surfaces hand the target to
a facade that signs it with the operator key (``attach_client``), so a
``site:v1:`` card is checked here (``site_approval.verify_site_card``) before a
target exists: a card whose words name a different site than its digest binds
raises :class:`~local_operator.browser_bridge.site_approval.SiteCardMismatch`
instead of being signed (review R1-1).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

#: How long a "this site card was checked" note waits for the signer to ask for
#: its challenge. The two happen in one user gesture — the card answer, then the
#: facade's ``bridge_decide`` asking the runtime for a challenge — so seconds
#: are plenty; the bound only keeps a note from outliving a gesture that failed
#: before it signed. 120 s is the runtime's own challenge TTL order.
_VERIFIED_TTL_S = 120.0
_VERIFIED_LOCK = threading.Lock()
_VERIFIED: dict[str, float] = {}


def note_site_card_verified(ask_id: str, question_id: str, word: str) -> None:
    """Record that THIS process checked the site card behind one decision.

    The signer half of the old-client guard (review U5,
    ``browser_bridge/site_approval.py``): the attach facade asks the node for a
    ``decide`` challenge with ``site_card_verified: true`` only for a target
    noted here, so a code path that reaches ``bridge_decide`` without going
    through :func:`bridge_decision_target` cannot sign a site card unchecked.
    """
    now = time.monotonic()
    with _VERIFIED_LOCK:
        for key, expiry in list(_VERIFIED.items()):
            if expiry <= now:
                del _VERIFIED[key]
        _VERIFIED[f"{ask_id}#{question_id}#{word}"] = now + _VERIFIED_TTL_S


def site_card_verified(request_id: str) -> bool:
    """Whether ``ask#question#word`` was noted by :func:`note_site_card_verified`.

    Single-use (popped): one checked card buys one challenge request, the same
    one-gesture-one-signature rule the facade's signature memo follows.
    """
    with _VERIFIED_LOCK:
        expiry = _VERIFIED.pop(str(request_id), None)
    return expiry is not None and expiry > time.monotonic()


def bridge_decision_target(
    row: Any, answers: Mapping[str, Sequence[str]]
) -> tuple[str, str, str] | None:
    """The ``(ask_id, question_id, word)`` a BRIDGE card answer decides.

    ``None`` for every ordinary ask: a decision exists only for a request whose
    question carries the producer's ``question_id`` AND whose picked value
    resolves to one of that question's own option ids. The word sent on the
    wire is the OPTION ID (``approve``/``deny``), never the drawn label — the
    door validates the word against the request's closed set, and a label is
    display copy that can be re-worded without touching the decision.

    Ambiguity is ``None`` rather than a guess: two picked request questions, or
    a value that maps to no option id, must fall back to the ordinary door and
    its refusal rather than decide something the human did not choose.

    A picked ``site:v1:`` question that its own ``site`` block does not rebuild
    RAISES ``SiteCardMismatch`` (a ``ValueError``) rather than returning
    ``None``: falling back to the plain door would only trade the honest
    refusal for a generic one. So does a site question that is not the ONLY
    question in its row (review R2-1): the node draws every question in the
    row, so a sibling could caption the site card with words the signer never
    checked ("the site question below grants docs.google.com only").
    """

    picked: list[tuple[str, str]] = []
    site_questions: list[Mapping[str, Any]] = []
    for question in row.questions:
        if not isinstance(question, Mapping):
            continue
        question_id = str(question.get("question_id") or "")
        if not question_id:
            continue
        selected = answers.get(str(question.get("id") or "")) or ()
        if len(selected) != 1:
            continue
        options = {
            str(option.get("label") or "").strip().lower(): str(option.get("id") or "")
            for option in (question.get("options") or ())
            if isinstance(option, Mapping)
        }
        value = str(selected[0]).strip()
        picked.append((question_id, options.get(value.lower()) or value.lower()))
        site_questions.append(question)
    if len(picked) != 1:
        return None
    from local_operator.browser_bridge.site_approval import (
        is_site_question_id,
        verify_site_card,
    )

    if is_site_question_id(picked[0][0]):
        from local_operator.browser_bridge.site_approval import SiteCardMismatch
        from local_operator.i18n.keys import browser_site_approval as _keys
        from local_operator.i18n.messages import render_resolved

        if len(tuple(row.questions)) != 1:
            raise SiteCardMismatch(render_resolved(_keys.refused_card_mismatch()))
        verify_site_card(site_questions[0])
        note_site_card_verified(str(row.ask_id), picked[0][0], picked[0][1])
    return (row.ask_id, picked[0][0], picked[0][1])
