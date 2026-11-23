"""The host's consent seam for Team Integrations mutations (RFC §8.2, S-82..S-87).

WHY A HOST SEAM, NOT A TOOL CHECK. Whether a person answered an ask is a fact
only the session holds: ``Session._run_answered_ask_ids`` is filled from rows
the session itself classified as a person's answer (``ask_answer_is_person``:
``attribution="user"`` and ``status == "answered"``), and it is reset every
run. A tool reading the ask log alone could not tell an answer the human gave
on a card from one an agent settled with ``ask_withdraw(answered_in_chat)`` or
from a timeout — so the session hands the tool this object and the tool cannot
reach consent any other way. It is the ``hub_pull`` ladder
(``Session._authorize_hub_pull``) applied to a different, narrower grant.

THE LADDER, for one action key (``integrations.action_key``):

1. GRANTED — a person answered, IN THIS RUN, a host-filed ask whose consent
   carries this action key, with the affirmative label persisted at filing
   time, the record is ``answered`` (not ``late``: the server commitment the
   ask froze has the same 15-minute life as the card), and the grant was not
   spent. It is spent HERE, before the caller makes the server call — an
   unrecorded spend could be spent twice, so a failed spend refuses.
2. DECLINED — the person picked the decline label, or the ask was declined
   ("I won't answer now") or dismissed. Nothing is re-asked in this run:
   RFC U5 state 4 is "zero retry".
3. QUEUED — an open ask already carries this key: reuse it, never file a
   second card for the same action.
4. TIMED OUT / NOT HUMAN — the ask lapsed, was withdrawn (``answered_in_chat``
   or ``moot``: an agent's claim, never a human answer), or was answered in an
   earlier run and never spent. The caller reports it and re-asks only when
   the model passes ``reask=true`` (RFC U5 state 5: preserve the draft, re-ask
   when the user returns).
5. NONE — no ask for this key: the caller obtains a commitment and files one
   through :meth:`file`.

A teammate's message, a peer, a typed "yes" (it is neither label) and any text
in the transcript are never consent: nothing here reads message text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from local_operator.team_bridge.integrations import (
    ASK_TIMEOUT_S,
    CONSENT_OP,
    QUESTION_ID,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsentOutcome:
    """The seam's answer. ``frozen`` is the consent record filed with the ask."""

    state: str  # granted | declined | queued | timed_out | not_human | none | unavailable
    ask_id: str = ""
    frozen: Mapping[str, Any] = field(default_factory=dict)
    reused: bool = False
    detail: str = ""


class IntegrationConsent:
    """Per-turn view over this session's ask queue and its person-answered ids."""

    def __init__(
        self,
        queue_getter: Callable[[], Any],
        answered_ids: Callable[[], set[str]],
        *,
        session_id: str = "",
    ) -> None:
        self._queue_getter = queue_getter
        self._answered_ids = answered_ids
        self.session_id = session_id

    def _matching(self, queue: Any, key: str, ask_id: str) -> list[dict[str, Any]]:
        out = []
        for record in queue.records():
            consent = record.get("consent")
            if not isinstance(consent, Mapping) or consent.get("op") != CONSENT_OP:
                continue
            if ask_id and record.get("ask_id") != ask_id:
                continue
            # A cited ask must still be FOR this action: an ask_id the model
            # remembers from a different mint is not consent for this one.
            if consent.get("action_key") != key:
                continue
            out.append(record)
        return out

    def lookup(self, key: str, *, ask_id: str = "") -> ConsentOutcome:
        queue = self._queue_getter()
        if queue is None:
            return ConsentOutcome("unavailable", detail="this session has no ask queue")
        records = self._matching(queue, key, ask_id)
        answered_now = self._answered_ids()
        # Newest first, so a fresh re-ask wins over an older lapsed one.
        records.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
        for record in records:
            rid = str(record.get("ask_id") or "")
            consent = record["consent"]
            status = str(record.get("status") or "")
            picks = [str(v) for v in (record.get("answers") or {}).get(QUESTION_ID) or ()]
            if record.get("consent_used"):
                continue
            if status in ("answered", "late") and picks == [str(consent.get("decline") or "")]:
                return ConsentOutcome("declined", rid, dict(consent), detail="cancel")
            if status == "answered" and rid in answered_now:
                if picks != [str(consent.get("affirm") or "")]:
                    # Typed text: neither yes nor no. Not consent; say so.
                    return ConsentOutcome("not_human", rid, dict(consent), detail="typed_answer")
                if not queue.mark_consent_used(rid):
                    return ConsentOutcome(
                        "unavailable", rid, detail="the grant could not be recorded"
                    )
                return ConsentOutcome("granted", rid, dict(consent))
            if status == "open":
                return ConsentOutcome("queued", rid, dict(consent), reused=True)
            if status in ("declined", "dismissed"):
                return ConsentOutcome("declined", rid, dict(consent), detail=status)
            if status == "withdrawn":
                return ConsentOutcome("not_human", rid, dict(consent), detail="withdrawn")
            if status in ("timed_out", "expired", "late"):
                return ConsentOutcome("timed_out", rid, dict(consent), detail=status)
            if status == "answered":
                # Answered by a person in an EARLIER run and never spent. The
                # commitment it froze is most likely dead; re-ask.
                return ConsentOutcome("timed_out", rid, dict(consent), detail="stale_answer")
        return ConsentOutcome("none")

    def file(
        self,
        question: Mapping[str, Any],
        consent: Mapping[str, Any],
        *,
        tool_call_id: str = "",
    ) -> ConsentOutcome:
        queue = self._queue_getter()
        if queue is None:
            return ConsentOutcome("unavailable", detail="this session has no ask queue")
        body = {"op": CONSENT_OP, **dict(consent)}
        receipt = queue.enqueue(
            [dict(question)], ASK_TIMEOUT_S, tool_call_id=tool_call_id, consent=body
        )
        if not receipt.get("ok"):
            logger.warning("integration consent ask not filed: %s", receipt.get("error"))
            return ConsentOutcome("unavailable", detail=str(receipt.get("error") or "ask refused"))
        return ConsentOutcome("queued", str(receipt["details"]["ask_id"]), body)


__all__ = ["ConsentOutcome", "IntegrationConsent"]
