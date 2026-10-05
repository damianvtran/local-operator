"""The per-session ask queue: caps, the fold, reconcile, and the deadline timer.

WHY THIS MODULE EXISTS (design ``docs/design/ask-nonblocking.md`` §2.2/§2.3).
``asks/store.py`` holds the durable truth and the pure fold; this is the object
that sits on a ``Session`` and turns that truth into (a) the receipts the tool
returns, (b) the transcript rows the model reads, and (c) the wire view a
surface paints. It is deliberately a plain class taking the session as a
collaborator rather than a mixin: nearly everything it needs — the transcript,
the steering queue, ``_prompt_messages`` — is session state, and passing the
session in keeps the queue testable with a small double instead of a Session.

**LEVEL-TRIGGERED, NEVER EVENT-DRIVEN.** ``reconcile`` reads the log, computes
the fold, and writes exactly the transcript rows that are MISSING for the
current status. There is no "mark delivered" write to lose in a crash: the row
itself is the marker (``transcript.has_entry``), so a delivery that happened
and a delivery that is about to happen are the same observation. That is what
makes boot-after-SIGKILL, answer-then-kill, and a cold timeout all converge on
exactly one row per (ask, kind).

**CONSUMPTION, NOT HAND-OFF (amended 2026-10-04).** That row counts as the
marker only once its append is DURABLE. Handing the message to a delivery path,
scheduling the turn and the answering op's ACK close nothing, so the revision
window (design §10) stays open until the model's conversation carries the
answer — and every delivery path re-resolves the row from the fold at its
append (``refresh_delivery_message``), so a revision accepted in between is
what lands. The in-process sets beside that flag are scheduling (``_handed``)
and the append-in-flight guard (``_committing``); neither is a reader's truth.

**THE TWO-ROW CASE.** A ``late`` ask needs BOTH a timeout row and a response
row, because the timeout genuinely fired before the answer arrived. When both
are missing in the same batch — a cold boot after an answer that arrived past
the deadline — only the RESPONSE is written (design §2.3, rule N8): replaying
"[Ask timed out] … you will be told" immediately before the answer it announces
reads to the model as a contradiction. Both orders are asserted by the tests.

**THE SHIPPED DEFAULT, AND STILL CONDITIONAL.** Nothing here runs unless
``asks.policy.NONBLOCKING_ASK`` is on AND a host installed an ask surface (the
same ``_ask_user`` hook that makes the tool exist). The flag is on unless the
operator sets the kill switch (``LOP_ASK_NONBLOCKING=0``), so on a normal host
the two conditions are the same one; with it off — or on a host that cannot
show a question — ``Session`` never constructs this object and every existing
path is untouched (§5 invariant). That is what keeps the blocking arm real
rather than vestigial, and it is why this class stays optional instead of
becoming unconditional.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from local_operator.asks import policy, render, store
from local_operator.harness.message_types import (
    ASK_RESPONSE_MESSAGE_TYPE,
    ASK_TIMEOUT_MESSAGE_TYPE,
)
from local_operator.harness.types import CustomMessage

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class AskQueue:
    """The session's queued asks: enqueue, answer, withdraw, fold, deliver."""

    def __init__(
        self,
        session: Any,
        *,
        config_dir: Path | str,
        session_id: str,
        cwd: str = "",
        clock: Any = None,
    ) -> None:
        self._session = session
        self._config_dir = Path(config_dir)
        self._session_id = session_id
        self._cwd = cwd
        #: Injectable clock (epoch ms). Only the FLOOR test needs a real one;
        #: everything else injects, because "wait two minutes and see" is not a
        #: test, it is a hope (AGENTS.md, timing section).
        self._now = clock or _now_ms
        self._timer: asyncio.Task[None] | None = None
        self._disposed = False
        #: Row ids handed to delivery by THIS process. SCHEDULING DEDUPE ONLY
        #: (amended 2026-10-04): it closes the gap between handing a message to
        #: the delivery path and the append that makes it durable, inside which
        #: ``has_entry`` is still false and a re-entrant reconcile would hand the
        #: same row twice. It is NOT the revision window and NOT the wire's
        #: ``delivered`` — those read the DURABLE row (``_present_row_ids``).
        #: Lifecycle: added at hand-off; dropped at the append outcome either
        #: way — success (the durable row carries it) or failure (the next
        #: reconcile re-hands, which is the retry the sticky form silently
        #: skipped).
        self._handed: set[str] = set()
        #: Row ids whose durable append is IN FLIGHT right now (set and cleared
        #: synchronously on the session loop by the session's append helper).
        #: The revision window is the durable row; this set closes the one true
        #: race the refresh cannot: content re-resolved at T0, append awaiting,
        #: a revision at T1>T0. Without the guard that revision would be
        #: ACCEPTED and then dropped by the append that already snapshotted —
        #: the interleaving §10 forbids by name — so a decision that sees the
        #: row here refuses in the delivered sentence (sub-ms, conservative).
        self._committing: set[str] = set()
        #: THE LEGACY DRAFT (design §4, A2 addendum): question id -> answer cell
        #: for an ask being answered ONE QUESTION AT A TIME by the old mirrored
        #: card. In-memory and per-runtime by design — it is not a durable fact
        #: (the log stays atomic, written once every question has an entry), and
        #: a runtime death simply returns the ask to its open state, which is the
        #: truthful outcome for a partial answer nobody finished submitting.
        self._drafts: dict[str, dict[str, list[str]]] = {}

    # -- paths -------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        return store.session_dir(self._config_dir, self._session_id)

    # -- reading -----------------------------------------------------------

    def present_row_ids(self) -> set[str]:
        """Row ids already DURABLE in this session's transcript.

        Constant-time per CANDIDATE id (``transcript.has_entry``) rather than a
        scan of every entry: the deadline tick calls this on every reconcile, and
        a long conversation's transcript is not something a one-minute timer
        should walk.

        DURABLE-ONLY, deliberately (amended 2026-10-04): ``_handed`` is the
        scheduling dedupe and is NOT folded in here — the revision window, the
        wire's ``delivered`` and every other reader mean CONSUMPTION, and only
        ``reconcile``'s own missing-row planning unions the dedupe locally.
        """
        return self._present_row_ids(store.read_events(self.session_dir))

    def _present_row_ids(self, events: Sequence[Mapping[str, Any]]) -> set[str]:
        transcript = getattr(self._session, "transcript", None)
        has_entry = getattr(transcript, "has_entry", None)
        seen: set[str] = set()
        if callable(has_entry):
            for ask_id in store.ask_ids(events):
                for row_id in (store.response_row_id(ask_id), store.timeout_row_id(ask_id)):
                    try:
                        if has_entry(row_id):
                            seen.add(row_id)
                    except Exception:  # noqa: BLE001 — unreadable is "absent"
                        pass
        return seen

    def _fold_state(self, now_ms: int) -> tuple[list[dict[str, Any]], set[str]]:
        """``(records, present row ids)`` from ONE read of the log."""
        events = store.read_events(self.session_dir)
        present = self._present_row_ids(events)
        return store.fold(events, now_ms, present_ids=present), present

    def records(self, now_ms: int | None = None) -> list[dict[str, Any]]:
        """The folded queue, newest-last, each with its ``delivered`` flag."""
        stamp = now_ms if now_ms is not None else self._now()
        return self._fold_state(stamp)[0]

    def open_records(self, now_ms: int | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records(now_ms) if r["status"] == store.STATUS_OPEN]

    # -- writing -----------------------------------------------------------

    def enqueue(
        self,
        questions: Sequence[Any],
        timeout_raw: Any,
        *,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """Validate, cap, append ``queued``, arm the timer; return a receipt.

        Returns ``{"ok": True, "text": …, "details": …}`` or
        ``{"ok": False, "error": …}``. Every refusal is a validation error the
        model reads, never a silent clamp or a dropped ask: the caps exist
        because non-blocking asking is free for the model and costly for the
        human, and a model that never learns it was refused keeps re-asking.

        ``tool_call_id`` is the model's own call id, passed IN rather than read
        off the session (review round 1, MINOR 5): the attribute it used to read
        does not exist anywhere in the tree, so every ask carried ``""`` and the
        card A2 draws could not point back at the call that asked.
        """
        timeout_s, bounds_error = policy.parse_timeout_param(timeout_raw)
        if bounds_error is not None:
            return {"ok": False, "error": bounds_error}
        now = self._now()
        records = self.records(now)
        open_rows = [r for r in records if r["status"] == store.STATUS_OPEN]
        if len(open_rows) >= policy.OPEN_ASK_CAP:
            return {
                "ok": False,
                "error": (
                    f"this session already has {len(open_rows)} open asks (the cap is "
                    f"{policy.OPEN_ASK_CAP}); wait for responses; do not re-ask."
                ),
            }
        shapes = [_question_shape(q) for q in questions]
        open_secret_keys = {
            str(q.get("id"))
            for r in open_rows
            for q in (r.get("questions") or [])
            if q.get("secret")
        }
        for shape in shapes:
            if shape["secret"] and shape["id"] in open_secret_keys:
                return {
                    "ok": False,
                    "error": (
                        f"a secret question for {shape['id']} is already open; wait for "
                        f"responses; do not re-ask."
                    ),
                }
        texts = {str(q.get("question") or "").strip() for q in shapes}
        for row in open_rows:
            for q in row.get("questions") or []:
                if str(q.get("question") or "").strip() in texts:
                    return {
                        "ok": False,
                        "error": (
                            "an open ask already carries that question text; wait for "
                            "responses; do not re-ask."
                        ),
                    }
        ask_id = self._mint_ask_id({r["ask_id"] for r in records})
        record = {
            "ask_id": ask_id,
            "created_at": now,
            "expires_at": now + timeout_s * 1000,
            "timeout_s": timeout_s,
            "urgent": policy.is_urgent(timeout_s),
            "tool_call_id": str(tool_call_id or ""),
            "questions": shapes,
        }
        event = {"v": store.EVENT_SCHEMA, "kind": store.EVENT_QUEUED, "at": now, **record}
        if not store.append_event(self.session_dir, event):
            return {
                "ok": False,
                "error": "the ask could not be recorded on this session's ask log.",
            }
        self._arm_deadline_wake(record)
        self._refresh(now)
        receipt = render.receipt_text(record, self._reach())
        details = {
            "ask_id": ask_id,
            "status": "queued",
            "question_ids": [q["id"] for q in shapes],
            "timeout_s": timeout_s,
            "expires_at": record["expires_at"],
            "urgent": record["urgent"],
            "secret": any(q["secret"] for q in shapes),
            "reach": self._reach(),
        }
        self.arm()
        return {"ok": True, "text": receipt, "details": details}

    def respond(
        self,
        ask_id: str,
        answers: Mapping[str, Sequence[str]],
        *,
        by: str = "unknown",
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """Append ``answered`` for the whole ask, atomically, and reconcile.

        Single-winner by the log: the first ``answered`` to land is the one the
        fold keeps, so two surfaces racing produce one winner and the loser
        reads :func:`asks.render.refusal_copy` — the same rule ``_resolve_pending``
        enforced in memory, moved onto the durable record.

        THIS IS THE ONE-WAY PATH, AND ITS ONE SANCTIONED EXCEPTION IS
        :meth:`revise` (design §10, #1936): a recorded answer can be changed ONLY
        by an explicit revision while it is still undelivered, never by a second
        ``respond``. A plain repeat — even with a different map — keeps its
        refusal here, because "a repeat tap is a retry, not a change of mind".

        TWO REFUSALS THAT ARE CONTRACTS RATHER THAN GUARDS. FIRST, the map must be
        COMPLETE (design §2.4, review round 1 QA Q1): every question id has to be
        present, because the row that lands is terminal — a surface that forgot a
        key would lose that question for good, and the user would have no way to
        tell. A question the user deliberately skipped is sent as an EMPTY LIST,
        which is how "no answer" is said without omitting the key, so the check is
        on the KEYS and not on the values. SECOND, for a SECRET question the
        recorded cell can only be a KEY NAME (review round 1, MINOR 6):
        :meth:`Session.respond_ask` substitutes the value before calling here, and
        this is what makes that hop load-bearing rather than merely the current
        caller's good manners — a cold CLI or route in B/C reaches the queue
        directly, and the value must not be able to reach the log through it.
        """
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        refusal = render.refusal_copy(record)
        if refusal:
            return {"ok": False, "error": refusal}
        cleaned, map_refusal = self._whole_ask_cells(record, answers)
        if map_refusal:
            return {"ok": False, "error": map_refusal}
        assert cleaned is not None  # ``_whole_ask_cells`` returns one with no refusal
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_ANSWERED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "answers": cleaned,
        }
        if tool_call_id or record.get("tool_call_id"):
            # The id of the tool call that QUEUED this ask, carried on the answer
            # so a card can link the two (review round 1, MINOR 5). The record's
            # own id is the fallback: the caller that answers a question asked in
            # an EARLIER turn has no tool call of its own to name.
            payload["tool_call_id"] = str(tool_call_id or record.get("tool_call_id") or "")
        if not store.append_event(self.session_dir, payload):
            return {"ok": False, "error": "the answer could not be recorded."}
        self._settled(ask_id)
        return {"ok": True}

    def revise(
        self,
        ask_id: str,
        answers: Mapping[str, Sequence[str]],
        *,
        by: str = "unknown",
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """SUPERSEDE a recorded answer while it is still UNDELIVERED (design §10).

        THE INTENT IS EXPLICIT, AND VALUE EQUALITY IS NEVER THE MARKER. A
        revision is its own op precisely so that :meth:`answer_one`'s rule — "a
        repeat tap is a retry, not a change of mind" — survives the retry case:
        a resent map is still refused there, and so is a different one. This is
        the single entry point that reads a second answer as a change of mind,
        and only while the change is still cheap.

        **THE WINDOW IS THE ROW'S DURABLE APPEND — consumption, not handoff**
        (amended 2026-10-04; it was the row's existence at hand-off, which the
        answering ACK raced). The single ``ask-response-<ask_id>`` row pins what
        the model reads, the transcript is append-only with one row and no
        rewrite, so the append IS the freeze and the last instant a revision can
        still change what the model will read (design §10). The wire's
        ``delivered`` reads the same fact: for ``answered``/``declined``/
        ``late`` it means exactly this row, so a durable deadline notice leaves
        a ``late`` answer reading ``false`` — the notice does not deliver the
        answer, and the window stays open exactly while it should. Handing the
        message to a delivery path, scheduling the turn and the answering op's
        ACK close nothing; a revision arriving while the append is in flight is
        refused too (``_committing``), conservatively: its content was already
        snapshotted, so accepting it would be accepted-and-then-dropped.

        ACCEPTED: the ask already carries an answer (``answered`` or ``late``)
        whose response row is not yet DURABLE — and not being appended right now
        — from ANY surface; a revision is not a race, so the single-winner rule
        that governs two in-flight ``respond``s is not a surface gate here.
        REFUSED once the response row is durable (or an append of it is in
        flight), in :data:`asks.render.REVISED_ALREADY_DELIVERED`'s words rather
        than the state table's "already answered by <surface>": the user's next
        move is a new message, not a different tap. ``declined``/``dismissed``/
        ``expired`` keep their state-mapped sentences — a revision of an answer
        that was never recorded is not what those states are.

        BEFORE any answer is recorded (``open``/``timed_out``) the intent degrades
        to the plain first answer, with ``revised: False`` on the verdict: the
        ``revised`` kind exists only to supersede, so with nothing to name the
        ask settles on one ``answered`` row and no revision event.

        THE RULE LIVES IN ``_revision_decision``, and :meth:`revision_refusal`
        exposes it read-only: a caller that must do work the refusal should
        suppress — ``Session.revise_ask`` stores a SECRET's value before the queue
        sees the map — asks the probe first rather than doing it speculatively for
        a revision this path is about to turn down.
        """
        now = self._now()
        record, refusal = self._revision_decision(ask_id, now)
        if refusal:
            return {"ok": False, "error": refusal}
        assert record is not None  # ``_revision_decision`` returns one with no refusal
        if record.get("status") in (store.STATUS_OPEN, store.STATUS_TIMED_OUT):
            outcome = self.respond(ask_id, answers, by=by, tool_call_id=tool_call_id)
            if outcome.get("ok"):
                outcome["revised"] = False
            return outcome
        cleaned, map_refusal = self._whole_ask_cells(record, answers)
        if map_refusal:
            return {"ok": False, "error": map_refusal}
        assert cleaned is not None  # ``_whole_ask_cells`` returns one with no refusal
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_REVISED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "answers": cleaned,
            # The ``at`` of the write this supersedes: the newest revision's when
            # one exists, else the first answer's. Naming it keeps the log a chain
            # a reader can walk rather than a pile of claims about one moment.
            "supersedes": int(record.get("revised_at") or record.get("answered_at") or 0),
        }
        if not store.append_event(self.session_dir, payload):
            return {"ok": False, "error": "the revision could not be recorded."}
        self._settled(ask_id)
        return {"ok": True, "revised": True}

    def revision_refusal(self, ask_id: str, now_ms: int | None = None) -> str:
        """Why a revision of this ask would be REFUSED, or ``""`` if it would be taken.

        The READ-ONLY half of :meth:`revise`'s STATE decision, split out for the
        one caller with a side effect it must not perform speculatively:
        :meth:`Session.revise_ask` stores a SECRET answer's value before the queue
        sees the map, and a revision the STATE window is going to refuse must not
        be the reason a pasted credential is stored and announced to later turns.
        The decision itself is ``_revision_decision``, shared with the write path,
        so the two can never disagree BY RULE — the sentence a caller sees here is
        the sentence the write's state decision would return.

        THE STATE DECISION IS ALL OF WHAT THIS PROBE COVERS, and that limit is
        worth stating here because it was claimed wider: the write path applies a
        SECOND rule afterwards — the whole-ask map contract, which refuses a map
        that omits a question — and this probe does not run it. A revision refused
        for an INCOMPLETE map therefore still stores a secret the caller supplied
        before the refusal. That is pre-existing and identical on
        :meth:`Session.respond_ask`, it is recoverable (the resend lands) and no
        value reaches a durable surface, and it is recorded as a DEFERRED finding
        on PR #1954; the remedy is to consult the map contract here too, the same
        shape as this probe. Do not read this docstring as covering it.

        THE CLOCK IS THE CALLER'S, and that is the one place they can part in
        fact: ``revise`` reads ``now`` again for its own fold, so a probe and a
        write straddling the ``LATE_WINDOW_S`` horizon — where an ask moves to
        ``expired`` and the verdict flips from admissible to refused — can land on
        opposite sides of it. Every other state transition is verdict-neutral
        (``open``/``timed_out`` are both admissible, ``answered``/``late`` are the
        same window), so that 7-day boundary is the whole of the difference, and
        it is why this is documented rather than papered over with a guard.

        It writes nothing and delivers nothing: the fold the write path reads, and
        the sentence it would return.
        """
        _record, refusal = self._revision_decision(
            ask_id, self._now() if now_ms is None else now_ms
        )
        return refusal

    def _revision_decision(self, ask_id: str, now_ms: int) -> tuple[dict[str, Any] | None, str]:
        """``(the folded record, the refusal sentence or "")`` for a revision.

        THE ONE ADMISSIBILITY RULE (design §10, amended 2026-10-04): an ask that
        already carries an answer (``answered``/``late``) whose
        ``ask-response-<ask_id>`` row is not yet DURABLE — and not being appended
        right now — is admissible, from ANY surface; a revision is not a race, so
        the single-winner rule is not a surface gate. A CONSUMED one is refused in
        the revision path's own words; ``declined``/``dismissed``/``expired`` keep
        the state table's sentences; an ask with no answer yet (``open``/
        ``timed_out``) is admissible as a plain first answer.

        THE WINDOW IS THE ROW'S DURABLE APPEND, and this check is exactly that
        term: ``present`` is durable-only (`_present_row_ids`) plus the COMMIT
        GUARD (``_committing``) for the append currently in flight — a revision
        arriving while the append awaits cannot be carried afterwards (the
        content was snapshotted) and must not be accepted-and-dropped, so it is
        refused in the delivered sentence, conservatively and for milliseconds.
        Hand-off, scheduling and the answering op's ACK close nothing.
        """
        records, present = self._fold_state(now_ms)
        record = next((item for item in records if item["ask_id"] == ask_id), None)
        if record is None:
            return None, render.refusal_copy(None)
        status = record.get("status")
        if status in (store.STATUS_OPEN, store.STATUS_TIMED_OUT):
            return record, ""
        if status not in (store.STATUS_ANSWERED, store.STATUS_LATE):
            # declined / dismissed / expired: the state table's own sentence,
            # byte-for-byte the one every other path gives for that state.
            return record, render.refusal_copy(record)
        response_id = store.response_row_id(ask_id)
        if response_id in present or response_id in self._committing:
            # The row's durable append is the consumption point (this module's
            # docstring; design §10), so this is the one check that means "the
            # model's conversation carries the answer" — or is carrying it as we
            # speak (the guard), which the no-accept-and-drop rule prices the
            # same way.
            return record, render.REVISED_ALREADY_DELIVERED
        return record, ""

    def _whole_ask_cells(
        self, record: Mapping[str, Any], answers: Mapping[str, Sequence[str]]
    ) -> tuple[dict[str, list[str]] | None, str]:
        """The whole-ask map contract, shared by :meth:`respond` and :meth:`revise`.

        Returns ``(cleaned cells, refusal sentence)``, the refusal empty when the
        map is usable. Both rules are the design's (§2.4, review round 1 QA Q1 and
        MINOR 6) and every entry point must agree on them — ``respond``,
        ``revise`` and :meth:`withdraw`'s chat path: the row that lands is
        terminal, so a surface that forgot a key would lose that question for
        good, and a SECRET cell may only ever be a KEY NAME: the ``[<key>]`` shape
        is what makes this hop load-bearing, because a cold CLI or relay route
        reaches the queue directly and the value must not reach the log through
        it. Kept in ONE function for the same reason ``refusal_copy`` is one
        sentence per state: two copies of a contract drift.
        """
        cleaned = {str(k): [str(v) for v in (vals or [])] for k, vals in answers.items()}
        unanswered = [
            str(q.get("id"))
            for q in (record.get("questions") or ())
            if str(q.get("id")) not in cleaned
        ]
        if unanswered:
            return None, _partial_answer_error(unanswered)
        return _guard_secret_cells(record, cleaned), ""

    def answer_one(
        self,
        ask_id: str,
        question_id: str,
        values: Sequence[str],
        *,
        by: str = "unknown",
    ) -> dict[str, Any]:
        """LEGACY INCREMENTAL ANSWER: one question of an ask, settled later.

        The old mirrored card is a per-question flow (the blocking path advanced
        one question at a time), so an old client can only ever send one cell.
        :meth:`respond` — the NEW whole-ask path — deliberately refuses a partial
        map, because the modern ops are atomic per ask and a partial submit is a
        lost race rather than a step. This method is the bridge between the two:
        it merges one cell into a per-ask draft, refuses a SECOND answer for the
        same question (a repeat tap is a retry, not a change of mind), and hands
        the completed draft to :meth:`respond` so the ask still settles in ONE
        atomic log write. The one sanctioned way to CHANGE a recorded answer is
        :meth:`revise`, and it never passes through here: the draft path keeps its
        refusal exactly so a retry stays a retry (design §10).

        Returns ``{"ok": True, "settled": bool, "waiting": [qid, ...]}`` for an
        accepted cell — ``waiting`` names the questions the card should offer
        next — or ``{"ok": False, "error": …}`` with the same refusal copy every
        other answer path uses.
        """
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        refusal = render.refusal_copy(record)
        if refusal:
            return {"ok": False, "error": refusal}
        ids = [str(q.get("id") or "") for q in (record.get("questions") or ())]
        key = str(question_id)
        if key not in ids:
            return {
                "ok": False,
                "error": f"{key!r} is not a question on ask {ask_id}.",
            }
        draft = self._drafts.setdefault(ask_id, {})
        if key in draft:
            return {
                "ok": False,
                "error": f"{key!r} is already answered on ask {ask_id}; "
                "answer the questions still waiting.",
            }
        draft[key] = [str(item) for item in (values or ())]
        waiting = [qid for qid in ids if qid not in draft]
        if waiting:
            # Publish the partial so the mirrored card advances to the next
            # question: the draft is what ``mirror_card`` reads to choose it.
            self._refresh(now)
            return {"ok": True, "settled": False, "waiting": waiting}
        outcome = self.respond(ask_id, draft, by=by)
        if not outcome.get("ok"):
            # A refusal at settle time (a lost race against another surface)
            # discards the draft with it: keeping it would let the next tap
            # settle an ask the log has already closed.
            self._drafts.pop(ask_id, None)
            return outcome
        return {"ok": True, "settled": True, "waiting": []}

    def draft_question_ids(self, ask_id: str) -> list[str]:
        """The question ids with an in-flight legacy answer, sorted."""
        return sorted(self._drafts.get(str(ask_id), {}))

    def decline(self, ask_id: str, *, by: str = "unknown") -> dict[str, Any]:
        """Terminal-on-write: append ``declined`` (today's Esc, made explicit)."""
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        refusal = render.refusal_copy(record)
        if refusal:
            return {"ok": False, "error": refusal}
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_DECLINED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
        }
        if not store.append_event(self.session_dir, payload):
            return {"ok": False, "error": "the decline could not be recorded."}
        self._settled(ask_id)
        return {"ok": True}

    def dismiss(self, ask_id: str, *, by: str = "unknown") -> dict[str, Any]:
        """View-only removal of a timed-out ask. Injects nothing, ever."""
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        if record["status"] != store.STATUS_TIMED_OUT:
            return {
                "ok": False,
                "error": "only a timed-out ask can be dismissed; it is still open.",
            }
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_DISMISSED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
        }
        if not store.append_event(self.session_dir, payload):
            return {"ok": False, "error": "the dismissal could not be recorded."}
        self._settled(ask_id)
        return {"ok": True}

    def withdraw(
        self,
        ask_id: str,
        *,
        reason: str,
        answers: Mapping[str, Sequence[str]] | None = None,
        message_id: str | None = None,
        by: str = "agent",
    ) -> dict[str, Any]:
        """THE ASKER'S OWN SETTLE (design §12): retract a question, or record the
        user's chat answer as one. Returns ``{"ok": True, "text", "details"}``
        or ``{"ok": False, "error": <sentence>}``, the same shape every op here
        returns.

        Every other write to the log is authored by a SURFACE on the user's
        behalf — :meth:`respond`, :meth:`revise`, :meth:`decline`, :meth:`dismiss`.
        Before this method the ASKER had no path, and two real shapes fell
        through: an ask whose answer stopped mattering sat ``open`` →
        ``timed_out`` → expiry, besetting every list as a question nobody will
        answer (``reason="moot"``); and a user who simply replied in the
        transcript left the ask reading "waiting" although the agent already had
        the answer (``reason="answered_in_chat"``).

        TWO REASONS, TWO ROWS, ONE APPEND EACH:

        * ``moot`` appends ``withdrawn`` — terminal-on-write, injects NOTHING
          (symmetric with ``dismissed``). The fold caveat lives in
          :func:`asks.store.fold`: every user-act terminal row present
          (``answered``, ``declined``, ``dismissed``) outranks a later
          ``withdrawn``, in either write order, because the user's own acts are
          never overridden by the asker's retraction. This method does not lean
          on that caveat for its own refusals — it checks the fold and refuses a
          settled ask in :func:`render.withdraw_refusal`'s words — so the caveat
          is belt-and-braces for the true cross-process race only.
        * ``answered_in_chat`` records through the same ``answered`` machinery
          :meth:`respond` uses: the cells are the user's words VERBATIM, one
          cell per question the message answers and an EMPTY LIST for a question
          it does not cover (the §2.4 completeness contract is on the KEYS, so a
          forgotten key is refused rather than lost), and ``by`` names the
          evidence — ``{"surface": "chat", "message_id": ...}``. The standard
          response row follows through ``reconcile``. A SECRET question is
          REFUSED for this reason with its own sentence: there is no
          masked-entry hop from chat text and there must not be one — the card
          stays the only secret path.

        Admissible exactly where an answer is: an open or timed-out ask (a
        post-deadline answer folds to ``late`` as it would from any surface); a
        settled ask is refused with no state change.

        RUNTIME-LOCAL, AGENT-ONLY. No new remote wire op and no relay change:
        the model's tool call runs inside this session, and
        ``Session.withdraw_ask`` wraps this method the way ``respond_ask`` wraps
        :meth:`respond`. The only wire-visible effect is the new folded status
        travelling out through the existing index/frame flow.
        """
        if reason not in ("moot", "answered_in_chat"):
            return {"ok": False, "error": render.WITHDRAW_BAD_REASON}
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        status = str(record.get("status") or "")
        settled = status not in (store.STATUS_OPEN, store.STATUS_TIMED_OUT)
        if reason == "moot":
            if answers:
                # Refused rather than ignored: a model that meant to record the
                # user's words and passed the wrong reason must hear it, because
                # the words would otherwise be silently discarded by the
                # withdrawal it did ask for. An EMPTY map passes deliberately —
                # it is "no answers", so there are no words for this guard to
                # protect (review round 1, NIT 5).
                return {"ok": False, "error": render.WITHDRAW_MOOT_TAKES_NO_ANSWERS}
            if settled:
                return {"ok": False, "error": render.withdraw_refusal(record, reason)}
            payload = {
                "v": store.EVENT_SCHEMA,
                "kind": store.EVENT_WITHDRAWN,
                "ask_id": ask_id,
                "at": now,
                "by": {"surface": by},
            }
            if not store.append_event(self.session_dir, payload):
                return {"ok": False, "error": "the withdrawal could not be recorded."}
            self._settled(ask_id)
            return {
                "ok": True,
                "text": render.withdraw_receipt(ask_id, reason),
                "details": {"ask_id": ask_id, "reason": reason},
            }
        if settled:
            return {"ok": False, "error": render.withdraw_refusal(record, reason)}
        if any(question.get("secret") for question in (record.get("questions") or ())):
            return {"ok": False, "error": render.WITHDRAW_SECRET_REFUSAL}
        cells, map_refusal = self._whole_ask_cells(record, answers or {})
        if map_refusal:
            return {"ok": False, "error": map_refusal}
        assert cells is not None  # ``_whole_ask_cells`` returns one with no refusal
        by_map: dict[str, Any] = {"surface": "chat"}
        if message_id:
            by_map["message_id"] = str(message_id)
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_ANSWERED,
            "ask_id": ask_id,
            "at": now,
            "by": by_map,
            "answers": cells,
        }
        if record.get("tool_call_id"):
            # The same carry as :meth:`respond`: a card can link the answer to
            # the tool call that queued the ask.
            payload["tool_call_id"] = str(record.get("tool_call_id"))
        if not store.append_event(self.session_dir, payload):
            return {"ok": False, "error": "the answer could not be recorded."}
        self._settled(ask_id)
        return {
            "ok": True,
            "text": render.withdraw_receipt(ask_id, reason),
            "details": {"ask_id": ask_id, "reason": reason},
        }

    def _settled(self, ask_id: str) -> None:
        """Post-write bookkeeping common to every terminal transition.

        Three effects, in order: the deadline row goes away (nothing to wake for),
        the derived index is rewritten (the cross-session view must not keep
        showing a settled ask as open), and a delivery is SCHEDULED — the ops are
        synchronous because they only append a row, so the reconcile they imply
        is scheduled rather than awaited.
        """
        self.retire_deadline_wake(ask_id)
        # A settled ask has no partial left to remember: the draft either just
        # became the log's answers or the ask was declined/dismissed out from
        # under it, and keeping it would advance a card for a closed ask.
        self._drafts.pop(str(ask_id), None)
        self._refresh()
        self._kick()

    def _reclaim_closed_drafts(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Drop in-flight taps for asks the FOLD no longer calls ``open``.

        The deadline is a DERIVED status (``store.fold`` reads the clock), so
        nothing is written when it elapses and :meth:`_settled` — the reclaim
        site for the three transitions that DO append a row — never runs for it.
        Reclaiming on the fold instead is what keeps a tap from outliving the ask
        it belongs to, and ``reconcile`` is the path a deadline is observed on:
        the armed ``ask_timeout`` wake fires it, and the boot drain reconciles
        with it. A draft is not a durable fact, so dropping it returns the ask to
        the state its log already states — closed.
        """
        for record in records:
            if record.get("status") != store.STATUS_OPEN:
                self._drafts.pop(str(record.get("ask_id")), None)

    def _kick(self) -> None:
        """Ask the session to reconcile soon. Best-effort, no loop required.

        With no running loop — a cold CLI answer against a finished session —
        there is nothing to schedule and nothing to do: the next runtime's boot
        reconcile is the same path the answer takes anyway, which is exactly what
        makes the queue session-agnostic.
        """
        spawn = getattr(self._session, "_spawn_background", None)
        if not callable(spawn):
            return
        try:
            spawn(self.reconcile())
        except Exception:  # noqa: BLE001 — the boot reconcile is the backstop
            logger.debug("ask: could not schedule a reconcile", exc_info=True)

    # -- delivery ----------------------------------------------------------

    async def reconcile(self, now_ms: int | None = None, *, load_time: bool = False) -> list[str]:
        """Deliver every transcript row the current fold calls for; return their ids.

        Idempotent and level-triggered: call it at boot, at every turn start,
        after every answer/decline/dismiss, and from the deadline tick. It
        delivers at most one row per (ask, kind) because the row IS the marker.

        ``load_time`` IS THE REOPEN FACT, and it is the only caller that arms the
        stop annotation (review round 2, M1). The design scopes the "lapsed while
        the session was stopped" sentence to the reopen (§2.2, §7: "on reopen the
        load-time reconcile delivers overdue notices annotated…"), and this is the
        same call on three other paths where the session is RUNNING by definition —
        the deadline tick (``arm``), the wake fire
        (:meth:`Session._deliver_wake`) and every turn start. A deadline that
        elapses on any of those did not lapse while anything was stopped, so
        annotating it would be false in both halves: nothing was lapsed by the
        stop, and no notice was withheld.
        """
        if self._disposed:
            return []
        now = now_ms if now_ms is not None else self._now()
        # Read ONLY on the load-time reconcile; ``None`` everywhere else, which is
        # what keeps the live paths off the marker read entirely.
        marker = self._deliberate_stop_marker() if load_time else None
        records, present = self._fold_state(now)
        # SCHEDULING PLANS WITH THE DEDUPE, EVERYONE ELSE DOES NOT (amended
        # 2026-10-04): ``present`` is durable-only now, so the missing-row
        # checks below union ``_handed`` LOCALLY. A row handed to a delivery
        # path whose append has not run yet must not be re-planned — that is
        # the double-hand this set exists to stop — while the window, the wire
        # and every other reader keep meaning CONSUMPTION.
        planned = present | self._handed
        # Before anything is delivered or published: this is the fold that can
        # show an ask CLOSED by its deadline, and a tap for it must not ride the
        # rows this reconcile goes on to publish (see
        # :meth:`_reclaim_closed_drafts` for why the deadline needs its own
        # reclaim site).
        self._reclaim_closed_drafts(records)
        timeouts: list[tuple[int, CustomMessage]] = []
        responses: list[tuple[int, CustomMessage]] = []
        for record in records:
            status = record["status"]
            if status not in store.INJECTING_STATUSES:
                continue
            response_id = store.response_row_id(record["ask_id"])
            timeout_id = store.timeout_row_id(record["ask_id"])
            want_response = status in (
                store.STATUS_ANSWERED,
                store.STATUS_DECLINED,
                store.STATUS_LATE,
            )
            want_timeout = status in (store.STATUS_TIMED_OUT, store.STATUS_LATE)
            response_missing = want_response and response_id not in planned
            # N8, AND IT IS PERMANENT RATHER THAN A ONE-BATCH DEFERRAL (review
            # round 1, MAJOR 2). A response row for this ask — present, or being
            # written in THIS batch — supersedes its deadline row for good. The
            # one-shot form only deferred the contradiction: after the response
            # landed, the next reconcile (and reconcile runs again immediately
            # after every answer, at every turn start and on the tick) wrote the
            # ``ask-timeout-`` row for an ask the model had just been told the
            # answer to, so it read "[Ask timed out] ... if they answer later you
            # will be told" immediately below the answer itself. A deadline that
            # genuinely fired first is unaffected: that is the ``timed_out``
            # status, where no response exists yet, and it is delivered in its own
            # reconcile. A ``late`` ask therefore carries ONE row, the response,
            # whose lead already says the window had closed — see
            # :data:`asks.render.LATE_LEAD` and :func:`store.expected_row_ids`.
            if want_timeout and timeout_id not in planned and not want_response:
                timeouts.append(
                    (
                        int(record.get("expires_at") or 0),
                        self._timeout_message(
                            record,
                            now,
                            lapsed_while_stopped=_lapsed_while_stopped(
                                record, marker, self._session_id
                            ),
                        ),
                    )
                )
            if response_missing:
                responses.append(
                    (int(record.get("created_at") or 0), self._response_message(record))
                )
        messages: list[CustomMessage] = []
        # Timeouts first: within a batch they are the older event, and a notice
        # that predates the answer it sits beside reads in the order it happened.
        for _, message in sorted(timeouts, key=lambda item: item[0]):
            messages.append(message)
        for _, message in sorted(responses, key=lambda item: item[0]):
            messages.append(message)
        if messages:
            for message in messages:
                self._handed.add(message.id)
                await self._emit_delivered(message)
            await self._session.deliver_ask_messages(messages)
        self._refresh(now)
        self.arm()
        return [m.id for m in messages]

    # -- the append side (design §2.3 as amended 2026-10-04) -----------------
    #
    # THE WINDOW CLOSES AT THE ROW'S DURABLE APPEND, which happens in the
    # SESSION (the four delivery paths call ``Session._append_injected_message``
    # around it). These four methods are that append's queue-side contract, and
    # nothing else in this module touches ``_committing`` or ``_handed``:

    def begin_row_commit(self, row_id: str) -> None:
        """Mark an ask row's durable append as IN FLIGHT (the commit guard).

        Set SYNCHRONOUSLY on the session loop before the append awaits, from
        the same loop that runs every other writer of the row — the discipline
        the serving handle's ``@_on_session_loop`` hop makes load-bearing. From
        here until :meth:`finish_row_commit`, a revision decision that sees
        this id refuses in the delivered sentence: the append snapshotted
        content already, so accepting one would be accepted-and-then-dropped.
        """
        self._committing.add(str(row_id))

    def finish_row_commit(self, row_id: str, *, durable: bool) -> None:
        """The append resolved: release the guard, settle the dedupe, publish.

        ``_handed`` drops on BOTH outcomes, and that is the lifecycle rather
        than an accident: success — the durable row now carries the id, so a
        re-hand would be deduped by ``has_entry`` anyway; failure — nothing
        carries it, and sticking would silently never retry a row a human is
        waiting to see (the pre-existing hole the sticky set had). On success
        the fold's ``delivered`` flag has just flipped — at the append, LATER
        than any reconcile publish — so the index and the wire are rewritten
        through :meth:`note_row_durable`.
        """
        row = str(row_id)
        self._committing.discard(row)
        self._handed.discard(row)
        if durable:
            self.note_row_durable(row)

    def note_row_durable(self, row_id: str) -> None:
        """The transcript now holds ask row ``row_id``: publish the new fold.

        Called by the session's append helper on every successful ask-row
        append. The ``delivered`` flag is CONSUMPTION-based now, so it flips at
        the append — after the reconcile that handed the message published its
        (still false) fold — and without this write the wire would keep
        reporting ``delivered:false`` for a row the model can already read.
        """
        self._refresh()

    def refresh_delivery_message(self, message: CustomMessage) -> CustomMessage:
        """Rebuild an ask-RESPONSE row from the CURRENT fold, at the append.

        THE MESSAGE RECONCILE BUILT IS A PREVIEW. The window closes at the
        durable append, so a revision accepted between the reconcile that built
        this message and the append that lands it must be what the append
        carries — otherwise the window would have "accepted" a change and
        delivered the stale map, the accepted-and-then-dropped interleaving §10
        forbids by name. Delivery is the last reader; the fold is the truth it
        re-derives from (latest ``answers``/``text``/``status``/``secret_lost``,
        same id — one response row per ask, ever).

        A record the fold no longer shows as an injectable response (the ask
        moved to a state with no response row in the very milliseconds since
        reconcile) is returned UNCHANGED: the reconcile that emitted this
        message already decided the row is owed, and dropping it here would be
        a silent loss — never accepted-and-then-dropped cuts both ways.
        Timeout rows need no rebuild (nothing about them is revisable) and are
        passed through by the caller.
        """
        details = getattr(message, "details", None)
        ask_id = str((details or {}).get("ask_id") or "") if isinstance(details, dict) else ""
        if not ask_id:
            return message
        record = self._find(ask_id, self._now())
        if record is None or record.get("status") not in (
            store.STATUS_ANSWERED,
            store.STATUS_DECLINED,
            store.STATUS_LATE,
        ):
            return message
        return self._response_message(record)

    async def _emit_delivered(self, message: CustomMessage) -> None:
        """Emit the paint-ahead event BEFORE the turn it triggers (design §2.3)."""
        from local_operator.harness.types import (
            AskResponseDeliveredEvent,
            AskTimeoutDeliveredEvent,
        )

        emit = getattr(self._session, "_emit", None)
        if not callable(emit):
            return
        details = message.details or {}
        if message.custom_type == ASK_TIMEOUT_MESSAGE_TYPE:
            event: Any = AskTimeoutDeliveredEvent(
                text=str(details.get("text") or ""),
                ask_id=str(details.get("ask_id") or ""),
                urgent=bool(details.get("urgent")),
            )
        else:
            event = AskResponseDeliveredEvent(
                text=str(details.get("text") or ""),
                ask_id=str(details.get("ask_id") or ""),
                status=str(details.get("status") or ""),
            )
        try:
            # ``_emit`` is probed with getattr, so what it returns is not
            # statically known to be awaitable; the codebase's established
            # spelling for that is ``inspect.isawaitable`` rather than a cast.
            result = emit(event)
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — a paint event must never fail a delivery
            logger.warning("ask: could not emit delivery event", exc_info=True)

    def _deliberate_stop_marker(self) -> dict[str, Any] | None:
        """This conversation's DELIBERATE stop marker, in ms-stamped form, or ``None``.

        ``runtime-stop.json`` is the durable evidence a stop leaves in the
        conversation directory (``registry.STOP_MARKER_NAME``, written by the party
        that acts), and it is the ONE artifact available at boot that says the
        session was stopped at a known time: the wake index's ``stopped_at`` is
        cleared by the open-time rewrite, so a boot reconcile cannot read it.

        ``deliberate`` is required and that is not pedantry: the same file also
        records INvoluntary deaths (a supervisor's reap, a stray kill), and telling
        a user their session "was stopped" when what happened was a crash is the
        kind of wrong sentence that makes the honest ones worthless.

        THE SESSION KEY IS CHECKED HERE, and it is the run-covers rule the
        classification reader applies (``attention._stop_marker_covers_run``): a
        marker is keyed to a conversation and to a RUN, so a marker belonging to
        another conversation must never narrate this one's ask — which matters
        because the file is read from a directory the queue was merely handed.
        Nothing else is left once the load-time gate has excluded the live paths,
        and that residue is stated at :func:`_lapsed_while_stopped` rather than
        implied.

        The read is lazy and total: a missing module or a malformed marker is "no
        evidence", never an exception on a delivery path.
        """
        try:
            from local_operator.session.runtime.registry import read_stop_marker

            raw = read_stop_marker(self.session_dir)
        except Exception:  # noqa: BLE001 — no evidence, not a delivery failure
            logger.debug("ask: could not read the stop marker", exc_info=True)
            return None
        if not isinstance(raw, Mapping) or not raw.get("deliberate"):
            return None
        if str(raw.get("session_id") or "") != str(self._session_id or ""):
            return None
        try:
            at_ms = int(float(raw.get("at") or 0) * 1000)
        except (TypeError, ValueError):
            return None
        # ``session_id`` rides along so the decision site can re-assert the same
        # clause against the id the QUEUE was built with, rather than trusting a
        # reader that already passed it: the two spellings of one check are what
        # makes a future second caller safe.
        return {"at_ms": at_ms, "session_id": str(raw.get("session_id") or "")}

    def _response_message(self, record: Mapping[str, Any]) -> CustomMessage:
        lost = self._secret_answer_lost(record)
        text = render.response_text(record, secret_lost=lost)
        return CustomMessage(
            custom_type=ASK_RESPONSE_MESSAGE_TYPE,
            attribution="user",
            id=store.response_row_id(str(record["ask_id"])),
            details={
                "ask_id": record["ask_id"],
                "status": record["status"],
                "questions": [dict(q) for q in (record.get("questions") or [])],
                "answers": {k: list(v) for k, v in (record.get("answers") or {}).items()},
                "at": record.get("answered_at") or record.get("at") or 0,
                # ``secret_lost`` is stated on the ROW as well as in the text: a
                # surface renders the fact (the card must not say "in hand" for a
                # credential the session no longer holds) and the model reads the
                # sentence it belongs to.
                "secret_lost": lost,
                "tool_call_id": str(record.get("tool_call_id") or ""),
                "text": text,
            },
        )

    def _secret_answer_lost(self, record: Mapping[str, Any]) -> bool:
        """Whether a secret this response announces is GONE from the session store.

        The verify-on-delivery half of design §2.4 (review round 1, MAJOR 3, and §8
        risk 3 — the one path where the user answered and the agent still cannot
        proceed). Session credentials live in memory unless ``persist`` promoted
        them, so a restart between an answer and its delivery leaves a row that
        announces a key the runtime does not hold; a model reading it proceeds to
        use a credential that is not there. Delivering
        :data:`render.SECRET_VALUE_LOST` instead is the difference between a bad
        turn and a plausible-looking wrong one.

        Fail-OPEN in one direction only: with no store to ask (a queue built
        without a session, the unit shapes) the answer is "not lost", because a
        gate that fired on an unanswerable question would decorate every secret
        response in every embedder. A key that WAS declined (the not-provided
        sentinel) is not "lost" either — nothing was ever handed over.
        """
        questions = [q for q in (record.get("questions") or ()) if q.get("secret")]
        if not questions:
            return False
        names = getattr(self._session, "credential_names", None)
        if not callable(names):
            variables = getattr(self._session, "_variables", None)
            names = getattr(variables, "credential_names", None)
        if not callable(names):
            return False
        # ``Any`` is deliberate, here and at the call: ``callable()`` narrows an
        # untyped attribute to ``Callable[..., object]``, and ``object`` is not
        # iterable to the checker even though every real answer is a list of
        # names — the same spelling the tool layer uses for its probed callables.
        reader: Any = names
        try:
            held = {str(name).upper() for name in reader()}
        except Exception:  # noqa: BLE001 — an unreadable store is not a "lost" claim
            logger.debug("ask: could not read the credential names", exc_info=True)
            return False
        answers = record.get("answers") or {}
        for question in questions:
            cell = [str(v) for v in (answers.get(str(question.get("id"))) or ())]
            key = cell[0].strip() if cell else ""
            if not key or key == _secret_not_provided():
                continue
            if key.upper() not in held:
                return True
        return False

    def _timeout_message(
        self,
        record: Mapping[str, Any],
        now_ms: int,
        *,
        lapsed_while_stopped: bool = False,
    ) -> CustomMessage:
        text = render.timeout_text(record, now_ms=now_ms, lapsed_while_stopped=lapsed_while_stopped)
        waited_s = max(0, now_ms - int(record.get("created_at") or now_ms)) // 1000
        return CustomMessage(
            custom_type=ASK_TIMEOUT_MESSAGE_TYPE,
            attribution="system",
            id=store.timeout_row_id(str(record["ask_id"])),
            details={
                "ask_id": record["ask_id"],
                "status": store.STATUS_TIMED_OUT,
                "waited_s": waited_s,
                "urgent": bool(record.get("urgent")),
                "lapsed_while_stopped": lapsed_while_stopped,
                # The questions ride the row so a transcript CARD can expand onto
                # what went unanswered (design §5.1: "collapsed one-liner,
                # expandable Q&A"). They are the same dicts the response row and
                # the wire carry, so a card cannot disagree with the picker about
                # what was asked; additive, and a reader that ignores it (the
                # phone's notice fold does) is unaffected.
                "questions": [dict(q) for q in (record.get("questions") or ())],
                "text": text,
            },
        )

    # -- the wire view and the derived index -------------------------------

    def projection(
        self, now_ms: int | None = None, *, drafts: bool = False
    ) -> list[dict[str, Any]]:
        """The ``PendingAsk`` rows for a surface, open first, capped.

        A2 publishes this on the frontend state; it is built here so the fold is
        computed once per change rather than per surface.

        THE FILTER IS THE SAME RULE AS THE INDEX'S STALENESS SWEEP, and that
        agreement is load-bearing: the fold reports an IN-WINDOW answer as
        ``answered`` for good (rule 1 is terminal), so a projection that kept
        every non-``expired`` row would have this writer re-create an entry the
        reader sweeps as stale — the two would disagree forever about whether a
        week-old answered ask is still worth showing. One horizon, both sides.
        """
        stamp = now_ms if now_ms is not None else self._now()
        horizon_ms = store.LATE_WINDOW_S * 1000
        rows: list[dict[str, Any]] = []
        for record in self.records(stamp):
            status = record["status"]
            if status in (store.STATUS_EXPIRED, store.STATUS_DISMISSED):
                continue
            expires_at = int(record.get("expires_at") or 0)
            if expires_at and stamp - expires_at > horizon_ms:
                continue
            # A DRAFT IS PUBLISHED ONLY FOR AN OPEN ASK (review round 2's one new
            # minor). Two reasons, and the first is the one that shows on the
            # wire: the mirrored card that consumes this field is open-only, so
            # the tap names a question nobody can still be offered; and the
            # deadline is a DERIVED status, so between the deadline elapsing and
            # the next ``reconcile`` the row already folds as ``timed_out`` while
            # the tap is still in memory — publishing it there would state an
            # in-flight answer for an ask the user can no longer answer.
            draft = (
                self._drafts.get(str(record.get("ask_id")))
                if drafts and status == store.STATUS_OPEN
                else None
            )
            rows.append(store.pending_row(record, draft))
        rows.sort(
            key=lambda row: (
                row.get("status") != store.STATUS_OPEN,
                -row.get("created_at", 0),
            )
        )
        return rows[: policy.PROJECTION_CAP]

    def _refresh(self, now_ms: int | None = None) -> None:
        """Rewrite the derived index. Best-effort by contract, like ``wakes/``."""
        now = now_ms if now_ms is not None else self._now()
        try:
            store.write_entry(
                self._config_dir,
                self._session_id,
                cwd=self._cwd,
                asks=self.projection(now),
            )
        except Exception:  # noqa: BLE001 — the log is the truth; the index heals
            logger.warning(
                "ask index: could not write entry for %s", self._session_id, exc_info=True
            )
        self._publish_state()

    def _publish_state(self) -> None:
        """Hand the new fold to the host's wire publisher, if it has one.

        ``_refresh`` is the ONE place the queue's visible state changes —
        enqueue, answer, decline, dismiss and every reconcile all end here — so
        it is also where the wire has to be told (design §4: the frontend state,
        the projection and the list rows publish the fold the moment it moves).
        A PROBE rather than a direct call, for the reason every other optional
        hook on this object is one: the session double in this package's tests
        has no publisher, and a queue must not require a wire to exist.
        """
        publish = getattr(self._session, "publish_ask_state", None)
        if not callable(publish):
            return
        try:
            publish()
        except Exception:  # noqa: BLE001 — a wire is never worth failing the log
            logger.debug("ask: could not publish the ask state", exc_info=True)

    # -- the deadline wake row ---------------------------------------------

    def _arm_deadline_wake(self, record: Mapping[str, Any]) -> None:
        """Arm the hidden ``ask_timeout`` wake row for a cold runtime (§2.2).

        The row is what makes a deadline survive a runtime that does not exist:
        the wake supervisor engages a runtime for it, boot ``reconcile`` then
        delivers the notice, and the fire itself is stale once the ask is
        terminal — the patience "watermark" rule, so no cross-process row
        deletion is needed.
        """
        session = self._session
        if getattr(session, "_wake", None) is None:
            return
        try:
            from local_operator.harness.wake_types import WakeSchedule

            row = WakeSchedule(
                id=store.timeout_row_id(str(record["ask_id"])),
                message=f"ask {record['ask_id']} deadline",
                next_due_at=int(record["expires_at"]),
                created_at=self._now(),
                kind="ask_timeout",
                hidden=True,
            )
            # The session owns ``_wake`` and its persist-and-arm path, so it owns
            # the write; the queue only mints the row (see ``Session.arm_ask_wake``).
            session.arm_ask_wake(row)
        except Exception:  # noqa: BLE001 — degrade to the in-runtime timer
            logger.warning("ask: could not arm the deadline wake row", exc_info=True)

    def retire_deadline_wake(self, ask_id: str) -> None:
        """Drop an ask's deadline row once it is terminal.

        Not required for correctness (the fire is stale by then) but it keeps
        the visible wake count and the index honest, and the supervisor from
        engaging a runtime for a deadline that has already been decided.
        """
        if getattr(self._session, "_wake", None) is None:
            return
        try:
            self._session.retire_ask_wake(store.timeout_row_id(ask_id))
        except Exception:  # noqa: BLE001 — best-effort, like every wake-index writer
            logger.warning("ask: could not retire the deadline wake row", exc_info=True)

    # -- the in-runtime timer ----------------------------------------------

    def arm(self) -> None:
        """Ensure exactly ONE deadline timer task exists for this session."""
        if self._disposed:
            return
        if self._timer is not None and not self._timer.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; the next reconcile (from a turn or boot) arms
        self._timer = loop.create_task(self._run_timer())

    async def _run_timer(self) -> None:
        """Sleep to the earliest deadline (≤60 s granularity), then reconcile.

        One task, not one per ask: N asyncio timers for N asks is the shape the
        design rejects, and a re-check on a bounded tick absorbs clock changes
        rather than arming a multi-hour timeout.
        """
        try:
            while not self._disposed:
                opens = [
                    int(r["expires_at"])
                    for r in self.records()
                    if r["status"] in (store.STATUS_OPEN,)
                ]
                if not opens:
                    return
                delay_s = min(policy.MAX_TICK_MS, max(0, min(opens) - self._now())) / 1000.0
                await asyncio.sleep(max(0.05, delay_s))
                if self._disposed:
                    return
                await self.reconcile()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 — a timer must never kill the session
            logger.warning("ask: deadline timer failed", exc_info=True)

    def dispose(self) -> None:
        self._disposed = True
        task = self._timer
        self._timer = None
        if task is not None and not task.done():
            task.cancel()

    # -- helpers -----------------------------------------------------------

    def _reach(self) -> str | None:
        probe = getattr(self._session, "ask_reach", None)
        if not callable(probe):
            return None
        try:
            value = probe()
        except Exception:  # noqa: BLE001 — an unreachable probe is "unreachable"
            return None
        if not value:
            return None
        if isinstance(value, (list, tuple, set, frozenset)):
            names = [str(v) for v in value]
            return ", ".join(names) if names else None
        return str(value)

    def find(self, ask_id: str, now_ms: int | None = None) -> dict[str, Any] | None:
        """The folded record for ``ask_id``, or ``None`` when this log never had it.

        ``None`` is its own answer (``render.refusal_copy``): a surface that
        tapped an id this session has never seen — a stale screen, or a card from
        another session — must be told exactly that rather than "already
        answered", which would be a claim about work nobody did.
        """
        now = now_ms if now_ms is not None else self._now()
        for record in self.records(now):
            if record["ask_id"] == ask_id:
                return record
        return None

    def _find(self, ask_id: str, now_ms: int) -> dict[str, Any] | None:
        return self.find(ask_id, now_ms)

    def _mint_ask_id(self, taken: Iterable[str]) -> str:
        used = set(taken)
        for _ in range(64):
            candidate = store.new_ask_id()
            if candidate not in used:
                return candidate
        return store.new_ask_id()


def _secret_not_provided() -> str:
    """The cell value that says the user declined to hand a secret over.

    Imported lazily from the tool module that owns it rather than re-spelled: the
    two must be the same string, or a declined secret would read as a lost one.
    """
    from local_operator.tools.builtin import ASK_SECRET_NOT_PROVIDED

    return ASK_SECRET_NOT_PROVIDED


def _credential_key_shape(text: str) -> str:
    """``text`` in the shape ``variables.normalize_credential_key`` produces.

    Re-spelled rather than imported because this module is on the harness's lean
    side of the import graph (``session.variables`` pulls the session engine in),
    and the property being compared is the normaliser's whole contract: letters
    and digits kept, every other run folded to one ``_``, upper-cased. The spelling
    is pinned against the real function by ``tests/unit/asks/test_queue.py`` so a
    drift here fails rather than silently widening the secret guard.
    """
    return "_".join(part for part in re.split(r"[^A-Za-z0-9]+", text.strip()) if part).upper()


def _secret_cell_ok(cell: Sequence[str], qid: str) -> bool:
    """Whether ``cell`` for secret question ``qid`` can only be a KEY NAME.

    The positive test is deliberately narrow — a single token with no whitespace
    whose key shape matches the question id's — because the thing being excluded
    is a pasted SECRET, and a secret is arbitrary bytes: it has no reason to equal
    the id it was asked under, while the key name the credential store returns for
    a question is that id in exactly this shape. The value is never logged,
    returned or quoted on the refusal path either.
    """
    head = str(cell[0]).strip() if cell else ""
    if not head or any(char.isspace() for char in head):
        return False
    return _credential_key_shape(head) == _credential_key_shape(qid)


def _guard_secret_cells(
    record: Mapping[str, Any], answers: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Drop anything that is not a key name from a SECRET question's cell.

    Defence in depth for the one hop that must never carry a value (review round
    1, MINOR 6). :meth:`Session.respond_ask` substitutes the key BEFORE calling
    :meth:`AskQueue.respond`, and this is what keeps that order load-bearing: a
    direct caller — a cold CLI or relay route in B/C is the expected one — gets
    "the user did not provide it" rather than a line in the log that outlives the
    session. The cell is replaced WHOLE rather than filtered element by element,
    because a leaked value sitting beside a legitimate key would be just as
    durable.
    """
    out = dict(answers)
    for question in record.get("questions") or ():
        if not question.get("secret"):
            continue
        qid = str(question.get("id"))
        if qid not in out:
            continue
        cell = list(out[qid])
        if cell and cell[0] == _secret_not_provided():
            continue
        if not _secret_cell_ok(cell, qid):
            logger.warning(
                "ask %s: refused a value-shaped answer for secret question %s",
                record.get("ask_id"),
                qid,
            )
            out[qid] = [_secret_not_provided()]
    return out


def _partial_answer_error(missing: Sequence[str]) -> str:
    """The refusal for an answer map that omits questions (design §2.4).

    NAMES the missing ids and says what to do instead: the submit is atomic, so a
    surface that has only some of the answers must wait rather than settle the ask
    — and a question the user deliberately skipped is sent as an empty list, which
    is how "no answer" is said while keeping the map complete.
    """
    listed = ", ".join(f"{qid!r}" for qid in missing)
    return (
        f"every question must be answered at once: {listed} has no entry. "
        "Send an empty list for a question the user skipped — a partial answer "
        "would settle the ask and lose the rest."
    )


def _lapsed_while_stopped(
    record: Mapping[str, Any], marker: Mapping[str, Any] | None, session_id: str
) -> bool:
    """Whether this ask's DEADLINE fell inside a deliberate, THIS-conversation stop.

    Three facts, and the caller supplies the fourth. ``marker`` is the run-covers
    checked marker (:meth:`AskQueue._deliberate_stop_marker`): deliberate, and
    named for this conversation. ``session_id`` is passed again so the check is
    visible at the decision site rather than only at the reader.

    THE WINDOW IS THE ASK'S OWN: it has to have been created before the stop (or
    the stop did not interrupt it) and its deadline has to fall after the stop
    began (or the session was running again before the window closed). Both ends
    matter — the marker is durable and outlives the stop, so "a marker exists" is
    not the question, and neither is "the marker is recent".

    THE FOURTH FACT IS THE LOAD-TIME GATE, and it is held by the CALLER rather
    than here (``reconcile(load_time=True)``): the sentence is owed only on the
    reopen, so this predicate is never consulted on the live deadline tick, the
    wake-fire path or a turn start — where the session is running by definition
    and the deadline therefore did not lapse while anything was stopped. With that
    gate in place the two remaining ends bracket the deadline into the stopped
    interval: created no later than the stop, deadline no later than the reopen
    (which is the moment the load-time reconcile is running).

    WHAT IS NOT CHECKED, stated so it is a boundary rather than a surprise. A
    marker is not withdrawn by a successful reopen, so a marker from an EARLIER
    stop survives into later runs; if a run in between passed through the deadline
    and the session then booted again, both window ends still hold and this would
    annotate a lapse that was actually live. The queue has no run key to close
    that: the marker's ``pid``/``started_at`` name a process the queue never saw,
    and the alternative — the transcript's per-run ``attention_started`` entry —
    would put the attention classifier on this package's import path, which
    ``asks/__init__`` and the import-graph pin exist to keep off. Recorded as a
    known residue rather than papered over; the common shape (stop → reopen → pick
    up) is covered, and the reviewer's false case (stop → reopen before the
    deadline → deadline elapses live) is excluded by the gate.
    """
    if marker is None:
        return False
    if str(marker.get("session_id") or "") != str(session_id or ""):
        return False
    stopped_at_ms = marker.get("at_ms")
    if not isinstance(stopped_at_ms, int) or not stopped_at_ms:
        return False
    created_at = int(record.get("created_at") or 0)
    expires_at = int(record.get("expires_at") or 0)
    return bool(created_at) and created_at <= stopped_at_ms <= expires_at


def _as_mapping(value: Any) -> dict[str, Any]:
    """``value`` as a plain dict: ``model_dump()`` when it has one, else itself.

    Written as an explicit two-branch probe rather than ``dict(value)`` for the
    same reason ``wakes/store._as_mapping`` is: ``dict()`` on an untyped value
    picks an overload the checker cannot verify, and this module is stdlib-only
    so it must not import pydantic to name the type it is unwrapping.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _question_shape(question: Any) -> dict[str, Any]:
    """One question as the plain dict the stdlib-only log stores.

    The log must not carry pydantic objects (``asks/store.py`` imports nothing of
    ours), and the projection deliberately carries the FULL question — options,
    multi, secret, persist — because a surface that can only see the id cannot
    draw a picker and would have to re-derive the ask from the model's tool call.
    """
    raw = _as_mapping(question)
    options: list[dict[str, Any]] = []
    for option in raw.get("options") or ():
        options.append(_as_mapping(option))
    return {
        "id": str(raw.get("id") or ""),
        "question": str(raw.get("question") or ""),
        "options": options,
        "multi": bool(raw.get("multi")),
        "recommended": raw.get("recommended"),
        "secret": bool(raw.get("secret")),
        "persist": bool(raw.get("persist")),
    }


__all__ = ["AskQueue"]
