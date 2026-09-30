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

**THE TWO-ROW CASE.** A ``late`` ask needs BOTH a timeout row and a response
row, because the timeout genuinely fired before the answer arrived. When both
are missing in the same batch — a cold boot after an answer that arrived past
the deadline — only the RESPONSE is written (design §2.3, rule N8): replaying
"[Ask timed out] … you will be told" immediately before the answer it announces
reads to the model as a contradiction. Both orders are asserted by the tests.

**DARK BY DEFAULT.** Nothing here runs unless ``asks.policy.NONBLOCKING_ASK``
is on AND a host installed an ask surface (the same ``_ask_user`` hook that
makes the tool exist). With the flag off, ``Session`` never constructs this
object and every existing path is untouched (§5 invariant).
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
    """The session's queued asks: enqueue, answer, fold, deliver."""

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
        #: Row ids handed to delivery by THIS process. The transcript row is the
        #: cross-process, cross-restart marker; this set closes the window
        #: between handing a message to the session and the turn that persists
        #: it, inside which ``has_entry`` is still false and a re-entrant
        #: reconcile would hand the same row twice.
        self._handed: set[str] = set()

    # -- paths -------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        return store.session_dir(self._config_dir, self._session_id)

    # -- reading -----------------------------------------------------------

    def present_row_ids(self) -> set[str]:
        """Row ids already durable in this session's transcript.

        Constant-time per CANDIDATE id (``transcript.has_entry``) rather than a
        scan of every entry: the deadline tick calls this on every reconcile, and
        a long conversation's transcript is not something a one-minute timer
        should walk.
        """
        return self._present_row_ids(store.read_events(self.session_dir))

    def _present_row_ids(self, events: Sequence[Mapping[str, Any]]) -> set[str]:
        transcript = getattr(self._session, "transcript", None)
        has_entry = getattr(transcript, "has_entry", None)
        seen: set[str] = set(self._handed)
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
        cleaned = {str(k): [str(v) for v in (vals or [])] for k, vals in answers.items()}
        unanswered = [
            str(q.get("id"))
            for q in (record.get("questions") or ())
            if str(q.get("id")) not in cleaned
        ]
        if unanswered:
            return {"ok": False, "error": _partial_answer_error(unanswered)}
        cleaned = _guard_secret_cells(record, cleaned)
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

    def _settled(self, ask_id: str) -> None:
        """Post-write bookkeeping common to every terminal transition.

        Three effects, in order: the deadline row goes away (nothing to wake for),
        the derived index is rewritten (the cross-session view must not keep
        showing a settled ask as open), and a delivery is SCHEDULED — the ops are
        synchronous because they only append a row, so the reconcile they imply
        is scheduled rather than awaited.
        """
        self.retire_deadline_wake(ask_id)
        self._refresh()
        self._kick()

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

    async def reconcile(self, now_ms: int | None = None) -> list[str]:
        """Deliver every transcript row the current fold calls for; return their ids.

        Idempotent and level-triggered: call it at boot, at every turn start,
        after every answer/decline/dismiss, and from the deadline tick. It
        delivers at most one row per (ask, kind) because the row IS the marker.
        """
        if self._disposed:
            return []
        now = now_ms if now_ms is not None else self._now()
        # The stop rule (design §2.2): an ask whose deadline passed while the
        # session was STOPPED gets its notice annotated, because "Timed out"
        # arriving at reopen reads as the agent having been slow rather than as
        # nothing having been running. Read once per reconcile; ``None`` on the
        # ordinary path, so the flag-off behaviour and the live behaviour are the
        # same bytes.
        stopped_at = self._deliberate_stop_ms()
        records, present = self._fold_state(now)
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
            response_missing = want_response and response_id not in present
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
            if want_timeout and timeout_id not in present and not want_response:
                timeouts.append(
                    (
                        int(record.get("expires_at") or 0),
                        self._timeout_message(
                            record,
                            now,
                            lapsed_while_stopped=_lapsed_while_stopped(record, stopped_at),
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

    def _deliberate_stop_ms(self) -> int | None:
        """When this session was last deliberately stopped, in ms, or ``None``.

        ``runtime-stop.json`` is the durable evidence the stop leaves in the
        conversation directory (``registry.STOP_MARKER_NAME``, written by the
        party that acts), and it is the ONE artifact available at boot that says
        the session was stopped at a known time: the wake index's ``stopped_at``
        is cleared by the open-time rewrite, so a boot reconcile cannot read it.

        ``deliberate`` is required and that is not pedantry: the same file also
        records INvoluntary deaths (a supervisor's reap, a stray kill), and telling
        a user their session "was stopped" when what happened was a crash is the
        kind of wrong sentence that makes the honest ones worthless. The read is
        lazy and total: a missing module or a malformed marker is "no evidence",
        never an exception on a delivery path.
        """
        try:
            from local_operator.session.runtime.registry import read_stop_marker

            marker = read_stop_marker(self.session_dir)
        except Exception:  # noqa: BLE001 — no evidence, not a delivery failure
            logger.debug("ask: could not read the stop marker", exc_info=True)
            return None
        if not isinstance(marker, Mapping) or not marker.get("deliberate"):
            return None
        try:
            return int(float(marker.get("at") or 0) * 1000)
        except (TypeError, ValueError):
            return None

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
                "text": text,
            },
        )

    # -- the wire view and the derived index -------------------------------

    def projection(self, now_ms: int | None = None) -> list[dict[str, Any]]:
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
            rows.append(store.pending_row(record))
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


def _lapsed_while_stopped(record: Mapping[str, Any], stopped_at_ms: int | None) -> bool:
    """Whether this ask's DEADLINE fell inside a deliberate stop (design §2.2).

    The window is the ask's own: it has to have been created before the stop (or
    the stop did not interrupt it) and its deadline has to fall after the stop
    began (or the session was running again before the window closed). Both ends
    matter — the marker is durable and outlives the stop, so "a marker exists" is
    not the question, and neither is "the marker is recent".

    The false positives this could still admit need a deliberate stop INSIDE the
    ask's window, and then a second outage in which the deadline passed: the
    sentence is true in that case too, and the notice is owed at boot exactly
    because no live runtime delivered it.
    """
    if stopped_at_ms is None:
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
