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

    def enqueue(self, questions: Sequence[Any], timeout_raw: Any) -> dict[str, Any]:
        """Validate, cap, append ``queued``, arm the timer; return a receipt.

        Returns ``{"ok": True, "text": …, "details": …}`` or
        ``{"ok": False, "error": …}``. Every refusal is a validation error the
        model reads, never a silent clamp or a dropped ask: the caps exist
        because non-blocking asking is free for the model and costly for the
        human, and a model that never learns it was refused keeps re-asking.
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
            "tool_call_id": str(getattr(self._session, "_current_tool_call_id", "") or ""),
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
    ) -> dict[str, Any]:
        """Append ``answered`` for the whole ask, atomically, and reconcile.

        Single-winner by the log: the first ``answered`` to land is the one the
        fold keeps, so two surfaces racing produce one winner and the loser
        reads :func:`asks.render.refusal_copy` — the same rule ``_resolve_pending``
        enforced in memory, moved onto the durable record.
        """
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        refusal = render.refusal_copy(record)
        if refusal:
            return {"ok": False, "error": refusal}
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_ANSWERED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "answers": {str(k): [str(v) for v in (vals or [])] for k, vals in answers.items()},
        }
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
            # N8: a timeout row is only written when the response it announces
            # is NOT in the same batch (see the module docstring).
            if want_timeout and timeout_id not in present and not response_missing:
                timeouts.append(
                    (
                        int(record.get("expires_at") or 0),
                        self._timeout_message(record, now),
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

    def _response_message(self, record: Mapping[str, Any]) -> CustomMessage:
        text = render.response_text(record)
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
                "text": text,
            },
        )

    def _timeout_message(self, record: Mapping[str, Any], now_ms: int) -> CustomMessage:
        text = render.timeout_text(record, now_ms=now_ms)
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
