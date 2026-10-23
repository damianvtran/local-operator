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
import hashlib
import inspect
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from local_operator.asks import policy, render, store
from local_operator.harness.message_types import (
    ASK_RELAY_MESSAGE_TYPE,
    ASK_RESPONSE_MESSAGE_TYPE,
    ASK_TIMEOUT_MESSAGE_TYPE,
)
from local_operator.harness.types import CustomMessage, ImageContent
from local_operator.team_bridge.ask_issuance import bridge_scope_digest

logger = logging.getLogger(__name__)

#: The most images ONE answer may carry, across all its questions. The same bound
#: the HTTP door (``Answer.images``) and a composer send (``Prompt.images``) use;
#: it is restated here because the queue is also reached directly (the TUI's
#: in-process owner, a cold relay engage) and the cap must hold wherever the
#: bytes enter, not only where the HTTP model happens to sit in front of them.
MAX_ANSWER_IMAGES = 8

#: The refusal for a revision that carries images (design D5: v1 revisions are
#: text only). One sentence, shared by the queue and the session hop in front of
#: it, so the wire and the dock say the same thing.
REVISION_IMAGES_REFUSED = (
    "a revision carries text only: the images already sent stay as they were and "
    "cannot be changed"
)


def _now_ms() -> int:
    return int(time.time() * 1000)


#: The statuses a relay's settle note is owed for (#2520). ``timed_out`` is
#: DELIBERATELY absent: it is lapsed, not terminal — the origin still takes a
#: late answer, and the note's row id is a one-shot marker, so delivering there
#: would leave the watcher permanently misinformed if that answer arrives. The
#: watcher is told when the pair can no longer move: an answer (or its absence,
#: at expiry), a decline, a dismissal, or a retraction.
_RELAY_SETTLED_STATUSES = frozenset(
    {
        store.STATUS_ANSWERED,
        store.STATUS_DECLINED,
        store.STATUS_DISMISSED,
        store.STATUS_WITHDRAWN,
        store.STATUS_EXPIRED,
        store.STATUS_LATE,
    }
)

#: The USER'S OWN ACTS in the fold's §12 carve-out, applied across the pair: an
#: answer, decline, dismissal or late answer on the ORIGIN outranks the watcher's
#: retraction of its relay, in either write order — the same rule
#: ``store.fold`` applies within one log.
_RELAY_USER_ACT_TERMINALS = frozenset(
    {store.STATUS_ANSWERED, store.STATUS_DECLINED, store.STATUS_DISMISSED, store.STATUS_LATE}
)


def _lapsed_visible_ms(config_dir: Path) -> int:
    """The ``asks.lapsed_visible_for`` window, for the deadline timer.

    A module-level seam so a test can pin the window without a ``config.yml``;
    the import is local because ``visibility`` reaches the config watcher, which
    this module's other paths never need.
    """
    from local_operator.asks.visibility import lapsed_visible_ms

    return lapsed_visible_ms(config_dir)


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
        #: Set by :meth:`arm` while a timer is alive: its current plan may miss a
        #: deadline added since it was made (see :meth:`_run_timer`).
        self._timer_stale = False
        #: Tasks currently inside :meth:`reconcile` (see the note there).
        self._reconciling: set[asyncio.Task[Any]] = set()
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
                for row_id in (
                    store.response_row_id(ask_id),
                    store.timeout_row_id(ask_id),
                    # The relay settle note (#2520) is a durable row of this
                    # engine too: without it here, every reconcile would re-plan
                    # a note it had already appended. One extra has_entry per
                    # ask, the same cost class as the two existing rows.
                    store.relay_row_id(ask_id),
                ):
                    try:
                        if has_entry(row_id):
                            seen.add(row_id)
                    except Exception:  # noqa: BLE001 — unreadable is "absent"
                        pass
        return seen

    def _fold_state(self, now_ms: int) -> tuple[list[dict[str, Any]], set[str]]:
        """``(records, present row ids)`` from ONE read of the log.

        Relay rows are resolved AFTER the local fold (#2520): their status,
        questions and answers live in the ORIGIN session's log, so the fold here
        produces the reference and this second pass fills it in. It is the only
        cross-session read in the engine, and it is done on every read rather
        than cached: the origin is a file another process appends to, and a
        cached derivation would report an answer the user has already given.
        """
        events = store.read_events(self.session_dir)
        present = self._present_row_ids(events)
        records = store.fold(events, now_ms, present_ids=present)
        self._apply_relay_derivation(records, now_ms)
        return records, present

    def _apply_relay_derivation(self, records: list[dict[str, Any]], now_ms: int) -> None:
        """Fill each relay record in from its ORIGIN's fold (#2520).

        ONE hop only, deliberately. A relay of a relay is refused at creation
        (``RELAY_OF_RELAY``), so there is no chain to follow; an origin record
        that is itself a relay can only come from a hand-edited log, and it is
        treated as gone rather than dereferenced — the alternative is an
        unbounded walk with no answerable end.

        The ORIGIN'S RULES DECIDE, not the watcher's: questions are copied
        VERBATIM (``AskQueue.respond``'s whole-ask contract means the watcher
        cannot reword what the human is answering), the deadline and urgency are
        the origin's, and the status is the origin's. The LOCAL OVERRIDES are
        the watcher's own settles — a ``withdrawn`` row (the asker retracting
        its relay) and a ``dismissed`` row (the user closing a lapsed card) —
        under the fold's §12 carve-out applied across the pair: a user act on
        the origin (answered, declined, dismissed, late) outranks them in either
        order.

        An unresolvable origin SETTLES the relay (``origin_gone``, ``expired``):
        a pointer with nothing behind it is not outstanding work, and leaving it
        open would make the watcher's list show a question no one can answer.
        A local settle already on the row is kept over that ``expired``: the
        card was closed before the pointer broke, and re-opening it to say the
        origin is gone would be the same dropped decision R1-1 fixed.
        """
        for record in records:
            ref = record.get("relay")
            if not isinstance(ref, Mapping):
                continue
            # THE WATCHER'S OWN SETTLES (#2673, R1-2/R1-3), read off the stamps
            # ``store._relay_record`` puts on the base row: a dismissal is the
            # user's view act, a withdrawal the asker's retraction, and a
            # dismissal outranks it (§12's order, applied across the pair). They
            # are LOCAL and view-only — nothing here is written — but they must
            # survive the origin's status or the card silently re-opens.
            local_settle = (
                store.STATUS_DISMISSED
                if record.get("relay_dismissed_at")
                else store.STATUS_WITHDRAWN if record.get("relay_withdrawn_at") else ""
            )
            origin_session = str(ref.get("session_id") or "")
            origin_ask = str(ref.get("ask_id") or "")
            relay: dict[str, Any] = {"session_id": origin_session, "ask_id": origin_ask}
            origin = (
                next(
                    (
                        candidate
                        for candidate in store.fold(
                            store.read_events(store.session_dir(self._config_dir, origin_session)),
                            now_ms,
                        )
                        if candidate.get("ask_id") == origin_ask and not candidate.get("relay")
                    ),
                    None,
                )
                if origin_session and origin_ask
                else None
            )
            if origin is None:
                relay["origin_gone"] = True
                relay["origin_status"] = "missing"
                record["relay"] = relay
                record["status"] = local_settle or store.STATUS_EXPIRED
                continue
            record["questions"] = [
                dict(question)
                for question in (origin.get("questions") or ())
                if isinstance(question, Mapping)
            ]
            record["expires_at"] = int(origin.get("expires_at") or 0)
            record["timeout_s"] = int(origin.get("timeout_s") or 0)
            record["urgent"] = bool(origin.get("urgent"))
            # Copied rather than referenced so a later read of this record cannot
            # see the origin mutate under it; only keys the origin actually holds
            # travel, so a still-open relay carries no empty ``answers``.
            for key in ("answers", "answered_at", "answered_by", "attachments"):
                if key in origin:
                    value = origin[key]
                    record[key] = dict(value) if isinstance(value, Mapping) else value
            origin_status = str(origin.get("status") or "")
            relay["origin_status"] = origin_status
            record["relay"] = relay
            record["status"] = origin_status
            if local_settle and origin_status not in _RELAY_USER_ACT_TERMINALS:
                # A USER ACT ON THE ORIGIN WINS (design §12, across the pair): an
                # answer, decline, dismissal or late answer there is what
                # happened to the QUESTION, so the watcher's own closed card must
                # not hide it. Anything else — the origin merely lapsed, expired
                # or still open — leaves the local settle standing.
                record["status"] = local_settle

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
        cap_refusal = _ask_cap_refusal(
            records,
            incoming_is_bridge=any(
                str(_as_mapping(question).get("question_id") or "") for question in questions
            ),
        )
        if cap_refusal:
            return {"ok": False, "error": cap_refusal}
        open_rows = [
            row for row in records if row["status"] == store.STATUS_OPEN and not row.get("relay")
        ]
        # The duplicate rules below are about the MODEL's own asks. A relay's
        # questions are the origin's and were copied VERBATIM, so counting them
        # would refuse a fresh ask in this session for text the model never
        # wrote; the relay row is excluded with the cap above and for the same
        # reason (#2520).
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

    def enqueue_idempotent(
        self,
        *,
        issuance_key: str,
        fingerprint: str,
        questions: Sequence[Mapping[str, Any]],
        timeout: Any = None,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """KEYED-IDEMPOTENT bridge enqueue (S-10), under ONE interprocess lock.

        The harness-only companion to :meth:`enqueue`: the durable ask id is
        DERIVED from the preallocated issuance identity (never caller/model/
        UI-chosen), and the exact-key lookup + fingerprint validation + append
        run inside ONE ``_NonBlockingLock`` section on the ask log — the same
        lock discipline as every other writer — and under lock contention this
        door FAILS CLOSED (a retryable refusal) rather than deciding on a read
        it cannot prove exclusive (TB-10). Same key + same fingerprint returns
        the SAME ask id (even after the ask was consumed); a mismatched
        fingerprint or multiple exact rows REFUSE. The generic heuristic
        duplicate-text refusal does not apply here: the KEY is the identity,
        and a retry legitimately carries the same wording.

        ``timeout`` follows :func:`policy.parse_timeout_param`: ``None`` (the
        default) means the policy default window, and anything out of range —
        an empty string included — is a validation error naming the bounds,
        exactly as for :meth:`enqueue`. It is never silently defaulted here,
        so a caller cannot ship an unintended window by passing ``""``.
        """

        if not issuance_key or not fingerprint:
            return {"ok": False, "error": "an issuance key and fingerprint are required."}
        # R1-3 (PR #2378): the derived id MUST sit inside the store's own
        # ``a-<hex>`` namespace — ``parse_mirror_request_id`` refuses anything
        # else — while staying DETERMINISTIC, so a retry/dedupe resolves to
        # the same id.
        ask_id = (
            "a-" + hashlib.sha1(("bridge-idem:" + issuance_key).encode("utf-8")).hexdigest()[:16]
        )
        timeout_s, bounds_error = policy.parse_timeout_param(timeout)
        if bounds_error is not None:
            return {"ok": False, "error": bounds_error}
        now = self._now()
        # The lock fd must be opened on the log itself, so the per-session
        # directory has to exist first — the same discipline as
        # ``store.append_event`` for every other writer. A first-ever ask for a
        # session (the bridge's usual shape) otherwise died on FileNotFoundError.
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                store.asks_log_path(self.session_dir),
                store.append_flags(),
                0o600,
            )
        except OSError:
            logger.warning("could not open the ask log in %s", self.session_dir, exc_info=True)
            return {"ok": False, "error": "the ask log could not be opened; not enqueuing."}
        try:
            lock = store._NonBlockingLock(fd)
            with lock:
                if not lock.acquired:
                    # TB-10: the lookup+mint DECISION needs the lock — the
                    # atomic append alone only protects the write — so under
                    # contention this door fails closed (retryable) instead of
                    # deciding on a read it cannot prove exclusive.
                    return {"ok": False, "error": "the ask log is contended; retry shortly."}
                try:
                    rows = self.records()
                except Exception:  # noqa: BLE001 — unreadable is its own answer
                    return {
                        "ok": False,
                        "error": "the ask log could not be read; not enqueuing.",
                    }
                exact = [r for r in rows if str(r.get("ask_id") or "") == ask_id]
                if len(exact) > 1:
                    return {
                        "ok": False,
                        "error": "multiple asks already carry this issuance key; not enqueuing.",
                    }
                if exact:
                    if _bridge_fingerprint_from_row(exact[0]) != fingerprint:
                        return {
                            "ok": False,
                            "error": (
                                "this issuance key already carries a different request; "
                                "not enqueuing."
                            ),
                        }
                    row = exact[0]
                    # SAME key + SAME fingerprint: the SAME ask, even when it
                    # has since been answered/consumed — a crash retry or a
                    # live+cold duplicate must never mint a second one.
                    # TB-08: the idempotent return keeps the deadline/wake
                    # behaviour of a fresh enqueue — a cold runtime still gets
                    # its timeout row armed and the fold republished.
                    self._arm_deadline_wake(row)
                    self._refresh()
                    return {
                        "ok": True,
                        "text": "",
                        "details": {
                            "ask_id": ask_id,
                            "status": "existing",
                            "question_ids": [
                                str(q.get("id") or "")
                                for q in (row.get("questions") or [])
                                if isinstance(q, Mapping)
                            ],
                            "timeout_s": row.get("timeout_s"),
                            "expires_at": row.get("expires_at"),
                            "urgent": bool(row.get("urgent")),
                            "secret": any(
                                bool(q.get("secret"))
                                for q in (row.get("questions") or [])
                                if isinstance(q, Mapping)
                            ),
                            "reach": self._reach(),
                        },
                    }
                cap_refusal = _ask_cap_refusal(
                    rows,
                    incoming_is_bridge=any(
                        str(_as_mapping(question).get("question_id") or "")
                        for question in questions
                    ),
                )
                if cap_refusal:
                    return {"ok": False, "error": cap_refusal}
                shapes = [_question_shape(q) for q in questions]
                # THE MINT MUST AGREE WITH ITS OWN FINGERPRINT. Every later
                # lookup compares a caller's fingerprint against the digest
                # RECOMPUTED from the stored question's bridge block (the log
                # keeps no separate fingerprint column), so the first writer's
                # claim is only ever checked if it is checked HERE. Without
                # this, a writer holding a conflicting fingerprint that landed
                # first minted the ask under the stored block's digest, and the
                # honest writer behind it then matched that digest and was
                # answered "existing" — two writers, both ok, for one key.
                if _bridge_fingerprint_from_row({"questions": shapes}) != fingerprint:
                    return {
                        "ok": False,
                        "error": (
                            "the fingerprint does not match the request's original scope; "
                            "not enqueuing."
                        ),
                    }
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
                os.write(
                    fd,
                    json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    + b"\n",
                )
        finally:
            os.close(fd)
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

    def enqueue_relay(self, relay_ref: str, *, tool_call_id: str = "") -> dict[str, Any]:
        """Record a RELAY of another session's pending ask (#2520).

        A relay is a POINTER, not a copy: this session's log gets one
        ``relayed`` row naming the origin ``(session_id, ask_id)`` and the
        origin's log gets one ``relay-surfaced`` row naming this session, so the
        origin can show "also raised elsewhere". Nothing about the question,
        its deadline or its answer is duplicated — the derivation in
        :meth:`_fold_state` reads them from the origin's fold, which is what
        makes one answer settle both by construction.

        REFUSAL ORDER is the order a model can act on: shape (correctable),
        self (answer it here), unknown (wrong id), chain (follow the one hop),
        settled (nothing left), secret (a boundary), held (already relayed from
        here), cap (wait). Each is its own sentence in ``render``; a generic
        failure teaches nothing and is retried.

        The origin's append is BEST-EFFORT (logged, not fatal): the relay is the
        durable fact and works from this session's log alone, while the surfaced
        note is a courtesy to a session whose directory this process may not own
        or be able to write. Failing the relay because a foreign directory is
        read-only would make the feature unusable for exactly the stopped
        sessions it exists to surface.
        """
        now = self._now()
        parsed = parse_relay_ref(relay_ref)
        if parsed is None:
            return {"ok": False, "error": render.RELAY_BAD_FORM}
        origin_session, origin_ask = parsed
        if origin_session == self._session_id:
            return {"ok": False, "error": render.RELAY_SELF}
        records = self.records(now)
        origin_events = store.read_events(store.session_dir(self._config_dir, origin_session))
        origin = next(
            (
                record
                for record in store.fold(origin_events, now)
                if record.get("ask_id") == origin_ask
            ),
            None,
        )
        if origin is None:
            return {"ok": False, "error": render.RELAY_ORIGIN_UNKNOWN}
        if origin.get("relay"):
            return {"ok": False, "error": render.RELAY_OF_RELAY}
        origin_status = str(origin.get("status") or "")
        # ``timed_out`` is RELAYABLE: the origin still accepts a late answer
        # (#2505), and relaying a lapsed ask is exactly how a watcher surfaces
        # one whose own session has stopped. Every user-act terminal and an
        # expiry is not.
        if origin_status not in (store.STATUS_OPEN, store.STATUS_TIMED_OUT):
            return {"ok": False, "error": render.RELAY_ORIGIN_SETTLED}
        if any(
            question.get("secret")
            for question in (origin.get("questions") or ())
            if isinstance(question, Mapping)
        ):
            # SECURITY, not a preference: a credential may only be typed on the
            # card of the session that asked for it. See ``RELAY_SECRET_ORIGIN``.
            return {"ok": False, "error": render.RELAY_SECRET_ORIGIN}
        # IDEMPOTENCE (#2673, R1-4): a ref this session already relays would
        # otherwise be minted twice — two pointers, two cards, two ``relayed``
        # rows and two surfaced notes for ONE question. The ordinary asks above
        # refuse a duplicate too; the held row's derived status is read here
        # (``records`` is the derivation), so a relay the user closed locally
        # does not block a deliberate re-raise.
        if any(
            record.get("relay")
            and str((record.get("relay") or {}).get("session_id") or "") == origin_session
            and str((record.get("relay") or {}).get("ask_id") or "") == origin_ask
            and record["status"] in (store.STATUS_OPEN, store.STATUS_TIMED_OUT)
            for record in records
        ):
            return {"ok": False, "error": render.RELAY_ALREADY_HELD}
        open_relays = [
            record
            for record in records
            if record.get("relay") and record["status"] == store.STATUS_OPEN
        ]
        if len(open_relays) >= policy.RELAY_CAP:
            return {
                "ok": False,
                "error": render.RELAY_CAP_REFUSAL.format(cap=policy.RELAY_CAP),
            }
        ask_id = self._mint_ask_id({record["ask_id"] for record in records})
        event = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_RELAYED,
            "ask_id": ask_id,
            "at": now,
            "relay": {"session_id": origin_session, "ask_id": origin_ask},
        }
        if not store.append_event(self.session_dir, event):
            return {
                "ok": False,
                "error": "the relay could not be recorded on this session's ask log.",
            }
        surfaced = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_RELAY_SURFACED,
            "ask_id": origin_ask,
            "at": now,
            "by": {"session_id": self._session_id, "ask_id": ask_id},
        }
        try:
            surfaced_ok = store.append_event(
                store.session_dir(self._config_dir, origin_session), surfaced
            )
        except OSError:  # pragma: no cover — only a foreign-dir permission failure
            surfaced_ok = False
        if not surfaced_ok:
            logger.warning(
                "ask relay %s: could not record the surfaced note on %s",
                ask_id,
                origin_session,
            )
        self._refresh(now)
        self.arm()
        return {
            "ok": True,
            "text": render.relay_receipt(
                ask_id,
                origin_session,
                origin_ask,
                reach=self._reach(),
            ),
            "details": {
                "ask_id": ask_id,
                "status": "queued",
                "relay": {
                    "session_id": origin_session,
                    "ask_id": origin_ask,
                    "origin_status": origin_status,
                },
                "reach": self._reach(),
            },
        }

    def _relay_respond(
        self,
        record: Mapping[str, Any],
        answers: Mapping[str, Sequence[str]],
        *,
        by: str,
        tool_call_id: str,
        attachments: Mapping[str, Sequence[ImageContent]] | None,
    ) -> dict[str, Any]:
        """Route an answer on a relay row to the ORIGIN's log (#2520).

        Nothing is appended to the watcher's log: the relay has no answer of its
        own, and a local ``answered`` row would be a second winner the origin
        never learns about. The origin's ``respond`` decides everything — the
        whole-ask completeness check, secrets, images, and the single-winner
        refusal on a second answer — because its log is the only place an answer
        is recorded.

        A refusal comes back with the origin's wording VERBATIM behind a lead
        that names WHERE it came from (``render.relay_refused_by_origin``): the
        same sentence unlabelled reads like the watcher's own rule, and a human
        who taps the wrong card then has no way to tell which ask refused.
        """
        relay = record.get("relay") or {}
        origin_session = str(relay.get("session_id") or "")
        origin_ask = str(relay.get("ask_id") or "")
        origin = AskQueue(
            None,
            config_dir=self._config_dir,
            session_id=origin_session,
            clock=self._now,
        )
        result = origin.respond(
            origin_ask,
            answers,
            by=by,
            tool_call_id=tool_call_id,
            attachments=attachments,
        )
        if not result.get("ok"):
            result["error"] = render.relay_refused_by_origin(
                origin_session, origin_ask, str(result.get("error") or "")
            )
            return result
        # THIS session's wire still has to move: its row derives ``answered`` from
        # the origin on the next fold, and the settle note is owed once. The
        # origin's own write already refreshed the origin's index inside
        # ``respond`` -> ``_settled``.
        self._refresh(self._now())
        self._kick()
        return result

    def _relay_decline(self, record: Mapping[str, Any], *, by: str) -> dict[str, Any]:
        """Route a DECLINE on a relay row to the ORIGIN's log (#2673, R1-1).

        Decline is the other TERMINAL settle, so it takes the same route an
        answer does: the origin's ``decline`` decides and writes, and nothing is
        appended here — the relay's status is derived from the origin on every
        fold, so a local ``declined`` row would be a second winner the origin
        never learns about (and would re-derive back to ``open``).

        A refusal comes back labelled by ``relay_refused_by_origin`` for the
        reason :meth:`_relay_respond` gives.
        """
        relay = record.get("relay") or {}
        origin_session = str(relay.get("session_id") or "")
        origin_ask = str(relay.get("ask_id") or "")
        origin = AskQueue(
            None,
            config_dir=self._config_dir,
            session_id=origin_session,
            clock=self._now,
        )
        result = origin.decline(origin_ask, by=by)
        if not result.get("ok"):
            result["error"] = render.relay_refused_by_origin(
                origin_session, origin_ask, str(result.get("error") or "")
            )
            return result
        self._refresh(self._now())
        self._kick()
        return result

    def respond(
        self,
        ask_id: str,
        answers: Mapping[str, Sequence[str]],
        *,
        by: str = "unknown",
        tool_call_id: str = "",
        attachments: Mapping[str, Sequence[ImageContent]] | None = None,
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

        ``attachments`` are IMAGES answering particular questions (the Other
        door's attachment half). They are content, and the row that lands is
        terminal, so they are never dropped on the way in: anything the queue
        cannot keep (a secret question, a question that is not on the ask, more
        than :data:`MAX_ANSWER_IMAGES`, a store that will not take the bytes)
        refuses the WHOLE answer and leaves the ask open. Accepted ones are
        written to the AttachmentStore FIRST and the ``answered`` event carries
        only refs -- see :meth:`_store_attachments` for the ordering argument.
        """
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        if isinstance(record.get("relay"), Mapping):
            # A RELAY'S ANSWER BELONGS TO THE ORIGIN (#2520). Before the ordinary
            # refusal ladder, because every rule on that ladder is stated about
            # the ORIGIN's log: the relay's own derived status already says what
            # that log holds, and letting the ladder speak here would (a) lose
            # the answer when the relay is open and (b) report the watcher's
            # derivation, not the origin's own refusal copy, for a settled one.
            return self._relay_respond(
                record,
                answers,
                by=by,
                tool_call_id=tool_call_id,
                attachments=attachments,
            )
        refusal = render.refusal_copy(record)
        if refusal:
            # F5-1: a bridge carriage's declined/dismissed row is a same-uid
            # RECORD, not proof that a human declined anything, and it carries no
            # answers for :meth:`revise` to supersede (the fold applies a
            # revision only on top of an ``answered`` event), so the human's
            # admitted decision is taken here as the FIRST answer — the fold's
            # own ordering does the rest, since a genuine ``answered`` event
            # outranks a recorded ``declined`` row. This is not the bypass: the
            # grant is minted by the authority seam, the door checks it, and the
            # dispatcher consumes it, so an answer without one still never
            # reaches the model.
            if not (
                str(record.get("status") or "") in (store.STATUS_DECLINED, store.STATUS_DISMISSED)
                and self._is_bridge_ask_record(record)
            ):
                return {"ok": False, "error": refusal}
        cleaned, map_refusal = self._whole_ask_cells(record, answers)
        if map_refusal:
            return {"ok": False, "error": map_refusal}
        assert cleaned is not None  # ``_whole_ask_cells`` returns one with no refusal
        refs: dict[str, list[dict[str, Any]]] = {}
        if attachments:
            image_refusal = self.attachment_refusal(record, attachments)
            if image_refusal:
                return {"ok": False, "error": image_refusal}
            refs, image_refusal = self._store_attachments(attachments)
            if image_refusal:
                return {"ok": False, "error": image_refusal}
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_ANSWERED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "answers": cleaned,
        }
        if refs:
            # REFS ONLY, never base64: ``asks.jsonl`` is one-line O_APPEND rows and
            # ``store.fold`` reads the WHOLE log on every refresh, so inline bytes
            # would make every later read of every ask pay for one screenshot.
            payload["attachments"] = refs
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
        attachments: Mapping[str, Sequence[ImageContent]] | None = None,
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
        if attachments and any(attachments.values()):
            # A REVISION CARRIES TEXT ONLY (v1): an answer's images are immutable
            # once sent, because the revision event replaces the answer TEXT and
            # the fold keeps the first ``answered`` event's attachments. Refused in
            # words BEFORE any decision so nothing is stored for a revision that
            # would not use it -- the alternative, silently keeping the old images
            # under new text, is the failure the user cannot see.
            return {"ok": False, "error": REVISION_IMAGES_REFUSED}
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

    def attachment_refusal(
        self, record: Mapping[str, Any], attachments: Mapping[str, Sequence[ImageContent]]
    ) -> str:
        """Why this answer's images would be REFUSED, or ``""`` if they would be taken.

        READ-ONLY, and public for the same reason :meth:`revision_refusal` is: the
        one caller with an effect it must not perform speculatively --
        ``Session.respond_ask`` stores a SECRET's value before the queue sees the
        map -- asks first, so an answer refused for its images never leaves a
        credential behind. The queue asks it again at the write (the authority),
        because a direct caller reaches ``respond`` without the session's hop.

        Every rule exists because the alternative is silent loss on a terminal
        answer: a question that is not on this ask, or one that is SECRET (a
        secret answer is a key name only and has no attachment hop, so a picture
        attached to it could only be dropped or leaked), or more images than one
        answer may carry.
        """
        questions = {str(q.get("id")): q for q in (record.get("questions") or ())}
        total = 0
        for qid, images in attachments.items():
            count = len(images or ())
            if not count:
                continue
            question = questions.get(str(qid))
            if question is None:
                return f"an image names question {qid!r}, which is not on this ask"
            if question.get("secret"):
                return (
                    f"question {qid!r} is a secret: a credential is answered by its key "
                    "name, so an image cannot be attached to it"
                )
            total += count
        if total > MAX_ANSWER_IMAGES:
            return f"an answer can carry at most {MAX_ANSWER_IMAGES} images; this one has {total}"
        return ""

    def _store_attachments(
        self, attachments: Mapping[str, Sequence[ImageContent]]
    ) -> tuple[dict[str, list[dict[str, Any]]], str]:
        """Write the images to the AttachmentStore; return ``(refs, refusal)``.

        **BYTES FIRST, THEN THE EVENT** -- the secrets' ordering rule: a value (here
        a picture) must be where the row points BEFORE the row exists. A crash
        between the two leaves ORPHAN BYTES, which the store's own contract
        tolerates (content-addressed, idempotent, never deleted); the reverse order
        would leave a durable ``answered`` row pointing at nothing, and the model
        would be told "image attached" about a picture nobody can load.

        A write the store reports as failed (``put`` -> ``None``: read-only home,
        full disk, undecodable) REFUSES the answer. The transcript path falls back
        to inline bytes for such a failure, but this log cannot: base64 does not go
        in ``asks.jsonl``, so there is no degraded form to fall back to, and
        recording the text while losing the picture is exactly the outcome the
        user cannot see.
        """
        from local_operator.session.attachments import store_for_transcript_dir

        blobs = store_for_transcript_dir(self.session_dir)
        refs: dict[str, list[dict[str, Any]]] = {}
        for qid, images in attachments.items():
            group: list[dict[str, Any]] = []
            for image in images or ():
                ref = blobs.put(image.data, image.mime_type)
                if ref is None:
                    return {}, (
                        "an attached image could not be saved, so the answer was not "
                        "recorded; try again or send the answer as text"
                    )
                group.append(
                    {"attachment": ref.digest, "mime_type": ref.mime_type, "bytes": ref.bytes}
                )
            if group:
                refs[str(qid)] = group
        return refs, ""

    def _resolve_attachments(
        self, record: Mapping[str, Any]
    ) -> tuple[dict[str, list[ImageContent]], int]:
        """The record's image refs as ``ImageContent`` by question, plus the number lost.

        Resolution can legitimately fail (a store pruned by hand, a damaged
        sidecar -- ``AttachmentStore.get`` returns ``None`` for both), and a
        delivery must survive that the way a resumed session survives a missing
        transcript attachment: the picture is omitted and the COUNT of what was
        lost is returned so the text can SAY so, rather than the model being told
        an image is "shown below" that is not.
        """
        refs = record.get("attachments") or {}
        if not refs:
            return {}, 0
        from local_operator.session.attachments import store_for_transcript_dir

        blobs = store_for_transcript_dir(self.session_dir)
        resolved: dict[str, list[ImageContent]] = {}
        missing = 0
        for qid, group in refs.items():
            for ref in group:
                found = blobs.get(str(ref.get("attachment") or ""))
                if found is None:
                    missing += 1
                    continue
                data, mime_type = found
                resolved.setdefault(str(qid), []).append(
                    ImageContent(data=data, mime_type=mime_type)
                )
        return resolved, missing

    @staticmethod
    def _images_in_question_order(
        record: Mapping[str, Any], images: Mapping[str, list[ImageContent]]
    ) -> list[list[ImageContent]]:
        """The resolved image groups in the order the ask's QUESTIONS were asked.

        The attachments map is keyed by question id and, after the log round trip,
        sorted alphabetically (see :meth:`_response_message`), so any consumer that
        needs question order must walk ``record["questions"]`` rather than the map.
        A group whose id is not on the ask (a hand-edited log -- the write path
        refuses such an answer) is kept at the end, in id order, rather than lost:
        dropping a picture silently is the failure this feature exists to prevent.
        """
        ordered: list[list[ImageContent]] = []
        seen: set[str] = set()
        for question in record.get("questions") or []:
            qid = str(question.get("id") or "")
            if qid in seen:
                continue
            seen.add(qid)
            if images.get(qid):
                ordered.append(images[qid])
        ordered.extend(images[qid] for qid in sorted(images) if qid not in seen)
        return ordered

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
        if isinstance(record.get("relay"), Mapping):
            # A relay has no answer row of its own to supersede (#2520): the
            # answer (and therefore its revision window) lives on the origin, so
            # a local ``revised`` event would be a write no reader ever consults.
            return record, render.RELAY_REVISE_UNSUPPORTED
        status = record.get("status")
        if status in (store.STATUS_OPEN, store.STATUS_TIMED_OUT):
            return record, ""
        if status not in (store.STATUS_ANSWERED, store.STATUS_LATE):
            # declined / dismissed / expired: the state table's own sentence,
            # byte-for-byte the one every other path gives for that state.
            # F5-1 does NOT change this: a revision supersedes a recorded
            # ANSWER, and ``fold`` applies one only on top of an ``answered``
            # event, so a revision of a declined carriage could carry nothing
            # (measured, not assumed) — the declined/dismissed BRIDGE case is
            # admitted as a FIRST answer in :meth:`respond` instead.
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
        map is usable. Three rules are the design's (§2.4, §16/SA-D1, review
        round 1 QA Q1 and MINOR 6) and every entry point must agree on them —
        ``respond``, ``revise`` and :meth:`withdraw`'s chat path: the row that
        lands is terminal, so a surface that forgot a key would lose that
        question for good, a key that is not a question id is refused by name
        rather than written where no reader can attribute it, and a SECRET cell
        may only ever be a KEY NAME: the ``[<key>]`` shape is what makes this hop
        load-bearing, because a cold CLI or relay route reaches the queue
        directly and the value must not reach the log through it. Kept in ONE
        function for the same reason ``refusal_copy`` is one sentence per state:
        two copies of a contract drift.
        """
        cleaned = {str(k): [str(v) for v in (vals or [])] for k, vals in answers.items()}
        ids = [str(q.get("id")) for q in (record.get("questions") or ())]
        # UNKNOWN KEYS BEFORE MISSING ONES: a typo is both, and the partial
        # sentence would answer it by telling the model to send an empty list for
        # a question it just answered under another spelling (field case, #2803).
        unknown = [key for key in cleaned if key not in ids]
        if unknown:
            return None, _unknown_answer_error(unknown, ids)
        unanswered = [qid for qid in ids if qid not in cleaned]
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
        """Terminal-on-write: append ``declined`` (today's Esc, made explicit).

        A RELAY takes the origin's route instead (#2673, R1-1): its status is
        derived from the ORIGIN's log, so a local ``declined`` row would be
        silently re-derived away and the user's terminal decision would come
        back as a still-open card. Before the refusal ladder, for the reason
        :meth:`respond` gives — every rule on it is stated about the origin.
        """
        now = self._now()
        record = self._find(ask_id, now)
        if record is None:
            return {"ok": False, "error": render.refusal_copy(None)}
        if isinstance(record.get("relay"), Mapping):
            return self._relay_decline(record, by=by)
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

    def dismiss(
        self, ask_id: str, *, by: str = "unknown", reason: str | None = None
    ) -> dict[str, Any]:
        """The operator puts an ask away — the LIVE arm.

        A TIMED-OUT ask is view-only, as before: what it can still OWE is the
        deadline notice it had not delivered (fork A, §14). An OPEN ask is
        accepted on EVERY live session (SA-D4, 2026-10-23, replacing §15's D6),
        and owes the agent an ``ask-response-<id>`` row that the reconcile
        :meth:`_settled` schedules delivers like a decline's: mid-turn it rides
        the steering queue to the next tool boundary of the SAME turn, and an
        idle session holds it durably with no turn (``Session
        .deliver_ask_messages``). The refusal comes from
        :func:`asks.render.dismiss_refusal`, shared with the cold door
        (``asks/dismiss.py``), so a surface cannot tell which one answered it.
        ``reason`` is the operator's word for why (``moot`` when missing); an
        unknown one is refused before anything is written.

        NO RUNTIME PROBE, deliberately: the queue used to fail CLOSED on a
        session that could not report whether work was running, which refused
        the dismissal exactly while the operator was chatting — the case the
        feature exists for. Nothing about the write depends on it: the log is
        appended the same way in every state, and the DELIVERY is the session's
        decision (see ``Session.deliver_ask_messages``).

        A RELAY'S DISMISSAL IS LOCAL (#2673, R1-2): the dismissal is a view act on
        this session's card, so the row written here is the whole fact — the
        fold's relay record applies it (``store._relay_record``) and the
        derivation keeps it over a merely lapsed origin. The COLD door cannot
        know the origin lapsed (only the live derivation may read it), so the
        state it reads after this tap is the same ``dismissed`` row, not a
        silently re-opened card.
        """
        if reason is not None and reason not in store.DISMISS_REASONS:
            return {"ok": False, "error": render.DISMISS_BAD_REASON}
        now = self._now()
        record = self._find(ask_id, now)
        refusal = render.dismiss_refusal(record)
        if refusal:
            return {"ok": False, "error": refusal}
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_DISMISSED,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "reason": reason or store.DISMISS_REASON_DEFAULT,
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
        if isinstance(record.get("relay"), Mapping) and reason != "moot":
            # ANSWERED_IN_CHAT ON A RELAY IS REFUSED (#2520): it would append an
            # ``answered`` row to THIS log, which the relay's derivation never
            # reads (the origin is the only record of an answer). The receipt
            # would then claim an answer the origin does not hold. Moot is the
            # one local retraction, and it needs no special case below: the
            # append already targets this session's log, which is exactly the
            # required scope.
            return {"ok": False, "error": render.RELAY_WITHDRAW_ANSWERED_IN_CHAT}
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
        # A CHAT ANSWER WITH NO WORDS IS NOT AN ANSWER (review round 1 m2): every
        # cell empty or blank is a model asking to record words nobody said, and
        # the empty-list recipe this release teaches every session makes it
        # reachable. Refused rather than rerouted to ``moot`` — a moot writes a
        # ``withdrawn`` row, a different fact about who acted — and nothing is
        # appended either way.
        if not any(any(text.strip() for text in values) for values in cells.values()):
            return {"ok": False, "error": render.WITHDRAW_NO_ANSWER_WORDS}
        # The questions the message did not answer close with the ask (design
        # §16/SA-D1), and the receipt says how many: the model is the only party
        # that can re-ask them. The ROW's own rule decides what "not answered"
        # means — ``_ask_report`` drops blank text — so a cell of ``[" "]``
        # counts as closed here too; counting only ``[]`` would tell the model
        # nothing was left behind while the row renders ``(not answered)``.
        closed_unanswered = sum(
            1 for values in cells.values() if not any(text.strip() for text in values)
        )
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
            "text": render.withdraw_receipt(ask_id, reason, unanswered=closed_unanswered),
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
        # Recorded so :meth:`rearm` can tell when the timer task is INSIDE this
        # call: cancelling it between ``_handed.add`` and the delivery hand-off
        # strands the id in ``_handed`` and the row is never delivered (review
        # round 2, R2-2). A set of tasks, because a turn start or an answer can
        # run a reconcile while the timer's own is suspended.
        current = asyncio.current_task()
        if current is not None:
            self._reconciling.add(current)
        try:
            return await self._reconcile(now_ms, load_time=load_time)
        finally:
            if current is not None:
                self._reconciling.discard(current)

    async def _reconcile(self, now_ms: int | None, *, load_time: bool) -> list[str]:
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
        relay_notes: list[tuple[int, CustomMessage]] = []
        for record in records:
            status = record["status"]
            if isinstance(record.get("relay"), Mapping):
                # A RELAY IS NOT THIS MODEL'S ASK (#2520). No response row and no
                # deadline row is owed here: the answer belongs to the ORIGIN's
                # conversation, and the origin's own reconcile has already told
                # its model. What this session gets is ONE quiet line about how
                # its pointer ended, whose row id is the delivery marker.
                note = self._relay_note(record)
                if note is not None and note.id not in planned:
                    relay_notes.append((int(record.get("created_at") or 0), note))
                continue
            # A DISMISSED ask whose notice was never delivered is an owed row like
            # any other (fork A, §14): the dismissal put the ask away, and nothing
            # ever told the model it had no answer. Its row is the same
            # ``ask-timeout-<id>`` the deadline would have written.
            owed_dismissed = status == store.STATUS_DISMISSED and bool(record.get("notice_owed"))
            # D1 (design §15): a PRE-deadline dismissal owes the RESPONSE row, which
            # is delivered exactly like a decline's.
            owed_dismissed_response = status == store.STATUS_DISMISSED and bool(
                record.get("response_owed")
            )
            if (
                status not in store.INJECTING_STATUSES
                and not owed_dismissed
                and not owed_dismissed_response
            ):
                continue
            response_id = store.response_row_id(record["ask_id"])
            timeout_id = store.timeout_row_id(record["ask_id"])
            want_response = owed_dismissed_response or status in (
                store.STATUS_ANSWERED,
                store.STATUS_DECLINED,
                store.STATUS_LATE,
            )
            want_timeout = status in (store.STATUS_TIMED_OUT, store.STATUS_LATE) or (owed_dismissed)
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
                            dismissed=owed_dismissed,
                        ),
                    )
                )
            if response_missing:
                if self._bridge_answer_is_proven(record):
                    responses.append(
                        (int(record.get("created_at") or 0), self._response_message(record))
                    )
                else:
                    # S3-1: the record holds an answer NOTHING admitted — the
                    # model must never read it as the user's pick, and the row
                    # is the delivery marker, so withholding it also keeps the
                    # revision window open for the verified door to supersede.
                    logger.warning(
                        "bridge ask %s carries an UNPROVEN answer — its response row is "
                        "withheld until the verified door decides it",
                        record.get("ask_id"),
                    )
        messages: list[CustomMessage] = []
        # Timeouts first: within a batch they are the older event, and a notice
        # that predates the answer it sits beside reads in the order it happened.
        for _, message in sorted(timeouts, key=lambda item: item[0]):
            messages.append(message)
        for _, message in sorted(responses, key=lambda item: item[0]):
            messages.append(message)
        # The watcher's settle notes last: they are about a pair the earlier rows
        # belong to, so the batch reads in the order the facts happened.
        for _, message in sorted(relay_notes, key=lambda item: item[0]):
            messages.append(message)
        if messages:
            for message in messages:
                self._handed.add(message.id)
                # A relay note is NOT an ask row: ``_emit_delivered`` builds a
                # response/timeout delivery event for ask rows, and a note about
                # a settlement is neither (the settlement happened on the
                # origin, whose own rows carry it).
                if message.custom_type in (ASK_RESPONSE_MESSAGE_TYPE, ASK_TIMEOUT_MESSAGE_TYPE):
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

    def _relay_note(self, record: Mapping[str, Any]) -> CustomMessage | None:
        """The WATCHER's one-line note that its relay settled, or ``None`` (#2520).

        Why the watcher and not the origin: the origin's model already gets the
        answer as a response row (it asked). The watcher's model pointed at an
        ask and never heard how it ended, so the note names the origin and the
        settlement in one line and nothing else — it is a fact about a pointer,
        not a question to act on.

        ``_RELAY_SETTLED_STATUSES`` withholds it while the pair can still move
        (an open or lapsed relay), and the row id is ``ask-relay-<id>`` so a
        second reconcile finds it durable and plans nothing: the row IS the
        marker, exactly like the response and timeout rows.
        """
        status = str(record.get("status") or "")
        if status not in _RELAY_SETTLED_STATUSES:
            return None
        ask_id = str(record.get("ask_id") or "")
        if not ask_id:
            return None
        return CustomMessage(
            custom_type=ASK_RELAY_MESSAGE_TYPE,
            attribution="system",
            id=store.relay_row_id(ask_id),
            details={
                "ask_id": ask_id,
                "status": status,
                "relay": dict(record.get("relay") or {}),
                "text": render.relay_settled_note(record),
            },
        )

    def _is_bridge_ask_record(self, record: Mapping[str, Any]) -> bool:
        """Whether a folded ask record is a BRIDGE request at all (structural).

        The SAME predicate the session's door uses (``_is_bridge_ask_record``:
        ANY question carrying a non-empty ``question_id``) — structure, never
        sniffing of which pick happened to resolve. Keying the provenance gate
        off "did one bridge pick resolve" was the S3-1 BLOCKER: a record whose
        bridge question lost its ``id``, carried two picks, or left its first
        bridge question unanswered fell through to the ordinary-ask arm, where
        the forged pick rendered as the user's and its append closed the
        revision window on the human's genuine tap.
        """

        return any(
            str(question.get("question_id") or "")
            for question in (record.get("questions") or ())
            if isinstance(question, Mapping)
        )

    def _bridge_picks(self, record: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Every bridge (question_id, word) the record's answers decided.

        One entry per bridge question that carries a single pick. A bridge
        question with zero or two-plus picks is NOT represented — the caller
        reads the count against the number of bridge questions to see that
        resolution failed and must withhold.
        """

        answers = record.get("answers") or {}
        picks: list[tuple[str, str]] = []
        for question in record.get("questions") or ():
            if not isinstance(question, Mapping):
                continue
            candidate = str(question.get("question_id") or "")
            if not candidate:
                continue
            values = [str(value) for value in ((answers.get(str(question.get("id") or "")) or ()))]
            if len(values) != 1:
                continue
            picks.append((candidate, values[0].strip().lower()))
        return picks

    def _bridge_question_count(self, record: Mapping[str, Any]) -> int:
        """How many bridge questions the record carries."""

        return sum(
            1
            for question in (record.get("questions") or ())
            if isinstance(question, Mapping) and str(question.get("question_id") or "")
        )

    def _bridge_answer_is_proven(self, record: Mapping[str, Any]) -> bool:
        """Whether a BRIDGE record's recorded answer may reach the model.

        The fold is a RECORD, not a proof (§1.4, S3-1): a forged line can claim
        any surface and any word, so a bridge ask's response row is built ONLY
        when the answer is backed by a decision THIS runtime's authority seam
        admitted for this session — the one-shot grant, live (the human's
        gesture, dispatched around the append) or already consumed by the
        dispatcher.

        FAIL-CLOSED ON RESOLUTION, not just on absence: a bridge record whose
        answers do not resolve to exactly ONE pick per bridge question (none,
        two, or a bridge question without its own ``id`` so its answer cannot
        be read) is WITHHELD — the ordinary-ask arm is only for records that
        carry no bridge question at all. An unproven answer is withheld rather
        than rendered as the user's: the human still decides through the
        verified door, which supersedes the forged record while the window is
        open.
        """

        if not self._is_bridge_ask_record(record):
            return True
        picks = self._bridge_picks(record)
        if len(picks) != self._bridge_question_count(record):
            # At least one bridge question did not resolve to a single pick:
            # there is no admitted decision to point at, so nothing may render.
            return False
        if not picks:
            return False
        from local_operator.harness.approval import bridge_decide_grant_seen

        session_id = str(self._session_id or "")
        ask_id = str(record.get("ask_id") or "")
        return all(
            bridge_decide_grant_seen(
                session_id=session_id, ask_id=ask_id, question_id=question_id, word=word
            )
            for question_id, word in picks
        )

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
        if record is None or not (
            record.get("status")
            in (
                store.STATUS_ANSWERED,
                store.STATUS_DECLINED,
                store.STATUS_LATE,
            )
            or (record.get("status") == store.STATUS_DISMISSED and record.get("response_owed"))
        ):
            return message
        if self._is_bridge_ask_record(record) and not self._bridge_answer_is_proven(record):
            # M3: this rebuild re-derives the row from the CURRENT fold, so the
            # provenance VERDICT is re-taken here — never inherited from the
            # preview's shape or from the earlier reconcile. A fold rewritten
            # between the gate and the append (or a grant withdrawn since) must
            # not have its content landed, and it must not carry the marker.
            # The preview the queue already deemed owed is returned unchanged:
            # that content came from the fold at the moment the gate passed —
            # INCLUDING the provenance marker the reconcile-time block stamped on
            # it — and dropping the row here would be the silent loss this
            # method's own contract forbids. So the row that lands is the
            # gate-approved PREVIEW: nothing that failed the gate is written, and
            # nothing owed is dropped, marker and all.
            logger.warning(
                "bridge ask %s no longer proves its decision at the append — the "
                "rebuilt row is refused and the pre-append preview is kept",
                ask_id,
            )
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
        images, images_missing = self._resolve_attachments(record)
        text = render.response_text(
            record,
            secret_lost=lost,
            image_counts={qid: len(group) for qid, group in images.items()},
            images_missing=images_missing,
        )
        extra: dict[str, Any] = {}
        # M3: the marker is stamped on a VERDICT, never on the shape of the
        # record — this builder is reached from the reconcile (always gated) and
        # from the append-time refresh (which re-takes the verdict), and a
        # record that cannot show an admitted decision must not be able to earn
        # the marker by merely looking like a bridge row.
        bridge_proven = self._is_bridge_ask_record(record) and self._bridge_answer_is_proven(record)
        if images:
            # THE IMAGES RIDE AS A TOP-LEVEL ``content`` EXTRA, not inside
            # ``details`` -- measured, not assumed (design §1.3, spikes A and B):
            # the transcript externalises only ``payload["content"]`` blocks into
            # the AttachmentStore, so base64 in ``details`` stays inline in the
            # row (16.9 kB for a 64x64 PNG) and never reaches the model, while the
            # same bytes as ``content`` become a 290 B row of digest refs that
            # replay byte-identically. ``CustomMessage`` is ``extra="allow"`` for
            # exactly this. The render hop turns these blocks into the
            # ``ImageContent`` of an ordinary user turn.
            #
            # THE ORDER IS THE QUESTIONS', NOT THE MAP'S. ``_ask_report`` points at
            # the pictures per question ("(+1 image, shown below)") and nothing else
            # tells the model which image answers which question -- position is the
            # whole attribution. ``images`` comes from the folded log, and the log
            # is written ``sort_keys=True``, so its key order is alphabetical by
            # question id (``a0`` before ``q1``; ``q10`` before ``q2``) whatever order
            # the questions were asked in. Walking ``record["questions"]`` is the one
            # place that restores the order the report text lists them in.
            extra["content"] = [
                block.model_dump(mode="json", exclude={"marker"})
                for group in self._images_in_question_order(record, images)
                for block in group
            ]
        details_attachments = record.get("attachments")
        return CustomMessage(
            custom_type=ASK_RESPONSE_MESSAGE_TYPE,
            attribution="user",
            id=store.response_row_id(str(record["ask_id"])),
            **extra,
            details={
                "ask_id": record["ask_id"],
                "status": record["status"],
                "questions": [dict(q) for q in (record.get("questions") or [])],
                "answers": {k: list(v) for k, v in (record.get("answers") or {}).items()},
                "at": record.get("answered_at")
                or record.get("dismissed_at")
                or record.get("at")
                or 0,
                # A dismissal row states who put the ask away and why, so the
                # receipt (``harness.rows.ask_response_notice``) can say so; absent
                # for every other status, which leaves those rows unchanged.
                **(
                    {
                        "reason": record.get("dismissed_reason"),
                        "by": dict(record.get("dismissed_by") or {}),
                    }
                    if record["status"] == store.STATUS_DISMISSED
                    else {}
                ),
                # ``secret_lost`` is stated on the ROW as well as in the text: a
                # surface renders the fact (the card must not say "in hand" for a
                # credential the session no longer holds) and the model reads the
                # sentence it belongs to.
                "secret_lost": lost,
                "tool_call_id": str(record.get("tool_call_id") or ""),
                "text": text,
                # R3-5/S3-1: a CORRELATION marker, not unforgeable provenance.
                # It records that THIS builder wrote the row under a confirmed
                # admitted verdict, which is what lets the recovery door tell a
                # row the harness rendered from a fold line a forger appended —
                # within the same-uid trust boundary, where the transcript file
                # is itself writable. The AUTHORITY stays the consumed grant; an
                # absent marker means "not written by this path" and is
                # non-authorizing on its own.
                **({"bridge_provenance": "authority_grant"} if bridge_proven else {}),
                # REFS for the cards (TUI, desktop, phone): thumbnails come from
                # the digest route, never from bytes on the row. Absent for a
                # text-only answer so the row is unchanged for every existing ask.
                **(
                    {
                        "attachments": {
                            str(qid): [dict(ref) for ref in group]
                            for qid, group in dict(details_attachments).items()
                        }
                    }
                    if details_attachments
                    else {}
                ),
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
        dismissed: bool = False,
    ) -> CustomMessage:
        """The ``ask_timeout`` row: the deadline notice, and the fork-A variant.

        ``dismissed`` says the user put the ask away before this row existed, so the
        TEXT gains the one sentence that says so (``render.timeout_text``) and the
        details gain an additive flag. The row's id and status are UNCHANGED —
        ``ask-timeout-<id>`` with ``status: "timed_out"`` is the delivery marker the
        fold, the transcript card and every reader already know (fork A is an
        amendment to WHEN this row is written, not a new row).
        """
        text = render.timeout_text(
            record,
            now_ms=now_ms,
            lapsed_while_stopped=lapsed_while_stopped,
            dismissed=dismissed,
        )
        waited_s = max(0, now_ms - int(record.get("created_at") or now_ms)) // 1000
        details: dict[str, Any] = {
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
        }
        if dismissed:
            details["dismissed"] = True
        return CustomMessage(
            custom_type=ASK_TIMEOUT_MESSAGE_TYPE,
            attribution="system",
            id=store.timeout_row_id(str(record["ask_id"])),
            details=details,
        )

    # -- the wire view and the derived index -------------------------------

    def lapsed_visible_ms(self) -> int:
        """This queue's ``asks.lapsed_visible_for`` window, read from ITS config
        directory (the one the log and index live under), for a surface that
        passes it to :meth:`projection`."""
        return _lapsed_visible_ms(self._config_dir)

    def projection(
        self,
        now_ms: int | None = None,
        *,
        drafts: bool = False,
        lapsed_visible_ms: int = store.DEFAULT_LAPSED_VISIBLE_MS,
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

        ``lapsed_visible_ms`` is the ``asks.lapsed_visible_for`` DISPLAY window
        (decision D6): a surface passes it to drop a lapsed row sooner. The
        index writer (:meth:`_refresh`) does NOT, so the derived index stays
        data at the 7 d horizon the sweep shares, and its readers apply the
        window themselves (``asks/store._shown_rows``).
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
            if store.lapsed_hidden(status, expires_at, stamp, lapsed_visible_ms):
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
            # SA-D7 (2026-10-23): the card's context line — "3 messages from you
            # since this was asked (last 19:36)" — rides the row as two INTs so
            # no surface has to re-derive them from a session it may not hold
            # (the desktop and the phone read rows, not sessions). Absent when
            # this runtime cannot count them exactly (see ``_card_context``),
            # which is also the shape every older runtime sends.
            since, last_user_at = self._card_context(record)
            rows.append(
                store.pending_row(
                    record,
                    draft,
                    user_messages_since=since,
                    last_user_message_at=last_user_at,
                )
            )
        rows.sort(
            key=lambda row: (
                row.get("status") != store.STATUS_OPEN,
                -row.get("created_at", 0),
            )
        )
        return rows[: policy.PROJECTION_CAP]

    def _card_context(self, record: Mapping[str, Any]) -> tuple[int | None, int | None]:
        """The ask card's context pair, from the session's user-message stamp.

        ``(user_messages_since, last_user_message_at)``, or ``(None, None)`` when
        the session cannot answer — a queue over a double or a reduced host with
        no stamp, a session that has admitted no operator message yet, or an ask
        older than the runtime's retained window. ABSENCE IS THE CONTRACT: every
        surface renders the line only when the count is there, so "cannot count"
        must never be sent as ``0`` (SA-D7; the same rule the ``attachments`` key
        follows).

        Probed, not assumed: ``AskQueue`` is constructed by the session but also
        by tests and embedders over small doubles, and a duck-typed absence is
        the difference between "no line" and an AttributeError on a per-frame
        path.
        """
        probe = getattr(self._session, "ask_context_line", None)
        if not callable(probe):
            return None, None
        try:
            # ``probe`` came from a getattr, so its return is not statically known;
            # the contract is the session's ``tuple[int, int] | None``.
            pair: Any = probe(int(record.get("created_at") or 0))
        except Exception:  # noqa: BLE001 — a repaint is never worth a failure
            logger.debug("ask: could not read the user-message stamp", exc_info=True)
            return None, None
        if not isinstance(pair, tuple) or len(pair) != 2:
            return None, None
        since, last_user_at = pair
        return int(since), int(last_user_at)

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
        """Ensure exactly ONE deadline timer task exists for this session.

        A live task is kept, but told its plan is stale: a timer waiting on a
        lapsed row's window edge does not re-read the log each tick (see
        :meth:`_run_timer`), so a change that may have added an earlier deadline
        — every enqueue and reconcile ends here — must wake it to re-plan.
        """
        if self._disposed:
            return
        if self._timer is not None and not self._timer.done():
            self._timer_stale = True
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; the next reconcile (from a turn or boot) arms
        self._timer = loop.create_task(self._run_timer())

    def rearm(self) -> None:
        """Replace the timer task, so its next wake is planned from scratch.

        For a change :meth:`arm` cannot see from inside a sleep: the
        ``asks.lapsed_visible_for`` window itself (QA round 1, Q2). A task
        already asleep holds a delay computed with the OLD window — up to a full
        tick with an open ask present, and the whole wait to the old edge with
        lapsed rows only — so a row crossing the new window soon after the edit
        was hidden late or never. Cancelling and re-creating it plans with the
        new window at once.

        EXCEPT MID-RECONCILE (review round 2, R2-2): a timer inside
        :meth:`reconcile` may be between handing a row id to ``_handed`` and
        delivering it, and a cancel there loses the row for this process. It is
        left to finish instead: both branches of :meth:`_run_timer` re-plan from
        the top right after their reconcile, reading the window afresh, which is
        all the replacement would have done.
        """
        task = self._timer
        if task is not None and task in self._reconciling:
            self._timer_stale = True
            return
        self._timer = None
        if task is not None and not task.done():
            task.cancel()
        self.arm()

    async def _run_timer(self) -> None:
        """Sleep to the earliest deadline (≤60 s granularity), then reconcile.

        One task, not one per ask: N asyncio timers for N asks is the shape the
        design rejects, and a re-check on a bounded tick absorbs clock changes
        rather than arming a multi-hour timeout.

        TWO KINDS OF INSTANT wake it (decision D9 ii): an ``open`` ask's
        deadline, and the first millisecond past a LAPSED ask's
        ``asks.lapsed_visible_for`` window. The second has no event of its own
        — nothing is written when a row stops being shown — so without it a
        running session would keep painting a row the setting says is gone
        until the next turn happened to republish. The reconcile at the wake
        ends in ``_refresh`` → ``publish_ask_state``, which re-reads the window.
        A changed window replaces this task (:meth:`rearm`), so the next wake
        is planned with it.

        WITH AN OPEN ASK the loop is today's: re-fold and reconcile every tick.
        WITH LAPSED ROWS ONLY nothing can move but the clock, so it waits to the
        window edge in the same bounded ticks (a suspend or clock change still
        costs at most one tick) WITHOUT re-reading the log or reconciling on
        each — at the 7 d default that was a log read and fold every minute for
        a week (review round 1, R1-4). Anything that could add an earlier
        deadline goes through :meth:`arm`, which flags the plan stale and sends
        the loop back to re-read.
        """
        try:
            while not self._disposed:
                now = self._now()
                records = self.records(now)
                due = [int(r["expires_at"]) for r in records if r["status"] == store.STATUS_OPEN]
                has_open = bool(due)
                window = self.lapsed_visible_ms()
                due.extend(
                    # ``+ 1``: the horizon is strict (``now - expires_at >
                    # window``), so this is the first instant the row is hidden.
                    int(r["expires_at"]) + window + 1
                    for r in records
                    if store.is_lapsed(r["status"])
                    and r.get("expires_at")
                    and not store.lapsed_hidden(r["status"], r["expires_at"], now, window)
                )
                if not due:
                    return
                target = min(due)
                if has_open:
                    delay_s = min(policy.MAX_TICK_MS, max(0, target - now)) / 1000.0
                    await asyncio.sleep(max(0.05, delay_s))
                    if self._disposed:
                        return
                    await self.reconcile()
                    continue
                self._timer_stale = False
                while not self._disposed and not self._timer_stale and self._now() < target:
                    delay_s = min(policy.MAX_TICK_MS, target - self._now()) / 1000.0
                    await asyncio.sleep(max(0.05, delay_s))
                if self._disposed:
                    return
                if self._now() >= target:
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


#: How much of an unknown answer key the refusal echoes (review round 1 NIT 3).
#: The refusal quotes the key so a typo is obvious, but the map comes from a
#: caller the queue does not trust — a hostile relay client can put a
#: megabyte-long key in it, and this sentence is model-facing text that rides a
#: turn. The ids the ask HOLDS are echoed whole: the model chose them.
UNKNOWN_KEY_ECHO_MAX = 5
UNKNOWN_KEY_ECHO_CHARS = 40


def _unknown_answer_error(unknown: Sequence[str], valid: Sequence[str]) -> str:
    """The refusal for an answer map naming a question that is not on the ask.

    NAMES the key it did not recognise and the ids the ask actually has, because
    the model is the one who mistyped: the field case behind this rule sent
    ``bill-merge-permission`` for an ask whose only question was
    ``merge-permission`` and was told it had left a question out. Before it, a map
    that carried every real id plus one unknown key settled the ask and wrote the
    stray cell into the log, where nothing can attribute it to a question. Shares
    :func:`_partial_answer_error`'s contract — the answer is atomic, so the whole
    map is refused and no cell is dropped. The echo is BOUNDED
    (:data:`UNKNOWN_KEY_ECHO_MAX`, :data:`UNKNOWN_KEY_ECHO_CHARS`) because the
    keys come from the caller.
    """
    shown = [
        key if len(key) <= UNKNOWN_KEY_ECHO_CHARS else key[:UNKNOWN_KEY_ECHO_CHARS] + "…"
        for key in unknown[:UNKNOWN_KEY_ECHO_MAX]
    ]
    listed = ", ".join(f"{key!r}" for key in shown)
    if len(unknown) > len(shown):
        listed += f" (and {len(unknown) - len(shown)} more)"
    known = ", ".join(f"{qid!r}" for qid in valid) or "(none)"
    verb = "is" if len(unknown) == 1 else "are"
    return (
        f"{listed} {verb} not a question on this ask — its questions are {known}. "
        "Use the ids exactly as they were asked."
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


def parse_relay_ref(value: Any) -> tuple[str, str] | None:
    """``"<session_id>/<ask_id>"`` → its two halves, or ``None`` (#2520).

    EXACTLY one separator, both halves non-empty. A session id is generated
    identifier text and never contains a slash, so a second one is a malformed
    ref rather than an id that happens to carry one — and accepting it would
    split the string at the wrong place and name a session that does not exist.
    The shape is checked HERE rather than in the tool schema so the model's
    refusal and the queue's are the same rule with the same sentence.
    """
    text = str(value or "").strip()
    if text.count("/") != 1:
        return None
    session_id, ask_id = (part.strip() for part in text.split("/"))
    if not session_id or not ask_id:
        return None
    return session_id, ask_id


def _ask_cap_refusal(rows: Sequence[Mapping[str, Any]], *, incoming_is_bridge: bool) -> str:
    """The typed cap refusal for one readiness window, split by custody (TB-13).

    OPEN_ASK_CAP is the MODEL's asking capacity — "free for the model and
    costly for the human" — and a remote sender is not the model, so bridge
    requests ride a SEPARATE bounded budget. EACH direction is checked against
    the INCOMING ask's own class: a sender burst never blocks the model's
    asks, and a full model queue never blocks a waiting request. A bridge ask
    is identified by its carriage (a non-empty ``question_id`` on any question
    — the same predicate ``asks_bridge.bridge_pending_count`` counts by),
    never by copy. ``""`` means the relevant budget has room.
    """

    open_rows = [row for row in rows if row["status"] == store.STATUS_OPEN]
    # RELAYS ARE NOT THE MODEL'S ASKING CAPACITY (#2520). A relay row is a
    # pointer at a question another session owns: the model here asked nothing,
    # so it must not consume one of THIS session's slots (nor be squeezed by a
    # full queue). Filtered here, before the custody split, so the cap cannot
    # drift between the plain and the bridge arm.
    open_rows = [row for row in open_rows if not row.get("relay")]
    bridge_rows = [
        row
        for row in open_rows
        if any(
            str(question.get("question_id") or "")
            for question in (row.get("questions") or ())
            if isinstance(question, Mapping)
        )
    ]
    if incoming_is_bridge:
        if len(bridge_rows) >= policy.BRIDGE_ASK_CAP:
            return (
                f"this session already has {len(bridge_rows)} waiting agent-messages "
                f"requests (the cap is {policy.BRIDGE_ASK_CAP}); wait for responses "
                f"before more arrive."
            )
        return ""
    model_rows = len(open_rows) - len(bridge_rows)
    if model_rows >= policy.OPEN_ASK_CAP:
        return (
            f"this session already has {model_rows} open asks (the cap is "
            f"{policy.OPEN_ASK_CAP}); wait for responses; do not re-ask."
        )
    return ""


def _bridge_fingerprint_from_row(row: Mapping[str, Any]) -> str:
    """The canonical scope fingerprint, recomputed from a FOLDED ask row.

    The bridge scope block IS the stored original block (with the issuance
    identity fields embedded), so the same canonical JSON hash the session
    computes server-side is recomputable from the log ALONE — no side table,
    no correlation-only ledger (S-10).
    """

    for question in row.get("questions") or []:
        if isinstance(question, Mapping):
            bridge = question.get("bridge")
            if isinstance(bridge, Mapping):
                return bridge_scope_digest(bridge)
    return ""


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
    shape = {
        "id": str(raw.get("id") or ""),
        "question": str(raw.get("question") or ""),
        "options": options,
        "multi": bool(raw.get("multi")),
        "recommended": raw.get("recommended"),
        "secret": bool(raw.get("secret")),
        "persist": bool(raw.get("persist")),
        # H5a: BRIDGE REQUESTS are human-required by contract (§10.3, #125) —
        # a bridge ask must ALWAYS land in the human queue, so the mark rides
        # the projection: the gate short-circuits on it (``Session._gate_ask``)
        # and the ask's ``question_id`` is the bridge message id the decide
        # receipt (`ask_ref`) binds to. Both default empty/false for every
        # existing ask, so their logs and folds are byte-compatible.
        "human_required": bool(raw.get("human_required")),
        "question_id": str(raw.get("question_id") or ""),
    }
    bridge = raw.get("bridge")
    if isinstance(bridge, Mapping):
        # H5b/S-3: the ORIGINAL bridge scope block travels WHOLE (it is the
        # provenance a later recovery re-validates), and the key is added ONLY
        # when present — an ask without it keeps the exact pre-scope shape, so
        # existing logs fold and serialize byte-identically.
        shape["bridge"] = dict(bridge)
    return shape


__all__ = ["AskQueue"]
