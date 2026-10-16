"""Dismiss a lapsed ask with NO runtime — the cold door (design §2.2, §14).

WHY THIS MODULE EXISTS (fork A, 2026-10-16). Every other ask op needs a runtime:
an answer is READ by the model, so a session with no runtime must be engaged to
receive one — that is the whole reason ``_ASK_ANSWER_OPS`` engages. A dismissal
has no reader. It is one append plus a derived-index rewrite, decided entirely by
the fold, and the phone's own promise has always been the plain one: "dismiss —
send no reply". Yet the only door was the engage, so a stopped conversation was
BOOTED, and the boot reconcile delivered the deadline notice the dismissal had not
cancelled yet, which started a paid turn (measured end to end; the architect's E1).

THREE PROPERTIES MAKE WRITING FROM HERE SAFE, and each one is a check below:

* **A live owner is refused, never written behind.** A runtime republishes its
  in-memory wake list over an outside write (``wakes/arm.py`` :41-49) and tracks
  the rows it has handed out internally (``AskQueue._handed``), so an outside
  append under a running session can be lost, or can deliver a notice without the
  turn the notice is for. The caller falls back to the dial on
  ``ask_owner_present`` — and ONLY on that, because every other refusal is an
  answer rather than a reason to boot something.
* **The fold decides, in the queue's own words.** The refusal sentence comes from
  :func:`asks.render.dismiss_refusal`, the same function ``AskQueue.dismiss``
  calls, so a surface cannot tell which door answered it by the wording.
* **The deadline wake row goes with it.** ``ask-timeout-<id>`` is what makes a
  deadline survive a runtime that does not exist: the supervisor engages a runtime
  for it. Removing it is the reason a dismissal boots nothing at all — and it is
  safe for exactly ONE reason, which is the condition this module is allowed to
  exist under:

  **Only a timed-out ask can be dismissed, so its deadline has already passed and
  the wake row has nothing left to deliver except the boot itself.** The notice
  the row would have delivered is not lost: the fold keeps it owed
  (``store.fold``'s ``notice_owed``) and the next real reconcile delivers it in
  its dismissed wording. If a dismissal is ever allowed BEFORE the deadline,
  removing the row becomes wrong — the fire would be the only thing that delivers
  a notice for a live ask — and this module must change in the same edit as that
  rule.

The dismissal is durable before anything else happens: ``append_event`` is the
truth (``O_APPEND``, one line) and the index is derived from it, so a failure to
rewrite the index or cancel the wake leaves a session the next reader heals rather
than a promise the phone cannot keep.

WHAT THE WAKE ROW COSTS A PARKED SESSION (agent review round 1, MAJOR). The cancel
is skipped when the wake entry is PARKED (``stopped_at``/``held_at``), because a
parked entry's row fires nothing — every supervisor decision skips a held entry —
while ``cancel_wake``'s empty remainder removes the whole index entry, and with it
the park marker that ``session_was_stopped``, ``attention._stopped_marker`` and the
desktop warm guards read to keep this conversation from being started. Skipping is
also what keeps the transcript untouched, which is what stops a dismissal
re-ranking the conversation in ``/resume`` (QA round 1, Q1). A COLD SESSION THAT IS
NOT PARKED still gets the row removed: nothing suppresses its fire there, so a
stale deadline row would boot exactly the runtime this module exists to avoid
booting.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from local_operator.asks import render, store
from local_operator.wakes.arm import WakeWriteError, cancel_wake, repair_index
from local_operator.wakes.store import is_held
from local_operator.wakes.store import read_entry as read_wake_entry

logger = logging.getLogger(__name__)

#: The surface word a cold dismissal records when its caller named none.
#:
#: The phone's ``ask_dismiss`` body carries no ``by`` (``mobile/daemon.py`` forwards
#: ``str(body.get("by") or "")``), and the LIVE door already answers that empty
#: string with this word (``session/runtime/serving.py``: ``by or "remote"``), so the
#: audit line cannot say which door recorded the tap (agent review round 1,
#: MINOR-3). One constant, the same word — the surface and the sentence agree.
_REMOTE_SURFACE = "remote"

#: The wake refusals a dismissal may meet, neither of which is one for it:
#: ``wake_not_found`` means the row is already gone (the ask was armed while a
#: runtime lived, and that runtime's retire landed), and ``wake_owner_present``
#: means a runtime claimed the conversation between the owner check below and this
#: write — it owns the list now and its own reconcile settles the row. Anything
#: ELSE is a failed cancel on a dismissal that has already landed, so it warns
#: rather than failing: the log is the truth, and the row is stale once the ask is
#: terminal.
_WAKE_REFUSALS = ("wake_not_found", "wake_owner_present")


async def dismiss_cold(
    config_dir: Path | str,
    session_id: str,
    ask_id: str,
    *,
    by: str = "unknown",
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Settle a TIMED-OUT ask and boot nothing. ``{"ok": True}`` or a refusal.

    The refusals are the queue's own ``{"ok": False, "error": <sentence>}`` shape
    plus one code no live dismissal can produce — ``ask_owner_present``, which the
    relay and the TUI answer by dialling the owner they just found. Writes nothing
    on every refusal, including that one.

    ``by`` is the surface the user dismissed from. An EMPTY one is the remote door
    (they sent nothing), which is the word the live path records for the same tap;
    an omitted ``by`` is the caller's honest "unknown".
    """
    root = Path(config_dir)
    directory = store.session_dir(root, session_id)
    now = now_ms if now_ms is not None else store.now_ms()

    # THE OWNER CHECK FIRST, before any read: a runtime that owns this session has
    # its own record of what it delivered and republishes its schedule list, so a
    # write from here would be lost — or would leave the owed notice delivered by
    # a boot with nobody's turn to read it.
    if _live_owner(root, session_id) is not None:
        return {"ok": False, "code": "ask_owner_present"}

    events = store.read_events(directory)
    record = next(
        (row for row in store.fold(events, now) if str(row.get("ask_id")) == ask_id),
        None,
    )
    refusal = render.dismiss_refusal(record)
    if refusal:
        return {"ok": False, "error": refusal}

    payload = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_DISMISSED,
        "ask_id": ask_id,
        "at": now,
        "by": {"surface": by or _REMOTE_SURFACE},
    }
    if not store.append_event(directory, payload):
        return {"ok": False, "error": "the dismissal could not be recorded."}

    _dismiss_in_index(root, session_id, ask_id)
    await _cancel_deadline_wake(root, session_id, ask_id)
    return {"ok": True}


def _live_owner(root: Path, session_id: str) -> int | None:
    """The pid hosting ``session_id``, or ``None`` — the liveness probe itself.

    Called instead of the mobile client's ``find_runtime_record``, which this door
    only ever used to throw the record away and keep this pid (agent review round
    1, NIT): that function asks :func:`local_operator.resume.live_runtime_pid`
    underneath and then scans the run directory for a record nobody here reads,
    and importing it pulls the phone client's whole closure (approval, keychain,
    signing) into a write that must stay cheap. The import is function-local for
    the same reason — ``resume`` is import-light, and a door imported on a cold
    path should not widen its own closure for a probe.
    """
    from local_operator.resume import live_runtime_pid

    return live_runtime_pid(root, session_id)


def _dismiss_in_index(config_dir: Path, session_id: str, ask_id: str) -> None:
    """Set this ask's row in the derived index to ``dismissed``.

    The index is what the cross-session view reads with no runtime to ask, so
    without this the phone keeps offering Dismiss on an ask the log has already
    settled until some runtime happens to reconcile. Best-effort by the same
    contract the queue's own writer keeps (``AskQueue._refresh``): the log is the
    truth and any writer heals the entry.
    """
    try:
        entry = store.read_entry(config_dir, session_id)
        if entry is None:
            return
        rows = [dict(row) for row in entry.get("asks") or ()]
        touched = False
        for row in rows:
            if str(row.get("ask_id")) == ask_id:
                row["status"] = store.STATUS_DISMISSED
                touched = True
        if not touched:
            return
        store.write_entry(config_dir, session_id, cwd=str(entry.get("cwd") or ""), asks=rows)
    except Exception:  # noqa: BLE001 — the log is the truth; the index heals
        logger.warning("ask: could not rewrite the index for %s", session_id, exc_info=True)


async def _cancel_deadline_wake(config_dir: Path, session_id: str, ask_id: str) -> None:
    """Drop the ask's hidden ``ask_timeout`` row (see the module docstring).

    A PARKED SESSION IS LEFT ALONE, and that is a decision the reviewer's evidence
    settles rather than a preference. ``cancel_wake`` removes the index entry when
    the list it leaves behind is empty (``wakes/arm.py``, ``store.write_entry``),
    and that entry is where ``stopped_at`` lives — the durable marker
    ``session_was_stopped``, ``attention._stopped_marker`` and the desktop warm
    guards read to decide this conversation must not be started. Dropping it turns
    "stopped, leave it alone" into "not proven stopped", which is a second door
    onto the boot this module exists to prevent. Skipping costs nothing: every
    supervisor decision skips a held entry (``wakes/supervisor.py``'s
    ``_is_held_entry``), so the row fires nothing while parked, and it is stale by
    then anyway — only a terminal ask reaches this door. Keeping the park marker
    instead of skipping would mean writing an empty-schedules entry, which the
    store's contract forbids ("no file" and "no wakes" are the same statement)
    and which would hold the cleanup reap guard open on a session that no longer
    carries a wake.
    """
    entry = await asyncio.to_thread(read_wake_entry, Path(config_dir), session_id)
    if is_held(entry):
        return
    try:
        await cancel_wake(config_dir, session_id, store.timeout_row_id(ask_id))
    except WakeWriteError as exc:
        if exc.code == "wake_not_found":
            # QA round 1, Q2: the cancel's BASE is the transcript, and the index
            # can carry a row the transcript does not (a hand-built pair, or a
            # crash between the two writes). The refusal is right — there is
            # nothing to cancel — but the index row it leaves behind is the
            # supervisor's input, so the divergence is healed with the module's
            # own repair: the transcript's rows are projected onto the index,
            # which drops a row the transcript never had and invents none.
            # Best-effort, like every other index write on this path: the
            # transcript is the truth and the next reader repairs the rest.
            await repair_index(Path(config_dir), session_id)
        elif exc.code not in _WAKE_REFUSALS:
            logger.warning(
                "ask: could not cancel the deadline wake row for %s", ask_id, exc_info=True
            )
