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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from local_operator.asks import render, store
from local_operator.mobile.attach_client import find_runtime_record
from local_operator.wakes.arm import WakeWriteError, cancel_wake

logger = logging.getLogger(__name__)

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
    """
    root = Path(config_dir)
    directory = store.session_dir(root, session_id)
    now = now_ms if now_ms is not None else store.now_ms()

    # THE OWNER CHECK FIRST, before any read: a runtime that owns this session has
    # its own record of what it delivered and republishes its schedule list, so a
    # write from here would be lost — or would leave the owed notice delivered by
    # a boot with nobody's turn to read it.
    _record, owner = find_runtime_record(root, session_id)
    if owner is not None:
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
        "by": {"surface": by},
    }
    if not store.append_event(directory, payload):
        return {"ok": False, "error": "the dismissal could not be recorded."}

    _dismiss_in_index(root, session_id, ask_id)
    await _cancel_deadline_wake(root, session_id, ask_id)
    return {"ok": True}


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
    """Drop the ask's hidden ``ask_timeout`` row (see the module docstring)."""
    try:
        await cancel_wake(config_dir, session_id, store.timeout_row_id(ask_id))
    except WakeWriteError as exc:
        if exc.code not in _WAKE_REFUSALS:
            logger.warning(
                "ask: could not cancel the deadline wake row for %s", ask_id, exc_info=True
            )
