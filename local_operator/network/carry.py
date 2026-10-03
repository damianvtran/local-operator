"""State carry-over across a mesh move: the derived indexes, rebuilt and pruned.

WHAT THIS MODULE OWNS, and why it is a module of its own rather than four calls
in ``mobility.py``. A moved session's wake and monitor state has two halves:
the TRANSCRIPT (``wake_schedules``/``monitor_schedules`` custom entries) rides
the existing copy set, and the DERIVED INDEXES
(``<config>/wakes/<sid>.json``, ``<config>/monitors/<sid>.json`` and
``<config>/monitors/state/<sid>/…``) live outside the session directory and are
deliberately NOT copied (``sync.COPY_SET_NAMES``' exclusion list; the stores'
own docstrings call them derived, never authoritative). Without the two steps
here a moved daily wake is invisible to the destination's supervisor until
somebody opens the session — the silent-loss class both store docstrings warn
about (design note §5.1; ``docs/design/remote-onboarding-note.md``).

THE TWO STEPS, in the only order that is safe (F5, design note §5.3 — read it
before touching the call sites):

1. ``rebuild_indexes`` runs at the DESTINATION's promote, from the copied
   transcript. It is idempotent: a second run is a no-op (it skips a write
   whose rows already match), so the promote path and the optional
   engage-on-arrival open cannot duplicate a row.
2. ``prune_after_commit`` runs at the SOURCE's commit, and ONLY after a
   SUCCESSFUL commit — the caller checks that the directory was actually
   removed. Prune-first would strand the source on a failed copy: the commit
   may still refuse (a digest mismatch, a live writer), and a source that had
   already forgotten its index would have a session whose wakes only come back
   when somebody opens it. The prune is idempotent for the same reason.

THE NO-DOUBLE-FIRE ORDERING (§5.3, frozen). The source runtime is retired
before the copy, and every engage against a session with a move in flight is
refused by ``placement.handoff_guard_refusal`` (wired at ``launch.py`` and
``session_factory._prepare``). A refused engage must NOT consume or advance
the row (``fired_count``/``next_due_at`` untouched): nothing in this module —
and nothing an engage refusal can reach — writes the transcript or an index on
the refusal path, so the row travels as stored and the destination fires it
exactly once. The ``[commit → promote-rebuild]`` gap is NAMED rather than
hidden: for the move's own seconds neither supervisor can see the row, and a
past-due wake fires on the destination's first eligible tick after the rebuild.
The e2e cell asserts the no-consume property, not just the happy path.

STDLIB-ONLY, and the reason is the same as ``wakes/store.py``'s ("why this
module must stay import-light"): this code runs inside the RELAY process — on
the promote and commit paths — which must not load the harness to answer a
one-row question. So the transcript scan below is a local backward walk,
adapted from ``session.transcript._iter_complete_lines_backward`` (same chunk
discipline and same O(row) fragment handling) rather than an import of the
session stack, and the custom-type strings and the transcript filename are
local literals, cross-checked against their canonical constants by
``tests/unit/network/test_carry.py`` — importing them would drag
``harness.wake``/``monitors.spec`` in for three strings.

MONITORS RE-BASELINE SILENTLY (OQ8). The spec rows are carried by the
transcript; the counters (``<mid>.json``) and the snapshot (``<mid>.snap``) are
device-local observations and are NOT carried — the rebuild writes only the
index, and a monitor with no counters is exactly the "no baseline at all"
branch ``MonitorScheduler._apply_success`` already handles by establishing a
baseline without firing (never a full-dump delivery). So the move itself
produces no alert. On the source, the prune removes the state directory too:
leaving it behind would make a later hand-back diff against a pre-move
snapshot, which is the opposite of the silent re-baseline the design chose.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The journal's file name (``session.transcript.TRANSCRIPT_FILENAME``). Local
#: literal on purpose: see the module docstring; a test pins the equality.
TRANSCRIPT_FILENAME = "transcript.jsonl"

#: ``harness.wake.WAKE_SCHEDULES_CUSTOM_TYPE`` and
#: ``monitors.spec.MONITOR_SCHEDULES_CUSTOM_TYPE``, same treatment.
WAKE_SCHEDULES_CUSTOM_TYPE = "wake_schedules"
MONITOR_SCHEDULES_CUSTOM_TYPE = "monitor_schedules"

#: ``session.transcript.ENTRY_CUSTOM``.
ENTRY_CUSTOM = "custom"

#: ``session.retention.DESKTOP_MARKER_NAME`` — the created-session marker a copy
#: carries, and the second source of the ``cwd`` a rebuilt index names.
DESKTOP_MARKER_NAME = "desktop.json"

#: Backward-read granularity for the transcript scan (the reference walker's
#: chunk; kept in one place so the two readers stay comparably shaped).
_BACKWARD_CHUNK_BYTES = 1 << 16


def _iter_lines_backward(
    handle: Any, end_of_file: int, *, chunk_bytes: int = _BACKWARD_CHUNK_BYTES
):
    """Yield the file's lines NEWEST FIRST, as raw bytes.

    Adapted row-for-row from ``session.transcript._iter_complete_lines_backward``
    (which carries the full reasoning for the fragment list — one pass per row,
    joined exactly once): a row larger than a chunk must not cost one copy per
    chunk. The file's opening fragment is yielded last and is treated as a
    complete row; the caller's parse decides whether its head was torn, exactly
    as the reference reader documents.
    """
    position = end_of_file
    # Fragments of one row, in READ order (newest first), each contiguous with
    # the next; never joined until the boundary that ends the row arrives.
    carried: list[bytes] = []
    while True:
        chunk_start = max(0, position - chunk_bytes)
        handle.seek(chunk_start)
        chunk = handle.read(position - chunk_start)
        position = chunk_start
        if position > 0:
            boundary = chunk.find(b"\n")
            if boundary < 0:
                carried.append(chunk)
                continue
            complete = chunk[boundary + 1 :].split(b"\n")
            if carried:
                complete[-1] += b"".join(reversed(carried))
            carried = [chunk[:boundary]]
        else:
            complete = chunk.split(b"\n")
            if carried:
                complete[-1] += b"".join(reversed(carried))
            carried = []
        for line in reversed(complete):
            yield line
        if position == 0:
            return


def latest_custom_details(session_dir: Path, custom_type: str) -> dict[str, Any] | None:
    """The newest ``custom_type`` row's ``payload.details``, or ``None``.

    Row-for-row compatible with ``session.transcript.read_latest_custom_entry``
    for the one question this module asks: the newest row with
    ``type == "custom"`` and ``str(payload.get("custom_type", ""))`` equal to
    ``custom_type``, with malformed rows skipped individually and an absent
    journal answering ``None``. The named divergence is the reference reader's
    own: bytes are decoded with ``errors="replace"``, so a byte-corrupt row
    that still parses is answered where the resident ``Transcript`` object
    would raise, and a row that no longer parses is skipped like any other
    malformed line.
    """
    path = Path(session_dir) / TRANSCRIPT_FILENAME
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        handle.seek(0, os.SEEK_END)
        for raw in _iter_lines_backward(handle, handle.tell()):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(entry, dict) or entry.get("type") != ENTRY_CUSTOM:
                continue
            payload = entry.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if str(payload.get("custom_type", "")) != custom_type:
                continue
            details = payload.get("details")
            return dict(details) if isinstance(details, Mapping) else {}
    return None


def resolve_cwd(session_dir: Path, previous: Mapping[str, Any] | None = None) -> str:
    """Which directory a rebuilt index should name as the session's cwd.

    The same three steps the arm modules use (``wakes/arm.py::_resolved_cwd``,
    ``monitors/arm.py::_resolved_cwd``, verbatim): the existing index entry's
    own value first — it is what the supervisor reads today — then the desktop
    marker a created session carries (it travels in the copy set, which is what
    makes it the right source on a device that never had an index), then the
    session directory as the last resort.
    """
    existing = (previous or {}).get("cwd")
    if isinstance(existing, str) and existing:
        return existing
    try:
        marker = json.loads((Path(session_dir) / DESKTOP_MARKER_NAME).read_text(encoding="utf-8"))
        cwd = marker.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    except (OSError, ValueError):
        pass
    return str(session_dir)


def _monitor_index_rows(specs: list[Any]) -> list[dict[str, Any]]:
    """The index rows for carried monitor specs, with NO counters (fresh defaults).

    The key set is ``MonitorScheduler.index_rows``' and
    ``monitors.arm._compose_index_rows``' — rebuilt here from specs alone
    because the destination has no counters files: the state is device-local
    and the first check re-baselines it silently. A zeroed ``next_due_at`` is
    deliberate: it cannot promise a due instant the destination has not
    computed, and every cold reader treats a missing due time as "rebuilt when
    its session opens" (``monitors/store.next_due_at`` skips non-int rows).
    """
    rows: list[dict[str, Any]] = []
    for raw in specs:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row.update(
            {
                "next_due_at": None,
                "last_check_at": 0,
                "checks": 0,
                "deliveries": 0,
                "consecutive_failures": 0,
                "disabled": bool(row.get("disabled")),
                "disabled_reason": str(row.get("disabled_reason") or ""),
            }
        )
        rows.append(row)
    return rows


def rebuild_indexes(
    config_dir: Path, session_id: str, *, session_dir: Path | None = None
) -> dict[str, Any]:
    """Rebuild the destination's derived indexes from the copied transcript.

    Idempotent and best-effort: each half swallows its own failure (a rebuild
    that raised would fail a move whose bytes are already committed and whose
    transcript still carries the rows — the indexes self-heal on the next
    open), and a half whose rows already match the entry on disk does not
    rewrite it, so "a second rebuild is a no-op" is literal rather than
    eventual.

    Returns a small report for the callers' logs and the tests:
    ``{"wakes": n|None, "monitors": n|None, "index_written": bool, ...}`` where
    ``None`` means the store half failed.
    """
    from local_operator.monitors import store as monitor_store
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    directory = Path(session_dir) if session_dir is not None else root / "sessions" / session_id
    report: dict[str, Any] = {"session_id": session_id, "wakes": None, "monitors": None}

    # -- wakes -----------------------------------------------------------------
    try:
        details = latest_custom_details(directory, WAKE_SCHEDULES_CUSTOM_TYPE) or {}
        rows = [row for row in details.get("schedules") or () if isinstance(row, Mapping)]
        existing = wake_store.read_entry(root, session_id)
        cwd = resolve_cwd(directory, existing)
        if existing is not None and _rows_equal(existing.get("schedules"), rows):
            report["wakes"] = len(rows)
        else:
            # Empty rows REMOVE the entry, which is the store's own contract
            # ("no file" and "no wakes" are the same statement) — and correct
            # on a rebuild: the latest transcript snapshot is the full list.
            wake_store.write_entry(root, session_id, cwd=cwd, schedules=rows, preserve=existing)
            report["wakes"] = len(rows)
    except Exception:  # noqa: BLE001 — derived state; the next open rebuilds it
        logger.warning("carry: could not rebuild the wake index for %s", session_id, exc_info=True)

    # -- monitors --------------------------------------------------------------
    try:
        details = latest_custom_details(directory, MONITOR_SCHEDULES_CUSTOM_TYPE) or {}
        specs = [row for row in details.get("monitors") or () if isinstance(row, Mapping)]
        rows = _monitor_index_rows(specs)
        existing = monitor_store.read_entry(root, session_id)
        cwd = resolve_cwd(directory, existing)
        if existing is not None and _rows_equal(existing.get("monitors"), rows):
            report["monitors"] = len(rows)
        else:
            monitor_store.write_entry(root, session_id, cwd=cwd, monitors=rows, preserve=existing)
            report["monitors"] = len(rows)
    except Exception:  # noqa: BLE001 — derived state; the next open rebuilds it
        logger.warning(
            "carry: could not rebuild the monitor index for %s", session_id, exc_info=True
        )

    report["index_written"] = report["wakes"] is not None or report["monitors"] is not None
    return report


def _rows_equal(existing: Any, rows: Sequence[Any]) -> bool:
    """Whether the index entry already carries these rows (the no-op check).

    Compared on the JSON form rather than ``==`` so a row read back from disk
    with its keys re-ordered still counts as equal — a second rebuild must not
    rewrite the file just to reorder it.
    """
    if not isinstance(existing, list) or len(existing) != len(rows):
        return False
    try:
        return json.dumps(existing, sort_keys=True) == json.dumps(
            [dict(row) for row in rows], sort_keys=True
        )
    except (TypeError, ValueError):
        return False


def prune_after_commit(config_dir: Path, session_id: str) -> dict[str, Any]:
    """Remove the SOURCE's derived state for a session that has just left.

    Called ONLY after a successful commit (§5.3, F5 — see the module docstring
    for why the order is not negotiable). Removes the wake index entry, the
    monitor index entry and the monitor state directory; every step is
    idempotent, so a retried commit is safe.

    Returns ``{"wakes": bool, "monitors": bool, "monitor_state": bool}`` —
    whether each removal actually found something to remove.
    """
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors import store as monitor_store
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    report: dict[str, Any] = {"session_id": session_id}
    try:
        report["wakes"] = bool(wake_store.remove_entry(root, session_id))
    except Exception:  # noqa: BLE001 — derived state; failing to prune strands nothing durable
        logger.warning("carry: could not prune the wake index for %s", session_id, exc_info=True)
        report["wakes"] = False
    try:
        report["monitors"] = bool(monitor_store.remove_entry(root, session_id))
    except Exception:  # noqa: BLE001
        logger.warning("carry: could not prune the monitor index for %s", session_id, exc_info=True)
        report["monitors"] = False
    try:
        monitor_state.remove_session_state(root, session_id)
        report["monitor_state"] = True
    except Exception:  # noqa: BLE001
        logger.warning("carry: could not prune monitor state for %s", session_id, exc_info=True)
        report["monitor_state"] = False
    return report


def ensure_supervisor(config_dir: Path) -> dict[str, Any]:
    """Install the wake supervisor on demand, and VERIFY it is running.

    The §5.3 ``ensure`` step: the destination must be able to FIRE the carried
    rows, and the supervisor is the process that fires wakes for a session with
    no runtime. Mirrors ``Session._ensure_wake_supervisor``'s contract (best
    effort, never raises, called only when something is scheduled): the
    installer is idempotent and the relay must not pay its cost for the
    overwhelming majority of sessions that have no wakes at all.

    THE VERIFY HALF IS THE POINT, and the carried-wake drill is why (a real
    two-device move carried a daily wake whose due time passed with the
    destination's supervisor loaded-but-stopped; nothing said so). The
    installer's ``installed`` means "a supervisor is now in place" — it may
    have started one a moment ago, and a process can be gone again by the time
    the move answers. So the operative fact this function reports is
    ``running``, read back from the same state probe ``lop wake status``
    uses; a caller that gets ``running=False`` must SAY so rather than move on.

    Returns ``{"installed": bool, "running": bool, "detail": str}``:
    ``installed`` is the installer's answer, ``running`` is the verified
    liveness for THIS store, and ``detail`` names why when it is not running.
    Never raises, and never lets the verification failure pass in silence
    either: both halves report through the outcome.
    """
    try:
        from local_operator.wakes.install import (
            ensure_supervisor_installed,
            supervisor_state,
        )

        outcome = ensure_supervisor_installed(Path(config_dir))
    except Exception as exc:  # noqa: BLE001 — a supervisor is not worth a failed move
        logger.warning("carry: could not ensure the wake supervisor", exc_info=True)
        return {"installed": False, "running": False, "detail": f"failed: {exc}"}
    try:
        state = supervisor_state(Path(config_dir))
        # FAIL-CLOSED, the same conjunct the status surface applies
        # (``cli.py``'s ``wake status``: ``state.running and state.verifiable``):
        # ``verifiable=False`` means the probe addressed a DIFFERENT store — the
        # global label, not this one — so nothing about it may read as this
        # store's supervisor running. Today every ``verifiable=False`` return also
        # carries ``running=False``; the conjunct is what keeps that a guarantee
        # rather than a coincidence (review round 1, R-MINOR).
        running = bool(state.running and state.verifiable)
        detail = str(state.detail or "") or str(outcome.reason or "")
    except Exception as exc:  # noqa: BLE001 — an unverifiable supervisor is not a running one
        logger.warning("carry: could not verify the wake supervisor", exc_info=True)
        return {"installed": bool(outcome.installed), "running": False, "detail": f"failed: {exc}"}
    return {"installed": bool(outcome.installed), "running": running, "detail": detail}


def supervisor_notice(wakes: int, device: str) -> str:
    """The LOUD fallback sentence for a move that carried wakes it cannot fire.

    THE COPY IS A CONTRACT, and design round 1 re-rendered it (D1/D2/D7):

      "1 wake carried; supervisor not running on <device> — run 'lop wake install' on <device>"

    Three deliberate properties: the noun pluralises with the count (the drill's
    own measured case is ONE daily wake); the command is anchored to the device
    NAMED IN THE SENTENCE, because ``lop wake install`` has no peer selector and
    a reader on the source who ran it there would fix the wrong machine — the
    exact silent-stopped outcome this line exists to prevent; and the command is
    single-quoted, the family's canonical spelling for a copy-pasteable remedy
    (``cli.py``'s ``wake status`` convention: a quoted command is one token).
    """
    word = "wake" if wakes == 1 else "wakes"
    return (
        f"{wakes} {word} carried; supervisor not running on {device} — "
        f"run 'lop wake install' on {device}"
    )


def wake_running_notice(wakes: int, device: str) -> str:
    """The quiet success line: carried wakes whose supervisor was VERIFIED running.

    Design round 1's D4: the loud half alone left a carried-and-live wake
    indistinguishable from no wake at all — the receipt said nothing when things
    were fine. This is the other half, rendered only when ``running`` is the
    verified fact (never on the not-running path, which has the loud sentence).
    """
    word = "wake" if wakes == 1 else "wakes"
    return f"{wakes} {word} carried — supervisor running on {device}"


def monitors_notice(count: int) -> str:
    """The up-front monitors statement for a move whose origin holds monitors.

    DESIGN ROUND 1 RE-CUT THIS SENTENCE (D3/D6), and the two stumbles it fixes are
    worth keeping named: the old copy ("N monitors here will not travel with their
    state") made THE MONITORS the subject of "not travel", which garden-paths as
    "your monitors were dropped" — the opposite of the fact, since the spec rows
    ride the transcript and the destination rebuilds their index. Here the subject
    of the negative is the STATE, matching the design note's "Monitor state does
    not travel with a move", and the continuation says in plain words what happens
    instead ("their checks start fresh on the destination") rather than meeting the
    reader with the codebase's internal "re-baseline" — which stays the term of
    art in the design note (``docs/design/mesh-session-mobility.md`` §7.2), not at
    the point of first contact.
    """
    noun = "monitor" if count == 1 else "monitors"
    return (
        f"{count} {noun} here: their state does not travel — "
        "their checks start fresh on the destination"
    )


def move_carry_block(config_dir: Path, wakes: int, device: str) -> dict[str, Any]:
    """The receipt's scheduled-state block for a promote that carried ``wakes``.

    THE §5.3 ``ensure`` STEP'S WHOLE OUTCOME, in one function, so the copy of the
    fallback sentence has a single definition (``supervisor_notice`` states it) and
    the promote only has to report it: install/start the supervisor, VERIFY it is
    running, and when it is not — including an ensure that answered a shape this
    module did not expect (measured 2026-10-03: a stub on the old string shape
    raised from inside the fallback logging and failed a whole move) — say so. The
    running case is NOT silent either (design round 1, D4): ``running_notice`` is
    the quiet confirmation, so a carried-and-live wake cannot read as no wake at
    all. Never raises: a promote must not fail here. The WHY stays in this device's
    relay log; the actionable line on the receipt is the notice.
    """
    running = False
    detail: Any = None
    try:
        supervision = ensure_supervisor(Path(config_dir))
        running = bool(supervision["running"])
        detail = supervision.get("detail")
    except Exception:  # noqa: BLE001 — the sentence below must still be said
        logger.warning("carry: could not ensure the wake supervisor", exc_info=True)
    block: dict[str, Any] = {"wakes": wakes, "supervisor": "running" if running else "not_running"}
    if running:
        # The quiet half (design round 1, D4): the block alone left the rendered
        # receipt silent on success, so a carried-and-live wake could not be told
        # from no wake at all. Rendered only when ``running`` is verified.
        block["running_notice"] = wake_running_notice(wakes, device)
    else:
        block["notice"] = supervisor_notice(wakes, device)
        logger.warning(
            "carry: wake supervisor not running for %s after %d carried wake(s): %s",
            config_dir,
            wakes,
            detail or "ensure failed",
        )
    return block


def monitor_count(config_dir: Path, session_id: str) -> int:
    """How many monitor rows THIS device holds for a session, or 0.

    The "origin can cheaply see its own monitors at move time" half of the
    move's monitors notice: one small file read
    (``<config>/monitors/<sid>.json`` — the same index ``lop monitor status``
    scans). Best effort by design — a count is not worth a failed move, so an
    unreadable or absent index answers 0.
    """
    try:
        from local_operator.monitors import store as monitor_store

        entry = monitor_store.read_entry(Path(config_dir), session_id)
    except Exception:  # noqa: BLE001 — a count is not worth a failed move
        logger.warning("carry: could not read the monitor index for %s", session_id, exc_info=True)
        return 0
    rows = (entry or {}).get("monitors")
    return len(rows) if isinstance(rows, list) else 0
