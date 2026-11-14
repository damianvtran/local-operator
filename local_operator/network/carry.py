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
    from local_operator.session import compression

    # Resolved through the codec: the carried payload must be the DECOMPRESSED
    # bytes. Opening the plain name here read ``None`` for every compressed
    # session, and opening an archive raw would decode compressed bytes as JSONL
    # and silently carry nothing.
    path = compression.transcript_for_read(Path(session_dir))
    if path is None:
        return None
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


def _home_relative_tail(recorded: str) -> str:
    """The part of another host's home path below its home directory, or ``""``.

    Splits on ``/`` deliberately rather than using local path semantics: the value
    was written on the other device, so it is a POSIX path on both macOS and Linux
    (``/home/<user>/…`` or ``/Users/<user>/…``), and the local OS's separator is not
    the question being asked.
    """
    parts = [part for part in str(recorded).split("/") if part]
    if len(parts) >= 3 and parts[0] in ("home", "Users"):
        return "/".join(parts[2:])
    return ""


def translate_host_path(recorded: str, *, home: str = "") -> str:
    """Map another host's absolute path onto THIS host, or ``""`` when it cannot.

    The same rule :func:`rewrite_host_cwd` applies to the marker, minus its
    always-succeeding fallback (the caller owns that choice): a path that exists
    here is kept; else the same trailing component under this user's home; else
    the tail below the other device's home, when that exists here — and a path
    that IS the other device's home maps to this home (``/Users/damian`` →
    ``/home/ec2-user``, the 2026-11-13 move). Relative or empty input is returned
    unchanged: a relative cwd was resolved by its writer. Never raises.
    """
    value = str(recorded or "")
    if not value or not os.path.isabs(value) or os.path.exists(value):
        return value
    home = home or os.path.expanduser("~")
    parts = [part for part in value.split("/") if part]
    if len(parts) == 2 and parts[0] in ("home", "Users"):
        return home
    same_basename = os.path.join(home, os.path.basename(value.rstrip("/")))
    if os.path.exists(same_basename):
        return same_basename
    tail = _home_relative_tail(value)
    if tail and os.path.exists(os.path.join(home, tail)):
        return os.path.join(home, tail)
    return ""


def _under(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or below it (string-wise; ``path`` may not exist)."""
    path = os.path.normpath(path)
    return path == root or path.startswith(root.rstrip("/") + "/")


def translate_monitor_cwds(
    specs: Sequence[Mapping[str, Any]], *, session_cwd: str, home: str = ""
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rewrite each carried monitor's ``cwd`` for THIS host.

    WHY (2026-11-13 move, Mac → lop-remote-node-2): three ``gh pr view`` monitors
    travelled in the transcript with ``cwd=/Users/damian`` and every first check on
    the node failed with "cannot start the command in '/Users/damian'" — the move
    translated the SESSION's cwd (``carry.cwd``) but not the monitors', which each
    capture their own absolute cwd at arm (``monitors/arm._captured_cwd``).

    A cwd that :func:`translate_host_path` can map is mapped; one it cannot is
    re-armed at ``session_cwd`` (the session's destination cwd) and COUNTED, so the
    receipt can say so rather than leaving a monitor that fails every tick.
    Returns ``(rows, {"translated": n, "fallback": n})``; rows whose cwd already
    fits are returned unchanged, so a second pass is a no-op.

    LOCAL PATHS ARE NEVER TRANSLATED (review r1 m2). The session-load seat runs
    this for EVERY session, moved or not, so a cwd under THIS host's own home is
    this host's path: a missing one (a removed worktree) is a real local failure
    that must keep failing loudly, not silently re-arm somewhere else. Only a cwd
    outside this home — another device's ``/Users/<u>`` or ``/home/<u>`` — is
    foreign and mapped.
    """
    rows: list[dict[str, Any]] = []
    counts = {"translated": 0, "fallback": 0}
    raw_home = home or os.path.expanduser("~")
    # Both spellings: a symlinked home (macOS ``/var`` → ``/private/var``) must not
    # make this host's own path read as foreign.
    local_homes = {os.path.normpath(raw_home), os.path.realpath(raw_home)}
    for raw in specs:
        row = dict(raw)
        cwd = row.get("cwd")
        if (
            isinstance(cwd, str)
            and cwd
            and os.path.isabs(cwd)
            and not os.path.exists(cwd)
            and not any(_under(cwd, local) for local in local_homes)
        ):
            mapped = translate_host_path(cwd, home=home)
            if mapped:
                counts["translated"] += 1
            else:
                mapped = session_cwd
                counts["fallback"] += 1
            row["cwd"] = mapped
        rows.append(row)
    return rows, counts


def _write_marker(marker: Path, document: Mapping[str, Any]) -> None:
    """Write ``desktop.json`` atomically, owner-only.

    A promote either leaves the old marker or the rewritten one — never a half
    written file a runtime would read as corrupt — and 0600 matches what the marker
    is: this device's own note about where the conversation lives, never shared.
    """
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, sort_keys=True))
    os.replace(temporary, marker)


def rewrite_host_cwd(
    target: Path, session_id: str, *, config_dir: Path, source_cwd: str = ""
) -> dict[str, str]:
    """Map a moved session's recorded cwd onto a path that exists on THIS host.

    WHY AT THE PROMOTE, before :func:`rebuild_indexes` reads it: the wake and monitor
    indexes are derived FROM the marker (``resolve_cwd`` reads it), so rewriting
    later would leave both indexes naming a directory this device does not have —
    the failure a moved conversation showed as an opened session whose working
    directory was another machine's home. The TRANSCRIPT IS NEVER TOUCHED: it is the
    verified conversation the copy was checked against byte-for-byte, and a cwd is
    host state, not conversation content.

    The three steps are the design's (clause 4): keep the recorded path when it
    exists here (or is relative — a relative cwd was already resolved against the
    session by its writer); else the same trailing component under this user's home;
    else the path's tail below the other device's home directory, when that exists
    here; else this user's home. ``config_dir`` is part of the signature so this
    lives beside the rest of the carry policy and a future writer has the root it
    needs — this function reads only the session's own marker.

    ``source_cwd`` IS THE MARKER-LESS CASE (the owner's report, clause 4): a session
    the TUI created may carry no ``desktop.json`` at all, so the source sends the cwd
    it knows on the wire instead (``_source_prepare``'s additive key, carried in
    ``ready.json``). When the marker is absent, unreadable or empty, that value is
    the recorded one — mapped by the SAME three steps and then WRITTEN as this
    host's ``desktop.json`` (0600, atomic), so the marker, the rebuilt wake index and
    the monitor index all carry one path. A marker that already records a cwd wins,
    and no ``source_cwd`` at all leaves the session exactly as before.

    Returns ``{"from": old, "to": new}`` when it rewrote, ``{}`` when it did not,
    which is what the receipt's ``carry.cwd`` reports. Creating the marker from
    ``source_cwd`` counts as a rewrite even when the mapped path equals the recorded
    one: the file did not exist before and does now. Never raises: a promote of
    verified bytes must not fail over a mapping it could not compute (the marker's
    reader tolerates the old value; the index self-heals on the next open).
    """
    marker_path = Path(target) / DESKTOP_MARKER_NAME
    marker: dict[str, Any] = {}
    try:
        loaded = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        loaded = None
    if isinstance(loaded, dict):
        marker = loaded
    recorded = marker.get("cwd")
    recorded_value = recorded if isinstance(recorded, str) and recorded.strip() else ""
    # True when this session never recorded a cwd of its own — the case the
    # ``source_cwd`` parameter exists for. No marker at all is the same statement.
    from_source = not recorded_value
    if from_source:
        recorded_value = source_cwd if source_cwd.strip() else ""
    if not recorded_value:
        return {}
    if not from_source:
        # AN EXISTING VALUE IS ONLY REWRITTEN WHEN IT DOES NOT FIT HERE: a relative
        # path was resolved by its writer, and one that exists is already usable.
        if not os.path.isabs(recorded_value) or os.path.exists(recorded_value):
            return {}
    home = os.path.expanduser("~")
    mapped = recorded_value
    if os.path.isabs(recorded_value) and not os.path.exists(recorded_value):
        same_basename = os.path.join(home, os.path.basename(recorded_value))
        if os.path.exists(same_basename):
            mapped = same_basename
        else:
            tail = _home_relative_tail(recorded_value)
            if tail and os.path.exists(os.path.join(home, tail)):
                mapped = os.path.join(home, tail)
            else:
                # THE FALLBACK THAT ALWAYS SUCCEEDS: home exists wherever a runtime
                # does, and a session opened at home is usable — a path that does not
                # exist is not. Nothing records that the mapping gave up beyond the
                # receipt's two paths.
                mapped = home
    try:
        marker["cwd"] = mapped
        _write_marker(marker_path, marker)
    except OSError:
        logger.warning("carry: could not rewrite the cwd marker for %s", session_id, exc_info=True)
        return {}
    logger.info(
        "carry: rewrote %s's cwd for this host (%s): %s -> %s",
        session_id,
        config_dir,
        recorded_value,
        mapped,
    )
    return {"from": recorded_value, "to": mapped}


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
        existing = monitor_store.read_entry(root, session_id)
        cwd = resolve_cwd(directory, existing)
        # The index names the cwd each check runs in; the transcript keeps the
        # source host's value (never edited — see ``rewrite_host_cwd``) and the
        # session re-applies this same translation when it loads the specs.
        specs, cwd_counts = translate_monitor_cwds(specs, session_cwd=cwd)
        if cwd_counts["translated"] or cwd_counts["fallback"]:
            report["monitor_cwds"] = cwd_counts
        rows = _monitor_index_rows(specs)
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


#: Bounds on what a move carries of the session's projects. The prepare answer
#: is one wire frame, and a project row's history is soft-bounded at
#: ``projects.UPDATES_MAX`` (500) entries of up to ``PROGRESS_MAX`` characters,
#: so the newest entries are carried and the count is stated in the receipt.
CARRIED_PROJECTS_MAX = 8
CARRIED_UPDATES_MAX = 100
#: Receiver-side byte caps (review r1 M1). After the entry trims above, an honest
#: row is ~100 KB at most (100 × 1000-character entries plus bounded fields), so
#: 256 KB a row and 1 MB a carry refuse only rows no current sender produces.
CARRIED_PROJECT_BYTES_MAX = 256 * 1024
CARRIED_TOTAL_BYTES_MAX = 1024 * 1024


def linked_project_rows(config_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """The SOURCE half of the project carry: rows this session is linked to.

    WHY: the project store is per-device (``<config>/projects/<id>.json``), so a
    moved session's linked project did not exist on the destination (2026-11-13,
    ``secret-guardrails-relax``: "no project named" on lop-remote-node-2). Rows are
    serialized as stored (minus attachment files, which stay here) with the newest
    :data:`CARRIED_UPDATES_MAX` history entries. Never raises: a project store that
    cannot be read must not refuse a move.
    """
    try:
        from local_operator.projects import ProjectRegistry

        registry = ProjectRegistry(Path(config_dir))
        projects = registry.projects_for_session(session_id)
    except Exception:  # noqa: BLE001 — see the docstring
        logger.warning("carry: could not read projects linked to %s", session_id, exc_info=True)
        return []
    rows: list[dict[str, Any]] = []
    for project in projects[:CARRIED_PROJECTS_MAX]:
        row = project.model_dump(mode="json", by_alias=True)
        updates = list(row.get("updates") or [])
        # Counted before they are dropped so the destination's receipt can say
        # the history arrived without its files (``Project`` ignores the key).
        row["attachments_not_carried"] = sum(
            len(entry.get("attachments") or []) for entry in updates
        )
        row["updates"] = [dict(entry, attachments=[]) for entry in updates][-CARRIED_UPDATES_MAX:]
        rows.append(row)
    return rows


def adopt_projects(config_dir: Path, session_id: str, rows: Any) -> dict[str, Any]:
    """The DESTINATION half: merge carried rows into this device's project store.

    Returns the receipt block ``{"created": [names], "merged": [names],
    "unlinked": [names], "failed": [names], "refused": [names], "truncated":
    [names], "attachments": "not carried", "limits": "<sentence>"}`` (keys present
    only when non-empty), or ``{}`` when nothing was carried. A row that cannot
    merge (for example one written by a newer build) is named under ``failed``
    rather than failing the promote of a verified copy.

    RECEIVER-SIDE CAPS (review r1 M1, reproduced: one carried row of 700 × 20 KB
    entries stored as a 14 MB project). The rows arrive from a peer, so the
    sender's bounds in :func:`linked_project_rows` are re-applied HERE rather than
    trusted: at most :data:`CARRIED_PROJECTS_MAX` rows, the newest
    :data:`CARRIED_UPDATES_MAX` history entries per row with each text cut to
    ``projects.PROGRESS_MAX`` (named under ``truncated``), and a row still larger
    than :data:`CARRIED_PROJECT_BYTES_MAX` — or one that would take the carry past
    :data:`CARRIED_TOTAL_BYTES_MAX` — is not adopted at all (named under
    ``refused``). Anything trimmed or refused is stated in ``limits`` so the
    receipt never implies the whole history arrived.
    """
    if not isinstance(rows, list) or not rows:
        return {}
    from local_operator.projects import PROGRESS_MAX, ProjectRegistry

    registry = ProjectRegistry(Path(config_dir))
    report: dict[str, list[str]] = {
        "created": [],
        "merged": [],
        "unlinked": [],
        "failed": [],
        "refused": [],
        "truncated": [],
    }
    attachments_dropped = False
    candidates = [row for row in rows if isinstance(row, Mapping)]
    over_count = max(0, len(candidates) - CARRIED_PROJECTS_MAX)
    total_bytes = 0
    for raw in candidates[:CARRIED_PROJECTS_MAX]:
        name = str(raw.get("name") or "")
        attachments_dropped = attachments_dropped or bool(raw.get("attachments_not_carried"))
        entries = [entry for entry in (raw.get("updates") or []) if isinstance(entry, Mapping)]
        kept = entries[-CARRIED_UPDATES_MAX:]
        trimmed = len(entries) != len(kept)
        bounded: list[dict[str, Any]] = []
        for entry in kept:
            text = str(entry.get("text") or "")
            if len(text) > PROGRESS_MAX:
                trimmed = True
                text = text[:PROGRESS_MAX]
            bounded.append({**dict(entry), "text": text})
        row = {**dict(raw), "updates": bounded}
        try:
            size = len(json.dumps(row, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            report["failed"].append(name)
            continue
        if size > CARRIED_PROJECT_BYTES_MAX or total_bytes + size > CARRIED_TOTAL_BYTES_MAX:
            logger.warning(
                "carry: refused project %r for %s (%d bytes after trimming)",
                name,
                session_id,
                size,
            )
            report["refused"].append(name)
            continue
        total_bytes += size
        if trimmed:
            report["truncated"].append(name)
        try:
            _project, action = registry.adopt_carried(row, session_id)
        except Exception:  # noqa: BLE001 — named in the receipt, never a failed promote
            logger.warning("carry: could not adopt project %r for %s", name, exc_info=True)
            report["failed"].append(name)
            continue
        if action == "created":
            report["created"].append(name)
        elif action == "unlinked":
            report["unlinked"].append(name)
        else:
            report["merged"].append(name)
    block: dict[str, Any] = {key: value for key, value in report.items() if value}
    if block and attachments_dropped:
        block["attachments"] = "not carried"
    limits: list[str] = []
    if report["truncated"]:
        limits.append(
            f"history kept to the newest {CARRIED_UPDATES_MAX} entries of up to "
            f"{PROGRESS_MAX} characters for {', '.join(report['truncated'])}"
        )
    if report["refused"]:
        limits.append(
            f"{', '.join(report['refused'])} not adopted: larger than this device "
            f"accepts ({CARRIED_PROJECT_BYTES_MAX // 1024} KB a project, "
            f"{CARRIED_TOTAL_BYTES_MAX // 1024} KB in total)"
        )
    if over_count:
        limits.append(
            f"{over_count} more linked project{'s' if over_count != 1 else ''} not carried "
            f"(at most {CARRIED_PROJECTS_MAX})"
        )
    if report["unlinked"]:
        limits.append(
            f"this session could not be linked to {', '.join(report['unlinked'])}: "
            "it already holds the most linked sessions a project may"
        )
    if limits:
        block["limits"] = "; ".join(limits)
    return block


def attachment_resolution(config_dir: Path, session_dir: Path) -> dict[str, Any]:
    """Whether the moved session's team/agent resolves on THIS device.

    WHY (2026-11-13): the moved session's prompt on lop-remote-node-2 carried its
    skills but not the lopdev team brief, and nothing said so. ``attachment.json``
    travels in the copy set and the session re-attaches from it by NAME at open
    (``Session._restore_attachment``), so the brief appears exactly when this
    device's registries resolve that name — which the source's definitions push
    (``mobility._push_move_definitions``, now on the queued path too) is meant to
    guarantee. This reads the same registries at the promote and reports the
    answer, so a brief that will NOT be there is named on the receipt rather than
    discovered from a prompt. ``{}`` when the session has no team or agent.
    Never raises.
    """
    try:
        from local_operator.resume import read_session_attachment

        stored = read_session_attachment(Path(session_dir))
    except Exception:  # noqa: BLE001 — see the docstring
        return {}
    if stored is None or not (stored.team or stored.agent):
        return {}
    block: dict[str, Any] = {}
    missing: list[str] = []
    if stored.team:
        try:
            from local_operator.teams import TeamRegistry

            resolved = TeamRegistry(Path(config_dir)).get_team_by_name(stored.team) is not None
        except Exception:  # noqa: BLE001 — unreadable registry reads as unresolved
            logger.warning("carry: could not resolve team %r", stored.team, exc_info=True)
            resolved = False
        block["name"] = stored.team
        block["resolved"] = resolved
        if not resolved:
            missing.append(f"team {stored.team!r}")
    else:
        # Review r1 N1: only when NO team is stored. A stored team owns the agent
        # slot, so ``Session._apply_stored_attachment`` treats a stored agent beside
        # it as superseded, never missing — naming it here would blame the wrong cause.
        try:
            from local_operator.agent_profiles import resolve_profile_or_specialist
            from local_operator.agents import AgentRegistry

            kind = resolve_profile_or_specialist(
                stored.agent, registry=AgentRegistry(Path(config_dir))
            )[0]
            resolved = kind is not None
        except Exception:  # noqa: BLE001 — unreadable registry reads as unresolved
            logger.warning("carry: could not resolve agent %r", stored.agent, exc_info=True)
            resolved = False
        block["agent"] = stored.agent
        block["agent_resolved"] = resolved
        if not resolved:
            missing.append(f"agent {stored.agent!r}")
    # Review r1 m4: the session's restore refuses to attach anything while the
    # profile registry is incomplete (``require_complete_metadata`` raising
    # ``ProfileRegistryUnavailable``), even for a name that resolves. Mirror that
    # gate, so ``applied`` is the answer the runtime's restore will actually reach.
    gate_error = ""
    try:
        from local_operator.agents import AgentRegistry
        from local_operator.session.errors import ProfileRegistryUnavailable

        try:
            AgentRegistry(Path(config_dir)).require_complete_metadata()
        except ProfileRegistryUnavailable as error:
            gate_error = str(error) or "the agent registry is incomplete"
    except Exception:  # noqa: BLE001 — a gate this cannot run is not a refusal
        logger.debug("carry: could not run the registry metadata gate", exc_info=True)
    # ``applied``: whether the brief is in the conversation's prompt when it opens
    # here. The runtime starts AFTER this promote (the engage does not cross the
    # done frame), so this is the restore's own decision computed against the same
    # registries; a runtime that opened unresolved re-resolves before each turn
    # (``Session._reresolve_unresolved_attachment``), so a definition that arrives
    # later is picked up without a reopen.
    block["applied"] = bool(resolved) and not gate_error
    if gate_error:
        block["notice"] = (
            f"the {'team' if stored.team else 'agent'} brief is not applied yet: {gate_error}"
        )
    elif missing:
        block["notice"] = (
            f"{missing[0]} is not defined on this device, so the conversation opens "
            "without that brief; send it with `lop network definitions push` from the "
            "device that has it — the conversation picks it up on its next turn"
        )
    return block
