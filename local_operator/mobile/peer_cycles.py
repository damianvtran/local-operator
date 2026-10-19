"""Optional owner-certified phone grouping; never authority from raw row details.

The neutral validator owns source/outcome proof. This adapter owns only mapping
those exact source IDs onto the phone's EXISTING row IDs and payload mapper.
Default folds receive no records and keep their old wire shape.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from local_operator.harness.peer_cycle_display import (
    validate_indexed_record,
    validate_record,
)
from local_operator.harness.peer_cycles import PeerCycleReplay
from local_operator.mobile.types import TranscriptEntry


def bind_owner_peer_cycles(
    fold: Any, session: Any, notify: Callable[[], None], loop: Any
) -> Callable[[], None]:
    """Observe the existing frontend publication, including metadata-only close.

    ToolEnd fan-out precedes the asynchronous display-record append. Watching
    ONLY tool events loses the close at idle. This one specialized subscriber
    also retries after raw-parent commits, without polling or rescanning history.
    The callback is marshalled to the same owner loop as ordinary fold events.
    """
    store = getattr(session, "_frontend_state_store", None)
    # The INTERNAL entry point: this observer is not a viewer, and must not
    # flip ``has_subscribers``. A viewer-less runtime otherwise started writing
    # the turn-end checkpoint a truly unattended one skips; that row carried a
    # raw transcript row id as ``history_cursor`` and broke the durable-journal
    # invariant the desktop ask-images e2e pins (see the store's
    # ``_internal_subscribers``).
    subscribe_internal = getattr(store, "subscribe_internal_threadsafe", None)
    if loop is None or store is None or subscribe_internal is None:
        return lambda: None
    closed = False
    context: dict[str, Any] = {"records": None, "epoch": None}

    def apply(update: Any) -> None:
        if closed:
            return
        changes = getattr(update, "changes", {})
        if "peer_cycle_display" in changes:
            context["records"] = changes["peer_cycle_display"]
        context["epoch"] = getattr(update, "epoch", context["epoch"])
        before = fold.projection.version
        try:
            fold.set_peer_cycle_source(
                context["records"], transcript=session.transcript, owner_epoch=context["epoch"]
            )
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
            # Optional display failure cannot abort the agent or print payloads.
            import logging

            logging.getLogger(__name__).warning(
                "phone peer display refused (%s)", type(error).__name__
            )
            return
        if fold.projection.version != before:
            notify()

    def enqueue(update: Any) -> None:
        if not closed and not loop.is_closed():
            loop.call_soon_threadsafe(apply, update)

    subscription = subscribe_internal(enqueue)
    snapshot = subscription.sync.snapshot
    if snapshot.session_id != fold.projection.session_id or not hasattr(
        snapshot, "peer_cycle_display"
    ):
        subscription.unsubscribe()
        return lambda: None
    context.update(records=snapshot.peer_cycle_display, epoch=snapshot.epoch)
    # Apply the captured snapshot before any queued post-join delta, never a
    # second independently timed snapshot of the store's mutable whole state.
    fold.set_peer_cycle_source(
        context["records"], transcript=session.transcript, owner_epoch=context["epoch"]
    )

    def unsubscribe() -> None:
        nonlocal closed
        closed = True
        subscription.unsubscribe()

    return unsubscribe


def retained_peer_graphs(rows: list[TranscriptEntry], session_id: str) -> list[dict[str, Any]]:
    """Retain only complete graphs this caller JUST certified, not their bodies."""
    by_id = {row.id: row for row in rows}
    retained = []
    for row in rows:
        stamp = row.details.get("peer_cycle")
        if (
            not isinstance(stamp, dict)
            or stamp.get("session_id") != session_id
            or any(key not in by_id for key in stamp.get("member_ids", ()))
        ):
            continue
        members = [by_id[key] for key in stamp["member_ids"]]
        retained.append(
            {
                "session_id": session_id,
                "quiet_call_id": stamp["quiet_call_id"],
                "member_ids": [
                    (
                        f"{member.id}:0"
                        if member.kind == "peer_message"
                        else f"tool:{member.tool_call_id}:0"
                    )
                    for member in members
                ],
            }
        )
    return retained


def project_peer_cycles(
    entries: list[TranscriptEntry],
    *,
    session_id: str | None = None,
    records: Any = None,
    history: Any = (),
    transcript: Any = None,
    child_row: Callable[[dict[str, Any]], TranscriptEntry] | None = None,
    certificates: Any = (),
    retained: Any = (),
    view_epoch: int = 0,
) -> list[TranscriptEntry]:
    # A custom producer can supply arbitrary details. Strip the RESERVED key
    # from a COPY before considering separately delivered, validated evidence.
    rows = [
        (
            replace(
                row,
                details={key: value for key, value in row.details.items() if key != "peer_cycle"},
            )
            if "peer_cycle" in row.details
            else row
        )
        for row in entries
    ]
    if not session_id:
        return rows
    graphs = []
    replay = PeerCycleReplay((session_id, 0))
    for message in history[-4096:]:
        closed = replay.step(message)
        if closed is not None:
            graphs.append((list(closed.member_ids), closed.quiet_call_id, []))
    for closed in certificates:
        if closed.session_id == session_id and closed.view_epoch == view_epoch:
            graphs.append((list(closed.member_ids), closed.quiet_call_id, []))
    # Retained stamps come ONLY from this fold's own previous successful joins,
    # never a reconstructed raw entry. They contain IDs, not another body cache.
    for stamp in retained:
        if stamp.get("session_id") == session_id:
            graphs.append((stamp["member_ids"], stamp["quiet_call_id"], []))
    remaining_bytes = 512 * 1024
    candidates = records.get("records", ()) if isinstance(records, dict) else ()
    if not isinstance(candidates, (list, tuple)) or len(candidates) > 4096:
        candidates = ()
    for candidate in candidates:
        if remaining_bytes <= 0:
            break
        if transcript is not None:
            record, consumed = validate_indexed_record(
                transcript, candidate, session_id, max_bytes=remaining_bytes
            )
            remaining_bytes -= consumed
        else:
            record = validate_record(candidate, history[-4096:], session_id)
        if record is not None:
            graphs.append((record["member_ids"], record["quiet_call_id"], record["child_rows"]))
    claimed: set[str] = set()
    for source_ids, quiet_call_id, child_rows in graphs:
        if claimed.intersection(source_ids):
            continue
        source_rows: dict[str, TranscriptEntry] = {}
        duplicate = False
        for row in rows:
            source = (
                f"{row.id}:0"
                if row.kind == "peer_message"
                else (
                    f"tool:{row.tool_call_id}:0"
                    if row.kind == "tool" and row.tool_call_id
                    else None
                )
            )
            if source is not None:
                if source in source_rows:
                    duplicate = True
                source_rows[source] = row
        if duplicate:
            continue
        additions: dict[str, list[TranscriptEntry]] = {}
        replacements: dict[str, TranscriptEntry] = {}
        available = dict(source_rows)
        for child in child_rows:
            if child["source_id"] in available:
                existing = available[child["source_id"]]
                if (
                    child_row is not None
                    and existing.kind == "tool"
                    and existing.tool_state == "done"
                    and not existing.error
                    and not existing.images
                    and not existing.videos
                    and existing.tool_name == child["tool_name"]
                    and existing.tool_call_id == child["tool_call_id"]
                ):
                    # The bounded post-redaction child is the only FULL source
                    # for this row. Replace an 8k preview, never the identity or
                    # reader-owned state, and never hide a conflicting failure.
                    complete = child_row(child)
                    restored = replace(existing, details=complete.details, text_complete=True)
                    available[child["source_id"]] = restored
                    replacements[existing.id] = restored
                continue
            root = available.get(f"tool:{child['root_id']}:0")
            if root is None or child_row is None:
                continue
            restored = child_row(child)
            if any(row.id == restored.id for row in rows) or any(
                row.id == restored.id for group in additions.values() for row in group
            ):
                continue
            available[child["source_id"]] = restored
            additions.setdefault(root.id, []).append(restored)
        member_ids = source_ids
        if any(source not in available for source in member_ids):
            continue
        proposed = []
        for row in rows:
            proposed.append(replacements.get(row.id, row))
            proposed.extend(additions.get(row.id, ()))
        members = [available[source] for source in member_ids]
        ids = [row.id for row in members]
        positions = [index for index, row in enumerate(proposed) if row.id in set(ids)]
        # No jumping over a public answer, attention, unknown row or page gap.
        # The complete finite graph must be one ordered physical interval.
        if (
            len(ids) != len(set(ids))
            or not positions
            or positions != list(range(positions[0], positions[0] + len(ids)))
            or [proposed[index].id for index in positions] != ids
            or members[0].kind != "peer_message"
            or any(
                row.kind != "tool"
                or row.tool_state != "done"
                or row.error
                or not row.text_complete
                or row.images
                or row.videos
                or row.tool_name not in {"send", "eval"}
                for row in members[1:]
            )
        ):
            continue
        stamp = {
            "version": 1,
            "session_id": session_id,
            "first_id": ids[0],
            "last_id": ids[-1],
            "member_ids": ids,
            "quiet_call_id": quiet_call_id,
        }
        first = replace(members[0], details={**members[0].details, "peer_cycle": stamp})
        rows = [first if row.id == first.id else row for row in proposed]
        claimed.update(member_ids)
    return rows
