"""Derive goal wakes from the owner's journal, not a second schedule to reconcile."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# A recovery probe is not the healthy turn loop. Match the existing five-minute
# residency sweep cadence so parking never turns an outage into a prompt storm.
GOAL_RETRY_S = 300.0


def retry_at(session_dir: Path) -> int | None:
    """Return the earliest recovery time, in epoch milliseconds.

    The sidecar's atomic write timestamps the last transition even if the owner
    exits just after writing it. Legacy waiting records already have this clock.
    """
    from local_operator.resume import (
        GOAL_SIDECAR_NAME,
        ORIGIN_FORK,
        USER_ORIGINS,
        _fork_instant,
        read_goal_record,
        session_origin,
    )
    from local_operator.session.goal_judge import GOAL_REARM_STATES

    try:
        updated = (session_dir / GOAL_SIDECAR_NAME).stat().st_mtime
    except OSError:
        return None
    record = read_goal_record(session_dir)
    if not record or record.get("status") != "active" or record.get("recover") is False:
        return None
    text, token, judge = record.get("goal"), record.get("token"), record.get("judge")
    if not (isinstance(text, str) and text.strip() and isinstance(token, str) and token):
        return None
    if not isinstance(judge, dict) or not isinstance(state := judge.get("state"), str):
        return None
    if state not in GOAL_REARM_STATES:
        return None
    origin = session_origin(session_dir)
    if origin and origin not in USER_ORIGINS:
        return None
    if origin == ORIGIN_FORK:
        # A fork may copy the parent's goal without the user opening the fork.
        # Only a subsequent goal transition on that fork can arm recovery.
        forked_at = _fork_instant(session_dir)
        if forked_at is not None and updated <= forked_at:
            return None
        if forked_at is None and record.get("recover") is not True:
            return None
    return int((updated + GOAL_RETRY_S) * 1000)


def candidates(config_dir: Path, index: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Find persisted goals, including those that never had an armed wake row."""
    from local_operator.resume import GOAL_SIDECAR_NAME, read_goal_record
    from local_operator.session.compression import has_transcript
    from local_operator.wakes.store import is_held

    rows = []
    for goal_file in (config_dir / "sessions").glob(f"*/{GOAL_SIDECAR_NAME}"):
        directory = goal_file.parent
        entry = index.get(directory.name) or {}
        if not isinstance(entry, dict) or is_held(entry) or directory.is_symlink():
            continue
        try:
            if (due := retry_at(directory)) is None or not has_transcript(directory):
                continue
        except OSError:
            continue
        record = read_goal_record(directory) or {}
        cwd = entry.get("cwd") or record.get("cwd")
        rows.append((directory.name, cwd if isinstance(cwd, str) else "", due))
    return rows
