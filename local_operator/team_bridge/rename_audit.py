"""Local audit trail for the bridge tool's owner-acting ``channel_rename``.

WHY. ``channel_rename`` is the one bridge op an agent runs AS ITS HUMAN OWNER:
it goes through ``TeamBridgeClient.conversation_rename`` with no
``actor_endpoint_id`` (the hub refuses agent renames in v1), so on the hub the
rename is indistinguishable from the owner clicking Rename in the desktop app.
The only record that an AGENT made the call, and which operator turn allowed
it, is this file (operator decision, 2026-11-21). Every attempt is written,
allowed or refused, so "who renamed #x, and did I ask for it?" is answerable
months later.

WHAT A ROW HOLDS, AND NEVER HOLDS. The conversation id (only when it is a
well-formed ``bcv_`` id; a malformed argument is recorded as ``null`` so free
model text never lands here), the requested name, the hub's
canonical name on success, the outcome code, the refusal reason, and
the authorizing operator row's transcript entry id. Never a message body,
never the operator's words, never a token or credential. The file is private
(``0600`` in a ``0700`` directory it creates), like the recipient-resolution
log beside it.

FAIL CLOSED. ``record`` raises :class:`RenameAuditUnavailable` when the row
cannot be written; the caller writes the ``authorized`` row BEFORE sending, so
an unrecordable rename is never sent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

RENAME_AUDIT_FILENAME = "conversation_renames.jsonl"


class RenameAuditUnavailable(OSError):
    """The audit row could not be written; nothing may be sent."""


def rename_audit_path(config_dir: Path | str) -> Path:
    """``<config_dir>/team_bridge/conversation_renames.jsonl``."""

    return Path(config_dir) / "team_bridge" / RENAME_AUDIT_FILENAME


def record(
    *,
    config_dir: Path | str,
    session_id: str,
    conversation_id: str | None,
    name: str,
    outcome: str,
    reason: str = "",
    authorized_by_entry_id: str = "",
    canonical_name: str | None = None,
    http_status: int | None = None,
) -> None:
    """Append one row. Raises :class:`RenameAuditUnavailable` on failure."""

    row: dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "op": "channel_rename",
        "acted_as": "owner",
        "session_id": session_id,
        "conversation_id": conversation_id,
        "name": name,
        "outcome": outcome,
        "reason": reason or None,
        "authorized_by_entry_id": authorized_by_entry_id or None,
    }
    if canonical_name is not None:
        row["canonical_name"] = canonical_name
    if http_status is not None:
        row["http_status"] = http_status
    path = rename_audit_path(config_dir)
    directory = path.parent
    try:
        # Tighten only a directory this writer creates: ``team_bridge`` is shared
        # with the store and the outbox (same rule as recipient_resolutions).
        created = not directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        if created:
            os.chmod(directory, 0o700)
        line = json.dumps(row, ensure_ascii=False, default=str)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            handle = os.fdopen(descriptor, "a", encoding="utf-8")
        except OSError:
            os.close(descriptor)
            raise
        # The handle owns the descriptor from here; ``with`` closes it once.
        with handle:
            handle.write(line + "\n")
    except OSError as error:
        raise RenameAuditUnavailable(
            "the rename could not be recorded in the local audit log, so nothing was "
            f"sent ({error.__class__.__name__})"
        ) from error


__all__ = ["RENAME_AUDIT_FILENAME", "RenameAuditUnavailable", "rename_audit_path", "record"]
