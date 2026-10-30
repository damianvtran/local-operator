"""Arrivals that woke NOBODY, made countable and visible (wake contract).

WHY THIS EXISTS. The connector's ``badge_only`` outcome is the one where a
conversation message reaches this machine and wakes no session at all: the owner
has agent messages on, the consent is accepted, and there is no designated
session for a card to land in (§5.6). That is a legitimate state, but it must
not be an INVISIBLE one — before this module the only trace of it was an
``action`` in an iteration summary nobody reads, so "why does my agent never
answer?" had no answer anywhere an owner could look.

WHAT IS RECORDED. A count per closed code and when it last happened. Nothing
else: no conversation id, no message id, no sender, no body. That an arrival
woke nobody is operational; the arrival itself is peer content and stays out of
this file (the same rule the receiver's own log lines follow).

SHAPE. A small JSON file beside the bridge store, written by atomic replace. The
connector holds the connector singleton lock, so there is exactly ONE writer;
the reader (``lop team-bridge receiver status``) is a different process, must
never create or repair the file, and answers "nothing recorded" for a missing
one. Deliberately NOT a table in ``state.sqlite3``: that schema is
migration-versioned, and this state is disposable — losing it loses a count.

NEVER RAISES. An arrival's handling must not fail because a diagnostics counter
could not be written; the WARNING line has already carried the fact by then.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The typed code the badge-only case reports. Closed vocabulary: the fence
#: says ``no_designation`` when consent is accepted but no session is
#: designated, and that is the code an operator greps for.
CODE_NO_DESIGNATION = "no_designation"

_SCHEMA = 1


def diagnostics_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "wake_diagnostics.json"


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def record_woke_nobody(config_dir: Path | str, *, code: str) -> None:
    """Log and count one arrival that woke nobody. Never raises."""

    # The line a person (or an agent reading the receiver log) actually sees:
    # one WARNING per arrival, with the typed code. Codes are a closed set from
    # the fence, never peer content, so this is safe to name literally.
    logger.warning("bridge: conversation arrival woke nobody code=%s", code)
    path = diagnostics_path(config_dir)
    by_code = _read(path).get("by_code")
    counts = dict(by_code) if isinstance(by_code, dict) else {}
    try:
        counts[code] = int(counts.get(code, 0)) + 1
    except (TypeError, ValueError):
        counts[code] = 1
    payload = {"schema": _SCHEMA, "by_code": counts, "last_at": time.time()}
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # A counter that cannot be written changes nothing about the arrival;
        # the log line above already said it woke nobody.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def woke_nobody_counts(config_dir: Path | str) -> dict[str, int]:
    """Counts per code since the file was last reset (``{}`` = none recorded).

    Read-only by construction: a missing or unreadable file is an empty answer,
    never a created one (the status command runs on machines that were never
    enrolled and must not leave files behind).
    """

    by_code = _read(diagnostics_path(config_dir)).get("by_code")
    if not isinstance(by_code, dict):
        return {}
    counts: dict[str, int] = {}
    for code, value in by_code.items():
        if isinstance(code, str) and isinstance(value, int) and not isinstance(value, bool):
            counts[code] = value
    return counts


__all__ = [
    "CODE_NO_DESIGNATION",
    "diagnostics_path",
    "record_woke_nobody",
    "woke_nobody_counts",
]
