"""Advisory session presence, never acknowledgement of an inbox revision.

The operator's presence addendum supersedes the original rung-0 creation
predicate. A focused session and fresh OS activity cannot prove that a new
callout rendered: it may be scrolled off, covered, in a child pane, or not
painted yet. This evaluator remains available to the banner lane as advisory
evidence; callers MUST NOT mark an item read from its verdict. Creation is
always unread. Only the per-item revision/anchor-bound receipt can auto-read.

Unknown or stale fields fail toward not active. The lease readers remain
stdlib-light for runtime and cold-route callers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The input-freshness window, in seconds. The SAME number as the TUI's
#: ``tui/attention.py`` ``ATTENTION_INPUT_EVIDENCE_S`` (120 s), restated rather
#: than imported because that module loads Textual-adjacent code; a test pins
#: the two equal so they cannot drift.
INPUT_EVIDENCE_S = 120.0


@dataclass(frozen=True)
class TuiEvidence:
    """What an attached terminal said about the operator, as the runtime knows it.

    ``focused`` is the attach connection's last focus report and ``input_at_ms``
    the epoch-ms of its last real Key/MouseDown edge (``tui/attention.py``:
    wheel, hover and forwarded events are not evidence). ``None`` means the
    terminal never reported one, which is "not active".
    """

    focused: bool = False
    input_at_ms: int | None = None


@dataclass(frozen=True)
class WatchingVerdict:
    """The rung-0 answer with the evidence it rests on, for the write log.

    ``surface`` is ``"desktop"`` or ``"tui"`` when watching, else ``""``.
    ``input_age_s`` is how old the evidence was, logged on every rung-0 write
    so a false suppression can be spot-checked (contract §14 risk 3).
    """

    watching: bool
    surface: str = ""
    input_age_s: float | None = None
    reason: str = ""


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _desktop_input_fields(root: Path | None) -> tuple[int | None, bool | None]:
    """``(input_at_ms, screen_locked)`` from the freshest live windowed record.

    Read directly from the lease records rather than through
    :class:`DesktopPresence`, because that dataclass is the shipped reader for
    rungs 1 and 2 and these two fields are additive: widening it would change
    a frozen shape every banner path already compares against. The same reaping
    (``_load_record``) and the same "freshest windowed record wins" rule apply,
    so the fields describe the very window whose ``session_id`` was matched.
    """
    from local_operator.session.runtime import presence as lease

    live: list[dict[str, Any]] = []
    for path in lease._record_paths(root):
        record = lease._load_record(path)
        if record is not None:
            live.append(record)
    windowed = [record for record in live if lease._window_of(record).get("exists")]
    if not windowed:
        return None, None
    newest = max(windowed, key=lambda record: float(record.get("heartbeat_at") or 0.0))
    locked = newest.get("screen_locked")
    return _int_or_none(newest.get("input_at_ms")), locked if isinstance(locked, bool) else None


def watching_and_active(
    session_id: str,
    *,
    root: Path | None = None,
    tui: TuiEvidence | None = None,
    now_ms: int | None = None,
) -> WatchingVerdict:
    """Rung 0, read fresh: is the operator in ``session_id`` and active right now?"""
    stamp = now_ms if now_ms is not None else int(time.time() * 1000)
    window_ms = int(INPUT_EVIDENCE_S * 1000)
    if not session_id:
        return WatchingVerdict(False, reason="no_session")
    try:
        from local_operator.session.runtime.presence import read_delivery

        # Uncached: this is a TERMINAL decision (an item recorded read is never
        # revisited), the `desktop_presence(cached=False)` rule.
        presence = read_delivery(root)
    except Exception:  # noqa: BLE001 — an unreadable lease is "not watching"
        presence = None
    if presence is not None and presence.attended and presence.session_id == session_id:
        try:
            input_at, locked = _desktop_input_fields(root)
        except Exception:  # noqa: BLE001 — unknown is "not active"
            input_at, locked = None, None
        if locked is False and input_at is not None:
            age = stamp - input_at
            if 0 <= age <= window_ms:
                return WatchingVerdict(True, "desktop", age / 1000.0, "desktop_input")
    if tui is not None and tui.focused and tui.input_at_ms is not None:
        age = stamp - int(tui.input_at_ms)
        if 0 <= age <= window_ms:
            return WatchingVerdict(True, "tui", age / 1000.0, "tui_input")
    return WatchingVerdict(False, reason="no_fresh_evidence")


def describe(verdict: WatchingVerdict) -> str:
    """A compact JSON line for the write log (contract §14 risk 3)."""
    return json.dumps(
        {
            "watching": verdict.watching,
            "surface": verdict.surface,
            "input_age_s": verdict.input_age_s,
            "reason": verdict.reason,
        },
        separators=(",", ":"),
    )
