"""Check a PARKED session's monitors without reopening it.

WHY THIS EXISTS. A parked session (``control.park_session``: runtime ended to
free memory, ``parked_at`` stamped, nothing held) keeps its monitors armed, and
the wake supervisor's monitors pass would cold-engage it for EVERY due tick —
a 60 s watch would bring a parked session back every minute, which hands the
memory straight back (field gap 2026-11-12: reopening sessions to keep their
watches alive "defeated the memory saving"). Most ticks are quiet, so the
supervisor runs the check here first and reopens the session only when there
is something for it to do.

THE RULE IS "QUIET OR ENGAGE", NEVER "DELIVER". This module never writes a
snapshot and never delivers a delta. It commits exactly one outcome itself —
the quiet tick (same normalized hash as the stored baseline): the counters
advance as the in-session scheduler's quiet path would (``checks``,
``last_check_at``, ``next_due_at`` via :func:`scheduler.advance_due`). Anything
else answers ``engage``: a change, a first check with no baseline, an error, an
empty/blind result, a monitor in a failing/blind/unavailable episode, or a tool
that cannot run outside a session (MCP, web tools needing session settings). The
supervisor then cold-engages the session exactly as for a due wake; the
session's own ``MonitorScheduler.load`` sees the row still due, re-runs the
check, and diffs it against the UNCHANGED baseline — so the delta is delivered
by the one owner of delivery, and nothing a parked session watched is missed.

SAME CODE, NOT A SECOND ALLOW-LIST. The tool objects are the builtin factories
the session itself uses (``build_bash_tool``/``build_read_tool``/
``build_grep_tool``); every check is re-gated by ``readonly.monitor_call_verdict``
at run time, as ``Session._run_monitor_check`` does; the comparable text is
``session._monitor_comparable_text`` and the hash is
``scheduler.normalize_result`` + ``diff.content_hash``. Any drift between this
and the session can only produce a spurious ``engage`` (the stored hash will
not match), never a missed change.

RUN AS A SHORT-LIVED SUBPROCESS (``python -m local_operator.monitors.parked``)
by the supervisor: importing the tool layer costs ~100 MB of RSS, which must not
become resident in the one long-lived process that runs during memory pressure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("local_operator.monitors.parked")

#: Builtin tools that run outside a session with no session-only state: their
#: factories need no ``ToolContext`` capability beyond ``cwd``. Every other tool
#: (MCP, web_fetch/web_search with session settings, eval…) answers ``engage``.
HEADLESS_TOOLS = ("bash", "read", "grep")

#: Counter keys whose presence means the monitor is mid-episode: a success there
#: owes a "restored"/"seeing again" notice that only the session can send.
_EPISODE_KEYS = ("unavailable_since", "failing", "blind", "disabled", "empty_checks")

QUIET = "quiet"
ENGAGE = "engage"


def is_parked(entry: Mapping[str, Any] | None) -> bool:
    """A monitor/wake index entry stamped by a PARK (and not also held)."""
    return (
        isinstance(entry, Mapping) and bool(entry.get("parked_at")) and not entry.get("stopped_at")
    )


def _build_tool(name: str) -> Any:
    from local_operator.tools import builtin

    if name == "bash":
        return builtin.build_bash_tool(None)
    if name == "read":
        return builtin.build_read_tool()
    if name == "grep":
        return builtin.build_grep_tool()
    return None


async def _check_one(
    config_dir: Path, session_id: str, row: Mapping[str, Any], settings: Any, now: int
) -> tuple[str, str]:
    """``(verdict, why)`` for one due monitor row; commits only a quiet tick."""
    from local_operator.harness.types import AbortSignal, ToolContext
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors.diff import content_hash
    from local_operator.monitors.readonly import (
        monitor_call_arguments,
        monitor_call_verdict,
    )
    from local_operator.monitors.scheduler import advance_due, normalize_result
    from local_operator.monitors.spec import MonitorSpec

    try:
        spec = MonitorSpec.model_validate(
            {key: row[key] for key in MonitorSpec.model_fields if key in row}
        )
    except Exception as exc:  # noqa: BLE001 — a bad row is the session's to report
        return ENGAGE, f"row unreadable ({type(exc).__name__})"
    counters = monitor_state.read_counters(config_dir, session_id, spec.id)
    if not counters or not counters.get("content_hash"):
        return ENGAGE, "no baseline yet"
    if any(counters.get(key) for key in _EPISODE_KEYS) or counters.get("consecutive_failures"):
        return ENGAGE, "monitor is in a failure/blind episode"
    tool = _build_tool(spec.tool)
    if tool is None:
        return ENGAGE, f"{spec.tool} cannot run outside a session"
    reason = monitor_call_verdict(tool, spec.arguments)
    if reason is not None:
        # Never run a call the gate refuses now; the session will count it.
        return ENGAGE, f"no longer read-only: {reason}"
    from local_operator.session.session import _monitor_comparable_text

    arguments = monitor_call_arguments(tool, spec.arguments)
    context = ToolContext(cwd=spec.cwd or ".", monitor_check=True, session_id=session_id)
    signal = AbortSignal()
    try:
        result = await asyncio.wait_for(
            tool.execute(f"parked-monitor-{spec.id}", arguments, signal, None, context),
            timeout=settings.run_timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        signal.abort("parked monitor check timed out")
        return ENGAGE, "check timed out"
    except Exception as exc:  # noqa: BLE001
        return ENGAGE, f"check raised {type(exc).__name__}"
    if getattr(result, "is_error", False):
        return ENGAGE, "check returned an error"
    details = getattr(result, "details", None)
    if isinstance(details, Mapping):
        exit_code = details.get("exit_code")
        if (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code != 0
            and str(details.get("stderr_head") or "").strip()
        ):
            return ENGAGE, "check failed"
        if details.get("backgrounded") or details.get("no_output") is True:
            return ENGAGE, "no comparable output"
    comparable, blind, handle = _monitor_comparable_text(getattr(result, "text", "") or "", details)
    if blind is not None or not str(comparable or "").strip():
        return ENGAGE, "no comparable output"
    new_hash = content_hash(normalize_result(str(comparable), spec, settings))
    if new_hash != str(counters.get("content_hash") or ""):
        return ENGAGE, "changed"
    # THE QUIET TICK, as ``MonitorScheduler._apply_success`` commits it.
    counters["checks"] = int(counters.get("checks") or 0) + 1
    counters["last_check_at"] = now
    counters["consecutive_failures"] = 0
    counters["last_error"] = ""
    if handle:
        counters["last_handle"] = handle
    counters["next_due_at"] = advance_due(spec, now)
    monitor_state.write_counters(config_dir, session_id, spec.id, counters)
    return QUIET, "unchanged"


async def probe_session(
    config_dir: Path, session_id: str, *, now_ms: int | None = None
) -> dict[str, Any]:
    """Check every DUE monitor of a parked session; ``verdict`` quiet/engage.

    ``engage`` as soon as one row needs the session (the rest stay due and the
    engaged session checks them itself). Checks run sequentially: the
    in-session ceiling is 2 concurrent checks per session, and the supervisor
    already bounds how many sessions it probes at once.
    """
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors import store as monitor_store
    from local_operator.monitors.settings import read_monitor_settings

    now = int(time.time() * 1000) if now_ms is None else now_ms
    entry = monitor_store.read_entry(config_dir, session_id)
    if not is_parked(entry):
        return {"verdict": ENGAGE, "why": "not parked", "checked": 0}
    assert entry is not None
    settings = read_monitor_settings()
    checked = 0
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        monitor_id = str(row.get("id") or "")
        counters = monitor_state.read_counters(config_dir, session_id, monitor_id) or {}
        due = counters.get("next_due_at", row.get("next_due_at"))
        if counters.get("disabled") or row.get("disabled"):
            continue
        until = row.get("until_at")
        if isinstance(until, int) and now >= until:
            continue
        if not isinstance(due, int) or isinstance(due, bool) or due > now:
            continue
        verdict, why = await _check_one(config_dir, session_id, row, settings, now)
        checked += 1
        if verdict == ENGAGE:
            return {"verdict": ENGAGE, "why": f"{monitor_id}: {why}", "checked": checked}
    return {"verdict": QUIET, "why": "unchanged", "checked": checked}


def main(argv: list[str] | None = None) -> int:
    """``python -m local_operator.monitors.parked --session <id>``: one JSON line."""
    parser = argparse.ArgumentParser(prog="parked-monitor-probe")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config-dir", default="")
    args = parser.parse_args(argv)
    from local_operator.paths import config_dir as ambient_config_dir

    root = Path(args.config_dir) if args.config_dir else ambient_config_dir()
    try:
        outcome = asyncio.run(probe_session(root, args.session))
    except Exception as exc:  # noqa: BLE001 — any failure means "let the session do it"
        outcome = {"verdict": ENGAGE, "why": f"probe raised {type(exc).__name__}", "checked": 0}
    print(json.dumps(outcome))
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    raise SystemExit(main())
