"""Which in-process session spawned a child process: ``pid -> session id``.

WHY (review m1 on PR #3987). Task subagents run IN-PROCESS (a child
``Session`` in the same harness process), and the eval tool keeps one kernel
per session in that same process. Every bash child and eval kernel of every
in-process session therefore has the harness pid as its parent, so parent-pid
ancestry alone (``provenance.cwd_holders``) cannot tell the deleting
session's own shell from a SIBLING subagent's: both read as family, and a
sibling's live cwd inside the deleting session's worktree did not keep it
busy.

This map records, at the two spawn sites that run agent work (the bash tool
and the eval kernel), which session started each direct child of the harness.
``cwd_holders`` consults it for the hop just below the harness pid and
attributes that process to the recorded session instead of to "this
process". Runtime records only: a process's environment or command line is
never read for attribution (either may be set by the agent).

CONSTRAINTS
- Pid reuse: an entry is believed only while the asyncio process object it
  was recorded with has not been reaped (``returncode is None``); a reaped
  child's pid may be reissued to an unrelated process, so a finished entry
  is ignored and pruned. Strong references are held only until reaping is
  observed, so the map stays bounded by live children.
- Process-local and in-memory: it describes children of THIS process only,
  which is exactly the set that shares the harness pid. Other harness
  processes have their own registry records and are attributed by those.
- An unmapped direct child of the harness (an MCP server, a language server)
  keeps the previous attribution — the harness's own — because no session
  spawned it as agent work.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
#: pid -> (session id, process object whose ``returncode`` proves liveness).
_spawned: dict[int, tuple[str, Any]] = {}


def record(pid: int | None, session_id: object, process: Any) -> None:
    """Note that ``session_id`` spawned ``pid`` (``process`` exposes ``returncode``)."""
    if not isinstance(pid, int) or not isinstance(session_id, str) or not session_id:
        return
    with _lock:
        # Prune reaped children here, so the map never outgrows live work.
        for stale in [p for p, (_s, proc) in _spawned.items() if _reaped(proc)]:
            del _spawned[stale]
        _spawned[pid] = (session_id, process)


def session_of(pid: int) -> str | None:
    """The session that spawned live child ``pid``, or ``None`` if unrecorded/reaped."""
    with _lock:
        entry = _spawned.get(pid)
        if entry is None:
            return None
        if _reaped(entry[1]):
            del _spawned[pid]
            return None
        return entry[0]


def _reaped(process: Any) -> bool:
    return getattr(process, "returncode", None) is not None
