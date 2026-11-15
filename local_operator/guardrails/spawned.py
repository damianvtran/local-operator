"""Which in-process session spawned a child process: ``pid -> session id``.

WHY (review m1 on PR #3987). Task subagents run IN-PROCESS (a child
``Session`` in the same harness process), and the eval tool keeps one kernel
per session in that same process. Every bash child and eval kernel of every
in-process session therefore has the harness pid as its parent, so parent-pid
ancestry alone (``provenance.cwd_holders``) cannot tell the deleting
session's own shell from a SIBLING subagent's: both read as family, and a
sibling's live cwd inside the deleting session's worktree did not keep it
busy.

This map records, at the spawn sites that run agent work (the bash tool, the
eval kernel, the grep tool's ``rg`` and the sessions tool's launcher), which
session started each direct child of the harness.
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
  spawned it as agent work. EXCEPT while an agent spawn is in flight
  (:func:`spawning`, QA Q3 on PR #3987): between the fork inside
  ``create_subprocess_exec`` and :func:`record`, the new child is unmapped
  and would read as the harness's (family) for a few event-loop ticks, so a
  sibling subagent's fresh shell inside the deleting session's tree did not
  hold it (6/6 trials in QA's rig). While :func:`pending` is non-zero an
  unmapped direct child counts as foreign instead — fail closed for that
  window; an MCP server inside a tree reads foreign for those ticks too.
- Reap-to-``returncode`` (review n1, PR #3987): on macOS the threaded child
  watcher reaps in a thread and sets ``returncode`` later via
  ``call_soon_threadsafe``, so a pid could be reissued before the entry
  reads reaped. Not exploitable: the reused pid would also have to be a
  direct child of the harness (only harness code spawns those), and a stale
  sibling entry errs toward foreign.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

_lock = threading.Lock()
#: pid -> (session id, process object whose ``returncode`` proves liveness).
_spawned: dict[int, tuple[str, Any]] = {}
#: Agent spawns between "about to fork" and :func:`record` (see CONSTRAINTS).
_pending = 0


@contextlib.contextmanager
def spawning() -> Iterator[None]:
    """Mark an agent spawn in flight; wrap ``create_subprocess_exec`` AND ``record``.

    Exited on every path (a failed spawn included), so the counter never
    sticks: a stuck counter would only fail closed, but would do so forever.
    """
    global _pending
    with _lock:
        _pending += 1
    try:
        yield
    finally:
        with _lock:
            _pending -= 1


def pending() -> int:
    """How many agent spawns are between fork and :func:`record` right now."""
    with _lock:
        return _pending


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
