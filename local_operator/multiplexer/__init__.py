"""Tell the terminal multiplexer which session this pane is holding.

WHY THIS EXISTS
---------------
A user runs a column of ``lop`` sessions as multiplexer panes (cmux
workspaces down a sidebar, tmux windows, zellij tabs). When the multiplexer
dies and comes back, the panes come back too — but each one opens a FRESH
shell and a FRESH ``lop``. Every conversation is still on disk under
``~/.local-operator/sessions``, and none of them is reachable except by hand:
the user has to remember which pane held which session and retype
``lop --resume <id>`` fifteen times. That is the failure this package fixes.

The fix is to publish, per pane, the one fact the multiplexer cannot derive
on its own: *this pane is holding session ``<id>``, and here is the argv that
reopens it*. What "publish" means is per-multiplexer, so each one is a
:class:`~local_operator.multiplexer.types.MultiplexerBackend` in a registry
(:mod:`.registry`) rather than a branch in a chain of ifs — a new
multiplexer is a new module, not an edit to this one.

RESTORE-AND-IDLE, NEVER AUTO-CONTINUE
-------------------------------------
The published command is always ``lop --resume <id>`` and never anything that
carries a prompt or continues a turn. This is the single most important
constraint here and it is a safety property, not a preference: a restore
happens when fifteen panes come back at once, unattended, typically after a
crash the user did not choose. ``--resume`` replays the transcript and waits
for the user, so the worst case of a spurious restore is fifteen idle
sessions. An "auto-continue the interrupted turn" binding would instead mean
fifteen agents resuming TOOL EXECUTION simultaneously with nobody watching.
:mod:`.broadcast` builds the argv so no call site can get this wrong.

BEST-EFFORT, ALWAYS
-------------------
Nothing in this package may prevent a session from starting, slow the TUI, or
surface an error. A missing binary, a socket that is mid-restart, a
multiplexer that is not running, a non-zero exit, a timeout: all are logged at
debug and otherwise ignored. Publication is bookkeeping ABOUT a session, and
taking a session down to record where it lives would be the more expensive
bug by far. Every subprocess is spawned detached with a short timeout and is
never waited on from the event loop.

NEVER FOR A SUBAGENT
--------------------
Only a user's own session is published. A subagent's child session is an
ephemeral directory with exactly the shape of a real conversation, and it runs
inside the SAME pane as its parent — so publishing one would overwrite the
pane's binding and the crash restore would reopen a delegated code review
instead of the user's work. :func:`~local_operator.resume.is_user_session`
(the ``origin.json`` marker) is the gate, the same one the ``/resume`` picker
uses. cmux's own omp integration carries an identical guard
(``isNestedArtifactSession``) for the same reason.

THE KILL SWITCH
---------------
``LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME`` disables all of it, mirroring the
``LOCAL_OPERATOR_NO_TERMINAL_TITLE`` convention in
:mod:`local_operator.tui.terminal_title`. Wanted by anyone who does not want a
pane's resume binding rewritten — a recording, a CI job, a session opened to
read someone else's transcript.

WHAT EACH BACKEND PUBLISHES
---------------------------
cmux has a real resume-binding API and gets a trusted auto-resume binding; see
:mod:`.cmux` for why that path is the socket RPC and not the ``cmux surface
resume set`` CLI. The others have no such API, so they publish a discoverable
per-pane marker instead — a tmux/wezterm pane option, or a state file under
``~/.local-operator/multiplexer/`` keyed by pane identity. The marker contract
is documented in :mod:`.markers` so a human or a shell script can implement
the restore side without reading this code.
"""

from __future__ import annotations

from local_operator.multiplexer.broadcast import (
    SessionBroadcast,
    broadcast_session,
    multiplexer_resume_enabled,
    retire_session,
)
from local_operator.multiplexer.registry import active_backend, backends
from local_operator.multiplexer.types import MultiplexerBackend, SessionBinding

__all__ = [
    "MultiplexerBackend",
    "SessionBinding",
    "SessionBroadcast",
    "active_backend",
    "backends",
    "broadcast_session",
    "multiplexer_resume_enabled",
    "retire_session",
]
