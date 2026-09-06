"""Tell the hosting Herdr pane what this session is doing.

WHY THIS EXISTS
---------------
`Herdr <https://herdr.dev>`_ is a terminal multiplexer whose differentiator
is a live **Agents** panel: one row per pane, each saying ``idle`` /
``working`` / ``blocked``. Left to itself Herdr fills that row by screen
detection — reading the pane's bottom buffer and guessing — which cannot tell
"the model is thinking" from "a tool approval is waiting on you". Local
Operator already knows that distinction exactly; it is the same three-valued
state the terminal title carries (``lo ›`` / ``lo ⣻`` / ``lo !``). This
package pushes that state through Herdr's documented custom-agent surface so
the panel shows what the session is actually doing.

No upstream Herdr change, no ``herdr integration install``, no plugin: the
CLI's ``pane report-agent`` / ``pane release-agent`` are documented "for
custom hooks", and a custom ``--source`` is first-class in ``agent list``.

ONE SIGNAL, ONE HOOK
--------------------
The reporter is driven from the SAME place the terminal title is —
``StatusLine._sync_terminal_title`` — rather than from new hooks scattered
through the turn loop. The title already coalesces ``streaming`` and
``attention`` into one external state; a second derivation would eventually
disagree with it, and the failure mode of two derivations is exactly a
sidebar row saying ``working`` while the tab title says idle.

BEST-EFFORT, ALWAYS
-------------------
Identical to :mod:`local_operator.multiplexer`'s contract. Nothing here may
prevent a session from starting, slow the TUI, or surface an error. A missing
binary, a socket mid-restart, a non-zero exit, a timeout: logged at debug and
otherwise ignored. Every CLI call runs on a worker thread with a short timeout
and is never awaited on the event loop, including the release at exit.

NEVER FOR A SUBAGENT
--------------------
A delegated child session runs inside its parent's pane. Reporting it would
overwrite the pane's row with the child's state — ``idle`` while the user's
own turn is parked on an approval. The app applies the same
``is_user_owned_session`` gate the multiplexer uses.

THE KILL SWITCH
---------------
``LOCAL_OPERATOR_NO_HERDR`` disables all of it, mirroring
``LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME`` and ``LOCAL_OPERATOR_NO_TERMINAL_TITLE``.
An environment gate only, with no ``/settings`` key: like the multiplexer
switch this writes nothing a user sees inside the app, so the reason to turn
it off is situational (this recording, this CI job) rather than a standing
preference — which is the line ``terminal_title`` draws between its env
switch and its config flag, and this integration has only the first half.
"""

from local_operator.herdr.reporter import (
    HERDR_AGENT,
    HERDR_SOURCE,
    HerdrReporter,
    HerdrState,
    herdr_binary,
    herdr_reporting_enabled,
    release_reporter,
    start_reporter,
)

__all__ = [
    "HERDR_AGENT",
    "HERDR_SOURCE",
    "HerdrReporter",
    "HerdrState",
    "herdr_binary",
    "herdr_reporting_enabled",
    "release_reporter",
    "start_reporter",
]
