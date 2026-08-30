"""Open a NEW terminal window (or cmux workspace) running a forked session.

WHY THIS EXISTS
---------------
``/fork`` branches the current conversation into a new session, and the whole
value of it is that the ORIGINAL keeps running: the point is to try a direction
without leaving the conversation that got you there. So the fork has to appear
somewhere else — a new window, a new cmux workspace — while this TUI carries on
untouched. That "somewhere else" is per terminal emulator, so each one is a
:class:`~local_operator.spawn.types.SpawnBackend` in a registry
(:mod:`.registry`) rather than a branch in a chain of ifs.

WHY NOT ``local_operator/multiplexer``
--------------------------------------
The two registries answer different questions about different populations.
``MultiplexerBackend`` asks "what MULTIPLEXER hosts this pane" and its members
are tmux/zellij/screen/wezterm/cmux; this one asks "what can open a WINDOW
here" and its members are terminal emulators plus cmux. The overlap is cmux and
WezTerm only, and the roles have different lifecycles — publish/retire against a
long-lived pane versus a one-shot launch. Bolting a ``spawn()`` method onto the
existing Protocol would force ``TmuxBackend`` and ``ScreenBackend`` to implement
window-opening that makes no sense for them.

The cmux CLIENT is not duplicated, though: :mod:`.cmux` imports
``multiplexer.cmux``'s binary resolution, RPC helper and surface target so the
PATH-vs-``CMUX_BUNDLED_CLI_PATH`` rule keeps exactly one definition.

SPAWNING IS BEST-EFFORT; THE FORK IS NOT
----------------------------------------
The invariant that makes every degradation recoverable: **the fork session is
created and durable on disk BEFORE any spawn is attempted.** Nothing in this
package may raise, and a backend that cannot open a window returns ``False`` so
the caller prints a receipt naming the fork id and the exact ``lop --resume
<id>`` command. A failed spawn costs a window, never a conversation — and "no
backend detected" (a bare tty, plain SSH with no window server, an unrecognised
emulator) is a NORMAL outcome carrying a ``note``, not an error.

RESTORE-AND-IDLE, INHERITED
---------------------------
Every backend builds its command from ``multiplexer.broadcast.resume_argv``,
which is documented there as a safety boundary: the argv replays a transcript
and then waits for the user, carrying no prompt and nothing that continues an
interrupted turn. A fork's opening message rides a sidecar in the fork's own
session directory (``fork.BOOT_PROMPT_NAME``) and never the command line, so
that invariant is inherited rather than bent — and so the message does not
appear in ``ps`` or need correct quoting on five backends.

FOCUS DISCIPLINE
----------------
Every cmux invocation passes ``--focus false`` explicitly. cmux's socket gate
only allows focus, window raise and workspace switching when a command carries
an explicitly truthy ``focus``, so this is what keeps a fork from yanking the
user's window while they are working in another one. The emulator backends open
real OS windows, which take focus on macOS; that is the platform's behaviour and
is one more reason cmux placement is the better experience where cmux exists.
"""

from local_operator.spawn.registry import active_backend, backends
from local_operator.spawn.types import ForkLaunch, SpawnBackend

__all__ = ["ForkLaunch", "SpawnBackend", "active_backend", "backends"]
