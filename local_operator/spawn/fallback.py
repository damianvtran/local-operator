"""No window can be opened here, and that is a NORMAL outcome.

This is not a backend the registry can select — it has no ``spawn`` — it is the
receipt the caller prints when nothing detected, plus the one piece of context
that makes the message read as competent rather than broken: naming SSH when
that is why there is no window server.

THE FORK IS NOT LOST ON THIS PATH. The session was created and is durable before
any spawn was attempted (see the package docstring), it is listed in ``/resume``
like any other conversation, and the receipt carries the exact command to reach
it. That is why the notice weight is ``note`` and not ``warning``: nothing went
wrong, this terminal simply cannot open windows.
"""

from __future__ import annotations

from local_operator import terminals
from local_operator.spawn.types import EnvMap


def fallback_receipt(session_id: str, env: EnvMap | None = None, *, failed: bool = False) -> str:
    """What to tell the user when the fork exists but no window opened.

    ``failed`` distinguishes the two ways to get here, and the distinction is
    worth a different sentence: a backend that DETECTED and then failed (a cmux
    socket mid-restart, a missing emulator binary) means "the window did not
    open", while no backend at all means "this terminal cannot open one". Both
    end with the same actionable command, which is the part that matters.
    """
    if failed:
        reason = "the new window could not be opened"
    elif terminals.is_ssh(env):
        # Worth naming: the user knows why, and a message that says it reads as
        # the tool understanding the situation rather than failing at it.
        reason = "no window server over ssh"
    else:
        reason = "no window could be opened here"
    return f"forked to {session_id} — {reason}; run `lop --resume {session_id}`"
