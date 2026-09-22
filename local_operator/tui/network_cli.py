"""Run ``lop network …`` for a TUI surface, bounded, and hand back what it said.

WHY A SUBPROCESS AND NOT AN IMPORT. ``/network`` is a front end to the CLI family
the agent guide drives (``docs/design/mesh-ui.md`` §1.1.3): the guards live in
``network/cli.py`` — who may be revoked, what a panic rotates, which audit record
is written once — and a TUI that re-derived them would be a second answer to
"what does disconnect do". Calling the handlers in-process would need this module
to rebuild the argparse Namespace the parser owns, i.e. a second parser; the
process boundary is the cheaper way to reuse the ONE parser, and it is also what
keeps a dial (a listing asks every member, budget 12 s) off the event loop.

THE CHILD IS BOUNDED AND REAPED BY ITS OWN GROUP. A ``join`` that wants a human,
or a listing waiting on a peer that never answers, must not leave a process
behind: ``start_new_session`` gives the child its own process group, and the
timeout kills the GROUP rather than the pid, so a grandchild (the relay client's
own helper) cannot outlive the call. ``stdin`` is /dev/null for the same reason —
a prompt nothing can answer must fail, not hang.

``--json`` IS FOR DATA, the human lines are for RECEIPTS. A receipt is the CLI's
own sentences (one implementation of the copy), while the panel parses a payload.
Asking for both would print JSON into a pipe that read lines, so each call picks
one and :func:`run_network` returns whichever channel it asked for.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from typing import Any

from local_operator.interpreter import python_argv

#: How long a listing may take before this surface gives up on it. The CLI's own
#: client budget for a listing is the relay's probe budget plus its slack
#: (``relay.LISTING_CLIENT_TIMEOUT_S`` = 12 s + 8 s, the one home for that
#: number); a TUI that cut that short would report a timeout about a command that
#: was still working, so the bound sits above it.
LISTING_TIMEOUT_S = 30.0

#: A dial-free verb — the local store, the identity file, an audit tail. Short
#: enough that a wedged relay is noticed at the composer, long enough for a cold
#: interpreter start (the child imports this project).
QUICK_TIMEOUT_S = 20.0

#: Verbs that ask a PEER and wait: a spawn plus its first turn's admission, and
#: the stop ladder's SIGTERM rung, which waits out a drain only the owning
#: machine can bound (``network/cli.py`` passes 120 s and 240 s respectively).
PEER_CALL_TIMEOUT_S = 260.0

#: The refusal sentences ``network/cli.py`` prints are ANSI-coloured for a human
#: watching a terminal. A notice is the wrong place for an escape sequence — a
#: captured string must be the sentence, not the sentence plus paint.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class NetworkRun:
    """One finished ``lop network`` call: its exit code and its two streams."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def lines(self) -> list[str]:
        """The receipt: stdout's non-blank lines, then stderr's. Never both empty.

        Stdout first because that is where ``_emit`` writes a receipt and where a
        refusal's ``--json`` body lands; stderr carries the coloured sentence for
        a human. A caller that got nothing on either stream says so itself —
        this returns ``[]`` rather than inventing a sentence about a silence it
        cannot explain.
        """
        out = [line for line in self.stdout.splitlines() if line.strip()]
        err = [line for line in self.stderr.splitlines() if line.strip()]
        return out + err

    def payload(self) -> dict[str, Any] | None:
        """The parsed ``--json`` body, or ``None`` when the call did not print one.

        ``None`` is not an error: it is the honest answer for a refused call whose
        body went to stderr, or for a subcommand that printed lines because a
        caller asked for lines. Callers that need a payload check for it.
        """
        text = self.stdout.strip()
        if not text.startswith("{"):
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None


def run_network(
    args: list[str],
    *,
    timeout: float = QUICK_TIMEOUT_S,
    json_output: bool = False,
) -> NetworkRun:
    """Run ``lop network <args>`` and return what it said. Blocking by design.

    Callers run this off the event loop (``asyncio.to_thread``): the listing verbs
    dial peers, and §1.1.3 requires all network work to stay outside the TUI loop.
    ``json_output`` appends ``--json`` so the payload is parseable; the flag is
    added here rather than at each call site so no caller can forget it and then
    wonder why :meth:`NetworkRun.payload` is empty.
    """
    argv = python_argv("-m", "local_operator.cli", "network", *args)
    if json_output:
        argv.append("--json")
    env = dict(os.environ)
    # A CHILD OF THE TUI IS NOT A TERMINAL. Inheriting the parent's stdout pipe
    # would make ``invite --print``'s TTY check answer for a surface nobody is
    # at — and the CLI refuses to print a token into a pipe on purpose (its own
    # docstring), which is exactly the behaviour we want to keep.
    env.pop("FORCE_COLOR", None)
    try:
        child = subprocess.Popen(  # noqa: S603 — argv is built here, never a shell
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # Its own group, so the timeout can reap the whole tree: a relay
            # client that forked a helper must not survive the surface that
            # started it.
            start_new_session=True,
        )
    except OSError as exc:
        return NetworkRun(tuple(argv), 127, stderr=f"could not start the CLI: {exc}")
    try:
        out, err = child.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _reap_group(child)
        out, err = child.communicate()
    return NetworkRun(
        tuple(argv),
        child.returncode if child.returncode is not None else -1,
        _ANSI_RE.sub("", out or ""),
        _ANSI_RE.sub("", err or ""),
        timed_out,
    )


def _reap_group(child: subprocess.Popen[str]) -> None:
    """Kill the child's whole process group, then let the caller drain it.

    By GROUP and by pid of the group we created, never by program name: this
    fleet runs ~25 concurrent agent sessions, and an unscoped kill has already
    taken out two other sessions' process trees. ``getpgid`` can race with the
    child exiting on its own, hence the guard — the fallback kill is the child
    itself, which is still by pid and still ours.
    """
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            child.kill()
        except OSError:
            pass
