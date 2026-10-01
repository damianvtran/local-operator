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
from typing import Any, Iterable

from local_operator.interpreter import python_argv
from local_operator.network.cli import PILOT_ACT_TIMEOUT_S

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

#: Verbs that DRIVE a session on a peer — ``/network sessions --send`` and its
#: two siblings. They are not one round trip: the CLI opens a viewer on the peer
#: and binds it, then waits for the act's own reply, and the whole act is bounded
#: by ``network/cli.PILOT_ACT_TIMEOUT_S``. DERIVED FROM THAT CONSTANT, not
#: restated: this number was written by hand once, as 480, under a comment that
#: said it sat above what the CLI waits — and it did not (2 x 120 s of dial plus
#: 300 s of turn is 540 s), so a `/network sessions --send …` typed in a session
#: reaped its child mid-act and the composer showed a timeout about a command
#: that was still working (review round 1, MAJOR-3). Importing the term is what
#: makes that impossible to repeat when one of the three budgets moves.
#:
#: The slack on top is for what the constant does not cover: this child is a
#: fresh interpreter that imports the whole application before its first frame,
#: and it tears a viewer down on the way out.
PILOT_CALL_TIMEOUT_S = PILOT_ACT_TIMEOUT_S + 60.0


def gesture_call_timeout() -> float:
    """The budget for a verb that WAITS ON A HUMAN GESTURE — ``approvals approve``.

    The store's signing call (``approval_store.sign_decision``, which IS the
    ``lop operator sign`` path) raises the OS key agent's sheet and blocks for
    as long as it is up; ``sign_message`` runs it with no timeout of its own, so
    the bound the gesture carries is the key agent's
    (``keyagent.SIGN_TIMEOUT_SECONDS``). DERIVED from it rather than restated,
    for the reason ``PILOT_CALL_TIMEOUT_S`` above is derived: a budget written
    by hand under the gesture's own bound would reap this child mid-prompt and
    report a timeout about a question the operator was still answering.

    The slack on top covers what that constant does not: a fresh interpreter
    importing the application before the sheet appears, and the store's
    cross-process lock around the decision write.

    The import is FUNCTION-LOCAL on purpose: this module is imported at
    ``app.py``'s module scope and pulls in only the stdlib, so the operator
    package stays off that path until a verb here actually raises a gesture.
    """
    from local_operator.operator.macos.keyagent import SIGN_TIMEOUT_SECONDS

    return SIGN_TIMEOUT_SECONDS + 60.0


#: The refusal sentences ``network/cli.py`` prints are ANSI-coloured for a human
#: watching a terminal. A notice is the wrong place for an escape sequence — a
#: captured string must be the sentence, not the sentence plus paint.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: The label ``--create`` puts in front of the id it minted. The line is
#: ``network/cli.py``'s own receipt (its first line is ``session: <id>``) and
#: :func:`created_session_id` is the ONE reader of it.
_CREATE_RECEIPT_ID_LABEL = "session"

#: The family's two spellings: the one a shell types, and the one this front end's
#: picker offers. Read by :func:`tui_spelling` only.
_CLI_SPELLING = "lop network "
_TUI_SPELLING = "/network "

#: The one carried verb the composer does NOT execute, and therefore the one
#: :func:`tui_spelling` leaves alone. `join` is in ``NETWORK_SUBCOMMANDS`` because
#: the picker offers it, but its answer in this front end is
#: ``_network_join_notice``'s sentence — pairing shows a code on each device for a
#: person to read across, so it needs a terminal, and the receipt that names it
#: (``invite``'s ``then, on the other device: …``) is addressed to the OTHER
#: machine, where a shell is what the reader has. Rewriting it here would hand the
#: reader a word this front end then refuses.
_NOT_TRANSLATED = frozenset({"join"})

#: The ONE verb this family spells differently on each side: the shell's `init`
#: and the composer's `new`, which is the only spelling ``NETWORK_SUBCOMMANDS``
#: carries. A line naming the shell's word at a composer hands the reader a
#: subcommand the caret's own table refuses — the D2 defect one token down from
#: the verb :func:`tui_spelling` was written for — and the reader meets it in a
#: REFUSAL as well as in a receipt, which is where a remedy is named (`lop
#: network init <name>` is what the empty-network refusal now tells a shell
#: reader to run). Read by :func:`tui_spelling` only.
_CLI_VERB_ALIASES = {"init": "new"}


def tui_spelling(line: str) -> str:
    """Rewrite a receipt's ``lop network <verb>`` into the composer's spelling.

    A RECEIPT IS THE CLI'S OWN SENTENCE, and that is right in the front ends that
    ARE a shell — but it is the wrong dialect at a composer. Measured on the real
    path: ``/network new devmesh`` prints ``next: lop network invite --role
    drive``, so the surface that had just accepted ``/network new`` handed its
    reader a command belonging to another surface's vocabulary (design round 1,
    D2). The panel's empty state and the README both say ``/network invite``, and
    this is what makes the three agree.

    ONLY VERBS THIS FRONT END CARRIES AND RUNS ARE TRANSLATED, which is why this
    consults a vocabulary instead of replacing a string. The same receipts also
    name ``lop network start``, ``serve`` and ``install``, and
    ``NETWORK_SUBCOMMANDS`` deliberately withholds those — a composer row that
    boots out the operator's relay, or deletes this device's identity, is the
    one-keystroke mistake the family's typed confirmations exist to prevent — so
    rewriting them would advertise a word this front end refuses. ``join`` is the
    subtler half of the same rule and has its own comment above. An unrecognised
    verb, and any line naming no verb at all, is returned exactly as it came.

    THIS RUNS OVER REFUSALS TOO, not only receipts. A refusal is where this family
    names the remedy — the empty-network one now ends with the command that
    creates the first network, and the join failures have ended that way for
    rounds — and a remedy is exactly the thing the reader is meant to RUN, so the
    spelling that reaches them has to be the one their caret accepts. The rule is
    unchanged by that: only a carried verb is rewritten, so a refusal naming
    ``lop network start`` still reads as the shell's.

    THE VERB IS PART OF THE SPELLING, which is what :data:`_CLI_VERB_ALIASES`
    carries: `init` becomes `new` before the vocabulary is consulted, so the two
    rewrites cannot disagree about which front end a line is being read in.
    """
    from local_operator.slash_commands import NETWORK_SUBCOMMANDS

    at = line.find(_CLI_SPELLING)
    if at == -1:
        return line
    rest = line[at + len(_CLI_SPELLING) :]
    verb = rest.split(" ", 1)[0].split("\n", 1)[0].strip()
    alias = _CLI_VERB_ALIASES.get(verb)
    if alias is not None:
        rest = alias + rest[len(verb) :]
        verb = alias
    if verb not in NETWORK_SUBCOMMANDS or verb in _NOT_TRANSLATED:
        return line
    return line[:at] + _TUI_SPELLING + rest


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


def created_session_id(lines: Iterable[str]) -> str:
    """The id a ``--create`` receipt names, or ``""`` when it names none.

    WHY A READER AND NOT A SECOND CALL. ``/new remote <peer>`` has to OPEN the
    session it just created on the peer, and the create's receipt is the only
    channel that carries the id back to this surface. Asking the CLI a second
    time in ``--json`` for the same fact would either print a payload into the
    transcript or make this front end re-render a receipt whose sentences are
    ``network/cli.py``'s — which is the reason the verb is a subprocess here at
    all (module docstring). So the line the CLI already prints is read.

    THE LABEL IS THE WRITER'S, not a shape guessed at this end, and a test drives
    the real CLI's create branch and asserts this reader recovers the id it
    minted: a reworded receipt fails that test instead of silently leaving every
    ``/new remote`` on the session it stood in before.

    The value is checked against ``session_directory_name`` — the store's own id
    admission — so a receipt line that carried something else cannot send a caller
    looking for a session that cannot exist.
    """
    from local_operator.session.catalog import session_directory_name

    for line in lines:
        label, separator, value = line.partition(":")
        if not separator or label.strip().casefold() != _CREATE_RECEIPT_ID_LABEL:
            continue
        candidate = value.strip()
        return candidate if session_directory_name(candidate) else ""
    return ""


def _argv_for(args: list[str], *, json_output: bool) -> list[str]:
    """The child's argv, with ``--json`` placed where nothing can read it as text.

    THE PLACEMENT IS THE POINT, not the flag. ``network sessions --send`` and its
    two siblings take their payload as an argparse REMAINDER — everything from the
    session id to the end of the command line is text — so a flag appended at the
    END of a tail that carries an act is delivered as part of the prompt and the
    payload channel comes back empty. Directly after the subcommand is where every
    ``network`` subcommand declares ``--json`` and where the parser still sees it
    as an option.

    Suffix-independent by construction: this inserts into the NETWORK arguments
    before they are prefixed, so it cannot depend on ``python_argv``'s shape.
    """
    network_args = list(args)
    if json_output:
        network_args.insert(1 if network_args else 0, "--json")
    return python_argv("-m", "local_operator.cli", "network", *network_args)


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

    WHERE that flag lands is load-bearing and is why this goes through
    :func:`_argv_for`: the pilot verbs take their text as an argparse REMAINDER,
    so every token after the session id is PAYLOAD — an appended flag would be
    delivered as part of the prompt (`--send <s> hi --json` would send the words
    "hi --json" and leave the payload channel empty).
    """
    argv = _argv_for(args, json_output=json_output)
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
