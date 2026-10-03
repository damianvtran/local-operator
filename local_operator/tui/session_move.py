"""``/move … --to``: the grammar, and the one call that runs a mesh move for the TUI.

WHY THIS IS NOT IN ``app.py``. Two pure pieces the app's handler needs and a test
must be able to reach without booting a Textual app: the parse (which is where
the "path or session?" ambiguity is decided, so it is the part a regression hides
in) and the subprocess call (bounded and reaped by its own group, the same
discipline as ``tui/network_cli.py``).

THE CLI OWNS THE PROTOCOL. ``lop sessions move <id> --to <peer|local> [--keep]
[--queue] --json`` is slice M's verb and prints the frozen ``session_move``
contract (``network/mobility.py``: ``SessionMoveResult`` / ``SessionMoveRefusal``).
This module runs it and parses that JSON — it re-implements no step of the move.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from local_operator.interpreter import python_argv
from local_operator.tui.network_cli import _ANSI_RE, _reap_group

#: The phases a move reports, in the contract's order (``MOVE_RESULT_PHASES``).
#: Restated rather than imported so the TUI's import of this module does not pull
#: the relay package; ``tests/unit/tui/test_session_move.py`` pins that the two
#: agree.
MOVE_PHASE_ORDER: tuple[str, ...] = ("prepared", "handing_off", "committed", "done")

#: The whole move, from this surface's side: the CLI's own relay budget is
#: ``MOVE_OP_DEADLINE_S + OFFLOAD_CONFIRM_WAIT_S`` plus slack (~130 s) and a
#: ``--keep`` copy may wait ``KEEP_COPY_WAIT_S`` (300 s). The bound sits above the
#: larger so a working move is never reported as a timeout.
MOVE_TIMEOUT_S = 360.0

#: The sentence for the one ambiguous shape (``mesh-ui.md`` §1.6, "move ambiguous").
AMBIGUOUS_MOVE = (
    "Use /move <path> for a working directory, or /move <session> --to <peer> to move " "a session."
)


@dataclass(frozen=True)
class MoveTo:
    """One parsed ``/move … --to`` request. ``error`` set means refuse with it."""

    session_id: str = ""
    to: str = ""
    keep: bool = False
    #: The queued form (design §5.4): the source records the intent and runs the
    #: move at its next safe point instead of refusing an attached-or-busy
    #: conversation. Ignored with ``keep`` by the CLI (a copy retires nothing,
    #: so it has no safe point to wait for), which is the CLI's contract to own.
    queue: bool = False
    error: str = ""


def parse_move_to(arg: str) -> MoveTo | None:
    """The mobility form of ``/move``, or ``None`` for today's path form.

    THE DISCRIMINANT IS THE PRESENCE OF ``--to`` (``mesh-ui.md`` §1.7, rule 1),
    never the shape of the first token: an id can look like a directory name and
    a directory name like an id. Without ``--to`` this returns ``None`` and the
    caller's path handling runs unchanged, byte for byte.

    Accepted: ``--to X``, ``--to=X``, an optional leading session id, ``--keep``
    and ``--queue`` anywhere. Refused (a ``MoveTo`` with ``error``): ``--to`` with
    no value, more than one leading word (a path with spaces, or a path plus an
    id — the ambiguous shape), any other flag — and ``--wait`` by name, because
    the producer's busy sentence sends readers to a shell for that flag
    (``mobility._busy_sentence``) and a bare "unknown flag" here would be the same
    dead end one surface over.

    ``--queue`` IS THE SPELLING THE PRODUCER'S VIEWED SENTENCE ALREADY POINTS AT
    (§5.4): the runtime's refusal says "queue the move to run at the next safe
    point", and before this flag the composer could not run the remedy it had
    just printed — the same papercut class the ``--wait`` refusal exists to
    avoid, one surface over.
    """
    try:
        tokens = shlex.split(arg)
    except ValueError:
        tokens = arg.split()
    if not any(token == "--to" or token.startswith("--to=") for token in tokens):
        return None
    to = ""
    keep = False
    # Defaulted here for EVERY form, not only where ``--queue`` is parsed: the
    # request build at the bottom reads this name, so a form without the flag must
    # still answer it — without the default, the bare ``--to`` forms raise
    # ``UnboundLocalError`` (the red shard that added this line).
    queue = False
    words: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--to":
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                return MoveTo(error="--to needs a device: /move --to <peer|local>")
            to = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--to="):
            to = token[len("--to=") :]
            if not to:
                return MoveTo(error="--to needs a device: /move --to <peer|local>")
        elif token == "--keep":
            keep = True
        elif token == "--queue":
            queue = True
        elif token.startswith("-"):
            if token == "--wait" or token.startswith("--wait="):
                # THE OTHER HALF OF THE BUSY PAPERCUT (``mobility._busy_sentence``):
                # the producer's refusal tells the reader to wait from a shell — this
                # view runs the CLI as one blocked subprocess with no phase feedback
                # until it answers, so it cannot wait — and a reader who types the
                # flag here anyway gets the shell route rather than a dead end about
                # an unknown option.
                return MoveTo(
                    error=(
                        "this view cannot wait — from a shell: "
                        "`lop sessions move <id> --to <to> --wait <seconds>`"
                    )
                )
            return MoveTo(error=f"/move --to takes only --keep and --queue, not {token!r}")
        else:
            words.append(token)
        index += 1
    if len(words) > 1:
        return MoveTo(error=AMBIGUOUS_MOVE)
    session_id = words[0] if words else ""
    if session_id and ("/" in session_id or session_id.startswith(("~", "."))):
        # A PATH WITH ``--to`` is the one shape the design refuses by name rather
        # than guessing (§1.7 table: "/move <path> --to … ⇒ refused").
        return MoveTo(error=AMBIGUOUS_MOVE)
    return MoveTo(session_id=session_id, to=to, keep=keep, queue=queue)


def run_session_move(
    session_id: str, to: str, *, keep: bool = False, queue: bool = False
) -> dict[str, Any]:
    """Run ``lop sessions move … --json`` and return its contract document.

    Blocking by design; the caller runs it off the loop. Every failure is a
    refusal in the contract's own shape (``ok: False``, ``code``, ``message``,
    ``changed: False``) so the caller has ONE branch for "it did not move", and a
    process that could not even start says so rather than looking like a refusal
    the relay gave.

    ``queue`` asks for the queued move (§5.4). Success still arrives in the same
    contract, with the record riding in the ``queue`` block and ``phase`` being
    the queue's own vocabulary — the caller tells the two apart by that block,
    never by a second call shape.
    """
    argv = python_argv("-m", "local_operator.cli", "sessions", "move", session_id, "--to", to)
    if keep:
        argv.append("--keep")
    if queue:
        argv.append("--queue")
    argv.append("--json")
    env = dict(os.environ)
    env.pop("FORCE_COLOR", None)
    try:
        child = subprocess.Popen(  # noqa: S603 — argv is built here, never a shell
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return _refusal(session_id, "cli_unavailable", f"could not start the CLI ({exc})")
    try:
        out, err = child.communicate(timeout=MOVE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _reap_group(child)
        child.communicate()
        return _refusal(
            session_id,
            "deadline_exceeded",
            "the move did not finish in time; `lop sessions move "
            f"{session_id} --to {to}` again reports where it got to",
            changed=True,
        )
    text = (out or "").strip()
    try:
        payload = json.loads(text) if text.startswith("{") else None
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        return payload
    lines = [line for line in _ANSI_RE.sub("", err or "").splitlines() if line.strip()]
    return _refusal(
        session_id,
        "cli_failed",
        lines[-1] if lines else f"the CLI exited {child.returncode} with no answer",
    )


def _refusal(session_id: str, code: str, message: str, *, changed: bool = False) -> dict[str, Any]:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "session_id": session_id,
        "phase_reached": None,
        "changed": changed,
    }
