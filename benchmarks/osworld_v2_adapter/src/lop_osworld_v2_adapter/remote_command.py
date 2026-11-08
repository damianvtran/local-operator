"""Text-returning commands over the existing guest control endpoint.

The server's own timeout is an HTTP 500 without partial output. A tiny guest
wrapper therefore owns the inner deadline and kills the shell process group,
then returns a bounded JSON value through /execute's stdout. The outer deadline
is transport slack, NOT a second attempt: a lost reply must never replay a shell
command which may already have changed target state. Output is spooled to guest
temporary files rather than accumulated unboundedly in either process's memory.

EVERY FAILURE THE TARGET REPORTS IS A RESULT, NEVER AN EXCEPTION (see
``decode_result``). The parent treats an exception from a mutating call as an
AMBIGUOUS MUTATION and poisons the session, which is the right rule for a batch
of actions and the wrong one for a shell command: the sealed run
``t27b-t003-20261108-090532`` lost an entire episode -- every later tool call
and ``score`` -- because the guest's own deadline killed this wrapper with
SIGTERM and answered ``returncode: -15``. A command's failure is an outcome the
model reads and reacts to. Only a reply that never arrived (a genuine channel
failure) may still raise, because a command that may already have run must not
be replayed.
"""

from __future__ import annotations

import json
import signal
from typing import Any

from pydantic import ValidationError

from local_operator.evaluation.adapters.api import (
    RunCommandParams,
    RunCommandResult,
    cap_command_output,
)

# Standard library only: the target already needs python3 for its control server.
# Keep both streams out of the wrapper's stdout until the single JSON reply.
GUEST_WRAPPER = r"""
import codecs, json, os, signal, subprocess, sys, tempfile
command, timeout, cap = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    child = subprocess.Popen(command, shell=True, stdout=out, stderr=err,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    timed_out = False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
    result = {"exit_code": None if timed_out else child.returncode,
              "timed_out": timed_out}
    for name, stream in (("stdout", out), ("stderr", err)):
        size = stream.seek(0, 2)
        stream.seek(0)
        data = stream.read(cap)
        # Ignore a split final UTF-8 code point only when the cap cut it;
        # genuine non-UTF-8 bytes are replaced, never silently dropped.
        truncated = size > len(data)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(data, final=not truncated)
        while len(text.encode("utf-8")) > cap:
            truncated = True
            text = text[:-1]
        result[name] = text
        result[name + "_bytes"] = size
        result[name + "_truncated"] = truncated
    print(json.dumps(result))
"""


#: Bound on the target's own failure text carried in ``RunCommandResult.error``.
#: A wrapper diagnostic quotes the guest's stderr, so it needs a ceiling of its
#: own: the RPC answer is one line and must stay one line.
MAX_ERROR_CHARS = 4096

#: The two keys every /execute answer carries. A non-2xx body that has both is
#: the guest's own answer (see ``answer_from_error_body``); anything else -- a
#: proxy page, a load balancer's JSON -- is not.
_ANSWER_KEYS = frozenset({"output", "returncode"})


def _exit_description(returncode: int) -> str:
    """Describe a guest exit status: an ordinary code, or death by signal.

    ``subprocess`` reports a signal-killed child as a NEGATIVE returncode, so
    ``exit -15`` in a diagnostic is really "SIGTERM killed it" wearing a
    misleading costume. A reader who does not know that convention reads -15 as
    an exit code and looks for a program that returned one; naming the signal
    removes a research step that has already cost a full investigation.

    It lives HERE rather than in ``providers/aws.py`` (which re-exports it)
    because the wrapped command's status is this module's vocabulary: both the
    ``execute`` diagnostic and the wrapper-failure answer name it, and two
    copies would drift.

    Deliberately says WHAT happened and not WHY. A signal death has many
    possible originators -- the OS, a supervisor, another process, or the
    command's own actions -- and this layer cannot distinguish them. Naming a
    specific cause here would encode a guess into every future incident report,
    which is the failure mode that made the last one expensive.

    Leaks nothing: the output is a signal NAME from the stdlib's fixed
    enumeration or a bare integer, never command text, guest stdout, or stderr.
    That is the same discipline ``execute`` applies to transport errors.
    """

    if returncode >= 0:
        return f"exit {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        # An unrecognised signal number is still worth reporting as a signal
        # rather than as a negative exit code.
        return f"terminated by signal {-returncode}"
    return f"terminated by signal {name}"


def _as_count(value: Any) -> int:
    """The wrapper's own byte count, or 0 when it did not give a usable one.

    ``type(...) is int`` rather than ``isinstance``: ``True`` is an int in
    Python, and a bool standing in for a byte count would silently mean 1.
    """

    return value if type(value) is int and value >= 0 else 0


def _bounded(text: str) -> str:
    """``text`` cut to ``MAX_ERROR_CHARS``, saying so when it was cut."""

    if len(text) <= MAX_ERROR_CHARS:
        return text
    note = f"... [truncated from {len(text)} characters]"
    return text[: MAX_ERROR_CHARS - len(note)] + note


def _unrun(note: str) -> RunCommandResult:
    """The shape for a command whose shell status is UNKNOWN, and why.

    Never an invented exit code and never a timeout flag: the two say things
    about the command that this layer did not observe. ``error`` is the field
    that carries "it did not run, or we cannot know how it ended", which is
    exactly the fact here -- and it keeps the two streams empty, because the
    target never produced them.
    """

    return RunCommandResult(
        exit_code=None,
        timed_out=False,
        error=_bounded(note),
        stdout="",
        stderr="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
    )


def _wrapper_failure(body: dict[str, Any], *, note: str | None = None) -> RunCommandResult:
    """The wrapper did not hand back a usable result: name what it did instead."""

    status = body.get("status")
    returncode = body.get("returncode")
    facts = [f"status={status!r}"]
    if type(returncode) is int:
        facts.append(f"wrapper {_exit_description(returncode)}")
    else:
        facts.append(f"wrapper returncode={returncode!r}")
    text = "the target command wrapper failed (" + ", ".join(facts) + ")"
    detail = body.get("error")
    if isinstance(detail, str) and detail.strip():
        text += f": {detail.strip()}"
    elif note is not None:
        text += f"; {note}"
    return _unrun(text)


def answer_from_error_body(content: bytes | None) -> dict[str, Any] | None:
    """The guest's own /execute answer inside a non-2xx response, or None.

    The guest control server answers SOME command outcomes with an error status
    while still carrying its normal four-key body -- upstream's timeout handler
    is one. That body is the TARGET TALKING: dropping it on the floor (the
    caller re-raising an HTTP error) turned a command outcome into a channel
    failure and poisoned the session. Anything without both ``output`` and
    ``returncode`` is not that body -- a redirect page, a reverse proxy's
    ``{"message": ...}`` -- so it stays a channel failure, where failing the
    call and poisoning are the safe (unreplayable) answers.
    """

    if not content:
        return None
    try:
        decoded = json.loads(bytes(content).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(decoded, dict) or not _ANSWER_KEYS <= set(decoded):
        return None
    return decoded


def payload_for(params: RunCommandParams) -> dict[str, Any]:
    """Use argv, not interpolation, so user quoting belongs to the inner shell."""
    return {
        "command": [
            "python3",
            "-c",
            GUEST_WRAPPER,
            params.command,
            str(params.timeout_s),
            str(params.max_output_bytes),
        ],
        "shell": False,
        "timeout": params.timeout_s + 5,
    }


def decode_result(body: dict[str, Any], params: RunCommandParams) -> RunCommandResult:
    """Turn the control server's answer into a result -- and never raise for one.

    Every failure the TARGET can report becomes a ``RunCommandResult`` the
    model can read and react to:

    * the wrapper exited nonzero (measured: the guest's own deadline SIGTERM'd
      it before it printed anything, so the shell's fate is unknown);
    * the wrapper hit its inner deadline and said so (``timed_out: true``,
      ``exit_code: null``) -- passed through as the fact it is;
    * the control server answered an error body for the command;
    * the payload was missing, not JSON, not an object, or not a command
      result;
    * a stream exceeded the cap, which is capped in place with the truncation
      flags -- reported as truncation, never as a failure, because the command
      did run and its exit status is real.

    A CHANNEL failure never reaches this function: the HTTP call that would
    raise it happens in the provider, and it stays a raise there for the one
    case that must not be replayed. So this function raises nothing at all.
    """

    if not isinstance(body, dict):
        return _unrun(f"the target answered a non-object body: {body!r}")
    status = body.get("status")
    returncode = body.get("returncode")
    if status != "success" or type(returncode) is not int or returncode != 0:
        return _wrapper_failure(body)
    raw = body.get("output")
    if not isinstance(raw, str) or not raw.strip():
        return _wrapper_failure(body, note="it produced no result payload")
    try:
        decoded = json.loads(raw)
    except ValueError as error:
        return _unrun(f"the target command wrapper's payload is not JSON: {error}")
    if not isinstance(decoded, dict):
        return _unrun("the target command wrapper's payload is not an object")
    # Cap BEFORE validating: a wrapper that ignored the cap (or even the
    # protocol's absolute ceiling) must report truncation, not become a
    # malformed payload that costs the caller the command's real status.
    for stream in ("stdout", "stderr"):
        text = decoded.get(stream)
        if not isinstance(text, str):
            continue
        kept, total, cut = cap_command_output(text, params.max_output_bytes)
        decoded[stream] = kept
        decoded[f"{stream}_bytes"] = max(_as_count(decoded.get(f"{stream}_bytes")), total)
        decoded[f"{stream}_truncated"] = bool(decoded.get(f"{stream}_truncated")) or cut
    try:
        return RunCommandResult.model_validate(decoded)
    except ValidationError as error:
        return _unrun(f"the target command wrapper's payload is not a command result: {error}")
