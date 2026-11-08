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
and ``score`` -- because the target had SIGTERM'd this wrapper and answered
``returncode: -15``, and this layer raised on that answer. A command's failure
is an outcome the model reads and reacts to. Only a reply that never arrived (a
genuine channel failure) may still raise, because a command that may already
have run must not be replayed.

A DEAD WRAPPER DOES NOT STOP THE COMMAND. Its child runs in its own session
(``start_new_session=True``), so whatever kills the wrapper leaves the shell
running and mutating the target. The guest's own deadline does exactly that
(upstream's handler kills only its direct child), and so did the measured case:
the command contained ``pkill -f citysheet`` and ``payload_for`` used to put the
command TEXT in the wrapper's argv, so the command matched and killed its own
wrapper -- which reported ``returncode: -15`` six seconds into a sixty-second
deadline. Both the argv leak (``payload_for`` now base64s the text) and the
silence about the surviving shell (``UNKNOWN_OUTCOME_CAUTION``) are fixed here
rather than in a caller, because only this layer knows either fact.
"""

from __future__ import annotations

import base64
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
import base64, codecs, json, os, signal, subprocess, sys, tempfile
command = base64.b64decode(sys.argv[1]).decode("utf-8")
timeout, cap = float(sys.argv[2]), int(sys.argv[3])
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


#: How many field-level pydantic errors a malformed payload reports. Remaining
#: ones are counted, not printed: the model needs the first wrong field, and
#: ``str(ValidationError)`` dumps one line per error plus a docs URL.
MAX_VALIDATION_ERRORS = 5

#: Bound on the target's own failure text carried in ``RunCommandResult.error``.
#: A wrapper diagnostic quotes the guest's stderr, so it needs a ceiling of its
#: own: the RPC answer is one line and must stay one line.
MAX_ERROR_CHARS = 4096

#: The two keys every /execute COMMAND answer carries. A non-2xx body that has
#: both is the guest's own answer (see ``answer_from_error_body``); anything else
#: -- a proxy page, a load balancer's JSON -- is not.
_ANSWER_KEYS = frozenset({"output", "returncode"})

#: The target server's own ``status`` vocabulary, read from the AMI's
#: ``server/main.py`` (upstream ``36a6adf4``): ``success``/``error`` from
#: ``/execute`` and ``verification_failed`` from its sibling route. A non-2xx
#: body that claims this field with a word outside the set is not the guest.
_GUEST_STATUSES = frozenset({"success", "error", "verification_failed"})

#: Appended to every result whose command outcome this layer cannot state. It is
#: the one thing a model must not get wrong: the sealed run's agent re-issued a
#: near-identical command one turn after the wrapper died, because the failure
#: text said nothing about the shell still running. Saying it costs one clause
#: and removes the only remaining way this module can cause a double execution.
UNKNOWN_OUTCOME_CAUTION = (
    "the command's outcome is unknown -- it may have run already, or it may "
    "still be running on the target -- so re-issuing it blindly could run it "
    "twice"
)


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
    """``text`` cut to ``MAX_ERROR_CHARS``, keeping BOTH ends when it is cut.

    The head says what happened; the tail is where a quoted diagnostic puts its
    own conclusion -- ``str(TimeoutExpired)`` ends with "timed out after N
    seconds" while everything in front of it is the command's whole argv. A
    head-only cut would drop the conclusion, so the middle is what goes.

    It also guarantees the closing caution survives: ``_unrun`` appends it, and
    a sentence a model must read before re-issuing a command is not something a
    length bound may remove.
    """

    if len(text) <= MAX_ERROR_CHARS:
        return text
    elided = len(text) - MAX_ERROR_CHARS
    # The marker's own length depends on the digit count of ``elided``, which
    # depends on the room left for the text; two passes settle it.
    while True:
        marker = f" ... [{elided} characters elided] ... "
        keep = MAX_ERROR_CHARS - len(marker)
        if len(text) - keep == elided:
            break
        elided = len(text) - keep
    head = text[: keep // 2]
    return head + marker + text[-(keep - len(head)) :]


def _unrun(note: str) -> RunCommandResult:
    """The shape for a command whose shell status is UNKNOWN, and why.

    Never an invented exit code and never a timeout flag: the two say things
    about the command that this layer did not observe. ``error`` is the field
    that carries "it did not run, or we cannot know how it ended", which is
    exactly the fact here -- and it keeps the two streams empty, because the
    target never produced them.

    Every one of these notes ends with ``UNKNOWN_OUTCOME_CAUTION``. A model that
    reads only "the wrapper failed" and re-issues the command is the measured
    failure this module exists to prevent, and the caution is the only place it
    can be told the shell outlives the wrapper.
    """

    return RunCommandResult(
        exit_code=None,
        timed_out=False,
        error=_bounded(f"{note}; {UNKNOWN_OUTCOME_CAUTION}"),
        stdout="",
        stderr="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
    )


def _guest_detail(body: dict[str, Any]) -> str:
    """The target's own words about a failure, when it gave any.

    ``error`` carries the command's/wrapper's stderr in the guest's command
    answer; ``message`` is what its exception handler returns instead
    (``execute_command``'s ``except Exception`` arm). Quoted, never
    paraphrased: this layer reports
    what was observed, and the target's sentence is the observation.
    """

    for key in ("error", "message"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _wrapper_failure(body: dict[str, Any]) -> RunCommandResult:
    """The wrapper reported no command status: say WHAT it did instead.

    Deliberately not WHY, the same rule ``_exit_description`` follows: a signal
    death has many possible originators -- the guest's deadline, a supervisor,
    another process, or the command's own ``pkill -f`` matching the wrapper --
    and this layer cannot tell them apart. Naming one would bake a guess into
    every future incident report, which is what made the last one expensive.

    The target's server status is quoted only when it is NOT ``success``: a
    server vouching for the request says nothing about the command, and
    ``status='success'`` printed beside ``exit_code: none`` reads as a
    contradiction to the model that has to act on it.
    """

    returncode = body.get("returncode")
    if type(returncode) is not int:
        # Only when the key is there: the guest's exception envelope has no
        # ``returncode`` at all, and printing ``returncode=None`` would report a
        # field the target never sent.
        present = f" (returncode={returncode!r})" if "returncode" in body else ""
        observation = f"the target command wrapper reported no result{present}"
    elif returncode == 0:
        # No exit code in the text: "exit 0" beside ``exit_code: none`` is the
        # same contradiction as a quoted ``success``, and it says nothing the
        # model can act on.
        observation = "the target command wrapper finished without reporting a result"
    elif returncode < 0:
        observation = (
            f"the target command wrapper was {_exit_description(returncode)} "
            "without reporting a result"
        )
    else:
        observation = f"the target command wrapper exited {returncode} without reporting a result"
    status = body.get("status")
    if isinstance(status, str) and status != "success":
        observation += f"; the target answered status={status!r}"
    detail = _guest_detail(body)
    if detail:
        observation += f": {detail}"
    return _unrun(observation)


def answer_from_error_body(content: bytes | None) -> dict[str, Any] | None:
    """The guest's own /execute answer inside a non-2xx response, or None.

    TWO shapes are the guest talking, and both are read from the AMI's own
    ``server/main.py`` (upstream ``36a6adf4``), not assumed:

    * its COMMAND answer -- ``{status, output, error, returncode}`` -- which that
      handler returns for a completed command, in ``execute_command``'s ``try``
      body. Upstream answers 200 for it, so this arm is defensive rather than
      routine;
    * its EXCEPTION envelope -- ``{"status": "error", "message": ...}`` -- which
      every failure of that handler returns with status 500, from its
      ``except Exception`` arm, ``subprocess.TimeoutExpired`` among them. It
      carries no command output at all, and answering it is the difference
      between a command the target reported on and a session-ending channel
      failure: the guest's deadline kills only the WRAPPER, whose shell keeps
      running.

    Anything else -- a redirect page, a reverse proxy's ``{"message": ...}`` --
    is not the guest, so it stays a channel failure, where failing the call and
    poisoning are the safe (unreplayable) answers. ``status`` must be one of the
    server's own words for the same reason: without it, any body that happened
    to carry ``output`` and ``returncode`` would be read as a command result.
    """

    if not content:
        return None
    try:
        decoded = json.loads(bytes(content).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(decoded, dict) or decoded.get("status") not in _GUEST_STATUSES:
        return None
    if _ANSWER_KEYS <= set(decoded):
        return decoded
    if decoded["status"] == "error" and isinstance(decoded.get("message"), str):
        return decoded
    return None


def payload_for(params: RunCommandParams) -> dict[str, Any]:
    """Use argv, not interpolation, so user quoting belongs to the inner shell.

    THE COMMAND TEXT NEVER APPEARS IN THE WRAPPER'S OWN ARGV, and that is
    load-bearing rather than tidiness. ``pkill -f``, ``pgrep -f`` and ``ps |
    grep`` match a process's whole argv, so a command containing one of them
    matched the wrapper that was running it: the sealed run's command contained
    ``pkill -f citysheet`` and SIGTERM'd its own wrapper six seconds into a
    sixty-second deadline. Base64 keeps the text out of the process table.

    stdin would be the cleaner channel and is not available: ``/execute`` gives
    its child no per-request stdin at all -- upstream's handler pipes only
    stdout and stderr (``execute_command``'s ``subprocess.run``) and the
    server's own stdin is
    shared by every request -- so argv is the only channel this endpoint has.
    The taint is therefore removed where it can be (the command), and the
    wrapper's source, which no command can usefully match on, remains.
    """

    encoded = base64.b64encode(params.command.encode("utf-8")).decode("ascii")
    return {
        "command": [
            "python3",
            "-c",
            GUEST_WRAPPER,
            encoded,
            str(params.timeout_s),
            str(params.max_output_bytes),
        ],
        "shell": False,
        "timeout": params.timeout_s + 5,
    }


def unreadable_answer(reason: object) -> RunCommandResult:
    """A 2xx answer this layer cannot read: the target SPOKE, unreadably.

    ``http_post_json`` raises for these itself -- ``response.json()`` on a
    non-JSON body, its own ``ValueError`` on a JSON body that is not an object --
    so they never reach ``decode_result`` and used to poison the session even
    though the target had answered. An unreadable answer says nothing about the
    command's fate, so it takes the same unknown-outcome shape as a wrapper that
    never reported.
    """

    return _unrun(f"the target answered with a body this layer cannot read: {reason}")


def decode_result(body: dict[str, Any], params: RunCommandParams) -> RunCommandResult:
    """Turn the control server's answer into a result -- and never raise for one.

    ONE RULE: the command payload, when the target sent a usable one, IS the
    result; everything else is an unknown outcome that says what was observed.
    The payload is written by the wrapper after it has waited for the shell, so
    a payload that parses is a completed command whatever the server's own
    ``status`` or the wrapper's exit code say -- and it is the only evidence
    about what the command actually did, so discarding it would lose the answer
    without gaining certainty.

    In order:

    * a payload that decodes to a ``RunCommandResult`` is the result;
    * a payload that is present but unusable (not JSON, not an object, not a
      command result) is reported as such -- the wrapper ran and its exit status
      was real, but the value never arrived;
    * a stream over the cap is capped in place with the truncation flags --
      truncation, never a failure, because the command did run and its exit
      status is real;
    * no payload at all means the wrapper never reported a status
      (``_wrapper_failure``), which is what the sealed run's ``returncode: -15``
      body was.

    ``timed_out: true`` and ``exit_code: null`` from the WRAPPER are passed
    through as the facts they are: the wrapper owns the inner deadline and its
    child is in its own session, so it can state that it killed the group.

    A CHANNEL failure never reaches this function: the HTTP call that would
    raise it happens in the provider, and it stays a raise there for the one
    case that must not be replayed. So this function raises nothing at all.
    """

    if not isinstance(body, dict):
        return _unrun(f"the target answered a non-object body: {body!r}")
    raw = body.get("output")
    if not isinstance(raw, str) or not raw.strip():
        return _wrapper_failure(body)
    return _decode_payload(raw, params)


def _decode_payload(raw: str, params: RunCommandParams) -> RunCommandResult:
    """The wrapper's JSON payload as a result, or the reason it is not one."""

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
        return _unrun(
            "the target command wrapper's payload is not a command result: "
            f"{_summarize_validation(error)}"
        )


def _summarize_validation(error: ValidationError) -> str:
    """A validation failure a model can act on, not a pydantic dump.

    The raw ``str()`` of a ``ValidationError`` is one line per error plus a URL
    to pydantic's docs -- eight of those for a body with two bad fields. What
    the model needs is which field was wrong, so each error is reduced to
    ``field: message`` and the list is bounded.
    """

    errors = error.errors()
    shown = errors[:MAX_VALIDATION_ERRORS]
    details = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<result>'}: {item['msg']}"
        for item in shown
    )
    more = len(errors) - len(shown)
    suffix = f"; {more} more" if more > 0 else ""
    return f"{len(errors)} validation errors: {details}{suffix}"
