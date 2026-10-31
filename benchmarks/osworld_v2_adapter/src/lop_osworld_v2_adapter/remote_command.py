"""Text-returning commands over the existing guest control endpoint.

The server's own timeout is an HTTP 500 without partial output. A tiny guest
wrapper therefore owns the inner deadline and kills the shell process group,
then returns a bounded JSON value through /execute's stdout. The outer deadline
is transport slack, NOT a second attempt: a lost reply must never replay a shell
command which may already have changed target state. Output is spooled to guest
temporary files rather than accumulated unboundedly in either process's memory.
"""

from __future__ import annotations

from typing import Any

from local_operator.evaluation.adapters.api import RunCommandParams, RunCommandResult

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
    """A server/wrapper failure is not a shell exit code or an invented timeout."""
    if (
        body.get("status") != "success"
        or type(body.get("returncode")) is not int
        or body["returncode"] != 0
    ):
        raise RuntimeError(f"target command wrapper failed: {body!r}")
    result = RunCommandResult.model_validate_json(body["output"])
    for stream in ("stdout", "stderr"):
        if len(getattr(result, stream).encode("utf-8")) > params.max_output_bytes:
            raise ValueError(f"target exceeded {stream} output cap")
    return result
