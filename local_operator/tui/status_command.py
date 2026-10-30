"""The operator's status command: one line of THEIR information on the band.

``display.composer.command`` names a shell command; the TUI runs it every
``display.composer.command_interval`` seconds and paints the first line it
prints as a band segment (``StatusLine.update(custom=...)``). It is the
extension point for facts the harness does not know and should not learn —
an account's quota from a plugin, a build or deploy state, a ticket the branch
belongs to — so it is deliberately a COMMAND rather than a list of built-in
segments: the operator's other agent tools already ship scripts that print
those facts, and any of them can be pointed at this key unchanged.

The contract mirrors the status-line hook other agent CLIs expose, so one
script can serve several harnesses:

- the command runs through ``/bin/sh -c`` in the TUI's working directory;
- it receives a JSON object on stdin (:func:`build_payload`) describing the
  session — ignore it freely, nothing requires reading stdin;
- the first non-blank stdout line is the segment; a non-zero exit, a timeout
  or empty output CLEARS the segment rather than painting an error, because a
  band reading `command failed` every 30 s is worse than a band without it.

Output is sanitised to plain text before it reaches the band. The band is one
row of measured cell arithmetic (see ``status_line``'s module docstring), so a
stray escape sequence, tab or newline would misplace every segment after it;
and a terminal escape from a script is a write to the operator's terminal that
bypasses Textual entirely. SGR colour is therefore DROPPED, not passed through.

Runs off the event loop's critical path: :func:`run` is a coroutine over
``asyncio.create_subprocess_exec`` and the app schedules it in an exclusive
worker group, so a slow script delays only its own segment. The child gets its
own session (``start_new_session``) and the whole process GROUP is killed on
timeout — a script that spawns a helper and hangs must not leave the helper
running for the life of the TUI (the same by-group reap ``network_cli`` uses).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from typing import Any, Mapping

from rich.cells import cell_len

logger = logging.getLogger(__name__)

#: Default seconds between runs. Long enough that a script calling a remote API
#: is not a load generator from every open TUI, short enough that a quota
#: reading is current within the time it takes to notice it.
DEFAULT_INTERVAL_S = 30
#: Bounds for ``display.composer.command_interval``. The floor stops a typo
#: (`1`) from forking a shell every second in every session on the machine; the
#: ceiling is an hour, past which "status" has stopped describing it.
MIN_INTERVAL_S = 5
MAX_INTERVAL_S = 3600
#: Hard timeout for one run. Fixed rather than derived from the interval: a
#: status script that needs more than this is doing work that belongs in a
#: cache it reads, and a shorter bound keeps a hung script from holding the
#: worker group across several ticks.
TIMEOUT_S = 10.0
#: How much of stdout is read. The segment is one line capped at
#: :data:`MAX_CELLS`; a script that dumps a log must not be buffered whole.
MAX_READ_BYTES = 4096
#: Cell cap for the painted segment. A BOUNDED rung, so it may sit anywhere in
#: the drop ladder (``status_line._UNBOUNDED_RUNGS``); 40 cells holds a
#: compound reading (`5h 42% · 7d 18% · resets 3h`) whole at ordinary widths.
MAX_CELLS = 40

# CSI (`ESC [ ... final`), OSC (`ESC ] ... BEL|ST`) and any other two-byte
# escape. Broader than ``network_cli._ANSI_RE`` (SGR only) on purpose: that one
# cleans our own CLI's output, this one cleans arbitrary third-party output.
_ESCAPE_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC, BEL- or ST-terminated
    r"|\x1b[@-Z\\-_]"  # two-byte escapes
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clamp_interval(value: Any) -> int:
    """``value`` as a usable interval: a hand-edited bad value never stalls it.

    The registry refuses out-of-range writes, but ``config.yml`` is also edited
    by hand, so the reader clamps instead of trusting it — the same rule the
    other bounded keys follow.
    """
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    return max(MIN_INTERVAL_S, min(MAX_INTERVAL_S, seconds))


def sanitize(raw: str) -> str:
    """The first non-blank line of ``raw`` as plain, single-row band text."""
    text = _ESCAPE_RE.sub("", raw)
    for line in text.splitlines():
        cleaned = " ".join(_CONTROL_RE.sub(" ", line).split())
        if cleaned:
            return _truncate(cleaned, MAX_CELLS)
    return ""


def _truncate(text: str, cells: int) -> str:
    if cell_len(text) <= cells:
        return text
    out = ""
    for char in text:
        if cell_len(out + char) > cells - 1:
            break
        out += char
    return out.rstrip() + "…"


def build_payload(
    *,
    session_id: str,
    cwd: str,
    provider: str,
    model_id: str,
    model_label: str,
    context_tokens: int,
    context_window: int,
    cost: str,
    version: str,
) -> dict[str, Any]:
    """The JSON object a status command receives on stdin.

    Additive-only once shipped: scripts parse this, so a field is never renamed
    or removed, and a new one is optional for every reader. ``cost`` is the
    band's own formatted string (``$0.12``, or empty when unpriced) rather than
    a float, because the band is the authority on how spend is shown and a
    script that wants the number can parse it.
    """
    return {
        "session_id": session_id,
        "cwd": cwd,
        "model": {"provider": provider, "id": model_id, "display_name": model_label},
        "context": {"tokens": context_tokens, "window": context_window},
        "cost": cost,
        "version": version,
    }


async def run(
    command: str,
    payload: Mapping[str, Any],
    *,
    cwd: str | None = None,
    timeout: float = TIMEOUT_S,
) -> str:
    """Run ``command`` once and return its sanitised first line ("" on failure).

    Never raises: every failure — no shell, a non-zero exit, a timeout, bytes
    that do not decode — reads as an empty segment and a debug log line. The
    band is a passive surface; it does not get to interrupt the operator with
    a script's problems.
    """
    if not command.strip():
        return ""
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001 — a missing shell is an empty segment
        logger.debug("status command: could not start", exc_info=True)
        return ""
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(data), timeout=timeout)
    except asyncio.TimeoutError:
        await _reap(process)
        logger.debug("status command: timed out after %.1fs", timeout)
        return ""
    except asyncio.CancelledError:
        # The worker was cancelled (app exit, or a newer run replaced this one
        # in the exclusive group). The child must not outlive the cancellation.
        await _reap(process)
        raise
    except Exception:  # noqa: BLE001 — e.g. BrokenPipe when the script ignores stdin and exits
        logger.debug("status command: communicate failed", exc_info=True)
        await _reap(process)
        return ""
    if process.returncode != 0:
        logger.debug("status command: exited %s", process.returncode)
        return ""
    return sanitize(stdout[:MAX_READ_BYTES].decode("utf-8", errors="replace"))


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Kill the child's whole process GROUP, by the pid we created, and wait.

    The wait is what closes the child's pipes on THIS loop: without it the
    transport is left for the garbage collector, which may run after the loop
    has closed and log a spurious "Event loop is closed" per killed run.
    Shielded so a second cancellation cannot abandon the wait half-way; that
    cancellation itself still propagates, so a cancelled worker stays cancelled.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        await asyncio.shield(asyncio.wait_for(process.wait(), timeout=2.0))
    except Exception:  # noqa: BLE001 — best effort; the kill already landed
        logger.debug("status command: reaped child did not report exit", exc_info=True)
