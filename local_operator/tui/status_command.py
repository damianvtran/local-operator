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
import unicodedata
from dataclasses import dataclass
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
#: How much of stdout is KEPT. The segment is one line capped at
#: :data:`MAX_CELLS`; anything past this is read and discarded (see
#: :func:`_collect`), so memory per run is bounded whatever the script prints.
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
    r"|\x9b[0-?]*[ -/]*[@-~]"  # 8-bit C1 CSI: the introducer alone is Cc-stripped
    # OSC, DCS, SOS, PM and APC are all STRINGS terminated by BEL or ST; strip
    # the payload with the introducer, or a DCS leaves its parameters behind as
    # junk text on the band (review round 1, m3).
    r"|\x1b[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[@-Z\\-_]"  # two-byte escapes
)
#: Unicode categories that never belong on a one-row status: controls (Cc),
#: format characters (Cf: bidi overrides/isolates such as U+202E and U+2066,
#: zero-width space/joiners, BOM) and line/paragraph separators (Zl, Zp). A
#: right-to-left override in a script's output visually reorders every segment
#: after it in a bidi-aware terminal — the band's alarms included — and a
#: zero-width character makes ``cell_len`` and the painted width disagree.
_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
#: Read size per ``StreamReader.read``. Independent of :data:`MAX_READ_BYTES`:
#: past the cap the reader still drains, and draining in 64 KiB steps keeps a
#: flood from costing a Python call per byte.
_CHUNK_BYTES = 65536


#: Consecutive failures after which a stale reading is cleared from the band.
#: Below it the last good line stays, dimmed (design round 1, T-3): a script
#: calling a network API fails now and then, and clearing on every miss made
#: three sibling segments appear and vanish each interval as the row re-fit.
STALE_AFTER_FAILURES = 3


@dataclass(frozen=True)
class Outcome:
    """One run's result. ``error`` is empty on success.

    A success with no output is NOT an error: a script printing nothing is
    saying "nothing to show right now", and the segment clears.
    """

    line: str = ""
    error: str = ""


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
    # Split on real newlines BEFORE category stripping: Zl/Zp would otherwise be
    # treated as line breaks by ``splitlines`` and promote a second line.
    for line in text.split("\n"):
        visible = "".join(
            " " if char == "\t" else char
            for char in line
            if char == "\t" or unicodedata.category(char) not in _STRIPPED_CATEGORIES
        )
        cleaned = " ".join(visible.split())
        if cleaned:
            return _truncate(cleaned, MAX_CELLS)
    return ""


def _truncate(text: str, cells: int) -> str:
    if cell_len(text) <= cells:
        return text
    used = 0
    end = 0
    for index, char in enumerate(text):
        width = cell_len(char)
        if used + width > cells - 1:
            break
        used += width
        end = index + 1
    return text[:end].rstrip() + "…"


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
) -> Outcome:
    """Run ``command`` once: its sanitised first line, or why it failed.

    Never raises: every failure — no shell, a non-zero exit, a timeout, a read
    error — comes back as an :class:`Outcome` with a short machine-ish
    ``error`` (``exit 127``, ``timed out after 10s``) the app can surface once.
    """
    if not command.strip():
        return Outcome()
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
        return Outcome(error="could not start /bin/sh")
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    try:
        kept = await asyncio.wait_for(_collect(process, data), timeout=timeout)
    except asyncio.TimeoutError:
        await _reap(process)
        logger.debug("status command: timed out after %.1fs", timeout)
        return Outcome(error=f"timed out after {timeout:g}s")
    except asyncio.CancelledError:
        # The worker was cancelled (app exit, or the command changed while this
        # run was in flight). The child must not outlive the cancellation.
        await _reap(process)
        raise
    except Exception:  # noqa: BLE001 — a read error is an empty segment
        logger.debug("status command: read failed", exc_info=True)
        await _reap(process)
        return Outcome(error="output could not be read")
    if process.returncode != 0:
        logger.debug("status command: exited %s", process.returncode)
        return Outcome(error=f"exit {process.returncode}")
    return Outcome(line=sanitize(kept.decode("utf-8", errors="replace")))


async def _collect(process: asyncio.subprocess.Process, data: bytes) -> bytes:
    """Feed stdin, keep the first :data:`MAX_READ_BYTES` of stdout, drain the rest.

    INCREMENTAL, never ``communicate()``: that buffers ALL of stdout before a
    slice could cap it, so a script that floods (``yes``, a chatty log) grew the
    TUI by gigabytes inside one timeout (review round 1, M1: 3.6 GB measured in
    one second). Past the cap the reader keeps DRAINING and discards, rather
    than stopping: a child blocked writing into a full pipe never exits, and a
    script that prints a lot and then exits 0 is still a success whose first
    line is in the kept prefix.
    """
    assert process.stdin is not None and process.stdout is not None
    stdin = process.stdin

    async def feed() -> None:
        # CONCURRENT with the read below (review round 2, N3): a script that
        # prints before it reads stdin would otherwise block on a full stdout
        # pipe while we block writing a payload larger than the stdin pipe.
        try:
            stdin.write(data)
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the script ignores stdin and already exited: allowed
        finally:
            try:
                stdin.close()
            except Exception:  # noqa: BLE001 — closing a dead pipe is not an error
                pass

    feeder = asyncio.ensure_future(feed())
    try:
        kept = bytearray()
        while chunk := await process.stdout.read(_CHUNK_BYTES):
            if len(kept) < MAX_READ_BYTES:
                kept += chunk[: MAX_READ_BYTES - len(kept)]
        await process.wait()
    finally:
        if not feeder.done():
            feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
    return bytes(kept)


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Kill the child's whole process GROUP, by the pid we created, and wait.

    The wait is what closes the child's pipes on THIS loop: without it the
    transport is left for the garbage collector, which may run after the loop
    has closed and log a spurious "Event loop is closed" per killed run. It
    DRAINS stdout first, because asyncio's ``wait()`` resolves only once every
    pipe has reported EOF, and a stream reader paused on a full buffer never
    reads to EOF — the round-1 flood case waited out the whole bound here and
    leaked the transport. Shielded so a second cancellation cannot abandon the
    wait half-way; that cancellation itself still propagates, so a cancelled
    worker stays cancelled.
    """
    try:
        # start_new_session makes the child's PID its group ID even after the
        # shell exits; getpgid(pid) would miss pipe-holding descendants then.
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass

    async def drain_and_wait() -> None:
        if process.stdout is not None:
            while await process.stdout.read(_CHUNK_BYTES):
                pass
        await process.wait()

    try:
        await asyncio.shield(asyncio.wait_for(drain_and_wait(), timeout=2.0))
    except Exception:  # noqa: BLE001 — best effort; the kill already landed
        logger.debug("status command: reaped child did not report exit", exc_info=True)
    # Close the transport on THIS loop even when the bounded wait ran out on a
    # loaded machine (review round 3, M1): left to the garbage collector it can
    # be finalised after the loop closes and log "Event loop is closed".
    transport = getattr(process, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:  # noqa: BLE001 — already closed, or the loop is going
            pass
