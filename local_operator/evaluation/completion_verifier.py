"""The session arm's completion verifier: config, and the guest inspection tool.

The verifier itself is generic (``Session.verify_completion``,
``harness/verifier.py``). This module is only what the OSWorld session arm adds
so the check can SEE a remote deliverable: an in-process ``inspect_command``
tool that runs read-only commands on the guest, and the arm's own switch.

WHY A SEPARATE CHANNEL FROM ``run_command``. The arm's own command path is the
action server (an MCP server declared ``ownTurnOnly``: children never inherit
it, and the manager refuses a call carrying a child's job id), and it takes the
bridge's ``_lock`` -- which ``ActionBridge.call`` HOLDS for the whole finish
branch, i.e. exactly while the verifier runs. Routing the verifier through it
would deadlock and would also make the measured surface reachable from a
child. ``inspect_command`` is therefore lock-free, lives only in the verifier
child, and is refused outside the verification window. Inside the window the
worker's turn is parked in ``call``, so the guest is quiescent apart from the
verifier's own reads.

READ-ONLY ON THE GUEST IS TWO HONEST MODES, recorded per verification as
``readonly_enforcement`` and never conflated:

* ``sandboxed`` -- a probe at the first inspection proved ``bwrap`` runs a
  command with exactly :data:`SANDBOX_GUARANTEES`: a read-only bind of ``/``,
  no network (which also hides abstract sockets), private PID and IPC
  namespaces (it cannot signal the desktop's processes), a new session (no
  terminal injection), and the path-based X11/ICE/D-Bus/user-runtime socket
  directories hidden under empty tmpfs with ``DISPLAY``/``DBUS_*``/
  ``XAUTHORITY``/``XDG_RUNTIME_DIR`` unset (it cannot type into apps or
  change settings over the session bus). The probe checks the write, the PID
  isolation and the hidden sockets; a probe that fails ANY of them is
  ``audit``. Every command is then wrapped the same way. Review round 1 (R1)
  found the first version bound only the filesystem while recording
  "no mutation" -- ``pkill``, ``gsettings set`` and ``xdotool`` would have
  reached the desktop. Nothing is SWEPT in this mode, so a command's
  ``mutation_detected`` is ``None`` (not measured), never ``False``.
* ``audit`` -- no working sandbox. Every command is followed by a sweep for
  files under ``$HOME`` and ``/tmp`` created or modified since the window
  opened, recorded as ``changed_files`` / ``mutation_detected``. It cannot
  see deletions or non-file state (a killed process, a changed setting).
  This DETECTS some mutation; it does not prevent it, and nothing here calls
  it read-only. Desktop applications write in the background too (autosave,
  recent-files), so a hit is a fact about the window, not proof the verifier
  wrote.

In BOTH modes obvious writes are refused lexically first (:func:`obvious_write`):
in ``sandboxed`` mode that is belt and braces, in ``audit`` the only gate.

``/tmp`` is deliberately NOT replaced by a tmpfs in the sandbox (the design
note's ``--tmpfs /tmp``): task deliverables live there often enough that
hiding them would make the check blind exactly where it must look.
"""

from __future__ import annotations

import asyncio
import math
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from local_operator.harness.verifier import (
    CANCEL_JOIN_S,
    DEFAULT_LOCAL_TOOL_ALLOWANCE,
    VerifierLimits,
)
from local_operator.harness.verifier_evidence import VerifierEvidence

#: Per-command bound for the verifier's guest reads. A read that needs more
#: than this is not a read; the worker's own commands keep their 300 s.
INSPECT_MAX_TIMEOUT_S = 30
INSPECT_DEFAULT_TIMEOUT_S = 15

#: The MCP client's default per-call deadline (``mcp/manager.py``
#: ``DEFAULT_MCP_TIMEOUT_MS`` = 300 s) bounds the worker's finish call, and
#: EVERYTHING the verifier costs runs inside that call: the once-per-episode
#: sandbox probe (:data:`PROBE_ENVELOPE_S`), the wall cap, and the cancel join
#: (``CANCEL_JOIN_S``; every wait inside ``verify_completion`` is clipped to
#: the wall, including the request a cap leaves the child for its verdict).
#: Review M2: the first version budgeted only wall + join (240 + 10) and the
#: probe pushed the worst case to 295 s. The 50 s delivery margin is the one
#: that version left for the challenge's own delivery.
MCP_FINISH_DEADLINE_S = 300.0
#: The adapter's per-command slack: ``session_arm._make_run_command`` gives
#: each guest command an RPC deadline of ``timeout_s +`` this, which is how
#: long a command may still be settling after its own timeout. THE single
#: source for that margin: the verifier's guest reads ride the same RPC path,
#: so a second constant would let its budget drift from the real deadline.
#: 20 s, not 15 (#3826): the OSWorld adapter waits ``timeout_s + 10`` for a
#: reply and then takes one liveness probe (worst case ~4 s) to answer a lost
#: reply instead of poisoning the session; at 15 that answer landed within
#: ~1 s of the RPC deadline. Raising it shrinks MAX_VERIFIER_WALL_S by the
#: same 5 s (195 -> 190) so the finish-call budget below still closes.
RUN_COMMAND_SLACK_S = 20.0
#: The probe's own timeout plus the adapter's per-command slack.
PROBE_ENVELOPE_S = float(INSPECT_MAX_TIMEOUT_S) + RUN_COMMAND_SLACK_S
FINISH_DELIVERY_MARGIN_S = 50.0
MAX_VERIFIER_WALL_S = (
    MCP_FINISH_DEADLINE_S - PROBE_ENVELOPE_S - CANCEL_JOIN_S - FINISH_DELIVERY_MARGIN_S
)


VerdictDisposition = Literal["accepted", "challenged"]


def verdict_disposition(status: str) -> VerdictDisposition:
    """What the session arm does with a verdict -- THE rule, read by the
    bridge's finish branch and written on every ``verifier_verdict`` row.

    Only ``PASS`` accepts the claim. ``FAIL`` and ``INCONCLUSIVE`` both become
    the claim's completion challenge (verifier-design.md: "FAIL/INCONCLUSIVE:
    send ``build_completion_challenge`` plus ... the findings"), so an
    INCONCLUSIVE is never a silent accept. Recorded explicitly because an
    ablation reading ``status`` alone cannot tell which way INCONCLUSIVE went.
    """

    return "accepted" if status == "PASS" else "challenged"


def verifier_envelope_s(wall_s: float) -> float:
    """The worst case one verification adds to a finish call, in seconds.

    What an embedder's fit check must compare against (``ActionBridge
    ._maybe_verify``), not the bare wall: the probe and the cancel join run
    outside the verifier's own wall cap. Conservative after the first round
    (the probe runs once per episode): over-skipping a check is recorded,
    overrunning the finish deadline loses the episode's finish.
    """

    return PROBE_ENVELOPE_S + float(wall_s) + CANCEL_JOIN_S


#: How much of one command's output reaches the verifier's context.
INSPECT_OUTPUT_CHARS = 12_000

ReadonlyMode = Literal["sandboxed", "audit"]


@dataclass(frozen=True)
class CompletionVerifierConfig:
    """The arm's switch. ``rounds=0`` (default) is today's arm, byte for byte.

    ``rounds`` is how many ``done`` claims are verified; a FAIL becomes the
    finish challenge, and once the rounds are spent the next claim takes the
    ordinary path (which, with the challenge unit already used, accepts it),
    so the verifier can never trap an episode in a loop. 0-2 only: past two,
    a cheap model arguing with its checker is cost without a measured gain.
    """

    rounds: int = 0
    # No independent ``max_requests``: the generic primitive derives it from
    # these (``VerifierLimits``) so the request cap always holds every
    # command plus the verdict -- the live t31v episode's 12-request cap
    # could not, and discarded a PASS.
    max_commands: int = 10
    local_tool_allowance: int = DEFAULT_LOCAL_TOOL_ALLOWANCE
    wall_s: float = 180.0

    def __post_init__(self) -> None:
        rounds = self.rounds
        if isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 2:
            raise ValueError("completion_verifier.rounds must be 0, 1 or 2")
        wall = self.wall_s
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(wall)
            or wall > MAX_VERIFIER_WALL_S
        ):
            raise ValueError(
                f"completion_verifier.wall_s must be finite and at most {MAX_VERIFIER_WALL_S:g} s "
                "(it runs inside the worker's 300 s finish call)"
            )
        # The rest of the validation is the generic primitive's own.
        self.limits()

    @property
    def enabled(self) -> bool:
        return self.rounds > 0

    def limits(self) -> VerifierLimits:
        return VerifierLimits(
            max_commands=self.max_commands,
            local_tool_allowance=self.local_tool_allowance,
            wall_s=self.wall_s,
        )

    def stamp(self) -> dict[str, Any]:
        """The comparability stamp: the arm a score came from."""

        return {"rounds": self.rounds, **self.limits().as_record()}


#: What ``sandboxed`` mode guarantees, recorded on the probe row so a record
#: says exactly what the label means (review R1).
SANDBOX_GUARANTEES: tuple[str, ...] = (
    "read-only-filesystem",
    "no-network",
    "private-pid-namespace",
    "private-ipc-namespace",
    "new-session",
    "x11-dbus-runtime-sockets-hidden",
)
#: Path-based desktop sockets live here: X11/ICE (``/tmp/.X11-unix/X0``), the
#: user's runtime dir (session bus, systemd --user, pulse, wayland) and the
#: system bus. Hidden only where they exist -- ``--tmpfs`` on a missing
#: directory would need to create it on the read-only root and fail -- and
#: ONLY these: ``/tmp`` itself stays visible because deliverables live there.
_SOCKET_DIRS = ("/tmp/.X11-unix", "/tmp/.ICE-unix", "/run/user", "/run/dbus")
_DESKTOP_ENV = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "DBUS_SESSION_BUS_ADDRESS",
    "DBUS_SYSTEM_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)
#: A shell prefix; the inner ``/bin/sh -c <command>`` follows it. ``env -u``
#: rather than ``bwrap --unsetenv`` so an older bwrap is not a probe failure.
_BWRAP = (
    "set --; for d in "
    + " ".join(_SOCKET_DIRS)
    + '; do [ -d "$d" ] && set -- "$@" --tmpfs "$d"; done; '
    'bwrap --ro-bind / / --dev /dev --proc /proc "$@" '
    "--unshare-net --unshare-pid --unshare-ipc --new-session --die-with-parent -- "
    "env " + " ".join(f"-u {name}" for name in _DESKTOP_ENV)
)


def sandboxed(command: str) -> str:
    """``command`` as it runs on the guest in ``sandboxed`` mode."""

    return f"{_BWRAP} /bin/sh -c {shlex.quote(command)}"


#: The probe proves ENFORCEMENT, not presence: inside the very wrapper every
#: command uses, a write must FAIL, the probe's own outer shell must be
#: unsignalable (PID namespace) and no desktop socket may be visible. Only
#: then is the token printed.
SANDBOX_TOKEN = "LOP_RO_SANDBOX_OK"
_PROBE_SCRIPT = (
    'p="$HOME/.lop-ro-probe-$$"; '
    'if : > "$p" 2>/dev/null; then rm -f "$p"; echo WRITABLE; exit 0; fi; '
    'if kill -0 "$1" 2>/dev/null; then echo PID-SHARED; exit 0; fi; '
    "for s in /tmp/.X11-unix/* /run/user/*/bus /run/dbus/*; do "
    'if [ -S "$s" ]; then echo SOCKET-VISIBLE; exit 0; fi; done; '
    f"echo {SANDBOX_TOKEN}"
)
SANDBOX_PROBE = (
    "command -v bwrap >/dev/null 2>&1 || exit 3; outer=$$; "
    f'{_BWRAP} /bin/sh -c {shlex.quote(_PROBE_SCRIPT)} probe "$outer"'
)

#: Lexical write-deny, applied in BOTH modes. It catches the obvious, not the
#: determined: the sandbox (or, in ``audit``, the changed-file sweep) is the
#: backstop. Tuned against plain reads that the first version refused, each
#: of which burned one of the verifier's ten commands (review M4):
#: ``open('a.xlsx','rb')`` (only an ``open()`` MODE argument counts now),
#: ``awk '$3 > 5'`` (redirection is matched outside quotes only) and
#: ``dpkg -l`` (package managers only on their mutating subcommands).
_WRITE_PATTERNS = re.compile(
    r"""
    \b(?:rm|rmdir|mv|cp|tee|dd|truncate|touch|mkdir|ln|chmod|chown|chgrp|install
          |shred|unlink|rsync|patch|kill|pkill|killall|reboot|shutdown|xdotool)\b
    | \bsed\s+(?:-\w*\s+)*-i | \bperl\s+(?:-\w*\s+)*-i
    | \b(?:apt|apt-get|aptitude)\s+(?:-\S+\s+)*(?:install|remove|purge|upgrade
          |dist-upgrade|full-upgrade|autoremove|update|reinstall)\b
    | \bdpkg\s+(?:-\S+\s+)*(?:-i|--install|-r|--remove|-P|--purge|--configure
          |--unpack|--set-selections)\b
    | \b(?:pip|pip3)\s+(?:-\S+\s+)*(?:install|uninstall)\b
    | \bnpm\s+(?:install|i|uninstall|ci|update|link)\b
    | \bsnap\s+(?:install|remove|refresh|revert)\b
    | \b(?:soffice|libreoffice)\b.*--convert-to
    | \bgit\s+(?:commit|checkout|reset|apply|stash|push|rm|mv|clean)\b
    | \bgsettings\s+(?:set|reset)\b | \bdconf\s+(?:write|reset|load)\b
    | \.save\s*\( | \.(?:write_text|write_bytes|unlink|rmdir|touch|mkdir|rename)\s*\(
    | \bos\.(?:remove|unlink|rename|replace|rmdir|removedirs|makedirs|mkdir|kill|system)\b
    | \bshutil\.(?:rmtree|move|copy\w*)\b
    | \bopen\s*\(\s*[^,()]+,\s*(?:mode\s*=\s*)?[rbt]?['"][rbtU]*[wax+]
    | \bmode\s*=\s*['"][rbtU]*[wax+]
    """,
    re.X,
)
#: A redirection into a file: not ``2>&1``/``>&2`` and not ``/dev/null``-ish.
_REDIRECT = re.compile(r"(?<![0-9&])>(?!&|\s*/dev/(?:null|stderr|stdout)\b)")
_QUOTED = re.compile(r"'[^']*'" + r'|"(?:\\.|[^"\\])*"')
#: ``sh -c '...'`` (and bash/dash/zsh): its quoted argument is itself a shell
#: line, so a redirection INSIDE it is still a redirection.
_SHELL_C = re.compile(r"\b(?:ba|da|z)?sh\s+(?:-\w+\s+)*-\w*c\w*\s*$")


def _redirects(command: str, depth: int = 0) -> bool:
    """Whether ``command`` redirects into a file, ignoring quoted text.

    A ``>`` inside quotes is data (``awk '$3 > 5'``, ``python3 -c "a > b"``),
    except in the quoted argument of ``sh -c``, which is checked as a shell
    line of its own (bounded recursion).
    """

    if _REDIRECT.search(_QUOTED.sub("''", command)):
        return True
    if depth >= 3:
        return False
    for match in _QUOTED.finditer(command):
        if _SHELL_C.search(command[: match.start()]):
            inner = match.group(0)[1:-1]
            if _redirects(inner, depth + 1):
                return True
    return False


def obvious_write(command: str) -> bool:
    return _WRITE_PATTERNS.search(command) is not None or _redirects(command)


RunCommand = Callable[[str, float], Awaitable[Any]]
RecordEvent = Callable[[str, dict[str, Any]], None]


class _NoTimeLeft(Exception):
    """A guest call that could not settle before the verification's deadline."""


class VerifierInspector:
    """The verifier's guest reads, gated on the bridge's verification window.

    ``window`` returns the current verification round (``None`` outside one);
    a call outside a window is refused before anything reaches the guest.
    No lock is taken anywhere here -- see the module docstring.

    A GUEST CALL IS NEVER CANCELLED (PR #3837 round 3, live episode
    ``t31v-t003``). ``run_command`` is the supervisor's ``_mutating_call``,
    which POISONS the adapter session on any unclassified failure -- and a
    ``CancelledError`` is one. The verifier child is cancelled by its caps
    while an ``inspect_command`` may be mid-RPC; in that episode the cancel
    landed there, the session was poisoned silently, and the worker's very
    next command failed ("adapter session is poisoned"), wedging the bridge
    with no score. So every guest call runs as its own task, awaited through
    ``asyncio.shield``: cancelling the child abandons the WAIT, while the call
    settles under its own RPC deadline. :meth:`drain` is the other half: the
    arm awaits every in-flight call before the worker resumes, so the guest
    is quiescent again and the next worker command is not racing a verifier
    read. #3836 does the same for the worker's own commands at the action
    server; this channel bypasses that path by design (module docstring).

    THE DEADLINE KEEPS THE ENVELOPE. :meth:`begin` takes the verification's
    hard deadline (its start + ``wall_s`` + ``CANCEL_JOIN_S``). A call is
    clipped so its RPC deadline (``timeout +`` :data:`RUN_COMMAND_SLACK_S`)
    lands before it, and refused with the verdict cue when no time is left,
    so draining never extends the finish call past :func:`verifier_envelope_s`.
    """

    def __init__(
        self,
        *,
        run_command: RunCommand,
        window: Callable[[], int | None],
        record_event: RecordEvent | None = None,
        job_id: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._run = run_command
        self._window = window
        self._record = record_event
        self._job_id = job_id
        self.mode: ReadonlyMode | None = None
        self._window_start: dict[int, str] = {}
        self._reported: set[str] = set()
        self._inflight: set[asyncio.Task[Any]] = set()
        self._deadline: float | None = None
        # Set from this round's verified observation, never from model paths.
        self.evidence: VerifierEvidence | None = None

    def begin(self, deadline: float) -> None:
        """Open one verification: guest calls must settle by ``deadline``
        (``time.monotonic()``)."""

        self._deadline = deadline

    async def _guest(self, command: str, timeout: float) -> Any:
        """One guest call: clipped to the deadline, tracked, never cancelled."""

        if self._deadline is not None:
            room = self._deadline - time.monotonic() - RUN_COMMAND_SLACK_S
            if room < 1.0:
                raise _NoTimeLeft
            timeout = min(timeout, room)
        operation = asyncio.ensure_future(self._run(command, timeout))
        self._inflight.add(operation)
        operation.add_done_callback(self._settled)
        # Shielded: a cancelled caller (the verifier child, cut by a cap)
        # stops waiting here; the RPC itself runs on to its own answer.
        return await asyncio.shield(operation)

    def _settled(self, operation: "asyncio.Task[Any]") -> None:
        self._inflight.discard(operation)
        # Retrieved so an abandoned call's failure is not an unhandled-task
        # warning; the caller that wanted it, if any, already has it.
        if not operation.cancelled():
            operation.exception()

    async def drain(self) -> int:
        """Wait for every in-flight guest call; return how many were awaited.

        Bounded by the deadline :meth:`begin` set (each call was clipped to
        settle before it). One still running at the deadline is LEFT running
        and recorded, never cancelled: cancelling it is exactly the poison
        this exists to prevent.

        Closes the verification: the deadline is cleared on the way out, so a
        later guest call (an ``ensure_mode`` re-probe after a cancelled first
        probe) is not clipped against a round that already ended (review
        round 3, N2) -- it gets its own unclipped bound, as the first probe.
        """

        try:
            pending = set(self._inflight)
            if not pending:
                return 0
            # Without a ``begin`` (a direct embedder), one call's longest RPC life.
            timeout = float(INSPECT_MAX_TIMEOUT_S) + RUN_COMMAND_SLACK_S
            if self._deadline is not None:
                timeout = max(0.0, self._deadline - time.monotonic())
            _done, still = await asyncio.wait(pending, timeout=timeout)
            if still:
                self._emit({"job_id": self._job_id(), "detached_in_flight": len(still)})
            return len(pending)
        finally:
            self._deadline = None

    async def ensure_mode(self) -> ReadonlyMode:
        """Probe once per episode; a failed probe is ``audit``, never a guess."""

        if self.mode is None:
            row: dict[str, Any] = {"probe": True, "command": SANDBOX_PROBE}
            probe_output: str | None = None
            try:
                result = await self._guest(SANDBOX_PROBE, float(INSPECT_MAX_TIMEOUT_S))
                # ``result`` is untyped (the guest seam returns Any), so a
                # declared ``str | None`` would not narrow; bind a plain str.
                stdout = str(result.stdout or "").strip()
                probe_output = stdout
                ok = result.exit_code == 0 and SANDBOX_TOKEN in stdout
                row["exit_code"] = result.exit_code
            except Exception as error:  # noqa: BLE001 -- an unreachable probe is no sandbox
                ok = False
                row["error"] = f"{type(error).__name__}: {error}"
            self.mode = "sandboxed" if ok else "audit"
            # The probe ran on the guest too, so it is on the record like any
            # verifier command, with the mode it decided.
            row["readonly_enforcement"] = self.mode
            if ok:
                row["guarantees"] = list(SANDBOX_GUARANTEES)
            elif probe_output is not None:
                # WHICH check failed (WRITABLE / PID-SHARED / SOCKET-VISIBLE),
                # so an ``audit`` episode says why it was not sandboxed.
                row["probe_output"] = probe_output[:200]
            self._emit(row)
        return self.mode

    async def _sweep(self, round_: int) -> list[str] | None:
        start = self._window_start.get(round_)
        if start is None:
            return None
        sweep = (
            f'find "$HOME" /tmp -xdev -type f -newermt @{start} 2>/dev/null '
            "| grep -v -E '/\\.cache/|/\\.lop-ro-probe' | head -n 50"
        )
        try:
            result = await self._guest(sweep, float(INSPECT_MAX_TIMEOUT_S))
        except Exception:  # noqa: BLE001 -- recorded as an unknown, never "clean"
            return None
        changed = [line for line in (result.stdout or "").splitlines() if line.strip()]
        fresh = [path for path in changed if path not in self._reported]
        self._reported.update(fresh)
        return fresh

    async def inspect(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        from local_operator.evaluation.adapters.api import MAX_COMMAND_LENGTH

        round_ = self._window()
        if round_ is None:
            return "inspect_command is only available while a completion check is running.", True
        command = arguments.get("command")
        timeout = arguments.get("timeout_s", INSPECT_DEFAULT_TIMEOUT_S)
        if (
            set(arguments) - {"command", "timeout_s"}
            or not isinstance(command, str)
            or not command.strip()
            or type(timeout) is not int
            or not 1 <= timeout <= INSPECT_MAX_TIMEOUT_S
        ):
            return (
                "Invalid arguments: supply a command and an integer timeout_s "
                f"between 1 and {INSPECT_MAX_TIMEOUT_S}.",
                True,
            )
        mode = await self.ensure_mode()
        row: dict[str, Any] = {
            "job_id": self._job_id(),
            "round": round_,
            "command": command,
            "readonly_enforcement": mode,
        }
        # Both modes (review R1): in ``sandboxed`` the wrapper is the real
        # gate and this only saves a command; in ``audit`` it is the only one.
        if obvious_write(command):
            row.update(refused="write-pattern", exit_code=None, mutation_detected=None)
            self._emit(row)
            return (
                "Refused: that command looks like it writes or changes state. "
                "Only read the deliverable.",
                True,
            )
        if mode == "audit":
            if round_ not in self._window_start:
                try:
                    stamp = await self._guest("date +%s", 10.0)
                    self._window_start[round_] = (stamp.stdout or "").strip() or "0"
                except Exception:  # noqa: BLE001
                    self._window_start[round_] = "0"
            wire = command
        else:
            wire = sandboxed(command)
        if len(wire) > MAX_COMMAND_LENGTH:
            return "Refused: the command is too long.", True
        try:
            result = await self._guest(wire, float(timeout))
        except _NoTimeLeft:
            row.update(refused="deadline", exit_code=None, mutation_detected=None)
            self._emit(row)
            return (
                "No time left for another command: give your VERDICT now from what you "
                "have already seen, ending with the VERDICT line.",
                True,
            )
        except Exception as error:  # noqa: BLE001 -- the verifier is told, the row records
            row.update(exit_code=None, error=f"{type(error).__name__}: {error}")
            self._emit(row)
            return f"The target could not run the command ({type(error).__name__}).", True
        row.update(exit_code=result.exit_code, timed_out=result.timed_out)
        if mode == "audit":
            changed = await self._sweep(round_)
            row["changed_files"] = changed
            row["mutation_detected"] = None if changed is None else bool(changed)
        else:
            # Nothing was checked after the command: the sandbox's guarantees
            # are on the probe row, and "not measured" is not "no mutation".
            row["mutation_detected"] = None
        self._emit(row)
        text = (
            f"exit_code: {result.exit_code if result.exit_code is not None else 'none'}\n"
            f"timed_out: {str(result.timed_out).lower()}\n"
        )
        if result.error is not None:
            text += f"error: {result.error}\n"
        for stream in ("stdout", "stderr"):
            body = getattr(result, stream) or ""
            if len(body) > INSPECT_OUTPUT_CHARS:
                body = body[:INSPECT_OUTPUT_CHARS] + f"\n[TRUNCATED: {len(body)} chars]"
            text += f"{stream}:\n{body}\n"
        return text, False

    def _emit(self, row: dict[str, Any]) -> None:
        if self._record is not None:
            self._record("verifier_command", dict(row))

    def tool(self) -> Any:
        """The in-process ``AgentTool`` handed to the verifier child only."""

        from local_operator.harness.types import AgentTool, TextContent, ToolResult

        async def execute(  # type: ignore[no-untyped-def]
            tool_call_id, args, signal=None, on_update=None, context=None
        ):
            arguments = dict(args or {})
            if "artifact_id" in arguments:
                item = (
                    next(
                        (
                            item
                            for item in self.evidence.items
                            if item.evidence_id == arguments["artifact_id"]
                        ),
                        None,
                    )
                    if self.evidence is not None
                    else None
                )
                if (
                    self._window() is None
                    or set(arguments) != {"artifact_id"}
                    or item is None
                    or (self._deadline is not None and time.monotonic() >= self._deadline)
                ):
                    return ToolResult(
                        tool_call_id=tool_call_id,
                        tool_name="inspect_command",
                        content=[
                            TextContent(
                                text="Unavailable artifact ID or invalid arguments. Use only a "
                                "registered evidence ID, without command/timeout_s."
                            )
                        ],
                        is_error=True,
                    )
                self._emit(
                    {
                        "job_id": self._job_id(),
                        "round": self._window(),
                        "artifact_id": item.evidence_id,
                    }
                )
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name="inspect_command",
                    content=[TextContent(text=f"Evidence ID: {item.evidence_id}"), item.content],
                )
            text, is_error = await self.inspect(arguments)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="inspect_command",
                content=[TextContent(text=text)],
                is_error=is_error,
            )

        return AgentTool(
            name="inspect_command",
            label="Inspect target",
            description=(
                "Run a READ-ONLY shell command on the machine where the task was done "
                "(cat, ls, python3 -c to read a file, pdftotext, unzip -p ...). "
                f"Never change anything. timeout_s 1-{INSPECT_MAX_TIMEOUT_S}. "
                "Alternatively supply only artifact_id to inspect registered typed evidence; "
                "never a path or URL. The same inspection budget applies."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command."},
                    "artifact_id": {
                        "type": "string",
                        "description": "A registered evidence ID; mutually exclusive with "
                        "command and timeout_s.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": INSPECT_MAX_TIMEOUT_S,
                    },
                },
                "additionalProperties": False,
            },
            approval_tier="read",
            execute=execute,
        )
