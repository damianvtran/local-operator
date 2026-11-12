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

* ``sandboxed`` -- a probe at the first inspection proved ``bwrap`` can run a
  command under a read-only bind of ``/`` with no network, and that a write
  inside it FAILS. Every command is then wrapped the same way.
* ``audit`` -- no working sandbox. Obvious writes are refused lexically, and
  every command is followed by a sweep for files under ``$HOME`` and ``/tmp``
  modified since the window opened, recorded as ``changed_files`` /
  ``mutation_detected``. This DETECTS mutation; it does not prevent it, and
  nothing here calls it read-only. Desktop applications write in the
  background too (autosave, recent-files), so a hit is a fact about the
  window, not proof the verifier wrote.

``/tmp`` is deliberately NOT replaced by a tmpfs in the sandbox (the design
note's ``--tmpfs /tmp``): task deliverables live there often enough that
hiding them would make the check blind exactly where it must look.
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from local_operator.harness.verifier import VerifierLimits

#: The MCP client's default per-call deadline (``mcp/manager.py``
#: ``DEFAULT_MCP_TIMEOUT_MS`` = 300 s) bounds the worker's finish call, and
#: the verifier runs INSIDE that call. 240 s leaves the 10 s cancel join in
#: ``verify_completion`` and the challenge's own delivery inside it.
MAX_VERIFIER_WALL_S = 240.0

#: Per-command bound for the verifier's guest reads. A read that needs more
#: than this is not a read; the worker's own commands keep their 300 s.
INSPECT_MAX_TIMEOUT_S = 30
INSPECT_DEFAULT_TIMEOUT_S = 15
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
    max_requests: int = 12
    max_commands: int = 10
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
            max_requests=self.max_requests, max_commands=self.max_commands, wall_s=self.wall_s
        )

    def stamp(self) -> dict[str, Any]:
        """The comparability stamp: the arm a score came from."""

        return {"rounds": self.rounds, **self.limits().as_record()}


#: The probe proves ENFORCEMENT, not presence: it attempts a write inside the
#: sandbox and only a refused write prints the token.
SANDBOX_TOKEN = "LOP_RO_SANDBOX_OK"
_BWRAP = "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-net --die-with-parent --"
SANDBOX_PROBE = (
    "command -v bwrap >/dev/null 2>&1 || exit 3; "
    f"{_BWRAP} /bin/sh -c "
    + shlex.quote(
        'p="$HOME/.lop-ro-probe-$$"; if : > "$p" 2>/dev/null; then rm -f "$p"; echo WRITABLE; '
        f"else echo {SANDBOX_TOKEN}; fi"
    )
)

#: Lexical write-deny for ``audit`` mode. It catches the obvious, not the
#: determined: the changed-file sweep is the backstop, and the mode's name
#: says so.
_WRITE_PATTERNS = re.compile(
    r"""
    (?<![0-9&])>(?!&|\s*/dev/(?:null|stderr|stdout)\b)   # redirection into a file
    | \b(?:rm|rmdir|mv|cp|tee|dd|truncate|touch|mkdir|ln|chmod|chown|chgrp|install
          |shred|unlink|rsync|patch|kill|pkill|killall|reboot|shutdown|xdotool)\b
    | \bsed\s+(?:-\w*\s+)*-i | \bperl\s+(?:-\w*\s+)*-i
    | \b(?:apt|apt-get|dpkg|pip|pip3|npm|snap)\b
    | \b(?:soffice|libreoffice)\b.*--convert-to
    | \bgit\s+(?:commit|checkout|reset|apply|stash|push|rm|mv|clean)\b
    | \bgsettings\s+set\b | \bdconf\s+(?:write|reset|load)\b
    | \.save\s*\( | \bopen\s*\([^)]*['"][wax+]
    """,
    re.X,
)


def obvious_write(command: str) -> bool:
    return _WRITE_PATTERNS.search(command) is not None


RunCommand = Callable[[str, float], Awaitable[Any]]
RecordEvent = Callable[[str, dict[str, Any]], None]


class VerifierInspector:
    """The verifier's guest reads, gated on the bridge's verification window.

    ``window`` returns the current verification round (``None`` outside one);
    a call outside a window is refused before anything reaches the guest.
    No lock is taken anywhere here -- see the module docstring.
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

    async def ensure_mode(self) -> ReadonlyMode:
        """Probe once per episode; a failed probe is ``audit``, never a guess."""

        if self.mode is None:
            row: dict[str, Any] = {"probe": True, "command": SANDBOX_PROBE}
            try:
                result = await self._run(SANDBOX_PROBE, float(INSPECT_MAX_TIMEOUT_S))
                ok = result.exit_code == 0 and SANDBOX_TOKEN in (result.stdout or "")
                row["exit_code"] = result.exit_code
            except Exception as error:  # noqa: BLE001 -- an unreachable probe is no sandbox
                ok = False
                row["error"] = f"{type(error).__name__}: {error}"
            self.mode = "sandboxed" if ok else "audit"
            # The probe ran on the guest too, so it is on the record like any
            # verifier command, with the mode it decided.
            row["readonly_enforcement"] = self.mode
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
            result = await self._run(sweep, float(INSPECT_MAX_TIMEOUT_S))
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
        if mode == "audit":
            if obvious_write(command):
                row.update(refused="write-pattern", exit_code=None, mutation_detected=None)
                self._emit(row)
                return (
                    "Refused: that command looks like it writes or changes state. "
                    "Only read the deliverable.",
                    True,
                )
            if round_ not in self._window_start:
                try:
                    stamp = await self._run("date +%s", 10.0)
                    self._window_start[round_] = (stamp.stdout or "").strip() or "0"
                except Exception:  # noqa: BLE001
                    self._window_start[round_] = "0"
            wire = command
        else:
            wire = f"{_BWRAP} /bin/sh -c {shlex.quote(command)}"
        if len(wire) > MAX_COMMAND_LENGTH:
            return "Refused: the command is too long.", True
        try:
            result = await self._run(wire, float(timeout))
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
            row["mutation_detected"] = False
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
            text, is_error = await self.inspect(dict(args or {}))
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
                f"Never change anything. timeout_s 1-{INSPECT_MAX_TIMEOUT_S}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command."},
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": INSPECT_MAX_TIMEOUT_S,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            approval_tier="read",
            execute=execute,
        )
