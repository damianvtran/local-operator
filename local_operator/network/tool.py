"""The ``network`` agent tool — the agent-facing half of R19 (mesh).

The agent drives the mesh through ONE surface: the authenticated ``lop network``
CLI, invoked with ``--json`` and parsed here. That is the design's decision
(``mesh-transport-identity.md`` §12.5, ``mesh-ui.md`` §3.2) and it is what keeps
the config store, the identity key and the relay's control key in one writer —
a tool that opened the store itself would be a second one.

Three rules shape the code below, and they are the reason it is not a thin
subprocess wrapper:

* **The token and the SAS never appear in a result.** ``invite`` returns the
  token's *path*; ``join`` is never completed from here. The CLI's stdout is
  PARSED, never passed through: a raw blob would put a token in the transcript
  the moment a command printed one (``mesh-ui.md`` §3.2), and every payload is
  scrubbed of secret-shaped keys on the way out as a second line of defence.
* **No confirmation, ever.** Pairing is two humans reading a code off each
  other's screens; ``panic``/``disconnect``/``member rm`` mutate the trust state
  of every device in a network. The CLI has no ``--yes``/``--confirm`` to pass
  and the argv table below cannot invent one, so a refusal comes back as the
  CLI's own sentence rather than as an action taken on the user's behalf.
* **Creation cannot be gated.** ``init`` is how the first network comes to
  exist, so the tool exists in every session (``build_network_tool`` never
  returns ``None`` — ``mesh-ui.md`` §3.2's ladder discussion).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    spill_truncate,
)

_TOOL = "network"

#: The CLI's own tier split (``mesh-ui.md`` §3.2): the reads never prompt, and
#: every op that changes a record, the trust state or the epoch does.
READ_ACTIONS = frozenset({"status", "ls", "show", "peers", "log", "doctor"})
WRITE_ACTIONS = frozenset({"init", "invite", "join", "member_rm", "disconnect", "panic"})

#: Local reads answer in well under a second; ``doctor`` dials endpoints. The
#: bound exists so a wedged subprocess can never hold a turn open.
_DEFAULT_TIMEOUT_S = 30.0
#: ``join`` performs a real handshake against the far device, which is bounded by
#: that relay's own timers, not by ours.
_JOIN_TIMEOUT_S = 120.0

#: A refusal sentence is one line; a traceback is not, and neither belongs in a
#: result whole.
_STDERR_CAP = 500

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Key-name fragments dropped from any payload before it reaches a result. The
#: CLI does not print key material today, and a tool result is the most-copied
#: text in the system — so the invariant is enforced here rather than trusted to
#: a future field name.
_SECRET_KEY_MARKERS = ("secret", "token", "password", "key", "material")

_ROW_CAP = 40

#: The tool's action vocabulary, named ONCE. ``NetworkParams.action`` and the argv
#: builder both read it, and a caller that wants to hold a map of actions (the
#: tests, and any future dispatcher) can annotate that map with this rather than
#: restating the list — a second spelling of it is how a valid action stops being
#: type-checked while still working.
NetworkAction = Literal[
    "status",
    "init",
    "invite",
    "join",
    "ls",
    "show",
    "peers",
    "member_rm",
    "disconnect",
    "panic",
    "log",
    "doctor",
]


class NetworkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: NetworkAction = Field(
        description=(
            "Which network operation to run. init/invite/join/member_rm/"
            "disconnect/panic all need a human or change what other devices "
            "trust; status/ls/show/peers/log/doctor only read."
        )
    )
    network: str = Field(
        default="",
        description=(
            "A network name or id. Required by show and member_rm; for init it "
            "is the new network's name. Optional for invite/disconnect/panic, "
            "where omitting it means 'the one network this device is in'. "
            "`log` reads every network this device is in."
        ),
    )
    token: str = Field(
        default="",
        description=(
            "For join: the invite token, or '@<path>' to a token file. "
            "Never echoed back, and the tool writes it to a private file so it "
            "does not land in this machine's process list."
        ),
    )
    role: Literal["read", "drive", "admin"] = Field(
        default="read",
        description=(
            "For invite: what the joining device may do. read = see sessions, "
            "drive = prompt/steer/stop them, admin = also manage membership. "
            "State it explicitly when the user asks for more than viewing."
        ),
    )
    device: str = Field(
        default="",
        description="For member_rm: the device id to revoke (see `lop network show`).",
    )
    since: str = Field(
        default="",
        description="For log: how far back to read, as a duration such as 15m or 2h.",
    )


def _cli_argv() -> list[str]:
    """The argv prefix that runs THIS build's CLI.

    ``sys.executable -m local_operator.cli``, not ``shutil.which("lop")``: the
    ``lop`` on PATH may be a different build (a global install, a generation
    pointer) than the code whose JSON shapes are parsed below, and the two could
    disagree about a flag or a field. The store is reached through the config
    directory either way, so pointing at this build costs nothing and removes the
    skew. The same spelling ``local_operator/update.py`` uses for its child
    processes.
    """
    return [sys.executable, "-m", "local_operator.cli"]


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _clean(text: str) -> str:
    """One bounded, ANSI-free, whitespace-collapsed message."""
    flat = " ".join(_strip_ansi(text).split())
    if len(flat) <= _STDERR_CAP:
        return flat
    return flat[: _STDERR_CAP - 1].rstrip() + "…"


def _scrub(value: Any) -> Any:
    """Drop secret-shaped keys anywhere in a payload, at every depth."""
    if isinstance(value, dict):
        return {
            str(key): _scrub(item)
            for key, item in value.items()
            if not any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS)
        }
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def _argv_for(params: NetworkParams) -> tuple[list[str], str]:
    """``(argv, error)`` — the CLI's argv for this call, or the sentence saying
    what the call is missing.

    Every branch is explicit and every flag is spelled here rather than
    constructed from user text, which is what makes "the tool can never pass a
    confirmation flag" a property of the code instead of a promise. ``--json`` is
    added once, at the end, for the same reason the CLI refuses ``--print``
    beside it: one of the two must win, and the agent's path is the JSON one.
    """
    action = params.action
    network = params.network.strip()

    if action == "status":
        argv = ["network", "status"]
    elif action == "ls":
        argv = ["network", "ls"]
    elif action == "show":
        if not network:
            return [], "action='show' needs 'network' (a name or id)."
        argv = ["network", "show", network]
    elif action == "peers":
        argv = ["network", "peers"]
    elif action == "doctor":
        argv = ["network", "doctor"]
    elif action == "log":
        argv = ["network", "log"]
        if params.since.strip():
            argv += ["--since", params.since.strip()]
    elif action == "init":
        if not network:
            return [], "action='init' needs 'network': the name of the new network."
        argv = ["network", "init", network]
    elif action == "invite":
        argv = ["network", "invite", "--role", params.role]
        if network:
            argv += ["--network", network]
    elif action == "join":
        if not params.token.strip():
            return [], (
                "action='join' needs 'token' (the invite token, or '@<path>' to "
                "the token file the other device minted)."
            )
        argv = ["network", "join", params.token.strip()]
    elif action == "member_rm":
        device = params.device.strip()
        if not network or not device:
            return [], "action='member_rm' needs 'network' and 'device'."
        argv = ["network", "member", "rm", network, device]
    elif action == "disconnect":
        argv = ["network", "disconnect"] + ([network] if network else [])
    elif action == "panic":
        argv = ["network", "panic"] + ([network] if network else [])
    else:  # pragma: no cover — the Literal above is the gate
        return [], f"{action!r} is not a network action."

    return argv + ["--json"], ""


def _describe_approval(args: dict[str, Any], _cwd: str) -> str:
    """What the approval prompt says (``mesh-ui.md`` §3.2's tier split)."""
    action = str(args.get("action") or "")
    target = str(args.get("network") or "").strip()
    detail = {
        "init": "create a network on this device",
        "invite": "mint a single-use invite token",
        "join": "join a network from an invite (needs a person at the terminal)",
        "member_rm": "revoke a member and rotate the secret",
        "disconnect": "leave the network and delete its local secret",
        "panic": "raise a panic: rotate the secret for every member",
    }.get(action, action)
    return f"{detail}{f' ({target})' if target else ''}"


async def _run_cli(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """``(returncode, stdout, stderr)`` from one CLI invocation.

    The child's stdin is /dev/null on purpose: the CLI's own human step (``join``)
    refuses a non-TTY rather than reading from whatever the session's stdin
    happens to be. A timed-out child is killed AND reaped — a killed-but-unwaited
    subprocess is how a tool leaves a process behind on the fleet.
    """
    process = await asyncio.create_subprocess_exec(
        *_cli_argv(),
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except (asyncio.TimeoutError, TimeoutError):
        process.kill()
        await process.wait()
        raise
    return (
        int(process.returncode or 0),
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _write_token_file(token: str) -> str:
    """Write a pasted token to a 0600 file and return its path.

    The CLI accepts a token in argv, and argv is readable by every process on the
    machine (``ps``). Since it also accepts ``@path``, a token handed to the tool
    as text goes to a private file first — the same reason ``lop network invite``
    refuses to print one and ``--json`` refuses to carry it.
    """
    handle, path = tempfile.mkstemp(prefix="lop-join-", suffix=".invite")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(token)
    os.chmod(path, 0o600)
    return path


def _networks_block(payload: dict[str, Any]) -> list[str]:
    from local_operator.network.relay import membership_marker

    rows = payload.get("networks") or []
    if not rows:
        return ["no networks on this device"]
    # THE COUNT TRAVELS WITH WHAT IT RESTS ON, through the same function the CLI's `ls`
    # uses: an agent reading this digest and an operator reading the terminal are
    # looking at one rendering of one fact, so neither can be told the member set was
    # checked when it was not (QA round 3, Q-R2-1).
    return [
        f"{row.get('name')}  {row.get('network_id')}  epoch {row.get('epoch')}  "
        f"{row.get('role')}  {row.get('members')} member(s)  {row.get('trust')}"
        + (f"  {row.get('links')} link(s)" if row.get("links") else "")
        + membership_marker(row)
        for row in rows
    ]


def _render(action: str, payload: dict[str, Any]) -> list[str]:
    """The human-readable digest of one parsed payload.

    Deliberately a summary: the parsed fields ride in ``details`` (which never
    reaches a provider), so the model reads what it needs and no field is
    silently the whole command output.
    """
    if action == "status":
        from local_operator.network.relay import audit_status_words

        relay = payload.get("relay") or {}
        # THE SAME THREE CASES THE CLI'S BLOCK HAS, from the same fields (Q-R3-4):
        # ``relay_running`` is a process, ``relay_answering`` is this probe being
        # answered, and only the pair together can describe a relay that is up and
        # silent. Reading the pid alone said "not running" about a running process —
        # and now that the audit line below says "the relay is not answering", the
        # block would have contradicted itself in two adjacent lines.
        if payload.get("relay_running"):
            pid = relay.get("pid") or (payload.get("record") or {}).get("pid")
            relay_line = (
                f"running, pid {pid}"
                if payload.get("relay_answering")
                else (
                    f"running (pid {pid}), NOT answering its control socket "
                    f"(state: {payload.get('relay_state')})"
                )
            )
        else:
            relay_line = "not running"
        lines = [
            f"installed: {'yes' if payload.get('installed') else 'no'}"
            + ("" if payload.get("supported") else "  (no launchd on this platform)"),
            f"identity:  {'present' if payload.get('identity_present') else 'missing'}",
            f"relay:     {relay_line}",
        ]
        # THE SAME WORDS THE CLI AND THE PANEL PRINT, through the one renderer (design
        # round 3, D40), at this register's own column: every value here starts at cell
        # 11 (`installed: `, `identity:  `, `log:       `). It matters more here than on
        # either: this digest is where "why is my session stuck" actually arrives, and
        # the payload's audit fields ride in ``details``, which never reaches a provider
        # — so without this line the model cannot know the distinction exists.
        audit_words = audit_status_words(payload)
        if audit_words:
            lines.append(f"audit:     {audit_words}")
        lines.append(f"log:       {payload.get('log')}")
        return lines + ["  " + line for line in _networks_block(payload)]
    if action == "ls":
        return _networks_block(payload)
    if action == "show":
        from local_operator.network.relay import membership_lines

        lines = [
            f"{payload.get('name')}  {payload.get('network_id')}  epoch "
            f"{payload.get('epoch')}  trust {payload.get('trust')}  "
            f"role {payload.get('role')}"
        ]
        # THIS DEVICE'S OWN STANDING, before the rows: a removed or refused device's
        # member table looks perfectly healthy, and the sentence is the only thing on
        # this surface that says otherwise (QA round 3, Q-R3-2).
        lines.extend(membership_lines(payload))
        for member in payload.get("members_detail") or []:
            mark = "active" if member.get("active") else "REMOVED"
            suspect = "  [suspect]" if member.get("suspect") else ""
            lines.append(
                f"  {mark:7} {member.get('device_id')}  {member.get('name')}  "
                f"{member.get('role')}  {', '.join(member.get('capabilities') or [])}{suspect}"
            )
        for invite in payload.get("invites") or []:
            lines.append(
                f"  invite {invite.get('invite_id')}  {invite.get('role')}  "
                f"{invite.get('state')}"
            )
        return lines
    if action == "peers":
        rows = payload.get("peers") or []
        if not rows:
            return ["no peers: this device is the only member of its networks"]
        # THE WORDS, NOT THE TOKEN (round 11, R11-0; QA round 25, Q-R25-1). This
        # branch printed the row's raw ``reason`` — a stage word, two endpoint
        # addresses and a Python class name — beside the 34-character device id, on a
        # digest the model reads the way a person reads `lop network peers`. The
        # ``doctor`` branch of this same function was already routed through its own
        # gloss for exactly that reason, and the asymmetry inside one function is what
        # round 11 filed: a sweep that counted the CLI and the TUI and stopped one
        # branch short of the agent's own digest. The gloss is the SHARED one (the
        # member table ``_peer_line`` reads, so a peer is described in one voice on
        # every surface), and the id is replaced by the NAME every other surface
        # addresses a peer by (``resume.UNNAMED_DEVICE`` when it has none, design
        # round 1 D8). Both the token and the id stay in this tool's ``details``, which
        # is the machine register.
        from local_operator.resume import UNNAMED_DEVICE, peer_reason_words

        return [
            f"{'reachable' if row.get('reachable') else 'unreachable':11} "
            f"{str(row.get('name') or '').strip() or UNNAMED_DEVICE}"
            + (
                ""
                if row.get("reachable")
                else f"  {peer_reason_words(str(row.get('reason') or ''))}"
            )
            for row in rows
        ]
    if action == "doctor":
        # THE SAME WORDS THE CLI SHOWS ITS OWN READER, through the same renderer
        # (round 24, Q-R24-2): every string in that field is written for a person
        # here, and the raw codes stay in ``checks[].detail``, which is what this
        # tool's ``details`` carries into the machine register. A doctor row is about
        # one address, which is why this is not ``peer_reason_words``.
        from local_operator.resume import doctor_detail_words

        lines = [
            f"{'ok  ' if check.get('ok') else 'FAIL'} {check.get('check')} "
            f"{check.get('device_id', '')} {check.get('endpoint', '')} "
            f"{doctor_detail_words(str(check.get('detail', '')))}".rstrip()
            for check in payload.get("checks") or []
        ]
        if not lines:
            lines = ["nothing to check: no networks, or no other members yet"]
        if not payload.get("identity_present", True):
            lines.append(
                "this device has no mesh identity: run `lop network init`, or "
                "re-pair with a new invite"
            )
        return lines
    if action == "log":
        records = payload.get("records") or []
        if not records:
            return ["no audit records in that window"]
        return [
            f"{record.get('ts_iso')}  {record.get('event')}  {record.get('outcome')}  "
            f"{record.get('network_id', '')}"
            for record in records[:_ROW_CAP]
        ] + ([f"… {len(records) - _ROW_CAP} more record(s)"] if len(records) > _ROW_CAP else [])
    if action == "init":
        return [
            f"created network {payload.get('name')} ({payload.get('network_id')}), "
            f"epoch {payload.get('epoch')}",
            f"this device is {payload.get('device_id')}",
            payload.get("relay") or "the relay was not started (--no-start)",
        ]
    if action == "invite":
        lines = [
            f"invite {payload.get('invite_id')} for role {payload.get('role')}, "
            f"expires in {int(float(payload.get('expires_in_s') or 0))}s",
            f"token written to {payload.get('path')} — single use",
            "hand that FILE to the other device out of band; do not paste the token "
            "into a chat, a commit or a ticket",
        ]
        if payload.get("relay"):
            lines.append(str(payload["relay"]))
        return lines
    if action == "join":
        lines = [
            f"joined {payload.get('name')} ({payload.get('network_id')}) at epoch "
            f"{payload.get('epoch')}, role {payload.get('role')}, "
            f"{payload.get('members')} member(s)",
            f"fingerprint {payload.get('fingerprint')}",
        ]
        return lines
    if action == "member_rm":
        lines = [
            f"removed {payload.get('removed')}; epoch is now {payload.get('epoch')}",
        ]
        if payload.get("queued"):
            lines.append(f"queued for {payload['queued']} offline peer(s)")
        if payload.get("relay"):
            lines.append(str(payload["relay"]))
        return lines
    if action == "disconnect":
        return [
            f"left the network ({payload.get('network_id')}); "
            f"{payload.get('reachable_peers', 0)} peer(s) notified, local secret "
            f"deleted, audit trail kept",
        ]
    if action == "panic":
        rotated = (
            ", every other device is told to stop trusting the network"
            if payload.get("rotated")
            else ""
        )
        return [
            f"panic raised: epoch {payload.get('epoch')}{rotated}",
            "each device must be re-admitted with `lop network trust <network> --active`",
        ]
    return [json.dumps(payload, sort_keys=True)]


def _hint(action: str) -> str:
    """The one sentence an agent needs after a refusal, where a generic failure
    message would leave it to guess (and to retry)."""
    if action == "join":
        return (
            "Pairing needs a person at a terminal on the joining device: have the "
            "user run `lop network join @<token-file>` themselves and read the code "
            "back from the other device's screen. No flag completes this for them, "
            "by design."
        )
    if action in ("panic", "disconnect", "member_rm"):
        return (
            "This one needs an explicit instruction naming the network and the "
            "action; it is never a retry after a failed command."
        )
    if action == "invite":
        return "The token is written to a file; the result carries its path, never the token."
    return ""


@_guard(_TOOL)
async def execute_network(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Run one ``lop network`` action and report what the CLI said."""
    try:
        params = NetworkParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, _TOOL, exc)

    argv, problem = _argv_for(params)
    if problem:
        return _error(tool_call_id, _TOOL, problem)

    token_path: str | None = None
    if params.action == "join":
        token_arg = params.token.strip()
        if not token_arg.startswith("@"):
            token_path = _write_token_file(token_arg)
            token_arg = f"@{token_path}"
        # Rebuilt rather than patched in place: the token is the ONE positional
        # this action takes, and a stray argv entry after it would be parsed as
        # one rather than reported.
        argv = ["network", "join", token_arg, "--json"]

    timeout = _JOIN_TIMEOUT_S if params.action == "join" else _DEFAULT_TIMEOUT_S
    try:
        code, stdout, stderr = await _run_cli(argv, timeout)
    except (asyncio.TimeoutError, TimeoutError):
        return _error(
            tool_call_id,
            _TOOL,
            f"`lop network {params.action}` did not finish within {int(timeout)}s.",
        )
    finally:
        if token_path:
            # The invite is single-use, so a leaked copy is not a live credential
            # forever — but it is one until it is redeemed or expires, and this
            # file is the one place the tool holds it.
            Path(token_path).unlink(missing_ok=True)

    payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(stdout) if stdout.strip() else None
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        # No passthrough: a refusal is a sentence on stderr, and stdout that is
        # not JSON must not be echoed into a transcript.
        detail = _clean(stderr) or "no output"
        hint = _hint(params.action)
        return _error(
            tool_call_id,
            _TOOL,
            f"`lop network {params.action}` exited {code}: {detail}"
            + (f"\n{hint}" if hint else ""),
        )

    scrubbed = _scrub(payload)
    if code != 0 or scrubbed.get("ok") is False:
        # ``code`` + ``message`` IS THE REFUSAL FAMILY'S SHAPE, so it is read
        # before ``error`` (which only the install/uninstall diagnostics still
        # use) and before stderr, which in ``--json`` mode is empty BY DESIGN —
        # reading it first is how a refusal became the bare text "exited 1", which
        # tells an agent neither what happened nor what to do (QA round 1, F-6).
        reason = str(scrubbed.get("code") or "")
        message = str(
            scrubbed.get("message") or scrubbed.get("error") or _clean(stderr) or f"exited {code}"
        )
        hint = _hint(params.action)
        return _error(
            tool_call_id,
            _TOOL,
            (f"{reason}: {message}" if reason else message) + (f"\n{hint}" if hint else ""),
        )

    body = "\n".join(_render(params.action, scrubbed))
    text, spill = spill_truncate(body, _TOOL, context)
    return _text(tool_call_id, _TOOL, text, details=spill or {"network": scrubbed})


def build_network_tool(context: ToolContext) -> AgentTool | None:
    """Always-on builder: an UNCONDITIONAL entry in the createIf table.

    It never returns ``None``, and that is a decision rather than an omission.
    A gate would have to be a prerequisite whose absence makes the tool unusable
    (``secret`` with no store), and there is no such prerequisite here: ``init``
    is how the first network comes to exist, so gating on "a network exists"
    would strip the tool from exactly the session that has to create one. The
    only condition that would justify a gate — no CLI reachable at all — cannot
    obtain inside a session that already has ``bash``, and ``bash`` would be the
    thing that reports it.
    """
    return AgentTool(
        name=_TOOL,
        label="Mesh network",
        description=(
            "Read and drive a lop mesh network from this device: pairing, peers, "
            "membership, the audit log, and the incident controls. Pairing and the "
            "incident controls need a human — this tool reports what the CLI "
            "refused and why. Sessions on other devices are not reachable in this "
            "build."
        ),
        parameters=NetworkParams.model_json_schema(),
        # Write tier because the highest op needs it (``panic`` rotates a secret
        # for every member); the reads downgrade per call so looking at the
        # network never prompts.
        approval_tier="write",
        call_approval_tier=lambda args: (
            "read" if str(args.get("action") or "") in READ_ACTIONS else "write"
        ),
        # EXCLUSIVE for the same reason ``hub`` is: the design asks for it on
        # panic/disconnect/member_rm, where two racing calls would mutate the
        # epoch or the trust state into a shape nobody designed. A static field
        # cannot vary per call, so the whole tool takes the exclusive slot — the
        # read ops are bounded by their subprocess anyway.
        concurrency="exclusive",
        # Likewise per-action in the design (reads and `join` interruptible): a
        # static field cannot vary, and aborting a `member_rm` mid-rotation is
        # worse than a delayed read, so the tool takes the safe side.
        interruptible=False,
        describe_approval=_describe_approval,
        execute=execute_network,
    )
