"""The ``network`` agent tool — the agent-facing half of R19 (mesh).

The agent drives the mesh through ONE surface: the authenticated ``lop network``
CLI, invoked with ``--json`` and parsed here. That is the design's decision
(``mesh-transport-identity.md`` §12.5, ``mesh-ui.md`` §3.2) and it is what keeps
the config store, the identity key and the relay's control key in one writer —
a tool that opened the store itself would be a second one.

Three rules shape the code below, and they are the reason it is not a thin
subprocess wrapper:

* **The invite token never appears in a result, and the SAS only as the code two
  people compare.** ``invite`` returns the token's *path*; the code a pairing shows
  is returned because the user has to be shown it. It is not a credential, but it
  DOES travel: both devices derive the same digits from the live handshake, and the
  joining device sends its transcription inside a SEALED frame for the inviter to
  compare against its own derivation (``handshake.pair_ready_frame``, compared in
  ``relay.py``'s ``net_pair_ready``). It therefore never appears in cleartext on the
  wire, in a log line or in an error — and its security never rested on secrecy in
  the first place: it rests on a human having read it off the other device's screen.
  The earlier wording here ("never travels") was wrong, and it was the premise this
  tool's own rewrite used to relax the design's invariant (agent review round 1,
  semantic finding 2). The CLI's
  stdout is PARSED, never passed through: a raw blob would put a token in the
  transcript the moment a command printed one (``mesh-ui.md`` §3.2), and every
  payload is scrubbed of secret-shaped keys on the way out as a second line of
  defence.
* **No confirmation, ever — the second phase is the PERSON's, and this tool
  cannot answer a park at all.** Pairing is two people reading a code off each
  other's screens. ``join`` opens the ceremony, parks it and hands back the code
  together with the sentence the user needs; the person then runs ``lop network join
  --confirm <code>`` at their own terminal (or answers the inviter's ``lop network
  confirm`` prompt), and ``_argv_for`` spells no ``--confirm`` anywhere — no field,
  no default, no branch that can produce one. The earlier revision took that code as
  a field on this tool, which is decorative rather than safe: the code both devices
  derive is the SAME digits, so a model holding the code it just printed can echo it
  back and satisfy the very comparison that exists to detect a substitution (agent
  review round 1, semantic finding 1; the operator's ruling was to remove the field,
  not to guard it). ``panic``/``disconnect``/``member rm`` mutate the trust state of
  every device in a network and still take no ``--yes``. So a refusal comes back as
  the CLI's own sentence rather than as an action taken on the user's behalf.
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
READ_ACTIONS = frozenset(
    {
        "status",
        "ls",
        "show",
        "peers",
        "log",
        "doctor",
        # Peer readiness: every fact comes from the peer's own relay (or from this
        # device's records when no relay answers), and the verb writes nothing —
        # the same discipline that keeps the listing reads on this side.
        "ready",
        "credentials",
        "definitions_state",
        # The user-scope MCP server list (``lop network mcp state``): a read of
        # this device's own mcp.json. Push is NOT an action here — the precedent
        # for the definitions push is the same (the tool has no push verb).
        "mcp_state",
        # Its BARE form is a listing, which is why it is on this side of the union the
        # tests assert; ``_approval_tier`` upgrades it per call when the arguments
        # carry a mutation.
        "sessions",
    }
)
WRITE_ACTIONS = frozenset({"init", "invite", "join", "member_rm", "disconnect", "panic", "trust"})

#: The one action whose tier depends on WHICH verb it carries: ``sessions`` presents
#: one peer's session list, which is a read, and can create, engage, stop or delete a
#: session on it, which are not. The split is therefore made on the arguments the
#: model actually sent, rather than on a second family of action names it would have
#: to keep in step with its own flags — a ``sessions_create`` that forgot ``peer``
#: would be the same call under a friendlier name.
_SESSION_MUTATIONS = ("create", "engage", "stop", "delete")

#: Local reads answer in well under a second; ``doctor`` dials endpoints. The
#: bound exists so a wedged subprocess can never hold a turn open.
_DEFAULT_TIMEOUT_S = 30.0
#: ``join`` performs a real handshake against the far device, which is bounded by
#: that relay's own timers, not by ours.
_JOIN_TIMEOUT_S = 120.0

#: How long a parked pairing's FIRST body may take. A dial and a handshake, which the
#: CLI bounds internally — so this is the "the pairing never announced itself" bound,
#: not a patience budget for the person who has to read a code off another screen.
_PARK_READY_TIMEOUT_S = 30.0

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
    # The readiness report (readiness.py): the install question ``doctor`` does not
    # ask — can a peer COMPLETE work offloaded to it — with the fix for each
    # blocker. A read of the peer's own state, or of this device's records.
    "ready",
    # The session plane's client half (``mesh-session-mobility.md`` §9.3): one peer's
    # sessions, and the four verbs that act on one. Before this the tool could see a
    # peer and drive nothing it held.
    "sessions",
    # Re-admitting a network this device marked untrusted is a WRITE about a record
    # every member's trust rests on, and it is the one command ``panic`` names as its
    # follow-up — a tool that could raise a panic and not clear it left the operator
    # with a terminal step the tool itself pointed at.
    "trust",
    # What this device owns and what it borrows (``lop network credentials``), and the
    # definitions it holds for its peers (``lop network definitions state``): both are
    # reads of this device's own files, and both answer a question an agent asked to
    # get here — "why does the peer resolve this name to the wrong thing".
    "credentials",
    "definitions_state",
    # The third install-wide sync's read (``lop network mcp state``, mcpdefs.py):
    # the user-scope MCP servers a peer would receive, their provenance, and the
    # reference keys a mirror still needs — the question a failed offloaded
    # server is usually the answer to. A read of this device's own mcp.json.
    "mcp_state",
]


class NetworkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: NetworkAction = Field(
        description=(
            "Which operation to run. Writes: init, invite, join, member_rm, trust, "
            "disconnect, panic, and sessions with a mutating field. Otherwise a read."
        )
    )
    network: str = Field(
        default="",
        description=(
            "A network name or id. Required by show, member_rm and trust; for init, "
            "the new name. Omitted means the one network this device is in."
        ),
    )
    token: str = Field(
        default="",
        description=(
            "For join: the invite token or '@<path>' to a token file. Never echoed "
            "back; kept out of `ps`."
        ),
    )
    expires: str = Field(
        default="",
        description="For invite: how long the token stays open, e.g. 30m or 2h (default 10m).",
    )
    peer: str = Field(
        default="",
        description=(
            "For sessions: the device to ask (a name from `peers`). For ready: the "
            "device to report on. Required by create/engage/stop/delete."
        ),
    )
    all_peers: bool = Field(default=False, description="For sessions: list every device.")
    create: bool = Field(default=False, description="For sessions: create a session on `peer`.")
    prompt: str = Field(default="", description="For create: the new session's first turn.")
    engage: str = Field(default="", description="For sessions: a session id on `peer` to warm up.")
    stop: str = Field(default="", description="For sessions: a session id on `peer` to end.")
    delete: str = Field(
        default="",
        description=(
            "For sessions: a session id on `peer` to delete. Always the owner's dry "
            "run — a real delete needs the user's `--yes`, so report no deletion."
        ),
    )
    trust_state: Literal["active", "untrusted"] = Field(
        default="active",
        description="For trust: 'active' re-admits an untrusted network, 'untrusted' refuses it.",
    )
    role: Literal["read", "drive", "admin"] = Field(
        default="read",
        description=(
            "For invite: what the joiner may do — read (see sessions), drive "
            "(prompt/stop), admin (also membership)."
        ),
    )
    device: str = Field(
        default="",
        description=(
            "For member_rm: the device id to revoke. For invite: a device id to bind "
            "the token to, so only it may redeem it (others are refused and the "
            "invite burns)."
        ),
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


#: The text spellings a model may send for "no". Absent for BOTH readers of the verb
#: set, because the approval tier is not a second opinion: a call tiered ``read`` that
#: still reaches the CLI as a mutation is the defect this constant exists to prevent.
_FALSY_TOKENS: frozenset[str] = frozenset({"", "0", "false", "no", "none", "null"})


def _session_verbs(args: dict[str, Any]) -> dict[str, str]:
    """The mutating verbs this ``sessions`` call really carries, normalised ONCE.

    ONE PREDICATE, TWO READERS — the approval tier and the argv table — because the
    tier claim is "a function of the arguments that reach argv". Review round 1
    (MINOR 4) reproduced the two disagreeing: ``{"stop": "0"}`` was tiered ``read``
    (``"0"`` being the text spelling of "no") while argv still spelled ``--stop 0``,
    so a mutation rode a call that raised no approval. That is latent rather than live
    only because session ids are ``uuid4().hex[:12]`` and none of those literals can
    name one — luck, not safety. ``create`` is a boolean and the other three are
    operands; a falsy spelling is no verb at all.
    """
    verbs: dict[str, str] = {}
    create = args.get("create")
    if create is True or (isinstance(create, str) and create.strip().lower() not in _FALSY_TOKENS):
        verbs["create"] = ""
    for name in _SESSION_MUTATIONS:
        if name == "create":
            continue
        value = args.get(name)
        if isinstance(value, str) and value.strip().lower() not in _FALSY_TOKENS:
            verbs[name] = value.strip()
    return verbs


def _session_mutation(args: dict[str, Any]) -> str:
    """Which mutating verb this ``sessions`` call carries, if any — the TIER's reader."""
    return next(iter(_session_verbs(args)), "")


def _approval_tier(args: dict[str, Any]) -> Literal["read", "write", "exec"]:
    """The tier of ONE call, as the CLI's own split states it (``mesh-ui.md`` §3.2).

    A read never prompts and a write always does, so the answer has to be a function
    of the arguments rather than of the action name alone: ``sessions`` is the one
    action that is both, and choosing its tier by which verb the call carries is what
    keeps a listing silent without making a stop silent too.
    """
    action = str(args.get("action") or "")
    if action == "sessions":
        return "write" if _session_mutation(args) else "read"
    return "read" if action in READ_ACTIONS else "write"


def _argv_for(params: NetworkParams) -> tuple[list[str], str]:
    """``(argv, error)`` — the CLI's argv for this call, or the sentence saying
    what the call is missing.

    Every branch is explicit and every flag is spelled here rather than
    constructed from user text, which is what makes three promises properties of the
    code instead of promises: no ``--yes`` and no ``--force`` is ever spelled, and
    ``--confirm`` is not spelled AT ALL — the human's second phase is not a field on
    this tool, so no argument combination can answer a park (agent review round 1,
    semantic finding 1). ``--json`` is added once, at the end, for the same reason the
    CLI refuses ``--print`` beside it: one of the two must win, and the agent's path
    is the JSON one.
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
    elif action == "ready":
        argv = ["network", "ready"]
        if params.peer.strip():
            argv += ["--peer", params.peer.strip()]
    elif action == "credentials":
        argv = ["network", "credentials"]
    elif action == "definitions_state":
        argv = ["network", "definitions", "state"]
    elif action == "mcp_state":
        argv = ["network", "mcp", "state"]
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
        if params.expires.strip():
            argv += ["--expires", params.expires.strip()]
        if params.device.strip():
            argv += ["--device", params.device.strip()]
    elif action == "join":
        if not params.token.strip():
            return [], (
                "action='join' needs 'token' (the invite token, or '@<path>' to "
                "the token file the other device minted). It PARKS the pairing and "
                "hands the person their step — this tool has no way to answer one."
            )
        # PHASE ONE parks the ceremony, which is what lets the code be handed to a
        # person: a pairing that only prompted would have nobody to prompt. Phase two
        # is theirs to run at their own terminal (`lop network join --confirm <code>`),
        # which is why no branch here spells it.
        argv = ["network", "join", params.token.strip(), "--park"]
    elif action == "sessions":
        # THE SAME PREDICATE THE TIER READS (``_session_verbs``), so argv and the
        # approval decision cannot disagree about whether this call mutates.
        verbs = _session_verbs(params.model_dump())
        mutation = next(iter(verbs), "")
        peer = params.peer.strip()
        if mutation and not peer:
            return [], (
                f"action='sessions' with '{mutation}' needs 'peer': the session lives on "
                "one device, and only its owner may act on it."
            )
        if mutation and params.all_peers:
            # REFUSED rather than sent: ``--all-peers`` is read only on the two listing
            # paths, so beside a mutating verb the CLI accepts the flag and silently
            # drops it — the class the CLI's own ``--force`` guard names ("a flag that
            # is accepted and then quietly dropped is the same class of untruth"), and
            # the agent would be told a stop happened across every device when exactly
            # one was asked for (agent review round 1, MINOR 5).
            return [], (
                f"action='sessions' with '{mutation}' acts on ONE device's session, so "
                "'all_peers' means nothing here and the CLI drops it silently. Send "
                "'peer' instead, or a plain listing with 'all_peers'."
            )
        if not mutation and not peer and not params.all_peers:
            return [], (
                "action='sessions' needs 'peer' (one device you are paired with) or "
                "'all_peers' (every device), or one of create/engage/stop/delete."
            )
        argv = ["network", "sessions"]
        if "create" in verbs:
            argv += ["--create"]
            if params.prompt.strip():
                argv += ["--prompt", params.prompt.strip()]
        if "engage" in verbs:
            argv += ["--engage", verbs["engage"]]
        if "stop" in verbs:
            argv += ["--stop", verbs["stop"]]
        if "delete" in verbs:
            # NEVER with ``--yes``: this verb's real form is the owner's deletion, and
            # the human's confirmation is the point of it, not a formality to skip.
            argv += ["--delete", verbs["delete"]]
        if peer:
            argv += ["--peer", peer]
        if params.all_peers:
            argv += ["--all-peers"]
    elif action == "trust":
        if not network:
            return [], "action='trust' needs 'network' (a name or id)."
        argv = ["network", "trust", network, f"--{params.trust_state}"]
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
        "join": "join a network from an invite (a person reads a code off the other device)",
        "member_rm": "revoke a member and rotate the secret",
        "disconnect": "leave the network and delete its local secret",
        "panic": "raise a panic: rotate the secret for every member",
        "trust": "change whether this device trusts the network",
        "sessions": "act on a session that runs on another device",
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


async def _reap(process: asyncio.subprocess.Process, *, grace: float = 5.0) -> None:
    """Wait for it to exit, and kill it if it will not: nothing outlives this call."""
    try:
        await asyncio.wait_for(process.wait(), grace)
    except (asyncio.TimeoutError, TimeoutError):
        process.kill()
        await process.wait()


def _reap_when_it_exits(pid: int) -> None:
    """Reap a DETACHED child from a thread, because this call's loop will be gone.

    The parked ceremony outlives the call that started it by design, and the loop it
    was started under does not: the harness runs each tool call in its own
    ``asyncio.run``, so asyncio's child watcher is dead long before the ceremony ends
    and something has to wait for the child or it stays a zombie for the rest of the
    session's life — one per completed pairing, each holding a pid, which is also what
    made a liveness probe answer "alive" for a process that had finished (agent review
    round 1, code finding 1). A blocking ``os.waitpid`` on ONE pid, in one daemon
    thread, is that waiter: it returns the moment this child exits, whenever that is.

    ``os.waitpid`` exists on POSIX only, and so do zombies — the capability check is
    the guard rather than a platform name. On a host that runs the tool inside a
    LONG-LIVED loop, asyncio's own watcher may win the race instead: this thread then
    gets ``ChildProcessError`` and does nothing, which is the intended outcome either
    way (the child is reaped exactly once).
    """
    if not hasattr(os, "waitpid"):  # pragma: no cover — no zombies off POSIX
        return
    import threading

    def wait() -> None:
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            # Somebody else reaped it first (an alive loop's watcher), or it was never
            # ours: either way the process table is already consistent.
            pass

    threading.Thread(target=wait, name="network-park-reap", daemon=True).start()


async def _start_parked_join(argv: list[str]) -> tuple[int, str, str]:
    """Start a parked pairing, return the FIRST JSON body it prints, leave it running.

    WHY THIS IS NOT ``_run_cli``, and it is the whole point of the two-phase pair: a
    parked ceremony ends when a PERSON answers it — minutes later, and from wherever
    they are — so waiting for the process here would burn the turn and then kill the
    very process the answer has to reach (``_run_cli`` kills a timed-out child). The
    child is therefore started in its own session, its first complete JSON document is
    read off stdout, and it is left holding the socket: the PERSON's own ``lop network
    join --confirm <code>`` (or the inviter's ``lop network confirm`` prompt) is what
    answers it, and it exits by itself at the end of its window.

    Nothing is left unreaped, but NOT because asyncio reaps it: that belief held only
    for a loop that outlives the child, and the harness runs each tool call under its
    own short-lived one — so a parked child that exits minutes later is a ZOMBIE, not a
    reaped process, and a liveness probe reads it as alive forever (agent review round
    1, code finding 1, which caught exactly that: ``state=Z`` for 300 consecutive
    polls). A child left holding the socket is therefore handed to
    ``_reap_when_it_exits``, a waiter of its own. A body that never arrives (the CLI
    refused first, or the dial hung) is a failure, and the child is killed and waited
    for before this returns.
    """
    process = await asyncio.create_subprocess_exec(
        *_cli_argv(),
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
        # Its own session, so a session restart or a killed tool call does not take the
        # ceremony with it mid-handshake.
        start_new_session=True,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _PARK_READY_TIMEOUT_S
    buffer = ""
    decoder = json.JSONDecoder()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0 or process.stdout is None:
            break
        try:
            line = await asyncio.wait_for(process.stdout.readline(), remaining)
        except (asyncio.TimeoutError, TimeoutError):
            break
        if not line:
            break
        buffer += line.decode("utf-8", "replace")
        text = _strip_ansi(buffer)
        start = text.find("{")
        if start < 0:
            continue
        try:
            document, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            # An unfinished document: the rest of it is still on its way.
            continue
        if not isinstance(document, dict):
            continue
        if document.get("ok") is False:
            # A refusal: printed, then the process exits. Wait for it, so the caller
            # reports it like any other refusal and nothing is left running.
            await _reap(process)
        else:
            # THE SOCKET IS STILL OPEN, so this child is still ours and will exit long
            # after this loop is gone: hand it a waiter that outlives the call.
            _reap_when_it_exits(process.pid)
        return 0, text[start : start + end], ""
    # No body, so nothing is parked and nobody could answer it: the socket goes now
    # rather than at the end of a window nothing is watching.
    await _reap(process)
    err = await process.stderr.read() if process.stderr is not None else b""
    return 1, "", _clean(err.decode("utf-8", "replace"))


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
            + ("" if payload.get("supported") else "  (no user service supervisor here)"),
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
        from local_operator.network import readiness as readiness_mod
        from local_operator.resume import UNNAMED_DEVICE, peer_reason_words

        # THE SAME BUILD SEGMENT THE CLI'S OWN ROWS CARRY (design §4), through the one
        # comparison and one spelling (``readiness.compare_builds`` / ``build_suffix``)
        # — a stale peer is exactly the fact an offload decision turns on, and this
        # digest is where the model reads it. Unknown stays silent (the old pinned
        # lines are the A/B rule), and the stamp is read lazily so a listing with no
        # known build pays nothing extra.
        own_stamp: dict[str, str] | None = None
        digest: list[str] = []
        for row in rows:
            line = (
                f"{'reachable' if row.get('reachable') else 'unreachable':11} "
                f"{str(row.get('name') or '').strip() or UNNAMED_DEVICE}"
            )
            if row.get("reachable"):
                build = row.get("build")
                if isinstance(build, dict) and build.get("version"):
                    if own_stamp is None:
                        from local_operator.network import relay as relay_mod

                        own_stamp = relay_mod.build_stamp()
                    line += readiness_mod.build_suffix(
                        readiness_mod.compare_builds(build, own_stamp)
                    )
            else:
                line += f"  {peer_reason_words(str(row.get('reason') or ''))}"
            digest.append(line)
        return digest
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
    if action == "ready":
        # THE SAME ROWS THE CLI SHOWS ITS OWN READER, through the shared
        # renderer (``readiness.render_check_lines``) — one loop, one register,
        # so the digest and the CLI cannot drift (agent review round 1, NIT-2),
        # and the reachability reading is this verb's own (a REFUSED connection
        # must not read as "nothing answered" in an agent's digest any more than
        # on a person's screen; the raw vocabulary stays in ``details``).
        from local_operator.network import readiness as readiness_mod

        ready_lines = readiness_mod.render_check_lines(payload.get("checks") or [])
        if not ready_lines:
            ready_lines = [readiness_mod.NOTHING_TO_CHECK_LINE]
        if not payload.get("identity_present", True):
            ready_lines.append(readiness_mod.NO_IDENTITY_LINE)
        return ready_lines
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
        if str(payload.get("status") or "") == "awaiting_confirmation":
            # PHASE ONE's receipt. It says what the person needs and what has NOT
            # happened yet, because the one way this digest could mislead is by reading
            # as a completed pair: nothing has been sent to the other device, and the
            # code below is THIS device's own derivation.
            code_text = str(payload.get("sas") or "")
            if len(code_text) == 6:
                code_text = f"{code_text[:3]} {code_text[3:]}"
            return [
                f"pairing started with {payload.get('name') or payload.get('network_id')} "
                f"({payload.get('inviter') or 'the other device'})",
                f"this device's code: {code_text}",
                f"{payload.get('sentence')}",
                f"{int(float(payload.get('seconds_left') or 0))}s left; nothing is "
                "joined until the person confirms, and the confirmation has to come "
                "from them.",
            ]
        # A finished ceremony's receipt. This tool cannot produce it — phase two is
        # the person's, run at their own terminal — but the CLI does report it, and a
        # report that arrived is rendered rather than dumped as JSON.
        lines = [
            f"joined {payload.get('name')} ({payload.get('network_id')}) at epoch "
            f"{payload.get('epoch')}, role {payload.get('role')}, "
            f"{payload.get('members')} member(s)",
            f"fingerprint {payload.get('fingerprint')}",
        ]
        # THE TWO CAUSES READ DIFFERENTLY (UX round 1, U2), the delta lines sit
        # beside the serving line rather than after "next:" (design round 1, D5),
        # and the subjects are the joiner's — "available here:" (UX round 1, U5)
        # — the same receipt the CLI renders from the same facts
        # (``offers.missing_share_lines`` owns the sentences for both paths).
        from local_operator.network.credentials import offers as offers_mod

        shares = [str(key) for key in payload.get("shares") or []]
        reduced = [str(key) for key in payload.get("reduced") or []]
        offered = [
            str(item.get("key"))
            for item in payload.get("offers") or []
            if isinstance(item, dict) and item.get("share")
        ]
        extras: list[str] = []
        if shares:
            extras.append(f"available here: {', '.join(shares)}")
        extras.extend(offers_mod.missing_share_lines(offered, shares, reduced))
        lines[1:1] = extras
        return lines
    if action == "sessions":
        rows = payload.get("sessions")
        if rows is None:
            # A MUTATION's receipt: the peer's own sentence, never a re-rendering
            # of it (the CLI renders the owner's words verbatim, and a second
            # vocabulary for "did it stop" is the thing that gets out of step).
            sentence = str(payload.get("detail") or payload.get("message") or "").strip()
            return [sentence] if sentence else [json.dumps(payload, sort_keys=True)]
        from local_operator.resume import peer_reason_words, session_state_words

        lines = [
            "  ".join(
                part
                for part in (
                    f"{session_state_words(str(row.get('state') or '')) or '?':10}",
                    str(row.get("conversation_name") or "").strip() or "(untitled)",
                    str(row.get("session_id") or ""),
                    "on "
                    + str(
                        (row.get("peer") or {}).get("name")
                        or (row.get("peer") or {}).get("device_id")
                        or "this device"
                    ),
                )
                if part
            )
            for row in rows[:_ROW_CAP]
        ]
        # Bound to a narrow local first: ``dict.get`` is `Any | None`, and the loop
        # below iterates it as a mapping (the CLI's own listing takes the same
        # precaution for the same reason).
        peers_block = payload.get("peers")
        facts: dict[str, Any] = peers_block if isinstance(peers_block, dict) else {}
        for device_id, block in sorted(facts.items()):
            if isinstance(block, dict) and not block.get("reachable"):
                lines.append(
                    f"{str(block.get('name') or device_id)}: did not answer "
                    f"({peer_reason_words(str(block.get('reason') or ''))})"
                )
        if not rows:
            # THE SENTENCE IS NARROWED TO WHAT WAS ESTABLISHED: with a device that
            # did not answer, "no sessions" is a claim about every device, and the
            # line above already says which one is missing from the answer.
            lines.append(
                "no sessions are held by the devices that answered"
                if lines
                else "no sessions are held by other devices right now"
            )
        return lines
    if action == "trust":
        lines = [f"{payload.get('network_id')} is now {payload.get('trust')}"]
        if payload.get("applied_locally"):
            lines.append("the relay is not running on this device: applied locally")
        return lines
    if action == "credentials":
        from local_operator.network import readiness as readiness_mod

        networks = payload.get("networks") or []
        lines: list[str] = []
        for network in networks:
            lines.append(f"{network.get('network')}:")
            for row in network.get("credentials") or []:
                owner = (
                    "this device"
                    if row.get("owned_here")
                    else (row.get("owner_device_name") or row.get("owner_device"))
                )
                # ``credential_name``, and the SPELLING is load-bearing: ``_scrub``
                # drops any field whose NAME contains a secret marker, the markers are
                # substring matches, and ``key`` — and ``credential_key``, the first
                # attempt at this fix — both contain "key". The production path therefore
                # rendered ``None`` where the credential's name belongs (agent review
                # round 1, code findings 4 and 5). The markers and the scrub boundary stay
                # exactly where they are; the CLI's field is named so the heuristic cannot
                # eat the one fact this listing exists to convey.
                lines.append(f"  {row.get('credential_name')}  {row.get('kind')}  owner: {owner}")
        # THE SAME BLOCK THE CLI PRINTS, through the one renderer
        # (``readiness.shareable_lines``): the device-level ledger of what this device
        # could offer — the shareability preflight design §2 added, so an agent asked
        # "why did the share refuse" (or "what can this device share") reads the answer
        # here instead of learning it at share time.
        lines.extend(readiness_mod.shareable_lines(payload.get("shareable") or []))
        return lines or ["nothing is shared with or by this device"]
    if action == "definitions_state":
        lines = []
        mirrored = payload.get("mirrored") or {}
        for kind in ("agents", "teams"):
            for name in sorted(payload.get(kind) or {}):
                origin = (mirrored.get(kind) or {}).get(name) or ""
                lines.append(
                    f"{kind[:-1]}: {name}"
                    + (f" (mirrored from {origin})" if origin else " (yours)")
                )
        return lines or ["no agent or team definitions on this device"]
    if action == "mcp_state":
        from local_operator.network import mcpdefs  # the ONE "looks like …" spelling

        lines = []
        for row in payload.get("servers") or []:
            origin = str(row.get("origin") or "")
            line = f"server: {row.get('name')}  {row.get('transport')}" + (
                f" (mirrored from {origin})" if origin else " (yours)"
            )
            # ``refs[].set`` is False only when the store was READ and does not
            # hold the key; ``None`` (unreadable store) is not a "needs".
            needs = [
                str(ref.get("id"))
                for ref in row.get("refs") or []
                if isinstance(ref, dict) and ref.get("set") is False
            ]
            if needs:
                line += " — needs: " + ", ".join(needs)
            # Same marker as the CLI's `state` (mcpdefs.state_rows carries the
            # label): a row the shape scan will never send says so before a push.
            withheld = str(row.get("withheld") or "")
            if withheld:
                line += f" — will not travel: {mcpdefs.shape_likeness(withheld)}"
            lines.append(line)
        return lines or ["no user-scope MCP servers on this device"]
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
        # THE SENTENCE THAT USED TO BE HERE TOLD THE AGENT TO PASS THE CODE BACK, and
        # its absence is load-bearing: the code a park prints is THIS device's own
        # derivation, and both devices derive the same digits, so a model echoing it
        # satisfies the comparison by construction (agent review round 1, semantic
        # finding 1 — the field is gone, so this hint must not describe one). What
        # survives is the property the two-phase pair exists to protect.
        return (
            "Pairing needs a person: `join` parks the ceremony and returns the code for "
            "THEM to read out, and the second phase is theirs — `lop network join "
            "--confirm <code>` at a terminal, or the inviter's own `lop network confirm` "
            "prompt. This tool has no way to answer a pairing."
        )
    if action in ("panic", "disconnect", "member_rm"):
        return (
            "This one needs an explicit instruction naming the network and the "
            "action; it is never a retry after a failed command."
        )
    if action == "invite":
        return "The token is written to a file; the result carries its path, never the token."
    if action == "sessions":
        return (
            "A session's owner is the device it runs on, so every verb here is asked "
            "of 'peer' rather than done locally."
        )
    if action == "trust":
        return (
            "`panic` and `disconnect` mark a network untrusted; `trust` is the only "
            "way back, and it does NOT restore a member that was removed."
        )
    if action == "ready":
        return (
            "Every FAIL row carries the command that clears it and the device to run it "
            "on; pass 'peer' (a name or id) to narrow the report to one device."
        )
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

    # PHASE ONE KEEPS RUNNING AFTER THIS CALL RETURNS, which is why it is not
    # ``_run_cli``: the ceremony is held by the process that dialled, and the code has
    # to reach a person before that process may finish. See ``_start_parked_join``.
    # EVERY join is phase one: there is no field that answers a park, so there is no
    # spelling of `join` on this tool that is anything but a park.
    parked = params.action == "join"
    token_path: str | None = None
    if parked:
        token_arg = params.token.strip()
        if not token_arg.startswith("@"):
            token_path = _write_token_file(token_arg)
            token_arg = f"@{token_path}"
        # Rebuilt rather than patched in place: the token is the ONE positional
        # this action takes, and a stray argv entry after it would be parsed as
        # one rather than reported.
        argv = ["network", "join", token_arg, "--park", "--json"]

    timeout = _JOIN_TIMEOUT_S if params.action == "join" else _DEFAULT_TIMEOUT_S
    try:
        if parked:
            code, stdout, stderr = await _start_parked_join(argv)
        else:
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
            # file is the one place the tool holds it. Deleting it here is safe for
            # the parked ceremony: the child read the token before it dialled, and
            # nothing after the handshake reads it again.
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
        if params.action == "ready" and isinstance(scrubbed.get("checks"), list):
            # AN UNHEALTHY REPORT IS STILL THE REPORT (QA round 1, Q-1): the
            # FAIL rows and their remedies are this verb's whole product, and
            # the generic refusal branch below kept only ``code: message`` —
            # dropping exactly the rows the hint says to read, on the one path
            # they exist for. The rows read through the same renderer the
            # healthy path uses; the raw payload still rides ``details``.
            body = "\n".join(_render(params.action, scrubbed))
            text, spill = spill_truncate(body, _TOOL, context)
            return _error(tool_call_id, _TOOL, text, details=spill or {"network": scrubbed})
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
            "Read and drive a lop mesh network from this device: peers, their sessions, "
            "creating a session on a peer, and the network's own lifecycle. Pairing and "
            "incident controls need a human: `join` parks a pairing and returns the code "
            "for the user to read out, answering it is theirs to run, and this tool "
            "reports what the CLI refused and why."
        ),
        parameters=NetworkParams.model_json_schema(),
        # Write tier because the highest op needs it (``panic`` rotates a secret
        # for every member); the reads downgrade per call so looking at the
        # network never prompts.
        approval_tier="write",
        # PER CALL, and a function of the arguments rather than of the action name:
        # a plain ``sessions`` listing is a read and the same action carrying
        # ``stop`` is not, so the tier is decided by ``_approval_tier`` beside the
        # argv table that spells both.
        call_approval_tier=_approval_tier,
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
