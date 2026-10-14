"""Ready-successor handoff for a listening daemon that must not be killed.

The update that installs a new build runs while the old daemon is still serving
operator work. Killing it loses in-flight turns, so the new generation is brought
up as a FENCED successor: it constructs its acceptors but serves nothing, the old
daemon drains its local work and stops accepting on one non-awaiting step, and
only then does the successor accept. Accepted connections and their tasks are
never torn down — the transfer is of the LISTENING socket, not of live traffic.

Scope and constraints:

* Capable builds only. A stock install whose API lacks the rolling contract
  cannot construct the successor and refuses; the one-time bootstrap for such an
  install is a separate, operator-owned step, not something this module fakes.
* POSIX passes the listener by ``os.dup`` over ``SCM_RIGHTS`` on a same-machine
  control channel — the same descriptor mechanism ``session.runtime.standby``
  already uses. Windows has no integer-FD equivalent here and refuses.
* Nothing here calls ``Server.shutdown``/``should_exit``, closes accepted
  transports, or touches the lifespan: ``pause_accept`` closes ACCEPTORS only,
  exactly the shape uvicorn's own reload uses (``server.py:285-286``).
* The transfer record rides the existing install-root transaction as one extra
  observation on the durable operation — no second lock, no CAS service.
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

HANDOFF_CONTRACT = "ready-successor-v1"
#: Busy-turn checkpoint vocabulary. Mirrors ``transcript.CHECKPOINT_PHASES``; kept
#: here so this module stays a leaf (the transcript module is heavier and imports
#: on the hot path), with a unit test pinning the two spellings together.
CHECKPOINT_PHASES = ("resume", "deferred")
#: Record states. ``ready`` is B announcing under the transaction; ``commit`` is
#: A's irreversible step; ``ack`` is B confirming it holds the listener; anything
#: else (abort/unknown) leaves BOTH roots and BOTH processes alive for recovery.
HANDOFF_STATES = ("ready", "commit", "ack", "aborted")
_CONTROL_HEADER = struct.Struct("!I")
_MAX_FRAME = 1 << 16


class HandoffError(RuntimeError):
    """An actionable refusal; never permission to force or to kill a daemon."""


class ListenerUnsupported(HandoffError):
    """This platform/build cannot transfer a listener; refuse, do not improvise."""


def _fd_socket_path_supported() -> bool:
    return hasattr(os, "dup") and hasattr(socket, "send_fds") and os.name != "nt"


class ListenerOwner:
    """Owns the master listening socket and the accept-only lifecycle.

    One per listening daemon. The acceptors are recreated from the retained
    master socket rather than rebound, so the bound port never closes and the
    listener inode is stable across a handoff — the successor verifies that inode
    before it accepts, which is what makes "the same port" a checkable fact
    instead of a hope.
    """

    def __init__(self, app: Any, servers: "list[Any]", listener: socket.socket) -> None:
        self.app = app
        # A DUP of the bound listener: closing an acceptor must not close the
        # master, and the original object stays owned by the caller's frame.
        self.socket = listener.dup() if hasattr(listener, "dup") else listener
        self.servers = servers
        self.accepting = True
        self._create: Callable[[], Awaitable[list[Any]]] | None = None

    @property
    def listener_ino(self) -> str:
        """A stable identity for THIS listener, so a successor can prove it."""

        info = os.fstat(self.socket.fileno())
        return f"{info.st_dev}:{info.st_ino}"

    def _bind_acceptor_factory(self, factory: Callable[[], Awaitable[list[Any]]]) -> None:
        self._create = factory

    async def start_fenced(self) -> None:
        """Build acceptors WITHOUT serving. Nothing is accepted until resume.

        ``start_serving=False`` is the whole point: a successor that accepted
        before the old daemon committed would serve two leaders for the same
        port. uvicorn 0.54's ``startup`` serves immediately, so the factory is
        supplied by the caller (which owns the protocol class) and the acceptors
        are created here, fenced.
        """

        if self._create is None:
            raise HandoffError("fenced start requires an acceptor factory from the caller")
        self.servers = await self._create()
        self.accepting = False

    def pause_accept(self) -> None:
        """Stop accepting NEW connections; leave accepted ones running.

        ``Server.close`` on an acceptor closes only the listening half — accepted
        transports hold their own descriptors and are untouched. This is the
        uvicorn reload shape, not ``Server.shutdown``.
        """

        for server in self.servers:
            server.close()
        self.accepting = False

    async def resume_accept(self) -> None:
        """Rebuild acceptors from the retained master, then serve.

        A closed acceptor is terminal — ``Server.close()`` nulls its ``_sockets``,
        so ``start_serving`` cannot revive it. Resume therefore recreates acceptors
        from ``self.socket`` (which was never closed, because the acceptor owns a
        dup) and starts them. Never rebinds: the bound port must not move.
        """

        if self._create is None:
            raise HandoffError("resume requires an acceptor factory from the caller")
        self.servers = await self._create()
        for server in self.servers:
            await server.start_serving()
        self.accepting = True

    def pass_control(self, channel: socket.socket, payload: bytes = b"") -> None:
        """Send the master listener descriptor over a same-machine channel.

        Only the LISTENER (and never accepted transports) crosses the boundary;
        the payload carries the transfer record the receiver verifies.
        """

        if not _fd_socket_path_supported():
            raise ListenerUnsupported(
                "listener handoff needs POSIX descriptor passing; refusing on this platform"
            )
        header = _CONTROL_HEADER.pack(len(payload))
        socket.send_fds(channel, [header + payload], [self.socket.fileno()])


def receive_listener(channel: socket.socket) -> "tuple[socket.socket, bytes]":
    """Read a transferred listener + record from a control channel.

    Reads the framed header with ``recv_fds`` so the descriptor arrives with the
    message that names it (the ``standby`` convention: a marker byte sent
    separately is indistinguishable from the frame's length prefix).
    """

    if not _fd_socket_path_supported():
        raise ListenerUnsupported(
            "listener receive needs POSIX descriptor passing; refusing on this platform"
        )
    header = b""
    fds: list[int] = []
    while len(header) < _CONTROL_HEADER.size:
        chunk, received, _flags, _addr = socket.recv_fds(
            channel, _CONTROL_HEADER.size - len(header), max(1, 4 - len(fds))
        )
        if not chunk:
            for fd in fds:
                os.close(fd)
            raise HandoffError("handoff control channel closed before the listener arrived")
        header += chunk
        fds.extend(received)
    if len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise HandoffError("handoff must carry exactly one listener descriptor")
    size = _CONTROL_HEADER.unpack(header)[0]
    if size > _MAX_FRAME:
        os.close(fds[0])
        raise HandoffError("handoff record frame is unreasonably large")
    payload = b""
    while len(payload) < size:
        chunk = channel.recv(size - len(payload))
        if not chunk:
            os.close(fds[0])
            raise HandoffError("handoff record truncated")
        payload += chunk
    return socket.socket(fileno=fds[0]), payload


async def _noop() -> None:  # pragma: no cover - reserved default factory shape
    return None


def _acceptor_factory(listener: socket.socket, protocol_factory: Callable[[], Any]):
    """Return an async factory that creates acceptors with ``start_serving=False``.

    The acceptor is built from a DUP of the master. ``loop.create_server(sock=)``
    takes ownership of the socket it is handed, so closing an acceptor (pause) would
    otherwise close the one master the handoff must still pass on. Each rebuild dups
    afresh, so a pause/resume cycle never exhausts or closes the retained master.
    """

    async def build() -> list[Any]:
        loop = asyncio.get_running_loop()
        owned = listener.dup()
        try:
            server = await loop.create_server(protocol_factory, sock=owned, start_serving=False)
        except BaseException:
            owned.close()
            raise
        return [server]

    return build


#: How long a `running` occurrence may be honoured before its birth is
#: re-checked. Only the birth-mismatch can reclassify it; age ALONE never does.
OCCURRENCE_STALE_S = 30 * 60.0


def suppressed_occurrences(
    occurrences: "dict[str, dict[str, Any]] | None",
    *,
    now: float | None = None,
    same_birth: "Callable[[str], bool] | None" = None,
) -> "tuple[set[str], set[str]]":
    """Split occurrence ids into (suppress, unreadable).

    The contract, in one place because two callers (scheduler, relay) share it:

    * ``None`` (unreadable/missing/unknown shape) ⇒ suppress NOTHING and report
      the set as unreadable. Unknown is not proven still running; a bounded,
      reported duplicate beats a permanently stalled schedule, and this matches
      the repo's rule that an unreadable probe must not decide the action.
    * an entry that is present and non-terminal ⇒ SUPPRESS that id only.
    * an entry present but terminal ⇒ never suppress.
    * a ``running`` entry older than :data:`OCCURRENCE_STALE_S` is reclassified
      ONLY when its recorded birth no longer matches a live process; age alone
      never reclassifies.

    Callers must re-read the map at each dispatch decision rather than caching
    the result: the suppression lifts the moment a terminal entry lands.
    """

    import time as _time

    if occurrences is None:
        return set(), {"<unreadable>"}
    stamp = _time.time() if now is None else now
    suppress: set[str] = set()
    for occurrence_id, entry in occurrences.items():
        if entry.get("state") != "running":
            continue
        recorded_at = entry.get("recorded_at")
        stale = isinstance(recorded_at, (int, float)) and (stamp - recorded_at) > OCCURRENCE_STALE_S
        birth = entry.get("birth")
        if stale and same_birth is not None and isinstance(birth, str):
            # The recorded producer is gone; the entry is not live evidence any
            # more. Reclassify for suppression ONLY on a proved birth mismatch.
            if not same_birth(birth):
                continue
        suppress.add(occurrence_id)
    return suppress, set()


# ---------------------------------------------------------------------------
# Per-unit inventory: how each owned central unit survives a swap.
#
# The handoff is not one mechanism applied N times: a unit that owns a bound
# listener transfers that listener, a unit that owns only timers/peers is fenced
# and re-homed, and a unit with no transferable state is reported pending.
# Keeping this as DATA the wiring reads (rather than prose) is how the
# classification cannot drift from the code that acts on it.
# ---------------------------------------------------------------------------

TransferKind = Literal["listener", "fenced", "service", "pending"]


@dataclass(frozen=True)
class Unit:
    """One owned central unit and the way a handoff must treat it."""

    name: str
    kind: TransferKind
    anchor: str
    note: str

    @property
    def transfers_listener(self) -> bool:
        return self.kind == "listener"

    @property
    def counts_as_moved(self) -> bool:
        """``pending`` is not moved; everything else is an owned transfer."""

        return self.kind != "pending"


#: The authoritative inventory; anchors name the source location the wiring
#: must attach to, so the claim can be audited against the tree.
UNITS: tuple[Unit, ...] = (
    Unit(
        "serve",
        "listener",
        "cli.serve_command / server/reload.py",
        "owns the API listener; wired through ListenerOwner",
    ),
    Unit(
        "mobile",
        "listener",
        "mobile/service.py amain",
        "own uvicorn listener on loopback; scan_loop fenced on admission",
    ),
    Unit(
        "relay",
        "listener",
        "network/relay.py RelayServer.bind/_accept_loop",
        "own peer listener; new admission fenced, existing links untouched",
    ),
    Unit(
        "browser",
        "listener",
        "browser_bridge/daemon.py _serve_main",
        "own uvicorn server; in-flight RPCs fenced before commit",
    ),
    Unit(
        "hub-scheduler",
        "fenced",
        "hub_sync/runner.py / scheduler_service.py",
        "fence already exists; occurrence inheritance needs a durable marker",
    ),
    Unit(
        "wake-monitor",
        "fenced",
        "wakes/supervisor.py / monitors/scheduler.py",
        "freeze kick/pump on admission; hand off via supervisor respawn",
    ),
    Unit(
        "tunnel",
        "pending",
        "tunnels/install.py",
        "no transferable state: relaunch is a fresh start, reported pending",
    ),
    Unit(
        "usage",
        "service",
        "providers/controller.py",
        "lives inside the serving API; covered by the API listener handoff",
    ),
)


def transferable_units() -> tuple[Unit, ...]:
    return tuple(entry for entry in UNITS if entry.transfers_listener)


def fenced_units() -> tuple[Unit, ...]:
    return tuple(entry for entry in UNITS if entry.kind == "fenced")


def pending_units() -> tuple[Unit, ...]:
    return tuple(entry for entry in UNITS if entry.kind == "pending")


def unit(name: str) -> Unit:
    for entry in UNITS:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown central unit: {name}")


def checkpoint_record(
    *,
    session_id: str,
    turn_id: str,
    committed_message_ids: "list[str]",
    boot_build: str = "",
    target_build: str = "",
    provider: str = "",
    model: str = "",
    completed_effects: "list[dict[str, Any]] | None" = None,
    pending_effect: "dict[str, Any] | None" = None,
    inbox_consumed_ids: "list[str] | None" = None,
    subagent_refs: "list[str] | None" = None,
    pty_refs: "list[str] | None" = None,
    job_refs: "list[str] | None" = None,
    phase: str = "resume",
) -> dict[str, Any]:
    """Build the durable checkpoint payload written with a committed pair.

    The record travels on the SAME atomic transcript write as the batch it
    describes, so there is never a committed batch without a checkpoint nor a
    checkpoint whose batch is not yet durable. ``completed_message_ids`` is the
    assistant+tool-result batch that just fsynced; ``completed_effects`` is the
    ``call_id``-keyed set the successor may skip; ``pending_effect`` is the one
    tool that had not committed and is the ONLY thing a replay may re-run.

    ``phase`` is ``resume`` when the turn can be carried forward and
    ``deferred`` when it cannot (provider stream, PTY, native child,
    non-idempotent effect): a deferred record is a refusal to fake continuation,
    not a partial success.
    """

    if phase not in CHECKPOINT_PHASES:
        raise HandoffError("checkpoint phase has no contract")
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "provider": provider,
        "model": model,
        "boot_build": boot_build,
        "target_build": target_build,
        "committed_message_ids": list(committed_message_ids),
        "completed_effects": list(completed_effects or []),
        "pending_effect": pending_effect or {},
        "inbox_consumed_ids": list(inbox_consumed_ids or []),
        "subagent_refs": list(subagent_refs or []),
        "pty_refs": list(pty_refs or []),
        "job_refs": list(job_refs or []),
        "phase": phase,
    }


def effect_already_committed(checkpoint: "dict[str, Any]", call_id: str) -> bool:
    """Has this tool call already durably committed on the predecessor?

    The exactly-once test for a replayed effect: a completed effect is never
    re-run, so the successor skips it. A call WITHOUT a stable ``call_id`` cannot
    be answered here and must be treated as not-committed by the caller (an
    at-least-once result), which is the declared limit rather than a guess.
    """

    if not call_id:
        return False
    for effect in checkpoint.get("completed_effects", []):
        if isinstance(effect, dict) and effect.get("call_id") == call_id:
            return True
    return False


def resumable(checkpoint: "dict[str, Any] | None") -> bool:
    """Whether a successor may carry this turn forward at all.

    Unreadable/missing (``None``) is NOT resumable, and neither is an explicit
    ``deferred`` phase: unknown is not permission to continue, matching the
    repo's rule that an unreadable probe must not decide the action.
    """

    return bool(checkpoint) and checkpoint.get("phase") == "resume"


@dataclass(frozen=True)
class ResumePlan:
    """What a successor may do with a checkpoint, derived — never guessed.

    ``deferred`` is a refusal to claim continuation, not a partial success: it
    is set whenever the record cannot be read or says so, and in that case
    ``replay_effect`` is empty and ``skip_call_ids`` is empty, so the caller
    cannot resume work on the strength of an unreadable record.
    """

    deferred: bool
    reason: str
    skip_call_ids: frozenset[str]
    replay_effect: dict[str, Any]
    inbox_consumed_ids: frozenset[str]


def resume_plan(checkpoint: "dict[str, Any] | None") -> ResumePlan:
    """Decide the exactly-once resume: skip committed calls, replay the one
    effect that did NOT commit, deliver each inbox/job id once.

    The rules, in one place so both the runtime and its tests read them:

    * Unreadable/``None`` or phase ``deferred`` ⇒ deferred, nothing replayed.
    * A completed effect (by ``call_id``) is SKIPPED — never re-run.
    * ``pending_effect`` is replayed ONLY because it never durably committed;
      an effect with no stable ``call_id`` cannot be proven uncommitted, so it
      is deferred rather than re-run (the declared at-least-once limit).
    * Inbox ids are a set, so a duplicate drain is a no-op.
    """

    if not resumable(checkpoint):
        return ResumePlan(
            True, "checkpoint is unreadable or deferred", frozenset(), {}, frozenset()
        )
    assert checkpoint is not None  # resumable() guaranteed it
    skip = frozenset(
        effect["call_id"]
        for effect in checkpoint.get("completed_effects", [])
        if isinstance(effect, dict) and isinstance(effect.get("call_id"), str) and effect["call_id"]
    )
    pending = checkpoint.get("pending_effect") or {}
    replay: dict[str, Any] = {}
    if isinstance(pending, dict) and pending:
        call_id = pending.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            # No stable id: we cannot prove it did not commit, so we must not
            # re-run it. A non-idempotent side effect is the operator's to
            # reconcile; a silent re-run is the harm this guards against.
            return ResumePlan(
                True,
                "pending effect has no stable call_id; not replayable exactly-once",
                skip,
                {},
                frozenset(),
            )
        replay = dict(pending)
    inbox = frozenset(
        item for item in checkpoint.get("inbox_consumed_ids", []) if isinstance(item, str) and item
    )
    return ResumePlan(False, "", skip, replay, inbox)


__all__ = [
    "HANDOFF_CONTRACT",
    "HANDOFF_STATES",
    "HandoffError",
    "ListenerOwner",
    "ListenerUnsupported",
    "TransferKind",
    "UNITS",
    "Unit",
    "fenced_units",
    "pending_units",
    "receive_listener",
    "suppressed_occurrences",
    "transferable_units",
    "unit",
    "checkpoint_record",
    "effect_already_committed",
    "resumable",
    "ResumePlan",
    "resume_plan",
    "_acceptor_factory",
]
