"""Session mobility across the mesh: move, recall, and archive/delete on a peer.

THIS IS THE P0 SEAM, NOT THE IMPLEMENTATION (mesh build plan §5). It exists so
three slices can build in parallel against one frozen shape:

* the relay registers ``net_session_move`` through :func:`install`, which today
  answers with the same by-name refusal the relay's fallback gives — the mobility
  slice (M) replaces the handler here, in this module, and never edits
  ``relay.py``;
* :func:`request_move` and :func:`lifecycle` are the entry points ``lop sessions
  move`` and ``lop network sessions --archive/--delete`` will call; their
  signatures are fixed now and they raise ``NotImplementedError`` until M lands;
* the ``session_move`` JSON CONTRACT below is what ``lop sessions move --json``
  prints, and it is what the TUI's ``/move --to`` (slice V) and the desktop
  transfer route (slice DB) parse. Frozen here so V and DB never wait on M.

Stdlib only, and nothing heavy at import: the relay imports this module at
construction, and the CLI will import it on ``lop sessions move``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

#: The owner-side deadline for one ``net_session_move`` request (seconds).
#:
#: ``prepare`` is the long phase: it retires the source runtime and waits for its
#: registry record to go (build plan §1.2). A busy session is REFUSED rather than
#: waited on (§1.3), so this bounds an IDLE runtime's exit plus the journal write,
#: not a turn. 90 s is the idle-exit path with room for a loaded host; past it the
#: requester is told the move did not finish and asks ``status`` before retrying.
MOVE_OP_DEADLINE_S = 90.0

# ---------------------------------------------------------------------------
# The ``session_move`` contract (frozen at P0; slices V and DB build against it)
# ---------------------------------------------------------------------------

#: The phases a move passes through, in order. MONOTONE: a result never reports
#: an earlier phase than one already reported for the same move.
#:
#: * ``prepared``    — the source retired its runtime and journalled the move;
#:                     nothing is copied yet and a rollback is still free.
#: * ``handing_off`` — the destination verified the copy and asked to commit; the
#:                     source is writing its tombstone. Not yet safe to open.
#: * ``committed``   — the DESTINATION OWNS THE SESSION. This is the phase a front
#:                     end acts on: open it on ``to_device``.
#: * ``done``        — the source confirmed its cleanup. Best-effort: a move that
#:                     stops at ``committed`` is complete from the user's side.
SessionMovePhase = Literal["prepared", "handing_off", "committed", "done"]

MOVE_RESULT_PHASES: tuple[SessionMovePhase, ...] = (
    "prepared",
    "handing_off",
    "committed",
    "done",
)

#: Phases after which the session is usable at its destination.
MOVE_OPENABLE_PHASES: frozenset[str] = frozenset({"committed", "done"})

#: ``mode``: ``move`` keeps the session id and retires the source; ``keep``
#: (``--keep``) mints a NEW id at the destination and leaves the source untouched.
SessionMoveMode = Literal["move", "keep"]

#: The refusal codes a front end may branch on. The sentence (``message``) is for
#: the person; a code outside this set is still a refusal, rendered by sentence.
MOVE_REFUSAL_CODES: frozenset[str] = frozenset(
    {
        "busy",  # the source has a turn in flight; message is its idle reason verbatim
        "viewed_elsewhere",  # another front end is attached to the source
        "unreachable",  # a device stopped answering; nothing changed
        "digest_mismatch",  # the copy did not verify; rolled back
        "revoked",  # a device lost its membership mid-move
        "not_authorised",  # a device lacks the ``move`` capability
        "not_owner",  # the source does not own the id (moved already?)
        "in_progress",  # another move of this id is under way
        "deadline_exceeded",  # the owner did not finish in time; ask status
        "not_implemented",  # this build does not move sessions yet
    }
)


class MoveDevice(TypedDict):
    """One end of a move. ``name`` is the mesh name, ``""`` when unknown."""

    device_id: str
    name: str


class MovePhaseStamp(TypedDict):
    phase: SessionMovePhase
    at: float  # unix seconds, the reporting device's clock


class SessionMoveResult(TypedDict):
    """``lop sessions move --json`` on success (exit status 0).

    ``session_id`` is the id the user asked to move; ``new_session_id`` is the id
    to OPEN — equal to ``session_id`` for ``mode == "move"``, freshly minted for
    ``keep``. ``phase`` is the last phase reached and is always in
    :data:`MOVE_OPENABLE_PHASES` on this shape; ``phases`` is the history.
    """

    ok: Literal[True]
    session_id: str
    new_session_id: str
    mode: SessionMoveMode
    from_device: MoveDevice
    to_device: MoveDevice
    phase: SessionMovePhase
    phases: list[MovePhaseStamp]


class SessionMoveRefusal(TypedDict):
    """``lop sessions move --json`` on refusal (exit status 1).

    ``phase_reached`` is ``None`` when nothing started. ``changed`` is the one
    field a front end must honour before retrying: ``False`` means nothing moved
    and a retry is safe; ``True`` means the move reached a phase it could not roll
    back from here, and the front end asks for status instead of retrying.
    """

    ok: Literal[False]
    code: str
    message: str
    session_id: str
    phase_reached: SessionMovePhase | None
    changed: bool


# ---------------------------------------------------------------------------
# Entry points (signatures frozen at P0)
# ---------------------------------------------------------------------------


def request_move(
    session_id: str,
    *,
    to: str,
    keep: bool = False,
    wait_s: float = 0.0,
    root: Path | None = None,
) -> SessionMoveResult | SessionMoveRefusal:
    """Move (or with ``keep``, copy) ``session_id`` to the device named by ``to``.

    ``to`` is a peer name or device id, or ``"local"`` to recall a session to this
    device. ``wait_s`` re-polls a busy source every 5 s for up to that long
    (``--wait N``). Returns one of the two contract shapes above and never raises
    for a refusal. Implemented by the mobility slice.
    """
    raise NotImplementedError("session moves land with the mobility slice (build plan §5, M)")


LifecycleAction = Literal["archive", "unarchive", "delete"]


def lifecycle(
    session_id: str,
    *,
    action: LifecycleAction,
    peer: str,
    confirmed: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    """Archive, restore or delete a session that lives on ``peer``.

    Runs the OWNER's own implementation (``archived.archive_change`` /
    ``cleanup.delete_session``) over ``net_session_lifecycle``; a delete without
    ``confirmed`` is the owner's dry run. Returns ``{"ok", "code", "message", ...}``
    in the family's shape. Implemented by the mobility slice.
    """
    raise NotImplementedError(
        "peer archive/delete lands with the mobility slice (build plan §5, M)"
    )


def install(server: RelayServer) -> None:
    """Register this slice's peer ops on ``server`` (see ``RelayServer.register_ops``).

    Today: ``net_session_move`` answers the by-name refusal, registered SLOW with
    its real deadline so the off-reader path and the requester's wait are the
    ones the implementation will use.
    """
    from local_operator.network.relay import not_implemented_peer_op

    server.register_ops(
        {"net_session_move": not_implemented_peer_op("net_session_move")},
        slow={"net_session_move": MOVE_OP_DEADLINE_S},
    )
