"""Open a session whose runtime lives on ANOTHER device, as an ordinary viewer.

WHY THIS EXISTS (mesh build plan §0, finding 2). ``projection.remote_owner_for``
and ``RemoteSessionClient`` were complete and reached only from tests, so every
surface answered a peer's session with a sentence ("<id> is running on <dev>")
instead of opening it — an offload you cannot keep driving is not an offload.
The facade seam was already there (``AttachedSession.cold(..., owner=, seed=)``,
``mesh-session-mobility.md`` §3.1): this module is the one place a caller turns
"that id is a peer's row" into that facade, so the TUI's sidebar pick,
``/resume``, the shared session factory and a ``/move`` that just committed to a
peer all build the SAME viewer rather than four spellings of it.

ZERO-PEER COST (R16 topology 0). Every entry point here answers ``None`` before
touching the mesh when this device has no relay running, and a session this
device holds is never asked about at all — the local path does not pay a dial.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from local_operator.resume import SessionRow
    from local_operator.session.attached import AttachedSession


def unreachable_peer_sentence(session_id: str, row: "SessionRow") -> str:
    """The ONE sentence for "that conversation is on a device we cannot reach".

    WHY IT IS A FUNCTION AND NOT A STRING AT EACH SURFACE. The TUI refuses the
    pick with it and the desktop backend answers the same state with it, and the
    requirement is that one situation is not described two ways on two surfaces
    (``mesh-ui.md`` §1.3's degraded states). The peer's own name, the reason in
    words and the command that diagnoses the link are three facts that came
    apart the first time they were written twice — so they are composed here, in
    the module that already owns "a peer row becomes a viewer".

    ``peer_reason_words`` rather than the raw token: the transport's reason is a
    token (``connect_failed:ConnectionRefusedError``), and a user surface that
    printed it named a Python class at the operator.
    """
    from local_operator.resume import UNNAMED_DEVICE, peer_reason_words

    device = row.owner_label or UNNAMED_DEVICE
    return (
        f"{session_id} is on {device}, which is unreachable "
        f"({peer_reason_words(row.unreachable_reason)}). /network doctor "
        f"{row.owner_device_name or row.owner_device} diagnoses the link."
    )


def remote_row_for(session_id: str, root: Path) -> "SessionRow | None":
    """The peer's row for ``session_id``, or ``None`` when it is not a peer's.

    Cache first (the producer the sidebar's poll fills), and ONE read only when
    this device holds no directory for the id — the same policy the TUI's guard
    used, so a local resume never waits on a peer. Blocking: call off the loop.
    """
    from local_operator.session.peer_rows import peer_session_row, peer_session_rows

    root = Path(root)
    row = peer_session_row(session_id, root)
    if row is None and not (root / "sessions" / session_id).is_dir():
        peer_session_rows(root)
        row = peer_session_row(session_id, root)
    if row is None or not row.is_remote or not row.owner_device:
        return None
    return row


async def open_remote_viewer(
    session_id: str,
    *,
    config_dir: Path,
    takeover: Callable[[], Awaitable[Any]],
    row: "SessionRow | None" = None,
    surface: str = "terminal",
) -> "AttachedSession | None":
    """A COLD viewer whose owner is the peer, or ``None`` when the id is not remote.

    Cold on purpose, exactly like a local ``lop`` boot: the first act that needs
    the runtime binds it through ``RemoteOwner.engage``/``locate``, so opening a
    peer's session costs no work on the peer until the user does something.

    ``takeover`` is required by the facade and never reached: a remote owner sets
    ``_can_go_cold``, so owner loss leaves the viewer cold rather than making this
    device a second writer (INV-1).
    """
    from local_operator.network.projection import PeerRow, remote_owner_for
    from local_operator.session.attached import AttachedSession

    config_dir = Path(config_dir)
    if row is None:
        row = await asyncio.to_thread(remote_row_for, session_id, config_dir)
    if row is None:
        return None
    peer_row = PeerRow(
        session_id=session_id,
        device_id=row.owner_device,
        device_name=row.owner_device_name,
        conversation_name=row.name,
        reachable=row.reachable,
    )
    owner = await asyncio.to_thread(
        remote_owner_for, session_id, config_dir=config_dir, row=peer_row, root=config_dir
    )
    return await AttachedSession.cold(
        session_id,
        config_dir=config_dir,
        # NO LOCAL PATH: a directory on this machine means nothing on the peer, and
        # an empty cwd lets the peer default to its own (§5.3 step 2).
        cwd="",
        takeover_factory=takeover,
        surface=surface,
        owner=owner,
        seed=owner.seed(),
    )
