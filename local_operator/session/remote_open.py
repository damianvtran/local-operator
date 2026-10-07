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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

if TYPE_CHECKING:
    from local_operator.resume import SessionRow
    from local_operator.session.attached import AttachedSession
    from local_operator.session.peer_rows import UnansweredPeer


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


def unresolved_peer_sentence(session_id: str, unanswered: Sequence["UnansweredPeer"]) -> str:
    """The ONE sentence for "we could not resolve that id, and a device stayed silent".

    WHY IT IS NOT THE UNREACHABLE SENTENCE, AND NOT A 404. The other refusal in
    this module names a KNOWN holder we cannot reach: the row is on the user's
    screen, so the device and the reason are facts. Here nothing names a holder
    at all — the resolution read missed AND at least one device did not answer
    it — so the two sentences must not be the same, or the surface would have to
    present a silence as a name (``mesh-ui.md`` §1.3's degraded states).

    THE THREE RULES THE COPY KEEPS, each one a way to be dishonest that was
    considered and refused:

    * the silent devices are named AS SILENT ("did not answer") — a device that
      did not reply is reported for what it did, never as a device that was
      searched and came back empty;
    * ownership is never claimed, and one device is never pinned as the holder:
      silence is not evidence about WHERE the conversation is, so the sentence
      says where it MAY be, and it says "one of them" whenever more than one
      device is silent;
    * absence is never stated: "it is not known to be gone" is the whole point
      of the state — the alternative copy is the reader's "no longer on this
      machine", which is a deletion claim nothing here can support.

    ``UNNAMED_DEVICE`` rather than a bare join: a device the membership never
    named still has to appear, and an empty string in the list would read as one
    fewer silent peer than there is.
    """
    from local_operator.resume import UNNAMED_DEVICE

    names = ", ".join(peer.name or UNNAMED_DEVICE for peer in unanswered)
    subject = "that device" if len(unanswered) == 1 else "one of them"
    return (
        f"{session_id} could not be resolved: {names} did not answer, so this "
        f"conversation may be on {subject}. That is not the same as gone — retry "
        "once the link is back; /network doctor diagnoses the link."
    )


class PeerSessionUnresolved(Exception):
    """A resolution miss where a device stayed SILENT — never "nobody holds it".

    THE STATE THIS EXISTS FOR. A miss on the peer listing answers ``None`` from
    :func:`remote_row_for`, and every surface above it turned that into the
    shared 404 — which the desktop renderer maps to ``missing``, "This
    conversation is no longer on this machine", with the composer refused. But
    a miss is only evidence of absence when every device ANSWERED the read. When
    the relay reports devices that did not reply, the same ``None`` means "we
    could not find out", and the honest answer keeps the composer open with a
    retry rather than closing the conversation the user is looking at.

    TYPED HERE RATHER THAN AS AN HTTP EXCEPTION, like :class:`PeerSessionUnreachable`
    in ``server/utils/desktop_sessions``, because the seams that raise it are not
    routes: the pool, the CLI and the TUI all reach ``open_remote_viewer``.

    WHY IT LIVES IN THIS MODULE rather than beside ``PeerSessionUnreachable``
    (design note ``mesh-wire-honesty.md`` §S2 names the type, not its file): it
    is raised at BOTH seams, and the second one is here — this module. The
    desktop server module is the HTTP layer and the tree keeps one direction of
    dependency (``session/`` never imports ``local_operator.server``; see
    ``session/store_failures.py``'s placement note for the same decision made
    the same way), so a type the session layer must raise cannot be defined in
    the server module without inventing that edge. Its sentence composer sits
    directly above it for the same reason ``unreachable_peer_sentence`` sits
    above the other refusal.
    """

    code = "session_unresolved"

    def __init__(self, session_id: str, unanswered: Sequence["UnansweredPeer"]) -> None:
        super().__init__(unresolved_peer_sentence(session_id, unanswered))
        self.session_id = session_id
        #: The devices the relay reported as not answering THIS read, carried
        #: rather than re-read: a surface that offers a retry, or names the
        #: silent devices, must be looking at the same answer the miss was.
        self.unanswered = tuple(unanswered)


def remote_row_for(session_id: str, root: Path) -> "SessionRow | None":
    """The peer's row for ``session_id``, or ``None`` when it is not a peer's.

    Cache first (the producer the sidebar's poll fills), and ONE read only when
    this device holds no directory for the id — the same policy the TUI's guard
    used, so a local resume never waits on a peer.

    THAT ONE READ IS A GENUINE READ (``ttl_s=0``). A miss means the cached
    listing was read BEFORE this id could be in it — the created-on-a-peer race
    the operator hit live: the create answered with the peer's minted id, and
    every route resolving it inside the listing's TTL was answered from a
    listing the id could not be in yet, each one refusing "This conversation no
    longer exists, so your message wasn't sent" about a conversation that had
    just been created. The TTL exists to keep the sidebar's two-second poll off
    the wire; this seam is the one caller whose cached answer is guaranteed
    stale by construction, so it pays the read it always documented (the TUI's
    own post-create open already pays the same forced read — ``app.py``,
    "ONE FORCED CATALOGUE READ"). The local-directory guard above keeps that
    cost off every id this device holds.

    Blocking: call off the loop.
    """
    from local_operator.session.peer_rows import peer_session_row, peer_session_rows

    root = Path(root)
    row = peer_session_row(session_id, root)
    if row is None and not (root / "sessions" / session_id).is_dir():
        peer_session_rows(root, ttl_s=0)
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
    slash_consumers: Sequence[str] | None = None,
) -> "AttachedSession | None":
    """A COLD viewer whose owner is the peer, or ``None`` when the id is not remote.

    ``None`` MEANS "THIS DEVICE HOLDS IT, OR NOBODY DOES" and nothing weaker:
    when the id does not resolve AND a device did not answer the read that
    missed, this raises :class:`PeerSessionUnresolved` instead. A caller that read
    ``None`` as "not a peer's", while a device had not answered, would go on to
    build a LOCAL viewer — a viewer for a conversation this device does not hold,
    whose first write would engage a runtime HERE under somebody else's id (the
    two-writer case INV-1 forbids) — so the two answers must not be collapsed.

    EVERY CALLER IN THIS TREE PASSES ``row=`` TODAY, so this guards the seam's
    contract rather than a live path: it is here because ``row=None`` is the
    documented way to ask this seam to resolve the id itself, and that is the
    call whose ``None`` would be ambiguous. The callers that consult
    ``remote_row_for`` directly (``cli.py``, ``network/cli.py``,
    ``desktop_mesh.py``) are unchanged by the design note's decision, and the CLI
    shell's own ``--resume`` keeps its local fall-through for a miss.

    Cold on purpose, exactly like a local ``lop`` boot: the first act that needs
    the runtime binds it through ``RemoteOwner.engage``/``locate``, so opening a
    peer's session costs no work on the peer until the user does something.

    ``takeover`` is required by the facade and never reached: a remote owner sets
    ``_can_go_cold``, so owner loss leaves the viewer cold rather than making this
    device a second writer (INV-1).

    ``slash_consumers`` is the viewer's own declaration of the action receipts it
    renders and submits (``AttachedSession``'s constructor argument of the same
    name). ``None`` means "whatever the facade does by default" — the full attached
    vocabulary, which is right for every existing caller (this function has four
    others: the TUI, the desktop server, ``lop --resume`` and the session factory,
    and three of them render a receipt's request themselves). A ONE-SHOT caller
    passes ``()`` instead, which declares nothing, so the owner runs the turn
    itself rather than standing down for a viewer that will print the receipt and
    exit (review round 1, MAJOR-2).
    """
    from local_operator.network.projection import PeerRow, remote_owner_for
    from local_operator.session.attached import (
        ATTACHED_SLASH_CONSUMERS,
        AttachedSession,
    )

    config_dir = Path(config_dir)
    if row is None:
        row = await asyncio.to_thread(remote_row_for, session_id, config_dir)
    if row is None:
        # A MISS IS NOT AN ABSENCE UNLESS SOMEBODY ANSWERED. ``remote_row_for``
        # ends in a GENUINE read (``ttl_s=0``), so the read that missed is the
        # read whose silence is consulted here: ``unanswered_peers`` rides the
        # same cache entry the rows were written under, which is why this check
        # costs no second dial on the path that just dialled. Without it this
        # seam's ``None`` means "not a peer's" when the truth may be "no peer
        # said" — and a caller that read it that way would build a LOCAL viewer
        # for somebody else's id, the two-writer case INV-1 forbids.
        from local_operator.session.peer_rows import unanswered_peers

        silent = await asyncio.to_thread(unanswered_peers, config_dir)
        if silent:
            raise PeerSessionUnresolved(session_id, silent)
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
        # ``None`` is THIS function's default (every caller that does not care),
        # so it resolves to the facade's own default rather than to "declare
        # nothing": the four other callers must keep the behaviour they have, and
        # three of them submit a receipt's request themselves — declaring nothing
        # for them would run the command twice.
        slash_consumers=(
            ATTACHED_SLASH_CONSUMERS if slash_consumers is None else tuple(slash_consumers)
        ),
    )
