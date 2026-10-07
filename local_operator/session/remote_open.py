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

from local_operator.network.types import MeshRefusal

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
    # ``owner_device_name or owner_device or UNNAMED_DEVICE``: the unnamed case
    # needs the fallback HERE too, or a row that carries neither reads through as
    # ``/network doctor  diagnoses the link.`` — two spaces and no device named to
    # go with the one the first half of the sentence just supplied (design round
    # 1, D6).
    return (
        f"{session_id} is on {device}, which is unreachable "
        f"({peer_reason_words(row.unreachable_reason)}). /network doctor "
        f"{row.owner_device_name or row.owner_device or UNNAMED_DEVICE} diagnoses the link."
    )


def unresolved_peer_sentence(session_id: str, unanswered: Sequence["UnansweredPeer"]) -> str:
    """The ONE sentence for "a device did not answer, so this id did not resolve".

    WHY IT IS NOT THE UNREACHABLE SENTENCE, AND NOT A 404. The other refusal in
    this module names a KNOWN holder we cannot reach: the row is on the user's
    screen, so the device and the reason are facts. Here nothing names a holder
    at all — the resolution read missed AND at least one device did not answer
    it — so the two sentences must not be the same, or the surface would have to
    present a silence as a name (``mesh-ui.md`` §1.3's degraded states).

    THE RULES THE COPY KEEPS, each one a way to be dishonest that was considered
    and refused:

    * **the SILENCE is the headline, never the failed resolution** (design round
      1, D1). Opening with "could not be resolved" uses this product's own word
      for NOT FOUND (``mini-copy.ts`` captions its *missing* state exactly that
      way) — the reading this whole state exists to prevent, stated in the
      vocabulary for it. What the reader is told first is what the devices DID;
    * **the silent devices are named AS SILENT** ("did not answer") — a device
      that did not reply is reported for what it did, never as a device that was
      searched and came back empty;
    * **ownership is never claimed, and one device is never pinned as the
      holder**: silence is not evidence about WHERE the conversation is, so the
      sentence says where it MAY be, and it says "one of them" whenever more than
      one device is silent;
    * **absence is never stated, and never conjured to be denied** (D2). The
      sentence neither asserts the conversation is gone nor introduces the word
      "gone" in order to argue with it: the positive claim ("may be on…")
      already carries the whole truth, and the negative restatement handed the
      reader a deletion word they had no reason to hold;
    * **the action names the control, and "connection" is the plural-safe word**
      (D3): "Retry once the connection is back" points at the Retry/reconnect
      affordance the surface contract requires, and holds whether one device or
      three just went quiet, where "the link" did not.

    TWO DEGENERATE INPUTS, both refused rather than rendered:

    * an EMPTY sequence raises (D5). Both raise sites guard on non-empty, but
      this is a public composer and ``PeerSessionUnresolved`` is a public type
      whose ``str()`` is what a surface prints, so the degenerate case is refused
      here instead of composing "`<id>:  did not answer`" — no device named,
      "one of them" referring to nothing;
    * a device the membership never NAMED still appears, as ``UNNAMED_DEVICE``:
      ``peer_rows`` builds ``name=str(facts.name or "")``, so an empty name is a
      real input, and dropping it would read as one fewer silent peer than there
      is.

    THE LIST IS JOINED WITH A COORDINATOR, not commas alone (D4): device names
    are free text (``lop network join --name``; default ``socket.gethostname()``)
    and nothing sanitises them, so a device called ``build-box, spare`` would read
    as two silent devices where one was.
    """
    from local_operator.resume import UNNAMED_DEVICE

    if not unanswered:
        raise ValueError(
            "unresolved_peer_sentence needs at least one silent device; "
            "an empty list composes a sentence naming nobody"
        )
    names = [peer.name or UNNAMED_DEVICE for peer in unanswered]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    subject = "that device" if len(names) == 1 else "one of them"
    return (
        f"{session_id}: {listed} did not answer, so this conversation may be on "
        f"{subject}. Retry once the connection is back; /network doctor "
        "diagnoses the link."
    )


class PeerSessionUnresolved(MeshRefusal):
    """A resolution miss where a device stayed SILENT — never "nobody holds it".

    THE STATE THIS EXISTS FOR. A miss on the peer listing answers ``None`` from
    :func:`remote_row_for`, and every surface above it turned that into the
    shared 404 — which the desktop renderer maps to ``missing``, "This
    conversation is no longer on this machine", with the composer refused. But
    a miss is only evidence of absence when every device ANSWERED the read. When
    the relay reports devices that did not reply, the same ``None`` means "we
    could not find out", and the honest answer keeps the composer open with a
    retry rather than closing the conversation the user is looking at.

    IT RIDES THE REFUSAL FAMILY (agent review round 1, R-4), because one of its
    callers is a ``lop network`` verb: ``_pilot_act`` resolves the id *inside*
    ``open_remote_viewer`` (it passes no ``row=``) and catches only
    ``TimeoutError``/``ConnectionError``, and ``network/cli.py``'s ``main``
    re-raises anything that is not a ``MeshRefusal`` — so a bare ``Exception``
    here would surface as a traceback, the one thing that file's own comment says
    this family never produces for a refusal. As a family member it already
    carries the two halves that surface prints: the machine ``code``
    (``session_unresolved``, the same one the desktop route answers with) and the
    sentence.

    TYPED HERE, IN THE SESSION LAYER, RATHER THAN BESIDE ``PeerSessionUnreachable``
    in ``server/utils/desktop_sessions`` (design note ``mesh-wire-honesty.md`` §S2
    names the type, not its file). It is raised at THREE seams — the desktop
    pool, ``open_remote_viewer`` below, and the shell's ``lop --resume`` — and two
    of the three are this module and ``cli.py``. Homing it in the server module
    would make this module import the HTTP layer to raise its own refusal, and
    ``local_operator.server`` is the one thing the session tree does not import
    at module scope: the single existing edge is function-local
    (``session/runtime/serving.py``'s completion announce, inside the method),
    and ``session/store_failures.py``'s placement note records the same decision
    made the same way. The module-scope import this class does take
    (``network/types.py``) is the opposite direction and a leaf — stdlib plus
    ``session.runtime.types`` — and ``session/credential_binding`` already imports
    a ``network`` leaf module at module scope.

    Its sentence composer sits directly above it, for the same reason
    ``unreachable_peer_sentence`` sits above the other refusal.
    """

    def __init__(self, session_id: str, unanswered: Sequence["UnansweredPeer"]) -> None:
        super().__init__("session_unresolved", unresolved_peer_sentence(session_id, unanswered))
        self.session_id = session_id
        #: The devices the relay reported as not answering THIS read, carried
        #: rather than re-read: a surface that offers a retry, or names the
        #: silent devices, must be looking at the same answer the miss was.
        self.unanswered = tuple(unanswered)


def remote_row_and_silence(
    session_id: str, root: Path
) -> tuple["SessionRow | None", tuple["UnansweredPeer", ...]]:
    """The peer's row for ``session_id``, AND the silence from the SAME read.

    WHY BOTH HALVES COME FROM ONE CALL (agent review round 1, R-1). A caller that
    asks "is this a peer's?" and then "and did anybody stay silent?" paid TWO
    fan-outs: the second question's freshness test is `(now − the moment the first
    read STARTED) < TTL`, so a listing that spent its own documented budget
    (``relay.LISTING_CLIENT_TIMEOUT_S``, which IS ``_TTL_S``) re-dialled — and
    ``unanswered_peers`` then returned only the re-dial's silence, DISCARDING the
    rows it had just read, so an id the listing held could answer "not a peer's"
    (on the pool path: the shared 404, i.e. the renderer's deleted-conversation
    claim, about an id the second read had in hand). Reading both halves here is
    one call, one dial, one answer — true by construction rather than by a
    duration argument.

    THE POLICY IS UNCHANGED from :func:`remote_row_for`: cache first, and ONE
    genuine read (``ttl_s=0``) only when this device holds no directory for the
    id, so a local resume never waits on a peer and the created-on-a-peer race
    still resolves. The local-directory case returns ``((None, ()))`` — no read
    means no silence to report, which is exactly the residue corner's answer.

    Blocking: call off the loop.
    """
    from local_operator.session.peer_rows import peer_session_row, read_listing

    root = Path(root)
    row = peer_session_row(session_id, root)
    if row is not None:
        return _as_peer_row(row), ()
    if (root / "sessions" / session_id).is_dir():
        return None, ()
    rows, unanswered = read_listing(root, ttl_s=0)
    for candidate in rows:
        if candidate.id == session_id:
            # A row the peers answered WITH is the device answering; the same
            # read cannot also be reporting that device silent (``_read``
            # excludes answered devices from ``unanswered``), so the empty half
            # is the read's own fact rather than a convenience.
            return _as_peer_row(candidate), ()
    return None, unanswered


def _as_peer_row(row: "SessionRow") -> "SessionRow | None":
    """The row as a PEER's row, or ``None`` when it is not one."""
    if not row.is_remote or not row.owner_device:
        return None
    return row


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

    Blocking: call off the loop. A caller that also needs the silence from this
    read wants :func:`remote_row_and_silence` — asking twice is what R-1 was.
    """
    return remote_row_and_silence(session_id, root)[0]


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

    FOUR OF THE FIVE PRODUCTION CALLERS PASS ``row=`` (agent review round 1,
    R-4 — the earlier claim here that all five did was false): the TUI's pick
    (``tui/app.py``), the desktop pool, the shared session factory and the shell's
    ``--resume``. ``_pilot_act`` (``network/cli.py``) does not, so it is the one
    caller that can reach this raise — which is why the exception rides
    ``MeshRefusal`` rather than being a bare ``Exception``.

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
        # A MISS IS NOT AN ABSENCE UNLESS SOMEBODY ANSWERED, AND THE SILENCE COMES
        # FROM THE SAME READ. ``remote_row_and_silence`` performs the miss's one
        # genuine read (``ttl_s=0``) and hands back both halves of it, so this
        # costs no second dial by construction and cannot consult a silence older
        # than the read that missed (agent review round 1, R-1: the two-call shape
        # re-dialled whenever the read outlasted the TTL, and the re-dial's rows
        # were discarded). Without this the seam's ``None`` means "not a peer's"
        # when the truth may be "no peer said" — and a caller that read it that
        # way would build a LOCAL viewer for somebody else's id, the two-writer
        # case INV-1 forbids.
        row, silent = await asyncio.to_thread(remote_row_and_silence, session_id, config_dir)
        if row is None and silent:
            raise PeerSessionUnresolved(session_id, silent)
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
        # ``None`` is THIS function's default (every caller that does not care),
        # so it resolves to the facade's own default rather than to "declare
        # nothing": the four other callers must keep the behaviour they have, and
        # three of them submit a receipt's request themselves — declaring nothing
        # for them would run the command twice.
        slash_consumers=(
            ATTACHED_SLASH_CONSUMERS if slash_consumers is None else tuple(slash_consumers)
        ),
    )
