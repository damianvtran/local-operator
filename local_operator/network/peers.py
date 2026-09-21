"""This device's OWN view of the peers it knows, read from disk and nothing else.

WHY THIS MODULE EXISTS, and why it is not ``projection.RelayPeerCatalog``. The
peer catalogue is a LIVE read: it dials every member on every listing (its own
docstring is the authority), which is the right answer for a listing a person
asked for and the wrong one for a list that opens on a KEYSTROKE. The TUI's
``/new`` autofill opens while the user is still typing, and ``/network``'s
vocabulary is consulted by the command-route validator — neither may pay a socket
budget, and a device whose relay is down must still offer the peers it knows,
because ``/new remote <peer>`` is the one form that reaches a device the local
relay cannot dial.

So the vocabulary comes from the member lists this device already holds on disk:
one small JSON read per network, no socket, no cache to invalidate. The trade is
stated rather than hidden — a peer this device has never paired with is not in
the list (it cannot be: nothing here has heard of it), and reachability is NOT
answered here at all. ``RelayPeerCatalog.peers()`` stays the live answer for
reachability, and the surfaces that show a peer's state ask that one.

THE ORDER IS THE WIRE'S, not this module's: networks come back sorted by name
then id (``store.list_networks``), and peers keep that order within a network, so
two surfaces that both list them cannot disagree about which peer is first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.network import store
from local_operator.network.types import MemberRecord

#: What separates a peer's name from its id in the ONE token the picker inserts
#: (``devon#9f2c``). The shape's ``remote <peer>`` form accepts either half
#: alone, so this is a convenience for a name that is ambiguous across two
#: networks — never a requirement, and never the only way to spell a peer.
PEER_ID_SEPARATOR = "#"


@dataclass(frozen=True)
class KnownPeer:
    """One peer device this device has paired with, as the member list records it.

    ``network_id``/``network_name`` are the membership's, not the device's: a
    device in two networks is two entries here, and folding them would be exactly
    the collapse ``docs/design/mesh-ui.md`` §2.8.1 records against the flat peer
    catalogue — this module has no business repeating it.
    """

    device_id: str
    name: str = ""
    role: str = "drive"
    network_id: str = ""
    network_name: str = ""
    kind: str = "device"

    @property
    def label(self) -> str:
        """What to show a person: the name, else the id's tail. Never empty."""
        return self.name or (self.device_id[:8] if self.device_id else "?")

    @property
    def token(self) -> str:
        """What the picker INSERTS for this peer: the name, or the id when unnamed.

        The id an id-shaped token carries is the FULL one, because the handler
        resolves either half and a truncated id would be a token this module
        invented rather than one the store can match.
        """
        return self.name or self.device_id


def known_peers(root: Path | None = None) -> list[KnownPeer]:
    """Every peer member this device currently holds, one entry per membership.

    The local device is EXCLUDED: this is the vocabulary for "somewhere else",
    and offering oneself would let ``/new remote <this device>`` route a session
    nowhere. Removed members are excluded too (``active_members``), because the
    tombstone is the record that they are gone — an autofill that kept offering a
    revoked device would be teaching a word the handler must then refuse.
    """
    peers: list[KnownPeer] = []
    for record in store.list_networks(root):
        for member in record.active_members():
            if not _is_peer(record, member):
                continue
            peers.append(
                KnownPeer(
                    device_id=member.device_id,
                    name=member.name,
                    role=member.role,
                    network_id=record.network_id,
                    network_name=record.name,
                    kind=member.kind,
                )
            )
    return peers


def known_peer_names(root: Path | None = None) -> tuple[str, ...]:
    """The typeable tokens for ``/new remote``: names first, then ids; each sorted.

    NAMES FIRST, and that is a claim about which half a person reaches for: a name
    is what they recognise, and an id is the fallback for a device that has none
    (an invite redeemed without ``--name`` yields a member with an empty name).
    Each half is sorted so the picker's ranking starts from a stable base — two
    invocations over the same store cannot disagree.

    Deduplicated: one device in two networks offers its name twice, and the picker
    offers a peer ONCE. Which network a session lands in is the relay's business,
    not the row's, so the vocabulary does not encode a membership.
    """
    names: set[str] = set()
    ids: set[str] = set()
    for peer in known_peers(root):
        if peer.name:
            names.add(peer.name)
        if peer.device_id:
            ids.add(peer.device_id)
    return tuple(sorted(names)) + tuple(sorted(ids))


def resolve_peer(target: str, root: Path | None = None) -> list[KnownPeer]:
    """Every membership ``target`` could mean: an exact id first, then a name.

    The SAME two-step ``store.match_networks`` uses for a ``<network>`` argument,
    and for the same reason: an id is unique and a name is not, so a caller that
    wants to disambiguate has to see every match rather than whichever one a
    dictionary happened to keep. An empty result is "no such peer on this
    device", which is a sentence the CALLER owns — this returns what it found and
    nothing about what that means.
    """
    wanted = target.strip()
    if not wanted:
        return []
    named, identified = [], []
    for peer in known_peers(root):
        if peer.device_id and peer.device_id == wanted:
            identified.append(peer)
        elif peer.name and peer.name == wanted:
            named.append(peer)
    return identified or named


def split_peer_token(token: str) -> tuple[str, str]:
    """``devon#9f2c`` → ``("devon", "9f2c")``; a bare word → ``(word, "")``.

    The picker inserts the ``#`` form only when a name is ambiguous across
    networks, so the handler must be able to read both halves out of one token.
    Split on the LAST separator, so an id that itself contains one (it does not
    today — ids are 12 hex — but the split should not depend on that) still
    leaves the name intact.
    """
    head, separator, tail = token.rpartition(PEER_ID_SEPARATOR)
    if not separator or not head:
        return token.strip(), ""
    return head.strip(), tail.strip()


def peer_facts_json(peers: list[KnownPeer]) -> list[dict[str, Any]]:
    """The listing shape for a ``--json`` consumer or a test that wants a digest.

    Kept here rather than on the surfaces so the keys are stated once: the TUI's
    picker reads :attr:`KnownPeer.label` directly and does not need this, but a
    future ``lop network devices`` (see the design's §2.8 list) is exactly one
    line away once the fields have one spelling.
    """
    return [
        {
            "device_id": peer.device_id,
            "name": peer.name,
            "role": peer.role,
            "kind": peer.kind,
            "network_id": peer.network_id,
            "network_name": peer.network_name,
        }
        for peer in peers
    ]


def _is_peer(record: Any, member: MemberRecord) -> bool:
    """Whether this member is a peer of the local device rather than the local one.

    A member with NO device id is skipped: it cannot be dialled or addressed, and
    offering it would put an untypeable row in the picker. Self is matched on the
    record's own ``self_device_id``, not on the member's kind, because the record
    is the field this device wrote about itself.
    """
    if not member.device_id:
        return False
    return member.device_id != getattr(record, "self_device_id", "")
