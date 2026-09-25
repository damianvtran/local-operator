"""The remote session projection: resolving an owner, and the viewer's client.

This module is the one ``mesh-transport-identity.md`` §2.3 reserves for session
mobility, and it implements ``mesh-session-mobility.md`` §2 (resolving the owner)
and §3 (the projection) from the viewer's side.

THREE THINGS LIVE HERE, AND WHY THEY ARE ONE MODULE.

1. **``resolve_owner``** — the §2.1 first-match-wins order that answers "which
   device owns this conversation, and is it me?". It is the only place the
   question is answered, so a listing, a resume, a `/stop` and a delete all
   route the same way.
2. **``RemoteOwner``** — the injected collaborator of §3.1 for a session whose
   runtime is on a peer. It answers exactly the three questions the facade asks
   (where, how to start, how to dial) and nothing else, which is what keeps
   ``AttachedSession`` transport-agnostic: "No other member of ``AttachedSession``
   learns about the mesh" (§3.1) is the falsifiable statement a reviewer checks.
3. **``RemoteSessionClient``** — an ``AttachClient`` whose endpoint is THIS
   device's relay rather than the runtime (§3.2). Everything except ``connect``
   is inherited untouched, because every other method is a *frame*: the relay
   forwards frames, it does not implement a parallel API.

THE ZERO-PEER PROPERTY (R16, spine §10 topology 0). Nothing in this module runs
for a device with no network: a facade built with no ``owner`` uses ``LocalOwner``
(``session/owner.py``), and ``resolve_owner`` answers from the local store without
touching the relay when no catalogue is supplied. That is asserted with a spy on
the relay entry point rather than by inspection, in
``tests/unit/session/test_owner_seam.py::test_a_facade_with_no_owner_gets_the_local_one_and_records_nothing``
(this module's own test file, ``tests/unit/network/test_projection.py``, covers
the CATALOGUE — the read side — against a real relay's reply).

NAMES IT DOES NOT INVENT. The peer-side ops are the transport's
(``net_catalog``, ``net_forward``, ``net_session_move`` …) plus the three the
mobility design adds (``net_session_create``, ``net_session_engage``,
``net_session_stop``). The LOCAL ops this module drives are the transport's
§2.5 set (``stream_open``/``stream_send``/``stream_close``) plus five
``peer_*`` names declared in ``network/types.py``: the local control surface has
its own vocabulary on purpose (a name in ``LOCAL_OPS`` may never appear in
``OP_CAPABILITY``, which ``authorizer.op_tables_are_total`` enforces), and a
viewer asking its OWN relay to ask a peer is a local act with a local name.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from local_operator import paths
from local_operator.mobile.attach_client import (
    _READ_LIMIT_BYTES,
    ACK_TIMEOUT_S,
    DESKTOP_WATCH_CAPABILITY,
    EVENT_MUTE_CAPABILITY,
    EXCLUSIVE_MOVE_CAPABILITY,
    OVERSIZED_FRAME_REASON,
    AttachClient,
    _projection_from_json,
)
from local_operator.network.types import (
    PEER_NUMBER_CEILING,
    normalise_pending,
    peer_number,
    peer_whole_int,
)
from local_operator.session.owner import SessionSeed
from local_operator.session.placement import (
    SessionPlacement,
    local_placement,
    read_stamp,
)
from local_operator.session.runtime.types import SessionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The local op vocabulary this module drives (declared in network/types.py)
# ---------------------------------------------------------------------------

OP_PEER_ROWS = "peer_session_rows"
OP_PEER_FACTS = "peer_session_facts"
OP_PEER_CREATE = "peer_session_create"
OP_PEER_ENGAGE = "peer_session_engage"
OP_PEER_STOP = "peer_session_stop"
OP_STREAM_OPEN = "stream_open"
OP_STREAM_SEND = "stream_send"
OP_STREAM_CLOSE = "stream_close"

#: The tombstones file, beside the rest of the mesh state (§1.2): every device
#: that has handed a session away keeps one, so a moved-away id is answerable
#: rather than merely absent.
TOMBSTONES_NAME = "tombstones.json"

#: How long a row served from a peer's answer is honoured before it is stale.
#: The transport's cache owns the fan-out TTL (§9.4); this is the fence around
#: a row that has already been read, and it exists so a viewer can say
#: ``unreachable`` with an age rather than rendering old facts as live.
ROW_TTL_S = 24 * 3600.0

OwnerKind = Literal["local", "remote", "unreachable", "unknown"]


# ---------------------------------------------------------------------------
# Refusal codes (one place, so a --json consumer and the tests agree)
# ---------------------------------------------------------------------------

#: Codes, in the operator's language alongside. Every one refuses CLOSED: the
#: resolver answers "unknown" rather than "local" when it cannot tell, because
#: engaging a successor for an id nobody owns is how a moved-away session gets
#: resurrected locally (§2.1 step 4).
CODE_OWNERSHIP_REFUSED = "session.ownership_refused"
CODE_PEER_UNREACHABLE = "session.peer_unreachable"
CODE_NO_OWNER = "session.no_owner"
CODE_NOT_IN_NETWORK = "session.not_in_network"
CODE_RELAY_UNAVAILABLE = "session.relay_unavailable"
CODE_PEER_REFUSED = "session.peer_refused"


class ProjectionRefusal(Exception):
    """A refusal with a machine ``code`` and a sentence for the operator.

    The same two-parts discipline ``network.types.MeshRefusal`` keeps, and for
    the same reason: the code is what a ``--json`` consumer branches on and
    what the tests key on, the sentence is what a person reads.
    """

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence


#: The marker for "the row did not carry this key at all", which is a DIFFERENT answer from
#: a key that is present and unreadable (``None``, ``0``, ``""``): absent means the peer says
#: nothing about the field, so the value this client was built with survives; present means
#: the peer made a claim, and an unreadable claim falls to that field's fail-safe.
_ABSENT = object()

#: The ``started`` a peer row carries when its own value could not be read: the epoch plus
#: a second, NOT zero (QA round 2 / review round 2 m1, one slice over). ``started`` is
#: rendered through ``as_record``, which spells an absent value as ``time.time()`` — "just
#: now" — and ``0.0`` is falsy, so a garbled ``started`` used to make a peer of unknown age
#: look BRAND NEW. A real epoch is a value the renderer cannot reinterpret, and it can only
#: read as an old peer.
STARTED_UNKNOWN_S = 1.0

#: The ``age_s`` a peer row carries when its own value could not be read. The age is a
#: LOWER BOUND on staleness, so the fallback is the largest number this protocol accepts
#: (``PEER_NUMBER_CEILING``) rather than 0: a garbled age must not read as "seen seconds
#: ago". Used as both the default and the cap, so every unreadable spelling — a string, a
#: negative, a 401-digit integer, a missing key — lands on the same value.
AGE_UNKNOWN_S = float(PEER_NUMBER_CEILING)


def _peer_age(value: Any) -> float:
    """``age_s`` from a peer: unreadable, AND A CLAIMED ZERO, both mean UNKNOWN.

    WHY ZERO IS ON THE STALE SIDE (review round 3, NIT 3). Every unreadable spelling lands
    on ``AGE_UNKNOWN_S``, which renders as "last seen a very long time ago" — except ``0``,
    which parses and renders as "just now", the one reading that makes a stale peer look
    fresh. ``0`` is exactly what the pre-validator spelling ``float(value or 0.0)`` wrote
    for an age it did not know, so a mixed fleet can still send it, and a device cannot
    truthfully send it: the value is the age of the peer's process, and 0.0 s means it
    started within this microsecond. Fail safe on the ambiguous value.
    """
    age = peer_number(value, default=AGE_UNKNOWN_S, maximum=AGE_UNKNOWN_S)
    return AGE_UNKNOWN_S if age <= 0 else age


# ---------------------------------------------------------------------------
# The peer catalogue seam (the transport's network/catalog.py implements it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerFacts:
    """One peer device, as this device knows it right now."""

    device_id: str
    name: str = ""
    network_id: str = ""
    reachable: bool = False
    reason: str = ""
    age_s: float = 0.0


@dataclass(frozen=True)
class PeerRow:
    """One of a peer's session rows, RE-ADDRESSED at our relay.

    ``peers`` is the topology's peer block on the federated row (§9.2):
    ``{device_id, name, network_id, reachable, age_s}``. It is part of the row
    here rather than assembled by the caller because the merge in
    ``session/catalog.py`` must be able to file a row under a device without
    inferring anything from an id's shape.
    """

    session_id: str
    device_id: str
    device_name: str = ""
    network_id: str = ""
    conversation_name: str = ""
    cwd: str = ""
    model_label: str = ""
    pid: int = 0
    kind: str = "daemon"
    state: str = ""
    busy: bool = False
    #: WHAT a person is being waited on (``"approval"``/``"ask"``), the same
    #: ``str | None`` the LOCAL row publishes (``SessionRecord.pending``, carried
    #: through ``info/collect``): the federated row may not change a field's TYPE,
    #: because one client reads the local and the remote rows together and a bool
    #: beside a string is a second vocabulary for one fact. An absent or empty
    #: value is no claim, never ``False``.
    pending: str | None = None
    detached: bool = False
    capabilities: tuple[str, ...] = ()
    started: float = 0.0
    age_s: float = 0.0
    reachable: bool = True
    placement: SessionPlacement = field(default_factory=local_placement)
    origin: dict[str, Any] = field(default_factory=dict)
    archived: bool | None = None

    @property
    def peer_block(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.device_name,
            "network_id": self.network_id,
            "reachable": self.reachable,
            "age_s": self.age_s,
        }

    def to_record(self) -> SessionRecord:
        """A ``SessionRecord`` for the facade, carrying the PEER's facts.

        The facade needs a record because it is the shape every viewer path
        already handles (``record.pid``, ``record.cwd``, ``record.version``).
        ``control_port``/``control_key`` are left at zero DELIBERATELY: the
        owner's real control key never leaves the owning device (§3.2), and a
        zero here is a value that cannot be dialled by accident.
        """
        return SessionRecord(
            pid=self.pid or 0,
            kind=self.kind if self.kind in ("tui", "exec", "daemon") else "daemon",
            session_id=self.session_id,
            conversation_name=self.conversation_name,
            cwd=self.cwd,
            model_label=self.model_label,
            control_port=0,
            control_key="",
            started_at=self.started or time.time(),
            capabilities=list(self.capabilities),
            busy=self.busy,
            pending=self.pending,
            detached=self.detached,
        )

    @staticmethod
    def from_json(data: dict[str, Any], *, device_id: str, device_name: str = "") -> PeerRow:
        def _bool(key: str) -> bool:
            return bool(data.get(key))

        return PeerRow(
            session_id=str(data.get("session_id") or ""),
            device_id=device_id,
            device_name=device_name,
            network_id=str(data.get("network_id") or ""),
            conversation_name=str(data.get("conversation_name") or ""),
            cwd=str(data.get("cwd") or ""),
            model_label=str(data.get("model_label") or ""),
            # EVERY NUMBER HERE IS THE PEER'S CLAIM (QA round 2, one slice over:
            # ``pid: "abc"`` raised ``ValueError`` into this reader, so one listed peer
            # broke the listing for all of them, and ``pid: 10**400`` was ACCEPTED as a
            # pid). ``peer_number``/``peer_int`` are the mesh's one validator for that
            # boundary — the same two helpers the credentials slice reads its frames with.
            #
            # THE FALLBACK DIRECTION IS THE POINT: a pid that could not be read becomes 0
            # ("no pid this device can dial"), never a value that looks live; an ``age_s``
            # becomes the stalest number this protocol can carry, never a fresh-looking 0;
            # a ``started`` becomes a real epoch rather than 0 (which the record facade
            # turns into "just now").
            pid=peer_whole_int(data.get("pid"), default=0),
            kind=str(data.get("kind") or "daemon"),
            state=str(data.get("state") or ""),
            busy=_bool("busy"),
            pending=normalise_pending(data.get("pending")),
            detached=_bool("detached"),
            capabilities=tuple(str(item) for item in (data.get("capabilities") or ())),
            started=peer_number(data.get("started"), default=STARTED_UNKNOWN_S),
            age_s=_peer_age(data.get("age_s")),
            reachable=bool(data.get("reachable", True)),
            placement=SessionPlacement.from_json(data.get("placement")),
            origin=dict(data["origin"]) if isinstance(data.get("origin"), dict) else {},
            archived=(bool(data["archived"]) if isinstance(data.get("archived"), bool) else None),
        )

    def to_row_json(self) -> dict[str, Any]:
        """The federated ROW this device shows for a peer's session (§9.2/§9.3).

        ``locality`` and ``peer`` are required on a remote row and ``peer`` is
        ``null`` on a local one — the transport's shape, and the reason a client
        can group by ``peer.name`` without ever inferring remoteness from an id.
        """
        return {
            "session_id": self.session_id,
            "conversation_name": self.conversation_name,
            "cwd": self.cwd,
            "model_label": self.model_label,
            "pid": self.pid,
            "kind": self.kind,
            "state": self.state,
            "busy": self.busy,
            "pending": self.pending,
            "detached": self.detached,
            "capabilities": list(self.capabilities),
            "started": self.started,
            "age_s": self.age_s,
            "locality": "remote",
            "peer": self.peer_block,
            "placement": self.placement.to_json(),
            "origin": dict(self.origin) or None,
            "last_synced_at": None,
        }


@runtime_checkable
class PeerCatalog(Protocol):
    """The catalogue fan-out seam (§9.4), which the transport's ``catalog.py`` owns.

    This module CONSUMES it and does not re-implement its cache: the TTL cache,
    the 60 s peer-to-peer cadence and the delta op are the transport slice's, and
    a second cache here would be a second staleness rule for one fact.
    """

    def peers(self) -> list[PeerFacts]: ...

    def rows(self, device_id: str = "") -> list[PeerRow]: ...

    def row(self, session_id: str) -> PeerRow | None: ...


class RelayPeerCatalog:
    """A LIVE read-through of this device's relay: one ``net_catalog`` fan-out.

    Deliberately not a cache. The transport owns ``network/catalog.py`` and its
    24 h/2 s staleness rules, and this class is the seam it will implement; until
    then a live read is the honest answer — every row it returns is one the peer
    just answered, which is strictly fresher than a cached row and never claims
    more than it can prove. A peer that does not answer contributes no row at
    all rather than a stale one (§8.3).
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._rows: list[PeerRow] | None = None

    # -- the relay call -----------------------------------------------------

    def _call(self, op: str, **fields: Any) -> dict[str, Any] | None:
        """One control call to this device's relay, with the ANSWER unwrapped.

        THE REPLY IS AN ENVELOPE, NOT THE ANSWER (QA round 10, Q-R10-1). The
        relay frames every control reply as ``{"op": "ack", "req", "detail"}``
        (``relay._control_connection``), and this method used to hand that
        envelope straight back while both readers below asked IT for
        ``peers``/``sessions``. Both were ``None`` on every call, so
        ``peers()`` returned ``[]``, ``_load()`` returned ``[]``, and
        ``peer_session_rows()`` returned ``()`` on every device that has a
        relay — the remote rows the sidebar's ``⇄`` tier, the per-device
        heading and the ``/resume`` guard all read. It survived a green suite
        because every producer test stood a fake in for this class, so
        ``_call`` was never exercised against a real reply.

        Unwrapping HERE rather than in the two readers keeps one unwrap for the
        one envelope: ``_relay_call`` is this module's own spelling of "run one
        local op and give me its detail", and the session-owner half of this
        file already goes through it. A second unwrap in ``peers()`` would be
        the copy that drifts when the envelope changes.

        THE BUDGET IS THE LISTING'S, NOT THE SOCKET DEFAULT (5 s). The relay
        fans out to the peers behind this op and bounds its own probe at
        ``relay.LISTING_PROBE_BUDGET_S``; a client that gave up first would
        return "no peers" for a mesh whose peer simply took longer than the
        default — the same silent-empty answer, one layer down, and the reason
        ``network/cli.py``'s listing calls wait out the relay's own budget.
        """
        from local_operator.network import relay

        return _relay_call(self._root, op, timeout=relay.LISTING_CLIENT_TIMEOUT_S, **fields)

    # -- the protocol -------------------------------------------------------

    def peers(self) -> list[PeerFacts]:
        detail = self._call(OP_PEER_ROWS)
        if not detail:
            return []
        raw_peers = detail.get("peers")
        peers: dict[str, Any] = raw_peers if isinstance(raw_peers, dict) else {}
        return [
            PeerFacts(
                device_id=str(device_id),
                name=str(block.get("name") or ""),
                network_id=str(block.get("network_id") or ""),
                reachable=bool(block.get("reachable")),
                reason=str(block.get("reason") or ""),
                age_s=_peer_age(block.get("age_s")),
            )
            for device_id, block in peers.items()
            if isinstance(block, dict)
        ]

    def rows(self, device_id: str = "") -> list[PeerRow]:
        if self._rows is None:
            self._rows = self._load()
        if not device_id:
            return list(self._rows)
        return [row for row in self._rows if row.device_id == device_id]

    def row(self, session_id: str) -> PeerRow | None:
        for row in self.rows():
            if row.session_id == session_id:
                return row
        return None

    def _load(self) -> list[PeerRow]:
        detail = self._call(OP_PEER_ROWS)
        if not detail:
            return []
        rows: list[PeerRow] = []
        for item in detail.get("sessions") or ():
            if not isinstance(item, dict):
                continue
            raw_peer = item.get("peer")
            peer: dict[str, Any] = raw_peer if isinstance(raw_peer, dict) else {}
            rows.append(
                PeerRow.from_json(
                    item,
                    device_id=str(peer.get("device_id") or ""),
                    device_name=str(peer.get("name") or ""),
                )
            )
        return rows


# ---------------------------------------------------------------------------
# Tombstones — the third durable carrier's read side (§1.2, §6.3)
# ---------------------------------------------------------------------------


def tombstones_path(config_dir: Path | None = None) -> Path:
    from local_operator.network.identity import network_root

    root = Path(config_dir) if config_dir is not None else paths.config_dir()
    return network_root(root) / TOMBSTONES_NAME


def read_tombstones(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every resent session id → where it went. Unreadable reads as empty.

    Empty is the safe answer: a tombstone that cannot be read costs one
    "unknown" row, while a tombstone INVENTED from a corrupt file would route a
    command at a device that never held the session.
    """
    try:
        data = json.loads(tombstones_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    raw_entries = data.get("sessions")
    entries: dict[str, Any] = raw_entries if isinstance(raw_entries, dict) else data
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def write_tombstone(
    session_id: str,
    *,
    device_id: str,
    device_name: str = "",
    network_id: str = "",
    config_dir: Path | None = None,
) -> Path:
    """Record that ``session_id`` was handed to ``device_id``.

    Read-modify-write of the whole file, and that is acceptable here for the
    reason ``archived.py`` gives for refusing the same shape: this file is
    written by the ONE device that handed the session away, at the commit of a
    move, and at no other time (§6.3) — it is not a per-listing write path.
    """
    path = tombstones_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = read_tombstones(config_dir)
    entries[session_id] = {
        "device_id": device_id,
        "device_name": device_name,
        "network_id": network_id,
        "moved_at": time.time(),
    }
    path.write_text(
        json.dumps({"version": 1, "sessions": entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def forget_tombstone(session_id: str, *, config_dir: Path | None = None) -> bool:
    """Drop one tombstone (the reverse of a move). Returns whether one was there."""
    entries = read_tombstones(config_dir)
    if session_id not in entries:
        return False
    entries.pop(session_id, None)
    path = tombstones_path(config_dir)
    try:
        path.write_text(
            json.dumps({"version": 1, "sessions": entries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Resolving the owner (§2.1)
# ---------------------------------------------------------------------------


class OwnerLocation(NamedTuple):
    """Where a conversation's owner is, for a caller that may not reach it.

    ``kind`` is the ROW's classification, not ``RuntimeLocality``:
    ``"unreachable"`` and ``"unknown"`` mean "the peer has not answered" and
    "nobody we can see owns this", and the ``"another-machine"`` widening of
    §1.3 does not touch this union (nor does it replace this value).
    """

    kind: OwnerKind
    session_id: str
    device_id: str = ""
    device_name: str = ""
    record: SessionRecord | None = None
    reason: str = ""

    @property
    def is_local(self) -> bool:
        return self.kind == "local"

    @property
    def is_remote(self) -> bool:
        return self.kind == "remote"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "reason": self.reason,
        }


def self_device_id(config_dir: Path | None = None) -> str:
    """This device's mesh id, or ``""`` when it has no identity file.

    ``identity.load`` and NOT ``load_or_mint``: a listing that minted a device
    key would give a device that never joined a network an identity as a side
    effect, which is both surprising and a durable write on a read path.
    """
    try:
        from local_operator.network import identity

        root = Path(config_dir) if config_dir is not None else paths.config_dir()
        loaded = identity.load(root)
    except Exception:  # noqa: BLE001 — an unreadable identity is "no identity"
        return ""
    return str(getattr(loaded, "device_id", "") or "")


def has_local_session(config_dir: Path, session_id: str) -> bool:
    return (Path(config_dir) / "sessions" / session_id).is_dir()


def resolve_owner(
    session_id: str,
    *,
    config_dir: Path,
    catalog: PeerCatalog | None = None,
) -> OwnerLocation:
    """Which device owns ``session_id`` — first match wins (§2.1).

    1. **The local store is authoritative for "is it here"**, and the stamp is
       what stops "here" from being read as "mine": a directory whose
       ``mesh.json`` names another device is a leftover a move must not have
       left (the crash window of §6.5), and resolving it as LOCAL is exactly how
       a moved-away session gets resurrected on the device it left.
    2. **Tombstone.** A moved-away id is answerable rather than merely absent.
    3. **The peer catalogue**, if one was supplied and has answered.
    4. **Nothing** → ``unknown``. NEVER ``local``: a mesh viewer must not engage
       a successor for an id nobody owns (§2.1 step 4).

    With ``catalog=None`` — the no-network case — steps 3 cannot run, so the
    answer costs one local ``is_dir`` and one tombstone read, and no relay is
    dialled. That is the zero-peer regression, and the test asserts it with a spy
    rather than by reading this comment.
    """
    config_dir = Path(config_dir)
    if not session_id:
        return OwnerLocation(
            kind="unknown",
            session_id="",
            reason="No conversation id was given.",
        )
    self_id = self_device_id(config_dir)

    # -- 1. the local store -------------------------------------------------
    if has_local_session(config_dir, session_id):
        stamp = read_stamp(config_dir, session_id)
        home = stamp.home_device if stamp is not None else ""
        # ``home == self_id`` is the whole test, and "we have no identity" does NOT
        # make another device's stamp ours — a stamp naming somebody else is the
        # leftover a handoff leaves in its crash window (§6.5), and resolving it
        # locally is how a moved-away session gets resurrected on the device it
        # left. The one exception is a device id WE used to have: a key rotation
        # changes the fingerprint and keeps the old id on the row's
        # ``previous_ids``, so an older stamp still on disk is still ours.
        if not home or home == self_id or home in _our_previous_ids(self_id, config_dir):
            record = None
            try:
                from local_operator.mobile.attach_client import find_runtime_record

                record, _pid = find_runtime_record(config_dir, session_id)
            except Exception:  # noqa: BLE001 — a scan failure is not an ownership answer
                record = None
            return OwnerLocation(
                kind="local",
                session_id=session_id,
                device_id=self_id,
                record=record,
            )
        # A directory stamped to another device: continue to the tombstone,
        # which is what a handoff leaves behind and what settles this.

    # -- 2. tombstone -------------------------------------------------------
    entry = read_tombstones(config_dir).get(session_id)
    if entry:
        device_id = str(entry.get("device_id") or "")
        device_name = str(entry.get("device_name") or "")
        members = _member_device_ids(self_id, config_dir)
        # A CURRENT MEMBER, or nobody: a tombstone naming a device we are not (or
        # are no longer) in a network with is answered "no longer in the network",
        # which is the honest sentence and not a guess that it is still there.
        if device_id and device_id in members:
            return OwnerLocation(
                kind="remote",
                session_id=session_id,
                device_id=device_id,
                device_name=device_name,
                reason=(
                    f"This conversation was moved to {device_name or device_id}."
                    if device_id
                    else ""
                ),
            )
        return OwnerLocation(
            kind="unknown",
            session_id=session_id,
            device_id=device_id,
            device_name=device_name,
            reason=(
                f"This conversation was moved to {device_name or device_id}, which is no "
                "longer in the network."
            ),
        )

    # -- 3. the peer catalogue ---------------------------------------------
    if catalog is not None:
        row = catalog.row(session_id)
        if row is not None:
            if row.reachable:
                return OwnerLocation(
                    kind="remote",
                    session_id=session_id,
                    device_id=row.device_id,
                    device_name=row.device_name,
                    record=row.to_record(),
                )
            return OwnerLocation(
                kind="unreachable",
                session_id=session_id,
                device_id=row.device_id,
                device_name=row.device_name,
                record=row.to_record(),
                reason=(
                    f"{row.device_name or row.device_id} is not answering right now"
                    + (f" (last seen {int(row.age_s)}s ago)" if row.age_s else "")
                    + "."
                ),
            )

    # -- 4. nothing ---------------------------------------------------------
    return OwnerLocation(
        kind="unknown",
        session_id=session_id,
        reason="No device in this network holds that conversation.",
    )


def _our_previous_ids(self_id: str, config_dir: Path) -> set[str]:
    """Device ids THIS device used to have, from its own member rows.

    A key rotation mints a new fingerprint (transport §3.3) and keeps the old id
    on the member row for a bounded window, so a session stamped before a
    rotation must not become unresolvable here.
    """
    if not self_id:
        return set()
    try:
        from local_operator.network import store

        records = store.list_networks(config_dir)
    except Exception:  # noqa: BLE001 — an unreadable store means no history known
        return set()
    previous: set[str] = set()
    for record in records:
        row = record.member(self_id)
        if row is not None:
            previous.update(str(item) for item in (row.previous_ids or ()))
    previous.discard(self_id)
    return previous


def _member_device_ids(self_id: str, config_dir: Path) -> set[str]:
    """Every OTHER device in any network this device is in, plus this one.

    Read from the network records rather than from reachability: a tombstone
    naming a member that is merely offline is still a member, and answering
    "no longer in the network" for it would be both wrong and alarming.
    """
    try:
        from local_operator.network import store

        records = store.list_networks(config_dir)
    except Exception:  # noqa: BLE001 — an unreadable store is "no members known"
        return set()
    ids: set[str] = {self_id} if self_id else set()
    for record in records:
        for member in record.active_members():
            ids.add(member.device_id)
    return ids


# ---------------------------------------------------------------------------
# The remote owner and its client (§3.1, §3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteSessionFacts:
    """A peer session's facts, plus the endpoint that reaches them.

    ``control_port``/``control_key`` are the LOCAL RELAY's, because that is what
    the viewer's client dials (§3.2). The owner's real control key never leaves
    the owning device — the peer relay supplies it on its own dial.
    """

    session_id: str
    device_id: str
    device_name: str = ""
    network_id: str = ""
    pid: int = 0
    conversation_name: str = ""
    cwd: str = ""
    model_label: str = ""
    capabilities: tuple[str, ...] = ()
    protocol: int = 0
    version: str = ""
    source_ref: str = ""
    state: str = ""
    reachable: bool = True
    control_port: int = 0
    control_key: str = ""
    relay_device: str = ""

    @classmethod
    def from_row(
        cls,
        row: PeerRow,
        *,
        relay_port: int = 0,
        relay_key: str = "",
        relay_device: str = "",
    ) -> RemoteSessionFacts:
        return cls(
            session_id=row.session_id,
            device_id=row.device_id,
            device_name=row.device_name,
            network_id=row.network_id,
            pid=row.pid,
            conversation_name=row.conversation_name,
            cwd=row.cwd,
            model_label=row.model_label,
            capabilities=row.capabilities,
            state=row.state,
            reachable=row.reachable,
            control_port=relay_port,
            control_key=relay_key,
            relay_device=relay_device,
        )

    def record(self) -> SessionRecord:
        return SessionRecord(
            pid=self.pid or 0,
            kind="daemon",
            session_id=self.session_id,
            conversation_name=self.conversation_name,
            cwd=self.cwd,
            model_label=self.model_label,
            control_port=self.control_port,
            control_key=self.control_key,
            capabilities=list(self.capabilities),
            version=self.version,
            source_ref=self.source_ref,
        )

    def seed(self) -> SessionSeed:
        return SessionSeed(
            name=self.conversation_name,
            model_label=self.model_label,
            cwd=self.cwd,
            device_name=self.device_name,
        )


class RemoteOwner:
    """``SessionOwner`` for a session whose runtime lives on a peer (§3.1).

    The three questions, answered over the mesh:

    * ``locate`` asks the peer, through this device's relay, whether a runtime
      holds the session there — reproducing BOTH of
      ``find_runtime_record``'s states, because ``_bind_under_lock`` branches on
      the difference between "no owner" and "an owner that published no record".
    * ``engage`` asks the peer to make one exist. It carries NO prompt: warming
      is ``net_session_engage``'s whole job, and a prompt smuggled into an
      engage would be a second way to start a turn.
    * ``make_client`` returns ``RemoteSessionClient``, whose endpoint is the
      local relay.
    """

    def __init__(
        self,
        *,
        config_dir: Path,
        session_id: str,
        facts: RemoteSessionFacts,
        root: Path | None = None,
    ) -> None:
        self._config_dir = Path(config_dir)
        self._session_id = session_id
        self._facts = facts
        self._root = root

    @property
    def facts(self) -> RemoteSessionFacts:
        return self._facts

    @property
    def placement(self) -> SessionPlacement:
        """``peer``, with the owning device and its network (§5.2).

        A remote owner is by definition not this device's placement, and saying
        so here is what makes ``AttachedSession.runtime_locality`` answer
        ``"another-machine"`` without the facade learning a second fact.
        """
        return SessionPlacement(
            mode="peer",
            network_id=self._facts.network_id,
            home_device=self._facts.device_id,
        )

    def seed(self) -> SessionSeed:
        return self._facts.seed()

    def locate(self) -> tuple[SessionRecord | None, int | None]:
        """The peer's answer, re-addressed (§3.4).

        ``(facts, pid)`` when the peer reports a runtime, ``(None, None)`` when
        it reports none, and ``(None, pid)`` when it reports a claim with no
        published record — the third registry state the bind loop distinguishes.
        A peer that cannot be asked raises nothing: it degrades to "no owner",
        which is what the bind loop's retry is for, and the row's own
        ``unreachable`` classification is what a surface reports to the user.
        """
        detail = _relay_call(
            self._root, OP_PEER_FACTS, peer=self._facts.device_id, session_id=self._session_id
        )
        if not detail:
            return None, None
        pid = detail.get("pid")
        pid_value = int(pid) if isinstance(pid, int) and pid > 0 else None
        if not detail.get("owned"):
            return None, pid_value
        row = detail.get("record")
        if not isinstance(row, dict):
            return (None, pid_value) if pid_value else (None, None)
        facts = _refresh_facts(self._facts, row)
        self._facts = facts
        if not detail.get("published", True):
            return None, pid_value
        return facts.record(), pid_value or facts.pid or None

    async def engage(self, *, cwd: str, warm: Any = None, **kwargs: Any) -> None:
        """Ask the peer to warm the session. Carries no prompt (§2.2).

        ``cwd`` is forwarded only when the CALLER supplied one: a remote viewer
        must not invent a path on another machine, and the peer defaults to its
        own home when the frame omits it (§5.3 step 2). ``warm`` is ignored by
        design — its fields are the local spawn's model choice, and the peer
        decides its own.
        """
        detail = _relay_call(
            self._root,
            OP_PEER_ENGAGE,
            peer=self._facts.device_id,
            session_id=self._session_id,
            cwd=cwd or "",
        )
        if detail is None:
            raise ProjectionRefusal(
                CODE_RELAY_UNAVAILABLE,
                "This device's network relay is not answering, so the session on "
                f"{self._facts.device_name or self._facts.device_id} cannot be started.",
            )
        if not detail.get("engaged"):
            raise ProjectionRefusal(
                CODE_PEER_REFUSED,
                str(detail.get("detail") or detail.get("message") or "the peer refused"),
            )

    def make_client(self, on_projection: Any, on_disconnected: Any, **kwargs: Any) -> Any:
        return RemoteSessionClient(on_projection, on_disconnected, facts=self._facts, **kwargs)


def _refresh_facts(facts: RemoteSessionFacts, row: dict[str, Any]) -> RemoteSessionFacts:
    """The peer's live answer, folded into the facts a client was built from.

    ``replace`` rather than a hand-listed construction: every field the row does
    not carry has to survive the refresh, and a listed copy is a field silently
    dropped the next time one is added.
    """
    # A KEY THAT IS PRESENT IS THE PEER'S CLAIM, whatever it holds; a key that is ABSENT is
    # not a claim at all. The old ``row.get("pid") or facts.pid or 0`` collapsed the two, so
    # a row that explicitly said ``pid: 0`` or ``pid: null`` (this session has no pid here)
    # resurrected the pid from the previous refresh — a value that was true minutes ago and
    # may look live now.
    row_pid = row.get("pid", _ABSENT)
    row_protocol = row.get("protocol", _ABSENT)
    return replace(
        facts,
        # ABSENT IS NOT UNREADABLE (the distinction the old ``or`` chain lost): a row that
        # does not restate ``pid`` leaves the value this client was built with, while a row
        # that restates it with something that cannot be read CLEARS it — an unreadable pid
        # must not leave the previous one looking live, and a 401-digit one must not become
        # a pid at all.
        pid=peer_whole_int(facts.pid if row_pid is _ABSENT else row_pid, default=0),
        conversation_name=str(row.get("conversation_name") or facts.conversation_name),
        cwd=str(row.get("cwd") or facts.cwd),
        model_label=str(row.get("model_label") or facts.model_label),
        capabilities=tuple(str(item) for item in (row.get("capabilities") or facts.capabilities)),
        # Same rule, and here the direction matters for a CHOICE: the attach path refuses a
        # peer below protocol 2, so an unreadable revision falls to 0 — the oldest thing
        # this protocol can be — rather than to whatever number would select the newest
        # path.
        protocol=peer_whole_int(
            facts.protocol if row_protocol is _ABSENT else row_protocol, default=0
        ),
        state=str(row.get("state") or facts.state),
        reachable=True,
    )


def remote_owner_for(
    session_id: str,
    *,
    config_dir: Path,
    row: PeerRow,
    root: Path | None = None,
) -> RemoteOwner:
    """Build the remote owner for one row, with the RELAY's endpoint filled in.

    The endpoint is read here rather than passed by the caller because a caller
    that had to supply it would be a caller that knows where this device's relay
    is — which is the one fact the projection exists to keep in one place. A
    device with no relay running gets zeroes, which is a client that cannot
    connect rather than one that connects somewhere unintended.
    """
    from local_operator.network import store

    relay_record = store.find_own_relay(root)
    facts = RemoteSessionFacts.from_row(
        row,
        relay_port=int(getattr(relay_record, "control_port", 0) or 0),
        relay_key=str(getattr(relay_record, "control_key", "") or ""),
        relay_device=str(getattr(relay_record, "device_id", "") or ""),
    )
    return RemoteOwner(config_dir=config_dir, session_id=session_id, facts=facts, root=root)


class RemoteSessionClient(AttachClient):
    """An ``AttachClient`` whose endpoint is the LOCAL relay, not the runtime.

    Only ``connect`` is overridden. Everything else — ``prompt``, ``steer``,
    ``abort``, ``slash_result``, ``approval_answer``, ``ask_answer``,
    ``history_page``, ``set_model``, ``set_effort``, ``complete_aside``,
    ``credential``, ``variables``, ``job_trajectory`` — is inherited untouched,
    because it is a FRAME and the relay forwards frames (§3.2). That inheritance
    is the whole point of R-IF-1's pass-through mode: a second client would be a
    second front-end path, which the spine forbids.

    TWO FACTS THIS CLASS RESPECTS, both from §3.2's diagram:

    * the viewer's per-connection auth fields (``events``, ``frontend_state``,
      ``display_window``, ``slash_consumers``, ``surface``) ride the
      ``stream_open`` frame, not a second auth frame — they are per-connection
      facts (§3.5) and the peer relay needs them to build ITS dial;
    * the owner-side control KEY is never sent by this client. The frame carries
      no key at all, and the peer relay supplies it on the dial it makes — so
      this process cannot leak a key it was never given.
    """

    def __init__(
        self,
        on_projection: Any,
        on_disconnected: Any,
        *,
        facts: RemoteSessionFacts,
        **kwargs: Any,
    ) -> None:
        super().__init__(on_projection, on_disconnected, locality="remote", **kwargs)
        self._facts = facts
        self._stream_id = ""

    async def connect(self, record: SessionRecord, session_id: str) -> None:
        """Open a forwarded stream to the owner and consume its welcome.

        The gates mirror ``AttachClient.connect``'s, one for one, because they
        are about the OWNER's build and the owner is the peer's runtime: a
        desktop surface still needs ``DESKTOP_WATCH_CAPABILITY``, the protocol
        still has to be >= 2, and ``display_window`` is still negotiated against
        the owner's advertised capabilities. A remote dial that skipped them
        would fail differently from a local one for no reason a user could see.
        """
        capabilities = record.capabilities or list(self._facts.capabilities)
        if self._surface == "desktop" and DESKTOP_WATCH_CAPABILITY not in capabilities:
            raise ConnectionError("This session needs a runtime update before desktop attachment.")
        if record.protocol < 2:
            raise ConnectionError(f"owner runs protocol v{record.protocol}; attach needs >= 2")
        if not self._facts.control_port:
            raise ConnectionError(
                "This device's network relay is not running, so a session on another "
                "device cannot be reached from here."
            )
        self._session_id = session_id
        self._drain_phrase = ""
        self._attention_supported = "completion-ack-v1" in capabilities
        self._event_mute_supported = EVENT_MUTE_CAPABILITY in capabilities
        self._exclusive_move_supported = EXCLUSIVE_MOVE_CAPABILITY in capabilities
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", int(self._facts.control_port), limit=_READ_LIMIT_BYTES
            )
        except OSError as exc:
            raise ConnectionError(f"the local relay is unreachable: {exc}") from exc
        self._reader = reader
        self._writer = writer
        auth: dict[str, Any] = {}
        if self._events:
            auth["events"] = True
        if self._frontend_state:
            auth["frontend_state"] = True
        if self._display_window and "display-history-window-v1" in capabilities:
            auth["display_window"] = True
            if "display-history-audit-v1" in capabilities:
                auth["display_history_audit"] = True
        if self._slash_consumers is not None:
            auth["slash_consumers"] = list(self._slash_consumers)
        if self._surface == "desktop":
            auth["surface"] = "desktop"
        writer.write(json.dumps({"key": self._facts.control_key, "client": "cli"}).encode() + b"\n")
        writer.write(
            json.dumps(
                {
                    "op": OP_STREAM_OPEN,
                    "req": 1,
                    "peer": self._facts.device_id,
                    "session_id": session_id,
                    "auth": auth,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        try:
            raw_opening = await asyncio.wait_for(reader.readline(), ACK_TIMEOUT_S)
            opening = json.loads(raw_opening.decode())
        except TimeoutError as exc:
            raise ConnectionError("the local relay did not answer") from exc
        except ValueError as exc:
            raise ConnectionError("the local relay sent a malformed frame") from exc
        if not isinstance(opening, dict) or opening.get("op") != "ack":
            raise ConnectionError(
                str((opening or {}).get("message") or "the relay refused to open that stream")
            )
        raw_detail = opening.get("detail")
        detail: dict[str, Any] = raw_detail if isinstance(raw_detail, dict) else {}
        self._stream_id = str(detail.get("stream") or "")
        # From here the socket IS the session pipe: the next frame the owner
        # sends is its welcome projection, exactly as it is for a local attach.
        # The identity check below is unchanged (attach_client.py's connect
        # documents it) — which is the whole reason the relay may not merge two
        # viewers onto one upstream connection.
        try:
            first = await asyncio.wait_for(reader.readline(), timeout=ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ConnectionError("the remote owner did not send its state") from exc
        except ValueError as exc:
            raise ConnectionError(OVERSIZED_FRAME_REASON) from exc
        if not first:
            raise ConnectionError(
                f"the peer serving that session closed the stream "
                f"({self._facts.device_name or self._facts.device_id})"
            )
        try:
            frame = json.loads(first.decode("utf-8", "replace"))
        except ValueError as exc:
            raise ConnectionError("the remote owner sent a malformed frame") from exc
        if frame.get("op") not in ("projection", "welcome"):
            raise ConnectionError(f"the remote owner replied {frame.get('op')!r}, not its state")
        projection = _projection_from_json(frame.get("data") or {}, record)
        if projection.session_id != session_id:
            raise ConnectionError(
                f"the peer moved to another conversation ({projection.session_id})"
            )
        self._connected = True
        self._reader_task = asyncio.get_running_loop().create_task(self._pump())
        self._on_projection(projection)


def _relay_call(root: Path | None, op: str, **fields: Any) -> dict[str, Any] | None:
    """Run one local op on this device's relay, or ``None`` when there is none.

    ``None`` and a refusal are kept distinct all the way up: "I could not ask"
    and "the peer said no" are different sentences to an operator, and collapsing
    them is how a broken relay reads as a refused session.
    """
    from local_operator.network import relay, store

    record = store.find_own_relay(root)
    if record is None:
        return None
    reply = relay.control_request(record, op, **fields)
    if reply is None:
        return None
    if reply.get("op") == "error":
        return {"refused": True, "detail": str(reply.get("message") or "the relay refused")}
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else {"value": detail}
