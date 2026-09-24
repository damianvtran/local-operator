"""Wire shapes for the desktop's mesh surface (``features.peers``/``session_transfer``).

THE CONTRACT IS THE RENDERER'S, frozen before this backend was written: the desktop
app (local-operator-ui ``src/shared/desktop-session-contract.ts``) consumes these
exactly, and the mesh build plan's Addendum 2 settles the three shapes the first
review found wrong (a per-DEVICE peer row, flat locality fields on a session row, a
transfer body that accepts ``request_id``). A field added here is a field that
renderer has to learn, so each one says what it may and may not claim.

REQUEST MODELS FORBID EXTRA KEYS, the rule every desktop request model on this plane
follows (``routes/desktop_sessions.Input``): a misspelt key is a 422 rather than a
silently ignored intent. That rule is exactly why ``request_id`` is DECLARED on
:class:`TransferSession` — the renderer sends one, and an undeclared key would have
refused every move (Addendum 2, C).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

#: A mesh device id (``d_`` + hex today) or a network id (``n_`` + hex). PATH-SAFE
#: rather than exact, the renderer's own rule: both reach a route path, so what
#: matters is that neither can carry ``/``, ``.`` or ``%``; the exact shapes are the
#: transport's to evolve.
MESH_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
MeshId = Annotated[str, Field(pattern=MESH_ID_PATTERN)]
RequestID = Annotated[
    str, Field(pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PeerRow(BaseModel):
    """One OTHER device, collapsed across every network this device shares with it.

    ONE ROW PER DEVICE (Addendum 2, A). The relay's peer table is one entry per
    network MEMBERSHIP, so a device in two networks arrived twice and the sidebar
    drew two sections holding the same chats. Membership-per-network is the
    Networks tab's question (:class:`NetworkTopology`), not this one.
    """

    device_id: str
    #: ``""`` when the device never told us a name; the renderer shows the id's tail.
    name: str
    #: True when ANY shared network reached it on this read.
    reachable: bool
    #: The reason in WORDS (``resume.peer_reason_words``), ``""`` when reachable.
    #: The relay's raw token stays in ``lop network peers --json``.
    unreachable_reason: str
    #: The newest ``last_seen_at`` any shared network recorded, or ``None``.
    last_seen_at: float | None = None
    #: Counted from the SAME cached row set ``GET /v1/desktop/sessions?include_peers``
    #: answers from, so the heading and this row cannot disagree on one screen.
    session_count: int
    #: ALWAYS ``None`` in this build — see ``utils.desktop_mesh.RTT_MS``.
    rtt_ms: float | None = None


class PeerList(BaseModel):
    peers: list[PeerRow]
    #: This device's id, when a network record names it; absent ⇒ no claim.
    self_device_id: str | None = None
    #: The session list's ``degraded`` vocabulary, mirrored because the renderer's own
    #: ``PeerList`` type requires the key. Always empty today: this route refuses (with
    #: the relay's sentence) rather than answering a partial catalogue, so there is no
    #: silently-omitted source for a token to name. Published so a client need not
    #: branch on its presence.
    degraded: list[str] = Field(default_factory=list)


class NetworkMember(BaseModel):
    """One device's membership of ONE network (per network, never collapsed)."""

    device_id: str
    name: str
    role: str
    capabilities: list[str]
    active: bool
    suspect: bool
    endpoints: list[str]
    last_seen_at: float | None = None
    reachable: bool
    #: In words, ``""`` when reachable.
    reason: str


class NetworkSummary(BaseModel):
    network_id: str
    name: str
    epoch: int
    trust: str
    members: list[NetworkMember]


class NetworkTopology(BaseModel):
    networks: list[NetworkSummary]
    #: Which member is THIS device. Optional in the contract: absent ⇒ no node is
    #: drawn as "this device", which is a missing fact rather than a wrong one.
    self_device_id: str | None = None


class InviteNetwork(_Input):
    """Mint an invite: an INVITE, never an "add" — admission is two-sided."""

    role: Literal["read", "drive", "admin"]
    #: Bind the invite to one device id, so only that device can redeem it.
    device: MeshId | None = None


class InviteReceipt(BaseModel):
    """Where the token was WRITTEN. The token itself never crosses this API."""

    token_path: str
    expires_at: float | None = None


class RemoveMember(_Input):
    """The network's NAME as the user typed it — the route checks it, not the UI."""

    confirm: str = Field(min_length=1, max_length=256)


class RemovedMember(BaseModel):
    network_id: str
    removed: str
    epoch: int


class TransferSession(_Input):
    """Move a conversation to ``to`` (a device id) or home (``"local"``)."""

    to: MeshId
    keep: StrictBool = False
    #: How long the source may take to go idle before the move refuses as busy.
    #: 300 s is the renderer's own ceiling.
    wait_s: float = Field(default=0.0, ge=0.0, le=300.0)
    #: Optional: when present the move is at-most-once per id (a retried request
    #: replays the recorded outcome instead of moving twice).
    request_id: RequestID | None = None


class TransferPhase(BaseModel):
    phase: str
    #: The OTHER end of the move — the device this session left or arrived from.
    peer: str
    #: The phase's position in the monotone phase list (prepared .25 … done 1.0);
    #: a step count, never a byte measure.
    progress: float


class TransferReceipt(BaseModel):
    """A move, reported as ONE answer when it settles (desktop IPC is not streamed)."""

    phases: list[TransferPhase]
    locality: Literal["local", "remote"]
    #: Where the session lives NOW (this device's id when ``locality == "local"``).
    owner_device: str
    #: True when the source's copy is gone (a ``move``); a ``keep`` never retires it.
    source_retired: bool
    session_id: str
    #: The id to open: equal to ``session_id`` for a move, new for ``keep``.
    new_session_id: str
    mode: Literal["move", "keep"]
    replayed: bool = False
