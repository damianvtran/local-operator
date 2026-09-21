"""The relay: one supervised process per install, and the membership it enforces.

WHAT IT IS, AND WHAT IT IS NOT (R1). The relay owns device identity, the network
records and their secrets, the peer listener and every link, invite state, the
authorisation decision for every inbound frame, the audit log's writer, and a
loopback control surface the CLI dials. It owns NO transcript: it never writes a
session file, never holds a session lease, never runs a turn, and it never
receives a session's control key. Its knowledge of the local session plane is a
read-through cache over ``session.runtime.registry.scan()``.

Why that distinction is enforced rather than intended: if the relay owned
sessions, two writers could exist for one transcript (the thing the lease and the
``exclusive-move-v1`` fence exist to prevent), ``lop network stop`` would be a way
to lose work, and a relay crash would orphan every session it owned on every
network.

THREADS, NOT ASYNCIO. Deliberate: the payload is a coalescible event stream rather
than a byte stream, and the CLI is synchronous, so a thread per link keeps the
whole transport in one programming model instead of two. The fleet is small by
construction (``MAX_LINKS``), each link is one reader and one writer, and the
alternative would put an event loop between the CLI and every peer.

THE LISTENER IS THE ONE NON-LOOPBACK LISTENER IN THIS TREE. That is a deliberate,
single exception to an invariant the repository states in three places, and it is
why: (a) the listener authenticates completely before any op dispatches, (b) it
never proxies a raw local control socket and never transmits a control key, (c) it
is bound by an explicit config key with a dial-only mode (``listen_address:
127.0.0.1``), and (d) its auth failure path is silence plus a local audit record —
no reply frame, no error oracle. Every other listener keeps binding loopback.

NOT IN THIS SLICE, and named so nobody assumes otherwise: ``net_sync``
(``mesh-compute-pool.md`` §7's sync primitive), ``net_broker``
(``mesh-credentials.md``), ``net_session_move`` (``mesh-session-mobility.md``
§6's mobility, the next slice), and archive/restore on a peer (its §8 — the
local implementation lives on ``feat/session-archive-delete`` and there is no
``session/archived.py`` in this build, so the op refuses BY NAME). Those ops are
AUTHORISED here — the chokepoint resolves the inner capability for
``net_forward`` — and then answered with a sentence naming the design document
that owns them, which is what keeps the seam visible instead of silent.

WHAT THIS SLICE DOES IMPLEMENT is the session plane's piloting half
(``mesh-session-mobility.md`` §2.2/§3.2/§4.3): ``net_session_create``,
``net_session_engage``, ``net_session_stop``, the ``net_forward`` carrier, the
``net_stream`` carrier that makes one viewer connection a pass-through, and the
five ``peer_*`` local ops a viewer's CLI drives."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import plistlib
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from local_operator.network import dial as session_dial
from local_operator.network import store, wire
from local_operator.network.audit import AuditEvent, AuditLog
from local_operator.network.authorizer import Authorizer, NetworkState
from local_operator.network.handshake import (
    HANDSHAKE_TIMEOUT_S,
    MAX_DECLARED_ENDPOINTS,
    REASON_SELF,
    Credential,
    Handshake,
    ListenerPolicy,
    pair_abort_frame,
    pair_result_frame,
    pair_timeout_seconds,
    sas_matches,
)
from local_operator.network.identity import (
    DeviceIdentity,
    IdentityUseTracker,
    load,
    load_or_mint,
    mint_instance_id,
    rotation_statement,
    verify_rotation_statement,
)
from local_operator.network.invite import (
    MintedInvite,
    claim_or_consume,
    consume,
    invite_credential_for,
    inviter_prompt_for,
    mark_redeemed,
)
from local_operator.network.invite import mint as mint_invite
from local_operator.network.types import (
    NEEDS_ASK,
    LinkContext,
    LinkPhase,
    MemberRecord,
    MeshRefusal,
    NetworkRecord,
    PairDecision,
    PairingRefusal,
    PeerRecord,
    PendingPairing,
    Refusal,
    SecretState,
    capabilities_for_role,
    trust_state,
)
from local_operator.paths import config_dir, log_dir
from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S, PROTOCOL_VERSION

# ---------------------------------------------------------------------------
# Defaults, beside their readers (AGENTS.md, "Adding a configuration key")
# ---------------------------------------------------------------------------

#: The relay's bind address. Three meaningful values: ``127.0.0.1`` is DIAL-ONLY
#: (this install never accepts an inbound link), ``0.0.0.0`` accepts on every
#: interface, or a specific interface address. Dial-only is a supported
#: configuration, not a degraded one: no design here assumes inbound reachability.
#:
#: THE WIDE DEFAULT WAS CHALLENGED, and it is kept (round-3 review, MAJOR 1).
#: The argument against it: no design here assumes inbound reachability, so a
#: default that opens every interface is a wider exposure than the feature needs,
#: and any stranger on the same LAN can open a connection to it. The argument for
#: it, which wins: the mesh's PRIMARY topology is one reachable device and one
#: that is not (``mesh-transport-identity.md`` §13: the AWS instance's security
#: group is open and the laptop is dial-only), and that needs a device accepting
#: from elsewhere out of the box — a loopback default would make every mesh need a
#: hand edit on the reachable device before the first pair could complete, which
#: is exactly the friction ``lop network init`` exists to avoid. What made the
#: wide bind DANGEROUS was not the bind, it was that an unauthenticated connection
#: cost a thread with nothing bounding how many: that is now bounded at the accept
#: (``DEFAULT_MAX_HANDSHAKES``), so the exposure is one silent close per
#: connection past the cap. An operator who wants no inbound reachability at all
#: sets this to ``127.0.0.1`` and says so once.
DEFAULT_LISTEN_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 4097  # 4098 mobile, 4099 browser bridge, 4100 tunnel gateway taken
DEFAULT_ADVERTISE_HOSTS: tuple[str, ...] = ()
DEFAULT_MAX_LINKS = 32
#: Handshakes allowed IN FLIGHT at once, counted BEFORE anything is authenticated.
#:
#: ``max_links`` bounds ESTABLISHED links and cannot bound this: a connection that
#: sends nothing has no link, so before this cap N silent connections held N
#: threads and N descriptors for up to ``handshake_timeout_s`` each (round-3
#: review, MAJOR 1: 40 connections, 40 live ``mesh-handshake`` threads, 0 links).
#: The bound is smaller than ``max_links`` on purpose — an unauthenticated
#: connection has proved nothing, and the cost of refusing one is a silent close
#: that the dialling peer retries on its ordinary reconnect backoff. It is a CAP
#: and not a share, exactly as ``probe_candidates`` is: past it, a connection is
#: closed rather than queued, because a queue is the resource being defended.
DEFAULT_MAX_HANDSHAKES = 8

#: How often a SATURATED pre-auth bound is allowed to say so in the local audit log.
#:
#: The peer is told nothing (see the call site: a silent close is what keeps the port
#: from being a probe oracle), so the ONLY record of a refusal is the operator's own
#: log — and the question it answers is the one a saturated relay produces: "my peer
#: cannot connect and this device looks idle". Rate limited because the traffic shape
#: that produces a drop is a flood: one record per connection would let an
#: unauthenticated stranger churn the log an incident review needs, which is the same
#: resource the cap itself exists to defend. One record per window names the cap; the
#: repetition and the subject address describe the volume.
HANDSHAKE_CAP_NOTICE_S = 60.0
DEFAULT_AUTOSTART = True

#: Reconcile grants: at most this many per device per network per hour. "I keep
#: presenting an old epoch" is also what a replayed credential looks like.
RECONCILE_MAX_PER_HOUR = 3
RECONCILE_WINDOW_S = 3600.0

#: A second LOCAL rotation inside this window is refused (``rotation_in_progress``):
#: concurrent rotations are made unlikely before the deterministic tie-break has to
#: resolve them.
ROTATION_LOCK_S = 30.0

#: The local session catalogue is cached this long, so a sidebar polling every
#: second does not turn into a scan storm.
CATALOG_CACHE_S = 2.0

#: The audit log has one writer, and it is this process. The heartbeat reuses the
#: session runtime's interval so a reader needs ONE freshness rule for every record
#: in the tree rather than one per namespace.
HEARTBEAT_S = float(HEARTBEAT_INTERVAL_S)

#: How often ONE link answers a member-table pull (`net_member_list`).
#:
#: MEMBERSHIP IS A DISTRIBUTED FACT AND A LINK IS NOT ENOUGH ON ITS OWN. The pull
#: used to run at link ESTABLISHMENT only, on the theory that "contact
#: re-evaluates membership" — but a healthy pair holds its link open indefinitely
#: (`wire.KEEPALIVE_S` against `LINK_IDLE_S`: the writer keepalives, the peer acks,
#: and the link never idles out), so on the ordinary history of a mesh no link is
#: ever established again and nothing is ever re-pulled. A member admitted
#: afterwards stayed invisible to every device whose links predated it, for as long
#: as they stayed up, and it failed between two DIRECTLY REACHABLE devices as
#: readily as across a NAT (QA round 3, Q-R2-1: `members: 3` against `4` for four
#: minutes, on the dial-only Mac AND on an in-VPC peer).
#:
#: So a pull is due on a SCHEDULE, not on an event, and the schedule is per link:
#: an established link is asked again once this interval has passed (see
#: :meth:`RelayServer.refresh_membership`). 15 s is one heartbeat: slow enough that
#: a 32-link relay spends ~2 frames/s on table refresh, fast enough that a member
#: admitted anywhere in a mesh is visible everywhere inside one watch of a listing.
MEMBERSHIP_PULL_MIN_INTERVAL_S = 15.0

#: How often the relay walks its live links looking for a table refresh that is due.
#:
#: SHORTER THAN THE INTERVAL ABOVE ON PURPOSE: this is the granularity of "due",
#: not the rate of asking, so a link whose interval expired is re-pulled promptly
#: after a learning event instead of waiting for the next tick of a coarse clock.
#: Nothing here needs a surface to be touched: a mesh converges with no operator
#: action and no restart, which is what the round-3 watch demanded and what a
#: cadence on a LISTING cannot provide (a device nobody lists is still a member).
MEMBERSHIP_PULL_PASS_S = 5.0


#: How long ONE member-table read may wait before the refresh gives up on it.
#:
#: NOT ``op_wait_s`` (10 s), which is what this read used: it runs INSIDE the
#: relay's membership loop, and that loop is serial, so ONE peer that accepts the
#: connection and then does not answer held up every other link for ten seconds per
#: pass — with two links that is a permanently stalled refresh, which is one of the
#: ways a two-hop mesh took minutes to converge on real hardware while a
#: loopback proof converged instantly (QA round 3, Q-R2-1). Bounded, that peer costs
#: one pass and the next pass asks it again.
MEMBERSHIP_PULL_TIMEOUT_S = 4.0

#: How long a link waits for its own writer before closing anyway.
#:
#: Long enough for the frames a caller queued one line earlier to reach the socket
#: (they are small, and the writer is already awake), short enough that a peer that
#: stopped reading cannot hold a close — and its callers — past a blink. See
#: :meth:`PeerLink.close` for what losing that window cost.
CLOSE_FLUSH_S = 1.0


@dataclass(frozen=True)
class NetworkSettings:
    """The relay's configuration, with the module-level defaults above.

    ``from_config`` reads the ``network.*`` keys; a missing key means the default
    here, which is what makes the relay work before anybody edits ``config.yml``.
    The ``/settings`` registry entries for these keys live with the settings
    registry and are the settings slice's to add — see this module's report note;
    the defaults beside the readers are the code's single source of truth either
    way.
    """

    listen_address: str = DEFAULT_LISTEN_ADDRESS
    port: int = DEFAULT_PORT
    advertise_hosts: tuple[str, ...] = DEFAULT_ADVERTISE_HOSTS
    handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S
    keepalive_s: float = wire.KEEPALIVE_S
    link_idle_s: float = wire.LINK_IDLE_S
    reconnect_max_s: float = wire.RECONNECT_MAX_S
    op_wait_s: float = wire.OP_WAIT_S
    queue_frames: int = wire.QUEUE_FRAMES
    queue_bytes: int = wire.QUEUE_BYTES
    max_inflight: int = wire.MAX_INFLIGHT
    max_links: int = DEFAULT_MAX_LINKS
    max_handshakes: int = DEFAULT_MAX_HANDSHAKES
    autostart: bool = DEFAULT_AUTOSTART

    @classmethod
    def from_config(cls, root: Path | None = None) -> NetworkSettings:
        """Read ``network.*`` from the config store, defaulting to the values above.

        Through ``store.read_config``, the ONE reader this package uses, so a key
        read here and a key read by ``AuditLog`` cannot land in different places.
        """
        from functools import partial

        read = partial(store.read_config, root=root)
        hosts = read(("network", "advertise_hosts"), list(DEFAULT_ADVERTISE_HOSTS))
        return cls(
            listen_address=str(read(("network", "listen_address"), DEFAULT_LISTEN_ADDRESS)),
            port=int(read(("network", "port"), DEFAULT_PORT)),
            advertise_hosts=tuple(str(host) for host in (hosts or [])),
            handshake_timeout_s=float(
                read(("network", "handshake_timeout_s"), HANDSHAKE_TIMEOUT_S)
            ),
            keepalive_s=float(read(("network", "keepalive_s"), wire.KEEPALIVE_S)),
            link_idle_s=float(read(("network", "link_idle_s"), wire.LINK_IDLE_S)),
            reconnect_max_s=float(read(("network", "reconnect_max_s"), wire.RECONNECT_MAX_S)),
            op_wait_s=float(read(("network", "op_wait_s"), wire.OP_WAIT_S)),
            queue_frames=int(read(("network", "queue_frames"), wire.QUEUE_FRAMES)),
            queue_bytes=int(read(("network", "queue_bytes"), wire.QUEUE_BYTES)),
            max_inflight=int(read(("network", "max_inflight"), wire.MAX_INFLIGHT)),
            max_links=int(read(("network", "max_links"), DEFAULT_MAX_LINKS)),
            max_handshakes=int(read(("network", "max_handshakes"), DEFAULT_MAX_HANDSHAKES)),
            autostart=bool(read(("network", "autostart"), DEFAULT_AUTOSTART)),
        )


#: How long a LISTING may spend probing peers before it stops trying and says so.
#: A listing is a read: it must return even when one member's address is a black
#: hole, and a surface that hung for a dead peer would be worse than one that
#: reports it as unreachable. Sized to cover a handshake (``HANDSHAKE_TIMEOUT_S``)
#: on a couple of endpoints, not to wait out an unreachable fleet.
LISTING_PROBE_BUDGET_S = 12.0

#: How long ONE candidate address may take to ACCEPT a connection before it is
#: written off. Sized for the question a probe asks — "does anything answer at
#: this address?" — not for the handshake that follows it: a healthy path connects
#: in one round trip (a loopback peer in 0.03 s, measured; a transatlantic one in
#: ~0.15 s), so the only dial that spends this whole budget is an address that is a
#: black hole, where the answer is already known and the cost is what it is. It is
#: a CAP, not a share: see :func:`probe_candidates`.
PROBE_CONNECT_TIMEOUT_S = 3.0

#: The one reason a member row carries when nothing was even attempted. Its own
#: spelling because "we could not try" and "we tried and nothing answered" are
#: different operator actions, and a budget that ran out must never be reported as
#: a reachability result (QA round 2, Q-R2-2).
NOT_ATTEMPTED_REASON = "not_attempted: the listing budget ran out before this member was probed"


@dataclass(frozen=True)
class CandidateAttempt:
    """What ONE declared address did, so a report can say which address and why.

    ``detail`` is the same vocabulary a peer row's ``reason`` uses (``ok``,
    ``connect_failed:<ExceptionClass>``, ``bad_endpoint``), plus the two answers
    that exist only here: ``not_attempted`` (the deadline had already passed when
    this candidate's turn came) and ``no_answer`` (it was dialled and did not
    answer before the budget ran out).
    """

    endpoint: str
    connected: bool
    detail: str
    latency_ms: float | None = None


@dataclass(frozen=True)
class CandidateProbe:
    """The result of dialling every address one member declared, all at once."""

    #: The first socket that connected, still open and unclaimed — the caller owns
    #: it and MUST either hand it to :meth:`RelayServer.dial` or close it.
    sock: socket.socket | None
    #: The endpoint that socket is connected to (``""`` when none did).
    winner: str
    #: One entry per declared endpoint, in the order the row declares them. An
    #: endpoint whose attempt was still in flight when the budget ran out is named
    #: ``no_answer`` rather than dropped: a missing row reads as "fine".
    #: READ ``complete`` BEFORE TREATING THIS AS EVERY CANDIDATE'S ANSWER — on the
    #: early-return path a candidate can be ``no_answer`` simply because the
    #: collector stopped waiting, not because anything was wrong with the address.
    attempts: list[CandidateAttempt]
    #: True when EVERY declared endpoint reported, so ``attempts`` is the complete
    #: per-address answer and each entry is that address's own outcome. False when
    #: the probe returned at the winner (the ``wait_all=False`` path, where one
    #: answer is all the caller asked for) or hit the deadline first, in which case
    #: ``no_answer`` means "not heard from yet" and nothing more. A caller that
    #: needs every address's answer asks for ``wait_all=True`` and should find this
    #: True; a caller that only needs a link uses ``sock`` and ``reason`` and can
    #: ignore both.
    complete: bool
    #: The one line a peer row carries, and EMPTY when an address answered: the
    #: caller's success signal is the socket, and a "reason" beside it would be a
    #: sentence about a link that exists. See :func:`probe_reason`.
    reason: str


@dataclass
class _AttemptOutcome:
    """One attempt's result, as it crosses from a probe thread to the collector."""

    endpoint: str
    sock: socket.socket | None
    detail: str
    latency_ms: float


def probe_candidates(
    endpoints: Sequence[str],
    *,
    deadline: float | None,
    connect_cap: float,
    wait_all: bool = False,
) -> CandidateProbe:
    """Dial every address a member declared, ALL of them at once, and take the first
    that answers.

    WHY PARALLEL, and not "walk the row under one shared deadline". A member row
    lists every address its declaring device believes it can be reached at, and
    which of them is a black hole depends on WHO IS DIALING: a public address is
    unroutable from inside the same VPC and a private one from outside it, so for
    any given dialler one entry of the row is usually dead — and the row's order is
    the DECLARER's, which no receiver can fix unilaterally. Walking the row in
    order under one deadline therefore spends the WHOLE budget on whichever entry
    happens to be first and reports the member unreachable while its reachable
    address sits untouched in the same row; in a mesh of three, the budget spent on
    that black hole is a THIRD member never probed at all (QA round 2, Q-R2-2:
    `connect_failed:TimeoutError` for a peer whose private socket connected in
    0.00 s in the same second, and the next member reported `not probed`).

    Dividing the budget between candidates instead (candidate *i* gets
    ``remaining / candidates_left``) only trades one lost member for a shrunken
    timeout: the entries later in the row get a smaller and smaller share of the
    same clock, so a slow-but-good address is written off for the accident of being
    declared last. All attempts in flight at once costs one connect per candidate,
    gives every candidate the FULL cap rather than a shrinking share, and bounds
    the whole probe by ``deadline`` — which is the property the caller needs.

    The handshake is deliberately NOT run here: exactly one candidate wins, and the
    caller runs ONE handshake on the winning socket (``dial(..., connected=...)``).
    N handshakes would be N links to one device. ``wait_all`` keeps collecting after
    the winner so a diagnostic (``doctor``) can report every address; a listing does
    not need that and returns at the winner.

    Losing connections are closed here, under the same lock that admits them, so a
    probe that returns early cannot leak a socket that connected a moment later.
    """
    results: queue.Queue[_AttemptOutcome] = queue.Queue()
    #: Sockets this probe opened that the collector has not claimed or closed.
    opened: list[socket.socket] = []
    opened_lock = threading.Lock()
    stop = threading.Event()
    started_at = time.monotonic()

    def attempt(endpoint: str) -> None:
        budget = connect_cap
        if deadline is not None:
            budget = min(connect_cap, max(0.0, deadline - time.monotonic()))
        if budget <= 0:
            results.put(_AttemptOutcome(endpoint, None, "not_attempted", 0.0))
            return
        address, _, port_text = endpoint.rpartition(":")
        try:
            port = int(port_text)
        except ValueError:
            results.put(_AttemptOutcome(endpoint, None, "bad_endpoint", 0.0))
            return
        started = time.monotonic()
        try:
            sock = socket.create_connection((address or endpoint, port), timeout=budget)
        except OSError as exc:
            results.put(
                _AttemptOutcome(endpoint, None, f"connect_failed:{exc.__class__.__name__}", 0.0)
            )
            return
        latency = round((time.monotonic() - started) * 1000, 1)
        with opened_lock:
            if stop.is_set():
                # The probe already has its winner. A connection landing now is not
                # a second answer, it is a socket nobody will ever read.
                _close_quietly(sock)
                return
            opened.append(sock)
        results.put(_AttemptOutcome(endpoint, sock, "ok", latency))

    threads = [
        threading.Thread(target=attempt, args=(endpoint,), name="mesh-probe", daemon=True)
        for endpoint in endpoints
    ]
    for thread in threads:
        thread.start()

    attempts: list[CandidateAttempt] = []
    winner_sock: socket.socket | None = None
    winner = ""
    settled = 0
    while settled < len(threads):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            break
        try:
            outcome = results.get(timeout=remaining)
        except queue.Empty:
            break
        settled += 1
        attempts.append(
            CandidateAttempt(
                endpoint=outcome.endpoint,
                connected=outcome.sock is not None,
                detail=outcome.detail,
                latency_ms=outcome.latency_ms or None,
            )
        )
        if outcome.sock is not None and winner_sock is None:
            winner_sock, winner = outcome.sock, outcome.endpoint
            if not wait_all:
                break

    stop.set()
    with opened_lock:
        for sock in opened:
            if sock is not winner_sock:
                _close_quietly(sock)
        opened.clear()

    # An attempt still in flight when the budget ran out never reported. Name it:
    # every declared endpoint appears in the result, so a caller reading the rows
    # cannot mistake "we stopped waiting" for "this address is fine". WHICH name
    # depends on whether anything was tried at all — a probe handed a deadline that
    # had already passed has not dialled anything and must say so, rather than
    # report a reachability result it never established.
    starved = deadline is not None and deadline <= started_at
    reported = {row.endpoint for row in attempts}
    for endpoint in endpoints:
        if endpoint not in reported:
            attempts.append(
                CandidateAttempt(endpoint, False, "not_attempted" if starved else "no_answer", None)
            )

    return CandidateProbe(
        sock=winner_sock,
        winner=winner,
        attempts=attempts,
        complete=settled == len(threads),
        reason="" if winner_sock is not None else probe_reason(attempts),
    )


def probe_reason(attempts: Sequence[CandidateAttempt]) -> str:
    """The one line a peer row carries when NO candidate produced a link.

    ``no_endpoint`` and ``not_a_member`` are decided by the caller: they are facts
    about the record, not about a dial. Everything here is about the dial, and the
    distinction a reader needs is "nothing answered" versus "nothing was tried".
    A row whose candidates all failed the SAME way reports that one code — which is
    what round 1's F-2 fix documented, and for the ordinary single-address row
    (a NAT-bound device declares one address) it is byte-identical to it. A row
    where they failed DIFFERENTLY names every address with its own answer, so a
    dead lease and a wrong port do not read as one failure; and a candidate that
    was never dialled says so instead of being folded in with the ones that were.

    A candidate that ANSWERED is dropped rather than reported: a probe with a
    winner has no reason, and a caller that asks anyway must not be handed
    ``unreachable: ... ok`` — a sentence that contradicts itself.
    """
    codes = [row.detail for row in attempts if row.detail != "ok"]
    if not codes:
        return ""
    if len(set(codes)) == 1:
        return codes[0]
    return "unreachable: " + "; ".join(f"{row.endpoint} {row.detail}" for row in attempts)


def adopt_members(record: NetworkRecord, rows: Sequence[Any]) -> tuple[bool, list[str]]:
    """Take a peer's member rows into this device's record. Returns (changed, added).

    THE SAME TRUST DECISION THE JOIN ALREADY MAKES, made again where it is needed.
    A joiner adopts the inviter's full member list verbatim when it pairs
    (``network/cli.py:_persist_join``); this is that list, read again from a member
    that holds it — not a second source of truth. The wire has exactly one
    membership read (``net_member_list``, §6.4) and this is its merge.

    THE RULES, and each one is a refusal to guess:

    * A device this record has TOMBSTONED, or one named in ``removed_ids``, is
      never revived. Removal is the epoch path's decision (§8.1) and a peer's
      stale snapshot must not be able to undo it — the frame that carries a
      removal is a rotation, and it is the only one that can.
    * A row this device already holds keeps its LOCAL role, capabilities and
      public key. Authority is not something a peer gets to assert about a third
      device; the row's own declaration is the only thing a peer can relay.
    * A row this device does NOT hold is adopted as the frame carries it, exactly
      as an admission frame's rows are, because without it the member is invisible
      to every surface here — listings, dialling, and the authoriser that has to
      recognise its frames.
    * Nothing is ever REMOVED here. A peer with an older table (a device that
      joined later than we did, or one that has not seen a newcomer) is the normal
      case, and shrinking the table on the weaker evidence is how a mesh loses
      members that are still members.
    """
    changed = False
    added: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            incoming = MemberRecord.from_json(row)
        except (TypeError, ValueError):
            continue
        device_id = incoming.device_id
        if not device_id or device_id in record.removed_ids:
            continue
        existing = record.member(device_id)
        if existing is None:
            record.members.append(incoming)
            added.append(device_id)
            changed = True
            continue
        # The address list is the declaring device's OWN answer, relayed: it is
        # the only thing a third party can pass on that this device cannot see for
        # itself, and the row it lands on is the only place `_ensure_link` dials.
        if incoming.endpoints and list(existing.endpoints) != list(incoming.endpoints):
            existing.endpoints = list(incoming.endpoints)
            changed = True
    return changed, added


def advertise_endpoints(settings: NetworkSettings, *, declared: Sequence[str] = ()) -> list[str]:
    """Where peers should TRY to reach a device, in preference order.

    THE UNION THE DESIGN ASKS FOR (§10.4: "from ``network.advertise_hosts`` PLUS
    detected local addresses"). What the operator declared comes first — only they
    know about a tunnel or a public address — then where this process can actually
    be reached, and the live listen port is what that second part is built from.

    A UNION RATHER THAN THE FIRST NON-EMPTY LIST, because the two sources answer
    different questions and either can be stale. A record's declared host is
    written by the process that ran `init`/`join`, from ITS config; a relay started
    with `--port` that differs from the config listens somewhere else. Keeping only
    the declared entry made the record's one-off value hide the live port, which is
    the same "nothing can dial it" outcome by a different route (QA round 1, F-2).
    ``_ensure_link`` tries them in order, so a stale first entry costs one failed
    dial rather than a lost peer.

    Module-level rather than a ``Relay`` method because ``init`` and ``join`` need
    the same answer in the CLI process, where no relay exists — and a second
    implementation of "what do we advertise" is how the row and the peer record
    disagreed in the first place.
    """
    hosts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in hosts:
            hosts.append(text)

    for host in (*declared, *settings.advertise_hosts):
        add(host)
    if settings.listen_address == "127.0.0.1":
        # DIAL-ONLY: loopback is the honest answer, and it says "you cannot reach
        # me from another machine" rather than naming an address that only fails.
        add(f"127.0.0.1:{settings.port}")
    else:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                # ``getaddrinfo`` types its sockaddr as a union that includes the
                # AF_UNIX/AF_INET6 shapes, so the index reads as ``str | int`` even
                # though AF_INET guarantees a textual address. The cast is the
                # narrowing, not a coercion.
                address = str(info[4][0])
                if not address.startswith("127."):
                    add(f"{address}:{settings.port}")
        except OSError:
            pass
    # Bounded by the number a RECEIVER keeps, so nothing published here is dropped
    # at the other end and silently missing from the row `_ensure_link` dials.
    return hosts[:MAX_DECLARED_ENDPOINTS]


# ---------------------------------------------------------------------------
# Membership: the ONLY writer of network records
# ---------------------------------------------------------------------------


@dataclass
class RotationOutcome:
    """A rotation, and how to talk to each recipient about it."""

    epoch: int
    previous_epoch: int
    removed: list[str]
    record: NetworkRecord


def epoch_frame(
    record: NetworkRecord,
    state: SecretState,
    *,
    reason: str,
    target_device_id: str = "",
    members_digest: str = "",
) -> dict[str, Any]:
    """The ``net_epoch`` broadcast frame — WITHOUT the secret for a removed peer.

    THE CONVERGENCE RULE, ENFORCED AT THE FRAME'S CONSTRUCTION. A rotation's whole
    purpose is that a removed device must not learn the new secret, so a frame
    built for a recipient named in ``removed`` (or already tombstoned) omits it.
    Panic is the one exception: ``panic`` broadcasts a secret to every reachable
    peer because the operator has declared the network compromised and every
    receiver goes untrusted — see :func:`panic_frame`.
    """
    frame: dict[str, Any] = {
        "op": "net_epoch",
        "epoch": record.epoch,
        "previous_epoch": state.previous_epoch,
        "sequence": record.sequence,
        "rotation_id": record.rotations.get(str(record.epoch), ""),
        "members_digest": members_digest or members_digest_of(record),
        "members": [member.to_json() for member in record.members],
        "removed": list(record.removed_ids),
        "reason": reason,
    }
    member = record.member(target_device_id) if target_device_id else None
    # THE CONVERGENCE RULE: a recipient named in ``removed``, one already
    # tombstoned, or one whose row is gone gets NO secret. ``target_device_id == ""``
    # means a broadcast, which is the panic path and keeps the secret.
    if target_device_id and (
        target_device_id in record.removed_ids or member is None or not member.active
    ):
        return frame
    frame["secret"] = state.secret
    return frame


def members_digest_of(record: NetworkRecord) -> str:
    """``sha256`` over the canonical member list, so a receiver can verify it landed.

    Computed from the MEMBERSHIP fields only (not ``last_seen_at`` or
    ``duplicate_count``, which change constantly): a digest that moved every time
    somebody's link refreshed would make "did your list match mine" unanswerable.
    """
    payload = [
        {
            "device_id": member.device_id,
            "public_key": member.public_key,
            "role": member.role,
            "capabilities": sorted(member.capabilities),
            "lifecycle": member.lifecycle,
            "removed_at": member.removed_at,
        }
        for member in sorted(record.members, key=lambda row: row.device_id)
    ]
    return wire.hex64(wire.sha256(wire.canonical_json(payload)))


def membership_conflict(record: NetworkRecord, device_id: str, public_key: str = "") -> str:
    """The sentence a member list refuses with, or ``""`` when it would not refuse.

    ONE OWNER, because this refusal is raised from TWO places now: the pair listener
    decides it before it asks a human (see ``_run_pair_listener``), and :func:`admit`
    is the last line of defence for every other path that can reach an admission
    (a rotation's member list, a test harness, a future caller). Two copies of the
    sentence would be free to drift, and the sentence is the operator's remedy —
    `device_id_conflict` alone names nothing (QA round 3, Q-R3-3).

    Two refusals, and the reasons they are refusals and not merges:

    * **A burned id.** ``removed_ids`` is forever (R5): re-admitting a removed device
      would make revocation a temporary state an attacker can wait out.
    * **A conflicting id.** The same ``device_id`` presented with a DIFFERENT public
      key: an id is a name, and two keys claiming one name is either a collision or
      an attack, and neither may be absorbed silently. This is also why the id is not
      authority — the key is.
    """
    if record.is_burned(device_id):
        return (
            f"{device_id} was removed from this network; a burned id is never admitted "
            "again, however the invite is minted"
        )
    existing = record.member(device_id)
    if (
        existing is not None
        and public_key
        and existing.public_key
        and existing.public_key != public_key
    ):
        return (
            f"{device_id} is already a member with a different public key; the member "
            "list refuses to reinterpret an id"
        )
    return ""


def admit(
    record: NetworkRecord,
    *,
    device_id: str,
    public_key: str,
    name: str = "",
    role: str = "read",
    capabilities: list[str] | None = None,
    added_by: str = "",
    added_via: str = "invite",
    endpoints: list[str] | None = None,
    kind: str = "device",
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> MemberRecord:
    """Write a member row, refusing the two cases a member list must never absorb.

    The refusals themselves live in :func:`membership_conflict` — one owner for the
    sentence, because the pair listener applies the same rule before it asks a human.
    This function is the last line of defence: every admission path that is not the
    pair ceremony (a rotation's member list, a harness, a future caller) still has to
    pass through here.
    """
    moment = time.time() if now is None else now
    conflict = membership_conflict(record, device_id, public_key)
    if conflict:
        raise MeshRefusal("device_id_conflict", conflict)
    existing = record.member(device_id)
    if existing is None:
        existing = MemberRecord(device_id=device_id)
        record.members.append(existing)
    existing.public_key = public_key or existing.public_key
    existing.name = name or existing.name
    existing.kind = kind  # type: ignore[assignment]
    existing.lifecycle = "active"
    existing.role = role
    existing.capabilities = sorted(
        capabilities if capabilities is not None else capabilities_for_role(role)
    )
    existing.added_at = existing.added_at or moment
    existing.added_by = added_by or existing.added_by
    existing.added_via = added_via
    existing.endpoints = list(endpoints or existing.endpoints)
    existing.removed_at = None
    existing.removed_by = None
    if persist:
        store.save(record, root)
    return existing


def remove_member(
    record: NetworkRecord,
    state: SecretState,
    *,
    device_id: str,
    by: str,
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> RotationOutcome:
    """Tombstone a member, rotate the secret and bump the epoch (R5).

    The order matters: the tombstone is written FIRST, so the very next handshake
    from that device fails the membership check even if the rotation has not been
    delivered anywhere yet. Revocation therefore takes effect on this device
    immediately and on the others as soon as they receive the frame.
    """
    moment = time.time() if now is None else now
    member = record.member(device_id)
    if member is None:
        raise MeshRefusal("unknown_member", f"{device_id} is not a member of {record.name}")
    member.removed_at = moment
    member.removed_by = by
    member.lifecycle = "expired"
    if device_id not in record.removed_ids:
        record.removed_ids.append(device_id)
    record.pending = [entry for entry in record.pending if entry.get("device_id") != device_id]
    outcome = rotate_epoch(
        record,
        state,
        by=by,
        reason="member_removed",
        removed=[device_id],
        root=root,
        now=moment,
        persist=persist,
    )
    return outcome


def rotate_epoch(
    record: NetworkRecord,
    state: SecretState,
    *,
    by: str,
    reason: str,
    removed: list[str] | None = None,
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> RotationOutcome:
    """Mint a fresh secret, bump the epoch, record who did it.

    The secret is NEW RANDOM BYTES rather than a ratchet: a ratchet buys forward
    secrecy against an attacker who already holds a device key, and that attacker
    can read the disk anyway; a fresh random value is simpler to reason about, is
    what an operator expects from "rotate", and cannot be predicted from a leaked
    older one.

    ``rotations[epoch] = by`` is both the audit of who rotated and the tie-break
    key for the concurrent-rotation rule. The lock is written too, so a second
    local rotation inside ``ROTATION_LOCK_S`` is refused with a sentence rather
    than producing two secrets nobody can reconcile.
    """
    from secrets import token_bytes

    moment = time.time() if now is None else now
    if record.rotation_lock_until > moment:
        raise MeshRefusal(
            "rotation_in_progress",
            f"a rotation of {record.name} is already in progress; wait "
            f"{int(record.rotation_lock_until - moment)}s and try again",
        )
    previous_epoch = record.epoch
    record.epoch = record.epoch + 1
    state.rotate(wire.b64u(token_bytes(32)), record.epoch)
    record.rotations[str(record.epoch)] = by
    record.rotation_lock_until = moment + ROTATION_LOCK_S
    record.sequence += 1
    record.trust = "active"
    if persist:
        store.save(record, root)
        store.save_secrets(state, root)
    return RotationOutcome(
        epoch=record.epoch,
        previous_epoch=previous_epoch,
        removed=list(removed or []),
        record=record,
    )


@dataclass
class ApplyOutcome:
    """What happened to a received epoch frame."""

    applied: bool
    detail: str


def apply_epoch(
    record: NetworkRecord,
    state: SecretState,
    frame: dict[str, Any],
    *,
    sender_device_id: str,
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> ApplyOutcome:
    """Apply a received ``net_epoch``, or refuse it with a reason.

    Acceptance rules, in order: the sender is an ACTIVE member; the epoch is
    strictly greater than ours (which absorbs duplicates and the ordinary race
    where both sides announce the same rotation); ``rotation_id`` matches the
    sender's own device id (a peer may announce a rotation, but not one attributed
    to somebody else); the member list is internally consistent; and the digest
    matches. Then the record and secret are written atomically and the caller
    re-handshakes.

    The receiver applies the new secret ONLY IF IT IS STILL A MEMBER, because a
    newly-removed device is named in the same frame: refusing here is what stops a
    removal from handing the removed device the key that would let it read on.
    """
    moment = time.time() if now is None else now
    sender = record.member(sender_device_id)
    if sender is None or not sender.active:
        return ApplyOutcome(False, "not_a_member")
    # A rotation frame is also proof the sender is alive, so it refreshes the row's
    # liveness stamp: it is the only signal an offline-ish peer gives us, and
    # discarding it would leave `lop network show` reporting a device as unseen
    # while it is in fact delivering traffic.
    sender.last_seen_at = moment
    incoming_epoch = int(frame.get("epoch") or 0)
    if incoming_epoch <= record.epoch:
        return ApplyOutcome(False, "already_at_epoch")
    if str(frame.get("rotation_id") or "") != sender_device_id:
        return ApplyOutcome(False, "rotation_id_mismatch")
    members = frame.get("members")
    if not isinstance(members, list) or not members:
        return ApplyOutcome(False, "members_missing")
    try:
        rows = [MemberRecord.from_json(row) for row in members if isinstance(row, dict)]
    except (TypeError, ValueError):
        return ApplyOutcome(False, "members_unparsable")
    removing_us = record.self_device_id in (frame.get("removed") or [])
    inconsistent = _members_inconsistent(
        rows, self_device_id=record.self_device_id, expect_self_active=not removing_us
    )
    if inconsistent:
        return ApplyOutcome(False, inconsistent)
    if frame.get("members_digest") and str(frame["members_digest"]) != members_digest_of(
        _record_with(record, rows)
    ):
        return ApplyOutcome(False, "members_digest_mismatch")
    if removing_us:
        # This device was removed by the rotation that carries it. It must NOT
        # learn the new secret: it is not a member any more, and the design says so
        # explicitly ("a removed device does not learn the new secret").
        record.members = rows
        record.epoch = incoming_epoch
        record.rotations[str(incoming_epoch)] = sender_device_id
        record.removed_ids = sorted(set(record.removed_ids) | set(frame.get("removed") or []))
        record.stale = "refused_by_peers"
        if persist:
            store.save(record, root)
        return ApplyOutcome(False, "removed_by_this_rotation")

    material = str(frame.get("secret") or "")
    if not material:
        return ApplyOutcome(False, "secret_missing")
    record.members = rows
    record.epoch = incoming_epoch
    record.rotations[str(incoming_epoch)] = sender_device_id
    record.sequence = max(record.sequence, int(frame.get("sequence") or 0))
    record.removed_ids = sorted(set(record.removed_ids) | set(frame.get("removed") or []))
    state.rotate(material, incoming_epoch)
    if persist:
        store.save(record, root)
        store.save_secrets(state, root)
    return ApplyOutcome(True, "applied")


@dataclass
class MembershipReport:
    """What a claim about a member table rests on, per network, per report.

    A MEMBER COUNT IS AN ANSWER SOMEBODY GAVE, and until round 4 the surfaces
    presented one without saying who had been asked. `lop network ls` printed
    "3 member(s)" while the fourth device had been admitted and was known to two
    of the four devices, and `--all-peers` merged an incomplete peer set as though
    it were the whole network — the failure mode an operator acts on, which is why
    QA called it out separately from the convergence bug itself (Q-R2-1: "an
    incomplete list presented as authoritative is worse than an error").

    So every surface that reports a member count now reports this beside it: how
    many peers were asked, which answered with a table, which could not be asked
    and why, and the age of the OLDEST answer — the weakest evidence in the
    argument, because one fresh answer and one stale one is only as good as the
    stale one.

    ``complete`` is the strongest honest claim available: EVERY other active member
    answered, so the table is a merge of every member's own view. It is not a proof
    that nobody is missing — a member this device cannot reach and cannot be told
    about by anyone else is invisible to it — and that is exactly why the
    incompleteness is reported as a list of names and reasons rather than as a
    boolean.
    """

    network_id: str
    refreshed_at: float
    #: Members this device holds a live link to, so their table could be read.
    answered: list[str] = field(default_factory=list)
    #: Age of each answer, positionally matching :attr:`answered`.
    answer_ages: list[float] = field(default_factory=list)
    #: Members that could NOT answer, each with the reason the surface prints.
    silent: list[dict[str, str]] = field(default_factory=list)
    #: Members whose answer is INSIDE the pull cadence, so this pass did not ask again.
    #: They are evidence (with an age), not gaps — see :meth:`sentence`.
    not_due: list[dict[str, Any]] = field(default_factory=list)
    #: Device ids this refresh ADDED to the table — the ones that were invisible.
    learned: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.silent and bool(self.answered)

    @property
    def oldest_answer_age_s(self) -> float | None:
        return max(self.answer_ages) if self.answer_ages else None

    def to_json(self) -> dict[str, Any]:
        return {
            "network_id": self.network_id,
            "complete": self.complete,
            "answered": list(self.answered),
            "not_due": [dict(row) for row in self.not_due],
            "not_answered": [dict(row) for row in self.silent],
            "oldest_answer_age_s": self.oldest_answer_age_s,
            "learned": list(self.learned),
            "sentence": self.sentence(),
        }

    def sentence(self) -> str:
        """One sentence a person reads, and the honest one in every case."""
        age = self.oldest_answer_age_s
        age_text = "just now" if age is None or age < 1.5 else f"{int(age)}s ago"
        if self.complete:
            return f"members verified with all {len(self.answered)} peer(s) ({age_text})"
        if not self.answered:
            detail = ", ".join(f"{row['device_id']} ({row['reason']})" for row in self.silent)
            return "members NOT verified: no peer answered a table read this time" + (
                f" — {detail}" if detail else ""
            )
        detail = ", ".join(f"{row['device_id']} ({row['reason']})" for row in self.silent)
        return (
            f"members verified with {len(self.answered)} of "
            f"{len(self.answered) + len(self.silent)} peer(s) ({age_text}); "
            f"NOT verified with {detail}"
        )


#: This device's standing in one network, as its own surfaces read it.
MEMBERSHIP_STATES = ("active", "removed", "refused", "untrusted", "disconnected")


def membership_state(record: NetworkRecord) -> dict[str, Any]:
    """What this device's OWN standing in ``record`` is, and what to do about it.

    THE DEVICE THAT WAS REMOVED IS THE ONE WHOSE OWN SURFACE LIED (Q-R3-2). It held
    a full member list, `trust: "active"`, and `handshake_failed:ConnectionError`
    beside it — a healthy-looking network whose every attempt failed as a transport
    error, which is precisely the shape an operator cannot diagnose. Two facts were
    available locally and unused: our own row, which the last applied rotation
    tombstoned (`removed_at`/`removed_by`), and ``record.stale``, which the refusal
    path sets to ``refused_by_peers`` (the design's own words for it, §8.3).

    So the standing is derived HERE, once, for every surface that shows it — `ls`,
    `show` and `doctor` — because three copies of this rule would disagree, and the
    one that disagreed would be the one nobody was looking at.

    A state this device cannot establish is never claimed: with a fresh table and
    no refusal the answer is ``active``, and an unreachable PEER is not evidence
    about OUR membership (that is the peer's row, reported by `peers`).

    The remedies name what actually works. `lop network trust <net> --active`
    re-admits a network marked UNTRUSTED after a panic — it is the documented path
    for that state and it is NOT a way to restore a member that was removed: the
    removed id is burned on every device that saw the rotation (``removed_ids``,
    §4.2, "forever"), so a fresh invite cannot revive it either. Measured in round
    4: `member rm` → `trust --active` → invite → join is still refused
    `device_id_conflict`; `identity rotate` + a fresh invite is admitted.
    """
    self_row = record.self_member()
    removed = record.self_device_id in record.removed_ids or (
        self_row is not None and not self_row.active
    )
    if removed:
        by = self_row.removed_by if self_row is not None else ""
        who = ""
        if by:
            remover = record.member(by)
            who = f" (removed by {remover.name if remover else by})"
        return {
            "state": "removed",
            "removed_by": by,
            "removed_at": self_row.removed_at if self_row is not None else 0.0,
            "sentence": f"this device is no longer a member of {record.name}{who}",
            "remedies": [
                "re-join with a FRESH identity: `lop network identity rotate` on this "
                "device, then `lop network join` with a new invite from an admin — the "
                "removed id is burned on every device that saw the rotation, so no "
                "invite revives it",
                "`lop network trust <network> --active` is for a DIFFERENT state — a "
                "network marked untrusted after a panic — and does not restore a "
                "removed member",
            ],
        }
    if record.stale == "refused_by_peers":
        return {
            "state": "refused",
            "removed_by": "",
            "removed_at": 0.0,
            "sentence": (
                f"peers are refusing this device's handshakes into {record.name}: every "
                "attempt reaches the peer and is closed during the handshake with no "
                "reason given, which is what this protocol's silent refusal looks like"
            ),
            "remedies": [
                "if an admin removed this device, re-join with a fresh identity "
                "(`lop network identity rotate`, then a new invite): a removed id "
                "cannot be re-admitted",
                "if the network was marked untrusted after a panic, an admin re-admits "
                "everyone with `lop network trust <network> --active`",
            ],
        }
    if record.trust != "active":
        return {
            "state": str(record.trust),
            "removed_by": "",
            "removed_at": 0.0,
            "sentence": f"this device has {record.trust} {record.name}",
            "remedies": (
                [f"re-admit it with `lop network trust {record.name} --active`"]
                if record.trust == "untrusted"
                else ["join it again with a fresh invite"]
            ),
        }
    return {
        "state": "active",
        "removed_by": "",
        "removed_at": 0.0,
        "sentence": f"this device is an active member of {record.name}",
        "remedies": [],
    }


def _record_with(record: NetworkRecord, rows: list[MemberRecord]) -> NetworkRecord:
    """A shallow copy of ``record`` carrying ``rows`` — for digest computation only."""
    clone = NetworkRecord(
        network_id=record.network_id,
        name=record.name,
        epoch=record.epoch,
        members=rows,
    )
    return clone


def _members_inconsistent(
    rows: list[MemberRecord], *, self_device_id: str, expect_self_active: bool = True
) -> str:
    """The consistency rules a received member list must satisfy, as a reason code.

    ``expect_self_active`` is FALSE FOR EXACTLY ONE FRAME: the rotation that removes
    THIS device. The rule below exists so a peer cannot quietly age us out of our own
    network with a member list that simply omits us — but the frame that removes us
    says so OUT LOUD, in its own ``removed`` array, signed by the rotation's initiator
    with a matching ``rotation_id`` and digest, and it necessarily carries our row as
    a tombstone rather than an active member. Requiring the active row there made the
    removal frame unappliable BY THE ONE DEVICE IT WAS ADDRESSED TO: the removed
    device refused it as `self_absent_from_members` and kept reporting its old epoch,
    its old member list and `trust: active` while every attempt to reach the network
    failed as a transport error (QA round 3, Q-R3-2, measured on the wire: the
    removing device's audit shows `not_a_member` and the removed device's own record
    shows nothing at all).

    The exemption is narrow on purpose — it is granted only when this device's id is
    named in the frame's ``removed`` list, never to a frame that merely drops us —
    and every other rule still applies to that frame: the sender is an active member,
    the epoch is strictly greater, the rotation is attributed to the sender, the rows
    are internally consistent and the digest matches (§8.1 step 4).
    """
    ids = [row.device_id for row in rows]
    if len(ids) != len(set(ids)):
        return "duplicate_member_ids"
    if expect_self_active:
        mine = [row for row in rows if row.device_id == self_device_id]
        if not mine or not mine[0].active:
            # A list that does not contain us as an active member is either a mistake
            # or an attempt to age us out of our own network; both are refusals.
            return "self_absent_from_members"
    for row in rows:
        if not row.public_key:
            return "member_without_key"
        if not set(row.capabilities) <= set(capabilities_for_role("admin")):
            return "member_capability_unknown"
    return ""


def lowest_id_admin(record: NetworkRecord, *, excluding: str = "") -> str:
    """The deterministic rotator after a member leaves: the lowest-id active admin.

    Lowest id rather than an election: every remaining member computes the same
    answer from the same list, so no round trip is needed and two devices cannot
    both decide they are the rotator.
    """
    candidates = sorted(
        row.device_id
        for row in record.active_members()
        if row.role == "admin" and row.device_id != excluding
    )
    return candidates[0] if candidates else ""


def leave(
    record: NetworkRecord,
    *,
    device_id: str,
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> MemberRecord:
    """Tombstone a peer that announced its own departure (``net_leave``).

    A leave is not a reason to trust the device less, but it IS a reason to stop it
    being able to read new traffic — the leaving device still holds the old secret
    — so the caller rotates afterwards, from the lowest-id active admin.
    """
    moment = time.time() if now is None else now
    member = record.member(device_id)
    if member is None:
        raise MeshRefusal("unknown_member", f"{device_id} is not a member of {record.name}")
    member.removed_at = moment
    member.removed_by = device_id
    member.lifecycle = "expired"
    if device_id not in record.removed_ids:
        record.removed_ids.append(device_id)
    if persist:
        store.save(record, root)
    return member


def membership_marker(row: dict[str, Any]) -> str:
    """The short suffix a listing puts after a member count, so the count is never bare.

    A COUNT WITHOUT ITS PROVENANCE IS THE DEFECT (QA round 3, Q-R2-1): ``3 member(s)``
    was printed by a device that held a four-member table's worth of evidence to the
    contrary and had asked nobody. A one-member network has nobody to ask and gets no
    marker; every other row says how many peers answered, or that none did. ONE OWNER,
    because the CLI and the agent's own tool render this line from the same JSON and a
    second copy would be free to disagree about what "verified" means.
    """
    if int(row.get("members") or 0) <= 1:
        return ""
    table = (row.get("membership") or {}).get("table") or {}
    answered = len(table.get("answered") or [])
    pending = len(table.get("not_answered") or [])
    if table.get("complete"):
        return f"  [members verified with all {answered} peer(s)]"
    if answered:
        return f"  [members verified with {answered} of {answered + pending} peer(s)]"
    states = ", ".join(
        f"{item.get('device_id')} ({item.get('reason')})"
        for item in (table.get("not_answered") or [])
    )
    return f"  [members NOT verified: no peer answered{': ' + states if states else ''}]"


def membership_lines(row: dict[str, Any]) -> list[str]:
    """The lines a surface prints when THIS device's own standing is not `active`.

    ``trust: active`` beside a network that cannot be reached was the shape an
    operator could not act on (QA round 3, Q-R3-2), so the state is stated, with the
    remedies the code actually supports — :func:`membership_state` owns that
    reasoning, and this renders it for every surface rather than one.
    """
    membership = row.get("membership") or {}
    if membership.get("state", "active") == "active":
        return []
    lines = [f"  this device: {membership.get('sentence')}"]
    lines.extend(f"    - {remedy}" for remedy in membership.get("remedies") or [])
    return lines


def announce_identity_rotation(
    server: RelayServer,
    old: DeviceIdentity,
    new: DeviceIdentity,
    *,
    root: Path | None = None,
) -> dict[str, int]:
    """Announce a device-key rotation to every network this device is in.

    One signed statement per network, because the statement names the network it
    rotates within (a statement usable anywhere would be a skeleton key for the
    device's other memberships). Live peers get it now; offline ones get it queued,
    which is the documented cost of rotating while nobody is reachable — if it is
    never delivered and the old key is gone, the device is simply unknown to its
    peers and must re-pair, which is correct, because nothing can prove continuity
    without the old key.
    """
    sent = 0
    queued = 0
    for record in store.list_networks(root or server.root):
        if record.self_device_id == new.device_id:
            continue
        statement = rotation_statement(old, new, record.network_id)
        frame = {
            "op": "net_identity_rotate",
            "statement": statement,
            "locality": "remote",
        }
        for link in list(server.links.values()):
            if link.network_id == record.network_id and link.send(dict(frame)):
                sent += 1
        for member in record.active_members():
            if member.device_id == record.self_device_id:
                continue
            if server._link_for(member.device_id) is not None:
                continue
            store.enqueue_frame(
                member.device_id, dict(frame), removed=False, root=root or server.root
            )
            queued += 1
        # The row's own id changes with the key, so the record is rewritten here
        # rather than by the caller: leaving the old id in `self_device_id` would
        # make the next handshake verify against a key this device no longer holds.
        old_member = record.self_member()
        if old_member is not None and old_member.device_id == old.device_id:
            old_member.previous_ids = [*old_member.previous_ids, old.device_id]
            old_member.device_id = new.device_id
            old_member.public_key = new.public_key
            old_member.rotated_at = time.time()
            record.self_device_id = new.device_id
            store.save(record, root or server.root)
    return {"sent": sent, "queued": queued}


def set_trust(
    record: NetworkRecord,
    *,
    trust: str,
    reason: str = "",
    root: Path | None = None,
    persist: bool = True,
) -> None:
    """Move a network between ``active`` and ``untrusted`` (or ``disconnected``).

    ``untrusted`` refuses every link for that network — the handshake checks it as
    step 3 and dispatch checks it again — so recovery is explicit and local
    (``lop network trust <net> --active``).
    """
    # ``types.trust_state`` is the package's one reader for this string; going
    # through it keeps the refusal code and sentence single-sourced, and the
    # attribute is assigned its declared ``TrustState`` rather than a bare ``str``.
    record.trust = trust_state(trust)
    record.untrusted_reason = reason
    if trust == "active":
        record.untrusted_reason = ""
    if persist:
        store.save(record, root)


def panic_frame(
    record: NetworkRecord, state: SecretState, *, reason: str = "operator_panic"
) -> dict[str, Any]:
    """The incident broadcast. It carries the secret to EVERYONE, on purpose.

    The one place the "no secret to a removed peer" rule is deliberately absent: a
    panic declares the network compromised, every receiver goes untrusted, and the
    operator re-admits devices by hand afterwards — so there is no set of
    recipients for whom withholding the new secret would help.
    """
    frame = {
        "op": "net_panic",
        "epoch": record.epoch,
        "sequence": record.sequence,
        "rotation_id": record.rotations.get(str(record.epoch), record.self_device_id),
        "secret": state.secret,
        "reason": reason,
        "members": [member.to_json() for member in record.members],
        "members_digest": members_digest_of(record),
    }
    return frame


def panic(
    record: NetworkRecord,
    state: SecretState,
    *,
    by: str,
    is_admin: bool,
    reason: str = "operator_panic",
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Raise the alarm: rotate if this device may, and return the frame to fan out.

    A NON-ADMIN panic rotates nothing and carries no secret (§8.2 / §16 Q3): the
    safe half of the same signal is "every receiver goes untrusted", which forces
    an operator to re-admit — a false alarm costs a human action, a suppressed
    alarm costs the network.
    """
    moment = time.time() if now is None else now
    if is_admin:
        rotate_epoch(record, state, by=by, reason="panic", root=root, now=moment, persist=persist)
        return panic_frame(record, state, reason=reason)
    return {
        "op": "net_panic",
        "epoch": record.epoch,
        "sequence": record.sequence,
        "rotation_id": by,
        "reason": reason,
    }


def apply_panic(
    record: NetworkRecord,
    frame: dict[str, Any],
    *,
    sender_device_id: str,
    root: Path | None = None,
    persist: bool = True,
) -> ApplyOutcome:
    """Mark this network untrusted on receipt. The secret, if any, is NOT applied.

    A panic does not rotate anything on the receiving side either: the network is
    untrusted, which refuses every link, so a new secret would be a secret for a
    network nobody may talk on.
    """
    set_trust(
        record,
        trust="untrusted",
        reason=f"panic received from {sender_device_id}",
        root=root,
        persist=persist,
    )
    return ApplyOutcome(True, "untrusted")


def apply_device_rotation(
    record: NetworkRecord,
    statement: dict[str, Any],
    *,
    root: Path | None = None,
    now: float | None = None,
    persist: bool = True,
) -> MemberRecord:
    """Rewrite a member's row in place after a key rotation (§3.3).

    The statement must be signed by the OLD key and the old id must be an ACTIVE
    member, so a rotation cannot be used to re-identify as anybody. The old id is
    kept in ``previous_ids`` for a bounded window so an in-flight link at the old
    id is not cut mid-turn.
    """
    old_id = str(statement.get("old_device_id") or "")
    member = record.member(old_id)
    if member is None or not member.active:
        raise MeshRefusal(
            "unknown_member",
            f"a device rotation names {old_id}, which is not an active member of {record.name}",
        )
    verify_rotation_statement(statement, member.public_key)
    new_id = str(statement["new_device_id"])
    other = record.member(new_id)
    if other is not None and other.device_id != member.device_id:
        raise MeshRefusal("device_id_conflict", f"{new_id} is already a member of this network")
    member.previous_ids = [*member.previous_ids, member.device_id]
    member.device_id = new_id
    member.public_key = str(statement["new_public_key"])
    member.rotated_at = time.time() if now is None else now
    if persist:
        store.save(record, root)
    return member


# ---------------------------------------------------------------------------
# The relay's view of the store, as the authoriser's ``NetworkState``
# ---------------------------------------------------------------------------


class StoreView(NetworkState):
    """The authoriser's world: the store, read per question rather than cached.

    Reading per question is what makes a relay restart stateless AND what makes the
    CLI's direct writes safe: a ``lop network init`` that lands while the relay is
    running is visible to the very next authorisation decision, with no cache to
    invalidate and no lock to take.
    """

    def __init__(self, root: Path | None = None, *, sessions: Callable[[], set[str]] | None = None):
        self._root = root
        self._sessions = sessions or (lambda: set())

    def network(self, network_id: str) -> NetworkRecord | None:
        try:
            return store.load(network_id, self._root)
        except FileNotFoundError:
            return None

    def local_session_ids(self) -> set[str]:
        return self._sessions()


#: How long a mesh-requested engage may take before the op answers with a
#: sentence. Longer than a local caller's own budget because this one spans a
#: spawn plus a construction on a machine that may be busy with its own work,
#: and shorter than any front end's patience with a "starting" row.
ENGAGE_DEADLINE_S = 60.0

#: Margin over the ladder's own rungs when this side budgets a FORCED stop.
#:
#: The two waits below are the bound; this covers what sits around them on the
#: OWNER's side — the silent-socket probe, the identity proofs (one of which
#: forks ``ps`` and ``lsof``), the stop marker and the record recovery.
_FORCED_STOP_MARGIN_S = 20.0


def forced_stop_deadline_s() -> float:
    """How long a requester must wait for a PEER'S own forced stop to resolve.

    A PEER STOP'S ANSWER IS THE OWNER'S RECEIPT, OR IT IS NOTHING (QA round 7,
    Q-R7-2). The owner runs its OWN kill-switch ladder and answers with the rung
    that acted; that ladder's one long silence is the SIGTERM rung's grace, which
    is derived from the receiver's drain bound and is MINUTES by construction.
    Budgeting the hop below it does not shorten the owner's work — it only
    throws the receipt away: measured on a busy, socket-silent target, the marker
    the relay itself wrote named rung ``sigterm`` while the caller was told
    ``rc 1 peer_unreachable`` with no outcome, no rung and no pid, because the hop
    waited 60 s (``max(op_wait_s, ENGAGE_DEADLINE_S)``) and the ladder needed
    ~150 s.

    Only the FORCED path can reach that wait: a plain stop refuses or skips a
    target whose socket is silent, and the busy skip sits before the identity
    gate (see ``control.stop_session``), so every other mode of this verb keeps
    the default hop budget.

    DERIVED from the ladder's own constants rather than re-typed: a second number
    here would go stale the moment either rung moves, and both move for reasons
    (``SIGNAL_DRAIN_S``) that have nothing to do with this hop.
    """
    from local_operator.session.runtime import control

    return control.SIGTERM_GRACE_S + control.SIGKILL_CONFIRM_S + _FORCED_STOP_MARGIN_S


#: ``control.StopOutcome.method`` (socket | sigterm | sigkill | gone | refused |
#: busy | draining) → the coarse word a viewer's receipt uses. The RUNG is reported
#: verbatim beside it; this map only answers "did it end?", and it lives here rather
#: than in ``control.py`` because a second caller of that ladder should not force
#: its vocabulary on the one that already had one.
#:
#: EVERY METHOD THE LADDER CAN RETURN HAS A WORD, and an UNKNOWN one does not fall
#: back to a word that reads as success: the old ``.get(method, "stopped")`` turned
#: ``draining`` — the target is alive and already leaving, which
#: ``control.LEFT_ALONE_METHODS`` groups with ``busy`` — into ``outcome: "stopped"``,
#: i.e. a receipt claiming this device ended a runtime it never signalled. An
#: unknown method now reads ``unknown``, which the viewer's own derivation treats
#: as "did not end" rather than as "ended" (Q-R4-1's class).
_STOP_OUTCOME_WORD: dict[str, str] = {
    "socket": "stopped",
    "sigterm": "stopped",
    "sigkill": "killed",
    "gone": "already-gone",
    "refused": "refused",
    "busy": "skipped",
    "draining": "skipped",
}

#: What this device says when the ladder returned a method this map does not know.
#: NOT a success word, and not silence either: the viewer's derivation is a positive
#: set of ENDED words, so this lands as a non-zero receipt naming the gap.
_STOP_OUTCOME_DEFAULT = "unknown"


@dataclass
class _Stream:
    """One forwarded viewer connection, from either end of the mesh (§3.2).

    The SAME object serves both halves of the pipe, because they are one fact
    seen from two relays: ``viewer_sock`` is set on the device the person is at,
    ``dial`` on the device that owns the session. Exactly one of the two is set
    at a time, and leaving both optional says so instead of inventing two classes
    that would drift apart.

    ``pending`` exists for ONE ordering hazard. The opening relay writes the
    stream's ack to its own viewer BEFORE any owner frame reaches the viewer, but
    the peer relay can push the welcome before that ack has been written — the
    two travel on different sockets. Frames arriving in that window are held here
    and flushed when the viewer socket is marked ready, which is deterministic
    where a sleep would merely usually work.
    """

    stream_id: str
    session_id: str
    peer_device_id: str
    link: "PeerLink | None" = None
    viewer_sock: "socket.socket | None" = None
    viewer: "session_dial.LineReader | None" = None
    dial: "session_dial.OwnerDial | None" = None
    ready: bool = False
    closed: bool = False
    pending: list[dict[str, Any]] = field(default_factory=list)
    #: One lock per stream: the link's reader thread pushes from the peer while
    #: this device's control connection thread writes the viewer's own frames,
    #: and two writers on one socket interleave into unparseable JSON.
    lock: threading.Lock = field(default_factory=threading.Lock)
    pump: "threading.Thread | None" = None

    def write_to_viewer(self, frame: dict[str, Any]) -> bool:
        """Write one session frame to the viewer socket, or buffer it."""
        with self.lock:
            if self.closed or self.viewer_sock is None or not self.ready:
                self.pending.append(frame)
                return True
            try:
                self.viewer_sock.sendall(
                    (json.dumps(frame, default=str, ensure_ascii=False) + "\n").encode("utf-8")
                )
            except OSError:
                self.closed = True
                return False
            return True

    def flush(self) -> None:
        """Deliver everything buffered before the viewer was ready."""
        with self.lock:
            queued = self.pending
            self.pending = []
            self.ready = True
        for frame in queued:
            if not self.write_to_viewer(frame):
                return


def local_session_ids(root: Path | None = None) -> set[str]:
    """Session ids that live on THIS device, from the session run directory.

    A read-through: the relay opens no session file and holds no lease. It only
    needs the ids, so it asks the registry (the same discovery every viewer uses)
    rather than reading a transcript.

    OWNERSHIP IS NOT LIVENESS, and this function answers the first question —
    it is the whole of the authoriser's session-scope rule (§7.2, INV-1). A
    session that is merely COLD still belongs to this device: its directory is
    here, no peer owns it, and refusing a session-scoped op against it would
    refuse exactly the ops that exist to bring it back (`net_session_engage`
    warms a cold session; `net_session_lifecycle` deletes a stopped one). So the
    answer is the union of the live records and the session directories this
    device owns — a directory whose ``mesh.json`` names another device is NOT
    ours, because that is the crash window a handoff can leave behind (§6.5) and
    answering "mine" for it is how a moved-away session is resurrected here.
    """
    from local_operator.session.placement import read_stamp
    from local_operator.session.runtime import registry

    owned = {record.session_id for record, _state in registry.scan(root)}
    sessions_dir = (Path(root) if root is not None else config_dir()) / "sessions"
    self_id = _self_device_id(root)
    try:
        children = list(sessions_dir.iterdir())
    except OSError:
        return owned
    for child in children:
        if not child.is_dir():
            continue
        stamp = read_stamp(Path(root) if root is not None else config_dir(), child.name)
        home = stamp.home_device if stamp is not None else ""
        if home and self_id and home != self_id:
            continue
        owned.add(child.name)
    return owned


def _self_device_id(root: Path | None = None) -> str:
    """This device's mesh id, or ``""`` when it has no identity file.

    ``identity.load`` and never ``load_or_mint``: a relay read that minted a
    device key would give a device that never joined a network an identity as a
    side effect of somebody listing a session.
    """
    try:
        from local_operator.network import identity

        loaded = identity.load(root)
    except Exception:  # noqa: BLE001 — an unreadable identity is "no identity"
        return ""
    return str(getattr(loaded, "device_id", "") or "")


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


class LinkKind:
    """The two queue classes. Dropping the wrong one loses work; blocking on the
    wrong one stalls a chat."""

    #: ``projection`` and other session-state pushes: at most ONE pending per
    #: (link, stream), a newer push replacing the older. The existing event
    #: vocabulary is already a full repaint with no deltas, which is exactly a
    #: coalescible frame.
    DROPPABLE = "droppable"
    #: Acks, catalogues, membership and rotations: the producer waits and then
    #: FAILS the op with a sentence rather than dropping it.
    RELIABLE = "reliable"


class PeerLink:
    """One authenticated link: a socket, a codec, a reader and a writer."""

    def __init__(
        self,
        *,
        server: RelayServer,
        sock: socket.socket,
        result: Any,
        codec: wire.LinkCrypto,
        settings: NetworkSettings,
        reader: wire.FrameReader | None = None,
    ) -> None:
        """``reader`` is the HANDSHAKE's reader, when it still holds bytes.

        A link takes over the socket the handshake just finished on, and the
        handshake reader may already own bytes of the first record (see
        ``wire.FrameReader.__init__``). Passing it here carries that buffer into
        the record phase; leaving it out starts a fresh reader and silently loses
        whatever the peer sent in the gap — the shape of a link that dies with a
        crypto error moments after a good handshake.
        """
        self.server = server
        self.sock = sock
        self.result = result
        self.codec = codec
        self.settings = settings
        #: The handshake's reader, when it had one: it may already own bytes of the
        #: first record, and the read loop below prefers it over a fresh reader.
        self._handshake_reader = reader
        self.link_id = result.link_id
        self.device_id = result.peer_device_id
        self.instance_id = result.peer_instance_id
        self.network_id = result.network_id
        self.epoch = result.epoch
        self.phase: LinkPhase = result.phase
        self.capabilities = frozenset(result.peer_capabilities)
        self.peer_addr = ""
        self.opened_at = time.time()
        self.last_frame_at = time.time()
        self.frames_in = 0
        #: When this link last ANSWERED a member-table pull (`net_member_list`).
        #: Zero means "never", and it is the whole of the due-gate in
        #: :meth:`RelayServer.refresh_membership`: membership is re-read on a
        #: SCHEDULE per link rather than at establishment only, because a healthy
        #: pair keeps its link open indefinitely and establishment therefore never
        #: comes round again (QA round 3, Q-R2-1).
        self.member_pulled_at = 0.0
        #: The device ids the last table pull ADDED, so a refresh pass can report
        #: what it learned rather than only that it asked.
        self.member_pull_added: list[str] = []
        #: Replies that matched no waiter — a late answer to a timed-out request, or a
        #: peer answering something nobody asked. Counted, never logged per frame (A7);
        #: see the drop path in ``_handle`` for why it must not be dispatched.
        self.stray_replies = 0
        self.frames_out = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self._closed = threading.Event()
        self._reliable: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=settings.queue_frames)
        self._droppable: dict[str, dict[str, Any]] = {}
        self._droppable_lock = threading.Lock()
        self._wake = threading.Event()
        # THE CLOSE HANDSHAKE (see :meth:`close`): ``_flushing`` asks the writer to
        # finish what is queued, and ``_flushed`` is the writer's answer that both
        # queues are empty and on the wire.
        self._flushing = threading.Event()
        self._flushed = threading.Event()
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._context = LinkContext(
            link_id=self.link_id,
            device_id=self.device_id,
            instance_id=self.instance_id,
            network_id=self.network_id,
            epoch=self.epoch,
            capabilities=frozenset(self.role_capabilities()),
            phase=self.phase,
            peer_addr="",
        )

    def role_capabilities(self) -> set[str]:
        """The capabilities resolved from the MEMBER ROW at the current epoch.

        Read per use rather than captured at the handshake, because a rotation that
        narrows a member's authority must take effect on an already-open link the
        moment the record changes — the same reason revocation is evaluated against
        the current member list rather than a cached one.
        """
        record = self.server.store_view.network(self.network_id)
        if record is None:
            return set()
        member = record.member(self.device_id)
        return set(member.capabilities) if member and member.active else set()

    @property
    def context(self) -> LinkContext:
        return LinkContext(
            link_id=self.link_id,
            device_id=self.device_id,
            instance_id=self.instance_id,
            network_id=self.network_id,
            epoch=self.epoch,
            capabilities=frozenset(self.role_capabilities()),
            phase=self.phase,
            peer_addr=self.peer_addr,
        )

    def start(self) -> None:
        self._reader = threading.Thread(
            target=self._read_loop, name=f"mesh-read-{self.link_id[:8]}", daemon=True
        )
        self._writer = threading.Thread(
            target=self._write_loop, name=f"mesh-write-{self.link_id[:8]}", daemon=True
        )
        self._reader.start()
        self._writer.start()

    @property
    def alive(self) -> bool:
        return not self._closed.is_set()

    def send(
        self, frame: dict[str, Any], *, kind: str = LinkKind.RELIABLE, stream: str = ""
    ) -> bool:
        """Queue a frame. Returns False when it could not be queued at all."""
        if self._closed.is_set():
            return False
        if kind == LinkKind.DROPPABLE:
            with self._droppable_lock:
                self._droppable[stream or "link"] = frame
            self._wake.set()
            return True
        deadline = time.monotonic() + self.settings.op_wait_s
        while True:
            try:
                self._reliable.put(frame, timeout=0.1)
                self._wake.set()
                return True
            except queue.Full:
                if time.monotonic() > deadline or self._closed.is_set():
                    # The producer FAILS the op rather than dropping it: a dropped
                    # ack or rotation is a state divergence, not a stale repaint.
                    return False

    def close(self, reason: str = "we-closed", *, flush_s: float = CLOSE_FLUSH_S) -> None:
        """Close the link — AFTER handing the writer what is already queued.

        ``send`` is asynchronous on purpose (a producer must not block on a peer's
        socket), and the write loop drains the queue on its own schedule. Closing the
        socket therefore used to DISCARD whatever was still queued, and the two frames
        that matter most are exactly the ones sent immediately before a close: the
        ``net_epoch`` rotation that ``_rehandshake_network`` pushes down every link to
        change keys, and the ``net_bye`` queued a line above the close it announces.
        Measured: a rotation to epoch 2 never reached the peer over a live link with
        no delay inserted, and DID reach it (and was then refused by the epoch gate —
        the separate defect in ``authorizer._check_epoch``) once the close was delayed
        by a second. Losing a rotation is losing the only signal that revokes a
        device's authority, one level below the bug that was being chased.

        So the close asks the writer to finish first, bounded: the handshake is an
        event pair the writer answers when both queues are empty (the frames are
        written synchronously by then, so "empty" means "on the wire"), and a wedged
        or dead writer cannot hold the close past ``flush_s``. A link whose peer is
        gone blocks in ``sendall`` for at most that same bound.
        """
        if self._closed.is_set():
            return
        self._flushing.set()
        self._wake.set()
        self._flushed.wait(flush_s)
        self._closed.set()
        self._flushing.clear()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.server.link_closed(self, reason)

    # -- threads ------------------------------------------------------------

    def _read_loop(self) -> None:
        # The reader the handshake left behind, when it left one: its buffer may
        # already hold the first record, and a fresh reader here would lose it.
        reader = (
            self._handshake_reader
            if self._handshake_reader is not None
            else wire.FrameReader(self.sock)
        )
        try:
            while not self._closed.is_set():
                deadline = time.monotonic() + self.settings.link_idle_s
                payload = reader.read_record_payload(deadline)
                self.bytes_in += len(payload)
                frame = self.codec.open(payload)
                self.frames_in += 1
                self.last_frame_at = time.time()
                self.server.identity_use.note_frame(self.device_id)
                self._handle(frame)
        except TimeoutError:
            self.server.audit_idle(self)
            self.close("timeout")
        except (wire.LinkCryptoError, ConnectionError, OSError, TimeoutError):
            # Every failure here is fatal to the LINK and none is repaired: there is
            # no resynchronisation and no partial-trust phase.
            self.close("error")
        except Exception:  # noqa: BLE001 — a handler must not kill the accept loop
            self.close("error")

    def _handle(self, frame: dict[str, Any]) -> None:
        if wire.is_keepalive(frame):
            self.send({"op": "ack", "req": frame.get("req"), "detail": "pong"})
            return
        if wire.is_bye(frame):
            self.server.audit_link(self, "link_closed", cause="peer-closed")
            self.close("peer-closed")
            return
        # A reply to a request THIS side made is delivered to the waiter that is
        # blocked on its ``req`` rather than dispatched: a link also receives
        # unsolicited events, and an ack that satisfied the wrong caller would be a
        # cross-wired answer that looks like success.
        if frame.get("op") in ("ack", "error"):
            if self.server.deliver_reply(self, frame):
                return
            # A reply nobody is waiting for is DROPPED, never dispatched. Dispatching
            # it is a livelock, not merely wasteful: an unknown op is refused with an
            # ``error`` frame, the peer refuses that in turn, and the two devices
            # answer each other's refusals until one link's queue fills — measured
            # here as ~360 frames in three seconds for one idle pair, and caught by
            # the e2e test rather than by review. The counter is kept rather than a
            # log line: a stray reply is usually a late answer to a timed-out
            # request, and a per-frame log line is exactly what A7 forbids.
            self.stray_replies += 1
            return
        if frame.get("op") == "net_stream":
            # STREAM PUSHES AND CLOSES ARE NOT REQUESTS. Checked BEFORE dispatch,
            # for the reason the stray-reply branch below gives: a push has no
            # reply, and answering one would start exactly the refuse-each-other
            # livelock that comment measures. Authorisation is this relay's own
            # stream table (only a peer that opened that stream can push on it),
            # which is a stronger check than a capability string and is why no
            # capability row is consulted here.
            action = str(frame.get("action") or "")
            if action == "push" and self.server.route_stream_push(self, frame):
                return
            if action == "closed" and self.server.route_stream_closed(self, frame):
                return
        reply = self.server.dispatch(self, frame)
        if reply is not None:
            self.send(reply)

    def _write_loop(self) -> None:
        try:
            while not self._closed.is_set():
                sent = self._drain()
                if self._flushing.is_set() and self._reliable.empty() and not self._droppable:
                    # BOTH QUEUES EMPTY AND DRAINED BY THIS THREAD: everything a
                    # closing caller queued has been written, which is what
                    # :meth:`close` is waiting for. Checked HERE rather than in the
                    # producer because "written" is only knowable on this side.
                    self._flushed.set()
                if sent:
                    continue
                if self._flushing.is_set():
                    # Do not emit a keepalive into a link that is closing: it would be
                    # a frame the peer reads after the peer's own close attempt.
                    self._wake.wait(timeout=0.02)
                    self._wake.clear()
                    continue
                # Keepalive on an idle link; the reader enforces the other half
                # (``link_idle_s`` without a frame closes it).
                if time.time() - self.last_frame_at > self.settings.keepalive_s:
                    record = self.server.peer_record()
                    self._write(wire.keepalive_frame(record.pid if record else 0))
                self._wake.wait(timeout=0.5)
                self._wake.clear()
        except (OSError, wire.LinkCryptoError):
            self.close("error", flush_s=0.0)

    def _drain(self) -> bool:
        wrote = False
        while True:
            try:
                frame = self._reliable.get_nowait()
            except queue.Empty:
                break
            self._write(frame)
            wrote = True
        with self._droppable_lock:
            pending = list(self._droppable.values())
            self._droppable.clear()
        for frame in pending:
            self._write(frame)
            wrote = True
        return wrote

    def _write(self, frame: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        payload = self.codec.seal(frame)
        self.sock.sendall(payload)
        self.bytes_out += len(payload)
        self.frames_out += 1

    def request(
        self, frame: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Send one RELIABLE op and wait for its reply.

        The reply is matched on ``req`` rather than on arrival order, because a
        link also receives unsolicited events; a caller waiting for its own ack
        must not be handed someone else's ping answer.
        """
        req = frame.get("req")
        waiter = self.server.expect_reply(self.link_id, req) if req is not None else None
        if not self.send(frame):
            return None
        if waiter is None:
            return None
        return waiter.wait(self.settings.op_wait_s if timeout is None else timeout)


# ---------------------------------------------------------------------------
# The listener's policy: the relay's answers to the handshake's questions
# ---------------------------------------------------------------------------


class ServerPolicy(ListenerPolicy):
    """``ListenerPolicy`` backed by the store — the handshake's only window onto it.

    Every method here is a security decision, so each one is as narrow as the
    question it answers: an ACTIVE member's key or ``None`` (never a tombstone's),
    the current epoch key or ``None``, and an invite's key only while that invite is
    minted, fresh and named to this device as its inviter.
    """

    def __init__(self, server: RelayServer, network_id: str) -> None:
        self._server = server
        self._network_id = network_id
        self._record: NetworkRecord | None = None
        try:
            self._record = store.load(network_id, server.root)
            self._state = store.require_secrets(network_id, server.root)
        except (FileNotFoundError, MeshRefusal):
            self._record = None
            self._state = SecretState(network_id=network_id, epoch=0)

    @property
    def current_epoch(self) -> int:
        return self._record.epoch if self._record else 0

    @property
    def previous_epoch(self) -> int | None:
        return self._state.previous_epoch

    def network_known(self, network_id: str) -> bool:
        return self._record is not None and self._record.network_id == network_id

    def trust_active(self, network_id: str) -> bool:
        return bool(self._record and self._record.trust == "active")

    def epoch_key(self, network_id: str, epoch: int) -> bytes | None:
        if not self._record:
            return None
        if epoch == self._state.epoch and self._state.secret:
            return wire.epoch_key(self._state.secret, network_id, epoch)
        if epoch == self._state.previous_epoch and self._state.previous_secret:
            return wire.epoch_key(self._state.previous_secret, network_id, epoch)
        return None

    def active_member_public_key(self, network_id: str, device_id: str) -> str | None:
        """The member row's key — ONLY for a live member at the CURRENT epoch.

        This one method carries R5's teeth: a tombstoned device, or one absent from
        the current list, resolves to ``None`` and is refused at step 6 before any
        key is tried.
        """
        if not self._record:
            return None
        member = self._record.member(device_id)
        if member is None or not member.active:
            return None
        return member.public_key or None

    def invite_credential(self, network_id: str, invite_id: str) -> Credential | None:
        """The invite's MAC key, or ``None`` when the token may not be redeemed.

        Freshness is judged against the MINTING device's clock, which is this one —
        the token carries a duration, never an absolute expiry, so pairing involves
        no cross-host clock comparison at all.
        """
        if not self._record or not invite_id:
            return None
        invite = self._record.invite(invite_id)
        if invite is None or invite.state != "minted" or not invite.is_fresh(time.time()):
            return None
        if invite.role not in ("read", "drive", "admin"):
            return None
        if not self._state.secret:
            return None
        return Credential(
            "invite",
            self._record.epoch,
            wire.invite_key(self._state.secret, network_id, invite.invite_id),
        )

    def invite_device_binding(self, network_id: str, invite_id: str) -> str:
        if not self._record:
            return ""
        invite = self._record.invite(invite_id)
        return invite.device_id if invite else ""


# ---------------------------------------------------------------------------
# The relay
# ---------------------------------------------------------------------------


class RelayServer:
    """The listener, the links, the dispatch, and the loopback control surface."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        settings: NetworkSettings | None = None,
        identity: DeviceIdentity | None = None,
        instance_id: str | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        # ROOT IS ALWAYS A REAL PATH. ``store`` and ``audit`` treat ``None`` as "the
        # ambient config dir" and resolve it per call, but code that BUILDS a path
        # from it does not: ``read_stamp(self.root, ...)`` and the create path's
        # ``self.root / "sessions" / session_id`` raise ``TypeError`` on None.
        # ``lop network serve`` constructs this with no root, so on the real path
        # every federated listing and every session a peer asked for failed with a
        # TypeError — while the in-process tests all pass an explicit root, which is
        # why CI never saw it (QA round 1: both failures sat behind F-2's link).
        self.root = Path(root) if root is not None else config_dir()
        self.settings = settings or NetworkSettings.from_config(self.root)
        self.identity = identity or load_or_mint(self.root)
        self.instance_id = instance_id or mint_instance_id()
        # `from_config`, not the default constructor: the cap, the generation count
        # and the age bound are registered settings (settings_io.SETTINGS, section
        # `network`), and an explicit `audit=` argument still wins for a test.
        self.audit = audit or AuditLog.from_config(self.root)
        self.store_view = StoreView(root, sessions=lambda: local_session_ids(root))
        #: Forwarded viewer streams, keyed by an unpredictable id (os.urandom).
        #: On the device the viewer sits at these are streams it OPENED; on the
        #: owning device they are streams it ACCEPTED. One table for both roles,
        #: because a stream id is unique per opening relay and the roles cannot
        #: collide. See _Stream for why the unguessable id is the authorisation
        #: for a PUSH (a push is not a request and is deliberately not
        #: dispatched — see PeerLink._handle).
        self._streams: dict[str, _Stream] = {}
        self._streams_lock = threading.Lock()
        #: req numbers for relay-initiated requests on a link (stream opens).
        #: A separate counter from any other user's, so a reply can never be
        #: matched by the wrong waiter.
        self._relay_req = 0
        self.authorizer = Authorizer(self.store_view, self.audit)
        self.identity_use = IdentityUseTracker()
        self.links: dict[str, PeerLink] = {}
        self._links_lock = threading.RLock()
        self._listener: socket.socket | None = None
        self._control: socket.socket | None = None
        self._control_key = wire.b64u(os.urandom(32))
        self._control_port = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._reply_waiters: dict[tuple[str, Any], "_ReplyWaiter"] = {}
        #: Serialises invite claim + mark, so two concurrent redemptions of one
        #: token cannot both pass the `minted` check.
        self._invite_lock = threading.Lock()
        #: The pre-auth bound (see DEFAULT_MAX_HANDSHAKES): acquired BEFORE a
        #: handshake thread exists and released by that thread on every exit path,
        #: so the cap counts connections that have not authenticated. Bounded, not
        #: plain: a release without an acquire is a bug this turns into a loud
        #: ValueError rather than a cap that silently grows.
        self._handshake_slots = threading.BoundedSemaphore(max(1, self.settings.max_handshakes))
        #: When the pre-auth bound last reported itself (``_note_handshake_cap``).
        #: Written and read by the accept loop alone, which is one thread, so it needs
        #: no lock here. The audit call it guards does lock — ``AuditLog`` is flushed
        #: from every thread the relay runs, ``_write_lock`` is what keeps one record
        #: from being published twice (see that module's docstring; assuming it was
        #: "the relay's own and already thread-safe" is exactly how the duplicate hid).
        self._cap_notice_at = 0.0
        self._reconcile_grants: dict[tuple[str, str], list[float]] = {}
        self._handlers: dict[str, Callable[[PeerLink, dict[str, Any]], dict[str, Any] | None]] = {
            "ping": self._op_ping,
            "net_bye": self._op_bye,
            "net_catalog": self._op_catalog,
            "net_member_list": self._op_member_list,
            "net_epoch": self._op_epoch,
            "net_reconcile": self._op_reconcile,
            "net_leave": self._op_leave,
            "net_panic": self._op_panic,
            "net_trust": self._op_trust,
            "net_identity_rotate": self._op_identity_rotate,
            # THE SESSION PLANE (mesh-session-mobility.md §2.2/§3.2/§4.3). The
            # carriers and the three ops that make a session on this device
            # reachable from another one. net_sync/net_broker/net_session_move
            # are deliberately ABSENT: they belong to other slices, and their
            # absence is answered by the same sentence-naming fallback dispatch
            # has always used (_owning_document).
            "net_forward": self._op_forward,
            "net_stream": self._op_stream,
            "net_session_create": self._op_session_create,
            "net_session_engage": self._op_session_engage,
            "net_session_stop": self._op_session_stop,
            "net_session_lifecycle": self._op_session_lifecycle,
            "net_pair_ready": self._op_pair_ready,
            "net_pair_abort": self._op_pair_abort,
        }
        self.started_at = time.time()
        #: Computed ONCE: the build stamp is decoration, and asking packaging
        #: metadata again on every heartbeat would be a per-15-seconds import for a
        #: string that cannot change while this process lives.
        self.build = _build_stamp()

    # -- lifecycle ----------------------------------------------------------

    def bind(self) -> tuple[str, int]:
        """Bind the peer listener. Returns the address it actually bound."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.settings.listen_address, self.settings.port))
        listener.listen(16)
        self._listener = listener
        host, port = listener.getsockname()[:2]
        # RECORD WHAT WE ACTUALLY BOUND. Every advertised endpoint is built from
        # `settings.port`, and with `port = 0` the kernel chooses — so leaving the
        # configured 0 in place would publish `127.0.0.1:0` into our own member row,
        # our peer record, and every peer that reads either: an address nothing can
        # dial, which is the same class of lie as the observed-address row F-2 was
        # about. `replace` rather than assignment so a settings object shared with
        # a second relay is not rewritten underneath it.
        self.settings = replace(self.settings, port=int(port))
        return str(host), int(port)

    def bind_control(self) -> tuple[str, int]:
        control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control.bind(("127.0.0.1", 0))
        control.listen(8)
        self._control = control
        host, port = control.getsockname()[:2]
        self._control_port = int(port)
        return str(host), int(port)

    def start(self, *, background: bool = False) -> None:
        if self._listener is None:
            self.bind()
        if self._control is None:
            self.bind_control()
        assert self._listener is not None and self._control is not None
        # BEFORE the accept loop starts: a peer that handshakes in the first
        # millisecond must find this device's endpoints already on the self row.
        self.sync_self_endpoints()
        for target, name in (
            (self._accept_loop, "mesh-accept"),
            (self._control_loop, "mesh-control"),
            (self._heartbeat_loop, "mesh-heartbeat"),
            # MEMBERSHIP CONVERGES ON ITS OWN CLOCK, not on a link being established
            # and not on someone running a listing: a member nobody lists is still a
            # member, and a healthy pair holds its link open forever, so there is no
            # "next contact" to hang the refresh on (Q-R2-1, QA round 3).
            (self._membership_loop, "mesh-membership"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.publish()
        self._flush_outboxes()

    def serve_forever(self) -> None:
        """The foreground runner (``lop network serve``): block until signalled."""
        self.start()
        stop = threading.Event()

        def _handle_signal(signum: int, _frame: Any) -> None:
            self.audit.record(
                AuditEvent(event="disconnect_initiated", actor="self", actor_kind="relay")
            )
            stop.set()

        previous = signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        try:
            while not stop.wait(0.5):
                pass
        finally:
            signal.signal(signal.SIGTERM, previous)
            self.stop()

    def stop(self) -> None:
        """Stop accepting, say goodbye on every link, unpublish, flush the audit log.

        Sessions are UNTOUCHED: a reconnecting peer sees them again on the next
        handshake, and nothing here owns a transcript to lose.
        """
        self._stop.set()
        with self._links_lock:
            links = list(self.links.values())
        for link in links:
            link.send({"op": "net_bye", "reason": "stopping"})
        time.sleep(0.05)
        for link in links:
            link.close("we-closed")
        for sock in (self._listener, self._control):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        store.unpublish_peer_record(os.getpid(), self.root)
        self.audit.flush()

    # -- discovery record ---------------------------------------------------

    def declared_endpoints(self) -> list[str]:
        """The endpoints this device's own records declare for peers to dial.

        ``lop network init --advertise-host`` writes them into
        ``record.listen.advertised``, and that value is the operator's sanctioned
        way to name a tunnel hostname or a public address. Before this was read
        here, the flag was written to the record and never published: a peer-b
        that was told to advertise ``52.27.70.210:4200`` still advertised its
        private ``172.31.36.64``, because only the config key reached
        ``advertised_endpoints`` (QA round 1, F-2).
        """
        hosts: list[str] = []
        for record in store.list_networks(self.root):
            for host in record.listen.get("advertised") or ():
                text = str(host or "").strip()
                if text and text not in hosts:
                    hosts.append(text)
        return hosts

    def advertised_endpoints(self) -> list[str]:
        """Where peers should try to reach this device.

        The records' declared hosts first (``--advertise-host``), then
        ``network.advertise_hosts`` (the sanctioned way to name a tunnel
        hostname), then the detected local addresses. Loopback is included ONLY in
        dial-only mode, where it is the honest answer: it says "you cannot reach me
        from another machine", which is exactly what a peer needs to know.
        """
        return advertise_endpoints(self.settings, declared=tuple(self.declared_endpoints()))

    def sync_self_endpoints(self) -> None:
        """Write this device's advertised endpoints into its OWN member row.

        The self row is the one row every other device receives at admission and
        in a member list, so it is where a peer learns how to dial us back. At
        ``init`` the row was written with ``endpoints=[]`` and nothing ever filled
        it, which is why a joiner held a member row for the inviter with no
        address to dial (QA round 1, F-2). Only writes when the value CHANGES,
        because this runs on every heartbeat.
        """
        endpoints = self.advertised_endpoints()
        if not endpoints:
            return
        for record in store.list_networks(self.root):
            member = record.member(self.identity.device_id)
            if member is None or list(member.endpoints) == endpoints:
                continue
            member.endpoints = list(endpoints)
            store.save(record, self.root)

    def _note_peer_endpoints(
        self, record: NetworkRecord, device_id: str, endpoints: list[str]
    ) -> None:
        """Record where a peer says it can be reached, on its member row.

        THE ROW IS THE ONLY PLACE ``_ensure_link`` DIALS, so a declaration that
        never lands here is a peer that is permanently unreachable. A peer that
        declares NOTHING leaves the row alone: the only address this device would
        otherwise have is the observed source address of the connection, whose
        port is ephemeral and closed by the time anything dials it, and writing
        that would be a worse lie than an honest empty list.
        """
        if not endpoints or not device_id:
            return
        member = record.member(device_id)
        if member is None or not member.active:
            return
        if list(member.endpoints) == list(endpoints):
            return
        member.endpoints = list(endpoints)
        store.save(record, self.root)

    # -- membership at rest, and the one read that keeps it current ---------

    def _pull_members(self, link: "PeerLink") -> str:
        """Learn the member table from a peer, WITHOUT ever failing the caller.

        Returns the empty string when the peer ANSWERED (whether or not its table
        told us anything new) and a short reason when it did not. The distinction
        is the whole of the reporting contract downstream: "the peer answered with
        the same table" is proof the table is current, and reporting it as a failure
        would leave `ls` unable to say what its member count rests on.

        This runs at link establishment AND from the refresh pass, so an exception
        escaping it would be reported as "the link failed" while the link is, in
        fact, up and started. A stale table is the state this exists to improve on;
        it is not a reason to drop a good link.
        """
        try:
            return self._learn_members_from(link)
        except Exception:  # noqa: BLE001 — see the docstring: never fails its caller
            return "error"

    def _learn_members_from(self, link: "PeerLink") -> str:
        """The pull itself: ask, merge, persist, audit. See :meth:`_pull_members`.

        THE WIRE HAS EXACTLY ONE MEMBERSHIP READ (``net_member_list``, §6.4) and
        this is its client half. It exists because the record is a SNAPSHOT taken
        when the member joined: a joiner is handed the full list in its admission
        frame (``pair_result_frame``), and every device already in the network is
        told NOTHING — so a member admitted afterwards is invisible to them
        forever, across restarts, and ``--all-peers`` silently under-reports on
        the normal history of a mesh (QA round 2, Q-R2-1: `members: 2` on the
        device that had joined first against `3` on both others).

        Contact is what re-evaluates membership (§8.4), and a link IS contact —
        but contact does not only mean ESTABLISHMENT: a healthy pair keeps its link
        open indefinitely, so "on the next contact" never arrives and a table can
        stay stale for the life of the link (QA round 3). What makes this converge
        is the schedule in :meth:`refresh_membership`; this method is the read, and
        the read TRANSFERS the table transitively — ``adopt_members`` merges the
        peer's whole member list, so a device one hop from a newcomer learns it
        from a device two hops away with no link to the newcomer at all.

        It is best effort by construction — an older peer answers ``unknown op``, a
        peer that has gone quiet costs one op wait and nothing else — because a
        membership refresh that could fail a link would be a worse bug than the
        stale table it fixes.
        """
        if not link.alive:
            return "link_down"
        reply = link.request(
            {"op": "net_member_list", "req": self._next_relay_req(), "locality": "remote"},
            timeout=MEMBERSHIP_PULL_TIMEOUT_S,
        )
        if reply is None:
            return "no_answer"
        if reply.get("op") != "ack":
            return str(reply.get("code") or reply.get("op") or "refused")
        detail = reply.get("detail")
        if not isinstance(detail, dict) or not isinstance(detail.get("members"), list):
            return "bad_answer"
        rows = detail["members"]
        record = next(
            (item for item in store.list_networks(self.root) if item.network_id == link.network_id),
            None,
        )
        if record is None:
            return "unknown_network"
        # STAMPED BEFORE THE MERGE, and stamped on an answer that changed nothing:
        # the stamp is evidence that this peer's TABLE answered, not that its table
        # differed, and the reporting surfaces read it as exactly that.
        link.member_pulled_at = time.time()
        changed, added = adopt_members(record, rows)
        link.member_pull_added = list(added)
        if not changed:
            return ""
        store.save(record, self.root)
        self.audit.record(
            AuditEvent(
                event="membership_learned",
                actor=link.device_id,
                subject=record.network_id,
                outcome="ok",
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "source": link.device_id,
                    "added": added,
                    "members": len(record.active_members()),
                    "members_digest": members_digest_of(record),
                },
            )
        )
        return ""

    def refresh_membership(self, *, budget_s: float | None = None) -> dict[str, MembershipReport]:
        """Re-pull the member table from every LIVE link whose pull is DUE.

        THE BLOCKER THIS CLOSES (QA rounds 2 and 3, Q-R2-1). Membership convergence
        may not depend on which links happen to exist, on which link was established
        last, or on an operator touching a surface. So the trigger is a per-link
        SCHEDULE — every link answers a table pull again once
        :data:`MEMBERSHIP_PULL_MIN_INTERVAL_S` has passed — and it runs with no
        surface involved (:meth:`_membership_loop`), which is what lets a mesh
        converge with nothing restarted. `show`/`ls`/`peers` also force a refresh
        before they report (:meth:`contact_peers`), so a command answers with the
        table it can vouch for rather than the one it happens to hold.

        NO DIALS HERE. This walks the links that exist; the surfaces' own contact
        path is what dials. That split is deliberate: a link held by a dial-only
        device is the only route to it, and re-pulling over it needs no address at
        all — the device behind a NAT is exactly the one whose table nobody can
        fetch by dialling.

        Returns one :class:`MembershipReport` per network, which is what the
        reporting surfaces attach to a member count; that return value is the only
        reason a caller can say not just how many members it holds but how many
        peers agreed. Best effort and bounded: it never raises, and a peer that does
        not answer costs its op wait and nothing else.
        """
        deadline = None if budget_s is None else time.monotonic() + budget_s
        now = time.time()
        reports: dict[str, MembershipReport] = {}
        for record in store.list_networks(self.root):
            report = MembershipReport(network_id=record.network_id, refreshed_at=now)
            for member in record.active_members():
                if member.device_id == record.self_device_id:
                    continue
                link = self._link_for(member.device_id)
                if link is None or not link.alive:
                    report.silent.append({"device_id": member.device_id, "reason": "no_live_link"})
                    continue
                age = now - link.member_pulled_at
                if age < MEMBERSHIP_PULL_MIN_INTERVAL_S:
                    # AN ANSWER INSIDE THE CADENCE IS STILL EVIDENCE. Skipping the ask
                    # must not turn into "no peer answered": a `show` one second after
                    # a refresh would otherwise claim the table was never checked
                    # while holding an answer that is one second old. The age is
                    # reported, so nothing is implied to be fresher than it is.
                    report.answered.append(member.device_id)
                    report.answer_ages.append(max(0.0, age))
                    report.not_due.append(
                        {"device_id": member.device_id, "age_s": round(max(0.0, age), 3)}
                    )
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    report.silent.append(
                        {
                            "device_id": member.device_id,
                            "reason": (
                                "not_asked: the refresh budget ran out before " "this peer's turn"
                            ),
                        }
                    )
                    continue
                reason = self._pull_members(link)
                if reason:
                    report.silent.append(
                        {"device_id": member.device_id, "reason": f"no_table:{reason}"}
                    )
                    continue
                report.learned.extend(link.member_pull_added)
                age = time.time() - link.member_pulled_at
                report.answered.append(member.device_id)
                report.answer_ages.append(max(0.0, age))
            reports[record.network_id] = report
        return reports

    def contact_peers(
        self, *, budget_s: float | None = LISTING_PROBE_BUDGET_S
    ) -> dict[str, MembershipReport]:
        """Contact every peer once, within ``budget_s``, and refresh the table.

        Called before a surface REPORTS membership (``net_show``, ``net_ls``,
        ``network_detail``, ``federated_rows``). A member table is a distributed
        fact, and a report built from the local snapshot alone is how a device that
        joined earlier never learns about a newcomer (Q-R2-1).

        TWO STEPS PER MEMBER, and the second is the one round 2's fix was missing:
        dial the member (which is what makes an unreachable one show up as
        unreachable rather than absent), and then RE-PULL over the link that exists —
        including a link that was established long ago, because
        ``_ensure_link_with_reason`` returns an existing link untouched and the
        pull it used to rely on therefore never ran again.

        Best effort and bounded: an unreachable peer costs its probe budget and
        contributes nothing, exactly as it does in ``peer_status``. It CANNOT raise:
        it runs inside the admit path (where an exception would be reported as a
        pairing refusal for a pairing that succeeded) and inside a listing (where it
        would replace the answer with an error), so a member that fails is skipped
        rather than propagated — a refresh that can fail its caller is a worse bug
        than the stale table it exists to fix.
        """
        deadline = None if budget_s is None else time.monotonic() + budget_s
        for record in store.list_networks(self.root):
            for member in record.active_members():
                if member.device_id == record.self_device_id:
                    continue
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                try:
                    self._ensure_link_with_reason(member.device_id, probe_timeout_s=remaining)
                except Exception:  # noqa: BLE001 — see the docstring: never fails its caller
                    continue
        return self.refresh_membership(
            budget_s=None if deadline is None else max(0.0, deadline - time.monotonic())
        )

    def _membership_loop(self) -> None:
        """The scheduled half of membership convergence, with no surface involved.

        A CADENCE ON A LISTING IS NOT CONVERGENCE: a member that nobody lists is
        still a member, and the round-3 watch showed a listing-driven refresh that
        only ran at link establishment leave four devices disagreeing for minutes
        (Q-R2-1). This pass exists so the table converges on its own, the way the
        heartbeat already republishes the relay record for the same reason.

        The interval is :data:`MEMBERSHIP_PULL_PASS_S` and the per-link gate is
        :data:`MEMBERSHIP_PULL_MIN_INTERVAL_S`, so the traffic is bounded per link
        whatever a mesh's diameter is. Every failure is swallowed and retried on the
        next pass: one unreachable peer must never be able to stop a relay's clock.
        """
        while not self._stop.wait(MEMBERSHIP_PULL_PASS_S):
            try:
                self.refresh_membership()
            except Exception:  # noqa: BLE001 — a refresh must never kill the loop
                continue

    def peer_record(self) -> PeerRecord:
        networks = []
        for record in store.list_networks(self.root):
            networks.append(
                {
                    "network_id": record.network_id,
                    "name": record.name,
                    "epoch": record.epoch,
                    "role": record.self_role,
                    "trust": record.trust,
                    "members": len(record.active_members()),
                    "links": sum(
                        1
                        for link in self.links.values()
                        if link.network_id == record.network_id and link.alive
                    ),
                }
            )
        build = self.build
        return PeerRecord(
            pid=os.getpid(),
            session_protocol=int(PROTOCOL_VERSION),
            device_id=self.identity.device_id,
            device_name=self.identity.name,
            instance_id=self.instance_id,
            control_port=self._control_port,
            # 0600 record = the loopback boundary; the key's protection is the
            # account, exactly as a session record's is.
            control_key=self._control_key,
            listen={
                "address": self.settings.listen_address,
                "port": self.settings.port,
                "advertised": self.advertised_endpoints(),
            },
            networks=networks,
            links=len([link for link in self.links.values() if link.alive]),
            capabilities=list(wire.LINK_CAPABILITIES),
            version=build.get("version", ""),
            source_ref=build.get("source_ref", ""),
            started_at=self.started_at,
        )

    def publish(self) -> None:
        # The self member row is refreshed BEFORE the record is published: a peer
        # that reads our peer record and then handshakes must find the same
        # endpoints in both places.
        self.sync_self_endpoints()
        store.publish_peer_record(self.peer_record(), self.root)

    def _heartbeat_loop(self) -> None:
        """Re-publish the record and flush the audit tail, on the session
        runtime's own heartbeat interval so a reader needs ONE freshness rule."""
        while not self._stop.wait(HEARTBEAT_S):
            try:
                self.publish()
                self.audit.flush()
            except OSError:
                continue

    # -- accepting links ----------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                sock, addr = self._listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                continue
            if len(self.links) >= self.settings.max_links:
                sock.close()
                continue
            # THE PRE-AUTH BOUND, checked HERE because this is the last moment
            # before the thread exists. ``max_links`` above counts LINKS, and a
            # connection that has authenticated nothing has none, so a silent
            # connection used to buy a thread and a descriptor for the whole
            # handshake budget (see ``DEFAULT_MAX_HANDSHAKES``). Past the cap the
            # socket is closed with no reply, exactly as a refused handshake is: an
            # error frame here would be a probe oracle for a stranger who never
            # authenticated.
            if not self._handshake_slots.acquire(blocking=False):
                self._note_handshake_cap(addr)
                sock.close()
                continue
            thread = threading.Thread(
                target=self._handshake_inbound,
                args=(sock, addr),
                name="mesh-handshake",
                daemon=True,
            )
            thread.start()

    def _note_handshake_cap(self, addr: Any) -> None:
        """Name a pre-auth cap drop in the LOCAL audit record, once per window.

        The dropped connection learns nothing (no reply, no frame) and that is
        deliberate, so this record is the only place the cap is ever named. Without
        it a saturated relay had no local trace at all: the operator's symptom is a
        peer that cannot connect while this device reports nothing, which is the
        "dead instrument" shape this repo treats as a defect of its own.

        One record per :data:`HANDSHAKE_CAP_NOTICE_S` window rather than one per
        connection — see that constant for why a stranger must not be able to churn
        the log. ``subject`` is the first refused peer of the window; later drops in
        the same window are the same story and are not re-told.
        """
        now = time.time()
        if self._cap_notice_at and now - self._cap_notice_at < HANDSHAKE_CAP_NOTICE_S:
            return
        self._cap_notice_at = now
        self.audit.record(
            AuditEvent(
                event="handshake_refused",
                actor="unknown",
                subject=f"{addr[0]}:{addr[1]}" if isinstance(addr, tuple) else str(addr),
                outcome="refused",
                cause="handshake_cap",
                detail={"cause": "handshake_cap", "mode": "unauthenticated"},
            )
        )

    def _handshake_inbound(self, sock: socket.socket, addr: Any) -> None:
        """One accepted connection: take a pre-auth slot, give it back on EVERY exit.

        The slot is released in a ``finally`` rather than at the end of the work
        because the handshake below returns from a dozen places — three refusals,
        a pair phase, a failed welcome write, an admitted link — and a cap that
        leaked a slot on any of them would be a slower version of no cap at all.
        """
        try:
            self._run_inbound_handshake(sock, addr)
        finally:
            self._handshake_slots.release()

    def _run_inbound_handshake(self, sock: socket.socket, addr: Any) -> None:
        """Complete a listener-side handshake, or close silently and audit.

        NO REPLY on any failure: an open port that answers wrong keys with errors is
        an oracle, and the reason is written to the LOCAL audit record instead.
        """
        deadline = wire.deadline_in(self.settings.handshake_timeout_s)
        mode = "member"
        network_id = ""
        peer_addr = f"{addr[0]}:{addr[1]}" if isinstance(addr, tuple) else str(addr)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            reader = wire.FrameReader(sock)
            # Peek at the hello only far enough to learn which network and mode it
            # claims (a policy object cannot exist before that); every field is then
            # re-validated by `accept_hello`, and nothing is trusted from it before
            # the verification order runs.
            peek = reader.read_line(deadline)
            network_id = str(peek.get("network_id") or "")
            mode = str(peek.get("mode") or "member")
            policy = ServerPolicy(self, network_id)
            handshake = Handshake.new(
                role="listener",
                identity=self.identity,
                network_id=network_id,
                epoch=int(peek.get("epoch") or 0),
                instance_id=self.instance_id,
                session_protocol=_session_protocol(),
                mode="join" if mode == "join" else "member",
                capabilities=list(wire.LINK_CAPABILITIES),
                build=self.build,
            )
            handshake.accept_hello(peek)
            invite_id = ""
            if handshake.mode == "join":
                # VALIDATED HERE, WRITTEN LATER. Two things about a join have to
                # happen before the challenge goes out: the invite's own checks
                # (state, freshness, epoch, and the optional device binding), and
                # the credential derived from the invite KEY, because that key is
                # what verifies the joiner's auth frame. NOTHING IS WRITTEN YET, and
                # that is the round-3 fix: marking the invite `redeemed` from the
                # HELLO alone burned a perfectly good invite when a connection
                # dropped before its auth frame, and the honest device's retry was
                # then refused `invite_in_use` — an unauthenticated peer has proved
                # nothing at this point. (The ONE durable write this block can still
                # make is a different rule of the state machine, and it is reasoned
                # about below.)
                #
                # THE ONE WRITE THIS BLOCK CAN STILL MAKE IS DELIBERATE (review round
                # 2, MINOR 1), and it is the reason this is ``claim_or_consume`` and
                # not ``claim``. Design §5.2/§5.4: a BOUND invite presented by
                # another device is CONSUMED, because a token that reached a second
                # device is the leak ``--device`` exists to contain. That decision is
                # taken from the hello's own ``device_id`` — an unauthenticated field
                # — and moving it after the auth frame would not authenticate it but
                # DISABLE it: ``verify_auth`` requires the auth frame to name the same
                # device the hello did and re-derives that id from the key that signs
                # it, so a device presenting a bound invite it does not own can never
                # reach the post-auth block at all. What the write can do is bounded
                # by the same fact: it authorises nothing, deletes nothing, and tells
                # no peer anything (every failure on this path is a silent close), and
                # it can only be reached by a peer able to NAME the invite id — which
                # lives in the token and in this device's own record and in no audit
                # record — i.e. by the leak this rule is for.
                with self._invite_lock:
                    joined = store.load(network_id, self.root)
                    invite_id = str(handshake.join_block.get("invite_id") or "")
                    try:
                        claim_or_consume(
                            joined,
                            invite_id,
                            device_id=handshake.peer_device_id,
                            epoch=handshake.epoch,
                        )
                    except PairingRefusal:
                        # ``claim_or_consume`` may have written the TERMINAL state
                        # (a bound invite presented by another device, design §5.4),
                        # so the save happens on this path too — otherwise the rule
                        # would live only in memory and the token would stay
                        # redeemable by exactly the device it was bound away from.
                        store.save(joined, self.root)
                        raise
                    joined_secrets = store.require_secrets(network_id, self.root)
                    handshake.credential = invite_credential_for(
                        joined, joined_secrets.secret, invite_id
                    )
            handshake.send_challenge(sock, policy)
            handshake.verify_auth(reader, deadline, policy)
            if handshake.mode == "join":
                # NOW the durable state change, and still BEFORE any human is shown
                # a code: the auth frame MACs under the invite key, so this is the
                # earliest moment at which `redeemed` means what it says. Re-checked
                # rather than assumed — a challenge round trip has happened since
                # the read above, and the record on disk is the authority.
                with self._invite_lock:
                    joined = store.load(network_id, self.root)
                    try:
                        claim_or_consume(
                            joined,
                            invite_id,
                            device_id=handshake.peer_device_id,
                            epoch=handshake.epoch,
                        )
                    except PairingRefusal:
                        store.save(joined, self.root)
                        raise
                    mark_redeemed(joined, invite_id, device_id=handshake.peer_device_id)
                    store.save(joined, self.root)
            result = handshake.establish()
        except MeshRefusal as refusal:
            self._audit_handshake_refusal(refusal, network_id, peer_addr, mode)
            _close_quietly(sock)
            return
        except (wire.LinkCryptoError, ConnectionError, OSError, TimeoutError):
            _close_quietly(sock)
            return
        except Exception as exc:  # noqa: BLE001 — never kill the accept loop
            self.audit.record(
                AuditEvent(
                    event="handshake_refused",
                    outcome="failed",
                    network_id=network_id,
                    cause="internal",
                    detail={"cause": str(exc)[:120], "mode": mode},
                )
            )
            _close_quietly(sock)
            return

        record = store.load(network_id, self.root) if network_id else None
        welcome = handshake.welcome_frame(
            phase=result.phase,
            epoch=result.epoch,
            nets=[_net_summary(record)] if record else [],
            capabilities=list(wire.LINK_CAPABILITIES),
            members_digest=members_digest_of(record) if record else "",
            network_name=record.name if record else "",
            endpoints=self.advertised_endpoints(),
        )
        try:
            handshake.send_welcome(sock, welcome)
        except OSError:
            _close_quietly(sock)
            return
        if result.phase == "pair":
            self._run_pair_listener(sock, handshake, result, peer_addr)
            return
        self.register_link(sock, handshake, result, peer_addr, reader=reader)

    def _audit_handshake_refusal(
        self, refusal: MeshRefusal, network_id: str, peer_addr: str, mode: str
    ) -> None:
        # A self-connection is its own event kind, not a generic refusal: it is the
        # one handshake failure that is usually the OPERATOR's mistake rather than an
        # attack, and `lop network log` should say which it was.
        if refusal.code == REASON_SELF:
            self.audit.record(
                AuditEvent(
                    event="self_link",
                    actor="self",
                    subject=peer_addr,
                    outcome="refused",
                    network_id=network_id,
                    detail={"instance_id": ""},
                )
            )
            return
        cause = {
            "unknown_network": "policy",
            "untrusted": "untrusted",
            "epoch_stale": "epoch_stale",
            "invite_epoch_stale": "epoch_stale",
            "not_a_member": "not_a_member",
            "bad_signature": "auth_failed",
            "bad_mac": "auth_failed",
            "protocol_mismatch": "protocol_mismatch",
            "invite_device_mismatch": "wrong_device",
            "invite_invalid": "policy",
            "self_link": "policy",
        }.get(refusal.code, "policy")
        self.audit.record(
            AuditEvent(
                event="handshake_refused",
                actor="unknown",
                subject=peer_addr,
                outcome="refused",
                network_id=network_id,
                cause=cause,
                detail={"cause": refusal.code, "mode": mode, "their_device": peer_addr},
            )
        )

    def register_link(
        self,
        sock: socket.socket,
        handshake: Handshake,
        result: Any,
        peer_addr: str,
        *,
        reader: wire.FrameReader | None = None,
    ) -> PeerLink:
        """Admit a fully-authenticated link, applying the duplicate-identity fence.

        ``reader`` is the handshake's own reader: it may hold the first bytes of the
        record phase already (``wire.FrameReader`` explains why), and handing it on
        is what keeps a peer that speaks immediately after its handshake from
        having that frame decrypted out of sequence.
        """
        link = PeerLink(
            server=self,
            sock=sock,
            result=result,
            codec=handshake.codec(),
            settings=self.settings,
            reader=reader,
        )
        link.peer_addr = peer_addr
        verdict = self.identity_use.observe(
            result.peer_device_id,
            instance_id=result.peer_instance_id,
            link_id=result.link_id,
        )
        if verdict.kind == "duplicate" and verdict.evicted is not None:
            evicted = self.links.get(verdict.evicted.link_id)
            if evicted is not None:
                # Newest wins in both branches, because refusing the new link would
                # let a stale copy pin a device's slot and deny service.
                evicted.send({"op": "net_bye", "reason": "duplicate_identity"})
                evicted.close("replaced")
            self._note_duplicate(result.peer_device_id, result.peer_instance_id)
        elif verdict.kind == "restart":
            self.audit.record(
                AuditEvent(
                    event="link_replaced",
                    actor=result.peer_device_id,
                    subject=result.network_id,
                    network_id=result.network_id,
                    epoch=result.epoch,
                    detail={
                        "instance_id": result.peer_instance_id,
                        "age_s": round(
                            time.time()
                            - (verdict.evicted.last_frame_at if verdict.evicted else 0.0),
                            2,
                        ),
                    },
                )
            )
        with self._links_lock:
            self.links[result.link_id] = link
        link.start()
        # Learn where the peer can be dialled back. This is a DIFFERENT direction
        # from the endpoints we just declared in our own welcome: the hello the
        # peer sent carries ITS endpoints, and this device is the only one that
        # saw it. A dial-only peer declares loopback, which is honest and simply
        # means "do not try to call me first".
        #
        # A RECORD THAT CANNOT BE RELOADED IS NOT AN ERROR HERE: the link is
        # already established and useful, and losing the address hint costs a
        # future dial attempt, not this connection.
        record = next(
            (
                item
                for item in store.list_networks(self.root)
                if item.network_id == result.network_id
            ),
            None,
        )
        if record is not None:
            self._note_peer_endpoints(record, result.peer_device_id, handshake.peer_endpoints)
            # The LISTENER's half of "a completed handshake clears the refusal mark":
            # this device did not dial, so nothing else here would notice that a peer
            # is talking to it again (see `_clear_refusal_mark`).
            self._clear_refusal_mark(record)
        self.audit.record(
            AuditEvent(
                event="link_opened",
                actor=result.peer_device_id,
                subject=result.network_id,
                network_id=result.network_id,
                epoch=result.epoch,
                detail={"role": "listener", "epoch": result.epoch, "phase": result.phase},
            )
        )
        if result.phase == "member":
            # CONTACT RE-EVALUATES MEMBERSHIP (§8.4) — and the LISTENER is the side
            # that would otherwise never hear: this device was dialled, so nothing
            # here asked for the other end's table, and the pull in `dial` only
            # runs on the side that dialled. Gated on the phase because a
            # reconcile-phase link may dispatch `ping` and `net_reconcile` and
            # nothing else (§8.4), so `net_member_list` there is a refusal by
            # design rather than a peer that cannot answer.
            self._pull_members(link)
        return link

    def _note_duplicate(self, device_id: str, instance_id: str) -> None:
        """Record a second live claim on one device id, and flag the member after
        three distinct instances inside the window. Detection is behavioural and the
        flag is ADVISORY: it denies nothing, and the operator's removal is what
        denies."""
        self.audit.record(
            AuditEvent(
                event="duplicate_identity",
                actor=device_id,
                outcome="ok",
                detail={
                    "instance_id": instance_id,
                    "duplicate_count": self.identity_use.recent_instance_count(device_id),
                },
            )
        )
        for record in store.list_networks(self.root):
            member = record.member(device_id)
            if member is None:
                continue
            member.duplicate_count += 1
            member.last_seen_instance = instance_id
            if self.identity_use.recent_instance_count(device_id) >= 3:
                member.suspect = True
            store.save(record, self.root)

    def link_closed(self, link: PeerLink, reason: str) -> None:
        with self._links_lock:
            self.links.pop(link.link_id, None)
        self.identity_use.released(link.device_id, link.link_id)
        self.audit.record(
            AuditEvent(
                event="link_closed",
                actor=link.device_id,
                subject=link.network_id,
                outcome="ok",
                network_id=link.network_id,
                epoch=link.epoch,
                detail={
                    "cause": reason,
                    "frames_in": link.frames_in,
                    "frames_out": link.frames_out,
                },
            )
        )

    def audit_idle(self, link: PeerLink) -> None:
        self.audit.record(
            AuditEvent(
                event="link_idle",
                actor=link.device_id,
                subject=link.network_id,
                network_id=link.network_id,
                epoch=link.epoch,
                cause="timeout",
                detail={
                    "last_seen_at": round(link.last_frame_at, 3),
                    "missed_beats": max(
                        1, int((time.time() - link.last_frame_at) / self.settings.keepalive_s)
                    ),
                },
            )
        )

    def audit_link(self, link: PeerLink, event: str, *, cause: str = "peer-closed") -> None:
        self.audit.record(
            AuditEvent(
                event=event,
                actor=link.device_id,
                subject=link.network_id,
                network_id=link.network_id,
                epoch=link.epoch,
                detail={"cause": cause},
            )
        )

    # -- dispatch (the ONE call site of the authoriser) ---------------------

    def dispatch(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any] | None:
        """Authorise, then handle. The chokepoint, and the only place it is called.

        ``check`` runs BEFORE any field of the frame other than ``op`` and ``req``
        is read, so an unauthorised frame cannot influence what the relay does by
        being malformed in an interesting way.
        """
        req = frame.get("req")
        try:
            granted = self.authorizer.check(link.context, frame)
        except Refusal as refusal:
            return wire.refusal_frame(req, refusal.sentence)
        # DISPATCH ON THE OP THAT ARRIVED, not on what the authoriser resolved it
        # TO. For every op but a carrier those are the same string; for
        # ``net_forward`` they are not — the chokepoint deliberately resolves the
        # INNER op (so the capability checked is the inner one's), while the
        # handler that must run is the CARRIER's, whose whole job is to carry that
        # inner frame. Looking up ``granted.action`` alone answered a forwarded
        # prompt with "prompt is not implemented in this build yet", which is how a
        # carrier that IS implemented reads as one that is not.
        carrier = str(frame.get("op") or "")
        handler = self._handlers.get(carrier) or self._handlers.get(granted.action)
        if handler is None:
            return wire.refusal_frame(
                req,
                f"{granted.action} is not implemented in this build yet "
                f"({_owning_document(granted.action)})",
            )
        if granted.action in ("net_pair_ready", "net_pair_abort", "net_pair_result"):
            return wire.refusal_frame(
                req, f"{granted.action} is only valid on a link that is still pairing"
            )
        try:
            result = handler(link, frame)
        except MeshRefusal as refusal:
            return wire.refusal_frame(req, refusal.sentence)
        except Exception as exc:  # noqa: BLE001 — a handler bug must not close the link
            self.audit.record(
                AuditEvent(
                    event="authorisation_refused",
                    actor=link.device_id,
                    subject=link.network_id,
                    network_id=link.network_id,
                    epoch=link.epoch,
                    outcome="failed",
                    cause="internal",
                    # THE EXCEPTION TEXT IS NOT RECORDED HERE, on purpose: the audit
                    # writer's per-event ``detail`` whitelist drops keys it does not
                    # know, so an ``error`` field here would never reach the file
                    # and would look like it had. The sentence goes to the PEER
                    # instead (below), and ``_fan_out_catalog`` now carries the
                    # peer's sentence into the listing's ``reason``, which is where
                    # the operator reads it.
                    detail={"op": granted.action, "capability": ""},
                )
            )
            return wire.refusal_frame(req, f"{granted.action} failed on this device: {exc}")
        # A REPLY IS ALWAYS AN ACK (design §10.2): ``{"op": "ack", "req": …,
        # "detail": …}``. A handler that answered with its own op name would be
        # indistinguishable from a REQUEST of that name at the peer, which is not a
        # cosmetic problem — two peers then answer each other's replies as if they
        # were new requests, forever (measured at ~600 frames a second on one idle
        # pair before this was fixed, and caught by the e2e test rather than by
        # review).
        if result is None:
            return {"op": "ack", "req": req, "detail": ""}
        if result.get("op") in ("ack", "error"):
            return result
        return {"op": "ack", "req": req, "detail": result}

    # -- reply plumbing -----------------------------------------------------

    def expect_reply(self, link_id: str, req: Any) -> "_ReplyWaiter":
        waiter = _ReplyWaiter()
        with self._links_lock:
            self._reply_waiters[(link_id, req)] = waiter
        return waiter

    def deliver_reply(self, link: PeerLink, frame: dict[str, Any]) -> bool:
        key = (link.link_id, frame.get("req"))
        with self._links_lock:
            waiter = self._reply_waiters.pop(key, None)
        if waiter is None:
            return False
        waiter.set(frame)
        return True

    # -- peer-scope handlers ------------------------------------------------

    def _op_ping(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        return {"op": "ack", "req": frame.get("req"), "detail": "pong"}

    def _op_pair_ready(self, link: PeerLink, _frame: dict[str, Any]) -> dict[str, Any]:
        """Refuse: a pairing frame on an ESTABLISHED link is a protocol error.

        The ceremony runs on a raw socket inside ``_run_pair_listener``, before any
        link exists — so a ``net_pair_ready`` arriving here means a peer is trying to
        drive the pairing state machine through the ordinary dispatch path, which is
        the shape of an attempt to reach a pair-phase exemption from a member link.
        """
        raise MeshRefusal(
            "protocol_error",
            "a pairing frame arrived on an established link; pairing happens before one exists",
        )

    def _op_pair_abort(self, link: PeerLink, _frame: dict[str, Any]) -> dict[str, Any]:
        """Same refusal as ``net_pair_ready``: see that handler."""
        raise MeshRefusal(
            "protocol_error",
            "a pairing frame arrived on an established link; pairing happens before one exists",
        )

    def _op_bye(self, link: PeerLink, frame: dict[str, Any]) -> None:
        self.audit_link(link, "link_closed", cause="peer-closed")
        link.close("peer-closed")
        return None

    def _op_catalog(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """This peer's session rows — a READ-THROUGH of the local run directory.

        One scan per request with a short TTL, because a sidebar polling every
        second must not become a scan storm. The rows carry no transcript content:
        an id, a name, a cwd, a model label and liveness, which is what a listing
        needs and nothing more.
        """
        return {
            "complete": True,
            "device": {
                "device_id": self.identity.device_id,
                "name": self.identity.name,
            },
            "generated_at": time.time(),
            "sessions": self.local_session_rows(),
        }

    def local_session_rows(self) -> list[dict[str, Any]]:
        """This device's rows: every LIVE session, then every stored one.

        A COLD SESSION IS STILL A ROW, and that is not a listing convenience: a
        session that is idle on a peer must still list as remote (mobility §1.2 —
        the record exists only while a runtime does, which is exactly why
        placement cannot live on the record alone), and `net_session_engage`'s
        whole job is to warm an id a viewer can already see. Live rows keep their
        registry state; a stored row says `stored` and carries no pid, so a
        viewer can tell "resident" from "on disk" without a second read.

        The stamp travels with the row (§5.1): placement is additive, always
        present on a row this build writes (`mode: "local"` for an ordinary
        session), so a reader can distinguish "local" from "written by a build
        that does not know about the mesh".
        """
        from local_operator.session.placement import local_placement, read_stamp
        from local_operator.session.runtime import registry

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record, state in registry.scan(self.root):
            seen.add(record.session_id)
            stamp = read_stamp(self.root, record.session_id)
            rows.append(
                {
                    "session_id": record.session_id,
                    "conversation_name": record.conversation_name,
                    "cwd": record.cwd,
                    "model_label": record.model_label,
                    "busy": record.busy,
                    "pending": record.pending,
                    "detached": record.detached,
                    "started": record.started,
                    "pid": record.pid,
                    "kind": record.kind,
                    "capabilities": list(record.capabilities),
                    "state": state,
                    "age_s": 0.0,
                    "placement": (
                        stamp.placement.to_json()
                        if stamp is not None
                        else local_placement().to_json()
                    ),
                    "origin": dict(stamp.origin) if stamp is not None and stamp.origin else None,
                    "archived": None,
                }
            )
        for row in self._stored_rows():
            if row.get("session_id") in seen:
                continue
            rows.append(row)
        return rows

    def _stored_rows(self) -> list[dict[str, Any]]:
        """Sessions with a directory and no live record — the idle half.

        Read through `session.catalog.load_catalog`, the same ranking the sidebar
        adopts as membership, so a stored row here names and dates a conversation
        exactly as the local UI does rather than by a second reading of the
        directory.

        Best effort BY DESIGN: a store that cannot be walked contributes no
        stored rows and leaves the live ones standing. This is a catalogue
        answer, and a device whose `sessions/` is unreadable is a device whose
        live sessions a viewer can still reach.
        """
        from local_operator.session.catalog import load_catalog
        from local_operator.session.placement import local_placement, read_stamp

        rows: list[dict[str, Any]] = []
        try:
            entries = load_catalog(self.root)
        except Exception:  # noqa: BLE001 — a listing is not an error path
            return rows
        for entry in entries:
            session_id = str(getattr(entry.row, "id", "") or "")
            if not session_id:
                continue
            stamp = read_stamp(self.root, session_id)
            rows.append(
                {
                    "session_id": session_id,
                    "conversation_name": str(getattr(entry.row, "name", "") or ""),
                    "cwd": "",
                    "model_label": "",
                    "busy": False,
                    # THE FIELD'S CONTRACT IS A STRING (Q-R7-1). This used to
                    # publish ``bool(entry.unseen)``, which crashed the ONE
                    # human reader of the field — `lop sessions --all-peers`
                    # and `--peer <dev>` render `pending` through
                    # ``rich.cells.cell_len`` — while the live half of the same
                    # catalogue published ``SessionRecord.pending``, a string.
                    # "An unread completion is waiting" is a NEEDS claim, so it
                    # says so in the record's own vocabulary rather than
                    # answering a what-is-needed question with a yes/no
                    # (``types.NEEDS_ASK``, beside the normaliser that reads the
                    # same field off the wire).
                    "pending": NEEDS_ASK if entry.unseen else None,
                    "detached": True,
                    "started": float(getattr(entry.row, "mtime", 0.0) or 0.0),
                    "pid": 0,
                    "kind": "daemon",
                    "capabilities": [],
                    "state": "stored",
                    "age_s": 0.0,
                    "placement": (
                        stamp.placement.to_json()
                        if stamp is not None
                        else local_placement().to_json()
                    ),
                    "origin": (dict(stamp.origin) if stamp is not None and stamp.origin else None),
                    "archived": None,
                }
            )
        return rows

    def _op_member_list(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        return {
            "op": "ack",
            "req": frame.get("req"),
            "detail": {
                "network_id": record.network_id,
                "epoch": record.epoch,
                "members": [member.to_json() for member in record.members],
                "members_digest": members_digest_of(record),
            },
        }

    def _op_epoch(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Apply a rotation from a peer, with the deterministic conflict rules."""
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        state = store.require_secrets(record.network_id, self.root)
        incoming = int(frame.get("epoch") or 0)
        if incoming == record.epoch and str(frame.get("rotation_id") or "") != record.rotations.get(
            str(record.epoch), ""
        ):
            # THE CONCURRENT-ROTATION RULE: same epoch, different rotator. The
            # receiver refuses, audits the conflict, and answers with its own state
            # so the sender learns it lost the race. Convergence comes from the
            # lowest-id rule when both frames arrive before either is applied.
            self.audit.record(
                AuditEvent(
                    event="epoch_conflict",
                    actor=link.device_id,
                    subject=record.network_id,
                    network_id=record.network_id,
                    epoch=incoming,
                    outcome="refused",
                    cause="epoch_stale",
                    detail={
                        "epoch": incoming,
                        "rotation_id": str(frame.get("rotation_id") or ""),
                        "winner": record.rotations.get(str(record.epoch), ""),
                    },
                )
            )
            self._queue_epoch_for(link.device_id, record, state, reason="epoch_conflict")
            return {
                "op": "ack",
                "req": frame.get("req"),
                "detail": {
                    "epoch": record.epoch,
                    "rotation_id": record.rotations.get(str(record.epoch), ""),
                },
            }
        outcome = apply_epoch(record, state, frame, sender_device_id=link.device_id, root=self.root)
        if outcome.applied:
            self.audit.record(
                AuditEvent(
                    event="epoch_rotated",
                    actor=link.device_id,
                    subject=record.network_id,
                    network_id=record.network_id,
                    epoch=record.epoch,
                    detail={
                        "epoch_before": state.previous_epoch,
                        "epoch_after": record.epoch,
                        "rotation_id": record.rotations.get(str(record.epoch), ""),
                        "removed": list(frame.get("removed") or []),
                    },
                )
            )
            self._rehandshake_network(record.network_id, reason="epoch_stale")
        return {
            "op": "ack",
            "req": frame.get("req"),
            "detail": {"epoch": record.epoch, "applied": outcome.applied, "reason": outcome.detail},
        }

    def _op_reconcile(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Answer a previous-epoch link with the current epoch state, and ONLY that.

        Rate limited per device per network: "I keep presenting an old epoch" is
        also what a replayed credential looks like, so a member gets three grants an
        hour and the fourth is refused with a named reason.
        """
        if link.phase != "reconcile":
            raise MeshRefusal("phase_forbidden", "this link is already at the current epoch")
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        member = record.member(link.device_id)
        if member is None or not member.active:
            # A REMOVED device presenting the previous epoch gets NOTHING here —
            # its row is a tombstone and that is checked before any key was tried.
            self.audit.record(
                AuditEvent(
                    event="reconcile_refused",
                    actor=link.device_id,
                    subject=record.network_id,
                    network_id=record.network_id,
                    epoch=link.epoch,
                    outcome="refused",
                    cause="not_a_member",
                    detail={"cause": "not_a_member", "grants_used": 0},
                )
            )
            raise MeshRefusal("not_a_member", "that device is not a member of this network")
        key = (record.network_id, link.device_id)
        now = time.time()
        grants = [
            stamp
            for stamp in self._reconcile_grants.get(key, [])
            if now - stamp < RECONCILE_WINDOW_S
        ]
        if len(grants) >= RECONCILE_MAX_PER_HOUR:
            self.audit.record(
                AuditEvent(
                    event="reconcile_refused",
                    actor=link.device_id,
                    subject=record.network_id,
                    network_id=record.network_id,
                    epoch=link.epoch,
                    outcome="refused",
                    cause="reconcile_rate_limited",
                    detail={"cause": "reconcile_rate_limited", "grants_used": len(grants)},
                )
            )
            raise MeshRefusal(
                "reconcile_rate_limited",
                "too many reconcile requests from this device in the last hour; re-pair if "
                "it is genuinely behind",
            )
        grants.append(now)
        self._reconcile_grants[key] = grants
        state = store.require_secrets(record.network_id, self.root)
        self.audit.record(
            AuditEvent(
                event="reconcile_granted",
                actor=link.device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "epoch_from": link.epoch,
                    "epoch_to": record.epoch,
                    "grants_used": len(grants),
                },
            )
        )
        return {
            "epoch": record.epoch,
            "secret": state.secret,
            "members": [row.to_json() for row in record.members],
            "members_digest": members_digest_of(record),
            "rotations": dict(record.rotations),
            "close_after": True,
        }

    def _op_leave(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """A peer announces its own departure; the lowest-id active admin rotates.

        The rotation is not optional: the leaving device still holds the old
        secret, so continuing to use it would let a device that has left read new
        traffic. Only one member rotates (the deterministic lowest id), so three
        peers receiving the same leave do not produce three epochs.
        """
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        state = store.require_secrets(record.network_id, self.root)
        leave(record, device_id=link.device_id, root=self.root)
        self.audit.record(
            AuditEvent(
                event="member_left",
                actor=link.device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={"epoch": record.epoch},
            )
        )
        rotator = lowest_id_admin(record)
        if rotator == record.self_device_id:
            outcome = rotate_epoch(
                record, state, by=record.self_device_id, reason="member_left", root=self.root
            )
            self._broadcast_epoch(record, state, reason="member_left", removed=outcome.removed)
        return {"op": "ack", "req": frame.get("req"), "detail": {"left": link.device_id}}

    def _op_panic(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        apply_panic(record, frame, sender_device_id=link.device_id, root=self.root)
        self.audit.record(
            AuditEvent(
                event="panic_received",
                actor=link.device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=int(frame.get("epoch") or 0),
                outcome="ok",
                detail={
                    "from_device": link.device_id,
                    "epoch_before": int(frame.get("epoch") or 0),
                    "epoch_after": record.epoch,
                    "reason": str(frame.get("reason") or ""),
                },
            )
        )
        # Every link for this network closes, and a connection that arrives
        # afterwards is refused at handshake step 3 — "refuse all further peer
        # traffic" is a state, not a one-off action.
        for other in list(self.links.values()):
            if other.network_id == record.network_id:
                other.send({"op": "net_bye", "reason": "untrusted"})
                other.close("we-closed")
        return {"op": "ack", "req": frame.get("req"), "detail": "untrusted"}

    def _op_trust(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        trust = trust_state(frame.get("trust") or "active")
        set_trust(record, trust=trust, reason=f"set by {link.device_id}", root=self.root)
        return {"op": "ack", "req": frame.get("req"), "detail": {"trust": trust}}

    def _op_identity_rotate(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        record = self.store_view.network(link.network_id)
        if record is None:
            raise MeshRefusal("not_a_member", "this device is not in that network")
        statement = frame.get("statement")
        if not isinstance(statement, dict):
            raise MeshRefusal("bad_rotation_statement", "no rotation statement was carried")
        member = apply_device_rotation(record, statement, root=self.root)
        self.audit.record(
            AuditEvent(
                event="device_rotated",
                actor=member.device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "old_device": str(statement.get("old_device_id") or ""),
                    "new_device": member.device_id,
                },
            )
        )
        return {"op": "ack", "req": frame.get("req"), "detail": {"device_id": member.device_id}}

    # -- the session plane: create / engage / stop / forward / stream --------
    #
    # WHAT IS HERE. `mesh-session-mobility.md` §2.2's three ops, its §3.3 refusal
    # rules, and the two carriers that make a remote session behave like a local
    # one: `net_forward` (one ControlOp frame, its reply returned) and
    # `net_stream` (a whole viewer connection, pass-through).
    #
    # THE SHAPE THAT MAKES IT SMALL: every one of these ops ends in the same two
    # moves — dial this device's runtime for a session it owns, then hand frames
    # to it. The peer relay implements no session API of its own; it is a dialer.
    # That is what keeps "a remote session is the SAME object as a local one"
    # true rather than aspirational: the vocabulary on the wire is the runtime's
    # own, unchanged.
    #
    # WHY THESE HANDLERS BLOCK. They run on the link's reader thread, so a
    # create/engage (a real spawn, 1-3 s) delays other inbound frames on that
    # link for its duration. Bounded and deliberate for v1: the client waits for
    # this op's answer anyway, and a per-op worker thread would need its own
    # ordering rule against stream pushes on the same link.

    def _peer_block(self, link: PeerLink) -> dict[str, Any]:
        """The dialing peer's identity, for a refusal that has to name it (§4.4)."""
        name = ""
        record = self.store_view.network(link.network_id)
        if record is not None:
            member = record.member(link.device_id)
            name = member.name if member is not None else ""
        return {"device_id": link.device_id, "name": name}

    def _session_record(self, session_id: str) -> Any:
        """The live discovery record for one of THIS device's sessions, or None.

        Read through `registry.scan` — the same discovery every viewer uses —
        rather than from a cache, because the answer is about a pid that may have
        died a heartbeat ago and a stale yes is the one that kills the wrong
        process.
        """
        from local_operator.session.runtime import registry

        for record, _state in registry.scan(self.root):
            if record.session_id == session_id:
                return record
        return None

    def _dial_owned(
        self,
        session_id: str,
        *,
        capabilities: Any = (),
        auth: dict[str, Any] | None = None,
        link: "PeerLink | None" = None,
    ) -> "session_dial.OwnerDial":
        """Dial a session THIS device owns, as a relay (locality `remote`).

        Raises `dial.OwnerUnreachable`. The capability set comes from the LINK,
        not from the frame: §3.3's rule is that the resolved set is the only
        input that decides the runtime's locality gates, and the relaying device
        is the one that resolved it.
        """
        return session_dial.dial_owner(
            self.root,
            session_id,
            capabilities=sorted(str(item) for item in capabilities),
            auth=auth or {},
            peer=self._peer_block(link) if link is not None else None,
            _record=self._session_record(session_id),
        )[0]

    # -- the ops ------------------------------------------------------------

    def _op_forward(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Carry ONE ControlOp frame to a session on this device (§3.3, §6.4).

        A carrier, NOT an authorisation bypass: the authoriser already resolved
        the inner frame's op through `INNER_OP_CAPABILITY` and the session-scope
        rule already proved this device owns the id, both before this ran. So
        the work is a dial, a write, and the reply.

        THE REPLY IS RE-CORRELATED to the OUTER req. The inner reply carries the
        req the inner frame used (the viewer's own numbering), and a requesting
        relay matches replies on the req IT sent — returning the inner reply
        verbatim would leave the requester's waiter unsatisfied and time the op
        out on an answer that had already arrived.
        """
        outer_req = frame.get("req")
        inner = frame.get("frame")
        if not isinstance(inner, dict):
            raise MeshRefusal(
                "protocol_error", "a net_forward frame must carry the frame it forwards"
            )
        session_id = str(inner.get("session_id") or "")
        if not session_id:
            raise MeshRefusal("protocol_error", "a forwarded session frame must name its session")
        try:
            dial = self._dial_owned(session_id, capabilities=link.context.capabilities, link=link)
        except session_dial.OwnerUnreachable as exc:
            raise MeshRefusal("session_unreachable", str(exc)) from exc
        try:
            reply = dial.exchange(inner, timeout_s=self.settings.op_wait_s)
        finally:
            dial.close()
        if reply is None:
            raise MeshRefusal(
                "session_unreachable",
                "that session's runtime stopped answering before it replied",
            )
        if reply.get("op") == "error":
            return {
                "op": "error",
                "req": outer_req,
                "message": str(reply.get("message") or "the owner refused that op"),
                "detail": {"frame": reply},
            }
        return {"op": "ack", "req": outer_req, "detail": {"frame": reply}}

    def _op_session_create(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Create a session ON this device, at another device's request (R8, §5.3).

        THE PEER MINTS THE ID, and that is the rule rather than an implementation
        detail: an id that exists in two places at once is the permanent routing
        ambiguity §2 exists to prevent. So a create frame naming a session id is
        refused, and the id this device mints is the id both ends use from then
        on.

        The steps are the ordinary local `/new` path, executed here: mint, CLAIM
        BEFORE mkdir (the fork invariant — a claimed directory is never one an
        idle sweep could reap), stamp `mesh.json` with `home_device = self`,
        engage a runtime, then admit the optional first prompt.
        """
        if frame.get("session_id"):
            raise MeshRefusal(
                "protocol_error",
                "the device that will own a session mints its id; this frame named one",
            )
        from local_operator.fork import new_session_id
        from local_operator.session.creation import ensure_session_created_at
        from local_operator.session.placement import (
            MeshStamp,
            SessionPlacement,
            write_stamp,
        )
        from local_operator.session.retention import claim_session, release_session

        session_id = new_session_id()
        session_dir = self.root / "sessions" / session_id
        cwd = str(frame.get("cwd") or "") or str(Path.home())
        stamp = MeshStamp(
            session_id=session_id,
            network_id=link.network_id,
            home_device=self.identity.device_id,
            placement=SessionPlacement(
                # `peer` rather than `local`: this device runs it, but it is here
                # because ANOTHER device asked for it, and the placement field's
                # job is to say how the session came to be where it is (§5.1).
                mode="peer",
                network_id=link.network_id,
                home_device=self.identity.device_id,
                policy="pinned",
                stamp_revision=1,
            ),
            origin={
                "kind": str(frame.get("origin") or "user"),
                "source_device": link.device_id,
                "source_session_id": "",
            },
        )
        try:
            # CLAIM BEFORE mkdir, then RELEASE — and the release is not an
            # afterthought. ``claim_session`` writes the CLAIMING process's pid
            # into ``.session.pid``, which is the transcript lease: in
            # ``fork_session`` the claimer and the owner are the same process, so
            # the marker is superseded by the spawned runtime's own claim a moment
            # later. Here the claimer is the RELAY, which owns no session at all
            # (R1) — and a lease naming the relay's live pid is exactly the state
            # ``launch`` reads as "somebody is constructing this right now", so the
            # runtime we then spawn waits out the whole engage deadline and the
            # create fails on a claim its own relay wrote. The claim exists for the
            # mkdir window (so a retention sweep never sees an unclaimed
            # directory); the ownership statement is the stamp, and the lease
            # belongs to whoever runs the session.
            claim_session(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
            ensure_session_created_at(session_dir, time.time())
            write_stamp(self.root, stamp)
            release_session(session_dir)
        except OSError as exc:
            raise MeshRefusal(
                "session_create_failed", f"the session directory could not be created: {exc}"
            ) from exc

        name = str(frame.get("name") or "")
        if name:
            try:
                (session_dir / "title.json").write_text(
                    json.dumps({"title": name, "names": [name]}), encoding="utf-8"
                )
            except OSError:
                # Cosmetic: a title that could not be written is the fork's
                # borrowed-name case, not a failed create.
                pass

        engage_error = self._engage_locally(session_id, cwd=cwd)
        if engage_error:
            return {
                "session_id": session_id,
                "admitted": False,
                "duplicate": False,
                "detail": engage_error,
                "record": self._row_for(session_id),
            }

        model_result: dict[str, Any] = {"applied": False, "detail": ""}
        model = frame.get("model")
        if isinstance(model, dict) and model.get("provider") and model.get("model_id"):
            model_result = self._set_model_on(session_id, model)
        admitted = False
        prompt_detail = ""
        prompt = str(frame.get("prompt") or "")
        if prompt:
            admitted, prompt_detail = self._prompt_on(session_id, prompt, frame.get("images"))
        elif frame.get("images"):
            # Images with no text are not a turn. Refused rather than silently
            # dropped, because a create that discarded them would look like it
            # had started something.
            raise MeshRefusal("protocol_error", "a create frame carried images but no prompt text")
        return {
            "session_id": session_id,
            "admitted": admitted,
            "duplicate": False,
            "detail": prompt_detail,
            "model": model_result,
            "record": self._row_for(session_id),
        }

    def _op_session_engage(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Make an owner exist for one of this device's sessions (warm).

        Carries NO prompt (§2.2): a prompt smuggled into an engage would be a
        second way to start a turn, and the one way is `prompt`.
        """
        session_id = str(frame.get("session_id") or "")
        cwd = str(frame.get("cwd") or "")
        error = self._engage_locally(session_id, cwd=cwd)
        if error:
            return {"engaged": False, "detail": error, "session_id": session_id}
        return {"engaged": True, "detail": "runtime joining", "session_id": session_id}

    def _op_session_stop(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Run THIS device's own kill-switch ladder for one of its sessions (§4.3).

        One implementation, four front ends: `lop stop`, `/stop`, the phone and
        now a peer all reach `control.stop_session`, so the pid-identity proofs,
        the escalation ladder and the refusal sentences are the ones the owning
        machine already has. The viewer renders the peer's vocabulary verbatim
        rather than re-deriving a sentence, because the peer is the only party
        that can see which rung fired.
        """
        from local_operator.session.runtime import control

        session_id = str(frame.get("session_id") or "")
        mode = str(frame.get("mode") or "graceful")
        record = self._session_record(session_id)
        if record is None:
            # The design's own vocabulary: a stop of a session that is not
            # running is an outcome, not an error (`StopOutcome` has "gone").
            return {
                "rung": "none",
                "outcome": "not_running",
                "pid": 0,
                "session_id": session_id,
                "detail": f"{session_id} is not running on this device.",
            }
        outcome = asyncio.run(
            control.stop_session(
                record,
                # `immediate` is the explicit opt-in the ladder documents: it
                # admits the record-field identity proof when the socket cannot
                # answer, instead of refusing on an unprovable pid.
                force=mode == "immediate",
                _root=self.root,
                _command="mesh net_session_stop",
            )
        )
        return {
            "rung": outcome.method,
            "outcome": _STOP_OUTCOME_WORD.get(outcome.method, _STOP_OUTCOME_DEFAULT),
            "pid": outcome.pid,
            "session_id": session_id,
            "wakes_dormant": outcome.wakes_dormant,
            "detail": outcome.line,
        }

    def _op_session_lifecycle(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """Archive/restore/delete on a peer — REFUSED BY NAME in this build.

        `mesh-session-mobility.md` §8 routes these to the OWNER's own
        implementation rather than replicating it: archive is
        `session/archived.py`'s durable index and delete is
        `cleanup.delete_session`, and neither exists in this build — both land
        with `feat/session-archive-delete` (PR #1328). Refusing by name, with the
        document and the module, is the honest answer: half-implementing a delete
        here would put a second `rmtree` of a session directory in the tree,
        which `tests/unit/session/test_no_session_deletion.py` exists to prevent.
        """
        action = str(frame.get("action") or "")
        raise MeshRefusal(
            "not_implemented",
            f"{action or 'that'} on a peer is not available in this build: it runs the "
            "owner's own implementation (mesh-session-mobility.md §8), which lands with "
            "session/archived.py and cleanup.delete_session",
        )

    # -- the two local helpers the ops above share --------------------------

    def _engage_locally(self, session_id: str, *, cwd: str) -> str:
        """Start or join a runtime for a session this device owns.

        Returns an EMPTY STRING on success and a sentence on failure, so the
        two callers (create, engage) can put the sentence in their own frame
        without either of them inventing a second error vocabulary.

        ``cwd`` defaults to this device's home when the caller supplied none —
        §5.3 step 2: an omitted working directory means "the peer decides", and
        the peer's decision is its own home rather than the requesting
        device's path, which would not exist here.
        """
        from local_operator.session.runtime.launch import WarmErrand, engage_runtime

        if not (self.root / "sessions" / session_id).is_dir():
            return f"this device does not hold a session {session_id}"
        started = cwd or str(Path.home())
        try:
            asyncio.run(
                engage_runtime(
                    session_id,
                    started,
                    # Delivers nothing: warming is the whole job here, and the
                    # runtime materialises what it needs when real work arrives.
                    WarmErrand(),
                    config_dir=self.root,
                    deadline_s=ENGAGE_DEADLINE_S,
                )
            )
        except (TimeoutError, RuntimeError, ConnectionError, OSError) as exc:
            return f"this device could not start that session: {exc}"
        return ""

    def _set_model_on(self, session_id: str, model: dict[str, Any]) -> dict[str, Any]:
        """Apply a create's model choice through the runtime's own `set_model`.

        Deliberately NOT parsed here: the runtime validates a model selection
        against the provider catalogue it actually has, and a second validator in
        the relay would be a second answer to "is this model usable". A refusal
        is reported (never swallowed) because a create that silently ran a
        different model than the caller asked for is the kind of quiet wrongness
        that costs a session.
        """
        try:
            dial = self._dial_owned(session_id, capabilities=())
        except session_dial.OwnerUnreachable as exc:
            return {"applied": False, "detail": str(exc)}
        try:
            reply = dial.exchange(
                {
                    "op": "set_model",
                    "req": 1,
                    "provider": str(model.get("provider")),
                    "model_id": str(model.get("model_id")),
                    **({"effort": str(model["effort"])} if model.get("effort") else {}),
                },
                timeout_s=self.settings.op_wait_s,
            )
        finally:
            dial.close()
        if reply is None:
            return {"applied": False, "detail": "the runtime did not answer"}
        if reply.get("op") == "error":
            return {"applied": False, "detail": str(reply.get("message") or "refused")}
        return {"applied": True, "detail": ""}

    def _prompt_on(self, session_id: str, text: str, images: Any) -> tuple[bool, str]:
        """Admit the first prompt of a freshly created session.

        The `command_id` is minted HERE and is a real idempotency key: it rides
        the runtime's own durable-admission path, so a retry of this create
        cannot run the turn twice.
        """
        try:
            dial = self._dial_owned(session_id, capabilities=("prompt",))
        except session_dial.OwnerUnreachable as exc:
            return False, str(exc)
        try:
            reply = dial.exchange(
                {
                    "op": "prompt",
                    "req": 1,
                    "command_id": str(uuid.uuid4()),
                    "text": text,
                    "images": list(images or []),
                },
                timeout_s=self.settings.op_wait_s,
            )
        finally:
            dial.close()
        if reply is None:
            return False, "the runtime did not admit the prompt"
        if reply.get("op") == "error":
            return False, str(reply.get("message") or "the runtime refused the prompt")
        detail = reply.get("detail")
        return True, "" if detail is None else str(detail)

    def _row_for(self, session_id: str) -> dict[str, Any] | None:
        """One session's row from the SAME builder `net_catalog` uses."""
        for row in self.local_session_rows():
            if row.get("session_id") == session_id:
                return row
        return None

    # -- net_stream: the carrier that makes one viewer connection a pipe -------
    #
    # WHY A SECOND CARRIER EXISTS. `net_forward` carries ONE frame and returns
    # its reply. An `AttachedSession` needs more than that: its attach handshake
    # is a welcome projection, then a continuous stream of events, frontend syncs
    # and acks, with the viewer's frames interleaved. `stream_open`'s
    # pass-through mode (mesh-transport-identity.md §2.5, R-IF-1) is the viewer
    # side of that, and `net_stream` is the LINK side of it.
    #
    # WHAT THE PEER RELAY IS, THEREFORE: a pump. On `open` it dials the owning
    # runtime ONCE and keeps the socket; its reader thread pushes every runtime
    # frame to the opening relay as `action: push`; `action: send` writes one
    # viewer frame into the same socket. It never interprets a session frame, so
    # the vocabulary on the wire is the runtime's own — which is what makes
    # `RemoteSessionClient` able to inherit every method of `AttachClient`.
    #
    # DIRECTION AND RELIABILITY. Viewer→peer frames are REQUESTS because a refusal
    # must reach the viewer (a read-only member's `prompt` through a stream is
    # refused with a sentence, not dropped). Peer→viewer frames are PUSHES and are
    # never answered: answering a push is the livelock `PeerLink._handle`
    # documents. Both directions ride the RELIABLE queue: the transport's
    # DROPPABLE class coalesces per (link, stream) and that is right for a
    # repaint, but a coalesced `event` frame is a LOST TRANSCRIPT ROW, so this
    # carrier does not use it. Cost stated plainly: one link round trip per frame
    # per direction until the transport's windowed send lands.

    def _next_relay_req(self) -> int:
        with self._streams_lock:
            self._relay_req += 1
            return self._relay_req

    def _op_stream(self, link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        """The link half of a forwarded viewer stream (§3.2)."""
        action = str(frame.get("action") or "")
        stream_id = str(frame.get("stream") or "")
        if not stream_id:
            raise MeshRefusal("protocol_error", "a net_stream frame must name its stream")
        if action == "open":
            return self._accept_stream(link, frame, stream_id)
        if action == "send":
            return self._accept_stream_frame(link, frame, stream_id)
        if action == "close":
            # CLOSING IS A REQUEST, so it is dispatched and answered rather than
            # routed — but it ENDS a stream, and it goes through the same
            # ownership rule the push path uses. Without it, a member that knew a
            # stream id could kill another member's stream: "unguessable" is not
            # "unforgeable" (round-3 review, MINOR 5). ``unknown_stream`` rather
            # than a distinct refusal, so the answer never confirms whether the id
            # exists on this device.
            if self._stream_for(link, stream_id) is None:
                raise MeshRefusal("unknown_stream", "that stream is not open on this device")
            self._close_stream(stream_id)
            return {"stream": stream_id, "closed": True}
        raise MeshRefusal("protocol_error", f"{action!r} is not a stream action")

    def _accept_stream(
        self, link: PeerLink, frame: dict[str, Any], stream_id: str
    ) -> dict[str, Any]:
        """Open the owner-side half: dial the runtime once, then pump.

        The session-scope rule has already run at the chokepoint, so a stream for
        an id this device does not own never reaches here (INV-1).
        """
        session_id = str(frame.get("session_id") or "")
        auth = frame.get("auth") if isinstance(frame.get("auth"), dict) else {}
        try:
            # CAPABILITIES COME FROM THE LINK, never from the frame (§3.3): the
            # relaying device is the one that resolved them from the member row,
            # and a frame claiming its own grant would be a client deciding its
            # own authority.
            dial, welcome = session_dial.dial_owner(
                self.root,
                session_id,
                capabilities=sorted(link.context.capabilities),
                auth=auth,
                peer=self._peer_block(link),
                _record=self._session_record(session_id),
            )
        except session_dial.OwnerUnreachable as exc:
            raise MeshRefusal("session_unreachable", str(exc)) from exc
        stream = _Stream(
            stream_id=stream_id,
            session_id=session_id,
            peer_device_id=link.device_id,
            link=link,
            dial=dial,
        )
        with self._streams_lock:
            self._streams[stream_id] = stream
        # The welcome goes back as the stream's first frame. It is queued BEFORE
        # this handler's ack, and that is safe rather than an ordering bug: the
        # opening relay buffers every push until it has written its own ack to
        # its viewer (see _Stream.pending), so the viewer still reads the
        # response to stream_open first.
        link.send({"op": "net_stream", "action": "push", "stream": stream_id, "frame": welcome})
        stream.pump = threading.Thread(
            target=self._stream_pump,
            args=(stream,),
            name=f"mesh-stream-{stream_id}",
            daemon=True,
        )
        stream.pump.start()
        self.audit.record(
            AuditEvent(
                event="session_stream_opened",
                actor=link.device_id,
                subject=session_id,
                outcome="ok",
                network_id=link.network_id,
                epoch=link.epoch,
                detail={"stream": stream_id},
            )
        )
        return {"stream": stream_id, "session_id": session_id, "opened": True}

    def _accept_stream_frame(
        self, link: PeerLink, frame: dict[str, Any], stream_id: str
    ) -> dict[str, Any]:
        """Write one viewer frame into the owner's socket, or refuse it by name.

        THE SECOND CAPABILITY CHECK, and the reason `net_stream` is not a bypass:
        opening a stream costs `view`, and that would let a read-only member then
        write anything it liked down the pipe. Each forwarded frame is therefore
        resolved through `INNER_OP_CAPABILITY` exactly as `net_forward`'s inner
        frame is — one table, one answer, for both carriers.
        """
        stream = self._stream_for(link, stream_id)
        if stream is None or stream.closed or stream.dial is None:
            raise MeshRefusal("unknown_stream", "that stream is not open on this device")
        inner = frame.get("frame")
        if not isinstance(inner, dict):
            raise MeshRefusal("protocol_error", "a stream frame must carry the frame it forwards")
        inner_op = str(inner.get("op") or "")
        from local_operator.network.types import INNER_OP_CAPABILITY

        required = INNER_OP_CAPABILITY.get(inner_op)
        if required is None:
            raise MeshRefusal(
                "unknown_op",
                f"the forwarded op {inner_op!r} has no capability decision, so it was "
                "refused rather than carried",
            )
        if required not in link.context.capabilities:
            raise MeshRefusal(
                "not_authorised",
                f"{link.device_id} may not do that here (it does not hold the "
                f"{required!r} capability)",
            )
        try:
            stream.dial.send(inner)
        except OSError as exc:
            self._close_stream(stream_id)
            raise MeshRefusal(
                "session_unreachable", f"the owner's socket went away: {exc}"
            ) from exc
        return {"stream": stream_id, "delivered": True}

    def _stream_pump(self, stream: _Stream) -> None:
        """Push the owner's frames to the opening relay until either end goes."""
        dialect = stream.dial
        assert dialect is not None
        while not stream.closed and not self._stop.is_set():
            frame = dialect.recv(0.5)
            if frame is None:
                if dialect.eof:
                    break
                continue
            if stream.link is None or not stream.link.send(
                {"op": "net_stream", "action": "push", "stream": stream.stream_id, "frame": frame}
            ):
                break
        stream.closed = True
        # Tell the opening relay the owner is gone so its viewer stops waiting.
        if stream.link is not None:
            stream.link.send(
                {
                    "op": "net_stream",
                    "action": "closed",
                    "stream": stream.stream_id,
                    "reason": "owner-gone",
                }
            )
        self._drop_stream(stream.stream_id)

    # -- the opening relay's half of a stream --------------------------------

    def _resolve_peer(self, target: str) -> str:
        """The DEVICE ID a person meant, from an id or a name they typed.

        A shell user types ``build-box`` and a device id is a fingerprint
        (``d_6010dd…``); every op below takes the id, so the translation happens
        once, here. An ambiguous name is REFUSED rather than guessed: two devices
        called ``laptop`` is a real possibility on a network with two of them, and
        asking the wrong one about a session is not a mistake a refusal fixes.
        A removed member is not a match at all — it is not a device you can ask.
        """
        if not target:
            raise MeshRefusal("peer_required", "name a device: --peer <device id or name>")
        matches: set[str] = set()
        for record in store.list_networks(self.root):
            for member in record.members:
                if member.device_id == self.identity.device_id or not member.active:
                    continue
                if target in (member.device_id, member.name):
                    matches.add(member.device_id)
        if not matches:
            raise MeshRefusal(
                "unknown_peer",
                f"this device is not in a network with anything called {target!r}",
            )
        if len(matches) > 1:
            raise MeshRefusal(
                "ambiguous_peer",
                f"{target!r} matches {len(matches)} devices; use the device id",
            )
        return matches.pop()

    def _ensure_link(self, device_id: str) -> "PeerLink | None":
        """A live link to ``device_id``, dialling its recorded endpoints if needed.

        A viewer's relay must be able to reach a peer the viewer was told about,
        and on a dial-only install the relay holds no inbound link at all — so a
        missing link is a dial to the endpoint the member row records, not an
        error. The member's own endpoints are the only place a device has ever
        advertised where it can be reached.
        """
        found = self._link_for(device_id)
        if found is not None:
            return found
        return self._ensure_link_with_reason(device_id)[0]

    def _ensure_link_with_reason(
        self, device_id: str, *, probe_timeout_s: float | None = None
    ) -> tuple["PeerLink | None", str]:
        """``_ensure_link`` plus WHY it failed, for a surface that must say so.

        A listing that reports ``reachable: false`` with no reason is the
        dead-instrument failure this repo warns about: it cannot be told from
        "the peer holds no sessions". The three honest answers are
        ``no_endpoint`` (nothing was ever declared for this member — the state
        every paired peer was in before F-2), the dial's own refusal code, and
        ``not_a_member`` when no network record knows the device at all.

        ``probe_timeout_s`` bounds the WHOLE probe, across every endpoint: a
        listing must return even when a member's address is a black hole, and a
        surface that hung for one dead peer would be worse than one that says it
        could not reach it. Every declared endpoint is dialled WITHIN that bound —
        all of them at once, so the row's order stops deciding who is reachable
        (:func:`probe_candidates`) — and a member the budget ran out before is
        reported as ``not_attempted`` rather than as unreachable.
        """
        found = self._link_for(device_id)
        if found is not None:
            return found, ""
        if probe_timeout_s is not None and probe_timeout_s <= 0:
            return None, NOT_ATTEMPTED_REASON
        deadline = None if probe_timeout_s is None else time.monotonic() + probe_timeout_s
        # A probe with a deadline is a listing's probe: cap one address at
        # PROBE_CONNECT_TIMEOUT_S so a black hole costs its own cap and not the
        # whole budget. A probe WITHOUT one is a real session op's dial (stream
        # open, session create) and keeps the full handshake budget per attempt,
        # which is what it had before this and what a slow WAN leg needs.
        cap = PROBE_CONNECT_TIMEOUT_S if deadline is not None else self.settings.handshake_timeout_s
        reason = "not_a_member"
        for record in store.list_networks(self.root):
            member = record.member(device_id)
            if member is None or not member.active:
                continue
            if not member.endpoints:
                reason = "no_endpoint"
                continue
            probe = probe_candidates(member.endpoints, deadline=deadline, connect_cap=cap)
            if probe.sock is None:
                reason = probe.reason
                continue
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                # The address answered and the budget ran out before the
                # handshake: "unreachable" would be a claim about the peer when
                # the truth is a claim about our clock.
                _close_quietly(probe.sock)
                return None, (
                    f"handshake_not_attempted: {probe.winner} answered and the listing "
                    "budget ran out before the handshake"
                )
            link, dial_reason = self.dial(
                record.network_id,
                host=probe.winner,
                epoch=record.epoch,
                timeout_s=remaining,
                connected=probe.sock,
            )
            if link is not None:
                return link, ""
            reason = dial_reason or probe.reason
        return None, reason

    def _open_viewer_stream(self, frame: dict[str, Any]) -> tuple[dict[str, Any], _Stream | None]:
        """The `stream_open` local op: reach a peer and open the pipe.

        The ack is written to the viewer BEFORE any owner frame, and the stream
        carries a random id rather than the session id so that a PUSH can be
        authorised by "you opened this stream" rather than by a capability the
        pushing side would have to be trusted about.
        """
        req = frame.get("req")
        peer = str(frame.get("peer") or "")
        session_id = str(frame.get("session_id") or "")
        auth = frame.get("auth") if isinstance(frame.get("auth"), dict) else {}
        if not peer or not session_id:
            return {
                "op": "error",
                "req": req,
                "message": "a stream needs both a peer and a session id",
            }, None
        link = self._ensure_link(peer)
        if link is None:
            return {
                "op": "error",
                "req": req,
                "message": f"{peer} cannot be reached from this device right now",
            }, None
        stream_id = "s" + os.urandom(8).hex()
        stream = _Stream(
            stream_id=stream_id,
            session_id=session_id,
            peer_device_id=peer,
            link=link,
        )
        with self._streams_lock:
            self._streams[stream_id] = stream
        reply = link.request(
            {
                "op": "net_stream",
                "req": self._next_relay_req(),
                "action": "open",
                "stream": stream_id,
                "session_id": session_id,
                "auth": auth,
                "locality": "remote",
            },
            timeout=self.settings.op_wait_s,
        )
        if reply is None or reply.get("op") != "ack":
            message = str(
                (reply or {}).get("message") or "the device holding that session did not answer"
            )
            self._drop_stream(stream_id)
            return {"op": "error", "req": req, "message": message}, None
        self.audit.record(
            AuditEvent(
                event="session_stream_opened",
                actor=self.identity.device_id,
                subject=session_id,
                outcome="ok",
                network_id=link.network_id,
                epoch=link.epoch,
                detail={"stream": stream_id, "peer": peer},
            )
        )
        return {
            "op": "ack",
            "req": req,
            "detail": {"stream": stream_id, "session_id": session_id, "peer": peer},
        }, stream

    def _forward_stream_frame(self, stream: _Stream, frame: dict[str, Any]) -> None:
        """One viewer frame down the pipe; a refusal is written back to it."""
        if stream.link is None:
            self._close_stream(stream.stream_id)
            return
        reply = stream.link.request(
            {
                "op": "net_stream",
                "req": self._next_relay_req(),
                "action": "send",
                "stream": stream.stream_id,
                "frame": frame,
            },
            timeout=self.settings.op_wait_s,
        )
        if reply is None or reply.get("op") == "error":
            message = str(
                (reply or {}).get("message") or "the device holding that session stopped answering"
            )
            stream.write_to_viewer({"op": "error", "req": frame.get("req"), "message": message})
            self._close_stream(stream.stream_id)

    def _stream_for(self, link: PeerLink, stream_id: str) -> "_Stream | None":
        """The stream ``stream_id`` names, IF this link is the one that owns it.

        THE OWNERSHIP RULE, in one place. A stream id is unpredictable
        (``os.urandom``) and that is what authorises a push — but unpredictable is
        not unforgeable: a member that learns an id from a log, a traceback or a
        bug must not be able to write into, or end, another member's stream. The
        push and closed paths below check this; ``send`` and ``close``, which
        arrive as REQUESTS through ``_op_stream``, did not, so knowing the id was
        enough (round-3 review, MINOR 5). Returns ``None`` rather than raising, so
        each caller answers with its own shape.
        """
        with self._streams_lock:
            stream = self._streams.get(stream_id)
        if stream is None or stream.link is not link:
            return None
        return stream

    def route_stream_push(self, link: PeerLink, frame: dict[str, Any]) -> bool:
        """Deliver one pushed session frame, or answer "not mine".

        Called from `PeerLink._handle` BEFORE dispatch, because a push is not a
        request: it has no reply, and answering it would be the livelock that
        method's stray-reply comment describes. Authorised by THIS device's own
        stream table — only a peer that successfully opened the (unpredictably
        named) stream can push on it.
        """
        stream_id = str(frame.get("stream") or "")
        stream = self._stream_for(link, stream_id)
        if stream is None:
            return False
        payload = frame.get("frame")
        if isinstance(payload, dict):
            stream.write_to_viewer(payload)
        return True

    def route_stream_closed(self, link: PeerLink, frame: dict[str, Any]) -> bool:
        """Mark a stream dead when the peer says the owner is gone."""
        stream_id = str(frame.get("stream") or "")
        if self._stream_for(link, stream_id) is None:
            return False
        self._close_stream(stream_id, notify_peer=False)
        return True

    def _drop_stream(self, stream_id: str) -> None:
        """Forget a stream without touching either socket."""
        with self._streams_lock:
            self._streams.pop(stream_id, None)

    def _close_stream(self, stream_id: str, *, notify_peer: bool = True) -> None:
        """End one stream: close what this device opened, and only that.

        QUIT SAFETY LIVES HERE. Closing the viewer's stream closes the peer
        relay's DIAL — one attach client — and nothing else: the runtime on the
        peer keeps its transcript, its lease and its life, so a laptop quitting
        its TUI cannot stop a session on another device. The op that would look
        like it is `retire_if_pristine`, and the peer relay never sends it on a
        remote viewer's behalf (§3.3).
        """
        with self._streams_lock:
            stream = self._streams.pop(stream_id, None)
        if stream is None:
            return
        stream.closed = True
        if stream.dial is not None:
            stream.dial.close()
        if stream.viewer_sock is not None:
            try:
                _close_quietly(stream.viewer_sock)
            except OSError:
                pass
        if notify_peer and stream.dial is not None and stream.link is not None:
            stream.link.send({"op": "net_stream", "action": "close", "stream": stream_id})

    # -- the pair ceremony, listener side -----------------------------------

    def _inviter_human_step(
        self,
        *,
        record: NetworkRecord,
        invite_id: str,
        result: Any,
        joiner_id: str,
        joiner_name: str,
        transcribed: str,
        peer_addr: str,
    ) -> PairDecision:
        """Design §5.3's second half: a person on the INVITING device confirms the code.

        The joiner's transcription has already been checked against this device's
        derivation. This is the other direction, and it is the half that makes the
        check mutual rather than decorative: the inviter's human reads the code THIS
        device derived and says whether the other screen shows the same digits. Both
        must hold, so a single mistyped digit anywhere ends the ceremony.

        WHY IT CAN BE ANSWERED LATER. The relay is usually a launchd daemon with no
        terminal, so the question is parked in a 0600 pending record that carries
        both codes, the relay prints it when it HAS a terminal, and otherwise the
        operator answers with `lop network confirm`. Parking it is what makes the
        prompt work in the deployment the design actually runs in; printing it and
        blocking would hang a daemon forever.
        """
        invite = record.invite(invite_id)
        role = (invite.role if invite is not None else "read") or "read"
        window = pair_timeout_seconds(invite.ttl_s if invite is not None else 0.0)
        pending = PendingPairing(
            invite_id=invite_id,
            network_id=record.network_id,
            network_name=record.name,
            joiner_device_id=joiner_id,
            joiner_name=joiner_name,
            sas=result.sas,
            fingerprint=wire.transcript_fingerprint(bytes.fromhex(result.transcript_hash)),
            transcribed=transcribed,
            peer_addr=peer_addr,
            expires_at=time.time() + window,
            prompt=inviter_prompt_for(
                network_name=record.name,
                role=role,
                device_id=joiner_id,
                name=joiner_name,
                transcribed=transcribed,
                derived=result.sas,
            ),
        )
        store.save_pending_pairing(pending, self.root)
        self.audit.record(
            AuditEvent(
                event="pairing_awaiting_confirmation",
                actor=joiner_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "subject": joiner_id,
                    "role": role,
                    "seconds_left": round(window, 1),
                },
            )
        )
        try:
            decision = self._await_pairing_decision(pending, window)
        finally:
            # Cleared either way: the code must not outlive the ceremony it belongs
            # to, and a leftover decision must never be read by the NEXT pairing.
            store.clear_pending_pairing(invite_id, self.root)
            store.clear_pair_decision(invite_id, self.root)
        self.audit.record(
            AuditEvent(
                event="pairing_confirmed" if decision.matched else "pairing_refused",
                actor=joiner_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                outcome="ok" if decision.matched else "refused",
                cause="" if decision.matched else (decision.reason or "declined"),
                detail={
                    "subject": joiner_id,
                    "role": role,
                    "answered_by": decision.answered_by,
                    "cause": decision.reason,
                },
            )
        )
        return decision

    def _await_pairing_decision(self, pending: PendingPairing, window: float) -> PairDecision:
        """The human's answer: inline when this process owns a terminal, else from disk."""
        if self._has_terminal():
            answer = self._ask_in_terminal(pending)
            if answer is None:
                # EOF or ctrl-c at the prompt. Not a licence to admit: an answer that
                # never arrived is a refusal, exactly like the timeout.
                return PairDecision(
                    invite_id=pending.invite_id,
                    decision="decline",
                    matched=False,
                    reason="unanswered",
                    answered_by="human",
                )
            return PairDecision(
                invite_id=pending.invite_id,
                decision="admit" if answer else "decline",
                matched=answer,
                reason="" if answer else "declined",
                answered_by="human",
            )
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            decision = store.pair_decision(pending.invite_id, self.root)
            if decision is not None:
                return decision
            time.sleep(0.2)
        return PairDecision(
            invite_id=pending.invite_id,
            decision="decline",
            matched=False,
            reason="timeout",
            answered_by="relay",
        )

    @staticmethod
    def _has_terminal() -> bool:
        """Whether THIS process can ask a human directly.

        A launchd daemon has no controlling terminal and a redirected run has no
        TTY either, so the answer is not assumed: `serve` in a terminal prompts
        inline, and every other shape reads the decision `lop network confirm`
        writes. `isatty` can raise on a closed stream, and a relay that crashed
        while asking would be a worse failure than one that asks on disk.
        """
        return _has_terminal()

    @staticmethod
    def _ask_in_terminal(pending: PendingPairing) -> bool | None:
        """Show both codes and take a yes/no. ``None`` means no answer arrived."""
        print(pending.prompt)
        sys.stdout.flush()
        try:
            answer = input(f"confirm {pending.joiner_device_id}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        return answer in ("y", "yes")

    def _run_pair_listener(
        self, sock: socket.socket, handshake: Handshake, result: Any, peer_addr: str
    ) -> None:
        """The inviter's half of the human step, then admission.

        Fail closed at every branch, and the invite is consumed in EVERY one of
        them: an attacker's next attempt then needs a fresh invite, which is
        another human action on the inviter.
        """
        # OUR OWN ENDPOINTS MUST BE ON THE SELF ROW OF THE FRAME THIS JOINER IS
        # ABOUT TO RECEIVE. The admission frame is the one carrier a joiner gets,
        # and a row written by `init` carries only what init could see. Syncing
        # here (once per pairing) makes the frame authoritative, and it is what
        # gives the joiner an endpoint to dial back (QA round 1, F-2).
        self.sync_self_endpoints()
        record = store.load(result.network_id, self.root)
        invite_id = str(handshake.join_block.get("invite_id") or "")
        joiner_id = result.peer_device_id
        joiner_key = str(handshake.join_block.get("joiner_public_key") or "")
        joiner_name = str(handshake.join_block.get("joiner_name") or "")
        codec = handshake.codec()
        reader = wire.FrameReader(sock)
        deadline = wire.deadline_in(
            min(self.settings.handshake_timeout_s, _ttl_of(record, invite_id))
        )
        member_row: MemberRecord | None = None
        try:
            ready = codec.open(reader.read_record_payload(deadline))
            if ready.get("op") != "net_pair_ready":
                raise PairingRefusal("protocol_error", "the joining device did not confirm a code")
            typed = str(ready.get("sas") or "")
            if not sas_matches(result.sas, typed):
                raise PairingRefusal("sas_mismatch", "the transcribed code did not match")
            # The SAS check is mutual and local: this device's own derivation is the
            # only thing it compares against, which is why a peer cannot echo it.
            #
            # A REFUSAL THIS DEVICE CAN ALREADY DECIDE IS DECIDED HERE, BEFORE A HUMAN
            # IS ASKED. Whether this joiner's id is burned, and whether it is claiming
            # an id this record already holds under another key, are LOCAL facts: the
            # record is in hand and the joiner has been named since the handshake.
            #
            # WHY THE POSITION MATTERS (measured, round-4 review): discovering it after
            # the human step made a CORRECT refusal depend on the confirmation window —
            # on a loaded host the same burned-joiner test came back as `timeout` in 2
            # runs of 5, because no confirmation landed inside the window and the
            # listener then refused with the window's expiry instead of with the reason
            # it was always going to refuse with. It also asked an operator to compare
            # six digits for a pairing that could not succeed. The sentence is
            # :func:`admit`'s (one owner), and this branch consumes the invite exactly
            # as every other refusal does.
            conflict = membership_conflict(record, joiner_id, joiner_key)
            if conflict:
                raise PairingRefusal("device_id_conflict", conflict)
            decision = self._inviter_human_step(
                record=record,
                invite_id=invite_id,
                result=result,
                joiner_id=joiner_id,
                joiner_name=joiner_name,
                transcribed=typed,
                peer_addr=peer_addr,
            )
            if not (decision.decision == "admit" and decision.matched):
                # A timeout and a decline are both refusals: an unanswered question
                # must never admit a device, and §5.4 makes every one of these
                # terminal for the invite.
                raise PairingRefusal(
                    "timeout" if decision.reason == "timeout" else "declined_remote",
                    (
                        "the pairing timed out before both people confirmed"
                        if decision.reason == "timeout"
                        else "the code was not confirmed on the inviting device, so nothing "
                        "was admitted"
                    ),
                )
            member_row = admit(
                record,
                device_id=joiner_id,
                public_key=joiner_key,
                name=joiner_name,
                role=_invite_role(record, invite_id),
                capabilities=sorted(_invite_capabilities(record, invite_id)),
                added_by=record.self_device_id,
                added_via="invite",
                # What the JOINER declared about itself in its hello, falling back
                # to the observed source address only when it declared nothing.
                # The observed address is an ephemeral port, so it is a last
                # resort: it is why every paired peer used to be unreachable the
                # moment the pairing link closed (QA round 1, F-2).
                endpoints=list(handshake.peer_endpoints) or [peer_addr],
                root=self.root,
            )
            consume(record, invite_id, outcome="admitted")
            store.save(record, self.root)
            state = store.require_secrets(result.network_id, self.root)
            frame = pair_result_frame(
                # The joiner's correlation id, echoed so it can match this answer
                # to its own request. It is read off an UNTRUSTED frame, so a
                # missing or non-integer id answers 0 rather than putting a null
                # (or a string) on the wire where the joiner's frame carried an int.
                req=int(ready["req"]) if isinstance(ready.get("req"), int) else 0,
                admit=True,
                network={
                    "network_id": record.network_id,
                    "name": record.name,
                    "epoch": record.epoch,
                    "sequence": record.sequence,
                    "trust": record.trust,
                },
                member=member_row.to_json(),
                members=[row.to_json() for row in record.members],
                members_digest=members_digest_of(record),
                material=state.secret,
                rotations=dict(record.rotations),
            )
            sock.sendall(codec.seal(frame))
            self.audit.record(
                AuditEvent(
                    event="member_admitted",
                    actor=joiner_id,
                    subject=record.network_id,
                    network_id=record.network_id,
                    epoch=record.epoch,
                    detail={
                        "role": member_row.role,
                        "member_kind": member_row.kind,
                        "epoch": record.epoch,
                    },
                )
            )
            # The link's phase flips to member IN PLACE: no re-handshake. The SAS
            # already proved the channel end to end with both humans at the
            # keyboard at that instant; the secrets did not change; and tearing down
            # and redialling would create a window in which the joiner holds the
            # secret but is not yet a member.
            result.phase = "member"
            link = PeerLink(
                server=self,
                sock=sock,
                result=result,
                codec=codec,
                settings=self.settings,
                reader=reader,
            )
            link.peer_addr = peer_addr
            with self._links_lock:
                self.links[result.link_id] = link
            link.start()
            # THE ADMITTING DEVICE OWES THE REST OF THE NETWORK THIS NEWS. The
            # joiner was handed the full member list in its admission frame; the
            # devices already in the network were told nothing, so a member
            # admitted later stays invisible to them forever (Q-R2-1). There is no
            # membership PUSH on the wire (§6.4 has one membership read,
            # `net_member_list`), so the delivery is a contact: this device makes
            # one, and every live link — INCLUDING the links that were already up
            # before this admission — re-pulls the table from here
            # (`refresh_membership`). Bounded and best effort: a peer this device
            # cannot dial is one that will pull when it next contacts, which is the
            # honest limit of a dial-only member.
            self.contact_peers()
        except (MeshRefusal, wire.LinkCryptoError, OSError) as exc:
            reason = getattr(exc, "code", "error")
            try:
                if record is not None and acquire_invite(record, invite_id):
                    consume(record, invite_id, outcome=reason)
                    store.save(record, self.root)
                # THE REFUSING DEVICE'S OWN SENTENCE GOES WITH THE CODE. It is the
                # only place the joiner can learn WHICH id was refused and what to do
                # about it: `device_id_conflict` alone is a dead end, and this is the
                # line where that explanation used to be dropped (Q-R3-3).
                sock.sendall(
                    codec.seal(
                        pair_abort_frame(
                            req=0,
                            reason=reason,
                            detail=getattr(exc, "sentence", "") or str(exc),
                        )
                    )
                )
            except OSError:
                pass
            self.audit.record(
                AuditEvent(
                    event="pairing_refused",
                    actor=joiner_id,
                    subject=record.network_id if record else "",
                    outcome="refused",
                    network_id=record.network_id if record else "",
                    cause=_PAIR_CAUSE.get(reason, "policy"),
                    detail={"cause": reason, "subject": joiner_id},
                )
            )
            _close_quietly(sock)

    # -- fan-out and the durable outbox -------------------------------------

    def _broadcast_epoch(
        self, record: NetworkRecord, state: SecretState, *, reason: str, removed: list[str]
    ) -> None:
        """Send ``net_epoch`` to every live link, and QUEUE it for the offline ones.

        The frame is built PER RECIPIENT by :func:`epoch_frame`, which withholds the
        secret from a device named in ``removed`` — and the queued copy goes through
        ``store.enqueue_frame``, which refuses to persist a secret for a removed
        recipient at all.
        """
        for link in list(self.links.values()):
            if link.network_id != record.network_id:
                continue
            link.send(epoch_frame(record, state, reason=reason, target_device_id=link.device_id))
        online = {
            link.device_id for link in self.links.values() if link.network_id == record.network_id
        }
        for member in record.active_members():
            if member.device_id in online or member.device_id == record.self_device_id:
                continue
            self._queue_epoch_for(member.device_id, record, state, reason=reason)

    def _queue_epoch_for(
        self, device_id: str, record: NetworkRecord, state: SecretState, *, reason: str
    ) -> None:
        member = record.member(device_id)
        removed = member is None or not member.active
        try:
            store.enqueue_frame(
                device_id,
                epoch_frame(record, state, reason=reason, target_device_id=device_id),
                removed=removed,
                root=self.root,
            )
        except MeshRefusal as refusal:
            # The rule fires here, at the writer, where it cannot be forgotten.
            self.audit.record(
                AuditEvent(
                    event="epoch_rotated",
                    actor=record.self_device_id,
                    subject=device_id,
                    network_id=record.network_id,
                    epoch=record.epoch,
                    outcome="refused",
                    cause="revoked",
                    detail={
                        "epoch_before": record.epoch - 1,
                        "epoch_after": record.epoch,
                        "rotation_id": record.rotations.get(str(record.epoch), ""),
                        "removed": [device_id],
                    },
                )
            )
            del refusal

    def _flush_outboxes(self) -> None:
        """Replay queued frames to peers that are reachable again."""
        for record in store.list_networks(self.root):
            for member in record.active_members():
                queued = store.queued_frames(member.device_id, self.root)
                if not queued:
                    continue
                link = self._link_for(member.device_id)
                if link is None:
                    continue
                for path, frame in queued:
                    if link.send(frame):
                        store.drop_frame(path)

    def _link_for(self, device_id: str) -> PeerLink | None:
        for link in self.links.values():
            if link.device_id == device_id and link.alive:
                return link
        return None

    def dial(
        self,
        network_id: str,
        *,
        host: str,
        epoch: int | None = None,
        mode: str = "member",
        invite: str = "",
        invite_material: str = "",
        joiner_name: str = "",
        timeout_s: float | None = None,
        connected: socket.socket | None = None,
    ) -> tuple[PeerLink | None, str]:
        """Dial a peer and complete the handshake. Returns the link and a reason.

        Dial-only installs reach their peers from here, which is the supported
        configuration where neither side can accept an inbound connection.

        ``invite_material`` is the secret the TOKEN carried. A join's only
        credential is that token — the joiner does not hold the network secret yet,
        which is the point of a join — so it arrives as a parameter rather than
        being looked up, because on this side there is nothing to look it up from.

        ``timeout_s`` OVERRIDES the handshake budget for one dial. A probe (a
        listing asking "is this peer reachable?") needs a bound shorter than the
        budget a real session op should get, and it must be able to say "could
        not reach it" rather than hang.

        ``connected`` is a socket ALREADY dialled by :func:`probe_candidates`,
        whose job was to choose WHICH of a member's declared addresses to use.
        Re-dialling here would throw that choice away — and, on the row this
        exists for, re-open the race where the first declared address is a black
        hole (QA round 2, Q-R2-2). ``host`` is still required: it names the
        address the probe settled on, and it is what the audit record and
        ``link.peer_addr`` report.
        """
        budget = self.settings.handshake_timeout_s if timeout_s is None else float(timeout_s)
        record = store.load(network_id, self.root)
        state = store.require_secrets(network_id, self.root)
        if connected is not None:
            sock = connected
        else:
            address, _, port_text = host.rpartition(":")
            try:
                port = int(port_text)
            except ValueError:
                return None, "bad_endpoint"
            try:
                sock = socket.create_connection((address or host, port), timeout=budget)
            except OSError as exc:
                return None, f"connect_failed:{exc.__class__.__name__}"
        deadline = wire.deadline_in(budget)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            handshake = Handshake.new(
                role="dialer",
                identity=self.identity,
                network_id=network_id,
                epoch=record.epoch if epoch is None else epoch,
                instance_id=self.instance_id,
                session_protocol=_session_protocol(),
                mode="join" if mode == "join" else "member",
                capabilities=list(wire.LINK_CAPABILITIES),
                build=self.build,
                # We DECLARE where we can be reached on every dial, so a peer that
                # can only ever answer learns an address to dial back. A dial-only
                # install declares loopback, which is the honest "not reachable"
                # answer rather than a private address that can only fail.
                endpoints=self.advertised_endpoints(),
            )
            if mode == "join":
                handshake.join_block = {
                    "invite_id": invite,
                    "joiner_public_key": self.identity.public_key,
                    "joiner_name": joiner_name or self.identity.name,
                }
                credential = Credential(
                    "invite",
                    handshake.epoch,
                    wire.invite_key(invite_material, network_id, invite),
                )
            else:
                key = wire.epoch_key(state.secret, network_id, record.epoch)
                credential = Credential("epoch", record.epoch, key)
            handshake.send_hello(sock)
            reader = wire.FrameReader(sock)
            handshake.read_challenge(reader, deadline)
            handshake.send_auth(sock, credential)
            handshake.read_welcome(reader, deadline)
            # ``read_welcome`` takes the phase off that very frame — the ONE place
            # it is decided (``Handshake.read_welcome``) — so re-reading it here
            # would be a second rule for one fact.
            result = handshake.establish()
            if result.phase == "pair":
                # Ownership of the socket belongs to whichever call opened it: here
                # that is this method, so the early return closes it. A caller that
                # handed in a `connected` socket relies on exactly that — a dial
                # that returns without a link must not leave the caller owning a
                # connection it was told nothing about.
                _close_quietly(sock)
                return None, "pair_phase_requires_the_ceremony"
            link = PeerLink(
                server=self,
                sock=sock,
                result=result,
                codec=handshake.codec(),
                settings=self.settings,
                reader=reader,
            )
            link.peer_addr = host
            with self._links_lock:
                self.links[result.link_id] = link
            link.start()
            # The listener's endpoints arrive in its ``welcome``; they are how this
            # device will re-open the link without being told the address again.
            self._note_peer_endpoints(record, result.peer_device_id, handshake.peer_endpoints)
            # CONTACT RE-EVALUATES MEMBERSHIP (§8.4). A link is the one moment both
            # ends are known to be up, and the member table is a distributed fact
            # the local record can hold a stale snapshot of (Q-R2-1). The stamp on
            # the link is what `refresh_membership` reads to decide what is due.
            self._pull_members(link)
            self._clear_refusal_mark(record)
            return link, "ok"
        except MeshRefusal as refusal:
            _close_quietly(sock)
            self._audit_handshake_refusal(refusal, network_id, host, mode)
            return None, refusal.code
        except (wire.LinkCryptoError, OSError, TimeoutError) as exc:
            _close_quietly(sock)
            self._note_refused_handshake(record, host, mode)
            return None, f"handshake_refused:{exc.__class__.__name__}"

    def _clear_refusal_mark(self, record: NetworkRecord) -> None:
        """A COMPLETED handshake clears a ``refused_by_peers`` mark.

        The mark says "peers refused this device"; the moment a peer accepts it,
        that sentence is no longer true and leaving it up would turn a transient
        into a permanent accusation — the same dead-instrument failure as asserting
        a state nobody re-checked (Q-R2-6). The refusal path sets it
        (:meth:`_note_refused_handshake`) and every successful handshake, in either
        direction, clears it here: one writer for each transition.
        """
        if record.stale != "refused_by_peers":
            return
        record.stale = ""
        store.save(record, self.root)

    def _note_refused_handshake(self, record: NetworkRecord, host: str, mode: str) -> None:
        """Name a peer's SILENT refusal of this device's handshake, LOCALLY.

        THE REFUSAL IS SILENT BY DESIGN AND THAT LEFT THIS DEVICE WITHOUT A CLUE. A
        listener that explains every refusal is an oracle — it would tell a stranger
        which tokens are real and which device ids exist — so every refusal closes
        the socket with no reply frame (``_audit_handshake_refusal``, §5.2). The
        cost of that trade was paid entirely on the refused side: the connection had
        been ACCEPTED (so this is not ``connect_failed:*``), the handshake was cut
        mid-way, and the only thing this device could say about the network it
        belongs to was ``handshake_failed:ConnectionError`` beside a full member list
        and ``trust: active``. An operator cannot act on a transport error, and QA
        round 3 (Q-R3-2) found exactly that on a device that had just been removed:
        its own audit held nothing at all about the removal, and its own surface
        described a healthy four-member mesh.

        So the LOCAL facts are written down and said out loud. Two of them:

        * the audit gains ``handshake_refused`` with the code
          ``peer_closed_silently`` — what was observed, in the words of what was
          observed, so a reviewer is not told a cause nobody established;
        * the network is marked ``stale: refused_by_peers`` — the design's own
          value for this state (§8.3: "its copy of the network is then marked
          ``stale: refused_by_peers`` so ``lop network ls`` says something true"),
          which is what makes every surface report the refusal instead of the
          member list it holds. ``membership_state`` turns that into the sentence a
          person reads, remedies included.

        WHAT THIS DELIBERATELY DOES NOT CLAIM: which of the three states that look
        like this it is. A removed device, a network marked untrusted after a panic,
        and a member whose epoch is behind all refuse this way, and the wire cannot
        tell them apart — that is the same silence, one level up. The sentence names
        all three and the remedies name the two that have one; claiming the removal
        would be the dead-instrument failure this repository has a section about.

        It is cleared on the next successful handshake (:meth:`dial`), so a peer that
        restarted mid-handshake does not leave a permanent mark.
        """
        self.audit.record(
            AuditEvent(
                event="handshake_refused",
                actor="unknown",
                subject=host,
                outcome="refused",
                network_id=record.network_id,
                cause="auth_failed",
                detail={
                    "cause": "peer_closed_silently",
                    "their_device": host,
                    "mode": mode or "member",
                },
            )
        )
        if record.stale != "refused_by_peers":
            record.stale = "refused_by_peers"
            store.save(record, self.root)

    def _rehandshake_network(self, network_id: str, *, reason: str) -> None:
        """Close and redial every live link at the new epoch.

        v1 closes and redials rather than rekeying in place: "one way to change
        keys" is worth more than the round trip, and an in-place rekey is a second
        protocol with its own bugs.
        """
        for link in list(self.links.values()):
            if link.network_id == network_id:
                link.send({"op": "net_bye", "reason": reason})
                link.close("epoch_stale")

    # -- the loopback control surface ---------------------------------------

    def _control_loop(self) -> None:
        assert self._control is not None
        while not self._stop.is_set():
            try:
                sock, _addr = self._control.accept()
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(
                target=self._control_connection, args=(sock,), name="mesh-control-conn", daemon=True
            ).start()

    def _control_connection(self, sock: socket.socket) -> None:
        """One local client: authenticate, then serve ops — or become a stream.

        The auth frame is compared with ``hmac.compare_digest`` against the key in
        the 0600 peers record, and anything else closes WITHOUT a reply — the same
        reasoning the session runtime's control socket states: an open port that
        answers wrong keys with errors is an oracle.

        A SUCCESSFUL ``stream_open`` CHANGES WHAT THIS CONNECTION ACCEPTS
        (transport §2.5, R-IF-1): from that point every line the viewer writes is a
        session frame forwarded verbatim and every frame the peer pushes is
        written back unmodified. The viewer's ack is written BEFORE the stream is
        marked ready, so the ordering the client depends on (response to
        ``stream_open`` first, then the owner's welcome) cannot be inverted by a
        push that arrived while the peer was still dialling.

        ONE READER FOR THE WHOLE CONNECTION, deliberately: the control ops and the
        session frames share a socket, and a second reader would lose whatever the
        first had already buffered. The session-frame bound is therefore the
        bound for both, which is the safe direction for an authenticated local
        client whose frames are otherwise capped at a welcome projection's size —
        AND IT IS THE BOUND THE CLIENT HALF NOW READS UNDER TOO
        (``control_request``, QA round 8 / Q-R8-1): the reply this loop writes is
        read under this same ``MAX_SESSION_FRAME_BYTES``, so the two halves of one
        socket cannot disagree about "too big" in either direction.
        """
        reader = session_dial.LineReader(sock)
        stream: _Stream | None = None
        try:
            first = reader.read_frame(10.0)
            if first is None or not hmac.compare_digest(
                str(first.get("key") or ""), self._control_key
            ):
                return
            while not self._stop.is_set():
                frame = reader.read_frame(3600.0)
                if frame is None:
                    if reader.eof:
                        break
                    continue
                if stream is not None:
                    self._forward_stream_frame(stream, frame)
                    if stream.closed:
                        break
                    continue
                op = str(frame.get("op") or "")
                if op == "stream_open":
                    reply, opened = self._open_viewer_stream(frame)
                    sock.sendall(wire.encode_line(reply))
                    if opened is None:
                        continue
                    stream = opened
                    stream.viewer_sock = sock
                    stream.viewer = reader
                    stream.flush()
                    continue
                reply = self.control_dispatch(op, frame)
                sock.sendall(wire.encode_line(reply))
        except (OSError, ConnectionError, TimeoutError):
            return
        finally:
            if stream is not None:
                # The viewer went away (a TUI quitting, a laptop closing). This
                # closes OUR stream and the relay's own dial on the far side —
                # never the runtime, which is the whole quit-safety guarantee.
                self._close_stream(stream.stream_id)
            _close_quietly(sock)

    def control_dispatch(self, op: str, frame: dict[str, Any]) -> dict[str, Any]:
        """The relay's LOCAL op vocabulary (§2.5). Not reachable over a peer link.

        Distinct from the peer ops on purpose: a reader can tell from the frame
        alone which boundary it crossed, which is what stops a local op being
        mistaken for something a peer may ask for.
        """
        req = frame.get("req")
        try:
            handler = self._control_handlers().get(op)
            if handler is None:
                # OPERATOR LANGUAGE, NOT A DEVELOPER STRING. This path is what a
                # viewer, the TUI or the agent tool reaches when it asks for
                # something this build does not serve — the session plane, until
                # that slice lands — and `unknown local op 'net_forward_session'`
                # told the reader nothing about what to do next. So the answer
                # names the capability the op needs and the document that owns it,
                # and says plainly that nothing changed (a refusal that leaves the
                # caller guessing whether it half-ran is a worse refusal).
                owner = _owning_document(op)
                planned = owner != "the design documents"
                return {
                    "op": "error",
                    "req": req,
                    "code": "not_implemented" if planned else "unknown_local_op",
                    "message": (
                        f"{op} is not in this build: {owner} owns that slice. Nothing "
                        "was changed and no session was touched. `lop network doctor` "
                        "reports what this build does serve."
                        if planned
                        else (
                            f"this relay does not know the action {op!r}. Nothing was "
                            "changed. Check the spelling, or run `lop network doctor` "
                            "for what this build serves."
                        )
                    ),
                }
            return {"op": "ack", "req": req, "detail": handler(frame)}
        except MeshRefusal as refusal:
            # THE CODE CROSSES THIS BOUNDARY TOO. A --json consumer branches on it
            # and the sentence is for the person; the same two-parts discipline
            # MeshRefusal states, kept all the way out to the CLI.
            return {
                "op": "error",
                "req": req,
                "code": refusal.code,
                "message": refusal.sentence,
            }

    def _control_handlers(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        return {
            "net_status": lambda frame: self.status(),
            "net_ls": self._ctl_ls,
            "net_show": lambda frame: self.network_detail(str(frame.get("network") or "")),
            "net_peer_ls": lambda frame: self.peer_status(),
            "net_invite": self._ctl_invite,
            "net_member_rm": self._ctl_member_rm,
            "net_trust_local": self._ctl_trust,
            "net_panic_local": self._ctl_panic,
            "net_pair_pending": self._ctl_pair_pending,
            "net_pair_confirm": self._ctl_pair_confirm,
            "net_disconnect": self._ctl_disconnect,
            "net_log": lambda frame: self.audit.tail(
                int(frame.get("limit") or 50),
                network_id=str(frame.get("network") or "") or None,
                since=frame.get("since"),
            ),
            "net_doctor": lambda frame: self.doctor(peer=str(frame.get("peer") or "")),
            # THE SESSION PLANE'S LOCAL VOCABULARY. `stream_open` is the transport's
            # §2.5 name and puts this connection into pass-through mode (handled in
            # _control_connection, because a mode change is not a value); the five
            # `peer_*` names are this slice's, and they are deliberately not
            # `net_*`: a name in LOCAL_OPS may never appear in OP_CAPABILITY
            # (authorizer.op_tables_are_total), and a viewer asking ITS OWN relay
            # to ask a peer is a local act with a local name.
            "stream_open": lambda _frame: _not_implemented(
                "stream_open", "network/control.py (handled in the control loop)"
            ),
            "stream_send": self._ctl_stream_send,
            "stream_close": self._ctl_stream_close,
            "peer_session_rows": lambda _frame: self.federated_rows(),
            "peer_session_facts": self._ctl_peer_facts,
            "peer_session_create": self._ctl_peer_create,
            "peer_session_engage": self._ctl_peer_engage,
            "peer_session_stop": self._ctl_peer_stop,
        }

    def _ctl_ls(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """``lop network ls``: every network this device is in, with its table.

        Contacts peers first, for the same reason ``network_detail`` does: "how
        many members" is a distributed fact and the local record is only a
        snapshot of it, taken when this device joined (Q-R2-1). The refresh's own
        result travels with the rows, so the count and its provenance cannot be
        separated by a consumer that only reads the number.
        """
        reports = self.contact_peers()
        return [
            self.network_summary(record, report=reports.get(record.network_id))
            for record in store.list_networks(self.root)
        ]

    def _ctl_invite(self, frame: dict[str, Any]) -> dict[str, Any]:
        record = self._require_network(str(frame.get("network") or ""))
        state = store.require_secrets(record.network_id, self.root)
        minted: MintedInvite = mint_invite(
            record,
            state.secret,
            role=str(frame.get("role") or "read"),
            ttl_s=float(frame.get("ttl_s") or 600.0),
            hosts=[str(host) for host in frame.get("hosts") or []] or None,
            device_id=str(frame.get("device_id") or ""),
        )
        record.invites.append(minted.record)
        store.save(record, self.root)
        path = store.save_invite_token(minted.record.invite_id, minted.token, self.root)
        self.audit.record(
            AuditEvent(
                event="invite_minted",
                actor=record.self_device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "role": minted.record.role,
                    "expires_at": minted.record.expires_at,
                    "bound_device": minted.record.device_id,
                },
            )
        )
        # The token is NEVER in the reply: a token in a JSON payload is a token in
        # the agent's transcript, and the transcript is replayed to the provider.
        return {
            "invite_id": minted.record.invite_id,
            "path": str(path),
            "expires_at": minted.record.expires_at,
            "expires_in_s": minted.record.ttl_s,
            "role": minted.record.role,
            "hosts": list(minted.record.hosts),
            "network_id": record.network_id,
            "network_name": record.name,
        }

    def _ctl_member_rm(self, frame: dict[str, Any]) -> dict[str, Any]:
        record = self._require_network(str(frame.get("network") or ""))
        state = store.require_secrets(record.network_id, self.root)
        device_id = str(frame.get("device_id") or "")
        outcome = remove_member(
            record, state, device_id=device_id, by=record.self_device_id, root=self.root
        )
        self.audit.record(
            AuditEvent(
                event="member_removed",
                actor=record.self_device_id,
                subject=device_id,
                network_id=record.network_id,
                epoch=outcome.epoch,
                detail={
                    "initiated_by": record.self_device_id,
                    "rekeyed": True,
                    "epoch_after": outcome.epoch,
                },
            )
        )
        self._broadcast_epoch(record, state, reason="member_removed", removed=outcome.removed)
        self._rehandshake_network(record.network_id, reason="epoch_stale")
        return {
            "network_id": record.network_id,
            "removed": device_id,
            "epoch": outcome.epoch,
            "queued": len(store.queued_frames(device_id, self.root)),
        }

    def _ctl_trust(self, frame: dict[str, Any]) -> dict[str, Any]:
        record = self._require_network(str(frame.get("network") or ""))
        before = record.trust
        set_trust(
            record,
            trust=trust_state(frame.get("trust") or "active"),
            reason=str(frame.get("reason") or "operator"),
            root=self.root,
        )
        self.audit.record(
            AuditEvent(
                event="trust_changed",
                actor=record.self_device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={"from": before, "to": record.trust, "reason": "operator"},
            )
        )
        return {"network_id": record.network_id, "trust": record.trust}

    def _ctl_pair_pending(self, _frame: dict[str, Any]) -> list[dict[str, Any]]:
        """The pairings parked for a human, oldest first. A list is the answer.

        A method rather than a lambda in the handler table: the CLI and the harness
        drive this path, and a missing method would fail as an `AttributeError`
        inside whatever is waiting for the answer — which is how a two-sided test
        ends up waiting for a timeout instead of reporting the real fault.
        """
        return [pending.to_json() for pending in store.pending_pairings(self.root)]

    def _ctl_pair_confirm(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Record the inviter's human answer to a parked pairing.

        The CLI does the ASKING (it has the terminal) and this stores the answer
        where the waiting pairing loop will find it — so the relay stays the only
        writer of pairing state and the human's answer is audited with everything
        else. ``matched`` must be true for an admission: a decision file that says
        admit without a matching comparison is a file nothing should have written.
        """
        invite_id = str(frame.get("invite_id") or "")
        pending = store.pending_pairing(invite_id, self.root) if invite_id else None
        if pending is None:
            open_pairings = store.pending_pairings(self.root)
            if not open_pairings:
                raise MeshRefusal(
                    "no_pending_pairing",
                    "no device is waiting to pair with this one right now. Run "
                    "`lop network confirm --json` to see the queue, or mint an invite "
                    "with `lop network invite`.",
                )
            pending = open_pairings[0]
        admit = str(frame.get("decision") or "admit") == "admit"
        matched = bool(frame.get("matched")) and admit
        decision = PairDecision(
            invite_id=pending.invite_id,
            decision="admit" if matched else "decline",
            matched=matched,
            reason="" if matched else str(frame.get("reason") or "declined"),
            answered_by=str(frame.get("answered_by") or "human"),
        )
        store.save_pair_decision(decision, self.root)
        return {
            "invite_id": pending.invite_id,
            "decision": decision.decision,
            "matched": decision.matched,
            "joiner_device_id": pending.joiner_device_id,
        }

    def _ctl_panic(self, frame: dict[str, Any]) -> dict[str, Any]:
        record = self._require_network(str(frame.get("network") or ""))
        state = store.require_secrets(record.network_id, self.root)
        member = record.self_member()
        is_admin = bool(member and "admin" in member.capabilities)
        outbound = panic(
            record,
            state,
            by=record.self_device_id,
            is_admin=is_admin,
            reason=str(frame.get("reason") or "operator_panic"),
            root=self.root,
        )
        self.audit.record(
            AuditEvent(
                event="panic_raised",
                actor=record.self_device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={
                    "epoch_before": record.epoch - (1 if is_admin else 0),
                    "epoch_after": record.epoch,
                    "reachable_peers": len(self.links),
                },
            )
        )
        # PANIC KEEPS ITS BROADCAST BEHAVIOUR: the frame carries the new secret to
        # every reachable peer.
        delivered = 0
        for link in list(self.links.values()):
            if link.network_id == record.network_id and link.send(dict(outbound)):
                delivered += 1
        for link in list(self.links.values()):
            if link.network_id == record.network_id:
                link.close("we-closed")
        set_trust(
            record,
            trust="untrusted",
            reason="this device raised a panic",
            root=self.root,
        )
        return {
            "network_id": record.network_id,
            "epoch": record.epoch,
            "rotated": is_admin,
            "broadcast_to": delivered,
        }

    def _ctl_disconnect(self, frame: dict[str, Any]) -> dict[str, Any]:
        """This device leaves: say goodbye, stop trusting, keep the audit trail.

        The local secret is DELETED — the record and its membership stay, so the
        operator can see what happened and re-pair deliberately. Leaving is not a
        reason to trust the device less; it is a reason to stop it being able to
        read new traffic, which the peers' rotation handles.
        """
        record = self._require_network(str(frame.get("network") or ""))
        reachable = 0
        for link in list(self.links.values()):
            if link.network_id != record.network_id:
                continue
            if link.send(
                {"op": "net_leave", "network_id": record.network_id, "locality": "remote"}
            ):
                reachable += 1
        time.sleep(0.05)
        for link in list(self.links.values()):
            if link.network_id == record.network_id:
                link.close("we-closed")
        secrets_file = store.secrets_path(record.network_id, self.root)
        if secrets_file.exists():
            secrets_file.unlink()
        set_trust(
            record,
            trust="disconnected",
            reason="this device disconnected",
            root=self.root,
        )
        self.audit.record(
            AuditEvent(
                event="disconnect_initiated",
                actor=record.self_device_id,
                subject=record.network_id,
                network_id=record.network_id,
                epoch=record.epoch,
                detail={"epoch": record.epoch, "reachable_peers": reachable},
            )
        )
        return {
            "network_id": record.network_id,
            "reachable_peers": reachable,
            "secret_deleted": True,
        }

    # -- the local session-plane ops (the viewer's CLI drives these) ----------
    #
    # ONE TRANSLATION, IN ONE PLACE. Each of these asks a peer to run ITS OWN
    # implementation of the same op (`net_session_create`/`_engage`/`_stop`). None
    # of them re-implements anything: the whole point of routing rather than
    # replicating (mobility §8.1) is that the guards, the ladder and the sentences
    # stay on the device that owns the disk.

    def _local_peer_call(
        self, op: str, peer: str, *, timeout: float | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Run one peer op on ``peer`` and return its detail.

        A refusal from the peer is turned into a MeshRefusal carrying the PEER'S
        sentence verbatim — never a sentence re-derived here. The peer is the only
        party that can see which guard or which ladder rung fired, and a second
        sentence table on this side is exactly how two devices come to disagree
        about the remedy (§8.2).

        ``timeout`` is per-op and defaults to ``max(op_wait_s, ENGAGE_DEADLINE_S)``
        — a spawn's budget. The one op that must outlast its own act on the far
        side passes its own (``_ctl_peer_stop``'s forced mode: see
        :func:`forced_stop_deadline_s`), because a hop that gives up first does not
        report a slow stop, it reports NOTHING — and the guide reads that answer as
        "the stop did not act".
        """
        link = self._ensure_link(self._resolve_peer(peer))
        if link is None:
            raise MeshRefusal(
                "peer_unreachable",
                f"{peer} cannot be reached from this device right now, so it was not asked",
            )
        reply = link.request(
            {"op": op, "req": self._next_relay_req(), "locality": "remote", **fields},
            timeout=(
                max(self.settings.op_wait_s, ENGAGE_DEADLINE_S) if timeout is None else timeout
            ),
        )
        if reply is None:
            raise MeshRefusal(
                "peer_unreachable",
                f"{peer} stopped answering before it replied",
            )
        if reply.get("op") == "error":
            raise MeshRefusal(
                str(reply.get("code") or "peer_refused"),
                str(reply.get("message") or "that device refused"),
            )
        detail = reply.get("detail")
        return detail if isinstance(detail, dict) else {"value": detail}

    def _ctl_peer_create(self, frame: dict[str, Any]) -> dict[str, Any]:
        return self._local_peer_call(
            "net_session_create",
            str(frame.get("peer") or ""),
            cwd=str(frame.get("cwd") or ""),
            model=frame.get("model"),
            name=str(frame.get("name") or ""),
            prompt=str(frame.get("prompt") or ""),
            images=list(frame.get("images") or []),
            origin=str(frame.get("origin") or "user"),
        )

    def _ctl_peer_engage(self, frame: dict[str, Any]) -> dict[str, Any]:
        return self._local_peer_call(
            "net_session_engage",
            str(frame.get("peer") or ""),
            session_id=str(frame.get("session_id") or ""),
            cwd=str(frame.get("cwd") or ""),
            warm=frame.get("warm") if isinstance(frame.get("warm"), dict) else {},
        )

    def _ctl_peer_stop(self, frame: dict[str, Any]) -> dict[str, Any]:
        mode = str(frame.get("mode") or "graceful")
        return self._local_peer_call(
            "net_session_stop",
            str(frame.get("peer") or ""),
            session_id=str(frame.get("session_id") or ""),
            mode=mode,
            # A FORCED STOP IS THE ONE HOP WHOSE ANSWER IS WORTH MINUTES (Q-R7-2).
            # ``immediate`` is the ladder's own opt-in, and the only shape that
            # takes it to the SIGTERM rung is a target whose socket is silent — the
            # exact case the flag exists for — so this is the hop that must outlast
            # the rung instead of reporting a refusal while the owner is still
            # working. Every other mode answers quickly (skip/refuse) and keeps the
            # default budget.
            timeout=forced_stop_deadline_s() if mode == "immediate" else None,
        )

    def _ctl_peer_facts(self, frame: dict[str, Any]) -> dict[str, Any]:
        """What the peer knows about one of its sessions, for a resolver (§3.4).

        ``published`` is the third registry state: a peer may report a claim with
        no usable record, and `_bind_under_lock` branches on that difference —
        so it is reported rather than collapsed into "not running".
        """
        peer = str(frame.get("peer") or "")
        session_id = str(frame.get("session_id") or "")
        try:
            # The resolver takes an id OR a name, like every other entry point a
            # person drives; an unresolvable target is not an error here because
            # this op's answer is ``owned: false`` either way.
            peer = self._resolve_peer(peer) if peer else ""
        except MeshRefusal:
            pass
        _peers, sessions = self._fan_out_catalog()
        row = next(
            (
                item
                for item in sessions
                if item.get("session_id") == session_id
                and (not peer or (item.get("peer") or {}).get("device_id") == peer)
            ),
            None,
        )
        if row is None:
            return {"owned": False, "published": False, "pid": None, "record": None}
        pid = int(row.get("pid") or 0)
        return {
            "owned": True,
            "published": bool(pid) and row.get("state") not in ("stored", ""),
            "pid": pid or None,
            "record": row,
        }

    def _ctl_stream_send(self, frame: dict[str, Any]) -> dict[str, Any]:
        """The multiplexed form: one frame on one opened stream (§2.5).

        Kept because the transport specifies it and because it is what a test can
        drive frame-at-a-time without holding a pass-through connection. The
        viewer path uses the pass-through mode instead.
        """
        stream_id = str(frame.get("stream") or "")
        stream = self._streams.get(stream_id)
        if stream is None:
            raise MeshRefusal("unknown_stream", "that stream is not open on this device")
        inner = frame.get("frame")
        if not isinstance(inner, dict):
            raise MeshRefusal("protocol_error", "a stream frame must carry the frame it forwards")
        self._forward_stream_frame(stream, inner)
        return {"stream": stream_id, "delivered": not stream.closed}

    def _ctl_stream_close(self, frame: dict[str, Any]) -> dict[str, Any]:
        stream_id = str(frame.get("stream") or "")
        existed = stream_id in self._streams
        self._close_stream(stream_id)
        return {"stream": stream_id, "closed": existed}

    def _fan_out_catalog(self) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Ask every peer for its rows, once. Returns ``(peers, sessions)``.

        A LIVE read rather than the transport's cached fan-out (§9.4): that cache,
        its TTL and its delta op belong to the transport slice, and a second
        cache here would be a second staleness rule for one fact. Every row this
        returns is one the peer just answered — strictly fresher than a cached
        row, and never a claim the peer did not make.

        A peer that does not answer contributes a ``reachable: false`` peer block
        and NO rows (§8.3): a listing must not show phantom rows for a device that
        is switched off, and it must still say the device exists.
        """
        peers: dict[str, dict[str, Any]] = {}
        sessions: list[dict[str, Any]] = []
        deadline = time.monotonic() + LISTING_PROBE_BUDGET_S
        for record in store.list_networks(self.root):
            for member in record.active_members():
                if member.device_id == self.identity.device_id:
                    continue
                # ``_ensure_link`` and NOT a bare lookup: a listing that only
                # read the links it happened to hold would report a peer we can
                # reach as unreachable, which is the one thing the reachability
                # field must never say. It dials the member's own recorded
                # endpoints, so a dial-only install lists like any other.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    link, unreachable_reason = None, NOT_ATTEMPTED_REASON
                else:
                    link, unreachable_reason = self._ensure_link_with_reason(
                        member.device_id, probe_timeout_s=remaining
                    )
                block = {
                    "device_id": member.device_id,
                    "name": member.name,
                    "network_id": record.network_id,
                    "reachable": link is not None,
                    "age_s": 0.0,
                    "endpoints": list(member.endpoints),
                    # WHY, not just that it is false: "no_endpoint" and
                    # "connect_failed:ConnectionRefusedError" are different
                    # operator actions, and an empty list of rows must not be
                    # indistinguishable from an unreachable device.
                    "reason": unreachable_reason,
                }
                if link is None:
                    peers.setdefault(member.device_id, block)
                    continue
                reply = link.request(
                    {"op": "net_catalog", "req": self._next_relay_req(), "locality": "remote"},
                    timeout=max(0.5, min(self.settings.op_wait_s, deadline - time.monotonic())),
                )
                if reply is None:
                    block["reachable"] = False
                    block["reason"] = "asked, and it did not answer"
                    peers.setdefault(member.device_id, block)
                    continue
                if reply.get("op") != "ack":
                    # THE PEER'S OWN SENTENCE, not a shrug. A refusal frame carries a
                    # message (the authoriser's refusal, or a handler failure), and
                    # reporting "it did not answer" while holding a reason is how a
                    # listing withholds the one fact the operator needs (QA round 1).
                    block["reachable"] = False
                    block["reason"] = str(reply.get("message") or "the peer refused the listing")[
                        :200
                    ]
                    peers.setdefault(member.device_id, block)
                    continue
                peers[member.device_id] = block
                raw_detail = reply.get("detail")
                detail: dict[str, Any] = raw_detail if isinstance(raw_detail, dict) else {}
                for row in detail.get("sessions") or ():
                    if not isinstance(row, dict):
                        continue
                    item = dict(row)
                    item["locality"] = "remote"
                    item["peer"] = block
                    sessions.append(item)
        return peers, sessions

    def federated_rows(self) -> dict[str, Any]:
        """The federated listing: this device's rows plus every peer's (R6, §9).

        Each row carries ``locality`` and, for a remote one, the peer block — the
        two fields §9.2 pins. The MERGE is by construction here rather than by a
        client comparison: a row is filed under the device that answered for it,
        so no surface has to infer remoteness from an id's shape.

        THE PEER SET IS REFRESHED FIRST, because a merge over a stale table
        silently omits a device and its sessions: `--all-peers` returning three of
        four peers as though that were the network is worse than an error, since an
        operator acts on it (Q-R2-1). The refresh is also what the ``membership``
        block reports, so the answer says how many peers agreed with the set.
        """
        rows: list[dict[str, Any]] = []
        for row in self.local_session_rows():
            item = dict(row)
            item["locality"] = "local"
            item["peer"] = None
            rows.append(item)
        # REFRESH OVER THE LINKS THAT EXIST, then let the fan-out dial (`_fan_out_catalog`
        # already probes every member it knows). One dial budget per listing: a listing
        # that both refreshed by dialling AND fanned out by dialling would pay the
        # probe budget twice for one answer.
        reports = self.refresh_membership()
        peers, sessions = self._fan_out_catalog()
        rows.extend(sessions)
        return {
            "ok": True,
            "sessions": rows,
            "peers": peers,
            "membership": {
                network_id: report.to_json() for network_id, report in sorted(reports.items())
            },
            "device_id": self.identity.device_id,
            "device_name": self.identity.name,
        }

    def _require_network(self, target: str) -> NetworkRecord:
        records = store.list_networks(self.root)
        if not target:
            if len(records) == 1:
                return records[0]
            raise MeshRefusal(
                "ambiguous_network",
                "name a network: this device is in "
                + (", ".join(record.name for record in records) or "none"),
            )
        matches = store.match_networks(records, target)
        if not matches:
            raise MeshRefusal(
                "unknown_network", f"this device is not in a network called {target!r}"
            )
        if len(matches) > 1:
            raise MeshRefusal(
                "ambiguous_network",
                f"{target!r} matches {len(matches)} networks; use the network id",
            )
        return matches[0]

    # -- reporting ----------------------------------------------------------

    def network_summary(
        self, record: NetworkRecord, *, report: MembershipReport | None = None
    ) -> dict[str, Any]:
        """One network's row, with what the member count RESTS ON.

        ``report`` is the refresh that just ran (:meth:`contact_peers`); when a
        caller has one it is attached, so no surface has to say "3 member(s)" on
        its own authority. ``membership_state`` is this DEVICE's standing, which is
        a different fact from any peer's reachability and is the one a removed
        device's operator needs (Q-R3-2).
        """
        links = [
            link
            for link in self.links.values()
            if link.network_id == record.network_id and link.alive
        ]
        state = membership_state(record)
        if report is not None:
            state = {**state, "table": report.to_json()}
            state["sentence"] = f"{state['sentence']}; {report.sentence()}"
        else:
            # NO REFRESH RAN, so no peer was asked, and the row says exactly that
            # instead of leaving a bare count to be read as authoritative — the
            # failure QA named separately from the convergence bug itself.
            state = {
                **state,
                "table": {
                    "complete": False,
                    "answered": [],
                    "not_answered": [],
                    "oldest_answer_age_s": None,
                    "learned": [],
                    "sentence": (
                        "members NOT verified: this row is the local table and no peer "
                        "was asked for its own"
                    ),
                },
            }
        return {
            "network_id": record.network_id,
            "name": record.name,
            "epoch": record.epoch,
            "role": record.self_role,
            "capabilities": list(record.self_capabilities),
            "trust": record.trust,
            "members": len(record.active_members()),
            "links": len(links),
            "stale": record.stale,
            "self_device_id": record.self_device_id,
            "membership_state": state["state"],
            "membership": state,
        }

    def network_detail(self, target: str) -> dict[str, Any]:
        record = self._require_network(target)
        # MEMBERSHIP IS A DISTRIBUTED FACT, so a report about it contacts the other
        # members first (Q-R2-1). This command is where the stale snapshot was
        # measured, and answering it from the local record alone is what made a
        # third member invisible to a device that had joined earlier. The refresh
        # RE-PULLS over links that already exist (`refresh_membership`), which is
        # the half round 2's fix was missing: a healthy pair never establishes its
        # link twice, so "on the next contact" never arrived.
        reports = self.contact_peers()
        base = self.network_summary(record, report=reports.get(record.network_id))
        # ONE DIGEST PER TABLE, and the SAME one the wire carries: `net_member_list`
        # and the admission frame both send `members_digest`, so two devices
        # comparing their `show` output can tell "same members" from "same count".
        base["members_digest"] = members_digest_of(record)
        base["members_detail"] = [
            {
                "device_id": member.device_id,
                "name": member.name,
                "kind": member.kind,
                "lifecycle": member.lifecycle,
                "role": member.role,
                "capabilities": list(member.capabilities),
                "endpoints": list(member.endpoints),
                "added_via": member.added_via,
                "active": member.active,
                "suspect": member.suspect,
                "duplicate_count": member.duplicate_count,
                "last_seen_instance": member.last_seen_instance,
            }
            for member in record.members
        ]
        base["rotations"] = dict(record.rotations)
        base["invites"] = [
            {
                "invite_id": invite.invite_id,
                "state": invite.state,
                "role": invite.role,
                "bound_device": invite.device_id,
                "expires_at": invite.expires_at,
            }
            for invite in record.invites
        ]
        base["audit_tail"] = self.audit.tail(5, network_id=record.network_id)
        return base

    def peer_status(self) -> list[dict[str, Any]]:
        """The peer table: every other member, and whether we can reach it NOW.

        A LIVE PROBE, not a link lookup. ``reachable`` answers "can I reach this
        device?", and a listing that read only the links it happened to hold
        would report a peer it can dial as unreachable — which is exactly the
        false negative that made every paired device look dead (QA round 1,
        F-2). The probe is bounded by ``LISTING_PROBE_BUDGET_S`` so the table
        still returns when a member's address is a black hole, and a peer that
        could not be probed says so in ``reason``.
        """
        peers: list[dict[str, Any]] = []
        deadline = time.monotonic() + LISTING_PROBE_BUDGET_S
        for record in store.list_networks(self.root):
            for member in record.active_members():
                if member.device_id == record.self_device_id:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    link, reason = None, NOT_ATTEMPTED_REASON
                else:
                    link, reason = self._ensure_link_with_reason(
                        member.device_id, probe_timeout_s=remaining
                    )
                peers.append(
                    {
                        "device_id": member.device_id,
                        "name": member.name,
                        "network_id": record.network_id,
                        "reachable": link is not None,
                        "reason": reason,
                        "endpoints": list(member.endpoints),
                        "last_seen_at": member.last_seen_at,
                        "suspect": member.suspect,
                    }
                )
        return peers

    def status(self) -> dict[str, Any]:
        """Install state, health, links, log paths — what ``lop network status`` prints."""
        record = self.peer_record()
        return {
            "pid": os.getpid(),
            "device_id": self.identity.device_id,
            "device_name": self.identity.name,
            "instance_id": self.instance_id,
            "listen": record.listen,
            "control_port": self._control_port,
            "networks": [self.network_summary(row) for row in store.list_networks(self.root)],
            "links": [
                {
                    "link_id": link.link_id,
                    "device_id": link.device_id,
                    "network_id": link.network_id,
                    "epoch": link.epoch,
                    "phase": link.phase,
                    "frames_in": link.frames_in,
                    "frames_out": link.frames_out,
                }
                for link in self.links.values()
            ],
            "audit_degraded": self.audit.degraded,
            "audit_degraded_reason": self.audit.degraded_reason,
            "audit_path": str(store.audit_path(self.root)),
            "log_path": str(log_path()),
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def doctor(
        self, *, peer: str = "", budget_s: float | None = LISTING_PROBE_BUDGET_S
    ) -> dict[str, Any]:
        """Diagnose a link: reachability, handshake, epoch skew, clock skew.

        It never claims reachability it has not just proven: every endpoint it
        reports on was dialled in this call, and a failure names which failure it
        was (``connect_timeout``, ``connection_refused``, a refusal code). Clock
        skew is REPORTED and never enforced — nothing in the design compares clocks
        across hosts, and a diagnostic that refused on skew would reintroduce the
        dependency the design removed.

        A PROBE DIALS, SO IT IS BOUNDED. ``budget_s`` is the whole run's budget and
        every dial inside it is capped (``PROBE_CONNECT_TIMEOUT_S``), because a
        diagnostic that can outlast its caller is a diagnostic whose answer is
        never read: the CLI gives the control call its own timeout, and a doctor
        whose reply arrives after it is a doctor whose output the operator never
        sees — which is exactly how this command came to report a relay state it
        had not checked (QA round 2, Q-R2-6). An endpoint the budget did not reach
        is reported as ``not_attempted``, never as unreachable.
        """
        findings: list[dict[str, Any]] = []
        deadline = None if budget_s is None else time.monotonic() + budget_s
        identity = store.network_root(self.root)
        if self.identity is None or not self.identity.device_id:
            findings.append({"check": "identity", "ok": False, "detail": "identity_missing"})
        for record in store.list_networks(self.root):
            if record.stale:
                findings.append(
                    {
                        "check": "network",
                        "network_id": record.network_id,
                        "ok": False,
                        "detail": record.stale,
                    }
                )
            # THIS DEVICE'S OWN STANDING, in words and with a code, because
            # ``refused_by_peers`` alone was the whole of what a removed device
            # could find: its own surface had no check that named the removal, so
            # `doctor` reported `unhealthy` about a per-endpoint list and left the
            # operator to guess (Q-R3-2). Absent on a healthy device — a check that
            # can only ever pass is noise, and the point of a diagnostic is that its
            # rows mean something.
            standing = membership_state(record)
            if standing["state"] != "active":
                findings.append(
                    {
                        "check": "membership",
                        "network_id": record.network_id,
                        "ok": False,
                        "code": standing["state"],
                        "detail": standing["sentence"],
                        "remedies": standing["remedies"],
                    }
                )
            for member in record.active_members():
                if member.device_id == record.self_device_id:
                    continue
                if peer and member.device_id != peer:
                    continue
                findings.extend(self._probe_member(record, member, deadline=deadline))
        return {
            "checks": findings,
            "identity_dir": str(identity),
            "listen": self.peer_record().listen,
            # THIS RELAY IS RUNNING: it is the process answering this call, so this
            # is a fact it holds rather than a claim about another machine. The
            # local fallback reports the same key with the same three states, and
            # it used to hardcode "not running" next to a machine whose relay was
            # demonstrably up (QA round 2, Q-R2-6).
            "relay": f"running, pid {os.getpid()}",
            "epochs": {row.network_id: row.epoch for row in store.list_networks(self.root)},
        }

    def _probe_member(
        self, record: NetworkRecord, member: MemberRecord, *, deadline: float | None
    ) -> list[dict[str, Any]]:
        """Every check row for ONE member: one per declared address, then the handshake.

        ALL ADDRESSES AT ONCE, and the handshake on whichever answered first — the
        same rule the listings follow, for the same reason (Q-R2-2): a member whose
        row leads with a black hole must be diagnosed, not written off, and the
        diagnostic owes that twice over because its whole purpose is to be the
        instrument that tells the difference. A SECOND address that also answers is
        reported as reachable — it connected, which is a fact about it — without a
        second handshake: two links to one device is not a diagnosis, it is a leak.
        """
        if not member.endpoints:
            return [
                {
                    "check": "reachability",
                    "device_id": member.device_id,
                    "endpoint": "",
                    "ok": False,
                    "detail": "no_endpoint",
                }
            ]
        probe = probe_candidates(
            member.endpoints,
            deadline=deadline,
            connect_cap=PROBE_CONNECT_TIMEOUT_S,
            wait_all=True,
        )
        winner_row: dict[str, Any] | None = None
        if probe.sock is not None:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                _close_quietly(probe.sock)
                winner_row = self._handshake_row(
                    member,
                    probe.winner,
                    ok=False,
                    detail=(
                        f"not_attempted: {probe.winner} answered and the doctor budget "
                        "ran out before the handshake"
                    ),
                )
            else:
                winner_row = self._handshake(record, member, probe.winner, probe.sock, remaining)
        rows: list[dict[str, Any]] = []
        for attempt in probe.attempts:
            if attempt.endpoint == probe.winner and winner_row is not None:
                rows.append(winner_row)
            elif attempt.connected:
                rows.append(
                    {
                        "check": "reachability",
                        "device_id": member.device_id,
                        "endpoint": attempt.endpoint,
                        "ok": True,
                        "latency_ms": attempt.latency_ms,
                        "detail": f"connected; the link was established at {probe.winner}",
                    }
                )
            else:
                rows.append(
                    {
                        "check": "reachability",
                        "device_id": member.device_id,
                        "endpoint": attempt.endpoint,
                        "ok": False,
                        "latency_ms": attempt.latency_ms,
                        "detail": attempt.detail,
                    }
                )
        return rows

    def _handshake_row(
        self, member: MemberRecord, endpoint: str, *, ok: bool, detail: str, **extra: Any
    ) -> dict[str, Any]:
        """One ``check: handshake`` row, the shape this command has always emitted."""
        return {
            "check": "handshake",
            "device_id": member.device_id,
            "endpoint": endpoint,
            "ok": ok,
            "detail": detail,
            **extra,
        }

    def _handshake(
        self,
        record: NetworkRecord,
        member: MemberRecord,
        endpoint: str,
        sock: socket.socket,
        remaining: float | None,
    ) -> dict[str, Any]:
        """Run the one handshake a member's probe earns, and close the link again."""
        started = time.monotonic()
        try:
            link, reason = self.dial(
                record.network_id,
                host=endpoint,
                epoch=record.epoch,
                timeout_s=remaining,
                connected=sock,
            )
        except MeshRefusal as refusal:
            return self._handshake_row(member, endpoint, ok=False, detail=refusal.code)
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        if link is None:
            return self._handshake_row(
                member, endpoint, ok=False, detail=reason or "unreachable", latency_ms=latency_ms
            )
        skew = None
        if link.epoch != record.epoch:
            skew = link.epoch - record.epoch
        link.close("we-closed")
        return self._handshake_row(
            member, endpoint, ok=True, detail="ok", latency_ms=latency_ms, epoch_skew=skew
        )

    # -- pair phase, listener-side helpers ---------------------------------


def _has_terminal() -> bool:
    """Whether THIS process can ask a human a question directly.

    One implementation for the two places that need the answer — the pairing loop's
    inline prompt and ``uninstall --purge-identity``'s confirmation — because the
    two must never disagree about whether a human is present. `isatty` can raise on
    a closed stream, and "cannot tell" must mean "no terminal": the failure
    direction of a wrong `True` is a destructive action taken without a human.
    """
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError):
        return False


def _owning_document(op: str) -> str:
    owners = {
        "net_forward": "mesh-session-mobility.md",
        "net_sync": "mesh-session-mobility.md (R22)",
        "net_broker": "mesh-credentials.md",
        "net_session_lifecycle": "mesh-session-mobility.md",
        "net_session_move": "mesh-session-mobility.md",
        "net_forward_session": "mesh-session-mobility.md",
    }
    return owners.get(op, "the design documents")


def _not_implemented(op: str, owner: str) -> Any:
    raise MeshRefusal(
        "not_implemented",
        f"{op} is not implemented in this build; {owner} owns that slice",
    )


def _close_quietly(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _session_protocol() -> int:
    from local_operator.session.runtime.types import PROTOCOL_VERSION

    return int(PROTOCOL_VERSION)


def _build_stamp() -> dict[str, str]:
    """This build's version and source ref, for the record and the hello frame.

    Best effort and never fatal: a stamp is decoration, and a relay that refused to
    start because packaging metadata was unreadable would be a worse failure than a
    record with an empty version. Both imports are function-local because both reach
    past the stdlib, and this module is imported while ``lop network --help`` builds.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            number = version("local-operator")
        except PackageNotFoundError:
            number = ""
    except Exception:  # noqa: BLE001
        number = ""
    ref = ""
    try:
        from local_operator.update import source_ref

        ref = str(source_ref() or "")
    except Exception:  # noqa: BLE001
        ref = ""
    return {"version": number, "source_ref": ref}


def _net_summary(record: NetworkRecord | None) -> dict[str, Any]:
    if record is None:
        return {}
    return {
        "network_id": record.network_id,
        "name": record.name,
        "epoch": record.epoch,
        "sequence": record.sequence,
        "trust": record.trust,
    }


def _ttl_of(record: NetworkRecord | None, invite_id: str) -> float:
    if record is None:
        return 60.0
    invite = record.invite(invite_id)
    return invite.ttl_s if invite else 60.0


def _invite_role(record: NetworkRecord, invite_id: str) -> str:
    invite = record.invite(invite_id)
    return invite.role if invite else "read"


def _invite_capabilities(record: NetworkRecord, invite_id: str) -> set[str]:
    invite = record.invite(invite_id)
    return set(invite.capabilities) if invite else set()


def acquire_invite(record: NetworkRecord, invite_id: str) -> bool:
    invite = record.invite(invite_id)
    return invite is not None and invite.state != "consumed"


_PAIR_CAUSE: dict[str, str] = {
    "sas_mismatch": "sas_mismatch",
    "declined_local": "policy",
    "timeout": "timeout",
    "invite_already_used": "policy",
    "invite_in_use": "policy",
    "protocol_error": "auth_failed",
}


class _ReplyWaiter:
    """A one-shot reply slot keyed by (link, req)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._frame: dict[str, Any] | None = None

    def set(self, frame: dict[str, Any]) -> None:
        self._frame = frame
        self._event.set()

    def wait(self, timeout: float) -> dict[str, Any] | None:
        if not self._event.wait(timeout):
            return None
        return self._frame


# ---------------------------------------------------------------------------
# Supervision: the launchd shape `lop mobile serve` established
# ---------------------------------------------------------------------------

LABEL = "com.local-operator.network"

#: The role template rendered into argv[0]. Defined HERE rather than in
#: ``procname`` because this slice may not edit that module; it is the same shape
#: and the same brand substitution, which is what keeps the four supervised
#: daemons from collapsing into one ``Local Operator`` row in Activity Monitor.
LABEL_TEMPLATE = "{brand} [network relay] port={port}"


def log_path() -> Path:
    return log_dir() / "network.log"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def is_supported() -> bool:
    return sys.platform == "darwin" and shutil.which("launchctl") is not None


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    """The supervised-unit plan, as one pure function every consumer reads."""
    from local_operator import procname

    return {
        "Label": LABEL,
        **procname.launchd_job(
            "local_operator.network.relay",
            "--port",
            str(port),
            label=procname.branded_argv0(LABEL_TEMPLATE, port=port),
        ),
        "RunAtLoad": True,
        # Crash restarts, a deliberate refusal (exit 2: no identity, unusable
        # store) does not flap.
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        # A relay holds long-lived sockets and timers; App Nap would suspend them.
        "ProcessType": "Interactive",
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _domain() -> str:
    """``gui/<uid>`` — the launchd domain every caller feeds to ``launchctl``.

    The platform guard is INSIDE this function rather than at its call sites,
    each of which is behind ``supervisors``/``is_supported()``: ``os.getuid``
    does not exist off POSIX, so an unguarded call was an ``AttributeError``
    waiting for any arm that forgot the check — and a guard at the call site is
    invisible to a reader and to the cross-platform scan that grades this
    branch (``scripts/xplat_probe.py``), neither of which can see that the arm
    is unreachable. Guarded here, the function is safe to call on its own
    merits. The same spelling, for the same reason, as ``wakes/install.py`` and
    ``mobile/install.py``.
    """
    if sys.platform != "darwin":
        raise RuntimeError("launchd domains exist only on macOS")
    return f"gui/{os.getuid()}"


def refresh_plist_if_stale() -> Any:
    """Bring the LaunchAgent up to date, and restart it if it changed.

    Every refresh goes through ``launchd.reload_job`` rather than ``kickstart -k``
    (measured: a kickstart after a rewrite keeps running the previous argv), and
    every path is guarded by ``launchd.is_own_plist`` so a sandboxed run cannot
    restart the operator's relay.
    """
    from local_operator import launchd

    name = "network"
    try:
        if not is_supported():
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = plist_path()
        if not launchd.is_own_plist(path, LABEL):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        port = launchd.int_arg(launchd.load(path), "--port", DEFAULT_PORT)
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(port))
        if outcome.kind != "repaired":
            return outcome
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop network install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        from local_operator import launchd

        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, Any]:
    """Install-or-refresh the LaunchAgent and verify the relay answers."""
    from local_operator import launchd

    steps: list[str] = []
    if not is_supported():
        return {
            "ok": False,
            "steps": steps,
            # ``reason`` beside the sentence, like this verb's two siblings:
            # ``service_action`` answers ``no_launchd`` for the same condition, and a
            # refusal with only prose is what the CLI's own caller has to switch on
            # by string. Additive — no caller reads this key today (the CLI refuses
            # before it gets here), which is exactly why naming it now costs nothing.
            "reason": "no_launchd",
            "error": (
                "install needs macOS launchd; run `lop network serve` in the foreground "
                "elsewhere"
            ),
        }
    if not _plist_is_addressable():
        # R6: say why, in the operator's terms. The launchd guard's own wording
        # ("is not the LaunchAgent the real home owns") is correct and about
        # launchd; the person reading this is running a redirected HOME and needs
        # to know it is expected and what to do instead.
        return {
            "ok": False,
            "steps": steps,
            "reason": "isolated_home",
            "error": (
                "no LaunchAgent is available here: this run's HOME is not the home "
                "launchd supervises, so launchd cannot own a unit for it and nothing "
                "was installed. That is what an isolated or redirected HOME looks "
                "like, and it is expected. Run the relay in the foreground with "
                "`lop network serve` (or skip the start with `lop network init "
                "--no-start` / `lop network serve --no-launchd`); from a normal login, "
                "`lop network start` uses the real home's LaunchAgent."
            ),
        }
    plist_path().parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        plist_path().write_bytes(plistlib.dumps(render_plist(port)))
    steps.append(f"wrote {plist_path()}")
    if dry_run:
        steps.append("dry run: skipped load and verification")
        return {"ok": True, "steps": steps}
    reloaded = launchd.reload_job(label=LABEL, path=plist_path(), runner=_launchctl)
    if not reloaded.ok:
        return {"ok": False, "steps": steps, "error": str(reloaded.detail)[:300]}
    steps.append("loaded the LaunchAgent")
    deadline = time.time() + 20
    while time.time() < deadline:
        probe = health(timeout=1.0)
        if probe is not None:
            steps.append("the relay answered its local control socket")
            return {"ok": True, "steps": steps}
        time.sleep(0.5)
    return {
        "ok": False,
        "steps": steps,
        "error": f"the relay did not answer within 20s; see {log_path()}",
    }


def uninstall(
    *,
    purge: bool = False,
    purge_identity: bool = False,
    networks: list[str] | None = None,
    root: Path | None = None,
    dry_run: bool = False,
    assume_tty: bool | None = None,
    answer: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Remove the LaunchAgent; ``--purge`` forgets networks; ``--purge-identity``
    destroys the device keypair, and only after a human confirms by name.

    ONE FLAG, ONE BLAST RADIUS (design §6 and §12, and the invariant
    ``purge_identity_needs_a_named_tty_confirmation``):

    * ``--purge`` deletes the network records, invites, outbound queues, parked
      pairings and the audit log for the networks being uninstalled. It does NOT
      touch the device identity keypair.
    * ``--purge-identity`` is what deletes that keypair, and it requires an
      interactive terminal plus a confirmation that NAMES every network still known
      to the identity. Without a terminal it is refused outright, naming ``--purge``
      — the flag that does work — because this key is unrecoverable and is what
      every OTHER network addresses this device by.

    ``assume_tty`` and ``answer`` are the test seam: the production path reads the
    real terminal, and a test injects both to exercise the confirmation WITHOUT
    pretending a redirected process has a TTY.
    """
    steps: list[str] = []
    known = store.list_networks(root)
    targets = [record.network_id for record in known] if networks is None else list(networks)
    # A TARGET THAT MATCHES NOTHING IS A REFUSAL, NOT A NO-OP (round 1's MINOR 8,
    # review round 2's MINOR 4). ``--network n_mistyped`` used to select nothing,
    # delete nothing and answer ``ok: true`` with an empty ``deleted`` block, so an
    # operator scoping a purge by hand got a receipt that read as success for a
    # network that was never touched. Naming the ids that matched nothing is the
    # whole remedy, and they are the operator's own input.
    known_ids = {record.network_id for record in known}
    unmatched = [target for target in targets if target not in known_ids]
    if unmatched:
        raise MeshRefusal(
            "unknown_network",
            "no network here matches "
            + ", ".join(repr(target) for target in unmatched)
            + "; nothing was deleted. `lop network ls` lists the networks this device knows.",
        )
    selected = [record for record in known if record.network_id in set(targets)]

    if purge_identity and not (_has_terminal() if assume_tty is None else assume_tty):
        raise MeshRefusal(
            "purge_identity_needs_tty",
            "deleting this device's identity keypair needs a terminal you can answer at, "
            "because the key is unrecoverable and every other network addresses this "
            "device by it. `lop network uninstall --purge` removes the network records "
            "without touching the keypair; run this from a terminal to go further.",
        )

    if not dry_run:
        # THE LAUNCHCTL CALL IS GATED ON ``is_supported()`` AS WELL AS ON
        # ``_plist_is_addressable()``. ``launchd.is_own_plist`` has no platform
        # check, so on Linux the second guard alone said "yes" and
        # ``subprocess.run(["launchctl", ...])`` raised ``FileNotFoundError``
        # BEFORE any of this verb's work — so `lop network uninstall` could not
        # clean anything up on the platform the peers actually run (QA round 1,
        # F-9). `install` has always consulted the platform guard; removal is the
        # other half of the same surface and must consult it too.
        addressable = _plist_is_addressable()
        if addressable and is_supported():
            _launchctl("bootout", _domain(), str(plist_path()))
        if plist_path().exists():
            plist_path().unlink()
        if not is_supported():
            steps.append(
                "no launchd on this platform: nothing was loaded or unloaded. `lop "
                "network serve` (or `--no-start`) is how the relay runs here."
            )
        else:
            steps.append(
                "removed the LaunchAgent and its plist"
                if addressable
                else (
                    "no LaunchAgent to remove here: this run's HOME is not the one launchd "
                    "supervises, so nothing was loaded or unloaded. `--no-start` (or "
                    "`serve --no-launchd`) is how to run without launchd at all."
                )
            )
    receipt: dict[str, Any] = {
        "ok": True,
        "steps": steps,
        "networks": [f"{record.name} ({record.network_id})" for record in selected],
        "deleted": {},
        "identity": "kept",
        "dry_run": dry_run,
    }
    if purge:
        if dry_run:
            receipt["deleted"] = {
                "networks": receipt["networks"],
                "invites": sum(len(record.invites) for record in selected),
                "queues": 0,
                "pending": 0,
                "audit_files": [],
                "audit_kept": False,
                "catalog": False,
            }
        else:
            receipt["deleted"] = store.purge_network_artifacts(targets, root)
        deleted = receipt["deleted"]
        steps.append(
            f"deleted {len(deleted['networks'])} network record(s), "
            f"{deleted['invites']} invite token file(s), {deleted['queues']} outbound "
            f"queue(s), {deleted['pending']} parked pairing file(s)"
        )
        if deleted["audit_files"]:
            steps.append(f"deleted the audit log ({', '.join(deleted['audit_files'])})")
        elif deleted.get("audit_kept"):
            steps.append(
                "kept the audit log: it is one file per install and records networks "
                "this purge did not cover. `lop network log --export` copies it first."
            )
        if deleted.get("catalog"):
            steps.append("deleted the cached session catalogue")
        steps.append(
            "the device identity keypair was NOT deleted: other networks address this "
            "device by it"
        )
    if purge_identity:
        steps.extend(_purge_identity_step(selected, root=root, dry_run=dry_run, answer=answer))
        receipt["identity"] = "deleted"
    return receipt


def _purge_identity_step(
    networks: list[NetworkRecord],
    *,
    root: Path | None,
    dry_run: bool,
    answer: Callable[[str], str] | None,
) -> list[str]:
    """The named confirmation, then the delete.

    The prompt LISTS the networks still known to this identity, because that list is
    what the human is being asked to destroy the key for: a device whose keypair
    goes away stops being addressable by every one of them, and the operator cannot
    weigh that if the command only says "delete the identity?".

    LOAD, NEVER ``load_or_mint`` (round 1's MINOR 8, review round 2's MINOR 4). This
    used to MINT a keypair on a device that had none, ask the human to confirm
    deleting it, and leave that freshly minted key on disk the moment the typed id
    did not match — a command whose whole job is to destroy an identity must not be
    the thing that creates one. A device with no identity answers by name instead:
    there is nothing here to delete.
    """
    identity = load(root)
    if identity is None:
        raise MeshRefusal(
            "no_identity",
            "this device has no identity keypair, so there is nothing to delete and "
            "nothing was created: `lop network init` mints one, and `--purge` removes "
            "the network records without touching it.",
        )
    listing = (
        ", ".join(f"{record.name} ({record.network_id})" for record in networks)
        or "no networks (this identity is not in any)"
    )
    prompt = (
        f"This deletes the device identity {identity.device_id}, permanently and with no "
        f"recovery. It is how these networks address this device: {listing}.\n"
        f"Type the device id to confirm: "
    )
    reader = answer if answer is not None else input
    try:
        typed = reader(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        typed = ""
    if typed != identity.device_id:
        raise MeshRefusal(
            "purge_identity_not_confirmed",
            "the device id was not typed back, so the identity keypair was left alone. "
            "Nothing was deleted. `lop network uninstall --purge` removes the network "
            "records without touching the keypair.",
        )
    if dry_run:
        return [f"dry run: would delete the device identity {identity.device_id}"]
    deleted = store.purge_identity(root)
    return [f"deleted the device identity {identity.device_id} ({', '.join(deleted) or 'nothing'})"]


def _plist_is_addressable() -> bool:
    """Whether a LaunchAgent here would be the one the REAL home owns.

    A redirected HOME (every isolated test run, and the harness) has no
    LaunchAgent: launchd supervises the passwd home's units only. Asking first
    keeps a sandbox from writing a plist into its own home and then reporting a
    launchd detail string the operator cannot act on.
    """
    from local_operator import launchd

    return launchd.is_own_plist(plist_path(), LABEL)


def service_action(action: str) -> dict[str, Any]:
    """start|stop|restart via launchctl, bootstrapping a plist that was never loaded.

    PLATFORM FIRST, HOME SECOND. ``_plist_is_addressable`` is a HOME question and
    answers "no" on a Linux host only by accident (the real home has no plist
    there), which would print the redirected-HOME explanation to an operator
    whose actual problem is that there is no launchd at all. `install` and
    `uninstall` both consult ``is_supported()`` first; this is the third half of
    one surface and does the same.
    """
    if not is_supported():
        return {
            "ok": False,
            "reason": "no_launchd",
            "error": (
                f"`lop network {action}` drives launchd, and this platform has none. Run "
                "the relay in the foreground with `lop network serve` instead."
            ),
        }
    if not _plist_is_addressable():
        return {
            "ok": False,
            "reason": "isolated_home",
            "error": (
                f"`lop network {action}` drives launchd, and this run's HOME is not the "
                "home launchd supervises, so there is no unit to drive. Run the relay in "
                "the foreground with `lop network serve`, or run this from a normal login "
                "where the real home's LaunchAgent exists."
            ),
        }
    if action in ("start", "restart") and plist_path().exists():
        printed = _launchctl("print", f"{_domain()}/{LABEL}")
        if printed.returncode != 0:
            bootstrap = _launchctl("bootstrap", _domain(), str(plist_path()))
            if bootstrap.returncode != 0:
                return {"ok": False, "error": bootstrap.stderr.strip()[:300]}
    if action == "start":
        result = _launchctl("kickstart", f"{_domain()}/{LABEL}")
    elif action == "stop":
        result = _launchctl("kill", "SIGTERM", f"{_domain()}/{LABEL}")
    else:
        result = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    ok = result.returncode == 0
    return {"ok": ok, "error": "" if ok else result.stderr.strip()[:300]}


def health(timeout: float = 3.0) -> dict[str, Any] | None:
    """Ask the running relay for its status over the loopback control socket.

    THE ONE PLACE A NAMED CONTROL REFUSAL IS NOT RE-RAISED, and the reason is this
    probe's own contract: every caller asks a BOOLEAN (``_autostart``, the identity
    rotation's announcement, the install readiness loop, ``status``) or a boolean plus
    the relay record (``cli._relay_state``), and each of their sentences falls back to
    ``store.scan_own_relay``, which reads the process rather than the socket. A
    ``MeshRefusal`` escaping here would traceback those four callers.

    What makes that safe rather than a swallow: ``net_status`` is the SMALLEST reply
    this socket serves — this device's own document, sized by its own networks — so a
    bound or parse failure on it is a bug in this build, not a size an operator can
    reach. The ops whose replies grow with the mesh are the LISTING family, and those
    refuse by name all the way out to the operator's terminal.
    """
    record = store.find_own_relay()
    if record is None:
        return None
    try:
        reply = control_request(record, "net_status", timeout=timeout)
    except MeshRefusal:
        return None
    if reply is None:
        return None
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else None


def status(port: int = DEFAULT_PORT) -> dict[str, Any]:
    """What a human needs: is it installed, is it running, what does it see.

    ONE ANSWER PER FACT, AND NO FIELD CONTRADICTS ANOTHER. This payload used to
    derive ``relay_running`` from the CONTROL SOCKET alone while deriving
    ``record`` from the record's liveness, so a relay whose process was alive but
    whose socket did not answer this probe (SIGSTOP is the clean way to make one)
    produced ``"relay_running": false`` in a payload whose own ``record`` block
    carried ``"pid": 21094`` — a diagnostic contradicting itself in the one place
    an operator looks first (QA round 3, Q-R3-4).

    So the record AND its verdict are read together (:func:`store.scan_own_relay`,
    which reports ``live`` and ``wedged`` — a wedged relay is a RUNNING process
    whose owner has not reported recently, never proof it is dead), the answer to
    "did it reply" is its own field, and ``relay_running`` is the union:

    * ``relay_running`` — a process of ours exists (a live record, or something that
      answered). True beside a pid, and only true when there is a process;
    * ``relay_answering`` — its control socket answered THIS probe;
    * ``relay_state`` — ``live`` (alive and reporting), ``wedged`` (alive, not
      reporting), ``stopped`` (no process).

    A wedged relay therefore reads ``running: true, answering: false, state:
    "wedged"``, which is what it is: the remedy is ``kill -CONT``/a restart, not
    "start it".
    """
    record, state = store.scan_own_relay()
    live = health()
    running = live is not None or state in ("live", "wedged")
    return {
        "installed": plist_path().exists(),
        "supported": is_supported(),
        "relay_running": running,
        "relay_answering": live is not None,
        "relay_state": state if running else "stopped",
        "relay": live,
        "record": record.to_json() if record is not None else None,
        "port": port,
        "log": str(log_path()),
        "listening": live.get("listen") if live else None,
        "identity_present": (store.network_root() / "identity" / "device.json").exists(),
        # Local records when no relay is running: this command is most useful
        # precisely when the relay is DOWN, and reporting "no networks" then would be
        # the status tool lying at the moment it is asked the only question that
        # matters. With a relay up, the relay's own view wins (it includes links).
        "networks": (
            live.get("networks")
            if live
            else [
                {
                    "network_id": row.network_id,
                    "name": row.name,
                    "epoch": row.epoch,
                    "role": row.self_role,
                    "trust": row.trust,
                    "members": len(row.active_members()),
                    "links": 0,
                    "stale": row.stale,
                }
                for row in store.list_networks()
            ]
        ),
    }


def control_request(
    record: PeerRecord, op: str, *, timeout: float = 5.0, **fields: Any
) -> dict[str, Any] | None:
    """Dial the relay's loopback control socket and run one op.

    The client half of §2.5. A CLI, a TUI or the desktop daemon all use this rather
    than opening peer links themselves, which is what keeps ONE place that speaks
    the peer protocol and one place that holds a control key.

    THE REPLY IS READ UNDER THE CONTROL SOCKET'S OWN BOUND, NOT THE HANDSHAKE'S
    (QA round 8, Q-R8-1). ``wire.MAX_HANDSHAKE_LINE`` is 16 KiB because a handshake
    frame is PRE-AUTH and attacker-controlled, and that tight bound is the property
    this package wants there. A control reply is neither: the socket is loopback-only
    and key-authenticated (the key lives in the 0600 peers record), and the reply
    grows with the thing it describes — the federated catalogue is one row per
    session on every device — so ``peer_session_rows`` legitimately passed 16 KiB at
    ~17 sessions. The reader refused the frame, this function returned ``None``, and
    every caller read that ``None`` as "this device's relay did not answer" — a
    sentence about a WEDGED RELAY for a relay that answered in milliseconds.

    So the reply is read with ``dial.LineReader`` under
    ``dial.MAX_SESSION_FRAME_BYTES`` — the reader and the number the RELAY half
    already frames this same socket with — which is what makes the two halves of one
    socket unable to disagree about "too big". The wire format is unchanged (JSON
    lines, one reply per request), deliberately: a ``lop`` CLI and a relay of
    adjacent builds talk over this socket during an update, and only the READER's
    bound changed, so an old relay and a new client keep working.

    A REPLY THAT CANNOT BE READ REFUSES BY NAME, and is never a ``None``: ``None``
    is reserved for "no relay answered me" (a refused connect, a closed socket, a
    deadline), which is the condition the callers' own sentences are about. An
    over-bound reply raises ``frame_too_large`` carrying this socket's bound and a
    lower-bound byte count; a reply that is not a JSON object raises
    ``frame_unreadable``. Both reach the operator with the relay named as the
    component that answered, not as a component that stayed silent.
    """
    try:
        sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=timeout)
    except OSError:
        return None
    try:
        # The bound is read here rather than left to the reader's default: this call
        # CHOOSES the control socket's bound, and ``dial.MAX_SESSION_FRAME_BYTES`` is
        # its one home.
        reader = session_dial.LineReader(
            sock, session_dial.MAX_SESSION_FRAME_BYTES, report_bad_frames=True
        )
        sock.sendall(wire.encode_line({"key": record.control_key, "client": "cli"}))
        sock.sendall(wire.encode_line({"op": op, "req": 1, **fields}))
        return reader.read_frame(timeout)
    except session_dial.FrameTooLarge as exc:
        raise MeshRefusal(
            "frame_too_large",
            f"this device's relay answered `{op}` with a line of at least {exc.size} bytes, "
            f"over the {exc.limit}-byte bound on one control reply, so the answer was "
            "refused rather than truncated; the relay itself is answering — nothing was "
            "listed and nothing was changed.",
        ) from exc
    except session_dial.FrameUnreadable as exc:
        raise MeshRefusal(
            "frame_unreadable",
            f"this device's relay answered `{op}` with a line this build cannot read "
            f"({exc.reason}), so whether the op ran is unknown; the relay answered — it is "
            "not a relay that stayed silent.",
        ) from exc
    except (OSError, ConnectionError, TimeoutError):
        # NO ANSWER, which is the only condition ``None`` means. A ``LinkCryptoError``
        # cannot reach here any more: the reader is ``dial.LineReader``, whose failures
        # are the two named ones above.
        return None
    finally:
        _close_quietly(sock)


# ---------------------------------------------------------------------------
# The foreground runner (`lop network serve`)
# ---------------------------------------------------------------------------


def amain(argv: list[str] | None = None) -> int:
    """``python -m local_operator.network.relay`` — the process launchd supervises.

    Foreground by design and installed with ``--module``; it never double-forks and
    never writes a pidfile of its own, because supervision belongs to launchd (or
    to a developer's terminal) and a self-daemonizing process is one nobody can
    stop.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the lop mesh relay in the foreground")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--address", default="")
    args = parser.parse_args(argv)

    settings = NetworkSettings.from_config()
    if args.port:
        settings = replace(settings, port=int(args.port))
    if args.address:
        settings = replace(settings, listen_address=args.address)

    from local_operator.logger import configure_console_logging, quiet_wire_clients

    configure_console_logging()
    # Without this pin a dependency's ``basicConfig`` floods the supervised log file
    # with one record per request — the same reason the mobile daemon calls both.
    quiet_wire_clients()
    server = RelayServer(settings=settings)
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover - the supervised entry point
    raise SystemExit(amain())
