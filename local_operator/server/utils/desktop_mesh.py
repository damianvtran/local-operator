"""The desktop's mesh reads, over THIS device's own relay (``features.peers``).

WHY A MODULE RATHER THAN CODE IN THE ROUTES. Three routes answer from the same two
relay reads (the peer table and the federated session rows), and a fourth joins
them per network. Written in the routes, each would open its own control connection
and its own copy of the collapse/gloss rules; here there is one seam that speaks to
the relay on the desktop's behalf, and the routes are HTTP plus a refusal mapping.

THE AUTHORISATION RULE, stated once because all three mutating routes share it.
Every peer is reached THROUGH this device's relay: this process never dials a peer,
never learns an endpoint from a renderer and holds no mesh key of its own
(``mesh-ui.md`` §2.2). The control socket is loopback-only and key-authenticated
with the 0600 record, so asking it is the operator's own act — the desktop token
gate is the outermost boundary and the relay's authoriser is the inner one. That is
deliberate rather than incidental: a renderer must not be able to do anything from
HTTP that the operator could not do from their own shell, and it must not be able to
do less in a way that invents a second policy. So these calls are the SAME ops the
CLI's verbs run (``net_invite``/``net_member_rm``/``session_move``), with the same
defaults, and every decision — who may be admitted, whether a member may be revoked,
whether a move is allowed — stays in the relay, together with the sentence a
refusal carries.

WHAT A PEER ROW MAY EXPOSE. A device id, a human name the peer chose, the transport's
reachability verdict and its glossed reason, a last-seen stamp, and a session COUNT.
Nothing a peer did not publish: no addresses beyond the member row the operator's own
network record already holds, no credentials, no transcript content. The count is
taken from the same cached projection the sidebar groups on, so a heading and this
catalogue cannot disagree on one screen.

ZERO-PEER COST IS ENFORCED, NOT ASSUMED. Every entry point here checks
:func:`has_any_network` FIRST, which is a read-only ``is_dir`` that never creates
``<config>/network``: a machine in no network answers an empty list having opened no
socket and written no file, which is what keeps every existing install byte-identical
and is the property ``session/peer_rows.py`` states for itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from local_operator.network.types import MeshRefusal

# GLOSSING IS CALLED DIRECTLY, NOT THROUGH A HELPER, and that is a guard's requirement
# rather than a style choice: ``tests/unit/network/test_reason_surfaces.py`` counts a
# ``reason`` read as glossed only when it is an argument of one of the project's three
# gloss functions. A local wrapper is invisible to that walk, so a read routed through
# one reads as an UNGLOSSED leak even while it glosses — which would force a declaration
# admitting the very thing this module must not do.
from local_operator.resume import peer_reason_words

#: ALWAYS ``None``, and this is a measurement rather than a placeholder.
#:
#: The contract declares ``rtt_ms: number | null`` and the renderer draws ``—`` for
#: null. The transport DOES measure a connect latency, but only inside a probe or a
#: ``doctor`` handshake (``relay.probe_candidates``/``_handshake``'s ``latency_ms``)
#: and it is printed, never persisted: neither ``peer_status`` nor ``network_detail``
#: carries it, and nothing on a link records a round trip. The only way to publish a
#: number here would be for THIS route to dial every peer on every poll — the
#: renderer polls at ``PEER_POLL_MS`` (30 s), so that would be a probe per row every
#: 30 s per client, to fill one hover-card field. That is a worse answer than an
#: absent one, and inventing a plausible figure is worse still. So the field is
#: published, honest, and null until the transport records a real one (the follow-up
#: is a ``latency_ms`` on the member/peer row, which is P0's file).
RTT_MS: float | None = None

#: The phase → progress table the transfer receipt publishes. A STEP count, not a
#: measure: the renderer draws a progress bar and there is no byte fraction to
#: report, so each phase answers "how far along the monotone list am I".
_PHASE_PROGRESS: dict[str, float] = {
    "prepared": 0.25,
    "handing_off": 0.5,
    "committed": 0.75,
    "done": 1.0,
}

#: How long the fast mesh verbs (mint an invite, revoke a member) may take. The
#: relay's own ops write one record under its lock and queue frames, so this is
#: generous rather than tight; a listing read uses the relay's listing budget
#: instead, because it dials every member.
_VERB_TIMEOUT_S = 20.0


def has_any_network(root: Path | None = None) -> bool:
    """Whether this device is in any network AT ALL, without creating anything.

    ``store.networks_dir`` mkdirs, which is right for a writer and wrong for a
    reader: a desktop route must not be the reason ``~/.local-operator/network``
    appears on a machine that has never joined one.
    """
    from local_operator.paths import config_dir

    return ((Path(root) if root is not None else config_dir()) / "network" / "networks").is_dir()


def _relay_record(root: Path | None) -> Any:
    from local_operator.network import store

    return store.find_own_relay(root)


def relay_unavailable() -> MeshRefusal:
    """The family's own "no relay answered" refusal, sentence included.

    Reusing ``network.cli``'s message rather than composing one here is the point:
    it distinguishes "the relay is not running" from "it is running and has stopped
    reporting" by reading the record, and a second copy would be free to drift from
    what the CLI's verbs say about the same machine.
    """
    from local_operator.network.cli import _relay_unavailable_message

    return MeshRefusal("relay_unavailable", _relay_unavailable_message())


def _call(root: Path | None, op: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
    """One control op on this device's relay, or a refusal naming the remedy.

    The relay's refusal crosses with its OWN code and sentence: the peer (or the
    relay) is the only party that can see which guard fired, and re-deriving a
    sentence on this side is how two surfaces come to disagree about the remedy.
    """
    from local_operator.network import relay as relay_mod

    record = _relay_record(root)
    if record is None:
        raise relay_unavailable()
    reply = relay_mod.control_request(record, op, timeout=timeout, **fields)
    if reply is None:
        raise relay_unavailable()
    if reply.get("op") == "error":
        raise MeshRefusal(
            str(reply.get("code") or "relay_refused"),
            str(reply.get("message") or "this device's relay refused that"),
        )
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else {"value": detail}


def _listing_call(root: Path | None, op: str, **fields: Any) -> dict[str, Any]:
    """A read that fans out to peers, under the relay's own listing budget.

    THE CLIENT'S BOUND MUST OUTLAST THE RELAY'S (``relay.LISTING_PROBE_BUDGET_S``):
    a client that gave up first would report "no peers" about a mesh whose peer was
    simply slower than the default, which is the silent-empty answer one layer down.
    """
    from local_operator.network import relay as relay_mod

    return _call(root, op, timeout=relay_mod.LISTING_CLIENT_TIMEOUT_S, **fields)


def peer_catalogue(root: Path | None = None) -> dict[str, Any]:
    """``GET /v1/desktop/peers``: one row per OTHER device, deduped across networks.

    ONE ROW PER DEVICE (Addendum 2, A). ``relay.peer_status`` answers per network
    MEMBERSHIP — a device in two networks appears twice — and a renderer grouping on
    those rows draws two sections holding the same chats. Membership-per-network is
    the Networks tab's question (:func:`network_topology`), which keeps it per
    network on purpose; this catalogue is the flat one the sidebar needs.

    ``session_count`` COMES FROM THE SIDEBAR'S OWN SET. It is counted from
    ``session.peer_rows.peer_session_rows`` — the TTL-cached projection the chat list
    groups on — rather than from a second fan-out, so the peer heading and this row
    are one answer rather than two reads that can disagree about a mesh that is
    moving. That read is shared with an ``include_peers`` listing, so one poll costs
    one fan-out, not two.
    """
    from local_operator.session.peer_rows import peer_session_rows

    if not has_any_network(root):
        return {"self_device_id": None, "peers": [], "degraded": []}

    table = _listing_call(root, "net_peer_ls")
    counts: dict[str, int] = {}
    for row in peer_session_rows(root):
        device = str(getattr(row, "owner_device", "") or "")
        if device:
            counts[device] = counts.get(device, 0) + 1

    collapsed: dict[str, dict[str, Any]] = {}
    for entry in table if isinstance(table, list) else table.get("value") or []:
        if not isinstance(entry, dict):
            continue
        device_id = str(entry.get("device_id") or "")
        if not device_id:
            continue
        reachable = bool(entry.get("reachable"))
        seen = _as_float(entry.get("last_seen_at"))
        current = collapsed.get(device_id)
        if current is None:
            collapsed[device_id] = {
                "device_id": device_id,
                "name": str(entry.get("name") or ""),
                # TRUE IF ANY SHARED NETWORK REACHED IT: the row answers "is this
                # device there", which is a fact about the device and not about the
                # network it happened to be listed under.
                "reachable": reachable,
                "unreachable_reason": (
                    "" if reachable else peer_reason_words(str(entry.get("reason") or ""))
                ),
                "last_seen_at": seen,
                "session_count": counts.get(device_id, 0),
                "rtt_ms": RTT_MS,
            }
            continue
        if reachable and not current["reachable"]:
            current["reachable"] = True
            current["unreachable_reason"] = ""
        elif not reachable and not current["reachable"] and not current["unreachable_reason"]:
            # The FIRST non-empty reason wins: every entry for one device describes
            # the same condition, and a device unreachable in one network and never
            # attempted in another would otherwise report the weaker sentence.
            current["unreachable_reason"] = peer_reason_words(str(entry.get("reason") or ""))
        if not current["name"]:
            current["name"] = str(entry.get("name") or "")
        if seen is not None and (current["last_seen_at"] is None or seen > current["last_seen_at"]):
            current["last_seen_at"] = seen

    self_device_id = ""
    for record in _networks(root):
        self_device_id = str(record.self_device_id or "")
        if self_device_id:
            break
    return {
        "peers": [collapsed[device] for device in sorted(collapsed)],
        "self_device_id": self_device_id or None,
        "degraded": [],
    }


def _networks(root: Path | None) -> list[Any]:
    """This device's network records, or ``[]``. Read here so nothing else mkdirs."""
    if not has_any_network(root):
        return []
    from local_operator.network import store

    return store.list_networks(root)


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def network_topology(root: Path | None = None) -> dict[str, Any]:
    """``GET /v1/desktop/networks``: per NETWORK members, never collapsed.

    A DEVICE IN TWO NETWORKS STAYS TWO MEMBERSHIPS (Addendum 2's reason for the
    per-network shape): the graph draws one node with two edges, and a flat
    per-device list cannot express the second edge at all.

    ``network_detail`` IS THE SOURCE for a running relay — it refreshes the member
    table by contacting peers first, so "how many members" is answered with its
    provenance rather than from a stale snapshot. It does NOT publish
    ``last_seen_at``, which the contract asks for, so that one field is read from
    the same record the relay just refreshed: the record is one file on this disk,
    and a member list built from two sources would be a second table.

    WITH NO RELAY the tab still answers, from this device's own record: the reach
    flags say ``reachable: false`` with the relay's absence named, because "I cannot
    ask anyone" is what a user opening the Networks tab on a stopped relay needs to
    read, and a 503 there would hide the networks they can see.
    """
    records = _networks(root)
    topology: list[dict[str, Any]] = []
    self_device_id = ""
    relay_up = _relay_record(root) is not None
    #: KEYED BY ``(network_id, device_id)``, NOT BY DEVICE. The peer table is one
    #: entry per MEMBERSHIP, and collapsing it here would make a device reachable in
    #: one network and unreachable in another report the same verdict on both edges —
    #: the per-network collapse this route exists NOT to do. (The peer CATALOGUE
    #: collapses them on purpose; the two reads answer different questions.)
    peer_table: dict[tuple[str, str], dict[str, Any]] = {}
    if relay_up:
        try:
            raw = _listing_call(root, "net_peer_ls")
            for entry in raw if isinstance(raw, list) else raw.get("value") or []:
                if isinstance(entry, dict) and entry.get("device_id"):
                    peer_table[(str(entry.get("network_id") or ""), str(entry["device_id"]))] = (
                        entry
                    )
        except MeshRefusal:
            # A refused peer table costs the reach columns, not the tab: the member
            # lists below still come from each record, and every member then reads
            # reachable: false with the reason named.
            relay_up = False

    for record in records:
        detail: dict[str, Any] = {}
        if relay_up:
            try:
                detail = _call(root, "net_show", timeout=_VERB_TIMEOUT_S, network=record.network_id)
            except MeshRefusal:
                detail = {}
        if not detail:
            # The local half: same shape, this device's own table, no live probe.
            detail = {
                "network_id": record.network_id,
                "name": record.name,
                "epoch": record.epoch,
                "trust": record.trust,
                "members_detail": [
                    {
                        "device_id": member.device_id,
                        "name": member.name,
                        "role": member.role,
                        "capabilities": list(member.capabilities),
                        "active": member.active,
                        "suspect": member.suspect,
                        "endpoints": list(member.endpoints),
                    }
                    for member in record.members
                ],
            }
        if not self_device_id:
            self_device_id = str(record.self_device_id or detail.get("self_device_id") or "")
        members: list[dict[str, Any]] = []
        for entry in detail.get("members_detail") or []:
            if not isinstance(entry, dict):
                continue
            device_id = str(entry.get("device_id") or "")
            member = record.member(device_id)
            reach = peer_table.get((record.network_id, device_id)) or {}
            reachable = bool(reach.get("reachable")) if relay_up else False
            # THREE CASES, decided before the gloss rather than after, because the
            # gloss NEVER returns empty (``peer_reason_words("")`` is "it did not
            # answer") and would therefore swallow the third case's sentence.
            if relay_up and reachable:
                reason = ""
            elif relay_up:
                # GLOSSED AT THE READ, never later: the guard that walks these modules
                # accepts a ``reason`` read only as an argument of the project's own
                # gloss function, so a raw read captured into a variable and glossed
                # afterwards reads as a leak.
                reason = peer_reason_words(str(reach.get("reason") or ""))
            else:
                # THIS FILE'S OWN PROSE, authored for the person reading the tab and
                # not a transport token: it says why nothing could be asked at all.
                reason = "the relay is not running, so no device was asked"
            if device_id and device_id == str(record.self_device_id):
                # THIS DEVICE IS HERE, and the relay never probed itself: the peer table
                # skips the device that owns the table, so a join that read absence as
                # "unreachable" drew the node for THIS machine as a dead device with
                # "it did not answer" beside it (measured on a live single-device
                # network). Reachability of oneself is not a probe result — it is what
                # "self" means — and ``self_device_id`` on the same answer is how the
                # renderer knows which node this is.
                reachable, reason = True, ""
            members.append(
                {
                    "device_id": device_id,
                    "name": str(entry.get("name") or ""),
                    "role": str(entry.get("role") or ""),
                    "capabilities": [str(item) for item in entry.get("capabilities") or []],
                    "active": bool(entry.get("active")),
                    "suspect": bool(entry.get("suspect")),
                    "endpoints": [str(item) for item in entry.get("endpoints") or []],
                    "last_seen_at": member.last_seen_at if member is not None else None,
                    "reachable": reachable,
                    "reason": "" if reachable else reason,
                }
            )
        topology.append(
            {
                # THE IDENTITY COMES FROM THE RECORD WE ASKED FOR, never from the
                # answer: the relay resolves a name to an id we already hold, and a
                # row keyed by whatever came back would file a network under the
                # wrong id the first time one answer described another network.
                "network_id": record.network_id,
                "name": str(detail.get("name") or record.name),
                "epoch": int(detail.get("epoch") or record.epoch),
                "trust": str(detail.get("trust") or record.trust),
                "members": members,
            }
        )
    return {"networks": topology, "self_device_id": self_device_id or None}


def remote_session_rows(
    root: Path | None, *, pins: set[str], query: str | None = None
) -> list[dict[str, Any]]:
    """Every session another device holds, as rows THIS desktop's list can paint.

    THE FLAT FIELDS ARE THE CONTRACT (Addendum 2, B): ``locality`` plus
    ``owner_device``/``owner_device_name``/``reachable``/``unreachable_reason`` on the
    row itself. The renderer groups and labels from those, and the nested transport
    block is deliberately NOT published here — a grouped-by-nested-block list filed
    every remote row under one heading in the first review.

    EVERY ROW CARRIES ALL SIX FIELDS, LOCAL VALUES INCLUDED, which is the merge rule
    the whole row shape follows: a client's merge is "an absent key is not a claim",
    so a row that MOVED home would otherwise keep a stale ``remote`` mark and stay
    filed under a peer it no longer lives on.

    ``pinned`` is read from THIS device's pin index because that is the store the
    desktop's own pin control writes; ``placement``/``origin`` are ``null`` because
    the federated row this reads does not carry the owner's stamp (see the module's
    report note) — a null is no claim, where the peer's home device guessed would be
    a wrong one.

    ``query`` filters by name and id exactly as the local search does (case-folded
    substring), and never by body: a peer's transcript is not on this disk, so a
    remote hit is never reported as a ``body_match``.
    """
    from local_operator.session.peer_rows import peer_session_rows

    if not has_any_network(root):
        return []
    needle = (query or "").strip().casefold()
    rows: list[dict[str, Any]] = []
    for row in peer_session_rows(root):
        session_id = str(getattr(row, "id", "") or "")
        name = str(getattr(row, "name", "") or "")
        if needle and needle not in name.casefold() and needle not in session_id.casefold():
            continue
        reachable = bool(getattr(row, "reachable", True))
        rows.append(
            {
                "id": session_id,
                "name": name,
                "mtime": float(getattr(row, "mtime", 0.0) or 0.0),
                "preview": "",
                "pinned": session_id in pins,
                # The PEER's own answer when it sent one; a peer does not offer an
                # archived session in its listing at all, so anything else is the
                # honest False rather than a claim that it is unarchived.
                "archived": bool(getattr(row, "archived", False)),
                "degraded": [],
                "subagents_running": None,
                "subagents_queued": None,
                "locality": "remote",
                "owner_device": str(getattr(row, "owner_device", "") or ""),
                "owner_device_name": str(getattr(row, "owner_device_name", "") or ""),
                "reachable": reachable,
                "unreachable_reason": (
                    ""
                    if reachable
                    else peer_reason_words(str(getattr(row, "unreachable_reason", "") or ""))
                ),
                "placement": None,
                "origin": None,
                "last_synced_at": None,
                "active": False,
                # The transport's own state token, in this list's vocabulary
                # (``session.catalog.live_state_from_flags``'s four words), or "" for
                # a cold row — the same spellings the local half publishes.
                "status": {
                    "code": _remote_status_code(row),
                    "label": _remote_status_label(row),
                },
                "binding": {"agent": None, "team": None},
            }
        )
    return rows


def _remote_status_code(row: Any) -> str:
    """The ``status.code`` a remote row publishes, from the peer's own two flags.

    A peer's row is decorated by its OWN catalogue: ``live_state`` (whose four words
    ``row_state_mark`` ranks) and ``pending``. This publishes the same precedence the
    local half uses, through the one function that owns it, so a remote row and a
    local row cannot answer "is this running, and is anyone watching it" differently.
    """
    from local_operator.session.catalog import status_of

    code, _label = status_of(row, None)
    return code


def _remote_status_label(row: Any) -> str:
    from local_operator.session.catalog import status_of

    _code, label = status_of(row, None)
    return label


def create_on_peer(
    root: Path | None, peer: str, *, model: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Mint a conversation ON ``peer``, through this device's relay.

    THE PEER MINTS THE ID and takes the runtime: this device asks its own relay to
    ask the peer, exactly as ``lop network sessions --create`` does, so there is one
    implementation of "a session this device does not host" and the id the peer
    answers with is the id both ends use from then on.

    ``cwd`` is NOT SENT AT ALL, which is how the peer defaults to its own home
    (``relay._op_session_create``: an absent cwd becomes ``Path.home()`` there).
    Sending this machine's directory would name a path on a disk the peer does not
    share, and the renderer's ``cwd`` always names one HERE. No first prompt is sent
    either: the desktop's ``/new`` opens a conversation, and a prompt from this device
    would be work the user has not asked for yet.
    """
    return _call(
        root,
        "peer_session_create",
        # A spawn plus its first admission, the CLI's own budget for this verb.
        timeout=120.0,
        peer=peer,
        model=model,
    )


def remote_owner(root: Path, session_id: str) -> tuple[str, str] | None:
    """``(device_id, name)`` when another device holds ``session_id``, else ``None``.

    Reads the SAME producer the sidebar's poll fills (``session.remote_open``'s
    ``remote_row_for``), so a row the user can see is a row this can act on: the
    cache answers for free on the path every archive/delete takes, and one relay read
    is paid only when this device holds no directory for the id. A local session is
    never asked about, which is what keeps the local path's cost unchanged.

    Blocking: the caller must run it off the event loop.
    """
    from local_operator.session.remote_open import remote_row_for

    row = remote_row_for(session_id, root)
    if row is None:
        return None
    return str(row.owner_device), str(row.owner_device_name or "")


def lifecycle_on_owner(
    root: Path | None,
    session_id: str,
    *,
    #: The OWNER's verbs, spelled here rather than imported from ``network.mobility``:
    #: that module pulls the whole relay and is not a boot-closure import, and these
    #: three names are the family's own (``LifecycleAction``).
    action: Literal["archive", "unarchive", "delete"],
    peer: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Archive, restore or delete a session that lives on ``peer``.

    ROUTED, NOT REPLICATED: the owner runs its own ``archive_change``/``delete_session``
    (``mobility.lifecycle``), so the guards, the confirmation semantics and the wake
    pruning stay on the device that owns the disk. A second ``rmtree`` of a session
    directory anywhere else is what ``test_no_session_deletion`` exists to prevent.

    A refusal comes back as ``{"ok": False, "code", "message"}`` in the family's
    shape; the caller maps it to a status and keeps the sentence.
    """
    from local_operator.network import mobility

    return mobility.lifecycle(session_id, action=action, peer=peer, confirmed=confirmed, root=root)


def invite(root: Path | None, network: str, *, role: str, device: str = "") -> dict[str, Any]:
    """Mint an invite and answer with the FILE PATH its token was written to.

    THE TOKEN IS NEVER RETURNED. The relay writes it to
    ``<config>/network/outbox/<id>.invite`` and answers with the path — the CLI's own
    rule, whose reason is that a token in a JSON payload is a token in a transcript
    — and this route keeps it, so the desktop hands the user a path to read rather
    than a secret to paste into a chat window.

    NO LOCAL FALLBACK: minting appends an invite row to a record the relay's own
    loops write, and the CLI's offline half re-implements that write. A second
    implementation of it here is exactly the drift the relay exists to prevent, so a
    machine with no relay is told to start one.
    """
    return _call(
        root,
        "net_invite",
        timeout=_VERB_TIMEOUT_S,
        network=network,
        role=role,
        device_id=device,
    )


def remove_member(root: Path | None, network: str, device: str) -> dict[str, Any]:
    """Revoke a member and rotate the network secret, through the relay.

    The typed confirmation is checked by the ROUTE against the network's name (see
    ``routes/desktop_mesh``), because the confirmation is a human act and the relay
    has no business holding a sentence a person typed. This function is the write
    itself: the tombstone, the rotation and the queued frames all happen on the
    relay, which owns the record's lock.
    """
    return _call(
        root,
        "net_member_rm",
        timeout=_VERB_TIMEOUT_S,
        network=network,
        device_id=device,
    )


def resolve_network(root: Path | None, target: str) -> Any:
    """One network by id or name, refusing an ambiguous name rather than guessing.

    The CLI's rule, including its wording: two networks may share a display name by
    design, so an ambiguous argument names the candidates instead of picking one.
    """
    from local_operator.network import store

    records = _networks(root)
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
        raise MeshRefusal("unknown_network", f"this device is not in a network called {target!r}")
    if len(matches) > 1:
        raise MeshRefusal(
            "ambiguous_network",
            f"{target!r} matches {len(matches)} networks; use the network id",
        )
    return matches[0]


def transfer(
    session_id: str, *, to: str, keep: bool = False, wait_s: float = 0.0, root: Path | None = None
) -> dict[str, Any]:
    """Move (or ``keep``-copy) a conversation, through ``mobility.request_move``.

    THE CLI'S OWN ENTRY POINT, deliberately: it already owns the action choice
    (recall/offload), the budget a slow retirement needs, and the refusal shapes a
    front end branches on. A route that composed its own ``session_move`` frame would
    be a second caller of the relay's most destructive local verb, and the two would
    drift the first time a phase changed.

    Blocking, and its caller must run it off the event loop: the relay holds this call
    for the whole move — a retire, a copy and a confirmation — which is seconds, and
    up to ``wait_s`` longer when the source is busy.
    """
    from local_operator.network import mobility

    return dict(mobility.request_move(session_id, to=to, keep=keep, wait_s=wait_s, root=root))


def transfer_receipt(
    result: dict[str, Any], *, session_id: str, keep: bool, to: str
) -> dict[str, Any]:
    """The move's result as the ONE answer the renderer consumes (Addendum 1, item 4).

    ``phases`` carries the monotone phase list with a progress step and the OTHER
    end of the move per phase. A per-phase device is not something the move records
    (a stamp is a name and a time), so ``peer`` is the move's other end throughout:
    the destination for an offload, the source for a recall. That is the fact the row
    being moved is about — which device it is moving to or from — and it is why the
    receipt's own ``owner_device`` is the destination rather than a per-phase tally.

    ``source_retired`` is ``mode == "move"``: a ``keep`` copy is minted at the
    destination and the source is untouched by design, which is the difference the
    two verbs exist to express.
    """
    phases = [
        {
            "phase": str(stamp.get("phase") or ""),
            # A phase this build does not know gets 0.0: the renderer's bar must not
            # claim a completion nobody reported.
            "peer": str((result.get("to_device") or {}).get("device_id") or to),
            "progress": _PHASE_PROGRESS.get(str(stamp.get("phase") or ""), 0.0),
        }
        for stamp in result.get("phases") or []
        if isinstance(stamp, dict)
    ]
    mode = str(result.get("mode") or ("keep" if keep else "move"))
    return {
        "phases": phases,
        "locality": "local" if to == "local" else "remote",
        "owner_device": str((result.get("to_device") or {}).get("device_id") or ""),
        "source_retired": mode == "move",
        "session_id": session_id,
        "new_session_id": str(result.get("new_session_id") or session_id),
        "mode": "keep" if mode == "keep" else "move",
    }
