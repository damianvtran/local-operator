"""The credential sync engine: generations, acks, and the copy path (design S3).

WHAT THIS IS (``mesh-consent-provisioning.md`` §5 and §9.2's S3 row). A credential
shared to an approved device is a COPY that must stay current while life goes on:
rotate a value here and the copy there converges within about a minute, with no
human on either side. Three pieces, one machine:

* **generations** — an owner-side, per-``(network, key)`` monotonic counter,
  bumped when the owner's stored value changes. Change detection diffs reads the
  tree already does (``updated_at`` per row plus a digest of the value) and is
  persisted in ``credentials/<network_id>/sync.json`` beside the placement
  documents (§5.2; the store-write hook can replace the diff later without a wire
  change — Q6).
* **the ack ledger** — owner-side ``{device: {key: {gen, digest, at}}}``: what
  each holder has confirmed it HOLDS. It is what makes staleness visible to the
  owner (``lop network credentials`` and ``doctor`` render it) and the whole of
  what an ack does: recording one never blocks anything (§5.5b).
* **the wire** — three new kinds ON THE EXISTING ``net_broker`` op, never a new
  op (``mesh-credentials.md`` §7.10: the transport authorises per op, so four ops
  for one authority would be four capability rows to keep identical):

  - ``announce`` (owner -> member): ``{kind, key, gen, digest, value_state}`` for
    a key whose generation moved since this member's last ack. Tiny, idempotent,
    RECOMPUTED on the next contact — a missed announce is not replayed (§5.1).
  - ``copy`` (member -> owner): the member asks for ``key`` naming the
    generation it holds; the owner answers with the value plus
    ``{gen, digest, provenance}`` inside the link's already-authenticated
    records (§5.1, §5.3: push the announcement, PULL the payload).
  - ``ack`` (member -> owner): ``{kind, key, gen, digest}`` — what the member
    holds NOW, so the owner can record it and stop announcing.

THE CARRIER IS THE DEFINITIONS SYNCER'S TICK (``definitions.add_tick_step`` —
read ``mcpdefs.mesh_tick_step``, the first tenant, before touching this). The
step runs after each member's definitions push, INSIDE the ``mesh-definitions``
thread and its floors (a 15 s tick, a 60 s floor to one reachable member), so
this slice starts no thread, invents no second retry policy, and its latency
budget is §5.3's ≈75 s p95 to a reachable member. The step is NON-BLOCKING per
member by construction: it enqueues onto the broker's event loop and returns;
the exchange runs there (``_BrokerLoop``'s executor, so nothing blocks the
loop).

THE NON-STALL PROPERTIES (§5.4), each pinned by a test:
1. use never awaits sync — nothing on a use path calls into this module;
2. sync never awaits a busy member and a busy member never defers — the tick
   step enqueues, and the member schedules its own pull off the responder
   thread, so a member under load is exactly the member that keeps syncing;
3. applying a copy is atomic per key — one ``upsert_credential`` write, visible
   in full or not at all, and the superseded row is swept AFTER it (insert
   first, delete second: a reader sees the old value or the new one, never
   neither).

WHAT COPIES, AND WHAT EACH SELECTION LAYER DOES. Copies flow for keys whose
class is copy-eligible (``COPY_KINDS``: the static classes — class 4's
``api-key-static`` and class 2's ``store-secret``, the encrypted store) AND
where the member is an ACTIVE HOLDER — the same authorisation the broker reads,
never a parallel one (§8.3: ``copy_requires_active_holder``). On top of that,
S4's selection POLICY (§4.2) decides which class-2 keys are SHARED BY DEFAULT:
the union of the ``ref:<NAME>`` names the pushed bundles declare
(:func:`needs_names`) and the keys the operator marked ``sync``; keys marked
``local-only`` never cross at all; everything else the device holds is OFFERED
— visible on the join list, not copied — and the card's reduce step trims the
served set (§1.1's pairing path). The receiving side re-seals class-2 copies
under its OWN key with a provenance marker, and wipe notices (announces with
``value_state: absent``) delete by that marker. The listing's freshness check
at use stays S5's, per the S3 PR's handoff note.

THE RESET EDGE, HANDLED RATHER THAN PAPERED OVER. If the owner's sync state is
lost, its counters restart BELOW what a member holds. The monotonically safe
convergence (§5.2's rule, kept absolute: a member applies only a generation
strictly greater than the one it holds) has two digest-driven paths: equal
digests ADOPT the owner's counter — no value moves, so a rollback is impossible
— and different digests pull, where the owner serves ``max(current, held + 1)``
so the member can move forward. Evidence moves counters; nothing moves a VALUE
backward. (Detected rather than assumed: see the ``reset`` cells in
``tests/unit/network/test_credentials_sync.py``.)

Stdlib only at import: the relay imports the credentials package at
construction, and the heavy reaches (``providers.auth_store``) happen inside the
functions that need them, exactly as ``credentials/__init__.py`` states.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

from local_operator.network.credentials.types import (
    SECRET_KIND,
    BrokerError,
    peer_int,
    secret_name_from_key,
)

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

logger = logging.getLogger("local_operator.network.credentials.sync")

#: The state document's schema, its name beside the placement files, and the lock
#: file beside it — the placement module's layout, one directory over.
SYNC_SCHEMA = 1
SYNC_FILENAME = "sync.json"
SYNC_LOCK_FILENAME = ".sync.lock"

#: Generations are peer-carried integers, so they are validated like every other
#: number a frame carries (``types.peer_int``, whose ceiling is 2**53).
GEN_CEILING = 2**53

#: The adoption bound for a member-supplied ``held`` (review round 1, R1): an
#: owner only ADOPTS a wire generation that still leaves this much room below
#: :data:`GEN_CEILING`, because every later change must bump PAST the number
#: adopted. A frame (or a corrupt row) naming the ceiling would freeze the
#: key's generation: bumps clamp at the ceiling and a member AT the ceiling
#: drops every reply (``served <= held``), a permanent liveness break for one
#: key from one frame. 2**20 is deliberately vast next to any real change count
#: (a counter that reached a million is already absurd) and tiny next to the
#: ceiling, so the bound refuses nonsense without ever touching an honest
#: device.
GEN_ADOPT_HEADROOM = 2**20

#: The highest generation this engine will ADOPT FROM THE WIRE, and the line
#: above which its own loaded rows are treated as corruption: rows at or above
#: it are dropped at load (the load docstring's ``unreadable falls back to a
#: fresh one`` rule), and a ``copy`` frame asking to continue from one is
#: refused by name instead of adopted.
GEN_SAFE_MAX = GEN_CEILING - GEN_ADOPT_HEADROOM

#: Bounded work per exchange (§5.1: "one bounded unit"): at most this many keys
#: are announced to one member in one tick, so a large copy-set cannot turn a
#: tick into a burst.
ANNOUNCE_CAP = 8

#: Wire timeouts. The announce is a few dozen bytes, so it gets the definitions
#: push's own order of magnitude; the copy carries a payload and gets more room,
#: still well under the owner-side ``net_broker`` deadline (75 s) whose slow-op
#: worker answers it.
ANNOUNCE_TIMEOUT_S = 10.0
COPY_TIMEOUT_S = 30.0

#: ``value_state`` on an announce/copy: whether the owner holds a value for the
#: key. ``absent`` is the WIPE NOTICE (§5.5c, §4.3): the member deletes its
#: copy by provenance and acks the deletion — the one ending the copy path has
#: short of rotation, and the shape that makes a delete "just another announce
#: recomputed on the next contact" (offline members get it on reconnect, never
#: a queued frame that can be lost).
VALUE_STATE_PRESENT = "present"
VALUE_STATE_ABSENT = "absent"

#: The selection marks of §4.2, owner-side and per (network, key), kept in the
#: sync document beside the generations they serve. ``sync`` — the operator's
#: standing "send this to approved nodes" mark, joined to every approved
#: device's copy-set by default. ``local-only`` — never crosses; the one mark a
#: candidate list must not soften, so the offer drops such keys ENTIRELY rather
#: than offering them off-by-default, and the serve paths refuse a grant that
#: raced the mark (§4.2's kill switch, enforced where the value would move).
MARK_SYNC = "sync"
MARK_LOCAL_ONLY = "local-only"
MARKS: frozenset[str] = frozenset({MARK_SYNC, MARK_LOCAL_ONLY})

#: The classes whose values are COPIED, by the §2.1 mechanism mapping ("classes
#: 2/4 = copy; 3/5(+6b) = broker; 1/6 = refuse"). ``api-key-static`` is class 4;
#: ``store-secret`` is class 2 — the encrypted ``lop secret`` store, re-sealed
#: under the receiving device's own key on arrival (§8.2's copy invariant).
#: Device-bound providers are excluded by NAME, not by this table: the placement
#: write refuses them at the door (``placement.refuse_device_bound``).
COPY_KINDS: frozenset[str] = frozenset({"api-key-static", SECRET_KIND})

#: A sanity bound on a class-2 copy's hex-encoded material as it arrives on the
#: wire. The store itself does not bound value length; a frame carrying megabytes
#: of "material" is already absurd, and the bound turns it into a dropped reply
#: rather than a multi-megabyte SQLite write behind one JSON parse.
_SECRET_HEX_MAX = 2 * 1024 * 1024

#: The provenance marker a received copy carries in its store row's data. It is
#: what makes the replace-sweep findable WITHOUT the sync state (a lost sidecar
#: must not accumulate rows), and it is the minimal form of §4.2 bound 4's
#: "origin/owner_device field or sidecar index on the receiving side" that S4's
#: wipe-by-provenance will read.
MESH_ORIGIN_KEY = "mesh_origin"

#: Durability markers a copy must NOT carry: they describe a LOCAL interaction
#: (a pending refresh POST, a provider refusal) that has no meaning on the
#: receiving device, and carrying them would poison the receiver's failover.
#: Spelled here rather than imported from ``providers.auth_store`` for the
#: package's stdlib-at-import rule; ``test_credentials_sync`` asserts the spellings
#: against the owning module so a rename fails a test, not production.
_TRANSIENT_MARKERS = ("grant_dead_at", "refresh_send_unconfirmed")

_DIGEST_PREFIX = "sha256:"
_DIGEST_HEX_LENGTH = 64


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def copies_by_class(kind: str) -> bool:
    """Whether credentials of ``kind`` are copied to holders, or broker-only.

    ONE table, because §4.2's mechanism mapping is one decision and a second
    copy of it is how a class silently gains or loses a copy path.
    """
    return kind in COPY_KINDS


def fingerprint(value: Any) -> str:
    """``sha256:<hex>`` over the canonical JSON of a copy's value payload.

    Canonical (sorted keys, tight separators) so both sides compute the SAME
    digest from the same payload across a JSON round trip — the digest is the
    announce's "is this the value you hold" test and the copy reply's integrity
    check, and both would be fooled by key order. ``""`` for a payload that
    cannot be canonicalised, which every caller treats as "no value": a digest
    that cannot be computed must not read as "unchanged".
    """
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return ""
    return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bounded_digest(value: Any) -> str:
    """A digest a PEER sent, or ``""`` — the boundary every carried digest gets."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text.startswith(_DIGEST_PREFIX):
        return ""
    hex_part = text[len(_DIGEST_PREFIX) :]
    if len(hex_part) != _DIGEST_HEX_LENGTH:
        return ""
    if any(character not in "0123456789abcdef" for character in hex_part):
        return ""
    return text


def _bounded_key(value: Any) -> str:
    """A placement key a state file or frame carried, or ``""``."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text if 0 < len(text) <= 200 else ""


def _copyable_payload(data: Any) -> dict[str, Any] | None:
    """The copy-able projection of a store row's ``data``, or ``None``.

    ONE normalisation for both sides, so the owner's digest and the member's
    recomputation cannot disagree: transient markers and any origin marker the
    owner's own row happens to carry are dropped, the class type is stated, and
    a payload without a usable ``key`` is not copyable at all.
    """
    if not isinstance(data, dict):
        return None
    secret = data.get("key")
    if not isinstance(secret, str) or not secret:
        return None
    payload = {
        name: value
        for name, value in data.items()
        if name not in _TRANSIENT_MARKERS and name != MESH_ORIGIN_KEY
    }
    payload["type"] = "api_key"
    return payload


def read_copy_value(
    store: Any, key: str, entry: Any, *, root: Path | None = None
) -> dict[str, Any] | None:
    """``{"value", "digest", "updated_at"}`` for ``key``, or ``None``.

    TWO CLASSES, ONE SHAPE. A ``store-secret`` key (class 2) reads the key's
    record from THIS device's encrypted store through
    :func:`_read_secret_value` — a read that must not create the store (the
    provider-role namespace is refused on that store's own surface, so a name
    that cannot be copied cannot be keyed here either). An ``api-key-static``
    key reads the auth store:

    THE ROW PICK IS THE NEWEST ENABLED ROW of the key's class, and the direction
    is deliberate. ``AuthStore`` lets one provider hold several rows ("each key
    is its own row"), and the resolver rotates among them per session; a copy
    mirrors ONE value, so the engine must say which. The newest row is the
    operator's LATEST answer for the key — a re-login writes a new row, and a
    rotation pool's newest member is the one most recently proven alive — so
    "paste a new key" propagates without a second mechanism, where a first-row
    pick would pin every copy to the OLDEST row and go stale silently.
    """
    if str(entry.kind) == SECRET_KIND:
        return _read_secret_value(key, root)
    if str(entry.kind) != "api-key-static":
        return None
    provider = str(entry.provider or key)
    try:
        rows = store.list_credentials(provider)
    except Exception:  # noqa: BLE001 — an unreadable store holds nothing to copy
        logger.debug("credentials sync: value read failed for %s", key, exc_info=True)
        return None
    for row in reversed(rows):
        if str(getattr(row, "credential_type", "")) != "api_key":
            continue
        payload = _copyable_payload(getattr(row, "data", None))
        if payload is None:
            continue
        digest = fingerprint(payload)
        if not digest:
            continue
        return {
            "value": payload,
            "digest": digest,
            "updated_at": int(getattr(row, "updated_at", 0) or 0),
        }
    return None


def _secret_copy_payload(name: str, description: str, value: bytes) -> dict[str, Any]:
    """The copy-able projection of a class-2 record — ONE normalisation, both ends.

    The value is hex-encoded for the same reason the store's own payload is: a
    secret may be arbitrary binary, and hex round-trips with no padding mode to
    get wrong. The NAME rides inside the payload so the member can refuse a
    reply that does not name the key it asked for — defence in depth under the
    digest pin, never instead of it.
    """
    return {
        "type": SECRET_KIND,
        "name": name,
        "value": value.hex(),
        "description": description,
    }


def _read_secret_value(key: str, root: Path | None) -> dict[str, Any] | None:
    """The class-2 half of :func:`read_copy_value`, opened through the secrets seam.

    The store is opened WITHOUT ``create``: an announce read must never be the
    reason a secret store appears on a device, and a device with no store holds
    nothing to copy. The value comes from ``read_for_copy`` — a value read that
    does NOT touch ``last_used_at`` or the audit chain, because this runs on
    every sync tick purely to recompute a digest (see that method's docstring).
    """
    try:
        name = secret_name_from_key(key)
    except ValueError:
        return None
    try:
        from local_operator.secrets import access

        store = access.open_store(root)
    except Exception:  # noqa: BLE001 — no store, no readable store: nothing to copy
        logger.debug("credentials sync: secret store for %s unavailable", key, exc_info=True)
        return None
    try:
        try:
            record, value = store.read_for_copy(name)
        except Exception:  # noqa: BLE001 — an absent or damaged record holds nothing
            logger.debug("credentials sync: secret read for %s failed", key, exc_info=True)
            return None
        payload = _secret_copy_payload(name, str(record.description or ""), value)
        digest = fingerprint(payload)
        if not digest:
            return None
        return {
            "value": payload,
            "digest": digest,
            "updated_at": int(getattr(record, "updated_at", 0) or 0),
        }
    finally:
        _close_quietly(store)


def _network_record(root: Path | None, network_id: str) -> Any:
    from local_operator.network import store as network_store

    for record in network_store.list_networks(root):
        if str(record.network_id) == network_id:
            return record
    return None


def _member_is_active(root: Path | None, network_id: str, device_id: str) -> bool:
    """Whether ``device_id`` is an ACTIVE member — the §8.1/§8.3 withholding rule.

    A copy is withheld from a non-active (removed, left, never-active) member
    however the request reached the handler: the epoch-secret rule extended to
    copies. The transport already refuses most of the ways such a request could
    arrive, and this is the check that does not depend on the transport's
    configuration to stay true.
    """
    if not device_id:
        return False
    record = _network_record(root, network_id)
    if record is None:
        return False
    member = record.member(device_id)
    return member is not None and bool(getattr(member, "active", False))


def _authenticated_sender(link: Any, frame: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """``(device, error_detail)``: WHO sent this, from the transport, never the frame.

    The member-side twin of ``MeshCredentialBroker._caller``'s rule: the
    handshake's ``link.device_id`` is the identity, the frame's ``from_device``
    is an assertion kept so a mismatch is refused by name. It is the member side,
    so the refusal is a plain error detail (no ``_refuse`` audit — nothing was
    lent, and an announce is a message about state, not a request for material).
    """
    authenticated = str(getattr(link, "device_id", "") or "")
    claimed = str(frame.get("from_device") or "")
    if not authenticated:
        return "", {
            "kind": "error",
            "code": "identity_mismatch",
            "key": "",
            "message": "this frame arrived on a link with no authenticated device; "
            "nothing was changed",
        }
    if claimed and claimed != authenticated:
        return authenticated, {
            "kind": "error",
            "code": "identity_mismatch",
            "key": "",
            "message": f"this frame named {claimed} as its sender but arrived from "
            f"{authenticated}; nothing was changed",
        }
    return authenticated, None


# ---------------------------------------------------------------------------
# The state document
# ---------------------------------------------------------------------------


def sync_path(network_id: str, root: Path | None = None) -> Path:
    """The state document's path. NEVER CREATES anything (placement's discipline)."""
    from local_operator.network.credentials import placement as placement_mod

    return placement_mod.credentials_root(root) / network_id / SYNC_FILENAME


@contextmanager
def _sync_lock(target: Path) -> Iterator[None]:
    """The placement document's two-level lock, for the sync document.

    In-process RLock first, then the cross-process flock beside the file, because
    the two writers here are the same two the placement lock exists for: the
    relay's syncer thread and a CLI process (``lop network credentials`` only
    reads, but a future repair verb will write). Reusing ``placement``'s
    primitives rather than restating them keeps ONE wait bound and ONE refusal
    sentence for "another writer is changing credential state on this device".
    """
    from local_operator.network import store as network_store
    from local_operator.network.credentials import placement as placement_mod

    with network_store._write_lock(target):
        with placement_mod._cross_process_lock(target.parent / SYNC_LOCK_FILENAME):
            yield


class SyncState:
    """One network's sync document: generations, acks, and the applied map.

    NOTHING HERE IS MATERIAL. Generations and digests are fingerprints and
    counters; the ack ledger is who-holds-what-how-fresh; the applied map names a
    store ROW ID (class 4) or RECORD ID (class 2), never a value; the marks are
    the operator's two selection words. The one secret-shaped thing this module
    must never write down is the value itself, and it never does: the value
    travels on the authenticated link and lands in this device's own credential
    storage — for a class-4 copy, the same 0600 ``auth.db`` row a local login
    writes; a class-2 copy is re-sealed into ``secrets/`` under THIS device's
    own master key (design §8.2), which the S4 copy path now implements.
    """

    def __init__(self, network_id: str, *, root: Path | None = None) -> None:
        self.network_id = network_id
        self.root = root
        #: ``{key: {"gen": int, "digest": str, "updated_at": int}}`` — owner side.
        self.generations: dict[str, dict[str, Any]] = {}
        #: ``{device_id: {key: {"gen", "digest", "at"[, "wiped": True]}}}``.
        #: A ``wiped: True`` row is an ENDING, not a copy: the member confirmed
        #: deleting by provenance, and nothing is owed that member for the key
        #: until a re-share re-announces it.
        self.acks: dict[str, dict[str, dict[str, Any]]] = {}
        #: ``{key: {"gen", "digest", "owner_device", "row_id", "record_id", "at"}}``.
        self.applied: dict[str, dict[str, Any]] = {}
        #: ``{key: {"mark": str, "set_at": float}}`` — the §4.2 selection marks,
        #: owner-side, per (network, key). Kept here rather than in a second
        #: document because the serve paths that must read a mark (announce,
        #: copy, wipe) already hold THIS document's lock; a second registry for
        #: one policy would be a second writer to keep in step with this one.
        self.marks: dict[str, dict[str, Any]] = {}
        self._dirty = False

    # -- persistence ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return sync_path(self.network_id, self.root)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SYNC_SCHEMA,
            "network_id": self.network_id,
            "written_by": "",
            "written_at": time.time(),
            "generations": {key: dict(row) for key, row in sorted(self.generations.items())},
            "acks": {
                device: {key: dict(row) for key, row in sorted(by_key.items())}
                for device, by_key in sorted(self.acks.items())
            },
            "applied": {key: dict(row) for key, row in sorted(self.applied.items())},
            "marks": {key: dict(row) for key, row in sorted(self.marks.items())},
        }

    def save(self) -> Path:
        """Write the document, when something changed. The one writer.

        THE CALLER'S LOCK IS THE LOCK. ``mutate`` holds the two-level lock across
        the whole read-modify-write (the same shape ``PlacementDocument.save``
        has), so this method takes only the in-process ``_write_lock`` the staged
        write itself uses. It must NOT re-take the flock: the first draft did,
        and on macOS that self-deadlocks — flock locks belong to the open file
        DESCRIPTION, so a second ``open`` of the same lock file by the same
        process conflicts with the first and the nested acquire times out as
        "busy" (measured on this suite). A standalone writer goes through
        ``mutate``.
        """
        from local_operator.network import store as network_store
        from local_operator.network.credentials import placement as placement_mod

        if not self._dirty:
            return self.path
        placement_mod._ensure_private_dir(self.path.parent)
        return network_store._write_private_json(self.path, self.to_json())

    @classmethod
    def load(cls, network_id: str, root: Path | None = None) -> "SyncState":
        """Read the document, or an empty one. Never raises for a missing file.

        A corrupt row is DROPPED, not repaired: a generation that cannot be read
        falls back to a fresh one (the diff recomputes), a lost ack re-announces,
        a lost applied re-pulls. Every degraded path is the safe direction —
        more traffic, never a wrong apply — which is why dropping is honest here
        where the placement document instead refuses everything.
        """
        state = cls(network_id, root=root)
        try:
            raw = state.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return state
        try:
            payload = json.loads(raw)
        except ValueError:
            return state
        if not isinstance(payload, dict):
            return state
        generations = payload.get("generations")
        if isinstance(generations, dict):
            for key, row in generations.items():
                name = _bounded_key(key)
                if not name or not isinstance(row, dict):
                    continue
                gen = peer_int(row.get("gen"), maximum=GEN_CEILING)
                digest = _bounded_digest(row.get("digest"))
                # A row AT OR ABOVE the adoption bound is corruption, not a
                # counter: dropped like an unreadable one (review round 1,
                # R1's corrupt-state path), so it cannot pin the key.
                if gen <= 0 or gen >= GEN_SAFE_MAX:
                    continue
                state.generations[name] = {
                    "gen": gen,
                    "digest": digest,
                    "updated_at": peer_int(row.get("updated_at"), maximum=GEN_CEILING),
                }
        acks = payload.get("acks")
        if isinstance(acks, dict):
            for device, by_key in acks.items():
                device_id = str(device or "")
                if not device_id or not isinstance(by_key, dict):
                    continue
                for key, row in by_key.items():
                    name = _bounded_key(key)
                    if not name or not isinstance(row, dict):
                        continue
                    gen = peer_int(row.get("gen"), maximum=GEN_CEILING)
                    # Same corruption rule as the generations loop above.
                    if gen <= 0 or gen >= GEN_SAFE_MAX:
                        continue
                    state.acks.setdefault(device_id, {})[name] = {
                        "gen": gen,
                        "digest": _bounded_digest(row.get("digest")),
                        "at": peer_int(row.get("at"), maximum=GEN_CEILING),
                        **({"wiped": True} if row.get("wiped") is True else {}),
                    }
        applied = payload.get("applied")
        if isinstance(applied, dict):
            for key, row in applied.items():
                name = _bounded_key(key)
                if not name or not isinstance(row, dict):
                    continue
                gen = peer_int(row.get("gen"), maximum=GEN_CEILING)
                # Same corruption rule: a dropped ``applied`` row makes this
                # device re-pull the key, which is the healing half of R1's
                # corrupt-member-state path.
                if gen <= 0 or gen >= GEN_SAFE_MAX:
                    continue
                state.applied[name] = {
                    "gen": gen,
                    "digest": _bounded_digest(row.get("digest")),
                    "owner_device": str(row.get("owner_device") or ""),
                    "row_id": peer_int(row.get("row_id"), maximum=GEN_CEILING),
                    # Class 2's record id is a string (the store's primary key),
                    # and it rides beside the int so one sidecar serves both
                    # classes. Bounded like every other carried string; the
                    # wipe does not TRUST it (it scans by provenance) — it is
                    # the fast path and the audit trail.
                    "record_id": str(row.get("record_id") or "")[:128],
                    "at": peer_int(row.get("at"), maximum=GEN_CEILING),
                }
        marks = payload.get("marks")
        if isinstance(marks, dict):
            for key, row in marks.items():
                name = _bounded_key(key)
                if not name or not isinstance(row, dict):
                    continue
                mark = str(row.get("mark") or "")
                if mark not in MARKS:
                    # An unknown mark is dropped toward the SAFE direction: an
                    # unmarked key is offered, never synced (§4.2's default is
                    # the needs-list, so dropping a `sync` mark copies LESS).
                    continue
                state.marks[name] = {
                    "mark": mark,
                    "set_at": peer_int(row.get("set_at"), maximum=GEN_CEILING),
                }
        state._dirty = False
        return state

    # -- owner-side accessors ------------------------------------------------

    def generation(self, key: str) -> dict[str, Any] | None:
        return self.generations.get(key)

    def record_generation(self, key: str, gen: int, digest: str, updated_at: int) -> None:
        self.generations[key] = {"gen": int(gen), "digest": digest, "updated_at": int(updated_at)}
        self._dirty = True

    def ack_for(self, device: str, key: str) -> dict[str, Any] | None:
        return (self.acks.get(device) or {}).get(key)

    # -- selection marks (§4.2) ----------------------------------------------

    def mark_for(self, key: str) -> str:
        """The operator's mark for ``key``: ``sync``/``local-only``/``""``."""
        row = self.marks.get(key)
        return str(row.get("mark") or "") if isinstance(row, dict) else ""

    def record_mark(self, key: str, mark: str | None) -> None:
        """Set or clear one mark. ``None``/``""`` clears; anything else must be one of ``MARKS``."""
        if not mark:
            if self.marks.pop(key, None) is not None:
                self._dirty = True
            return
        if mark not in MARKS:
            raise ValueError(f"mark must be one of {', '.join(sorted(MARKS))} or absent")
        self.marks[key] = {"mark": mark, "set_at": time.time()}
        self._dirty = True

    def clear_applied(self, key: str) -> None:
        """Forget what this device held for ``key`` (the wipe's sidecar half).

        Safe by construction: a missing ``applied`` row only makes the next
        announce pull again, and the next contact re-announces anything still
        served — the same heal a dropped row gets at load.
        """
        if self.applied.pop(key, None) is not None:
            self._dirty = True

    def record_ack(
        self, device: str, key: str, *, gen: int, digest: str, at: float, wiped: bool = False
    ) -> None:
        """Record what a member holds — or, with ``wiped``, that it holds NOTHING.

        A late ack for an older generation is normally a reordering, not news —
        keeping it would regress the ledger and buy a pointless announce round.
        The exception (review round 1, Q-2) is the counter-reset edge: after the
        owner's generation row is lost, the member adopts the fresh counter DOWN
        to the number the owner now serves and acks it, and THAT ack carries the
        same digest as the row it supersedes. Refusing it (the old rule) left the
        ledger reading ``stale (gen 4 of 1)`` forever and re-announced every
        cycle; an equal digest is proof no value moved, so the ledger follows the
        member down. A lower gen with a DIFFERENT digest is still refused: that
        one can only be a reorder from before a value change, and the announce
        gate converges it.

        ``wiped`` is the wipe-ack form: the member deleted by provenance and is
        confirming the ENDING. It obeys the same monotonic rule (a late wipe-ack
        for an older generation must not overwrite a newer live copy that a
        re-share put back), and it is the only form allowed to REPLACE a live
        row: nothing is owed a member whose ledger says wiped until a re-share
        re-announces the key.
        """
        by_key = self.acks.setdefault(device, {})
        existing = by_key.get(key)
        if existing is not None and int(existing.get("gen") or 0) > int(gen):
            if str(existing.get("digest") or "") != str(digest or ""):
                return
        row: dict[str, Any] = {"gen": int(gen), "digest": digest, "at": float(at)}
        if wiped:
            row["wiped"] = True
        by_key[key] = row
        self._dirty = True

    # -- member-side accessors ----------------------------------------------

    def applied_for(self, key: str) -> dict[str, Any] | None:
        return self.applied.get(key)

    def record_applied(
        self,
        key: str,
        *,
        gen: int,
        digest: str,
        owner_device: str,
        row_id: int,
        at: float,
        record_id: str = "",
    ) -> None:
        self.applied[key] = {
            "gen": int(gen),
            "digest": digest,
            "owner_device": owner_device,
            "row_id": int(row_id),
            "record_id": str(record_id or "")[:128],
            "at": float(at),
        }
        self._dirty = True


@contextmanager
def mutate(network_id: str, root: Path | None = None) -> Iterator[SyncState]:
    """Read-modify-write the state under its lock (``placement.mutate``'s shape).

    Network I/O NEVER happens inside this block: every caller computes under the
    lock, saves, and dials after — the non-stall property (§5.4) would not
    survive a peer's link deadline being held across this document's lock.
    """
    path = sync_path(network_id, root)
    with _sync_lock(path):
        state = SyncState.load(network_id, root=root)
        yield state
        state.save()


def needs_names(root: Path | None = None) -> frozenset[str]:
    """The secret names this device's pushed bundles declare — "the keys to set".

    Read through ``mcpdefs.state_rows``, the design's own pointer (§4.2), so the
    copy-set default and the push cannot disagree about which references exist:
    every ``ref:<NAME>`` an MCP server row declares is a name the approved node
    will need to run that row. A device with no MCP rows reports the empty set,
    and a document that cannot be read is the empty set too — the CLOSED
    direction, because a wrong needs-list would copy MORE, never less.
    """
    try:
        from local_operator.network import mcpdefs

        if root is None:
            from local_operator.paths import config_dir

            root = config_dir()
        rows = mcpdefs.state_rows(root)
    except Exception:  # noqa: BLE001 — unreadable is empty, the safe direction
        logger.debug("credentials sync: needs-list read failed", exc_info=True)
        return frozenset()
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for ref in row.get("refs") or []:
            if isinstance(ref, Mapping):
                ref_id = str(ref.get("id") or "")
                if ref_id:
                    names.add(ref_id)
    return frozenset(names)


def secret_mark_default(mark: str, name: str, needs: frozenset[str]) -> bool:
    """§4.2's default for one class-2 key: ``sync`` marks win, else the needs-list.

    The three postures, in the operator's own vocabulary: ``sync`` — joins every
    approved device's copy-set; ``local-only`` — never crosses (callers drop it
    before defaults apply; this answers False for completeness); unmarked — in
    the set exactly when the pushed bundles declare it. "Everything else is
    offered, not copied" is the caller's half: the key still appears on the
    join list, with ``share`` false, and only the reduce step or a mark changes
    that.
    """
    if mark == MARK_SYNC:
        return True
    if mark == MARK_LOCAL_ONLY:
        return False
    return name in needs


def _ensure_generation(state: SyncState, key: str, meta: Mapping[str, Any]) -> int:
    """The current generation for ``key``, bumped when the value moved.

    The change signal is BOTH halves of what the design names: the row's
    ``updated_at`` (the cheap metadata read §5.2 describes) and the value digest
    (which sees a same-millisecond rewrite the stamp cannot). A metadata-only
    touch therefore announces once and converges without a transfer — the
    member's digest comparison adopts the counter (§ reset edge above).
    """
    digest = str(meta.get("digest") or "")
    updated_at = int(meta.get("updated_at") or 0)
    row = state.generation(key)
    if row is None:
        state.record_generation(key, 1, digest, updated_at)
        return 1
    if str(row.get("digest") or "") != digest or int(row.get("updated_at") or 0) != updated_at:
        bumped = int(row.get("gen") or 1) + 1
        if bumped > GEN_CEILING:
            bumped = GEN_CEILING
        state.record_generation(key, bumped, digest, updated_at)
        return bumped
    return int(row.get("gen") or 1)


def _ack_matches(acked: Mapping[str, Any] | None, gen: int, digest: str) -> bool:
    """Whether this member's ack already names exactly this generation+value.

    A WIPED row never matches: it says the member holds NOTHING, so the same
    generation+digest arriving after a re-share must announce and pull again —
    matching it would leave the re-shared copy silently undelivered forever.
    """
    if not isinstance(acked, Mapping) or acked.get("wiped") is True:
        return False
    return int(acked.get("gen") or 0) == int(gen) and str(acked.get("digest") or "") == digest


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

_SYNC_ATTR = "_mesh_credential_sync"
_SYNC_CREATE_LOCK = threading.Lock()


class SyncEngine:
    """One relay's sync machinery: the owner exchange and the member pulls.

    Built lazily per relay (``sync_for_relay``) and shared by the tick step, the
    relay's peer handler, and the broker's dispatch — one object, so the
    per-member in-flight guards are one set and two overlapping ticks cannot
    double-announce.
    """

    def __init__(
        self,
        server: Any,
        *,
        root: Path | None = None,
        self_device: str = "",
        self_device_name: str = "",
        audit: Any = None,
        auth_store_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._server = server
        self._root = root
        self._self_device = self_device
        self._self_device_name = self_device_name
        self._audit = audit
        #: A test seam, like the broker's ``github_minter``: the cross-root cells
        #: hand the member's store in explicitly, because ``AuthStore`` derives
        #: its database from the ambient config root and a unit cell often runs
        #: both devices in one process. Production leaves it ``None``.
        self._auth_store_factory = auth_store_factory
        self._loop: Any = None
        self._guard = threading.Lock()
        self._inflight_members: set[str] = set()
        self._pulling: set[tuple[str, str]] = set()

    # -- lazy pieces ---------------------------------------------------------

    def _broker_loop(self) -> Any:
        if self._loop is None:
            from local_operator.network.credentials.owner import _BrokerLoop

            self._loop = _BrokerLoop(self._self_device[-8:] or "sync")
        return self._loop

    def _store(self) -> Any:
        if self._auth_store_factory is not None:
            return self._auth_store_factory()
        from local_operator.providers.auth_store import AuthStore

        return AuthStore(config_dir=self._root)

    def _placement(self) -> Any:
        """This device's placement document, re-read per use (F2's discipline).

        Never cached: a share that arrives after the relay started, or a revoke
        made by a CLI process, must be seen by the very next exchange.
        """
        from local_operator.network.credentials import placement as placement_mod

        return placement_mod.PlacementDocument.resolve(self._root, self_device=self._self_device)

    def _audit_row(self, event: str, *, network_id: str = "", **fields: Any) -> None:
        """One audit record, best effort (the broker's own discipline).

        ``actor`` and ``network_id`` ride the row the way the owner's
        ``credential.copy`` sets them: a member-side row that leaves ``actor``
        at the dataclass default ``"self"`` reads as actor-less on the one
        surface an incident review reads (review round 1, N2).
        """
        if self._audit is None:
            return
        try:
            from local_operator.network.audit import AuditEvent

            self._audit.record(
                AuditEvent(
                    event=event,
                    network_id=network_id,
                    epoch=None,
                    actor=self._self_device,
                    actor_name=self._self_device_name,
                    actor_kind="device",
                    **fields,
                )
            )
        except Exception:  # noqa: BLE001 — a log that cannot be written is not a refusal
            pass

    # -- the owner side: announce --------------------------------------------

    def enqueue_owner_exchange(self, device_id: str) -> str:
        """Schedule one announce exchange for ``device_id``. NEVER blocks.

        The cheap gate runs synchronously (a placement read: does this device own
        ANY copy-eligible key this member holds?); the store reads and the dial
        run on the broker's loop. The per-member in-flight guard is what keeps a
        slow exchange from stacking behind the syncer's own floors — "no new
        floors" means the definitions cadence bounds this, and this bounds it
        again per member so a 30 s exchange cannot overlap the next tick.
        """
        if not device_id:
            return "in_sync"
        document = self._placement()
        if document is None:
            return "in_sync"
        owned = [
            key
            for key in document.keys_owned_by(self._self_device)
            if copies_by_class(str(document.entry(key).kind))
            and document.entry(key).is_holder(device_id)
        ]
        if not owned:
            # THE ENDING HALF OF THE GATE (§5.5c): a member whose grants were
            # all revoked still owes its wipes, and the owned-keys check above
            # would skip exactly that member. Read-only and small; the precise
            # answer is computed under the state lock in ``_pending_announces``.
            if not self._has_pending_wipes(document, device_id):
                return "in_sync"
        with self._guard:
            if device_id in self._inflight_members:
                return "in_flight"
            self._inflight_members.add(device_id)
        future = asyncio.run_coroutine_threadsafe(
            self._exchange_async(device_id), self._broker_loop().loop()
        )
        future.add_done_callback(lambda done: self._exchange_done(device_id, done))
        return "scheduled"

    async def _exchange_async(self, device_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._exchange_blocking, device_id)

    def _exchange_done(self, device_id: str, future: Any) -> None:
        with self._guard:
            self._inflight_members.discard(device_id)
        try:
            future.result()
        except Exception:  # noqa: BLE001 — a cadence step logs, never raises
            logger.debug("credentials sync: exchange for %s failed", device_id, exc_info=True)

    def _exchange_blocking(self, device_id: str) -> None:
        network_id, announces = self._pending_announces(device_id)
        if not announces:
            return
        link = self._server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
        if link is None:
            # Catch-up, never invalidate (§5.5a): the next contact recomputes.
            return
        for item in announces:
            frame = {
                "op": "net_broker",
                "kind": "announce",
                "network_id": network_id,
                "from_device": self._self_device,
                "from_device_name": self._self_device_name,
                "req": self._server._next_relay_req(),  # noqa: SLF001 — the relay's own counter
                **item,
            }
            try:
                reply = link.request(frame, timeout=ANNOUNCE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 — one member's link is that member's problem
                logger.debug("credentials sync: announce to %s failed", device_id, exc_info=True)
                return
            detail = reply.get("detail") if isinstance(reply, dict) else None
            if not isinstance(detail, dict) or str(detail.get("kind") or "") == "error":
                # A refusal is an answer, not a flap: leave it to the next tick,
                # whose floors the syncer already owns.
                logger.debug("credentials sync: announce to %s answered %s", device_id, detail)
                return
            if str(item.get("value_state") or "") == VALUE_STATE_ABSENT:
                self._record_wipe_reply(network_id, device_id, item, detail)

    def _record_wipe_reply(
        self,
        network_id: str,
        device_id: str,
        item: Mapping[str, Any],
        detail: Mapping[str, Any],
    ) -> None:
        """Record a wipe the member CONFIRMED on the announce's own reply.

        WHY THE REPLY IS THE ACK. An un-approved member cannot open a frame to
        its owner — the transport gates ``net_broker`` on the sender's
        ``broker_credential`` capability, which a deactivated member row no
        longer carries — so the one channel left is the ANSWER to the request
        this owner dialled, and the member answers it AFTER deleting. A reply
        that does not claim the wipe (a local delete that failed, an older
        build) is left to the next tick, which recomputes the notice.
        """
        if str(detail.get("action") or "") != "wiped":
            return
        key = _bounded_key(item.get("key"))
        gen = peer_int(item.get("gen"), maximum=GEN_CEILING)
        digest = _bounded_digest(item.get("digest"))
        if not key or gen <= 0 or not digest:
            return
        try:
            with mutate(network_id, self._root) as state:
                if state.ack_for(device_id, key) is not None:
                    state.record_ack(
                        device_id, key, gen=gen, digest=digest, at=time.time(), wiped=True
                    )
        except Exception:  # noqa: BLE001 — a lost confirmation is retried next tick
            logger.debug("credentials sync: wipe confirmation for %s lost", key, exc_info=True)

    def _pending_announces(self, device_id: str) -> tuple[str, list[dict[str, Any]]]:
        """The frames this member is owed, computed under the lock.

        TWO KINDS, AND THE ENDINGS GO FIRST. A WIPE notice (``value_state``
        ``absent``) is owed for every ledger row that is not already ``wiped``
        and whose key this owner will no longer serve that member — unshared,
        value deleted, marked ``local-only``, or the member stopped being
        active (§5.5c: the ending reaches a removed member, where a copy would
        be withheld; a wipe is bounded by the marker it deletes, never by the
        grant it outlives). While an ending is owed, positive announces wait —
        one revoked key must not be buried under a page of refreshes. Then,
        when no ending is owed: owned here, copy-eligible, held by this member,
        value readable, not ``local-only``, and the member's ack does not
        already name this generation+digest.
        """
        document = self._placement()
        if document is None:
            return "", []
        network_id = str(document.network_id)
        active = _member_is_active(self._root, network_id, device_id)
        with mutate(network_id, self._root) as state:
            wipes = self._pending_wipes(document, state, device_id, active=active)
            if wipes or not active:
                return network_id, wipes
            announces: list[dict[str, Any]] = []
            for key in document.keys_owned_by(self._self_device):
                entry = document.entry(key)
                if not copies_by_class(str(entry.kind)) or not entry.is_holder(device_id):
                    continue
                if state.mark_for(key) == MARK_LOCAL_ONLY:
                    # THE KILL SWITCH, enforced where the value would move: a
                    # grant that raced the mark (or predates it) cannot copy,
                    # and the ledger row it left behind turns into a wipe.
                    continue
                meta = self._read_value(key, entry)
                if meta is None:
                    continue
                gen = _ensure_generation(state, key, meta)
                digest = str(meta.get("digest") or "")
                if not digest or _ack_matches(state.ack_for(device_id, key), gen, digest):
                    continue
                announces.append(
                    {
                        "key": key,
                        "gen": gen,
                        "digest": digest,
                        "value_state": VALUE_STATE_PRESENT,
                    }
                )
                if len(announces) >= ANNOUNCE_CAP:
                    break
            return network_id, announces

    def _pending_wipes(
        self, document: Any, state: SyncState, device_id: str, *, active: bool
    ) -> list[dict[str, Any]]:
        """The wipe frames owed to ``device_id``, in key order, capped.

        A ledger row is owed a wipe unless this owner would still SERVE the key
        to this member right now: still owned here, still copy-eligible, still
        held, value readable, not marked ``local-only``. The serveability check
        costs one store read per candidate key, bounded by the ledger and only
        for keys the positive path is not already serving.
        """
        rows = state.acks.get(device_id) or {}
        wipes: list[dict[str, Any]] = []
        for key in sorted(rows):
            row = rows[key]
            if not isinstance(row, dict) or row.get("wiped"):
                continue
            digest = str(row.get("digest") or "")
            gen = int(row.get("gen") or 0)
            if gen <= 0 or not digest:
                continue
            if active:
                entry = document.entry(key)
                if (
                    entry is not None
                    and str(entry.owner_device) == self._self_device
                    and copies_by_class(str(entry.kind))
                    and entry.is_holder(device_id)
                    and state.mark_for(key) != MARK_LOCAL_ONLY
                    and self._read_value(key, entry) is not None
                ):
                    continue
            wipes.append(
                {"key": key, "gen": gen, "digest": digest, "value_state": VALUE_STATE_ABSENT}
            )
            if len(wipes) >= ANNOUNCE_CAP:
                break
        return wipes

    def _has_pending_wipes(self, document: Any, device_id: str) -> bool:
        """Cheap gate half: does this member's ledger owe an ENDING (§5.5c)?

        Any non-``wiped`` ledger row answers yes, because this is only reached
        when the owned-keys check found nothing to SERVE — every remaining row
        is by definition a key that cannot be served to this member anymore,
        and the precise per-key decision runs under the state lock in
        ``_pending_wipes``. A state that cannot be read answers no: an
        unreadable document must not spin a dial per member per tick.
        """
        try:
            state = SyncState.load(str(document.network_id), root=self._root)
        except Exception:  # noqa: BLE001 — unreadable state: nothing owed
            return False
        rows = state.acks.get(device_id) or {}
        for row in rows.values():
            if isinstance(row, dict) and not row.get("wiped"):
                return True
        return False

    def _read_value(self, key: str, entry: Any) -> dict[str, Any] | None:
        """Read the owner's value for a copy, via the owner's own store.

        The store object is opened per read and closed in ``finally``: the
        syncer thread is long-lived and must not leak a SQLite handle per tick.
        """
        try:
            store = self._store()
        except Exception:  # noqa: BLE001 — an unopenable store holds nothing
            return None
        try:
            return read_copy_value(store, key, entry, root=self._root)
        finally:
            _close_quietly(store)

    # -- the member side: announce in, apply, ack out ------------------------

    def on_announce(self, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Serve an ``announce`` (member side). Validates, schedules, replies.

        THE PULL IS NOT DONE HERE. This runs on the member's slow-op worker:
        scheduling the pull on the broker's loop and answering "noted" keeps a
        busy member from ever missing its own sync (§5.4 property 2), and a pull
        lost to a crash is recomputed on the next contact (§5.1) — nothing here
        is a queue that can be lost.
        """
        by, refusal = _authenticated_sender(link, frame)
        if refusal is not None:
            return refusal
        key = _bounded_key(frame.get("key"))
        gen = peer_int(frame.get("gen"), maximum=GEN_CEILING)
        digest = _bounded_digest(frame.get("digest"))
        value_state = str(frame.get("value_state") or VALUE_STATE_PRESENT)
        if not key or gen <= 0 or not digest:
            return _sync_error(
                "malformed_announce",
                key,
                "the announcement carried no usable key, generation or digest; "
                "nothing was pulled",
            )
        if gen >= GEN_SAFE_MAX:
            # An owner at the adoption bound cannot be followed (review round
            # 1, R1's member-side belt): adopting its number would pin THIS
            # device the moment the owner runs out of headroom.
            return _sync_error(
                "generation_out_of_range",
                key,
                "the announced generation is outside the range this device serves; "
                "nothing was pulled",
            )
        document = self._placement()
        if document is None:
            return _sync_error(
                "unknown_key",
                key,
                "this device holds no sharing list, so the announcement names nothing "
                "here; nothing was pulled",
            )
        entry = document.entry(key)
        if value_state == VALUE_STATE_ABSENT:
            # THE WIPE NOTICE (§5.5c, §4.3 as-built), handled BEFORE the holder
            # checks because it must be handleable AFTER the grant is gone —
            # that is the whole point of a wipe. The ownership check still
            # gates it when an entry exists (a device cannot order a patch on
            # what it does not own here); when no entry exists (a removed
            # member whose placement is gone), the deletion is bounded instead
            # BY THE MARKER: only records whose provenance names the
            # authenticated sender are ever touched.
            if entry is not None and str(entry.owner_device) != by:
                return _sync_error(
                    "not_owner",
                    key,
                    f"the wipe notice for {key!r} arrived from a device that does "
                    "not own it on this device; nothing was deleted",
                )
            return self._wipe_now(by, key, gen, digest)
        if value_state != VALUE_STATE_PRESENT:
            return _sync_error(
                "malformed_announce",
                key,
                "the announcement carried an unknown value_state; nothing was pulled",
            )
        if entry is None:
            return _sync_error(
                "unknown_key",
                key,
                f"this device's sharing list has no entry for {key!r}; nothing was pulled",
            )
        if str(entry.owner_device) != by:
            return _sync_error(
                "not_owner",
                key,
                f"the announcement for {key!r} arrived from a device that does not own "
                "it on this device; nothing was pulled",
            )
        if not entry.is_holder(self._self_device):
            return _sync_error(
                "not_a_holder",
                key,
                f"this device does not hold {key!r} in this network; nothing was pulled",
            )
        if not copies_by_class(str(entry.kind)):
            return _sync_error(
                "not_copyable",
                key,
                f"{key!r} is not a copy-eligible credential; it stays on the brokering "
                "path and nothing was pulled",
            )
        self._schedule_pull(by, key, gen, digest)
        return {"kind": "ack", "key": key, "action": "noted"}

    def _schedule_pull(self, owner: str, key: str, gen: int, digest: str) -> None:
        token = (owner, key)
        with self._guard:
            if token in self._pulling:
                return
            self._pulling.add(token)
        future = asyncio.run_coroutine_threadsafe(
            self._pull_async(owner, key, gen, digest), self._broker_loop().loop()
        )
        future.add_done_callback(lambda done: self._pull_done(token, done))

    async def _pull_async(self, owner: str, key: str, gen: int, digest: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pull_blocking, owner, key, gen, digest)

    def _pull_done(self, token: tuple[str, str], future: Any) -> None:
        with self._guard:
            self._pulling.discard(token)
        try:
            future.result()
        except Exception:  # noqa: BLE001 — a pull logs, never surfaces into a turn
            logger.debug("credentials sync: pull %s failed", token, exc_info=True)

    def _wipe_now(self, owner: str, key: str, gen: int, digest: str) -> dict[str, Any]:
        """Delete this device's copies for ``key``, INLINE, and answer the notice.

        BY PROVENANCE, NEVER BY TRUST IN THE FRAME: only rows whose marker names
        the AUTHENTICATED sender (``origin.owner_device == owner``) and this key
        are touched — a wipe can never reach records another owner wrote, and the
        scan stays findable even when the ``applied`` sidecar was lost, the same
        property the apply-sweep keeps from the other side.

        WHY INLINE, when the pull is scheduled off this worker: the confirmation
        must ride THIS reply. An un-approved member cannot open a frame — its
        ``broker_credential`` capability went with the grant — so a wipe-ack sent
        as a fresh request is refused at the owner's door (measured), and the
        answer to the request the owner dialled is the only channel left. The
        work is local and bounded (two store scans, no dial, no transfer), so the
        §5.4 property that matters here — never a reader thread, never a wait on
        the network — holds; only the delete of a few rows runs here.
        """
        removed_auth = self._wipe_auth_copies(owner, key)
        removed_secrets = self._wipe_secret_copies(owner, key)
        if removed_auth is None or removed_secrets is None:
            # A scan that could not run proves nothing: answer receipt-only and
            # let the owner's next contact recompute the notice.
            return {"kind": "ack", "key": key, "action": "noted"}
        document = self._placement()
        network_id = str(document.network_id) if document is not None else ""
        if network_id:
            with mutate(network_id, self._root) as state:
                state.clear_applied(key)
        self._audit_row(
            "credential.copy_wiped",
            network_id=network_id,
            subject=owner,
            detail={
                "credential_key": key,
                "act": self._self_device,
                "sub": owner,
                "rows": removed_auth + removed_secrets,
            },
        )
        return {
            "kind": "ack",
            "key": key,
            "action": "wiped",
            "gen": gen,
            "digest": digest,
            "value_state": VALUE_STATE_ABSENT,
        }

    def _wipe_auth_copies(self, owner: str, key: str) -> int | None:
        """Delete this device's class-4 copies of ``key`` from ``owner``.

        ``None`` means the scan itself could not run (an unopenable store), and
        the caller must not claim a wipe on it; ``0`` means it ran and found
        nothing, which IS a wipe (nothing held is the ending asked for).
        """
        try:
            store = self._store()
        except Exception:  # noqa: BLE001 — an unopenable store proves nothing
            return None
        removed = 0
        try:
            for row in store.list_credentials(None, include_disabled=True):
                origin = getattr(row, "data", {}).get(MESH_ORIGIN_KEY)
                if not isinstance(origin, dict):
                    continue
                if str(origin.get("key") or "") != key:
                    continue
                if str(origin.get("owner_device") or "") != owner:
                    continue
                try:
                    store.delete_credential(int(getattr(row, "id", 0) or 0))
                    removed += 1
                except Exception:  # noqa: BLE001 — one row must not stop the rest
                    logger.debug("credentials sync: wipe of an auth row failed", exc_info=True)
            return removed
        except Exception:  # noqa: BLE001 — a failed scan claims nothing
            logger.debug("credentials sync: auth wipe scan failed", exc_info=True)
            return None
        finally:
            _close_quietly(store)

    def _wipe_secret_copies(self, owner: str, key: str) -> int | None:
        """Delete this device's class-2 copies of ``key`` from ``owner``.

        Same ``None`` contract as the class-4 half. A device with NO secret
        store answers 0 without opening one: it holds nothing, and a read must
        not be the reason a store appears (the announce reader's own rule).
        """
        try:
            from local_operator.secrets.keys import store_path

            if not store_path(self._root).exists():
                return 0
        except Exception:  # noqa: BLE001 — cannot tell: claim nothing
            return None
        try:
            from local_operator.secrets import access

            store = access.open_store(self._root)
        except Exception:  # noqa: BLE001 — no store openable: claim nothing
            return None
        removed = 0
        try:
            for record in store.list():
                origin = getattr(record, "origin", None)
                if not isinstance(origin, dict):
                    continue
                if str(origin.get("key") or "") != key:
                    continue
                if str(origin.get("owner_device") or "") != owner:
                    continue
                try:
                    store.delete(record.name)
                    removed += 1
                except Exception:  # noqa: BLE001 — one row must not stop the rest
                    logger.debug("credentials sync: wipe of a secret failed", exc_info=True)
            return removed
        except Exception:  # noqa: BLE001 — a failed scan claims nothing
            logger.debug("credentials sync: secret wipe scan failed", exc_info=True)
            return None
        finally:
            _close_quietly(store)

    def _pull_blocking(
        self, owner: str, key: str, announced_gen: int, announced_digest: str
    ) -> None:
        """One key's pull/apply/ack, off every responder thread."""
        document = self._placement()
        if document is None:
            return
        entry = document.entry(key)
        if (
            entry is None
            or str(entry.owner_device) != owner
            or not entry.is_holder(self._self_device)
            or not copies_by_class(str(entry.kind))
        ):
            return
        state = SyncState.load(document.network_id, root=self._root)
        held = state.applied_for(key)
        if held is not None and str(held.get("owner_device") or "") != owner:
            # A DIFFERENT owner's copy of the same key name: not this entry's
            # history. (Two owners for one key cannot happen in one placement
            # document; this guards a rotated owner row mid-flight.)
            held = None
        held_gen = int(held.get("gen") or 0) if held else 0
        held_digest = str(held.get("digest") or "") if held else ""
        if held_digest and held_digest == announced_digest:
            # THE SAME VALUE: adopt the owner's counter (up OR down — the digest
            # proves no value moves) and confirm. This is the reset edge's cheap
            # path and the ordinary "metadata-only touch" path at once.
            if announced_gen != held_gen:
                with mutate(document.network_id, self._root) as editable:
                    editable.record_applied(
                        key,
                        gen=announced_gen,
                        digest=announced_digest,
                        owner_device=owner,
                        row_id=int(held.get("row_id") or 0) if held else 0,
                        at=time.time(),
                    )
            self._send_ack(document.network_id, owner, key, announced_gen, announced_digest)
            return
        link = self._server._ensure_link(owner)  # noqa: SLF001 — the one dial seam
        if link is None:
            return
        frame = {
            "op": "net_broker",
            "kind": "copy",
            "network_id": document.network_id,
            "from_device": self._self_device,
            "from_device_name": self._self_device_name,
            "req": self._server._next_relay_req(),  # noqa: SLF001
            "key": key,
            "gen": announced_gen,
            "held": held_gen,
        }
        try:
            reply = link.request(frame, timeout=COPY_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — recomputed on the next contact
            logger.debug("credentials sync: copy request for %s failed", key, exc_info=True)
            return
        detail = reply.get("detail") if isinstance(reply, dict) else None
        if not isinstance(detail, dict) or str(detail.get("kind") or "") != "copy":
            logger.debug("credentials sync: copy for %s answered %s", key, detail)
            return
        if _bounded_key(detail.get("key")) != key:
            return
        value = detail.get("value")
        reply_digest = _bounded_digest(detail.get("digest"))
        if not isinstance(value, dict) or not reply_digest:
            return
        if fingerprint(value) != reply_digest:
            # INTEGRITY: the bytes do not match the digest that pins them. The
            # link is authenticated, so this is a build bug or a corrupted
            # payload — either way, nothing is written and the row says why.
            self._audit_row(
                "credential.copy_refused",
                network_id=document.network_id,
                subject=owner,
                detail={
                    "credential_key": key,
                    "act": self._self_device,
                    "sub": owner,
                    "reason": "digest_mismatch",
                },
            )
            return
        if str(detail.get("value_state") or VALUE_STATE_PRESENT) != VALUE_STATE_PRESENT:
            return
        served = peer_int(detail.get("gen"), maximum=GEN_CEILING)
        if served >= GEN_SAFE_MAX:
            # AN OWNER AT THE BOUND CANNOT BE FOLLOWED (review round 1, R1's
            # member-side belt): applying its number would pin this device, so
            # the reply is dropped whole.
            logger.debug(
                "credentials sync: %s served an out-of-range generation for %s", owner, key
            )
            return
        if served <= held_gen:
            # NEVER A VALUE ROLLBACK (§5.2's monotonicity rule): a generation at
            # or below what this device holds is dropped however it arrived.
            return
        provenance = detail.get("provenance")
        if not isinstance(provenance, Mapping):
            provenance = None
        applied = self._apply_value(key, entry, value, owner, served, provenance=provenance)
        if applied is None:
            # NOTHING WAS WRITTEN, SO NOTHING IS HELD: a failed apply must not
            # be recorded or acked — the owner would stop announcing a value
            # this device never received, and the member would sit silently
            # behind (§5.1's recompute-on-contact is the repair, but only if
            # the ack was never sent).
            return
        row_id, record_id = applied
        with mutate(document.network_id, self._root) as editable:
            editable.record_applied(
                key,
                gen=served,
                digest=reply_digest,
                owner_device=owner,
                row_id=row_id,
                record_id=record_id,
                at=time.time(),
            )
        self._audit_row(
            "credential.copy_applied",
            network_id=document.network_id,
            subject=owner,
            detail={"credential_key": key, "act": self._self_device, "sub": owner, "gen": served},
        )
        self._send_ack(document.network_id, owner, key, served, reply_digest)

    def _apply_value(
        self,
        key: str,
        entry: Any,
        value: Mapping[str, Any],
        owner: str,
        gen: int,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> tuple[int, str] | None:
        """Write a received copy into THIS device's own store.

        Returns ``(row_id, record_id)`` — the class-4 row's integer id or the
        class-2 record's string id, one slot each — or ``None`` when nothing
        was written. ``None`` is load-bearing at the caller: a failed apply
        must not be recorded or acked as held, or the owner stops announcing a
        value this device never received; a not-applied pull is recomputed on
        the next contact (§5.1: nothing here is a queue that can be lost).

        ORDER, AND WHY: the new value is upserted FIRST and any superseded row
        is swept after, so a reader racing the apply sees the old value or the
        new one — never neither (§5.4 property 3). The sweep — and the wipe —
        find previous copies by the ORIGIN MARKER, not by the sidecar alone, so
        a lost sync document cannot accumulate rows on the member.

        CLASS 4 (``api-key-static``): the received payload IS a row's ``data``
        (the origin marker rides in it), so the write is the store's ordinary
        credential upsert — the same 0600 row a local login writes.

        CLASS 2 (``store-secret``): the value is re-sealed into THIS device's
        own encrypted store under ITS OWN master key, with the provenance
        marker inside the sealed payload — never the owner's key, never a
        plaintext file (§8.2's copy invariant). An existing record under the
        name is UPDATED: the default copy-set is the needs-list — keys the node
        is missing — so colliding with a LOCAL value is the operator's
        deliberate force-add, and "this device's value for this key" is what
        was asked for; refusing instead would leave a key the owner keeps
        announcing permanently un-acked.
        """
        if str(entry.kind) == SECRET_KIND:
            return self._apply_secret_value(key, value, owner, gen, provenance)
        if str(entry.kind) != "api-key-static":
            return None
        store = self._store()
        try:
            provider = str(entry.provider or key)
            payload = _copyable_payload(value)
            if payload is None:
                return None
            payload[MESH_ORIGIN_KEY] = {
                "owner_device": owner,
                "key": key,
                "gen": int(gen),
                "applied_at": time.time(),
            }
            new_row = store.upsert_credential(provider, payload)
            new_id = int(getattr(new_row, "id", 0) or 0)
            for row in store.list_credentials(provider, include_disabled=True):
                if int(getattr(row, "id", 0) or 0) == new_id:
                    continue
                origin = getattr(row, "data", {}).get(MESH_ORIGIN_KEY)
                if not isinstance(origin, dict):
                    continue
                if str(origin.get("key") or "") != key:
                    continue
                if str(origin.get("owner_device") or "") != owner:
                    continue
                try:
                    store.delete_credential(int(row.id))
                except Exception:  # noqa: BLE001 — a failed sweep must not fail the apply
                    logger.debug("credentials sync: sweep of row %s failed", row.id, exc_info=True)
            return new_id, ""
        except Exception:  # noqa: BLE001 — a failed apply is retried on the next contact
            logger.debug("credentials sync: apply for %s failed", key, exc_info=True)
            return None
        finally:
            _close_quietly(store)

    def _apply_secret_value(
        self,
        key: str,
        value: Mapping[str, Any],
        owner: str,
        gen: int,
        provenance: Mapping[str, Any] | None,
    ) -> tuple[int, str] | None:
        """The class-2 half of :func:`_apply_value`: re-seal into ``secrets/``.

        The store is opened WITH ``create``: unlike the owner's announce read
        (which must never be the reason a store appears), this call exists
        because a copy is arriving for exactly this device, and refusing to
        create the store would make the first copy of a node's life
        unappliable forever. What is NOT created is any derivation of the value
        outside the sealed store: the bytes go from the link into ``secrets/``
        in one write, under this device's own master key.
        """
        try:
            name = secret_name_from_key(key)
        except ValueError:
            return None
        if str(value.get("name") or "") != name:
            # The payload must name the key it claims to be: defence in depth
            # under the digest pin — a reply mis-addressed at either end is
            # dropped whole rather than written under the wrong name.
            return None
        raw = value.get("value")
        if not isinstance(raw, str) or not raw or len(raw) > _SECRET_HEX_MAX:
            return None
        try:
            material = bytes.fromhex(raw)
        except ValueError:
            return None
        description = str(value.get("description") or "")
        origin: dict[str, Any] = {
            "owner_device": owner,
            "key": key,
            "gen": int(gen),
            "applied_at": time.time(),
        }
        owner_name = str((provenance or {}).get("owner_device_name") or "")
        if owner_name:
            origin["owner_device_name"] = owner_name
        try:
            from local_operator.secrets import access
            from local_operator.secrets.errors import SecretNotFound, SecretStoreError

            store = access.open_store(self._root, create=True)
        except Exception:  # noqa: BLE001 — no store openable: nothing is written
            logger.debug("credentials sync: secret store for apply failed", exc_info=True)
            return None
        try:
            try:
                store.describe(name)
                exists = True
            except SecretNotFound:
                exists = False
            except SecretStoreError:
                # NO STORE ON DISK YET is not an error for a WRITE path: the first
                # copy of a node's life is exactly this state, and the ``set``
                # below is what creates the store. Any OTHER read failure (a
                # damaged row, key trouble) re-raises — a blind ``set`` over an
                # unreadable record would destroy what it cannot read.
                from local_operator.secrets.keys import store_path

                if store_path(self._root).exists():
                    raise
                exists = False
            if exists:
                record = store.update(name, material, origin=origin)
            else:
                record = store.set(name, material, description=description, origin=origin)
            return 0, str(getattr(record, "record_id", "") or "")
        except Exception:  # noqa: BLE001 — a failed apply is retried on the next contact
            logger.debug("credentials sync: secret apply for %s failed", key, exc_info=True)
            return None
        finally:
            _close_quietly(store)

    def _send_ack(self, network_id: str, owner: str, key: str, gen: int, digest: str) -> None:
        """Tell the owner what this device holds now. Best effort, never blocking."""
        if gen <= 0 or not digest:
            return
        link = self._server._ensure_link(owner)  # noqa: SLF001 — the one dial seam
        if link is None:
            return
        frame = {
            "op": "net_broker",
            "kind": "ack",
            "network_id": network_id,
            "from_device": self._self_device,
            "from_device_name": self._self_device_name,
            "req": self._server._next_relay_req(),  # noqa: SLF001
            "key": key,
            "gen": gen,
            "digest": digest,
        }
        try:
            link.request(frame, timeout=ANNOUNCE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — the next announce recomputes the need
            logger.debug("credentials sync: ack for %s failed", key, exc_info=True)


def sync_for_relay(server: Any) -> SyncEngine | None:
    """The engine for ``server``, built on first need and kept — or ``None``.

    ``None`` when the relay has no identity or no root: nothing can be synced
    from a device that does not know who it is, and the member-side announce
    handler answers such a frame with its own error rather than crashing a
    reader.
    """
    if server is None:
        return None
    root = getattr(server, "root", None)
    identity = getattr(server, "identity", None)
    self_device = str(getattr(identity, "device_id", "") or "")
    if root is None or not self_device:
        return None
    with _SYNC_CREATE_LOCK:
        engine = getattr(server, _SYNC_ATTR, None)
        if engine is not None:
            return engine
        engine = SyncEngine(
            server,
            root=root,
            self_device=self_device,
            self_device_name=str(getattr(identity, "name", "") or ""),
            audit=getattr(server, "audit", None),
        )
        try:
            setattr(server, _SYNC_ATTR, engine)
        except Exception:  # noqa: BLE001 — a server that refuses attributes pays a rebuild
            pass
        return engine


# ---------------------------------------------------------------------------
# The tick step (registered on the definitions syncer)
# ---------------------------------------------------------------------------


def credentials_sync_step(step_server: "RelayServer", device_id: str) -> str | None:
    """One member's credential sync, run INSIDE the mesh-definitions tick.

    Registered with ``definitions.add_tick_step`` (declare the registration in
    ``credentials.install``), so this cadence rides the existing thread and its
    floors — 15 s tick, 60 s to a reachable member, failures retried next tick —
    exactly as ``mcpdefs.mesh_tick_step`` does, and for the same reason: a
    second thread for a shared question is the seam comment's own argument
    against itself. The capability check is the same one every sibling cadence
    makes (``net_broker`` requires ``broker_credential``): a member that cannot
    hold the op is skipped with a named outcome and NO wire traffic.
    """
    server = step_server
    document_engine = sync_for_relay(server)
    if document_engine is None:
        return None
    from local_operator.network import definitions

    blocked = definitions.unholdable_capability(server, device_id, "net_broker")
    if blocked:
        return f"skipped:no_{blocked}"
    return document_engine.enqueue_owner_exchange(device_id)


def install() -> None:
    """Register this slice's cadence on the definitions syncer's ONE seam.

    Idempotent by identity (``definitions.add_tick_step``), and that is
    load-bearing rather than tidy: ``install`` runs at relay CONSTRUCTION and
    the suite builds hundreds of relays that never tick. Called from
    ``credentials.install``, so the step exists exactly where the ops do.
    """
    from local_operator.network import definitions

    definitions.add_tick_step(credentials_sync_step)


# ---------------------------------------------------------------------------
# Owner-side handlers (called from the broker's dispatch)
# ---------------------------------------------------------------------------


def _sync_error(code: str, key: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "key": key, "message": message}


def owner_copy(broker: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Serve a ``copy`` request (member -> owner). Returns a ``detail`` object.

    THE AUTHORISATION IS THE BROKER'S OWN CHECK, not a parallel one (§8.3:
    ``copy_requires_active_holder``): the entry exists, it is owned HERE, and
    the sender is a HOLDER — plus the one belt the copy path adds, that the
    holder is an ACTIVE member, so a removed member is withheld a copy however
    the frame reached the handler.

    The served generation is ``max(current, held + 1)``: normally the current
    generation, which is strictly greater than the member's held one; after an
    owner-side state loss (the member is AHEAD of our counter), the member's own
    number + 1, adopted into the document so both sides converge forward. A
    member never applies at or below what it holds, so this is the one shape
    that lets a reset heal without any value ever moving backward.
    """
    key = _bounded_key(frame.get("key"))
    if not key:
        return BrokerError(
            code="malformed_frame", message="the copy request named no key"
        ).to_detail()
    by, refused = broker._caller(link, frame, key, key)
    if refused is not None:
        return refused
    requested = peer_int(frame.get("gen"), maximum=GEN_CEILING)
    held = peer_int(frame.get("held"), maximum=GEN_CEILING)
    if held >= GEN_SAFE_MAX:
        # THE ADOPTION BOUND (review round 1, R1): adopting ``held + 1`` at the
        # ceiling would freeze this key's generation forever — bumps clamp and
        # the member drops every reply. A number this high is a forgery or a
        # corrupt member state, never an honest counter, so it is refused by
        # name and NOTHING is recorded from it. (The member heals itself: its
        # own load drops the corrupt row and it re-pulls from zero.)
        return broker._refuse(
            link,
            key,
            key,
            by,
            "generation_out_of_range",
            "the generation this request asked to continue from is not one this "
            "device will adopt; nothing was copied",
        )
    entry = broker._entry(key)  # noqa: SLF001 — the broker's own document read
    if (
        entry is None
        or str(entry.owner_device) != str(broker.self_device)
        or not entry.is_holder(by)
    ):
        return broker._refuse(
            link,
            key,
            key,
            by,
            "not_a_holder",
            f"this device does not own {key!r} for that member; nothing was copied",
        )
    if not copies_by_class(str(entry.kind)):
        return broker._refuse(
            link,
            key,
            key,
            by,
            "not_copyable",
            f"{key!r} is not a copy-eligible credential; it stays on the brokering "
            "path and nothing was copied",
        )
    if not _member_is_active(broker.root, broker.network_id, by):
        return broker._refuse(
            link,
            key,
            key,
            by,
            "member_not_active",
            f"{by} is not an active member of this network; nothing was copied",
        )
    with mutate(str(broker.network_id), broker.root) as state:
        if state.mark_for(key) == MARK_LOCAL_ONLY:
            # §4.2's kill switch, enforced where the value would move: the mark
            # can land after a grant was written, and a grant must not outrun
            # it. The ledger row this leave behind turns into a wipe on the
            # next pass, so an already-held copy ends too.
            return broker._refuse(
                link,
                key,
                key,
                by,
                "local_only",
                f"{key!r} is marked local-only on this device, so it never crosses; "
                "nothing was copied",
            )
    meta = None
    try:
        meta = read_copy_value(broker._auth_store_instance(), key, entry, root=broker.root)
    except Exception:  # noqa: BLE001 — an unreadable store holds nothing to copy
        logger.debug("credentials sync: owner read for %s failed", key, exc_info=True)
    if meta is None:
        return broker._refuse(
            link,
            key,
            key,
            by,
            "no_value",
            f"this device has no copyable value for {key!r} right now; nothing was copied",
        )
    served = _served_generation(
        broker.root, broker.network_id, key, meta, requested=requested, held=held
    )
    broker._audit(
        "credential.copy",
        actor=broker.self_device,
        subject=by,
        detail={
            "credential_key": key,
            "act": broker.self_device,
            "sub": by,
            "gen": served,
        },
    )
    return {
        "kind": "copy",
        "key": key,
        "gen": served,
        "digest": str(meta.get("digest") or ""),
        "value_state": VALUE_STATE_PRESENT,
        "value": meta.get("value"),
        "provenance": {
            "owner_device": broker.self_device,
            "owner_device_name": broker.self_device_name,
        },
    }


def _served_generation(
    root: Path | None,
    network_id: str,
    key: str,
    meta: Mapping[str, Any],
    *,
    requested: int,
    held: int,
) -> int:
    """The generation a copy reply carries, persisted so the ledger can converge.

    ``requested`` names the generation the member saw announced; ``held`` is
    what the member said it holds. The rule (module docstring's reset edge):
    serve the current generation when it is strictly ahead of ``held``, else
    adopt ``held + 1`` — evidence of the member's number moving the counter
    FORWARD, never backward, and never a value change by itself.
    """
    del requested  # Named for the wire's sake; `held` is the strictly safer term.
    with mutate(network_id, root) as state:
        current = _ensure_generation(state, key, meta)
        served = current if current > held else held + 1
        if served > GEN_CEILING:
            served = GEN_CEILING
        if served != current:
            state.record_generation(
                key, served, str(meta.get("digest") or ""), int(meta.get("updated_at") or 0)
            )
        return served


def owner_ack(broker: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Record an ``ack`` (member -> owner). Idempotent; never blocks anything.

    TWO ACKS ARRIVE HERE, under DIFFERENT authority. A PRESENT ack claims a
    held copy: only a current holder of a key this device owns is recorded —
    the ledger's job is not to collect assertions. An ABSENT ack confirms a
    WIPE (§5.5c): it necessarily arrives after the grant was removed, so the
    holder check cannot apply; the ledger entry the wipe was sent for is the
    authority it is recorded against, and a wipe-ack with no such row (a lost
    owner state) is receipt-only — the next announce pass recomputes the need.
    """
    key = _bounded_key(frame.get("key"))
    if not key:
        return BrokerError(code="malformed_frame", message="the ack named no key").to_detail()
    by, refused = broker._caller(link, frame, key, key)
    if refused is not None:
        return refused
    gen = peer_int(frame.get("gen"), maximum=GEN_CEILING)
    digest = _bounded_digest(frame.get("digest"))
    value_state = str(frame.get("value_state") or VALUE_STATE_PRESENT)
    entry = broker._entry(key)  # noqa: SLF001 — the broker's own document read
    owned_here = entry is not None and str(entry.owner_device) == str(broker.self_device)
    if value_state == VALUE_STATE_ABSENT:
        if owned_here and gen > 0 and digest:
            with mutate(broker.network_id, broker.root) as state:
                if state.ack_for(by, key) is not None:
                    state.record_ack(by, key, gen=gen, digest=digest, at=time.time(), wiped=True)
        return {"kind": "ack", "key": key, "action": "recorded"}
    if value_state != VALUE_STATE_PRESENT:
        return BrokerError(
            code="malformed_frame", message="the ack carried an unknown value_state"
        ).to_detail()
    if owned_here and entry is not None and entry.is_holder(by):
        if gen > 0 and digest:
            with mutate(broker.network_id, broker.root) as state:
                state.record_ack(by, key, gen=gen, digest=digest, at=time.time())
    return {"kind": "ack", "key": key, "action": "recorded"}


def member_announce(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Serve an ``announce`` (owner -> member). The member-side entry point.

    Reached from TWO places, because a holder may or may not own keys itself:
    ``relay_handler``'s pre-broker branch (a device that owns nothing serves no
    ``net_broker`` — but it can still be a HOLDER, and the announce is addressed
    to exactly that device) and ``MeshCredentialBroker.on_broker`` (a device
    that owns something runs both roles on one handler).
    """
    engine = sync_for_relay(server)
    if engine is None:
        return _sync_error(
            "unavailable",
            "",
            "this relay has no identity to sync credentials with; nothing was pulled",
        )
    return engine.on_announce(link, frame)


# ---------------------------------------------------------------------------
# The staleness surface (shared by `lop network credentials` and `doctor`)
# ---------------------------------------------------------------------------


def sync_segment(*, acked: Mapping[str, Any] | None, current: Mapping[str, Any] | None) -> str:
    """One member's copy state, in the vocabulary §5.2 fixes.

    ``synced (gen 7)`` / ``stale (gen 6 of 7, last acked 14:02)`` — the two forms
    the design names, plus ``not yet synced (gen 7)`` for a holder that has
    never acked (a fresh share, before its first pull). ``""`` when there is no
    generation to speak of: the segment is ABSENT, never "failed", for a member
    whose copy relationship has no state yet (the rollout segment's own rule).
    """
    if not isinstance(current, Mapping):
        return ""
    gen = peer_int(current.get("gen"), maximum=GEN_CEILING)
    if gen <= 0:
        return ""
    if acked is not None and acked.get("wiped") is True:
        # THE ENDING FORM (§4.3 as-built): what this member holds now is
        # nothing, confirmed by its own delete-ack. It outranks the counter
        # comparison — nothing is stale when nothing is held — and a re-share
        # clears it back to the ordinary forms by announcing again.
        return f"wiped (gen {gen})"
    if acked is None:
        return f"not yet synced (gen {gen})"
    acked_gen = peer_int(acked.get("gen"), maximum=GEN_CEILING)
    if acked_gen == gen and str(acked.get("digest") or "") == str(current.get("digest") or ""):
        return f"synced (gen {gen})"
    at = peer_int(acked.get("at"), maximum=GEN_CEILING)
    stamp = time.strftime("%H:%M", time.localtime(at)) if at else ""
    tail = f", last acked {stamp}" if stamp else ""
    return f"stale (gen {acked_gen} of {gen}{tail})"


def sync_checks(record: Any, *, root: Path | None = None) -> list[dict[str, Any]]:
    """The ``credential_sync`` check rows for ONE network (doctor's derivation).

    One row per (copy-eligible owned key, active holder) that has a generation —
    read from the state document only, so a doctor run never dials, never reads
    a store, and never mutates. ``ok`` is the segment's own answer: synced rows
    pass; stale/pending rows FAIL with a remedy that says the true thing — this
    self-heals on the member's next contact, and NOTHING is blocked meanwhile
    (§5.5b: staleness never blocks a session).
    """
    from local_operator.network.credentials import placement as placement_mod

    state = SyncState.load(str(record.network_id), root=root)
    document = placement_mod.PlacementDocument.load(
        str(record.network_id), root=root, self_device=str(record.self_device_id)
    )
    rows: list[dict[str, Any]] = []
    for key in document.keys_owned_by(str(record.self_device_id)):
        entry = document.entry(key)
        if entry is None:
            continue
        if not copies_by_class(str(entry.kind)):
            continue
        current = state.generation(key)
        if current is None:
            continue
        for holder in entry.holders:
            if str(holder.device) == str(record.self_device_id):
                continue
            member = record.member(str(holder.device))
            if member is None or not bool(getattr(member, "active", False)):
                continue
            name = str(getattr(member, "name", "") or "")
            acked = state.ack_for(str(holder.device), key)
            if isinstance(acked, Mapping) and acked.get("wiped") is True:
                # A confirmed ending owes nothing: the row would read "wiped"
                # and pass, which is noise, not a check. A re-share re-announces
                # and the row returns on its own.
                continue
            segment = sync_segment(acked=acked, current=current)
            if not segment:
                continue
            synced = segment.startswith("synced")
            row: dict[str, Any] = {
                "check": "credential_sync",
                "network_id": str(record.network_id),
                "credential_name": key,
                "device_id": str(holder.device),
                "device_name": name,
                "ok": synced,
                "detail": segment,
            }
            if not synced:
                row["remedies"] = [
                    f"{name or holder.device} pulls the current value on its next "
                    "contact; nothing is blocked meanwhile"
                ]
            rows.append(row)
    return rows


def _close_quietly(store: Any) -> None:
    try:
        store.close()
    except Exception:  # noqa: BLE001 — a failed close must not fail the work it followed
        pass
