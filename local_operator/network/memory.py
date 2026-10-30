"""Incremental ALL-scope memory sync over existing receiver-admin peer links.

No grant/trust/enrollment changes. A policy-filtered stream carries bounded
sequence-only skips for excluded scopes; scoped cursor epochs prevent a later
scope expansion from losing old rows behind an already-advanced global vector.
Skipped sequence numbers disclose neither content nor owner IDs. Tombstone proofs
travel before redacted historical puts, so forgetting is order independent.

Tombstones are never garbage-collected in v1: the hard owner cap refuses writes
instead of discarding deletion/conflict history. Hence no unsafe offline GC
horizon exists; an arbitrarily old active member can still catch up incrementally.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from local_operator.compaction.tokens import count_text_tokens
from local_operator.memory import (
    DEFAULT_EXCLUDED_DEVICES,
    LEGACY_ITEM_WIRE_TOKENS,
    MAX_BATCH_BYTES,
    SCOPES,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    _dump,
    safe_segment,
    validate_op,
)
from local_operator.memory_capture import MEMORY_DEFAULT_ENABLED, policy
from local_operator.network import definitions
from local_operator.network import store as network_store
from local_operator.network import wire
from local_operator.network.types import MeshRefusal


def settings(root: Path) -> tuple[bool, list[str], list[str]]:
    from local_operator.config import read_config_values

    values = read_config_values(root) or {}
    network = values.get("network", {})
    memory = network.get("memory", {}) if isinstance(network, dict) else {}
    if not isinstance(memory, dict):
        raise MemoryError("Invalid network.memory policy.")
    scopes = memory.get("scopes", list(SCOPES))
    excluded = memory.get("exclude_devices", list(DEFAULT_EXCLUDED_DEVICES))
    if not isinstance(scopes, list) or len(scopes) > 4 or any(s not in SCOPES for s in scopes):
        raise MemoryError("Memory scopes must be an explicit subset of the four scopes.")
    if not isinstance(excluded, list) or len(excluded) > 256:
        raise MemoryError("Memory peer exclusions must be bounded device IDs.")
    for device in excluded:
        # Exclusion entries are device ids (``d_`` + hex), i.e. structural:
        # a hex tail can collide with the IBAN/card shapes, so they are
        # checked for the safe-segment shape, not the free-text guard.
        safe_segment(device, structural=True)
    enabled = memory.get("enabled", MEMORY_DEFAULT_ENABLED) is True and policy(root)["enabled"]
    return enabled, sorted(set(scopes)), excluded


def _store(root: Path) -> MemoryStore:
    return MemoryStore(root, MemoryIdentity())


def _epoch(scopes: list[str]) -> str:
    return hashlib.sha256(_dump(sorted(scopes)).encode()).hexdigest()[:16]


def _key(peer: str, scopes: list[str]) -> str:
    return "memory-cursor:" + safe_segment(peer, structural=True) + ":" + _epoch(scopes)


def cursor(
    memory: MemoryStore, peer: str, scopes: list[str], generation: int = 0
) -> dict[str, int]:
    if not memory.path.exists():
        return {}
    with memory.connection() as db:
        row = db.execute("SELECT value FROM meta WHERE key=?", (_key(peer, scopes),)).fetchone()
        if not row:
            return {}
        value = json.loads(row[0])
        return value.get("vector", {}) if value.get("generation") == generation else {}


def generation(memory: MemoryStore) -> int:
    if not memory.path.exists():
        return 0
    with memory.connection() as db:
        row = db.execute("SELECT value FROM meta WHERE key='stream-generation'").fetchone()
        return int(row[0]) if row else 0


def _checked_vector(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or len(value) > 256:
        raise MemoryError("Invalid bounded memory vector.")
    for origin, seq in value.items():
        # Vector origins are device ids (``d_`` + hex): structural, so a hex
        # tail that happens to satisfy the IBAN/card shape must not refuse an
        # otherwise valid sync vector (the permanent-refusal class fixed in
        # MemoryStore.put; see safe_segment).
        safe_segment(origin, structural=True)
        if type(seq) is not int or not 0 <= seq <= 2**63 - 1:
            raise MemoryError("Invalid memory sequence vector.")
    return value


class _LegacyWithheld(MemoryError):
    """An op skipped only because the peer predates ``memory-item-512``."""


def batch(
    memory: MemoryStore,
    vector: dict[str, int],
    scopes: list[str],
    origins: set[str],
    source_generation: int = 0,
    withheld: list[str] | None = None,
    *,
    legacy_item_cap: bool = False,
    legacy_withheld: list[str] | None = None,
) -> dict[str, Any]:
    """Bounded operations and content-free skipped ranges, never guessed bodies.

    ``legacy_item_cap`` is set for a peer that did not advertise
    ``wire.MEMORY_ITEM_512``: its validator still refuses a fact over
    ``LEGACY_ITEM_WIRE_TOKENS`` wire tokens, and one refused op fails the whole
    batch and strands that peer behind a sequence gap. Such a fact is withheld
    as a content-free skip instead — the same channel an unverifiable row uses —
    and is reported in ``withheld``. Its op_id is ALSO appended to
    ``legacy_withheld``: the skip advances that peer's cursor, so the ordinary
    stream would never offer this revision again, and ``_push_to_peer`` persists
    the id and redelivers the op once the peer advertises the capability.
    """
    _checked_vector(vector)
    result: dict[str, Any] = {
        "v": 1,
        "generation": source_generation,
        "scopes": scopes,
        "ops": [],
        "skips": [],
        "proofs": [],
        "after": dict(vector),
    }
    if not memory.path.exists():
        return result
    with memory.connection() as db:
        for origin in sorted(origins):
            rows = db.execute(
                "SELECT data,op_id,seq FROM ops WHERE origin=? AND seq>? " "ORDER BY seq LIMIT 128",
                (origin, vector.get(origin, 0)),
            ).fetchall()
            expected = vector.get(origin, 0) + 1
            for row in rows:
                before = json.loads(_dump(result))
                seq = row[2]
                if seq > expected:
                    # Portable session indexes can lack other namespaces' ops.
                    # A later fill increments the source generation and resets
                    # this peer cursor, so previously skipped rows are not lost.
                    result["skips"].append([origin, expected, seq - 1])
                try:
                    op = json.loads(row[0])
                    validate_op(op, allow_redacted=True)
                    if op["origin_device"] != origin or op["seq"] != seq or op["op_id"] != row[1]:
                        raise MemoryError("Memory operation/index identity mismatch.")
                    if (
                        legacy_item_cap
                        and op["scope"] in scopes
                        and op["fact"] is not None
                        and count_text_tokens(_dump(op["fact"])) > LEGACY_ITEM_WIRE_TOKENS
                    ):
                        if legacy_withheld is not None:
                            legacy_withheld.append(op["op_id"])
                        raise _LegacyWithheld()
                    if op["scope"] in scopes:
                        if op["kind"] == "put" and op["fact"] is None:
                            proof = db.execute(
                                "SELECT data FROM ops WHERE entity_id=? "
                                "AND kind='tombstone' ORDER BY op_id LIMIT 1",
                                (op["entity_id"],),
                            ).fetchone()
                            if not proof:
                                raise MemoryError("A redacted memory put has no tombstone proof.")
                            tomb = json.loads(proof[0])
                            validate_op(tomb)
                            if tomb["origin_device"] not in origins:
                                raise MemoryError(
                                    "A tombstone origin is no longer an active member."
                                )
                            if all(p["op_id"] != tomb["op_id"] for p in result["proofs"]):
                                result["proofs"].append(tomb)
                        result["ops"].append(op)
                    else:
                        result["skips"].append([origin, seq, seq])
                except (MemoryError, ValueError, TypeError) as exc:
                    # A legacy-cap skip is reported through ``legacy_withheld`` (and
                    # redelivered later), not as an unverifiable/unknown-origin row.
                    if (
                        withheld is not None
                        and len(withheld) < 20
                        and not isinstance(exc, _LegacyWithheld)
                    ):
                        withheld.append(_safe_ref(row[1]))
                    result["skips"].append([origin, seq, seq])
                result["after"][origin] = seq
                expected = seq + 1
                if (
                    len(_dump(result).encode()) > MAX_BATCH_BYTES - 1024
                    or sum(len(result[key]) for key in ("ops", "skips", "proofs")) > 120
                ):
                    return before
    return result


#: Bound on the op_ids remembered per peer as withheld for lack of
#: ``wire.MEMORY_ITEM_512``. Past it the peer is marked ``overflow`` and, once it
#: upgrades, re-baselined through a new stream generation (a full, idempotent
#: resend) instead of an id-by-id redelivery: the list is a meta row rewritten on
#: every push, so it must stay small.
LEGACY_WITHHELD_CAP = 256
#: Ops per redelivery request; each is <= ITEM_BYTES, so 24 stay well inside
#: ``MAX_BATCH_BYTES`` with the envelope.
_REDELIVER_OPS = 24


def _legacy_key(peer: str) -> str:
    return "memory-withheld:" + safe_segment(peer, structural=True)


def legacy_pending(memory: MemoryStore, peer: str) -> dict[str, Any]:
    """The ops withheld from ``peer`` for its legacy item cap, not yet redelivered.

    WHY this is persisted: a withheld op is a content-free skip that ADVANCES the
    peer's cursor, so the ordinary stream never offers it again. Without this
    record a large fact written once (a goal, a decision) would stay missing on
    that peer forever after it upgraded, while status reported ``in_sync``.
    """
    if not memory.path.exists():
        return {"ids": [], "overflow": False}
    with memory.connection() as db:
        row = db.execute("SELECT value FROM meta WHERE key=?", (_legacy_key(peer),)).fetchone()
        value = json.loads(row[0]) if row else {}
        ids = _still_held(db, list(value.get("ids", [])))
    return {"ids": ids, "overflow": bool(value.get("overflow"))}


def _still_held(db: Any, ids: list[str]) -> list[str]:
    """``ids`` minus any op this store no longer holds, order kept.

    WHY: journal compaction (now automatic, ``memory_maintenance``) can delete an
    owed op — a superseded revision, or a put under a tombstone. Nothing can ever
    redeliver it, so keeping its id made a still-legacy peer report
    ``withheld_pending`` for an op that no longer exists (PR #3057 review r5, m1).
    Dropping it loses nothing: the op's successor (the head or the tombstone) is
    an ordinary journal row the stream already carries.
    """
    if not ids:
        return ids
    held: set[str] = set()
    for start in range(0, len(ids), 500):  # SQLite's bound-parameter ceiling
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        held.update(
            row[0] for row in db.execute(f"SELECT op_id FROM ops WHERE op_id IN ({marks})", chunk)
        )
    return [op_id for op_id in ids if op_id in held]


def _owed(pending: dict[str, Any]) -> int | str:
    """The receipt's ``pending`` value: a count, or ``"overflow"`` when more were
    owed than the list remembers (a ``0`` there would read as nothing owed)."""
    return "overflow" if pending["overflow"] else len(pending["ids"])


def _save_legacy_pending(memory: MemoryStore, peer: str, ids: list[str], overflow: bool) -> None:
    ids = list(dict.fromkeys(ids))
    with memory.connection() as db:
        # Drop ids this store no longer holds BEFORE the cap. Journal compaction deletes
        # superseded ops, so a long-lived peer's withheld list accumulates dead ids that
        # nothing can ever redeliver; counting those against ``LEGACY_WITHHELD_CAP``
        # collapsed a list that already fit into ``overflow`` — a full resend forced by
        # ids that no longer exist (PR #3057 review r6, n1).
        ids = _still_held(db, ids)
        if len(ids) > LEGACY_WITHHELD_CAP:
            ids, overflow = [], True
        if ids or overflow:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                (_legacy_key(peer), _dump({"ids": ids, "overflow": overflow})),
            )
        else:
            db.execute("DELETE FROM meta WHERE key=?", (_legacy_key(peer),))
        db.commit()


def receive(
    server: Any, peer: str, payload: dict[str, Any], scopes: list[str], origins: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or len(_dump(payload).encode()) > MAX_BATCH_BYTES:
        raise MemoryError("Invalid or oversized memory payload.")
    # Do NOT guard() the whole payload: its structural identifiers (the origin
    # keys of the ``after`` vector, ``skips`` origins and each op's
    # origin_device/owner_key/op_id) are minted device/UUID shapes that can
    # legitimately collide with the free-text IBAN/card patterns — the same
    # permanent-refusal class MemoryStore.put fixed, reached here over the
    # mesh so an affected install could write locally yet never sync. They are
    # validated structurally instead (_checked_vector →
    # safe_segment(structural=True), and validate_op below). Every USER-TEXT
    # payload field still passes the free-text guard: each op's fact is
    # validated by validate_op → validate_fact, which refuses a real secret in
    # a synced body exactly as an admitted local write does.
    if type(payload.get("v")) is not int or payload["v"] != 1 or payload.get("scopes") != scopes:
        raise MemoryError("Memory payload version/scopes differ; update both peers.")
    source_generation = payload.get("generation")
    if type(source_generation) is not int or not 0 <= source_generation <= 2**63 - 1:
        raise MemoryError("Invalid memory stream generation.")
    memory = _store(server.root)
    before = {
        origin: seq
        for origin, seq in cursor(memory, peer, scopes, source_generation).items()
        if origin in origins
    }
    after = _checked_vector(payload.get("after"))
    operations, proofs, skips = (payload.get(key) for key in ("ops", "proofs", "skips"))
    # Ops this sender once withheld from us (``memory-item-512``) and now
    # redelivers. They sit BEHIND our cursor by construction, so they are kept out
    # of the contiguity accounting below; anything at or past the cursor is refused,
    # so this list can never be used to step around the sequence check. Only a
    # sender that saw us advertise the capability sends it, i.e. a build with this
    # code; an older receiver never sees the key.
    redeliver = payload.get("redeliver", [])
    if not isinstance(redeliver, list) or len(redeliver) > _REDELIVER_OPS:
        raise MemoryError("Invalid bounded memory redelivery list.")
    if any(
        not isinstance(value, list) or len(value) > 256 for value in (operations, proofs, skips)
    ):
        raise MemoryError("Invalid bounded memory batch lists.")
    intervals: dict[str, list[tuple[int, int]]] = {}
    assert isinstance(operations, list) and isinstance(proofs, list)
    assert isinstance(skips, list)
    for op in operations + proofs:
        try:
            validate_op(op, allow_redacted=True)
        except MemoryError as exc:
            reference = (
                _safe_ref(op.get("op_id")) if isinstance(op, dict) else "invalid-operation-id"
            )
            raise MemoryError(f"Memory operation {reference} refused: {exc}") from None
        if op["origin_device"] not in origins or op["scope"] not in scopes:
            raise MemoryError("Memory origin/scope is not active and authorised.")
    for op in redeliver:
        try:
            validate_op(op)
        except MemoryError as exc:
            reference = (
                _safe_ref(op.get("op_id")) if isinstance(op, dict) else "invalid-operation-id"
            )
            raise MemoryError(f"Memory operation {reference} refused: {exc}") from None
        if op["origin_device"] not in origins or op["scope"] not in scopes:
            raise MemoryError("Memory origin/scope is not active and authorised.")
        if op["seq"] > before.get(op["origin_device"], 0):
            raise MemoryError("A redelivered memory operation is not behind this peer's cursor.")
    for op in operations:
        intervals.setdefault(op["origin_device"], []).append((op["seq"], op["seq"]))
    for row in skips:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or row[0] not in origins
            or any(type(number) is not int or not 1 <= number <= 2**63 - 1 for number in row[1:])
            or row[1] > row[2]
        ):
            raise MemoryError("Invalid content-free memory sequence range.")
        intervals.setdefault(row[0], []).append((row[1], row[2]))
    for origin in set(before) | set(after) | set(intervals):
        if origin not in origins:
            raise MemoryError("Memory vector origin is not an active member.")
        expected = before.get(origin, 0) + 1
        for start, end in sorted(intervals.get(origin, [])):
            if start != expected:
                raise MemoryError("Memory sequence gap/overlap; refresh peer state and retry.")
            expected = end + 1
        if expected - 1 != after.get(origin, 0):
            raise MemoryError("Memory sequence vector does not cover the batch.")
    # Ops are atomic and idempotent. Cursor is committed only AFTER them: a
    # process crash between commits can cause a replay, never a lost operation.
    changed = memory.apply_ops(operations + proofs + redeliver, origins=origins, contiguous=False)
    if after != before:
        with memory.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                (_key(peer, scopes), _dump({"generation": source_generation, "vector": after})),
            )
            db.commit()
    return {"phase": "push", "changed": changed, "vector": after, "scopes": scopes}


def make_handler(server: Any) -> Any:
    def handle(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        try:
            enabled, local_scopes, excluded = settings(server.root)
            if not enabled or not local_scopes or link.device_id in excluded:
                raise MeshRefusal(
                    "policy", "Memory replication is disabled or this peer is excluded."
                )
            if wire.MEMORY_SYNC_V1 not in link.capabilities:
                raise MeshRefusal("peer_too_old", "Both peers need memory-sync-v1; run lop-update.")
            record = network_store.load(link.network_id, server.root)
            origins = {member.device_id for member in record.active_members()}
            requested = frame.get("scopes", local_scopes)
            if (
                not isinstance(requested, list)
                or len(requested) > 4
                or any(scope not in SCOPES for scope in requested)
            ):
                raise MemoryError("Memory scope negotiation refused.")
            scopes = sorted(set(requested) & set(local_scopes))
            if frame.get("phase") == "push" and sorted(set(requested)) != scopes:
                raise MemoryError("Memory push contains unnegotiated scopes.")
            if frame.get("phase") == "state":
                source_generation = frame.get("generation")
                if type(source_generation) is not int or not 0 <= source_generation <= 2**63 - 1:
                    raise MemoryError("Invalid memory stream generation.")
                return {
                    "phase": "state",
                    "v": 1,
                    "scopes": scopes,
                    "generation": source_generation,
                    "vector": {
                        origin: seq
                        for origin, seq in cursor(
                            _store(server.root), link.device_id, scopes, source_generation
                        ).items()
                        if origin in origins
                    },
                }
            if frame.get("phase") == "push":
                payload = frame.get("payload")
                if not isinstance(payload, dict):
                    raise MemoryError("Invalid or oversized memory payload.")
                return receive(server, link.device_id, payload, scopes, origins)
            raise MemoryError("Unknown memory sync phase.")
        except MemoryError as exc:
            raise MeshRefusal("bad_memory", str(exc)) from None

    return handle


def _safe_ref(value: Any) -> str:
    # An op-id reference is structural (minted UUID/hex), so it is shape-checked
    # rather than free-text guarded — consistent with validate_op's structural
    # fields, and it keeps a legitimate hex id from being blanked as "sensitive".
    try:
        return safe_segment(value, structural=True)
    except MemoryError:
        return "invalid-operation-id"


def _peer_code(value: Any, fallback: str) -> str:
    allowed = {
        "policy",
        "bad_memory",
        "refused",
        "not_authorised",
        "capability_denied",
        "phase_forbidden",
        "not_a_member",
        "peer_too_old",
    }
    return value if isinstance(value, str) and value in allowed else fallback


#: Receipt codes that mean the peer is receiving normally. Everything else is shown
#: as a failing peer in `lop memory status`; a code this module has never heard of is
#: treated as failing too, because claiming "synced" is the failure mode we are fixing.
HEALTHY_PEER_CODES = frozenset(
    {
        "in_sync",
        "catching_up",
        # Some origins are withheld from this scope (an unknown device); the peer is
        # receiving everything authorised, so this is not a stalled stream.
        "withheld",
        # Nothing is shared BY CONFIGURATION; there is no failure to report.
        "no_shared_scopes",
    }
)


def _peer_message(value: Any) -> str:
    """A peer's sentence, flattened and bounded for a receipt/status line.

    Peer text is UNTRUSTED display input, not a protocol field: a newline would
    break the one-line status receipt, and an unbounded sentence would bloat the
    meta row rewritten on every tick. The peer's refusal sentences are value-free
    by construction (MemoryError paths), but this bounds them anyway.
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:500]


def _push_to_peer(server: Any, device_id: str) -> dict[str, Any]:
    enabled, scopes, excluded = settings(server.root)
    if not enabled or not scopes or device_id in excluded:
        return {"ok": True, "code": "disabled_or_excluded"}
    blocked = definitions.unholdable_capability(server, device_id, "net_memory")
    if blocked:
        return {"ok": False, "code": "skipped:no_" + blocked}
    link = server._ensure_link(device_id)
    if link is None:
        return {"ok": False, "code": "unreachable"}
    if wire.MEMORY_SYNC_V1 not in link.capabilities:
        return {
            "ok": False,
            "code": "peer_too_old",
            "message": "Run lop-update on both peers for memory-sync-v1.",
        }
    active = {
        m.device_id for m in network_store.load(link.network_id, server.root).active_members()
    }

    memory = _store(server.root)
    source_generation = generation(memory)

    def request(phase: str, **extra: Any) -> Any:
        return link.request(
            {
                "op": "net_memory",
                "req": server._next_relay_req(),
                "locality": "remote",
                "phase": phase,
                "generation": source_generation,
                "scopes": scopes,
                **extra,
            },
            timeout=30,
        )

    state = request("state")
    if not isinstance(state, dict) or state.get("op") == "error" or state.get("ok") is False:
        return {
            "ok": False,
            "code": (
                _peer_code(state.get("code"), "refused")
                if isinstance(state, dict)
                else "state_failed"
            ),
            # The only actionable sentence in a refusal is the PEER's; dropping it
            # here left the operator with a code and no remedy (round-2 MAJOR).
            "message": _peer_message(state.get("message")) if isinstance(state, dict) else "",
        }
    remote = state.get("detail", state)
    if (
        remote.get("v") != 1
        or remote.get("generation") != source_generation
        or not isinstance(remote.get("scopes"), list)
        or any(scope not in scopes for scope in remote["scopes"])
    ):
        return {"ok": False, "code": "scope_or_version_mismatch"}
    scopes = remote["scopes"]
    if not scopes:
        return {"ok": True, "code": "no_shared_scopes", "batches": 0}
    vector = _checked_vector(remote.get("vector"))
    batches = 0
    withheld: list[str] = []
    if memory.path.exists():
        with memory.connection() as db:
            for row in db.execute("SELECT DISTINCT origin FROM ops LIMIT 257"):
                if row[0] not in active and len(withheld) < 20:
                    reference = db.execute(
                        "SELECT op_id FROM ops WHERE origin=? LIMIT 1", (row[0],)
                    ).fetchone()
                    withheld.append(_safe_ref(reference[0]))
    legacy = wire.MEMORY_ITEM_512 not in link.capabilities
    pending = legacy_pending(memory, device_id)
    if not legacy and pending["overflow"]:
        # Too many withheld ids to remember (or a redelivery was refused): re-baseline
        # instead. A new stream generation makes every peer's cursor for us start
        # empty, so the next pass is a full idempotent resend that includes every
        # skipped op. The generation is DEVICE-WIDE, so up-to-date peers resend too:
        # correct (apply is idempotent) and rare, the price of not tracking cursors
        # per peer here. This push itself sends nothing; the next one does the resend.
        with memory.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('stream-generation',?)",
                (str(source_generation + 1),),
            )
            db.commit()
        _save_legacy_pending(memory, device_id, [], False)
        return {"ok": True, "code": "catching_up", "batches": batches, "withheld": withheld}
    elif not legacy and pending["ids"]:
        # The peer now accepts 512-content facts: redeliver what its legacy cap
        # made us skip, in the ``redeliver`` list the receiver applies without
        # contiguity accounting.
        with memory.connection() as db:
            ops = []
            for op_id in pending["ids"]:
                row = db.execute("SELECT data FROM ops WHERE op_id=?", (op_id,)).fetchone()
                op = json.loads(row[0]) if row else None
                # Gone (compacted away as superseded) or redacted under a tombstone:
                # the stream already carried what replaced it, so nothing is owed.
                if op is None or op["fact"] is None or op["scope"] not in scopes:
                    continue
                # ONLY ops behind the peer's cursor (from its ``state`` reply for
                # this generation) are redeliverable; the receiver refuses anything
                # else. An id can legitimately be ahead of it: the stream generation
                # moved (gap fill, another peer's re-baseline) so the cursor reset, or
                # the push that would have advanced past it failed. Such an op is
                # dropped from the owed list, NOT retried: the ordinary stream now
                # carries it unwithheld (the peer has the capability). Retrying it
                # here was review round 2's M4, a refusal that stalled every scope.
                if op["seq"] <= vector.get(op["origin_device"], 0):
                    ops.append(op)
        remaining = [op["op_id"] for op in ops]
        for start in range(0, len(ops), _REDELIVER_OPS):
            chunk = ops[start : start + _REDELIVER_OPS]
            answer = request(
                "push",
                payload={
                    "v": 1,
                    "generation": source_generation,
                    "scopes": scopes,
                    "ops": [],
                    "skips": [],
                    "proofs": [],
                    "after": dict(vector),
                    "redeliver": chunk,
                },
            )
            if isinstance(answer, dict) and (
                answer.get("op") == "error" or answer.get("ok") is False
            ):
                # The peer REFUSED this redelivery. Whatever the reason, retrying the
                # same ops would refuse again and block every later push, so this
                # chunk is dropped from the owed list and the peer is re-baselined
                # (``overflow``): the next push opens a new stream generation whose
                # full resend carries these ops through the ordinary, validated path.
                # One refusal therefore costs one resend, never a permanent stall.
                remaining = remaining[len(chunk) :]
                _save_legacy_pending(memory, device_id, remaining, True)
                return {
                    "ok": False,
                    "code": _peer_code(answer.get("code"), "refused"),
                    "message": _peer_message(answer.get("message")),
                    "batches": batches,
                    # Same shape as every other receipt from this phase, so the
                    # owed state survives into `lop memory status --json`.
                    "pending": _owed({"ids": remaining, "overflow": True}),
                }
            if not isinstance(answer, dict):
                # No answer (the peer went away mid-push): nothing was refused, and
                # every remaining op is still behind its cursor, so keep them owed.
                _save_legacy_pending(memory, device_id, remaining, False)
                return {
                    "ok": False,
                    "code": "push_failed",
                    "message": "",
                    "batches": batches,
                    "pending": _owed({"ids": remaining, "overflow": False}),
                }
            remaining = remaining[len(chunk) :]
            batches += 1
        _save_legacy_pending(memory, device_id, remaining, False)
        pending = {"ids": remaining, "overflow": False}
    while batches < 8:
        newly: list[str] = []
        payload = batch(
            memory,
            vector,
            scopes,
            active,
            source_generation,
            withheld,
            legacy_item_cap=legacy,
            legacy_withheld=newly,
        )
        if newly:
            # Recorded BEFORE the push that advances the peer's cursor past them:
            # a skip without this record is the permanent loss this list prevents.
            pending = {"ids": pending["ids"] + newly, "overflow": pending["overflow"]}
            _save_legacy_pending(memory, device_id, pending["ids"], pending["overflow"])
            pending = legacy_pending(memory, device_id)
        if payload["after"] == vector:
            owed = bool(pending["ids"] or pending["overflow"])
            return {
                "ok": not withheld and not owed,
                "code": (
                    "withheld"
                    if withheld
                    else (
                        "withheld_pending"
                        if owed
                        else (
                            "in_sync" if generation(memory) == source_generation else "catching_up"
                        )
                    )
                ),
                "batches": batches,
                "withheld": withheld,
                "pending": _owed(pending),
            }
        answer = request("push", payload=payload)
        if not isinstance(answer, dict) or answer.get("op") == "error" or answer.get("ok") is False:
            return {
                "ok": False,
                "code": (
                    _peer_code(answer.get("code"), "refused")
                    if isinstance(answer, dict)
                    else "push_failed"
                ),
                "message": _peer_message(answer.get("message")) if isinstance(answer, dict) else "",
                "batches": batches,
            }
        reply = answer.get("detail", answer)
        if reply.get("vector") != payload["after"]:
            return {"ok": False, "code": "ack_mismatch"}
        vector = reply["vector"]
        batches += 1
    owed = bool(pending["ids"] or pending["overflow"])
    return {
        "ok": not withheld and not owed,
        "code": "withheld" if withheld else "withheld_pending" if owed else "catching_up",
        "batches": batches,
        "withheld": withheld,
        "pending": _owed(pending),
    }


def peer_status(root: Path) -> list[dict[str, Any]]:
    memory = _store(root)
    if not memory.path.exists():
        return []
    with memory.connection() as db:
        return [
            json.loads(row[0])
            for row in db.execute(
                "SELECT value FROM meta WHERE key LIKE 'sync-status:%' ORDER BY key LIMIT 256",
            )
        ]


def push_to_peer(server: Any, device_id: str) -> dict[str, Any]:
    try:
        result = _push_to_peer(server, device_id)
    except Exception:
        result = {"ok": False, "code": "memory_sync_failed"}
    memory = _store(server.root)
    if memory.path.exists() and result.get("code") != "disabled_or_excluded":
        try:
            safe_segment(device_id, structural=True)
            with memory.connection() as db:
                key = "sync-status:" + device_id
                present = db.execute("SELECT 1 FROM meta WHERE key=?", (key,)).fetchone()
                count = db.execute(
                    "SELECT count(*) FROM meta WHERE key LIKE 'sync-status:%'"
                ).fetchone()[0]
                if present or count < 256:
                    previous_receipt = db.execute(
                        "SELECT value FROM meta WHERE key=?", (key,)
                    ).fetchone()
                    previous_receipt = json.loads(previous_receipt[0]) if previous_receipt else {}
                    receipt = {
                        "device": device_id,
                        "code": result["code"],
                        "at": time.time(),
                        "batches": result.get("batches", 0),
                        "last_withheld": result.get("withheld")
                        or previous_receipt.get("last_withheld", []),
                        # Facts this peer has not received because it predates
                        # memory-item-512; redelivered automatically once it updates.
                        "pending": result.get("pending", 0),
                        # A healthy answer has no message, which CLEARS a stale
                        # refusal sentence from the previous receipt; otherwise a
                        # recovered peer would keep showing its old failure reason.
                        "message": _peer_message(result.get("message")),
                    }
                    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, _dump(receipt)))
                    db.commit()
        except Exception:
            pass  # observational receipts cannot obstruct sync or leak payloads
    return result


def mesh_tick_step(server: Any, device_id: str) -> str:
    if not _store(server.root).path.exists():
        return "in_sync"  # symmetric pushes: an empty bank has nothing to offer
    try:
        return str(push_to_peer(server, device_id).get("code", "failed"))
    except Exception:
        # Content never enters cadence logs/diagnostics, even through exceptions.
        return "memory_sync_failed"


def install(server: Any) -> None:
    definitions.add_tick_step(mesh_tick_step)
    server.register_ops({"net_memory": make_handler(server)}, slow={"net_memory": 60.0})
