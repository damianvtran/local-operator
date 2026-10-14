"""Forward-only credential lifecycle generations and the recipient's observation store.

``<config>/network/credentials/<network_id>/lifecycle.json``, 0600, METADATA ONLY:
integers, booleans and reference strings. Design authority:
``docs/design/mesh-credential-lifecycle.md`` ("Generations and metadata-only
transport"). This slice has no wire handler, owner dispatch or capability
advertisement; it is the persistence + arithmetic those later slices call.

Two independent counters per ``(credential key, account_scope_ref)`` on the OWNER:

* ``material_generation`` advances on a native presence/absence change (login or
  logout) or when the owner row's ``updated_at`` moves (rotation, expiry repair).
  It invalidates borrowed material/status evidence held by recipients.
* ``login_chain_revision`` advances on a new native login, an authorised re-enable
  or an explicit scope change — and NEVER on rotation. It is the evidence for one
  bounded MCP retry/reconnect; bumping it on rotation would turn every token
  refresh into a reconnect storm, which is exactly what two counters exist to stop.

Both are FORWARD-ONLY. Persisted state that is unreadable, malformed, of an
unknown schema, for another network or carrying an out-of-range counter is NOT
re-initialised from zero: it reports :data:`NEEDS_AUTHORITATIVE_BASELINE`, refuses
every positive application (nothing is written, the file is left for repair) and
never upgrades consent or clears a deny. A silent restart at 1 would let a replayed
older announcement look newer than everything the recipients already saw.
Known limit: a file that is simply ABSENT is a genesis, indistinguishable here from
a device that never ran lifecycle; the owner-authored checkpoints that would prove
otherwise belong to the recovery slice.

THE COPY-PROTOCOL MACHINERY IS DELIBERATELY NOT REUSED. ``sync``'s ``fingerprint``,
equal-material-digest comparison, downward adoption (``record_ack``/generation
adoption) and counter clamping belong to the copy classes and are banned for
lifecycle by the approved design: a digest is derived from material (a lifecycle
frame must not carry material-derived values) and downward adoption is a silent
downgrade. Only ``sync``'s locking and bounds conventions are mirrored.

Sticky withdrawal: ``withdrawn`` is a per-row flag set by the owner when a deny is
executed. Logout and re-login move the counters and ``present`` but leave
``withdrawn`` standing; only an explicit authorised re-enable clears it. The
authoritative deny list stays in ``consent.json``; this flag is its projection and
this module never reads or writes that file.

LOCK ORDER: consent -> lifecycle -> placement. This file takes only its own lock
(in-process ``store._write_lock`` then the cross-process ``.lifecycle.lock``) and
never calls into consent or placement while holding it, so a caller that already
holds consent may take lifecycle, and nothing here can invert that.

Pure derivation (:func:`derive`: facts -> next row) is separate from persistence
(:class:`LifecycleStore`) so the counter rules are testable without a filesystem.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn, Optional

from local_operator.network import store
from local_operator.network.credentials import consent, placement
from local_operator.network.types import MeshRefusal

SCHEMA = 1
LIFECYCLE_FILENAME = "lifecycle.json"
LIFECYCLE_LOCK_FILENAME = ".lifecycle.lock"

#: Counters are positive integers bounded like consent revisions, keeping headroom
#: below 2**53 (JSON numbers must round-trip through JavaScript peers).
MAX_REVISION = consent.MAX_REVISION
#: A single forward step a recipient will adopt without an authoritative baseline.
MAX_RECOVERY_JUMP = 2**20
#: Bounds on the document, mirroring the lifecycle frame's own small caps.
MAX_ROWS = 4096
MAX_FILE_BYTES = 1_048_576

#: The named unprovable state. Also the ``MeshRefusal`` code on the owner side.
NEEDS_AUTHORITATIVE_BASELINE = "needs_authoritative_baseline"

STATE_AVAILABLE = "available"
STATE_ABSENT = "absent"
STATE_WITHDRAWN = "withdrawn"

OBSERVED = "applied"
OBSERVED_STALE = "stale"
OBSERVED_REPLAY = "replayed"
OBSERVED_JUMP = "jump_too_large"
OBSERVED_INVALID = "invalid"
#: A NEW (owner, key, scope) ident refused because the table is at its bound.
OBSERVED_TABLE_FULL = "observation_table_full"
#: ``MeshRefusal`` code for the same condition on the owner counters side.
TABLE_FULL = "lifecycle_table_full"


@dataclass(frozen=True)
class CounterRow:
    """One owner row. ``row_updated_at`` is the owner row's own clock, not wall time."""

    key: str
    account_scope_ref: str
    material_generation: int
    login_chain_revision: int
    present: bool
    withdrawn: bool = False
    row_updated_at: Optional[float] = None

    @property
    def state(self) -> str:
        # Withdrawal outranks availability: a sticky deny is never shown as available.
        if self.withdrawn:
            return STATE_WITHDRAWN
        return STATE_AVAILABLE if self.present else STATE_ABSENT


@dataclass(frozen=True)
class NativeFacts:
    """What the owner just observed. Event flags are the owner's, never a peer's."""

    present: bool
    row_updated_at: Optional[float] = None
    new_login: bool = False
    scope_changed: bool = False
    authorised_reenable: bool = False
    withdrawn: bool = False


@dataclass(frozen=True)
class Observation:
    owner: str
    key: str
    account_scope_ref: str
    login_chain_revision: int
    material_generation: int


@dataclass(frozen=True)
class ObservationResult:
    applied: bool
    reason: str
    login_chain_revision: int = 0
    material_generation: int = 0


def derive(prev: Optional[CounterRow], key: str, scope: str, facts: NativeFacts) -> CounterRow:
    """Facts -> next row. Pure; never lowers a counter.

    A first observation seeds both counters at 1 (positive integers, per the frame
    validator) and is not itself a bump. After that, material advances at most once
    per call however many reasons coincide (a re-login that also moves
    ``updated_at`` is ONE new generation), and the chain only on login/re-enable/
    scope change.
    """
    if prev is None:
        return CounterRow(
            key,
            scope,
            1,
            1,
            facts.present,
            facts.withdrawn,
            facts.row_updated_at if facts.present else None,
        )
    presence_changed = prev.present != facts.present
    rotated = (
        prev.present
        and facts.present
        and facts.row_updated_at is not None
        and facts.row_updated_at != prev.row_updated_at
    )
    material_bump = presence_changed or rotated
    login = (
        (facts.present and not prev.present)
        or (facts.present and facts.new_login)
        or facts.authorised_reenable
        or facts.scope_changed
    )
    withdrawn = (prev.withdrawn or facts.withdrawn) and not facts.authorised_reenable
    material = prev.material_generation + (1 if material_bump else 0)
    chain = prev.login_chain_revision + (1 if login else 0)
    if material > MAX_REVISION or chain > MAX_REVISION:
        # Exhausted headroom needs an authoritative checkpoint repair, not wraparound.
        raise MeshRefusal(
            "lifecycle_counter_exhausted", "credential lifecycle refused: counter exhausted"
        )
    return replace(
        prev,
        material_generation=material,
        login_chain_revision=chain,
        present=facts.present,
        withdrawn=withdrawn,
        row_updated_at=facts.row_updated_at if facts.present else None,
    )


def classify_observation(
    high_water: Optional[tuple[int, int]], observed: tuple[int, int]
) -> tuple[str, tuple[int, int]]:
    """Pure recipient rule: ``(reason, new_high_water)``. ``reason == OBSERVED`` applies.

    Each counter is judged independently. Any counter below its high-water is stale
    (an owner never lowers either), all-equal is a replay, and a forward step over
    :data:`MAX_RECOVERY_JUMP` is refused rather than adopted from a bare peer
    number. The first observation is bounded only by :data:`MAX_REVISION`.
    """
    chain, material = observed
    if high_water is None:
        return OBSERVED, (chain, material)
    hw_chain, hw_material = high_water
    if chain < hw_chain or material < hw_material:
        return OBSERVED_STALE, high_water
    if chain == hw_chain and material == hw_material:
        return OBSERVED_REPLAY, high_water
    if chain - hw_chain > MAX_RECOVERY_JUMP or material - hw_material > MAX_RECOVERY_JUMP:
        return OBSERVED_JUMP, high_water
    return OBSERVED, (chain, material)


def lifecycle_path(network_id: str, root: Path | None = None) -> Path:
    """The document's path. NEVER CREATES anything."""
    return placement.credentials_root(root) / network_id / LIFECYCLE_FILENAME


def _revision_ok(value: Any) -> bool:
    return type(value) is int and 1 <= value <= MAX_REVISION


def _ref_ok(value: Any) -> bool:
    return isinstance(value, str) and consent.REFERENCE.match(value) is not None


def _baseline() -> NoReturn:
    raise MeshRefusal(
        NEEDS_AUTHORITATIVE_BASELINE,
        "credential lifecycle state on this device cannot be proven; an authoritative "
        "baseline is required and nothing was changed",
    )


class LifecycleDocument:
    """In-memory view of ``lifecycle.json``. ``needs_baseline`` marks unprovable state."""

    def __init__(self, network_id: str) -> None:
        self.network_id = network_id
        self.rows: dict[tuple[str, str], CounterRow] = {}
        #: (owner, key, scope) -> (login_chain_revision, material_generation) high-water
        self.high_water: dict[tuple[str, str, str], tuple[int, int]] = {}
        self.needs_baseline = False

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "network_id": self.network_id,
            "counters": [
                {
                    "key": r.key,
                    "account_scope_ref": r.account_scope_ref,
                    "material_generation": r.material_generation,
                    "login_chain_revision": r.login_chain_revision,
                    "present": r.present,
                    "withdrawn": r.withdrawn,
                    "row_updated_at": r.row_updated_at,
                }
                for _, r in sorted(self.rows.items())
            ],
            "observations": [
                {
                    "owner": o,
                    "key": k,
                    "account_scope_ref": s,
                    "login_chain_revision": hw[0],
                    "material_generation": hw[1],
                }
                for (o, k, s), hw in sorted(self.high_water.items())
            ],
        }

    @classmethod
    def from_json(cls, network_id: str, data: Any) -> "LifecycleDocument":
        """Exact-schema parse; ANY deviation flags ``needs_baseline`` (never partial)."""
        doc = cls(network_id)
        try:
            if (
                not isinstance(data, dict)
                or set(data) != {"schema", "network_id", "counters", "observations"}
                or type(data["schema"]) is not int
                or data["schema"] != SCHEMA
                or data["network_id"] != network_id
                or not isinstance(data["counters"], list)
                or not isinstance(data["observations"], list)
                or len(data["counters"]) > MAX_ROWS
                or len(data["observations"]) > MAX_ROWS
            ):
                raise ValueError
            row_fields = {
                "key",
                "account_scope_ref",
                "material_generation",
                "login_chain_revision",
                "present",
                "withdrawn",
                "row_updated_at",
            }
            for item in data["counters"]:
                if not isinstance(item, dict) or set(item) != row_fields:
                    raise ValueError
                updated = item["row_updated_at"]
                if (
                    not _ref_ok(item["key"])
                    or not _ref_ok(item["account_scope_ref"])
                    or not _revision_ok(item["material_generation"])
                    or not _revision_ok(item["login_chain_revision"])
                    or type(item["present"]) is not bool
                    or type(item["withdrawn"]) is not bool
                    or not (
                        updated is None
                        or (type(updated) in (int, float) and updated == updated)  # no NaN
                    )
                ):
                    raise ValueError
                ident = (item["key"], item["account_scope_ref"])
                if ident in doc.rows:
                    raise ValueError
                doc.rows[ident] = CounterRow(**item)
            obs_fields = {
                "owner",
                "key",
                "account_scope_ref",
                "login_chain_revision",
                "material_generation",
            }
            for item in data["observations"]:
                if not isinstance(item, dict) or set(item) != obs_fields:
                    raise ValueError
                if (
                    not _ref_ok(item["owner"])
                    or not _ref_ok(item["key"])
                    or not _ref_ok(item["account_scope_ref"])
                    or not _revision_ok(item["login_chain_revision"])
                    or not _revision_ok(item["material_generation"])
                ):
                    raise ValueError
                ident3 = (item["owner"], item["key"], item["account_scope_ref"])
                if ident3 in doc.high_water:
                    raise ValueError
                doc.high_water[ident3] = (
                    item["login_chain_revision"],
                    item["material_generation"],
                )
        except (ValueError, TypeError):
            bad = cls(network_id)
            bad.needs_baseline = True
            return bad
        return doc


class LifecycleStore:
    """Owner counters + recipient high-waters for one network, under one lock."""

    def __init__(self, network_id: str, root: Path | None = None) -> None:
        store.safe_component(network_id)
        self.network_id = network_id
        self.root = root
        self.path = lifecycle_path(network_id, root)
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise MeshRefusal("lifecycle_path_invalid", "credential lifecycle path refused")

    @contextmanager
    def locked(self) -> Iterator[None]:
        placement.credentials_dir_for(self.network_id, self.root)
        with store._write_lock(self.path):
            with placement._cross_process_lock(self.path.with_name(LIFECYCLE_LOCK_FILENAME)):
                yield

    def load(self) -> LifecycleDocument:
        """Read without creating anything. Absent file = genesis; unreadable = baseline."""
        if not self.path.exists():
            return LifecycleDocument(self.network_id)
        try:
            raw = self.path.read_bytes()
            if len(raw) > MAX_FILE_BYTES:
                raise ValueError
            data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (OSError, ValueError):
            doc = LifecycleDocument(self.network_id)
            doc.needs_baseline = True
            return doc
        return LifecycleDocument.from_json(self.network_id, data)

    def _save(self, doc: LifecycleDocument) -> None:
        """Persist, refusing any document :meth:`load` would reject.

        WHY: the reader treats an over-bound file as ``needs_authoritative_baseline``,
        which locks every later call out and has no repair path. A peer chooses the
        ``account_scope_ref`` values it announces, so the writer must hold the same
        bounds the reader enforces; a refusal leaves the existing file byte-identical
        (the write is staged-then-renamed, so nothing is touched before this check).
        """
        payload = doc.to_json()
        if (
            len(payload["counters"]) > MAX_ROWS
            or len(payload["observations"]) > MAX_ROWS
            or len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_FILE_BYTES
        ):
            raise MeshRefusal(
                TABLE_FULL, "credential lifecycle table is full; nothing was changed"
            )
        store._write_private_json(self.path, payload)

    def status(self) -> str:
        return NEEDS_AUTHORITATIVE_BASELINE if self.load().needs_baseline else "ok"

    def snapshot(self, key: str, scope: str) -> Optional[CounterRow]:
        doc = self.load()
        if doc.needs_baseline:
            _baseline()
        return doc.rows.get((key, scope))

    def scopes_for(self, key: str) -> frozenset[str]:
        """Every account scope this owner holds a row for under ``key`` (may be empty)."""
        doc = self.load()
        if doc.needs_baseline:
            _baseline()
        return frozenset(scope for (row_key, scope) in doc.rows if row_key == key)

    def record_native(self, key: str, scope: str, facts: NativeFacts) -> CounterRow:
        """Owner side: fold one native observation into the counters and persist."""
        if not _ref_ok(key) or not _ref_ok(scope):
            raise MeshRefusal("lifecycle_invalid", "credential lifecycle refused: invalid ref")
        with self.locked():
            doc = self.load()
            if doc.needs_baseline:
                _baseline()
            prev = doc.rows.get((key, scope))
            nxt = derive(prev, key, scope, facts)
            if nxt != prev:
                # Only a NEW row grows the table; moving an existing row (a logout,
                # a withdraw) must never be blocked by the bound, and ``_save``
                # refuses by name if the file bound itself would be crossed.
                if prev is None and len(doc.rows) >= MAX_ROWS:
                    raise MeshRefusal(
                        TABLE_FULL, "credential lifecycle table is full; nothing was changed"
                    )
                doc.rows[(key, scope)] = nxt
                self._save(doc)
            return nxt

    def observe(self, obs: Observation) -> ObservationResult:
        """Recipient side: apply an observation only if it is strictly newer."""
        if not (
            _ref_ok(obs.owner)
            and _ref_ok(obs.key)
            and _ref_ok(obs.account_scope_ref)
            and _revision_ok(obs.login_chain_revision)
            and _revision_ok(obs.material_generation)
        ):
            return ObservationResult(False, OBSERVED_INVALID)
        ident = (obs.owner, obs.key, obs.account_scope_ref)
        with self.locked():
            doc = self.load()
            if doc.needs_baseline:
                return ObservationResult(False, NEEDS_AUTHORITATIVE_BASELINE)
            reason, hw = classify_observation(
                doc.high_water.get(ident), (obs.login_chain_revision, obs.material_generation)
            )
            if reason == OBSERVED:
                # A fresh ident grows the table; an existing one only moves forward,
                # so restrictive updates still apply when the table is full.
                if ident not in doc.high_water and len(doc.high_water) >= MAX_ROWS:
                    return ObservationResult(False, OBSERVED_TABLE_FULL)
                doc.high_water[ident] = hw
                try:
                    self._save(doc)
                except MeshRefusal as refusal:
                    if refusal.code != TABLE_FULL:
                        raise
                    return ObservationResult(False, OBSERVED_TABLE_FULL)
            return ObservationResult(reason == OBSERVED, reason, hw[0], hw[1])


# ---------------------------------------------------------------------------
# Wire entries (shared by the owner's read/ack handlers and the recipient's announce)
# ---------------------------------------------------------------------------

#: Per-key states the owner reports on ``lifecycle_read``. The design's enum
#: (``available``/``absent``/``withdrawn``/``unsupported``/``non_portable``) plus
#: ``scope_unknown``: an unresolved required scope is a REFUSAL to infer, reported
#: by name so the recipient shows why instead of guessing a grant.
STATE_UNSUPPORTED = "unsupported"
STATE_NON_PORTABLE = "non_portable"
STATE_SCOPE_UNKNOWN = "scope_unknown"
WIRE_STATES = frozenset(
    {
        STATE_AVAILABLE,
        STATE_ABSENT,
        STATE_WITHDRAWN,
        STATE_UNSUPPORTED,
        STATE_NON_PORTABLE,
        STATE_SCOPE_UNKNOWN,
    }
)

#: Upper bound on entries per frame; the design's "1-8 unique entries".
MAX_ENTRIES = 8


def parse_entries(frame: Any, *, required: tuple[str, ...]) -> list[dict[str, Any]]:
    """The frame's ``entries`` as validated dicts, or ``ValueError`` (nothing applied).

    Every entry must carry exactly ``credential_key`` + ``account_scope_ref`` plus the
    ``required`` extras and nothing else: unknown fields are REJECTED, not ignored,
    because an ignored field is a side channel for material the protocol forbids.
    ``(credential_key, account_scope_ref)`` pairs must be unique within a frame.
    """
    entries = frame.get("entries") if hasattr(frame, "get") else None
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES:
        raise ValueError(f"lifecycle frame needs 1-{MAX_ENTRIES} entries")
    allowed = {"credential_key", "account_scope_ref", *required}
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != allowed:
            raise ValueError("lifecycle entry has missing or unknown fields")
        key, scope = item["credential_key"], item["account_scope_ref"]
        if not (_ref_ok(key) and _ref_ok(scope)):
            raise ValueError("lifecycle entry has an invalid credential or scope reference")
        for name in required:
            if name == "state":
                if item[name] not in WIRE_STATES:
                    raise ValueError("lifecycle entry state is not in the declared enum")
            elif not _revision_ok(item[name]):
                raise ValueError(f"lifecycle entry {name} is not a valid revision")
        if (key, scope) in seen:
            raise ValueError("lifecycle frame repeats an entry")
        seen.add((key, scope))
        out.append(dict(item))
    return out


# ---------------------------------------------------------------------------
# The recipient half of ``lifecycle_announce``
# ---------------------------------------------------------------------------

#: States an announcement may carry. ``scope_unknown`` is a READ-side answer only
#: (the owner declining to infer a scope for a specific read), not something an
#: owner pushes.
_ANNOUNCE_STATES = WIRE_STATES - {STATE_SCOPE_UNKNOWN}

#: Evidence that can only RESTRICT. Applying one again at an unchanged counter is
#: safe and necessary: the owner's sticky deny does not move either counter, so a
#: forward-only rule alone would drop the very announcement that withdraws access.
_RESTRICTIVE_STATES = frozenset({STATE_WITHDRAWN, STATE_NON_PORTABLE, STATE_UNSUPPORTED})


def _announce_error(code: str, key: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "key": key, "message": message}


def member_lifecycle_announce(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Serve a ``lifecycle_announce`` (owner -> recipient). Returns a ``detail`` object.

    Analog of ``sync.member_announce``. Order, each step refusing BEFORE any write:
    the frame validator, the authenticated sender (``link.device_id``, never the
    frame), the sender being the placement OWNER of every named key, the frame's
    ``recipient_device`` being THIS device, then the forward-only high-water.

    THE ACK PROVES METADATA WAS APPLIED, NOT AUTH. Nothing here can make a
    credential usable: the observation only records pending/withdrawn/unsupported
    evidence (``effective.note_lifecycle_observation``); material still arrives
    only through ``grant``, and only a successful resolve marks access usable.
    """
    from local_operator.network.credentials import effective
    from local_operator.network.credentials.placement import PlacementDocument
    from local_operator.network.credentials.types import validate_lifecycle_frame

    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if server is None or root is None or not self_device:
        return _announce_error(
            "unavailable",
            "",
            "this device has no identity to observe lifecycle metadata; nothing applied",
        )
    sender = str(getattr(link, "device_id", "") or "")
    claimed = str(frame.get("from_device") or "")
    if not sender or (claimed and claimed != sender):
        return _announce_error(
            "identity_mismatch", "", "the announcement's sender is not the authenticated device"
        )
    try:
        validate_lifecycle_frame(frame)
        entries = parse_entries(
            frame, required=("state", "material_generation", "login_chain_revision")
        )
        if any(item["state"] not in _ANNOUNCE_STATES for item in entries):
            raise ValueError("lifecycle announcement carries a state an owner does not push")
    except ValueError as exc:
        return _announce_error("malformed_announce", "", str(exc))
    if frame.get("recipient_device") != self_device:
        return _announce_error(
            "identity_mismatch",
            "",
            "the announcement is addressed to another device; nothing applied",
        )
    if frame.get("owner_device") != sender:
        return _announce_error(
            "not_owner",
            "",
            "the announcement names an owner other than its sender; nothing applied",
        )
    network_id = str(getattr(link, "network_id", "") or "")
    document = (
        PlacementDocument.load(network_id, root, self_device=self_device)
        if network_id
        else PlacementDocument.resolve(root, self_device=self_device)
    )
    if document is None:
        return _announce_error(
            "unknown_key", "", "this device holds no sharing list; nothing applied"
        )
    providers: dict[str, str] = {}
    for item in entries:
        entry = document.entry(item["credential_key"])
        if entry is None:
            return _announce_error(
                "unknown_key", item["credential_key"], "no such shared credential; nothing applied"
            )
        if str(entry.owner_device) != sender:
            return _announce_error(
                "not_owner",
                item["credential_key"],
                "the announcement came from a device that does not own this credential; "
                "nothing applied",
            )
        providers[item["credential_key"]] = str(entry.provider or item["credential_key"])
    life = LifecycleStore(document.network_id, root)
    results: list[dict[str, Any]] = []
    for item in entries:
        key, scope = item["credential_key"], item["account_scope_ref"]
        result = life.observe(
            Observation(
                sender, key, scope, item["login_chain_revision"], item["material_generation"]
            )
        )
        if result.reason == NEEDS_AUTHORITATIVE_BASELINE:
            return _announce_error(
                NEEDS_AUTHORITATIVE_BASELINE,
                key,
                "this device's lifecycle record needs an authoritative baseline; nothing applied",
            )
        applied = result.applied or (
            result.reason == OBSERVED_REPLAY and item["state"] in _RESTRICTIVE_STATES
        )
        if applied:
            effective.note_lifecycle_observation(root, providers[key], item["state"], owner=sender)
        results.append(
            {
                "credential_key": key,
                "account_scope_ref": scope,
                "applied": applied,
                "reason": result.reason,
            }
        )
    return {"kind": "ack", "entries": results}
