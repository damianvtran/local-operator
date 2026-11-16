"""Future-only fenced update operations: protocol ``update-operation.v2``.

WHY THIS EXISTS. Legacy recovery records (``.update-operations/<id>.json``) can
never be positively sealed: any process holding only ``(root, id)`` -- including
one rehydrated from a retained generation sidecar by an already-loaded,
immutable older build -- may still ``advance``/``record_occurrence``/
``note_capacity``/``advance_handoff`` them, and none of those mutators checks a
seal. A seal written into the same JSON would be preserved as an unknown key
while the legacy writer changed everything beneath it. Legacy finality is
therefore UNPROVABLE, and legacy records stay PROTECTED; nothing here changes
that (architect boundary, last-writer-authority note, alternative 3).

WHAT THIS PROTOCOL ADDS, for NEW operations only:

A. Address fence. Records live in ``.update-operations-v2/`` and generations get
   ``.update-operation-v2.json``. Legacy ``_operation_path`` always resolves
   ``.update-operations/`` and legacy ``operation_for_generation`` reads only
   ``.update-operation.json``, so no unchanged legacy API can address a v2
   record. A v2 id is NEVER dual-published into a legacy sidecar, and a legacy
   generation/record is NEVER migrated in place.
B. Discovery is observation, not a write right. The v2 sidecar is an observer
   locator; :func:`observe_generation` returns status and ``writer: None``. The
   only ways to obtain a :class:`WriterContext` are :func:`start_operation`
   (the coordinator) and :func:`enroll_writer` (explicit, durable, coordinator
   only, while enrollment is open). Register-before-effect: the context exists
   only after the slot is persisted.
C. One shared fence. Every v2 mutator goes through :func:`_fenced`, which under
   the EXISTING root transaction checks protocol, id, epoch, not-sealed, an
   admitted slot whose role matches, the slot still open, and the role's
   permission for this action. :func:`finish` irrevocably closes a slot; no API
   reopens a slot, reopens enrollment, or unseals.
D. One positive seal point. :func:`try_seal` seals only when enrollment is
   closed, every admitted slot is finished and every admitted obligation is
   terminal. PID liveness, age, phase and report outcome are never inputs.

WHAT IT DELIBERATELY DOES NOT DO. It does not wire the updater, reporter,
controller tail, scheduler or capacity note to v2 (those callers keep the legacy
protocol until a later, reviewed slice). It does not raise the 256 cap or create
a second capacity pool: v2 admission counts legacy AND v2 entries against the
same limit. It is a protocol fence among cooperating local code, not an OS
security boundary: a :class:`WriterContext` is a local handle, not a credential.
"""

from __future__ import annotations

import copy
import os
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from local_operator import install_transaction as _tx
from local_operator.install_transaction import (
    CAPACITY_STATES,
    OCCURRENCE_KINDS,
    OCCURRENCE_TERMINALS,
    InstallTransactionError,
)

#: Defined in ``install_transaction`` so the read-only capacity dry run shares it.
PROTOCOL = _tx.V2_PROTOCOL
#: A record's epoch is fixed at creation. A context minted for another epoch
#: (a future protocol revision) is refused rather than reinterpreted.
EPOCH = 1
#: Defined in ``install_transaction`` so legacy admission counts it too.
V2_OPERATION_DIR = _tx.V2_OPERATION_DIR
#: Both sidecar names live in ``install_transaction`` (one source of truth); this
#: module only READS the legacy one, to refuse mixed generations.
V2_SIDECAR_NAME = _tx.V2_SIDECAR_NAME
LEGACY_SIDECAR_NAME = _tx.LEGACY_SIDECAR_NAME
#: The repo's one platform question for owner checks (see ``secrets/protocol.py``):
#: ``os.getuid`` does not exist off POSIX, and a Windows ``st_uid`` is always 0,
#: so ownership is only compared where a uid means something. Elsewhere the
#: other refusals (symlink, type, identity, lock) still apply.
_UID_IS_MEANINGFUL = os.name == "posix"

#: Permissions are closed per role. ``finish`` is implicit for every role (a
#: writer must always be able to give up its own authority); ``try_seal`` is not
#: an action here because it never adds domain state (see :func:`try_seal`).
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "coordinator": frozenset(
        {
            "advance",
            "occurrence",
            "capacity",
            "handoff",
            "continuation",
            "enroll",
            "close_enrollment",
            "link",
        }
    ),
    "reporter": frozenset({"advance", "capacity", "continuation"}),
    "service_tail": frozenset({"advance", "handoff", "continuation"}),
    "scheduler": frozenset({"occurrence"}),
    "capacity": frozenset({"capacity"}),
}
FINISH_DISPOSITIONS = ("completed", "failed", "abandoned")
CONTINUATION_TERMINALS = ("done", "failed", "cancelled")
#: ``partial`` is deliberately NOT terminal: some pressure remains unresolved,
#: and requested-or-unknown capacity must never be sealed away.
CAPACITY_TERMINALS = ("moved", "reclaimed")
#: Generic ``advance`` may not touch handoff/occurrences/capacity/continuations:
#: whole-map replacement would bypass the domain ordering and the obligation
#: ledger (base ``advance(handoff=...)`` replaces the nested map wholesale).
ADVANCE_OBSERVATIONS = frozenset({"candidate", "current", "receipts", "failure"})
_RECORD_KEYS = frozenset(
    {
        "protocol",
        "epoch",
        "operation_id",
        "baseline",
        "target",
        "route",
        "phase",
        "candidate",
        "current",
        "receipts",
        "failure",
        "enrollment",
        "enrollment_closed_by",
        "writers",
        "obligations",
        "occurrences",
        "capacity",
        "handoff",
        "continuations",
        "linked_generations",
        "sealed",
        "created_at",
        "updated_at",
        "last_writer",
    }
)
_MAX_NAME = 128


class FencedOperationError(InstallTransactionError):
    """A v2 write was refused by the fence, or a v2 record is not trustworthy.

    Subclasses the shared refusal so existing CLI/TUI reporting keeps working;
    every refusal happens BEFORE persistence, so nothing was changed.
    """


@dataclass(frozen=True)
class WriterContext:
    """Explicit delegated write authority for ONE slot of ONE v2 operation.

    Never reconstructed from a sidecar or an id: only :func:`start_operation`
    and :func:`enroll_writer` return one. Carrying it does not grant authority
    by itself -- every use is re-checked against the durable slot, so a context
    whose slot finished (or whose operation sealed) is inert.
    """

    root: Path
    operation_id: str
    slot: str
    role: str
    epoch: int = EPOCH


def _valid_hex(identifier: object) -> bool:
    return (
        isinstance(identifier, str)
        and len(identifier) == 32
        and all(character in "0123456789abcdef" for character in identifier)
    )


def _valid_name(name: object) -> bool:
    return isinstance(name, str) and 0 < len(name) <= _MAX_NAME and ":" not in name


def _v2_path(root: Path, identifier: str, *, create: bool = False) -> Path:
    """Distinct namespace; same ownership/symlink refusals as the legacy path."""
    if not _valid_hex(identifier):
        raise FencedOperationError("fenced update operation identity is invalid")
    directory = root / V2_OPERATION_DIR
    if create and not directory.exists() and not directory.is_symlink():
        directory.mkdir(mode=0o700)
    _require_safe_directory(directory)
    return directory / f"{identifier}.json"


def _require_safe_directory(directory: Path) -> None:
    if directory.exists() or directory.is_symlink():
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
            raise FencedOperationError(
                "update operation directory is unsafe; no record was written"
            )


#: ONE shared count for both admission paths (legacy ``start_operation`` and
#: :func:`start_operation` here), so the 256 cap stays a single pool.
active_recovery_entries = _tx.active_recovery_entries


def _load(path: Path, identifier: str) -> dict[str, Any]:
    """Fail closed on anything that is not a well-formed v2 record of this id."""
    record = _tx.read_record(path)
    if record is None:
        raise FencedOperationError("fenced update operation is unreadable; preserve it")
    if (
        record.get("protocol") != PROTOCOL
        or record.get("operation_id") != identifier
        or not isinstance(record.get("epoch"), int)
        or not isinstance(record.get("writers"), dict)
        or not isinstance(record.get("obligations"), dict)
        or record.get("enrollment") not in ("open", "closed")
    ):
        raise FencedOperationError("fenced update operation has an untrusted shape; preserve it")
    return record


def start_operation(root: Path, *, baseline: str, target: str, route: str) -> WriterContext:
    """Admit a new v2 operation and its coordinator slot in ONE transaction."""
    root = _tx.canonical_root(root)
    identifier = uuid.uuid4().hex
    slot = uuid.uuid4().hex
    now = time.time()
    with _tx.transaction(root, operation="register fenced update operation") as held:
        held.validate(root)
        # Capacity first, so a refusal does not even create the v2 directory.
        # The SAME gate as legacy admission (rotation, then the shared cap), so
        # neither path can retire records the other would protect.
        _tx._admit(root, held)
        path = _v2_path(root, identifier, create=True)
        _tx._persist_operation(
            path,
            {
                "protocol": PROTOCOL,
                "epoch": EPOCH,
                "operation_id": identifier,
                "baseline": baseline,
                "target": target,
                "route": route,
                "phase": "queued",
                "candidate": "",
                "receipts": [],
                "enrollment": "open",
                "writers": {
                    slot: {
                        "role": "coordinator",
                        "epoch": EPOCH,
                        "state": "open",
                        "admitted_by": None,
                        "admitted_at": now,
                    }
                },
                "obligations": {},
                "occurrences": {},
                "continuations": {},
                "linked_generations": [],
                "sealed": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        held.validate(root)
    return WriterContext(root, identifier, slot, "coordinator")


@contextmanager
def _fenced(context: WriterContext, action: str) -> Iterator[dict[str, Any]]:
    """THE shared fence: every v2 mutation passes here or does not happen.

    Checked under the existing root transaction, after reading the durable
    record, so a concurrent finish/seal linearises before or after this write,
    never beside it. The body mutates the yielded dict; it is persisted only if
    the body returns normally.
    """
    if action != "finish" and action not in ROLE_PERMISSIONS.get(context.role, frozenset()):
        raise FencedOperationError(f"writer role {context.role!r} may not {action}")
    root = _tx.canonical_root(context.root)
    with _tx.transaction(root, operation=f"fenced update operation {action}") as held:
        held.validate(root)
        path = _v2_path(root, context.operation_id)
        record = _load(path, context.operation_id)
        if record["epoch"] != context.epoch:
            raise FencedOperationError("writer epoch does not match the operation")
        if record.get("sealed"):
            raise FencedOperationError("fenced update operation is sealed; no further writes")
        slot = record["writers"].get(context.slot)
        if (
            not isinstance(slot, dict)
            or slot.get("role") != context.role
            or slot.get("epoch") != context.epoch
        ):
            raise FencedOperationError("writer is not admitted to this operation")
        if slot.get("state") != "open":
            raise FencedOperationError("writer finished; its write authority is revoked")
        yield record
        record["updated_at"] = time.time()
        record["last_writer"] = context.slot
        _tx._persist_operation(path, record)
        held.validate(root)


def _open_obligation(record: dict[str, Any], name: str, kind: str, owner: str) -> None:
    record["obligations"][name] = {
        "kind": kind,
        "owner": owner,
        "state": "pending",
        "outcome": "",
        "updated_at": time.time(),
    }


def _close_obligation(record: dict[str, Any], name: str, outcome: str) -> None:
    entry = record["obligations"][name]
    entry.update(state="terminal", outcome=outcome, updated_at=time.time())


def advance(context: WriterContext, phase: str, **observations: Any) -> None:
    """Phase/observation write; never a domain map (see ADVANCE_OBSERVATIONS)."""
    if phase not in _tx._OPERATION_PHASES:
        raise FencedOperationError("update operation phase has no verified completion contract")
    if set(observations) - ADVANCE_OBSERVATIONS:
        raise FencedOperationError("fenced advance cannot replace domain state; use its owning API")
    with _fenced(context, "advance") as record:
        record.update(observations)
        record["phase"] = phase


def enroll_writer(context: WriterContext, role: str) -> WriterContext:
    """Durably admit a delegated writer BEFORE it may act (register-before-effect)."""
    if role not in ROLE_PERMISSIONS or role == "coordinator":
        raise FencedOperationError("delegated writer role has no verified contract")
    slot = uuid.uuid4().hex
    with _fenced(context, "enroll") as record:
        if record["enrollment"] != "open":
            raise FencedOperationError("writer enrollment is closed; nothing was admitted")
        record["writers"][slot] = {
            "role": role,
            "epoch": record["epoch"],
            "state": "open",
            "admitted_by": context.slot,
            "admitted_at": time.time(),
        }
    return WriterContext(context.root, context.operation_id, slot, role, context.epoch)


def close_enrollment(context: WriterContext) -> None:
    """Irrevocable: no API sets enrollment back to ``open``."""
    with _fenced(context, "close_enrollment") as record:
        record["enrollment"] = "closed"
        record.setdefault("enrollment_closed_by", context.slot)


def record_occurrence(
    context: WriterContext,
    occurrence_id: str,
    *,
    kind: str,
    state: str,
    terminal: str = "",
) -> None:
    """Occurrence lifecycle as an owned obligation: running opens, terminal closes."""
    if not _valid_name(occurrence_id):
        raise FencedOperationError("occurrence identity is invalid")
    if kind not in OCCURRENCE_KINDS:
        raise FencedOperationError("occurrence kind has no durable contract")
    if state not in ("running", "terminal"):
        raise FencedOperationError("occurrence state has no durable contract")
    if state == "terminal" and terminal not in OCCURRENCE_TERMINALS:
        raise FencedOperationError("terminal occurrence needs a terminal outcome")
    name = f"occurrence:{occurrence_id}"
    with _fenced(context, "occurrence") as record:
        existing = record["obligations"].get(name)
        if state == "running":
            if existing is not None:
                raise FencedOperationError("occurrence already recorded; refusing to reopen it")
            _open_obligation(record, name, "occurrence", context.slot)
        else:
            if (
                not isinstance(existing, dict)
                or existing.get("owner") != context.slot
                or existing.get("state") != "pending"
            ):
                raise FencedOperationError("only the running occurrence's owner may end it")
            _close_obligation(record, name, terminal)
        record["occurrences"][occurrence_id] = {
            "kind": kind,
            "state": state,
            "terminal": terminal,
            "writer": context.slot,
            "recorded_at": time.time(),
        }


def note_capacity(
    context: WriterContext,
    *,
    state: str,
    threshold: str,
    cohort: Sequence[str] = (),
) -> dict[str, Any]:
    """Capacity as one owned obligation; only ``moved``/``reclaimed`` close it."""
    if state not in CAPACITY_STATES:
        raise FencedOperationError("capacity state has no verified contract")
    if threshold not in ("count", "free", "both"):
        raise FencedOperationError("capacity threshold has no verified contract")
    name = "capacity"
    with _fenced(context, "capacity") as record:
        existing = record["obligations"].get(name)
        if isinstance(existing, dict) and existing.get("owner") != context.slot:
            raise FencedOperationError("capacity obligation belongs to another writer")
        if existing is None or state in ("requested", "partial"):
            # A new request after a resolution reopens the obligation while this
            # writer is still open: pressure seen later must block the seal.
            _open_obligation(record, name, "capacity", context.slot)
        if state in CAPACITY_TERMINALS:
            _close_obligation(record, name, state)
        raw_previous = record.get("capacity")
        previous: dict[str, Any] = raw_previous if isinstance(raw_previous, dict) else {}
        entry: dict[str, Any] = {
            **previous,
            "state": state,
            "threshold": threshold,
            "cohort": [str(item) for item in cohort],
            "requested_at": previous.get("requested_at") or time.time(),
            "recorded_at": time.time(),
            "writer": context.slot,
        }
        record["capacity"] = entry
        return copy.deepcopy(entry)


_HANDOFF_FIELDS = frozenset(
    {
        "nonce",
        "from_pid",
        "from_birth",
        "to_pid",
        "to_birth",
        "target_generation",
        "listener_ino",
        "control_fd_role",
        "inherited",
    }
)
_HANDOFF_REQUIRED: dict[str, str | None] = {
    "ready": None,
    "commit": "ready",
    "ack": "commit",
    "aborted": None,
}


def advance_handoff(context: WriterContext, state: str, **fields: Any) -> dict[str, Any]:
    """Legacy handoff ordering, plus the fence and an obligation until ack/abort.

    Successor and predecessor are different admitted slots in a real handoff, so
    ordering (not slot identity) governs who may advance it; the fence still
    requires each to be an open, admitted ``handoff``-permitted writer.
    """
    if state not in _HANDOFF_REQUIRED:
        raise FencedOperationError("handoff state has no verified contract")
    if set(fields) - _HANDOFF_FIELDS:
        raise FencedOperationError("handoff field is unsupported")
    with _fenced(context, "handoff") as record:
        raw_previous = record.get("handoff")
        previous: dict[str, Any] = raw_previous if isinstance(raw_previous, dict) else {}
        prior_state = previous.get("state")
        if prior_state != _HANDOFF_REQUIRED[state]:
            expected = _HANDOFF_REQUIRED[state] or "no prior handoff"
            raise FencedOperationError(
                f"handoff {state} requires {expected}; refusing out of order"
            )
        if state == "ready":
            _open_obligation(record, "handoff", "handoff", context.slot)
        elif state == "ack":
            _close_obligation(record, "handoff", "ack")
        elif state == "aborted":
            _open_obligation(record, "handoff", "handoff", context.slot)
            _close_obligation(record, "handoff", "aborted")
        merged: dict[str, Any] = dict(previous)
        merged.update(fields)
        merged["state"] = state
        merged["writer"] = context.slot
        record["handoff"] = merged
        return copy.deepcopy(merged)


def declare_continuation(context: WriterContext, name: str, *, detail: str = "") -> None:
    """Admit a pending continuation obligation owned by this writer."""
    if not _valid_name(name):
        raise FencedOperationError("continuation name is invalid")
    key = f"continuation:{name}"
    with _fenced(context, "continuation") as record:
        if key in record["obligations"]:
            raise FencedOperationError("continuation already declared; refusing to reopen it")
        _open_obligation(record, key, "continuation", context.slot)
        record["continuations"][name] = {"detail": str(detail)[:512], "writer": context.slot}


def resolve_continuation(context: WriterContext, name: str, *, outcome: str) -> None:
    """Only the declaring writer may resolve it: nobody guesses another's outcome."""
    if outcome not in CONTINUATION_TERMINALS:
        raise FencedOperationError("continuation outcome has no verified contract")
    key = f"continuation:{name}"
    with _fenced(context, "continuation") as record:
        existing = record["obligations"].get(key)
        if (
            not isinstance(existing, dict)
            or existing.get("owner") != context.slot
            or existing.get("state") != "pending"
        ):
            raise FencedOperationError("only the declaring writer may resolve a continuation")
        _close_obligation(record, key, outcome)


def link_generation(context: WriterContext, generation: Path) -> None:
    """Publish an OBSERVER locator; refuse any generation carrying a legacy sidecar.

    Dual publication would let legacy ``operation_for_generation`` (or this
    module) confuse the protocols; a legacy generation is never migrated.
    """
    root = _tx.canonical_root(context.root)
    with _fenced(context, "link") as record:
        if generation.is_symlink() or generation.parent != root / "generations":
            raise FencedOperationError(
                "candidate recovery is outside the canonical generation store"
            )
        info = generation.lstat()
        if not stat.S_ISDIR(info.st_mode) or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()):
            raise FencedOperationError("candidate generation is unsafe; nothing was linked")
        legacy = generation / LEGACY_SIDECAR_NAME
        if legacy.exists() or legacy.is_symlink():
            raise FencedOperationError(
                "generation carries a legacy recovery sidecar; never dual-publish"
            )
        locator = generation / V2_SIDECAR_NAME
        if locator.exists() or locator.is_symlink():
            existing = _tx.read_record(locator)
            if existing is None or existing.get("operation_id") != context.operation_id:
                raise FencedOperationError("generation already names another operation")
        else:
            _tx._persist_operation(
                locator,
                {"protocol": PROTOCOL, "operation_id": context.operation_id, "access": "observe"},
            )
        linked = record.get("linked_generations")
        linked = list(linked) if isinstance(linked, list) else []
        if str(generation) not in linked:
            linked.append(str(generation))
        record["linked_generations"] = linked


def seal_blockers(record: dict[str, Any], root: Path | None = None) -> list[str]:
    """Every reason the operation cannot be sealed; empty means sealable.

    Unknown fields, unknown writer/obligation shapes and a legacy alias of the
    same id all BLOCK: an unsupported consumer's obligation must stay pending,
    never be silently discharged.
    """
    blockers: list[str] = []
    for key in sorted(set(record) - _RECORD_KEYS):
        blockers.append(f"unknown_field:{key}")
    if record.get("enrollment") != "closed":
        blockers.append("enrollment_open")
    writers = record.get("writers")
    if not isinstance(writers, dict) or not writers:
        blockers.append("writers_unknown")
    else:
        for slot, entry in sorted(writers.items()):
            if not isinstance(entry, dict) or entry.get("role") not in ROLE_PERMISSIONS:
                blockers.append(f"writer_unknown:{slot}")
            elif entry.get("state") != "finished":
                blockers.append(f"writer_open:{slot}")
    obligations = record.get("obligations")
    if not isinstance(obligations, dict):
        blockers.append("obligations_unknown")
    else:
        for name, entry in sorted(obligations.items()):
            if not isinstance(entry, dict) or entry.get("state") not in ("pending", "terminal"):
                blockers.append(f"obligation_unknown:{name}")
            elif entry.get("state") != "terminal":
                blockers.append(f"obligation_pending:{name}")
    if root is not None and _valid_hex(record.get("operation_id")):
        legacy = root / _tx.OPERATION_DIR / f"{record['operation_id']}.json"
        if legacy.exists() or legacy.is_symlink():
            blockers.append("legacy_alias")
    return blockers


def _seal(record: dict[str, Any], slot: str) -> None:
    record["sealed"] = {
        "sealed_at": time.time(),
        "sealed_by": slot,
        "phase": record.get("phase"),
        "writers": {
            key: value.get("disposition") for key, value in sorted(record["writers"].items())
        },
        "obligations": {
            key: value.get("outcome") for key, value in sorted(record["obligations"].items())
        },
    }


def finish(context: WriterContext, disposition: str, *, seal: bool = False) -> dict[str, Any]:
    """Irrevocably close THIS writer's slot; optionally try the seal atomically.

    Pending obligations owned by this writer stay pending: finishing with them
    open makes the operation permanently unsealable (protected), never
    discharges them. A finishing coordinator also closes enrollment, because no
    other role may admit writers once it is gone.
    """
    if disposition not in FINISH_DISPOSITIONS:
        raise FencedOperationError("writer disposition has no verified contract")
    root = _tx.canonical_root(context.root)
    with _fenced(context, "finish") as record:
        record["writers"][context.slot].update(
            state="finished", disposition=disposition, finished_at=time.time()
        )
        if context.role == "coordinator":
            record["enrollment"] = "closed"
            record.setdefault("enrollment_closed_by", context.slot)
        blockers = seal_blockers(record, root)
        sealed = seal and not blockers
        if sealed:
            _seal(record, context.slot)
    return {"finished": True, "sealed": sealed, "blockers": blockers}


def try_seal(context: WriterContext) -> dict[str, Any]:
    """The ONLY positive finality transition, callable by an admitted slot.

    Unlike :func:`_fenced`, the caller's slot may already be finished: the last
    writer to finish is exactly who should seal. This is safe because the write
    adds only the seal, and only when no admitted writer can write any more and
    no obligation is pending -- it never adds domain state. Liveness of the
    calling process is irrelevant: authority was revoked logically, not by exit.
    """
    root = _tx.canonical_root(context.root)
    with _tx.transaction(root, operation="seal fenced update operation") as held:
        held.validate(root)
        path = _v2_path(root, context.operation_id)
        record = _load(path, context.operation_id)
        slot = record["writers"].get(context.slot)
        if (
            record["epoch"] != context.epoch
            or not isinstance(slot, dict)
            or slot.get("role") != context.role
        ):
            raise FencedOperationError("writer is not admitted to this operation")
        if record.get("sealed"):
            return {"sealed": True, "already": True, "blockers": []}
        blockers = seal_blockers(record, root)
        if blockers:
            return {"sealed": False, "already": False, "blockers": blockers}
        _seal(record, context.slot)
        record["updated_at"] = time.time()
        _tx._persist_operation(path, record)
        held.validate(root)
    return {"sealed": True, "already": False, "blockers": []}


def operation_status(root: Path, identifier: str) -> dict[str, Any] | None:
    """Observe only: no transaction, no lock creation, no liveness, no writer."""
    root = _tx.canonical_root(root)
    path = _v2_path(root, identifier)
    if not path.exists() and not path.is_symlink():
        return _archived_status(root, identifier)
    record = _load(path, identifier)
    return {
        **copy.deepcopy(record),
        "writer": None,
        "seal_blockers": [] if record.get("sealed") else seal_blockers(record, root),
    }


def _archived_status(root: Path, identifier: str) -> dict[str, Any] | None:
    """A rotated (archived) sealed record, read in place; ``None`` when absent.

    Rotation moves only SEALED records whose generations nothing points at, but
    those generations may still carry their observer locator. Without this,
    :func:`observe_generation` on such a generation would report a false
    "identity disagrees" instead of the finished operation. Read-only: no lock,
    no restore, no rotation; ``archived: True`` tells the caller where it is.
    """
    archive = _tx._archive_directory(root, create=False)
    found = _tx._archive_index(archive).get(identifier)
    if archive is None or found is None or found[1] != "v2":
        return None
    record = _load(archive / found[2], identifier)
    if not isinstance(record.get("sealed"), dict):
        raise FencedOperationError("archived update operation is not sealed; preserve it")
    return {**copy.deepcopy(record), "writer": None, "seal_blockers": [], "archived": True}


def observe_generation(generation: Path) -> dict[str, Any] | None:
    """Observer discovery from a v2 locator; never yields a writer.

    A generation with only a legacy sidecar is ignored (``None``): it belongs to
    the legacy protocol and gives no v2 finality. A generation with BOTH fails
    closed -- something dual-published, and neither side can be trusted.
    """
    legacy = generation / LEGACY_SIDECAR_NAME
    locator = generation / V2_SIDECAR_NAME
    if not locator.exists() and not locator.is_symlink():
        return None
    if legacy.exists() or legacy.is_symlink():
        raise FencedOperationError("generation carries both recovery protocols; preserve both")
    record = _tx.read_record(locator)
    if (
        record is None
        or record.get("protocol") != PROTOCOL
        or record.get("access") != "observe"
        or not _valid_hex(record.get("operation_id"))
    ):
        raise FencedOperationError("generation recovery locator is untrusted; preserve it")
    root = generation.parent.parent
    status = operation_status(root, record["operation_id"])
    if status is None or str(generation) not in (status.get("linked_generations") or []):
        raise FencedOperationError("candidate recovery identity disagrees; preserve operation")
    return status
