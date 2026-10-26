"""A durable metadata record for one cold replay, written behind the replay itself.

WHY THIS MODULE EXISTS. A cold facade's cost is dominated by the replay it reads
to build its FIRST frontend state: the parser walks the journal (to BOF when
there is no compaction boundary), materialises every ``Message``, and resolves
attachments — measured at 1,619 / 533 / 685 ms of backend CPU before the first
frame on 8 / 30 / 100 MB journals, with the journal read 6.1x / 2.2x / 2.1x per
cold open. The FRAME costs 3-8 ms. The desktop rebuilds that facade per request
(snapshot, ``/history``, the SSE frame), and the replay cache removes the repeat
WITHIN a process; this module is what survives a RESTART.

WHAT IS STORED, AND WHY IT IS NOT A SECOND ACCOUNTING FOLD. The record carries
exactly the four state values :func:`~local_operator.session.replay_cache.
build_cold_replay` already computes — the durable frontend checkpoint, the
opportunistic spend row, their journal order, and the usage seed — plus the
witness it was computed for. It is that function's own output, serialised; there
is no second derivation here and none may grow here. The MESSAGES are
deliberately not stored: they are the part that costs the parse and the part
that a stale record would be most dangerous to serve, and the design's bounded
slice (:file:`CORE-PROFILE-SUPPLEMENT.md` §2) defers them rather than caching
them.

WHAT THE WITNESS PROVES, AND WHAT IT DOES NOT — read this before trusting a hit.
The witness is the replay cache's own: the session directory, plus the journal's
``(st_ino, st_size, st_mtime_ns, st_ctime_ns)``, plus the request's cut and wanted
custom types. Equality of that tuple means the file has not been written since
the record was: an append moves size, an atomic replace moves the inode, and ANY
write — including a same-size interior rewrite whose mtime was then restored
(``Transcript._write_entries``' ``preserve_mtime``, or its rollback, which
truncates and restores both mtime and size) — moves ``st_ctime_ns``, which an
unprivileged process cannot set back. ctime is in the witness for exactly that
case: since the record carries the model SELECTION (``ColdMetadata.selection``),
a missed rewrite would serve a stale model, not merely stale roster facts
(review of #2960 reproduced it before ctime was added). The reader also
re-stats the journal AFTER reading the record, and refuses the record if the
file moved meanwhile, so a record read is never combined with a journal that
changed underneath it.

It is still NOT a byte proof. ctime is a kernel stamp, not a content digest: a
privileged writer can reset the clock, a filesystem may coarsen the stamp, and a
copy/restore tool can carry foreign stamps. Those are OUTSIDE this product's
writers and are not detected here. This record carries no durable
writer-maintained generation and no content digest, and nothing may present a
metadata hit as content-verified. Closing that residual properly needs a durable
generation committed with the journal and is a separate, larger change.

WRITE-BEHIND, NEVER LOAD-BEARING. The record is written by
:func:`~local_operator.session.replay_cache.build_cold_replay` after a replay
answer is final and its after-read recheck passed, inside a guard that swallows
every failure: a cache that cannot be written costs the next open a parse it was
already paying, and must never fail the open, the append or the turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.harness.types import Usage
from local_operator.session.compression import transcript_source
from local_operator.session.model_selection import (
    SelectionReading,
    StoredModelSelection,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, the import would be a cycle
    from local_operator.session.replay_cache import ReplayKey

logger = logging.getLogger(__name__)

#: Bumped when the record's shape changes. An older or unknown schema is a MISS,
#: never an upgrade in place: the eager replay recomputes and rewrites it, which
#: is the only migration that cannot produce a half-migrated record.
#:
#: 2: the record carries the journal's settled model SELECTION (``selection``).
#: A schema-1 record is a miss here and a schema-2 record is a miss to a schema-1
#: reader, so neither build can misread the other's shape.
COLD_METADATA_SCHEMA = 2

#: Beside the other derived per-session caches under ``<config>/cache`` — the
#: one place a user or an operator can clear without touching a durable store.
CACHE_DIRNAME = "cold_replay"

#: The allow-list ``code_requests.ledger`` applies to its own derived stems. An
#: id that cannot become a path component must not become one: a ``/`` or ``..``
#: here would write outside the cache directory, and the ids reaching this come
#: from a request rather than from anything this process minted.
_SESSION_STEM = re.compile(r"[A-Za-z0-9_-]{1,128}")


def cold_metadata_path(config_dir: str | Path, session_id: str) -> Path:
    """The record for one session's journal: one file, atomically replaced.

    ONE FILE PER SESSION rather than one per journal version: a keyed filename
    would accumulate a record for every append forever, and there is no index to
    sweep them. The version lives INSIDE the record, so a changed journal makes
    the old record useless (a miss) and the next replay replaces it in place.

    ``ValueError`` for an id that must not become a path component; both
    accessors beside this turn that into a miss or a skipped write, because a
    cache that cannot name its file has nothing to say about an open.
    """
    if not _SESSION_STEM.fullmatch(session_id or ""):
        raise ValueError(f"not a session id: {session_id!r}")
    return Path(config_dir) / "cache" / CACHE_DIRNAME / f"{session_id}.json"


def safe_metadata_path(config_dir: str | Path, session_id: str) -> Path | None:
    """``cold_metadata_path``, or ``None`` for an id that cannot name a file.

    The callers are replays and opens, which have a correct slow answer for
    every reason a cache cannot be addressed. Raising at the call site would
    turn "nothing is cached under this id" into a failed open.
    """
    try:
        return cold_metadata_path(config_dir, session_id)
    except ValueError:
        return None


@dataclass(frozen=True)
class ColdMetadataWitness:
    """What a record must describe to be usable: the journal version and the request.

    Both halves, for the reason :class:`~local_operator.session.replay_cache.
    ReplayKey` carries both: the same bytes answer differently under a different
    ``through_id`` or a different wanted-custom-type pair, so a record made for
    one request must never serve another.
    """

    #: The session directory exactly as ``replay_key`` spells it (``str(Path(...))``),
    #: so two spellings of one directory compare equal here as they do there.
    journal_dir: str
    inode: int
    size: int
    mtime_ns: int
    #: See the module docstring: the stamp that catches a same-size rewrite
    #: whose mtime was restored. A stored witness without it is a MISS.
    ctime_ns: int
    through_id: str | None
    strict_cut: bool
    checkpoint_types: tuple[str, ...]
    opportunistic_types: tuple[str, ...]

    def matches(self, other: "ColdMetadataWitness") -> bool:
        return self == other


@dataclass(frozen=True)
class ColdMetadata:
    """One journal version's cold facts, in the shape ``_read_transcript`` needs.

    The four state fields are the load-bearing part and are named as
    ``ColdReplay`` names them, because a caller installs them into the same four
    attributes (``_cold_checkpoint``, ``_cold_spend``, ``_cold_order``,
    ``_cold_seed_usage``). ``bytes_read`` is the evidence of the read that
    produced this record, and has two provenances: a record filed by the eager
    replay holds that replay's full journal read, while one filed by the bounded
    tail (``replay_cache.build_cold_tail_metadata``) holds only the tail's scan
    window (at most 4 MiB from EOF), NOT a full parse. Kept so a hit can report
    the work behind it rather than inventing one; nothing reads it back for a
    decision today, so do not treat it as the journal's size.
    """

    witness: ColdMetadataWitness
    checkpoint: dict[str, Any] | None = None
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)
    seed_usage: Usage | None = None
    bytes_read: int = 0
    #: The journal's model selection, read for EXACTLY this witness's version.
    #:
    #: WHY IT IS HERE: on a record HIT the cold open no longer parses the
    #: journal, and the selection read (settle scan + forward fold) became the
    #: largest CPU cost left in the open — 29.0 of 79.4 ms measured on a warm
    #: 32k-row open with no selection row. The witness already proves the journal
    #: has not moved since this was read, so a later ``selected_model`` append
    #: cannot be hidden by it: the append moves the size, the record misses.
    #:
    #: THREE STATES, and the outer ``None`` is NOT absence. ``None`` = NOT
    #: RECORDED (the writer had no reading for this version, or could not prove
    #: one) — the reader reads the journal exactly as before. A reading whose own
    #: ``selection`` is ``None`` is PROVED ABSENCE — the rules ran on these bytes
    #: and found nothing usable. A reading is also refused, back to the journal,
    #: when its ``rules`` fingerprint differs from this process's
    #: (``cold_model.resolve_conversation_model``): the answer depends on the
    #: provider registry, which a later build may have changed.
    selection: SelectionReading | None = None

    def to_payload(self) -> dict[str, Any]:
        witness = self.witness
        return {
            "schema": COLD_METADATA_SCHEMA,
            "witness": {
                "journal_dir": witness.journal_dir,
                "inode": witness.inode,
                "size": witness.size,
                "mtime_ns": witness.mtime_ns,
                "ctime_ns": witness.ctime_ns,
                "through_id": witness.through_id,
                "strict_cut": witness.strict_cut,
                "checkpoint_types": list(witness.checkpoint_types),
                "opportunistic_types": list(witness.opportunistic_types),
            },
            "checkpoint": self.checkpoint,
            "checkpoints": self.checkpoints,
            "order": self.order,
            # ``mode="json"`` so the record is plain JSON: a ``Usage`` carries
            # enums and nested models, and round-tripping its python-mode dump
            # would need a second envelope. json's float repr is shortest-round-
            # trip, so the value recovered is the value stored.
            "seed_usage": self.seed_usage.model_dump(mode="json") if self.seed_usage else None,
            "bytes_read": self.bytes_read,
            "selection": _selection_payload(self.selection),
        }


#: Every ``StoredModelSelection`` field, serialised by name. Listed rather than
#: taken from ``dataclasses.fields`` so a field added there is a deliberate
#: decision here (and a schema bump), not a silently-null column.
_SELECTION_FIELDS = (
    "provider",
    "model_id",
    "effort",
    "authoritative",
    "recovered",
    "boot_selector",
    "fast",
)


def _selection_payload(reading: SelectionReading | None) -> dict[str, Any] | None:
    if reading is None:
        return None
    value = reading.selection
    return {
        "rules": reading.rules,
        "source": list(reading.source) if reading.source is not None else None,
        "value": (
            None if value is None else {name: getattr(value, name) for name in _SELECTION_FIELDS}
        ),
    }


class _Invalid(Exception):
    """A stored selection whose shape this build does not write."""


def _selection_of(raw: Any, witness: ColdMetadataWitness) -> SelectionReading | None:
    """The stored reading, ``None`` when not recorded; ``_Invalid`` for a bad shape.

    STRICT, because a reading served from here replaces a journal read: every
    field must carry exactly the type the writer emits, and the reading's own
    source must name the witness's journal version (the writer only files one
    that does, so anything else is a record this build did not write).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"rules", "source", "value"}:
        raise _Invalid
    rules = raw["rules"]
    source = raw["source"]
    value = raw["value"]
    if not isinstance(rules, str):
        raise _Invalid
    stamp = (witness.inode, witness.size, witness.mtime_ns, witness.ctime_ns)
    if source != list(stamp):
        raise _Invalid
    if value is None:
        return SelectionReading(None, rules, stamp)
    if not isinstance(value, dict) or set(value) != set(_SELECTION_FIELDS):
        raise _Invalid
    if not isinstance(value["provider"], str) or not isinstance(value["model_id"], str):
        raise _Invalid
    for name in ("effort", "boot_selector"):
        if value[name] is not None and not isinstance(value[name], str):
            raise _Invalid
    for name in ("authoritative", "recovered"):
        if not isinstance(value[name], bool):
            raise _Invalid
    if value["fast"] is not None and not isinstance(value["fast"], bool):
        raise _Invalid
    return SelectionReading(
        StoredModelSelection(**{name: value[name] for name in _SELECTION_FIELDS}),
        rules,
        stamp,
    )


def witness_for(key: "ReplayKey") -> ColdMetadataWitness:
    """The witness a record must carry to answer ``key``'s request.

    THE ONE MAPPING, in one direction, and it exists because the two dataclasses
    are not the same type: ``ReplayKey.directory`` names the journal's directory
    and this module stores it as ``journal_dir``. Comparing the objects directly
    is how a record matches nothing at all — the fields all agree and the classes
    do not — so every caller converts here rather than comparing across them.

    ``ReplayKey`` is imported for the type only: ``replay_cache`` imports this
    module at run time, so the reverse import would be a cycle.
    """
    return ColdMetadataWitness(
        journal_dir=key.directory,
        inode=key.inode,
        size=key.size,
        mtime_ns=key.mtime_ns,
        ctime_ns=key.ctime_ns,
        through_id=key.through_id,
        strict_cut=key.strict_cut,
        checkpoint_types=tuple(key.checkpoint_types),
        opportunistic_types=tuple(key.opportunistic_types),
    )


def write_cold_metadata(path: Path, record: ColdMetadata) -> bool:
    """Persist ``record`` atomically. False when it could not land, never raises.

    PID-SUFFIXED TEMP + ``os.replace``, the idiom
    :mod:`local_operator.code_requests.ledger` measured: a fixed temp path has
    two writers of one session tearing each other's document, and the server and
    a TUI can both replay one journal. A reader therefore either sees the whole
    old record or the whole new one, never half of either.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(record.to_payload(), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        return True
    except (OSError, TypeError, ValueError):
        # TypeError/ValueError are reachable: ``json.dumps`` is being handed
        # parsed journal values, and a value it cannot serialise is a miss for
        # the next open rather than an error for this one.
        logger.debug("cold metadata: could not write %s", path, exc_info=True)
        return False


def read_cold_metadata(path: Path, expected: ColdMetadataWitness) -> ColdMetadata | None:
    """The record for ``expected``'s journal version, or ``None`` for a MISS.

    ``None`` is the ONLY failure mode and it always means "take the eager path":
    absent, unreadable, another schema, another request, another journal version,
    a journal that moved while the record was being read, or a payload that no
    longer validates. None of those is an error worth raising — the caller has a
    correct slow answer, and this record exists only to skip work the answer does
    not depend on.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        logger.debug("cold metadata: unreadable record %s", path, exc_info=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != COLD_METADATA_SCHEMA:
        return None
    # EVERY FIELD MUST BE PRESENT, including the ones whose value is legitimately
    # null. A record that lacks one was written by a build whose shape this one
    # does not know, and "the reading is absent" and "the build never stored a
    # reading" are different facts: the first is a valid answer (a pre-ledger
    # journal has no usage seed), the second is a record this code cannot read
    # and must not fill in from the default. Eager, always.
    if not _REQUIRED_FIELDS <= payload.keys():
        logger.debug("cold metadata: record %s predates this schema", path)
        return None
    witness = _witness_of(payload.get("witness"))
    if witness is None or not witness.matches(expected):
        # A record for another journal version or another request. Replaced by
        # the next replay of THIS request; a reader never rewrites it.
        return None
    # THE OTHER END OF THE READ, and the reason a hit is not a stat race: an
    # append that landed while this record was being read leaves the record
    # describing the version we asked about, which is exactly the version the
    # caller would then combine it with. Refusing here is what keeps a record
    # from ever being paired with a journal it does not describe.
    if not _journal_still_at(expected):
        return None
    checkpoint = payload.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, dict):
        return None
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, dict) or any(
        not isinstance(value, dict) for value in checkpoints.values()
    ):
        return None
    order = payload.get("order")
    if not isinstance(order, dict):
        return None
    try:
        seed_usage = (
            Usage.model_validate(payload["seed_usage"])
            if payload.get("seed_usage") is not None
            else None
        )
    except Exception:  # noqa: BLE001 — a stale reading is a miss, never a raise
        logger.debug("cold metadata: usage no longer validates in %s", path, exc_info=True)
        return None
    bytes_read = payload.get("bytes_read")
    try:
        selection = _selection_of(payload.get("selection"), witness)
    except _Invalid:
        # THE WHOLE RECORD is discarded, not just the selection, deliberately:
        # this writer never emits a malformed selection, so one here means the
        # file was written by another build or damaged, and its other fields
        # carry no better provenance than this one. Taking the eager path costs
        # one parse; trusting the rest of a record we cannot account for could
        # serve facts nobody computed for this journal.
        logger.debug("cold metadata: selection no longer validates in %s", path)
        return None
    return ColdMetadata(
        witness=witness,
        checkpoint=checkpoint,
        checkpoints=checkpoints,
        order={str(key): int(value) for key, value in order.items()},
        seed_usage=seed_usage,
        bytes_read=bytes_read if isinstance(bytes_read, int) else 0,
        selection=selection,
    )


#: The keys this build's records always carry. See the guard that uses them.
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "witness",
        "checkpoint",
        "checkpoints",
        "order",
        "seed_usage",
        "bytes_read",
        "selection",
    }
)


def _witness_of(raw: Any) -> ColdMetadataWitness | None:
    """The record's own witness, or ``None`` when the stored shape is wrong."""
    if not isinstance(raw, dict):
        return None
    journal_dir = raw.get("journal_dir")
    inode = raw.get("inode")
    size = raw.get("size")
    mtime_ns = raw.get("mtime_ns")
    ctime_ns = raw.get("ctime_ns")
    through_id = raw.get("through_id")
    strict_cut = raw.get("strict_cut")
    checkpoint_types = raw.get("checkpoint_types")
    opportunistic_types = raw.get("opportunistic_types")
    if not isinstance(journal_dir, str) or not isinstance(strict_cut, bool):
        return None
    if through_id is not None and not isinstance(through_id, str):
        return None
    if not isinstance(checkpoint_types, list) or not isinstance(opportunistic_types, list):
        return None
    if not all(isinstance(item, str) for item in checkpoint_types):
        return None
    if not all(isinstance(item, str) for item in opportunistic_types):
        return None
    # Narrowed one at a time rather than in a loop: a loop over the three does
    # not narrow any of them for a type checker, and these three are the witness
    # a record is accepted on.
    if not isinstance(inode, int) or isinstance(inode, bool):
        return None
    if not isinstance(size, int) or isinstance(size, bool):
        return None
    if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool):
        return None
    # A record from a build whose witness had no ctime (an unreleased schema-2
    # shape) is refused rather than defaulted: without the stamp the record
    # cannot rule out the same-size rewrite the stamp exists to catch.
    if not isinstance(ctime_ns, int) or isinstance(ctime_ns, bool):
        return None
    return ColdMetadataWitness(
        journal_dir=journal_dir,
        inode=inode,
        size=size,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        through_id=through_id,
        strict_cut=strict_cut,
        checkpoint_types=tuple(checkpoint_types),
        opportunistic_types=tuple(opportunistic_types),
    )


def _journal_still_at(witness: ColdMetadataWitness) -> bool:
    """Whether the journal is still at the version ``witness`` names.

    Through the ARCHIVE-AWARE resolver, not a plain-name probe: a session whose
    journal was compressed is a different physical file from the one the record
    describes (it records the plain journal's inode), so an archive is a MISS by
    construction — the record cannot describe it and must not be paired with it.
    Probing ``transcript.jsonl`` directly would answer the same thing here by
    accident, which is exactly the silent surface the archive-aware guard exists
    to close.
    """
    path, codec = transcript_source(Path(witness.journal_dir))
    if codec is not None:
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        stat.st_ino == witness.inode
        and stat.st_size == witness.size
        and stat.st_mtime_ns == witness.mtime_ns
        and stat.st_ctime_ns == witness.ctime_ns
    )


__all__ = [
    "CACHE_DIRNAME",
    "COLD_METADATA_SCHEMA",
    "ColdMetadata",
    "ColdMetadataWitness",
    "cold_metadata_path",
    "read_cold_metadata",
    "safe_metadata_path",
    "witness_for",
    "write_cold_metadata",
]
