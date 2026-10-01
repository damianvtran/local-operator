"""The approval store: one durable, signed, single-use onboarding request per file.

WHAT THIS IS FOR (remote-onboarding design note §2, revision 3). An agent that
wants to onboard a remote device needs ONE gesture from the operator, made
remotely, answerable from the operator's own surfaces, and durable enough to
survive a daemon restart between the gesture and its execution. That is a
RECORD, not a live gate: ``<config>/network/approvals/<approval_id>.json``,
beside the mesh store's other durable objects and written with its conventions
(``network/store.py``: staged temp file, 0600, ``os.replace``). The parked CARD
stays a live gate (an in-process future in the runtime that owns the session);
this module is only ever the durable half.

FOUR CONTRACTS ARE FROZEN HERE, and each one names its shape so a later slice
cannot re-invent it:

* **F1, idempotency.** A ``request_id`` (minted by the requesting surface) is
  bound to ``request_digest = "sha256:" + sha256(jcs(immutable_request))``.
  Same id + same digest returns the existing record verbatim; same id +
  different digest refuses ``approval_request_conflict`` and writes nothing;
  a re-request after a terminal state returns that terminal record unchanged;
  a retry or a new target is a NEW ``request_id``. The 30-day prune leaves a
  TOMBSTONE in ``index.jsonl``, and re-minting a tombstoned id refuses
  ``approval_request_conflict`` for 180 days.
* **F2, the transition matrix.** :data:`_ALLOWED_TRANSITIONS` is the ONE table;
  every writer routes through :func:`_require_transition`, so "only the frozen
  matrix transitions; anything else fails closed" is a property of the data
  rather than of five call sites remembering.
* **F3, cross-process write discipline.** Three-plus processes read-modify-
  write these records (CLI, the desktop daemon's routes, the runner), so every
  mutation holds a per-record ``flock(LOCK_EX)`` across the read AND the write
  (:func:`_record_lock`) — ``store.py``'s own lock is in-process only. The
  decision is WRITE-ONCE (a second decision refuses ``approval_decision_conflict``),
  receipt appends never touch ``state``/``signature``, and the store is
  DEVICE-LOCAL: never synced, never copied by a move.
* **F4, the signed payload.** ``signed_payload`` composes the canonical,
  domain-separated, length-prefixed message the decision signature covers
  (domain tag beside the module's other tags in ``operator/verify.py``); the
  decision write verifies it against the LOCAL operator key, and
  :func:`verify_for_run` re-derives the digest and re-verifies before every
  step of an execution.

TWO KINDS RIDE ONE MACHINERY. ``device_onboard`` carries a ``device`` block
(host, user, transport, host-key fingerprint); ``local_authority`` bootstraps
operator authority on the reader's OWN machine and carries a ``machine`` block
instead, ``credential_ref: null`` and the receipt vocabulary
``proposed → consent → generated → installed → verified``. Badge, gesture,
receipts fold and retention are identical; only the steps and the consent
channel differ — which is what a kind is for.

STDLIB-ONLY, AND PROVEN SO. The badge is a cold directory scan by design and
the module is importable from surfaces that must not carry the harness;
``tests/unit/test_import_graph.py`` pins the property in a fresh interpreter.
The operator module (crypto, anchors) and the audit writer are imported
FUNCTION-LOCALLY: a listing must not pay for ``cryptography``, and nothing here
may raise through an audit write.

READS FOLD, WRITERS MATERIALIZE. ``expired`` is a PURE FOLD of
``(state, expires_at, now)``: a reader presents it the moment the window has
passed, and the first WRITER to observe it materializes the transition under
the lock (the asks-store discipline). No timer process exists, and the
retention sweep materializes it for records nothing else touches.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from secrets import token_bytes
from typing import Any, Iterator, Mapping

from local_operator.network.identity import network_root
from local_operator.network.types import MeshRefusal
from local_operator.network.wire import canonical_json, crockford

#: The record schema this module writes and reads. Bumped only with a migration.
SCHEMA = 1

APPROVALS_DIRNAME = "approvals"
INDEX_FILENAME = "index.jsonl"

#: The two kinds v1 mints. An OPEN enum: a record with another kind parses (the
#: fold helpers are kind-agnostic) but nothing here creates one.
KIND_DEVICE_ONBOARD = "device_onboard"
KIND_LOCAL_AUTHORITY = "local_authority"
KINDS: tuple[str, ...] = (KIND_DEVICE_ONBOARD, KIND_LOCAL_AUTHORITY)

STATE_REQUESTED = "requested"
STATE_APPROVED = "approved"
STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"
STATE_DENIED = "denied"
STATE_EXPIRED = "expired"
STATE_FAILED = "failed"
STATES: tuple[str, ...] = (
    STATE_REQUESTED,
    STATE_APPROVED,
    STATE_CONNECTING,
    STATE_CONNECTED,
    STATE_DENIED,
    STATE_EXPIRED,
    STATE_FAILED,
)

#: Terminal states (§2.4): never re-openable, never re-runnable. ``failed`` is
#: deliberately NOT one — it is retry-eligible until the window closes.
TERMINAL_STATES = frozenset({STATE_CONNECTED, STATE_DENIED, STATE_EXPIRED})

#: The frozen matrix's rows and nothing else: from-state → the states a writer
#: may move it to. Each caller additionally checks its own guard (signature,
#: receipts, expiry) and its own sentence; this table is only WHICH moves exist.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    # create-if-absent, digest-bound (§2.2): the record IS the requested state.
    STATE_REQUESTED: frozenset({STATE_APPROVED, STATE_DENIED, STATE_EXPIRED}),
    # deny allowed while NO receipt exists (nothing ran yet); the runner opens
    # ``connecting`` on its first credentialed step.
    STATE_APPROVED: frozenset({STATE_CONNECTING, STATE_DENIED, STATE_EXPIRED}),
    # MID-RUN DENY (frozen revision, slice (a) remediation): the operator may deny
    # while ``connecting``. The deny WRITE lands (write-once, first decision wins)
    # and the RUNNER observes it at its next step check and stops, the receipts
    # recording where — the one write that cannot take an executed step back.
    STATE_CONNECTING: frozenset({STATE_CONNECTED, STATE_FAILED, STATE_DENIED, STATE_EXPIRED}),
    # RETRY re-enters execution on the SAME record with a new run_id; a window
    # that passes with no retry expires; a failed-but-retryable request the
    # operator no longer wants is ABANDONED (denied, write-once, never reset).
    STATE_FAILED: frozenset({STATE_CONNECTING, STATE_DENIED, STATE_EXPIRED}),
}

#: One use, one window: 60 minutes by default (§2.1). The invite's own TTL is
#: the pairing's limit; this is the approval's.
DEFAULT_EXPIRY_S = 60 * 60.0

#: Retention (§2.4): terminal records pruned after 30 days; the tombstone row
#: (which guards the id) survives 180.
TERMINAL_PRUNE_AGE_S = 30 * 24 * 60 * 60.0
TOMBSTONE_PRUNE_AGE_S = 180 * 24 * 60 * 60.0

#: The frozen receipt vocabulary for the local bootstrap (§2.2, sibling kind).
#: ``device_onboard``'s steps come from the runner's own sequence (§3.3) and are
#: not frozen here.
LOCAL_AUTHORITY_STEPS: tuple[str, ...] = (
    "proposed",
    "consent",
    "generated",
    "installed",
    "verified",
)

#: The signed payload's domain tag lives in ``operator/verify.py`` beside every
#: other tag ("so every tag lives in one module"); imported function-locally by
#: :func:`signed_payload` to keep this module stdlib-only at import time. The
#: literal is repeated in ``operator.verify.APPROVAL_DOMAIN`` and a test pins the
#: two equal, which is what keeps the framing honest across the import seam.
APPROVAL_DOMAIN_LITERAL = b"lop-approval-v1\x00"


# ---------------------------------------------------------------------------
# Layout. Pure resolvers; ensure_* twins create (the store.py discipline)
# ---------------------------------------------------------------------------


def approvals_dir(root: Path | None = None) -> Path:
    """``<config>/network/approvals`` — the path, never created by a reader.

    A listing over a directory that is not there answers "nothing pending",
    which is the truth on a fresh install; writers use
    :func:`ensure_approvals_dir`.
    """
    return network_root(root) / APPROVALS_DIRNAME


def ensure_approvals_dir(root: Path | None = None) -> Path:
    """``<config>/network/approvals``, created 0700 down the chain — WRITERS ONLY."""
    from local_operator.network.identity import ensure_network_root

    ensure_network_root(root)
    path = approvals_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def record_path(approval_id: str, root: Path | None = None) -> Path:
    return approvals_dir(root) / f"{approval_id}.json"


def lock_path(approval_id: str, root: Path | None = None) -> Path:
    """The per-record flock target. One name per record, and the NAME is stable
    across processes — that is the whole mechanism (two opens of one file are
    two lock holders only if the path, and so the inode, agrees)."""
    return approvals_dir(root) / f"{approval_id}.lock"


def index_path(root: Path | None = None) -> Path:
    """The append-only tombstone index. Rows outlive the records they guard."""
    return approvals_dir(root) / INDEX_FILENAME


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def new_approval_id() -> str:
    """``ap_<crockford(8 bytes)>`` — 64 random bits, human-safe alphabet."""
    return f"ap_{crockford(token_bytes(8))}"


def new_request_id() -> str:
    """``req_<crockford(8 bytes)>`` — minted by the REQUESTING surface (§2.2)."""
    return f"req_{crockford(token_bytes(8))}"


def new_run_id() -> str:
    """``run_<crockford(8 bytes)>`` — one per execution attempt.

    A retry re-enters the SAME record with a NEW run id (§2.4), so the receipts
    of an earlier attempt stay readable as their own run rather than being
    attributed to the retry.
    """
    return f"run_{crockford(token_bytes(8))}"


# ---------------------------------------------------------------------------
# Locking and atomic writes (F3)
# ---------------------------------------------------------------------------


@contextmanager
def _record_lock(approval_id: str, root: Path | None = None) -> Iterator[None]:
    """Hold ``flock(LOCK_EX)`` on this record's lock file across a read-modify-write.

    WHY A CROSS-PROCESS LOCK AT ALL (F3): ``store.py``'s lock serialises threads
    of ONE process, and this store has three-plus writers (CLI approve/deny,
    the desktop daemon's routes, the runner's step transitions) whose whole job
    is to edit one record under a decision that must win exactly once. The
    kernel's flock is that lock; it is taken BEFORE the read, so the record a
    body edits cannot predate a write it never saw — the ``store.mutate``
    argument, one level down.

    ``fcntl`` is imported INSIDE, the same way ``asks/store.py`` does: the
    import is POSIX-only and this module must import on every platform.
    """
    import fcntl

    ensure_approvals_dir(root)
    target = lock_path(approval_id, root)
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_record(record: Mapping[str, Any], root: Path | None = None) -> Path:
    """Stage + 0600 + ``os.replace``, through the network store's ONE writer.

    The staging name, the mode and the replace are ``store._write_private_json``
    (the established in-package call — ``credentials/state.py`` and
    ``credentials/placement.py`` do the same), so this store adds the
    cross-process flock rather than a second write path to keep in step.
    """
    from local_operator.network import store

    ensure_approvals_dir(root)
    return store._write_private_json(record_path(str(record["approval_id"]), root), dict(record))


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The record: immutable fields, digest, folds
# ---------------------------------------------------------------------------


def _immutable_request(record: Mapping[str, Any]) -> dict[str, Any]:
    """The fields the request digest binds (§2.2, F1).

    Exactly ``{kind, requested_by, device|machine, what, credential_ref,
    created_at, expires_at}``: the device block for ``device_onboard``, the
    machine block for ``local_authority`` (the sibling kind REPLACES the block,
    §2.2). Everything else — state, receipts, the signature, the id itself — is
    mutable or informational and deliberately outside the digest.
    """
    block_key = "machine" if record.get("kind") == KIND_LOCAL_AUTHORITY else "device"
    return {
        "kind": record.get("kind"),
        "requested_by": record.get("requested_by"),
        block_key: record.get(block_key),
        "what": record.get("what"),
        "credential_ref": record.get("credential_ref"),
        "created_at": record.get("created_at"),
        "expires_at": record.get("expires_at"),
    }


def request_digest(record: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical JSON of the immutable request (F1).

    ``canonical_json`` is the mesh transcript's one serialisation (sorted keys,
    tight separators, UTF-8), so two implementations that disagree about
    nothing produce the same bytes — the property the whole idempotency rule
    rests on.
    """
    payload = canonical_json(_immutable_request(record))
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_adopting_window(
    candidate: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> str:
    """The candidate's digest rebuilt on the RECORD's own window — the retry rule.

    QA round 1, Q1: the mint surfaces re-derive their window
    (``created_at = now``, ``expires_at = created_at + window``) on every
    invocation — the CLI passes its own computed pair, so a byte-identical
    retry is impossible — and the F1 promise, "a surface that retries its own
    command gets its first record back", was unreachable through it. Resolution
    (the reviewer's second option, chosen for its simplicity): the WINDOW IS NOT
    PART OF THE IDEMPOTENCY INTENT. On a request-id hit the comparison adopts the
    STORED window (rebuilding the digest on it) after every other immutable has
    been compared; a matching intent returns the first record, and the window
    stays moored to the first acceptance of the id — a caller that needs a
    different window files a NEW request id, which is the same rule a changed
    host follows.
    """
    rebuilt = dict(candidate)
    rebuilt["created_at"] = existing.get("created_at", rebuilt.get("created_at"))
    rebuilt["expires_at"] = existing.get("expires_at", rebuilt.get("expires_at"))
    return request_digest(rebuilt)


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def presented(record: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    """The record as a READER sees it: ``expired`` folds in without a write.

    A pure function of ``(state, expires_at, now)`` — the asks-store discipline.
    A terminal state is never re-folded (''a deny is never reset and an expiry
    is never extended''), and a missing/zero ``expires_at`` cannot expire.
    """
    view = dict(record)
    moment = time.time() if now is None else now
    expires_at = view.get("expires_at")
    if (
        isinstance(expires_at, (int, float))
        and expires_at > 0
        and view.get("state") not in TERMINAL_STATES
        and moment >= float(expires_at)
    ):
        view["state"] = STATE_EXPIRED
    return view


def _load_raw(approval_id: str, root: Path | None = None) -> dict[str, Any]:
    """The file's own content, unfolded. Raises ``unknown_approval`` when absent.

    A record that fails to parse is ALSO unknown: this store is device-local and
    its writer always writes a whole staged file, so a torn read here means
    somebody edited the file by hand — and answering "no such approval" is the
    fail-closed direction (a corrupt record must not be re-interpreted as a
    valid authority).
    """
    data = _read_json(record_path(approval_id, root))
    if not isinstance(data, dict) or data.get("approval_id") != approval_id:
        # Design round 1, D5: a pointer, not a command — this sentence travels
        # verbatim to the desktop route's ``{code, message}``, where "type this
        # verb" is not an action the reader can take.
        raise MeshRefusal(
            "unknown_approval",
            f"no approval {approval_id!r} on this device — it may already have been "
            "answered, denied or expired; the pending list shows what is here",
        )
    return data


def badge_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One record in the FROZEN list/badge shape (§3.5), shared by every surface.

    The frozen key spells the where-block ``device`` for ``device_onboard``; a
    ``local_authority`` record carries ITS block under ``machine`` — the kind is
    the difference a reader keys on, rather than a renamed key hiding which
    machine the block is about. One builder, so the CLI and the desktop route
    cannot drift into two spellings of the same read.
    """
    row: dict[str, Any] = {
        "approval_id": record.get("approval_id"),
        "state": record.get("state"),
        "what": record.get("what") or {},
        "requested_by": record.get("requested_by") or {},
        "expires_at": record.get("expires_at"),
    }
    row["machine" if "machine" in record else "device"] = (
        record.get("machine") or record.get("device") or {}
    )
    return row


def load_record(
    approval_id: str, *, root: Path | None = None, now: float | None = None
) -> dict[str, Any]:
    """The folded view of one record. Never creates anything, never writes."""
    return presented(_load_raw(approval_id, root), now)


def list_records(*, root: Path | None = None, now: float | None = None) -> list[dict[str, Any]]:
    """Every record, folded, oldest first. The cold scan the badge is built on.

    A missing directory answers ``[]`` (the ``store.networks_dir`` rule: a
    reader must not be the reason ``network/`` appears). Unreadable files are
    skipped rather than fatal — a listing is how a person finds out something is
    wrong, so it must be the last surface to break.
    """
    directory = approvals_dir(root)
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("ap_*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("approval_id"), str):
            records.append(presented(data, now))
    records.sort(key=lambda row: float(row.get("created_at") or 0.0))
    return records


def _find_by_request_id(request_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """The live record for a request id, or ``None`` — the idempotency lookup.

    A scan, not an index: the directory is small by design (the flat-store
    discipline, §2.3) and only CREATE pays it.
    """
    directory = approvals_dir(root)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("ap_*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("request_id") == request_id:
            return data
    return None


# ---------------------------------------------------------------------------
# The tombstone index
# ---------------------------------------------------------------------------


def _append_index_row(row: Mapping[str, Any], root: Path | None = None) -> None:
    """Append one tombstone row under the index's own flock, ``O_APPEND``.

    The index has MANY writers (every prune) and one reader (create's
    tombstone guard), so it gets the same discipline the ask log does: a whole
    line per write, appended at the end, torn lines skipped by the reader.
    """
    import fcntl

    ensure_approvals_dir(root)
    target = index_path(root)
    lock = target.with_name(f".{INDEX_FILENAME}.lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line)
    finally:
        os.close(fd)


def read_index(root: Path | None = None) -> list[dict[str, Any]]:
    """Every parseable tombstone row. A missing index answers ``[]``."""
    path = index_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _tombstone_for(
    request_id: str, root: Path | None = None, *, now: float | None = None
) -> dict[str, Any] | None:
    """A tombstone guarding ``request_id``, or ``None`` once it has aged out.

    The 180-day rule is the caller's of :func:`_tombstone_for` rather than a
    separate sweep: a row older than :data:`TOMBSTONE_PRUNE_AGE_S` reads as
    absent (the id may be re-minted), and :func:`sweep` is what physically drops
    it. Both agree because both compare against the same constant.
    """
    moment = time.time() if now is None else now
    for row in read_index(root):
        if row.get("request_id") != request_id:
            continue
        pruned_at = row.get("pruned_at")
        if (
            isinstance(pruned_at, (int, float))
            and moment - float(pruned_at) < TOMBSTONE_PRUNE_AGE_S
        ):
            return row
    return None


# ---------------------------------------------------------------------------
# Creating a request (F1)
# ---------------------------------------------------------------------------

_SURFACE_LIMIT = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MeshRefusal("approval_invalid", message)


def create_request(
    *,
    kind: str,
    request_id: str,
    requested_by: Mapping[str, Any],
    what: Mapping[str, Any],
    credential_ref: Mapping[str, Any] | None,
    device: Mapping[str, Any] | None = None,
    machine: Mapping[str, Any] | None = None,
    created_at: float | None = None,
    expires_at: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create the record, or answer the one this request id already made.

    THE IDEMPOTENCY RULES, exactly as frozen (§2.2), with the retry rule QA
    round 1 (Q1) pinned down: the lookup is by ``request_id``; a hit with the
    SAME digest returns the existing record verbatim (no merge, no state change
    — including a terminal state, so a re-request never resets a deny or an
    expiry); a hit whose immutables match apart from the WINDOW returns the
    first record too — the window is moored to the first acceptance of the id,
    so a retry cannot shift it (see ``_digest_adopting_window``); any OTHER
    difference refuses ``approval_request_conflict`` and writes nothing; a
    tombstoned id refuses the same way for 180 days. This function is NOT authority-increasing: it
    creates a REQUEST, and approve still needs the operator's signature (deny
    stays ordinary — §2.4).

    The digest is computed BEFORE the approval id is minted (the id is not part
    of the immutable request), so a surface that retries its own command gets
    its first record back — including a surface that re-derives its window per
    call, which is every shipped mint's shape.
    """
    _require(kind in KINDS, f"unknown approval kind {kind!r}; known: {', '.join(KINDS)}")
    _require(
        isinstance(request_id, str) and request_id.startswith("req_"),
        "the request id is minted by the requesting surface as 'req_<crockford>'",
    )
    _require(isinstance(requested_by, Mapping), "requested_by must be an object")
    _require(isinstance(what, Mapping), "what must be an object")
    surface = str(requested_by.get("surface") or "")
    _require(0 < len(surface) <= _SURFACE_LIMIT, "requested_by.surface must name the surface")
    if kind == KIND_DEVICE_ONBOARD:
        _require(isinstance(device, Mapping), "a device_onboard request needs a device block")
        _require(machine is None, "device_onboard carries a device block, not a machine block")
    else:
        _require(isinstance(machine, Mapping), "a local_authority request needs a machine block")
        _require(device is None, "local_authority carries a machine block, not a device block")

    moment = time.time()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "approval_id": "",  # minted below, after the digest
        "kind": kind,
        "request_id": request_id,
        "requested_by": dict(requested_by),
        "what": dict(what),
        "credential_ref": dict(credential_ref) if isinstance(credential_ref, Mapping) else None,
        "state": STATE_REQUESTED,
        "created_at": float(created_at if created_at is not None else moment),
        "decided_at": 0.0,
        "expires_at": 0.0,  # filled below: explicit value, or the default window
        "signature": None,
        "receipts": [],
        "audit": ["onboard_requested"],
    }
    if expires_at is not None:
        record["expires_at"] = float(expires_at)
    else:
        record["expires_at"] = record["created_at"] + DEFAULT_EXPIRY_S
    if kind == KIND_DEVICE_ONBOARD:
        record["device"] = dict(device or {})
    else:
        record["machine"] = dict(machine or {})
    record["request_digest"] = request_digest(record)

    # The create target's lock is keyed on the REQUEST id: the approval id does
    # not exist yet, and two concurrent creates of the same id must not both
    # mint one (the digest lookup is the thing being serialised).
    with _record_lock(request_id, root):
        existing = _find_by_request_id(request_id, root)
        if existing is not None:
            if existing.get("request_digest") == record["request_digest"]:
                return presented(existing, moment)
            if existing.get("request_digest") == _digest_adopting_window(record, existing):
                return presented(existing, moment)
            raise MeshRefusal(
                "approval_request_conflict",
                f"request {request_id} already exists with a different payload; "
                "a changed request is a NEW request id",
            )
        tomb = _tombstone_for(request_id, root, now=moment)
        if tomb is not None:
            raise MeshRefusal(
                "approval_request_conflict",
                f"request {request_id} was already used and pruned "
                f"({tomb.get('terminal_state', 'terminal')}); re-minting a spent id is refused",
            )
        for _ in range(5):
            approval_id = new_approval_id()
            if not record_path(approval_id, root).exists():
                break
        else:  # pragma: no cover — 64 random bits, five collisions is not a real case
            raise MeshRefusal("approval_invalid", "could not mint a free approval id")
        record["approval_id"] = approval_id
        _write_record(record, root)
    # RETENTION RIDES CREATION (§2.4): the sweep has no timer process, so the
    # one writer every approval flow passes through is where untouched records
    # get their due expiry materialized and 30-day terminals pruned. Best-effort
    # — retention must never be the reason a request cannot be filed.
    try:
        sweep(root=root)
    except Exception:  # noqa: BLE001 — see above; the record is already written
        pass
    _audit(
        "onboard_requested",
        actor="self",
        subject=record["approval_id"],
        root=root,
        detail={
            "kind": kind,
            "request_id": request_id,
            "surface": surface,
        },
    )
    return presented(record, moment)


# ---------------------------------------------------------------------------
# Signature contract (F4)
# ---------------------------------------------------------------------------


def _lp(value: str) -> bytes:
    """4-byte big-endian length + UTF-8 — ``operator/verify._lp``'s exact framing.

    Kept local so this module imports without the operator tree; the equality is
    not left to trust: a test signs through :func:`signed_payload` with a real
    key and verifies through ``operator.verify.verify_signature``, so a drift in
    this framer fails a test rather than shipping a second protocol.
    """
    raw = value.encode("utf-8", "surrogatepass")
    return len(raw).to_bytes(4, "big") + raw


def decided_at_text(value: float) -> str:
    """UTC epoch seconds as ``"%.6f"`` — the ONE textual form of ``decided_at``.

    The signature covers the TEXT, so the signer and every later verifier must
    render the same float the same way; rounding a timestamp is how a valid
    signature goes unexplained.
    """
    return f"{float(value):.6f}"


def signed_payload(
    *,
    kind: str,
    request_id: str,
    request_digest: str,
    decision: str,
    decided_at: float,
) -> bytes:
    """The canonical decision message (§2.4, F4). THE one builder.

    ``b"lop-approval-v1\\x00" || _lp(kind) || _lp(request_id) ||
    _lp(request_digest) || _lp(decision) || _lp(decided_at_text(decided_at))``

    A NEW versioned domain beside ``lop-operator-v1``/``lop-operator-device-v1``
    (both in ``operator/verify.py``), so a signature from any other protocol can
    never be replayed here, and the request digest binds every immutable field —
    including ``expires_at``, which is what makes an extension of the window
    detectable.
    """
    from local_operator.operator.verify import APPROVAL_DOMAIN

    if APPROVAL_DOMAIN != APPROVAL_DOMAIN_LITERAL:  # pragma: no cover — a drift guard
        raise MeshRefusal("approval_invalid", "the approval domain tag drifted between modules")
    return (
        APPROVAL_DOMAIN
        + _lp(str(kind))
        + _lp(str(request_id))
        + _lp(str(request_digest))
        + _lp(str(decision))
        + _lp(decided_at_text(decided_at))
    )


#: The refusal every surface raises when this machine holds no operator key to
#: sign with. USER-VISIBLE COPY: it names the product action (the local setup,
#: §3.7) and no terminal command (§2.9).
NO_SIGNING_SURFACE_SENTENCE = (
    "approving needs the operator key's consent, and operator authority is not set "
    "up on this machine yet — ask Local Operator to set it up for this machine (one "
    "approval and one admin password prompt), then approve again"
)


def sign_decision(
    *, kind: str, request_id: str, request_digest: str, decided_at: float, decision: str = "approve"
) -> str:
    """Sign one approval decision with the LOCAL operator key. This call PROMPTS.

    THE `lop operator sign` PATH, not a second one: the same signer resolution
    (the anchor names the backend), the same ``sign_message`` over the canonical
    payload, the same gesture where the host offers one. Every surface that
    answers a card — the CLI verb and the desktop route today — goes through
    here, so "who may approve" has one implementation to audit.

    A host with no usable key refuses BEFORE any prompt with
    :data:`NO_SIGNING_SURFACE_SENTENCE`; a gesture that failed or timed out
    carries the backend's own reason. Both are ``approval_signing_unavailable``
    because the surface's next move depends on "could not sign", not on which
    of the two it was.
    """
    message = signed_payload(
        kind=kind,
        request_id=request_id,
        request_digest=request_digest,
        decision=decision,
        decided_at=decided_at,
    )
    from local_operator.operator.keychain import KeyBackendError
    from local_operator.operator.sign import (
        load_signer,
        resolve_backend_name,
        sign_message,
    )
    from local_operator.paths import config_dir

    root = config_dir()
    signer = None
    try:
        signer = load_signer(config_root=root, backend_name=resolve_backend_name(root))
    except KeyBackendError:
        signer = None
    if signer is None:
        raise MeshRefusal("approval_signing_unavailable", NO_SIGNING_SURFACE_SENTENCE)
    try:
        return sign_message(signer, message, timeout=None).sig
    except KeyBackendError as exc:
        raise MeshRefusal(
            "approval_signing_unavailable",
            f"the operator key did not sign, so nothing was approved: {exc}",
        ) from exc
    finally:
        signer.close()


def spki_fingerprint(spki: bytes) -> str:
    """The human-comparable fingerprint of a public key: ``9A3C-12EF-3456``.

    Three groups of four uppercase hex characters from ``sha256(spki)`` — the
    shape the card renders beside "matches your operator key". Derived, never
    stored: the stored value is the digest itself.
    """
    digest = hashlib.sha256(bytes(spki)).hexdigest().upper()
    return f"{digest[0:4]}-{digest[4:8]}-{digest[8:12]}"


def local_anchor_trio(root: Path | None = None) -> dict[str, Any] | None:
    """``{key_id, spki_fp, statement_digest, spki}`` from the LOCAL operator store.

    The provenance trio of §2.5/§2.6, derived in ONE place so mint, approve and
    the runner cannot disagree: the installed (root-owned, usable) anchor wins;
    otherwise the STAGED statement — which is what the local bootstrap has
    between ``init`` and the privileged install, and what its install step
    consumes. ``None`` when neither exists: there is nothing to plant or to
    verify against. The staged read resolves against ``root`` (the ambient
    config dir when omitted), so a caller acting on an explicit store root
    reads THAT root's statement.
    """
    from local_operator.operator.trust import (
        anchor_bytes,
        load_anchor,
        load_staged_anchor,
    )
    from local_operator.paths import config_dir

    loaded = load_anchor()
    anchor = loaded.anchor if loaded.usable else None
    if anchor is None:
        anchor = load_staged_anchor(root if root is not None else config_dir())
    if anchor is None:
        return None
    statement = anchor_bytes(anchor)
    return {
        "key_id": anchor.key_id,
        "spki_fp": spki_fingerprint(anchor.spki),
        "statement_digest": "sha256:" + hashlib.sha256(statement).hexdigest(),
        "spki": anchor.spki,
    }


def _verify_decision_signature(
    record: Mapping[str, Any],
    *,
    decision: str,
    decided_at: float,
    signature_hex: str,
    spki: bytes,
) -> None:
    """Verify a decision signature or refuse ``approval_signature_invalid``."""
    from local_operator.operator.verify import verify_signature

    try:
        signature = bytes.fromhex(str(signature_hex))
    except ValueError:
        signature = b""
    message = signed_payload(
        kind=str(record.get("kind")),
        request_id=str(record.get("request_id")),
        request_digest=str(record.get("request_digest")),
        decision=decision,
        decided_at=decided_at,
    )
    if not verify_signature(spki=spki, message=message, signature=signature):
        raise MeshRefusal(
            "approval_signature_invalid",
            "the decision signature did not verify against this machine's operator key; "
            "nothing was written",
        )


# ---------------------------------------------------------------------------
# Transitions (F2) — the ONE gate every writer routes through
# ---------------------------------------------------------------------------


def _require_transition(record: Mapping[str, Any], target: str) -> None:
    """Fail closed unless ``state → target`` is a frozen matrix row.

    Every mutation calls this FIRST (after materializing a due expiry), so "only
    §2.4's matrix transitions; anything else fails closed" cannot be forgotten
    at one of the writers.
    """
    state = str(record.get("state") or "")
    allowed = _ALLOWED_TRANSITIONS.get(state, frozenset())
    if target not in allowed:
        raise MeshRefusal(
            "approval_decision_conflict",
            f"an approval in state {state!r} cannot become {target!r}; the first decision wins",
        )


def _materialize_expiry(record: dict[str, Any], now: float, root: Path | None) -> bool:
    """A WRITER's view of the fold: if the window has passed, land ``expired``.

    Returns True when it changed the record (and wrote it). Only writers call
    this — readers fold without writing (:func:`presented`). The change is a
    matrix row (any live state → ``expired``), audited as ``onboard_expired``.
    """
    expires_at = record.get("expires_at")
    if (
        not isinstance(expires_at, (int, float))
        or expires_at <= 0
        or record.get("state") in TERMINAL_STATES
        or now < float(expires_at)
    ):
        return False
    _require_transition(record, STATE_EXPIRED)
    record["state"] = STATE_EXPIRED
    record["decided_at"] = float(now)
    record["audit"].append("onboard_expired")
    _write_record(record, root)
    _audit(
        "onboard_expired",
        actor="self",
        subject=str(record.get("approval_id")),
        root=root,
        detail={"kind": str(record.get("kind"))},
    )
    return True


def _record_decision(
    approval_id: str,
    *,
    decision: str,
    decided_at: float | None,
    signature_hex: str | None,
    root: Path | None,
) -> dict[str, Any]:
    """The shared spine of approve/deny: guards, signature, write-once, audit."""
    with _record_lock(approval_id, root):
        record = _load_raw(approval_id, root)
        moment = time.time()
        if _materialize_expiry(record, moment, root):
            raise MeshRefusal(
                "approval_expired",
                "this approval's window has passed; it cannot be answered — "
                "file a new request to try again",
            )
        state = str(record.get("state"))
        target = STATE_APPROVED if decision == "approve" else STATE_DENIED

        if state in TERMINAL_STATES:
            if state == STATE_EXPIRED:
                raise MeshRefusal(
                    "approval_expired",
                    "this approval's window has passed; it cannot be answered — "
                    "file a new request to try again",
                )
            if state == STATE_CONNECTED and decision == "deny":
                raise MeshRefusal(
                    "approval_already_connected",
                    "this approval is already connected; denying it now would not undo "
                    "anything — remove the device instead if that is the intent",
                )
            raise MeshRefusal(
                "approval_decision_conflict",
                f"this approval was already decided ({state}); the first decision wins, "
                "and a retry is a new request",
            )
        if decision == "deny" and state == STATE_CONNECTING:
            # MID-RUN DENY: the write lands (write-once) and the RUNNER stops at
            # its next step check (``verify_for_run`` refuses a denied record). A
            # receipt lands with the decision so the record itself shows where the
            # deny arrived — the steps already executed stay, above it.
            run_id = ""
            for prior in reversed(record.get("receipts") or []):
                if isinstance(prior, Mapping) and prior.get("run_id"):
                    run_id = str(prior["run_id"])
                    break
            record.setdefault("receipts", []).append(
                {
                    "run_id": run_id,
                    "step": "deny",
                    "at": float(decided_at if decided_at is not None else moment),
                    "ok": True,
                    "detail": "the operator denied this while it was running; the runner "
                    "stops at its next step check",
                    "digest": "",
                }
            )
        _require_transition(record, target)

        decided = float(decided_at if decided_at is not None else moment)
        signature: dict[str, Any] | None = None
        if decision == "approve":
            trio = local_anchor_trio(root)
            if trio is None:
                raise MeshRefusal(
                    "approval_anchor_unavailable",
                    "this machine has no operator key to verify an approval against; "
                    "finish operator setup here first",
                )
            anchor = record.get("what")
            anchor = anchor.get("anchor") if isinstance(anchor, Mapping) else None
            if not isinstance(anchor, Mapping):
                # FAIL CLOSED (R1-5 / QA Q2): every shipped mint records the trio
                # and the card's "matches your operator key" presumes it, so a
                # record WITHOUT provenance has nothing to compare — there is no
                # way to say the decision matches the request it claims to answer.
                # Refusing here points the same direction as a mismatch; approving
                # anyway was the one branch that skipped the comparison entirely.
                raise MeshRefusal(
                    "approval_anchor_missing",
                    "this request carries no operator-key provenance, so a decision cannot "
                    "be checked against it; nothing was written",
                )
            if (
                str(anchor.get("key_id") or "") != str(trio["key_id"])
                or str(anchor.get("spki_fp") or "") != str(trio["spki_fp"])
                or str(anchor.get("statement_digest") or "") != str(trio["statement_digest"])
            ):
                raise MeshRefusal(
                    "approval_anchor_mismatch",
                    "the operator key on this machine no longer matches the one this "
                    "request was filed for; nothing was written",
                )
            if not signature_hex:
                raise MeshRefusal(
                    "approval_signature_invalid",
                    "approving needs the operator's signature; this surface supplied none",
                )
            _verify_decision_signature(
                record,
                decision="approve",
                decided_at=decided,
                signature_hex=signature_hex,
                spki=trio["spki"],
            )
            signature = {
                "by": f"operator:{trio['key_id']}",
                "alg": "ES256",
                "sig": str(signature_hex),
                "key_id": str(trio["key_id"]),
                "cert": None,
            }
        elif signature_hex:
            # A signed DENY is accepted when a surface offers one (the payload
            # format carries ``deny`` too); it is never REQUIRED, because a deny
            # settles in the safe direction and must keep working from every
            # surface (§2.4).
            trio = local_anchor_trio(root)
            if trio is not None:
                _verify_decision_signature(
                    record,
                    decision="deny",
                    decided_at=decided,
                    signature_hex=signature_hex,
                    spki=trio["spki"],
                )
                signature = {
                    "by": f"operator:{trio['key_id']}",
                    "alg": "ES256",
                    "sig": str(signature_hex),
                    "key_id": str(trio["key_id"]),
                    "cert": None,
                }

        record["state"] = target
        record["decided_at"] = decided
        record["signature"] = signature
        record["audit"].append(f"onboard_{'approved' if decision == 'approve' else 'denied'}")
        _write_record(record, root)
        if decision == "approve" and signature is not None:
            # The audit carries WHICH public key the authorisation named (public
            # fingerprints only — never material): an incident review of "what
            # could be installed from this decision" starts here.
            planted = record.get("what")
            planted = planted.get("anchor") if isinstance(planted, Mapping) else None
            audit_detail: dict[str, Any] = {
                "kind": str(record.get("kind")),
                "key_id": str(signature.get("key_id") or ""),
                "spki_fp": (
                    str((planted or {}).get("spki_fp") or "")
                    if isinstance(planted, Mapping)
                    else ""
                ),
            }
        else:
            audit_detail = {"kind": str(record.get("kind"))}
        _audit(
            f"onboard_{'approved' if decision == 'approve' else 'denied'}",
            actor="self",
            subject=approval_id,
            root=root,
            detail=audit_detail,
        )
        return presented(record, moment)


def approve(
    approval_id: str,
    *,
    signature_hex: str,
    decided_at: float,
    root: Path | None = None,
) -> dict[str, Any]:
    """``requested → approved``: the operator's gesture, verified before the write.

    The signature is over :func:`signed_payload` with ``decision="approve"`` and
    the SAME ``decided_at`` the caller is about to store — it is verified
    against the local operator key BEFORE the file is touched
    (``approval_signature_invalid``), and the anchor provenance is re-derived
    and compared first (``approval_anchor_mismatch``). Write-once: any state
    that has already been decided refuses (``approval_decision_conflict``).
    """
    return _record_decision(
        approval_id,
        decision="approve",
        decided_at=decided_at,
        signature_hex=signature_hex,
        root=root,
    )


def deny(
    approval_id: str,
    *,
    decided_at: float | None = None,
    signature_hex: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """``requested|approved|connecting|failed → denied``: ordinary, write-once, safe direction.

    Allowed from ``requested`` and from ``approved`` any time before the runner's
    next step check; from ``connecting`` (the deny write lands and the RUNNER
    observes it and stops at its next step check, the receipts showing where);
    and from ``failed`` (an operator abandoning a failed-but-retryable request).
    Refused after ``connected`` (``approval_already_connected``) and on any
    already-decided record (``approval_decision_conflict``).
    """
    return _record_decision(
        approval_id,
        decision="deny",
        decided_at=decided_at,
        signature_hex=signature_hex,
        root=root,
    )


# ---------------------------------------------------------------------------
# The runner's half: run ids, receipts, terminal transitions
# ---------------------------------------------------------------------------


def begin_run(approval_id: str, *, run_id: str, root: Path | None = None) -> dict[str, Any]:
    """``approved|failed → connecting``: the runner opens (or retries) a run.

    A retry is the SAME record with a NEW ``run_id`` (§2.4); the receipts of
    earlier runs stay, so the record reads as a history rather than a reset.
    Refused with the record's own sentence for every other state.
    """
    _require(bool(str(run_id)), "a run needs a run id")
    with _record_lock(approval_id, root):
        record = _load_raw(approval_id, root)
        moment = time.time()
        if _materialize_expiry(record, moment, root):
            raise MeshRefusal(
                "approval_expired",
                "this approval's window has passed and it cannot run; file a new request",
            )
        state = str(record.get("state"))
        if state not in (STATE_APPROVED, STATE_FAILED):
            raise MeshRefusal(
                "approval_not_runnable",
                f"this approval is {state!r} and cannot run; "
                + (
                    "it still needs the operator's approval"
                    if state == STATE_REQUESTED
                    else "a terminal record is never re-runnable"
                ),
            )
        _require_transition(record, STATE_CONNECTING)
        record["state"] = STATE_CONNECTING
        _write_record(record, root)
        return presented(record, moment)


def append_receipt(
    approval_id: str,
    *,
    run_id: str,
    step: str,
    ok: bool,
    detail: str = "",
    digest: str = "",
    at: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Append one step receipt while a run is in flight.

    A receipt NEVER touches ``state``/``signature`` (F3) and is refused on a
    record that is not ``connecting`` — in particular on a decided record, with
    one exception baked into the design: the runner's own in-flight path, which
    is exactly what ``connecting`` is. ``detail`` is the step's fixed-vocabulary
    observation, never credential material.
    """
    _require(bool(str(step)), "a receipt needs a step name")
    with _record_lock(approval_id, root):
        record = _load_raw(approval_id, root)
        if str(record.get("state")) != STATE_CONNECTING:
            raise MeshRefusal(
                "approval_receipt_refused",
                f"a receipt can only land while this approval is running "
                f"(it is {record.get('state')!r})",
            )
        record.setdefault("receipts", []).append(
            {
                "run_id": str(run_id),
                "step": str(step),
                "at": float(at if at is not None else time.time()),
                "ok": bool(ok),
                "detail": str(detail)[:400],
                "digest": str(digest),
            }
        )
        _write_record(record, root)
        return presented(record)


def mark_failed(
    approval_id: str,
    *,
    run_id: str,
    step: str,
    detail: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """``connecting → failed``: a step failed; the receipt names it.

    ``failed`` is retry-eligible until the window closes (``begin_run``
    re-enters with a new run id) — which is why the failure receipt is written
    in the same locked mutation as the state.
    """
    with _record_lock(approval_id, root):
        record = _load_raw(approval_id, root)
        _require_transition(record, STATE_FAILED)
        record["state"] = STATE_FAILED
        record.setdefault("receipts", []).append(
            {
                "run_id": str(run_id),
                "step": str(step),
                "at": time.time(),
                "ok": False,
                "detail": str(detail)[:400],
                "digest": "",
            }
        )
        record["audit"].append("onboard_failed")
        _write_record(record, root)
        _audit(
            "onboard_failed",
            actor="self",
            subject=approval_id,
            root=root,
            detail={"kind": str(record.get("kind")), "step": str(step)},
        )
        return presented(record)


def mark_connected(
    approval_id: str,
    *,
    run_id: str,
    step: str = "verify",
    detail: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """``connecting → connected``: every step verified. TERMINAL.

    The record is never re-openable and never re-runnable after this (§2.2's
    single-use definition); the final receipt lands in the same mutation.
    """
    with _record_lock(approval_id, root):
        record = _load_raw(approval_id, root)
        _require_transition(record, STATE_CONNECTED)
        record["state"] = STATE_CONNECTED
        record.setdefault("receipts", []).append(
            {
                "run_id": str(run_id),
                "step": str(step),
                "at": time.time(),
                "ok": True,
                "detail": str(detail)[:400],
                "digest": "",
            }
        )
        record["audit"].append("onboard_connected")
        _write_record(record, root)
        _audit(
            "onboard_connected",
            actor="self",
            subject=approval_id,
            root=root,
            detail={"kind": str(record.get("kind"))},
        )
        return presented(record)


# ---------------------------------------------------------------------------
# Verification for the runner (F4, point 2)
# ---------------------------------------------------------------------------


def verify_for_run(approval_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Re-check a record before a step: the decision, then digest, then signature.

    THE ENFORCEMENT TOKEN CLAIM (§2.5) against a same-uid file: a launderer who
    edits the immutable fields changes the re-derived digest and is refused
    ``approval_record_tampered``; one who edits anything else cannot mint a new
    signature, so the re-verification fails. Returns the folded record when it
    holds; raises the typed refusal otherwise — the caller records a failed
    receipt naming the check and refuses to run.

    FIRST, the operator's DECISION: a deny that landed while the run was in
    flight refuses ``approval_denied`` here, which is the runner's stop signal
    between steps.
    """
    record = _load_raw(approval_id, root)
    # THE RUNNER'S DECISION CHECK (frozen revision, slice (a) remediation): a
    # deny may land mid-run, and the runner must observe it BEFORE every step.
    # ``denied`` is reachable from ``connecting`` now, so a step loop that calls
    # this between steps stops here rather than executing past the operator's
    # decision; the receipts on the record show where it stopped.
    if str(record.get("state")) == STATE_DENIED:
        raise MeshRefusal(
            "approval_denied",
            "this approval was denied while it was running; the runner stops here — "
            "no further step is executed",
        )
    derived = request_digest(record)
    if derived != str(record.get("request_digest") or ""):
        raise MeshRefusal(
            "approval_record_tampered",
            "this approval's request no longer matches the digest it was signed over; "
            "it will not be executed — file a new request",
        )
    signature = record.get("signature")
    if not isinstance(signature, Mapping) or not signature.get("sig"):
        raise MeshRefusal(
            "approval_record_tampered",
            "this approval carries no decision signature; it will not be executed",
        )
    trio = local_anchor_trio(root)
    if trio is None:
        raise MeshRefusal(
            "approval_anchor_unavailable",
            "this machine has no operator key to re-verify the approval against; "
            "finish operator setup here first",
        )
    message = signed_payload(
        kind=str(record.get("kind")),
        request_id=str(record.get("request_id")),
        request_digest=str(record.get("request_digest")),
        decision="approve",
        decided_at=float(record.get("decided_at") or 0.0),
    )
    from local_operator.operator.verify import verify_signature

    try:
        raw_sig = bytes.fromhex(str(signature.get("sig")))
    except ValueError:
        raw_sig = b""
    if not verify_signature(spki=trio["spki"], message=message, signature=raw_sig):
        raise MeshRefusal(
            "approval_record_tampered",
            "the decision signature no longer verifies against this machine's operator "
            "key; the approval will not be executed — file a new request",
        )
    return presented(record)


# ---------------------------------------------------------------------------
# Retention: the sweep (§2.4)
# ---------------------------------------------------------------------------


def _terminal_at(record: Mapping[str, Any]) -> float:
    """When a terminal record BECAME terminal, best-effort.

    ``decided_at`` for a decision or an expiry; otherwise the newest receipt
    (``connected`` is written with its final receipt in one mutation); otherwise
    ``created_at``. The fallback can only over-report age by the record's own
    lifetime, and the sweep's only use for it is a 30-day threshold.
    """
    decided = float(record.get("decided_at") or 0.0)
    if decided > 0:
        return decided
    receipts = record.get("receipts")
    if isinstance(receipts, list) and receipts:
        last = receipts[-1]
        if isinstance(last, Mapping) and isinstance(last.get("at"), (int, float)):
            return float(last["at"])
    return float(record.get("created_at") or 0.0)


def sweep(*, root: Path | None = None, now: float | None = None) -> dict[str, int]:
    """Materialize due expiries, prune 30-day-terminal records, age tombstones.

    NO TIMER PROCESS EXISTS by design (§2.4): this runs opportunistically from
    writer paths (create, below), and the counts it returns are for the caller's
    audit line. Best-effort by construction — a record locked by a live runner
    is skipped rather than waited on.
    """
    moment = time.time() if now is None else now
    materialized = pruned = 0
    directory = approvals_dir(root)
    if directory.is_dir():
        for path in sorted(directory.glob("ap_*.json")):
            data = _read_json(path)
            if not isinstance(data, dict) or not isinstance(data.get("approval_id"), str):
                continue
            approval_id = data["approval_id"]
            try:
                with _record_lock(approval_id, root):
                    record = _load_raw(approval_id, root)
                    if _materialize_expiry(record, moment, root):
                        materialized += 1
                    if record.get("state") in TERMINAL_STATES and (
                        moment - _terminal_at(record) >= TERMINAL_PRUNE_AGE_S
                    ):
                        _append_index_row(
                            {
                                "request_id": record.get("request_id"),
                                "request_digest": record.get("request_digest"),
                                "terminal_state": record.get("state"),
                                "pruned_at": moment,
                            },
                            root,
                        )
                        os.unlink(record_path(approval_id, root))
                        lock = lock_path(approval_id, root)
                        try:
                            os.unlink(lock)
                        except OSError:
                            pass
                        pruned += 1
            except MeshRefusal:
                continue
    # Tombstones age out at 180 days: rewrite the index without them. The
    # rewrite is atomic (staged replace), so a crash leaves the old index whole.
    rows = read_index(root)
    fresh = [
        row
        for row in rows
        if isinstance(row.get("pruned_at"), (int, float))
        and moment - float(row["pruned_at"]) < TOMBSTONE_PRUNE_AGE_S
    ]
    if len(fresh) != len(rows):
        from local_operator.network import store

        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in fresh)
        store._write_private_text(index_path(root), text)
    return {"materialized": materialized, "pruned": pruned, "tombstones": len(fresh)}


# ---------------------------------------------------------------------------
# Audit (best-effort; never raises)
# ---------------------------------------------------------------------------


def _audit(
    event: str,
    *,
    actor: str,
    subject: str,
    root: Path | None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """One mesh-audit event for a transition. Never raises.

    The audit log's own contract is "a lost record is worse than a dropped key"
    — and its writer already reports its own failures. A store mutation must not
    fail because an audit write did.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        log = AuditLog.from_config(root)
        try:
            log.record(
                AuditEvent(
                    event=event,
                    actor=actor,
                    subject=subject,
                    detail=dict(detail or {}),
                )
            )
        finally:
            log.close()
    except Exception:  # noqa: BLE001 — the mutation already landed; never raise here
        pass
