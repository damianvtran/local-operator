"""The SQLite-backed secret store (design §4).

SQLite in WAL mode rather than one file per secret, and the reason is measured
rather than stylistic: the operator runs ~10 lop sessions at once, so the store
needs a concurrency protocol, and SQLite already has one that works. The design
spike put 10 concurrent readers doing 60 decrypts each against a live writer at
21 ms with zero errors; per-file storage would have meant inventing a lock
protocol here. ``tests/unit/secrets/test_concurrency.py`` re-runs that shape.

Names and descriptions are stored ENCRYPTED, not just values. A store listing
``MINERVA_PROD_DB_PASSWORD`` in the clear tells an attacker where to go next,
so the only plaintext-searchable column is the blind index — an HMAC of the
normalised name under a key derived from the master key. That is why ``list``
needs the key: enumerating which secrets exist is itself privileged.

This module knows nothing about *how* the master key was obtained. It is handed
one. That is the seam the broker PR slides into: today
:func:`local_operator.secrets.access.open_store` reads the key file directly,
and it will instead ask the broker, with this layer unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from local_operator.secrets import audit
from local_operator.secrets.crypto import (
    RECORD_FORMAT_VERSION,
    RecordMetadata,
    key_fingerprint,
    name_index,
    open_record,
    seal,
    validate_name,
)
from local_operator.secrets.errors import (
    IncompatibleStore,
    InvalidSecretName,
    SecretCorrupt,
    SecretExists,
    SecretNotFound,
    SecretStoreError,
    StaleKeyEpoch,
)
from local_operator.secrets.keys import (
    FILE_MODE,
    check_mode,
    ensure_secrets_dir,
    replace_master_key,
    store_path,
)

#: Layout version of the tables themselves. A runtime that finds a HIGHER value
#: refuses to write and refuses to read records whose format it does not know
#: (§13): version skew is routine here because the operator updates the runtime
#: while sessions are live, and a downward migration would be unrecoverable.
SCHEMA_VERSION = 1

#: Valid values of the ``kind`` column. ``file`` marks a secret whose plaintext
#: is a file's contents — materialised to a private temporary path for the
#: duration of one command by ``lop secret file`` (design §7).
KINDS = ("string", "file")

#: Descriptions are operator-written labels shown by ``list``; the ceiling only
#: has to keep a pathological paste from bloating every record.
MAX_DESCRIPTION_LENGTH = 1024

#: ``meta`` key holding the fingerprint of the master key the store is sealed
#: under. This is the store's KEY EPOCH, and it exists because ``BEGIN
#: IMMEDIATE`` serialises statements but not key epochs: a session that loaded
#: the key before a rotation would otherwise commit a record sealed under the
#: old key and indexed under the old index key, unreachable and undecryptable
#: forever, while reporting success. Every write compares its own key against
#: this row inside its own transaction and fails closed on a mismatch.
#:
#: Absent on a store created before this row existed. That case is treated as
#: "epoch unknown", not as a mismatch, and the next write backfills it —
#: refusing to write to a pre-existing store would be a worse failure than the
#: race this guards.
KEY_FINGERPRINT_KEY = "key_fingerprint"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v BLOB);
CREATE TABLE IF NOT EXISTS secrets(
  id             TEXT PRIMARY KEY,
  name_index     BLOB NOT NULL UNIQUE,
  key_generation INTEGER NOT NULL,
  format_version INTEGER NOT NULL,
  nonce          BLOB NOT NULL,
  ciphertext     BLOB NOT NULL,
  kind           TEXT NOT NULL,
  created_at     REAL NOT NULL,
  updated_at     REAL NOT NULL,
  last_used_at   REAL
);
CREATE TABLE IF NOT EXISTS audit(
  ts REAL NOT NULL, event TEXT NOT NULL, secret_id TEXT,
  session_id TEXT, pid INTEGER, exe TEXT, outcome TEXT,
  prev_hash BLOB, hash BLOB
);
"""

_RECORD_COLUMNS = (
    "id, name_index, key_generation, format_version, nonce, ciphertext, kind,"
    " created_at, updated_at, last_used_at"
)


@dataclass(frozen=True)
class SecretRecord:
    """A decrypted record, minus the value.

    Returned by ``list`` and ``describe``. The value is a separate call
    precisely so that no listing path can print one by accident: there is no
    field here to print.
    """

    record_id: str
    name: str
    description: str
    kind: str
    key_generation: int
    created_at: float
    updated_at: float
    last_used_at: float | None


def _connect(path: Path) -> sqlite3.Connection:
    """Open the database with the pragmas this store depends on.

    ``journal_mode=WAL`` is what lets ~10 sessions read while one writes.
    ``synchronous=FULL`` because this store is small and written rarely, so
    durability is worth more than write throughput — a secret that reports
    "stored" and is gone after a power cut is the worst outcome available.
    ``busy_timeout`` covers the brief writer-lock contention concurrent
    sessions do still produce; without it a racing writer gets an immediate
    ``database is locked`` instead of waiting the few milliseconds needed.

    ``isolation_level=None`` turns off the driver's implicit transaction
    management so the explicit ``BEGIN IMMEDIATE`` blocks below mean what they
    say — sqlite3's default would otherwise open its own transaction on the
    first DML statement and leave the read that preceded it outside.
    """
    connection = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


def _read_meta_int(connection: sqlite3.Connection, key: str, default: int) -> int:
    """Read an integer ``meta`` row, refusing a corrupted one cleanly.

    The conversion is guarded because ``meta.v`` is a BLOB column that SQLite
    will happily hold TEXT or NULL in: a hand-corrupted row made ``bytes()`` or
    ``int()`` raise something that is not a :class:`SecretStoreError`, which
    escaped the CLI's one-line error contract and printed a 29-line traceback
    into a command routinely run inside ``$( )``. The value is unusable either
    way; the only question is whether the operator gets a sentence or a stack.
    """
    row = connection.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    if row is None or row[0] is None:
        return default
    try:
        return int(bytes(row[0]).decode("ascii"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise IncompatibleStore(
            f"This secret store's {key!r} metadata is not a number ({row[0]!r}). "
            "The store's metadata is damaged; it cannot be read safely."
        ) from exc


def _read_meta_blob(connection: sqlite3.Connection, key: str) -> bytes | None:
    row = connection.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    if row is None or row[0] is None:
        return None
    return bytes(row[0])


def _write_meta_blob(connection: sqlite3.Connection, key: str, value: bytes) -> None:
    connection.execute(
        "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, value),
    )


def _write_meta_int(connection: sqlite3.Connection, key: str, value: int) -> None:
    connection.execute(
        "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, str(value).encode("ascii")),
    )


def _payload(name: str, description: str, value: bytes) -> bytes:
    """The plaintext sealed inside the ciphertext.

    JSON with the value hex-encoded rather than raw bytes: a value may be
    arbitrary binary (a service-account key, a DER certificate) and JSON cannot
    carry those. Hex over base64 only because it round-trips with no padding
    mode to get wrong; the 2x size cost is paid on a store measured in
    kilobytes.

    The name is in here as well as in the blind index because the index is
    one-way — this is the only copy of the label that can be read back for
    ``list``, and being inside the AEAD is what authenticates it.
    """
    return json.dumps(
        {"name": name, "description": description, "value": value.hex()},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def recorded_key_fingerprint(base: Path | None = None) -> bytes | None:
    """The fingerprint of the key the store on disk is currently sealed under.

    Read without a key, because the caller of this function is deciding WHICH
    key to use — it is how an interrupted rotation is detected and completed
    (see :func:`local_operator.secrets.access.resolve_master_key`). Returns
    ``None`` when there is no store, no fingerprint row (a store predating the
    row), or a damaged one: every one of those means "cannot tell", and the
    caller falls back to the installed key rather than guessing.
    """
    path = store_path(base)
    if not path.exists():
        return None
    try:
        with closing(_connect(path)) as connection:
            return _read_meta_blob(connection, KEY_FINGERPRINT_KEY)
    except sqlite3.DatabaseError:
        return None


def install_master_key_if_current(key: bytes, base: Path | None = None) -> bool:
    """Install ``key`` as ``master.key`` ONLY if the store is still sealed under it.

    A compare-and-swap, and the reason it exists is that installing
    unconditionally destroys the store. ``rotate`` is stage → COMMIT → install,
    and the epoch guard inside the re-seal transaction serialises the COMMITs —
    but it said nothing about the installs that follow them, which ran outside
    any transaction. So a rotator that committed EARLIER could install its now
    superseded key LATER::

        A: COMMIT  -> db sealed under Ka
        B: COMMIT  -> db sealed under Kb   (legitimate; B adopted Ka first)
        B: install Kb, discard staged Kb   (correct)
        A: install Ka, discard staged Ka   (last writer wins -- Ka is STALE)

    Final state: the database needs Kb, ``master.key`` holds Ka, and both staged
    copies are gone, so Kb exists NOWHERE. Every secret is unrecoverable and
    both processes exit 0 printing ``rotated N secret(s)``. Reproduced
    deterministically 3/3 with two real ``lop secret rotate`` processes and no
    crash anywhere, and 7/8 from ordinary contention between four of them.
    :func:`local_operator.secrets.access.resolve_master_key` cannot repair it:
    it adopts a STAGED key matching the database, and the matching one was
    legitimately discarded by the rotation that installed it.

    **The fix is to make commit-and-install atomic with respect to other
    rotators by putting the install under the database's OWN write lock.**
    ``BEGIN IMMEDIATE`` here takes the same lock the re-seal transaction takes,
    so an install and a competing COMMIT cannot interleave: either this call
    finishes first and the competitor's later COMMIT+install lands on top of it,
    or the competitor commits first and the fingerprint read below no longer
    matches, and this install is REFUSED instead of clobbering.

    Chosen over an exclusive rotation lock held across commit and install, for
    three reasons. It adds no new lock primitive, so there is no second lock to
    order against SQLite's and therefore no deadlock to reason about. It is not
    an ``flock``, so it cannot reproduce #401 — a thread parked in ``flock()``
    blocks a sibling's ``close()`` of that descriptor on macOS, whereas SQLite
    contention is bounded by ``busy_timeout`` and raises rather than hangs. And
    it fails CLOSED by construction: a lock that cannot be taken tempts a
    degrade-to-unlocked path, which here would be exactly the unguarded install
    that loses the store.

    Ordinary writes are NOT serialised behind rotation by this. They already
    contend for the same write lock for their own transactions; this adds one
    short read-only transaction per rotation, not a lock that spans one.

    **The install stays AFTER the COMMIT and is deliberately not folded into the
    re-seal transaction.** Writing ``master.key`` before that COMMIT would leave
    a crash window where the database is still under the OLD key while the new
    one is installed and the old one is gone — the R2 invariant (at every
    instant, a key that opens the store exists on disk) inverted. With the
    ordering kept, R2 holds under concurrency too: before this call the staged
    key is the copy that opens the committed database; after ``os.replace``
    ``master.key`` is; a crash between them leaves the staged file for
    ``resolve_master_key`` to adopt; and a refusal leaves the winner's install
    untouched, so the key on disk is the one the database is sealed under at
    every instant. Re-proven with four SIGKILLs landing inside this function
    while three rotators contended: 0 stores left unopenable.

    Returns ``True`` when the key was installed, ``False`` when another rotation
    superseded it. Nothing is unlinked on either path: this function is reached
    both by a rotator installing its OWN staged key and by
    :func:`local_operator.secrets.access.resolve_master_key` completing somebody
    ELSE's interrupted rotation, and only the owner is entitled to discard a
    staged file (see
    :func:`local_operator.secrets.keys.discard_staged_master_key` — removing
    another rotation's staged key is the loss that check exists to prevent).
    The rotator therefore does its own discard on a refusal.

    A store with no database or no fingerprint row cannot be superseded by
    anything, so the install proceeds unconditionally; that is the same
    "cannot tell, do not guess" fallback :func:`recorded_key_fingerprint` uses.
    """
    path = store_path(base)
    if not path.exists():
        replace_master_key(base, key)
        return True

    with closing(_connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            recorded = _read_meta_blob(connection, KEY_FINGERPRINT_KEY)
            if recorded is not None and recorded != key_fingerprint(key):
                # Superseded. Refusing is what keeps the winner's key installed.
                return False
            replace_master_key(base, key)
            return True
        finally:
            # Read-only throughout: the transaction exists solely to hold the
            # write lock across the fingerprint check and the file install.
            connection.execute("ROLLBACK")


class SecretStore:
    """Read/write access to the encrypted store, given a master key.

    Not a long-lived object: each public method opens a connection, does its
    work in one transaction and closes it. Ten sessions each holding an open
    handle for their whole lifetime would keep WAL readers pinned and stop the
    checkpointer from ever truncating the log, for no benefit — these
    operations are milliseconds and rare.
    """

    def __init__(self, master_key: bytes, base: Path | None = None) -> None:
        self._master_key = master_key
        self._path = store_path(base)
        self._base = base

    @property
    def path(self) -> Path:
        """The database file this instance talks to."""
        return self._path

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Create the database and its tables, 0600 inside a 0700 directory.

        The explicit ``chmod`` after creation is load-bearing: SQLite creates
        its file honouring the process umask, so a session running under
        ``umask 000`` would otherwise leave a world-readable database. The
        ``-wal`` and ``-shm`` sidecars get the same treatment — they carry the
        same ciphertext, and a private database with a world-readable WAL is
        not private.
        """
        ensure_secrets_dir(self._base)
        with closing(_connect(self._path)) as connection:
            connection.executescript(_SCHEMA)
            if _read_meta_int(connection, "schema_version", 0) == 0:
                _write_meta_int(connection, "schema_version", SCHEMA_VERSION)
            if _read_meta_int(connection, "key_generation", 0) == 0:
                _write_meta_int(connection, "key_generation", 1)
            if _read_meta_blob(connection, KEY_FINGERPRINT_KEY) is None:
                _write_meta_blob(connection, KEY_FINGERPRINT_KEY, key_fingerprint(self._master_key))
        self._restrict_modes()

    def exists(self) -> bool:
        """Whether a store has been created yet."""
        return self._path.exists()

    def _restrict_modes(self) -> None:
        for candidate in (
            self._path,
            self._path.with_name(self._path.name + "-wal"),
            self._path.with_name(self._path.name + "-shm"),
        ):
            # The exists()/chmod pair is a TOCTOU on the sidecars, not on the
            # database: SQLite unlinks `-wal` and `-shm` when the last
            # connection closes, so a concurrent session can remove one inside
            # this gap and fail an operation that was otherwise fine (observed
            # once naturally in ~200 rotation-heavy concurrent runs). A file
            # that no longer exists carries no ciphertext and needs no mode, so
            # the disappearance is the desired end state rather than an error.
            # Only FileNotFoundError is suppressed; a permission failure on a
            # file that IS there still raises, because that one leaves real
            # ciphertext at a mode this function promised to remove.
            with contextlib.suppress(FileNotFoundError):
                if candidate.exists():
                    os.chmod(candidate, FILE_MODE)

    def _open(self, *, for_write: bool) -> sqlite3.Connection:
        """Open an existing store, refusing an incompatible or exposed one."""
        if not self._path.exists():
            raise SecretStoreError(
                f"No secret store found at {self._path}. "
                "Create one by storing a secret: lop secret set NAME"
            )
        check_mode(self._path)
        connection = _connect(self._path)
        found = _read_meta_int(connection, "schema_version", SCHEMA_VERSION)
        if found != SCHEMA_VERSION:
            connection.close()
            # Both directions are refused, but only the newer one is expected:
            # the operator updates the runtime under live sessions, so an older
            # runtime meeting a newer store is routine (§13) and must never
            # migrate it downward.
            direction = "newer" if found > SCHEMA_VERSION else "older"
            action = (
                "upgrade this runtime (lop update)"
                if found > SCHEMA_VERSION
                else "the store predates this runtime's schema and has no migration"
            )
            raise IncompatibleStore(
                f"This secret store uses schema version {found}; this runtime speaks "
                f"{SCHEMA_VERSION}. It was written by a {direction} local-operator — {action}."
            )
        return connection

    def key_generation(self) -> int:
        """The generation new records are sealed under."""
        with closing(self._open(for_write=False)) as connection:
            return _read_meta_int(connection, "key_generation", 1)

    def _guard_key_epoch(self, connection: sqlite3.Connection) -> None:
        """Refuse a write whose key is no longer the store's key.

        MUST be called inside the caller's ``BEGIN IMMEDIATE`` transaction, not
        before it. That is the entire protection: ``BEGIN IMMEDIATE`` takes the
        write lock, so a ``rotate`` cannot land between this check and the
        INSERT that follows it. Checking outside the transaction would restore
        the race it exists to close.

        The failure this prevents is not hypothetical. A session that loaded
        the master key before another session's ``rotate`` would seal its
        record under the superseded key and index it under the superseded index
        key, then COMMIT and report "stored". The row is then unreachable by
        name (the index does not match) and undecryptable (the key is gone) —
        and because it is undecryptable, it also poisons every enumeration that
        touches it. Refusing the write costs one retry with a fresh key.

        A store with no fingerprint row predates this guard; it is backfilled
        rather than refused, because breaking every existing store would be a
        worse outcome than the race.
        """
        recorded = _read_meta_blob(connection, KEY_FINGERPRINT_KEY)
        mine = key_fingerprint(self._master_key)
        if recorded is None:
            _write_meta_blob(connection, KEY_FINGERPRINT_KEY, mine)
            return
        if recorded != mine:
            raise StaleKeyEpoch(
                "This store's master key was rotated by another session after this one "
                "loaded it, so this write was refused rather than being sealed under a "
                "key that no longer exists. Re-run the command; it will pick up the "
                "current key."
            )

    # -- record helpers ----------------------------------------------------

    def _decode(
        self, row: Sequence[Any], master_key: bytes | None = None
    ) -> tuple[SecretRecord, bytes]:
        """Authenticate and decode one row into ``(record, value)``.

        The AAD is rebuilt from the row's own COLUMNS — never from anything
        inside the ciphertext, which would be circular (see
        :func:`crypto.build_aad`). So every column an attacker with database
        write access can edit is an AEAD input, and editing one makes the tag
        check fail rather than relabelling a record that still decrypts.

        The re-derived blind index check at the end closes the remaining gap:
        the name inside the ciphertext and the ``name_index`` column are two
        copies of the same fact, and this is where they are made to agree.
        """
        key = self._master_key if master_key is None else master_key
        record_id = str(row[0])
        stored_index = bytes(row[1])
        key_generation = int(row[2])
        format_version = int(row[3])
        if format_version > RECORD_FORMAT_VERSION:
            raise IncompatibleStore(
                f"Secret record {record_id} uses record format {format_version}; this runtime "
                f"understands {RECORD_FORMAT_VERSION}. It was written by a newer "
                "local-operator — upgrade this runtime (lop update) to read it."
            )
        kind = str(row[6])
        metadata = RecordMetadata(
            record_id=record_id,
            name_index=stored_index,
            kind=kind,
            key_generation=key_generation,
            format_version=format_version,
        )
        plaintext = open_record(key, metadata, bytes(row[4]), bytes(row[5]))
        payload = json.loads(plaintext.decode("utf-8"))

        # The sealed name must still hash to the column the row was found by.
        # Belt and braces over the AAD: it means an attacker cannot construct
        # any row whose two copies of the name disagree, even with the key.
        if name_index(key, payload["name"]) != stored_index:
            raise SecretCorrupt(
                f"Secret record {record_id} is inconsistent: its sealed name does not match "
                "its lookup index. Refusing to return a value."
            )
        record = SecretRecord(
            record_id=record_id,
            name=payload["name"],
            description=payload["description"],
            kind=kind,
            key_generation=key_generation,
            created_at=float(row[7]),
            updated_at=float(row[8]),
            last_used_at=None if row[9] is None else float(row[9]),
        )
        return record, bytes.fromhex(payload["value"])

    def _row_for(self, connection: sqlite3.Connection, name: str) -> Sequence[Any]:
        index = name_index(self._master_key, validate_name(name))
        row = connection.execute(
            f"SELECT {_RECORD_COLUMNS} FROM secrets WHERE name_index = ?", (index,)
        ).fetchone()
        if row is None:
            raise SecretNotFound(f"No secret named {name!r} in this store.")
        return row

    # -- verbs -------------------------------------------------------------

    def set(
        self,
        name: str,
        value: bytes,
        *,
        description: str = "",
        kind: str = "string",
        session_id: str | None = None,
    ) -> SecretRecord:
        """Store a NEW secret. Refuses to overwrite an existing name.

        Overwriting is ``update``'s job and is spelled differently on purpose:
        the previous value is retained nowhere, so a ``set`` that silently
        replaced a live credential because of a mistyped name would be
        unrecoverable.
        """
        canonical = validate_name(name)
        description = _validate_description(description)
        if kind not in KINDS:
            raise InvalidSecretName(f"kind must be one of {', '.join(KINDS)}; got {kind!r}.")
        self.initialize()
        now = time.time()
        record_id = str(uuid.uuid4())
        index = name_index(self._master_key, canonical)
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._guard_key_epoch(connection)
                generation = _read_meta_int(connection, "key_generation", 1)
                if connection.execute(
                    "SELECT 1 FROM secrets WHERE name_index = ?", (index,)
                ).fetchone():
                    raise SecretExists(
                        f"A secret named {canonical!r} already exists. "
                        "Use `lop secret update` to change its value."
                    )
                metadata = RecordMetadata(
                    record_id=record_id,
                    name_index=index,
                    kind=kind,
                    key_generation=generation,
                )
                nonce, ciphertext = seal(
                    self._master_key, metadata, _payload(canonical, description, value)
                )
                connection.execute(
                    f"INSERT INTO secrets({_RECORD_COLUMNS})"
                    " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        record_id,
                        index,
                        generation,
                        RECORD_FORMAT_VERSION,
                        nonce,
                        ciphertext,
                        kind,
                        now,
                        now,
                    ),
                )
                audit.append(
                    connection,
                    event="set",
                    ts=now,
                    outcome="ok",
                    secret_id=record_id,
                    session_id=session_id,
                    pid=os.getpid(),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            generation_written = generation
        self._restrict_modes()
        return SecretRecord(
            record_id=record_id,
            name=canonical,
            description=description,
            kind=kind,
            key_generation=generation_written,
            created_at=now,
            updated_at=now,
            last_used_at=None,
        )

    def update(
        self,
        name: str,
        value: bytes,
        *,
        description: str | None = None,
        session_id: str | None = None,
    ) -> SecretRecord:
        """Re-seal an existing secret with a new value under a FRESH nonce.

        The nonce comes from :func:`crypto.seal`, which generates one per call.
        An update that reused the record's stored nonce under the same key
        would leak the XOR of the old and new values and the authentication
        key — the classic GCM misuse, and the reason no call site here is given
        the opportunity to supply a nonce.
        """
        canonical = validate_name(name)
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._guard_key_epoch(connection)
                existing, _ = self._decode(self._row_for(connection, canonical))
                new_description = (
                    existing.description
                    if description is None
                    else _validate_description(description)
                )
                metadata = RecordMetadata(
                    record_id=existing.record_id,
                    name_index=name_index(self._master_key, canonical),
                    kind=existing.kind,
                    key_generation=existing.key_generation,
                )
                nonce, ciphertext = seal(
                    self._master_key, metadata, _payload(canonical, new_description, value)
                )
                connection.execute(
                    "UPDATE secrets SET nonce = ?, ciphertext = ?, updated_at = ? WHERE id = ?",
                    (nonce, ciphertext, now, existing.record_id),
                )
                audit.append(
                    connection,
                    event="update",
                    ts=now,
                    outcome="ok",
                    secret_id=existing.record_id,
                    session_id=session_id,
                    pid=os.getpid(),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        self._restrict_modes()
        return SecretRecord(
            record_id=existing.record_id,
            name=canonical,
            description=new_description,
            kind=existing.kind,
            key_generation=existing.key_generation,
            created_at=existing.created_at,
            updated_at=now,
            last_used_at=existing.last_used_at,
        )

    def get(self, name: str, *, session_id: str | None = None) -> bytes:
        """Return the exact stored bytes, or raise.

        ``last_used_at`` and the audit row are written on the way out, which is
        why a read verb takes a write connection: a retrieval that leaves no
        trace is the one an attacker most wants.
        """
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                record, value = self._decode(self._row_for(connection, name))
                connection.execute(
                    "UPDATE secrets SET last_used_at = ? WHERE id = ?", (now, record.record_id)
                )
                audit.append(
                    connection,
                    event="get",
                    ts=now,
                    outcome="ok",
                    secret_id=record.record_id,
                    session_id=session_id,
                    pid=os.getpid(),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return value

    def describe(self, name: str) -> SecretRecord:
        """Metadata for one secret. Never returns the value."""
        with closing(self._open(for_write=False)) as connection:
            record, _ = self._decode(self._row_for(connection, name))
        return record

    def list(self) -> list[SecretRecord]:
        """Every readable record's metadata, name-sorted. Never returns values.

        Damaged records are REPORTED, not propagated. This used to raise on the
        first record that failed to authenticate, on the reasoning that the
        operator must learn about corruption the first time they look. The
        reasoning was right and the mechanism was wrong: raising means one bad
        row takes down ``list``, ``status`` and every subsequent ``rotate``,
        so the operator learns that something is broken and simultaneously
        loses every tool for finding out what, including access to the
        untouched secrets sitting beside it. Enumeration is the surface an
        operator reaches for *when* the store is damaged; it is the one path
        that must keep working.

        So a record that will not open is surfaced through
        :meth:`damaged_records` and through ``list``'s and ``status``'s output
        rather than by making the store unusable, and :meth:`delete_record_id`
        gives the operator a way to remove it.
        """
        return self._enumerate()[0]

    def damaged_records(self) -> list[str]:
        """Record ids present in the store that cannot be authenticated.

        Empty in every healthy store. A non-empty result means tampering, disk
        damage, or a record sealed under a key that no longer exists — the last
        of which is what a pre-guard concurrent rotation used to produce.
        """
        return self._enumerate()[1]

    def _enumerate(self) -> tuple[list[SecretRecord], list[str]]:
        """Split every row into the records that open and the ids that do not.

        One pass, so ``list`` and ``status`` cannot disagree about which rows
        are damaged. Only :class:`SecretCorrupt` and
        :class:`IncompatibleStore` are caught per row: those mean "this record
        is unreadable", which is precisely the condition to report and step
        over. A failure to read the store at all still propagates.
        """
        with closing(self._open(for_write=False)) as connection:
            rows = connection.execute(f"SELECT {_RECORD_COLUMNS} FROM secrets").fetchall()
        records: list[SecretRecord] = []
        damaged: list[str] = []
        for row in rows:
            try:
                records.append(self._decode(row)[0])
            except (SecretCorrupt, IncompatibleStore):
                damaged.append(str(row[0]))
        return sorted(records, key=lambda record: record.name), sorted(damaged)

    def delete_record_id(self, record_id: str, *, session_id: str | None = None) -> bool:
        """Remove a row by its id, without needing to decrypt it.

        The repair path for a damaged record. :meth:`delete` looks a row up by
        blind index and decodes it, so it cannot touch a record whose name is
        unreadable — the operator was left with a row that broke enumeration
        and that no command could remove. This deletes by primary key, which
        needs no key material for the row itself.

        Returns whether a row was removed. Still audited, and still guarded by
        the key epoch: removing a record is a write.
        """
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._guard_key_epoch(connection)
                cursor = connection.execute("DELETE FROM secrets WHERE id = ?", (record_id,))
                removed = cursor.rowcount > 0
                if removed:
                    audit.append(
                        connection,
                        event="delete",
                        ts=now,
                        outcome="ok",
                        secret_id=record_id,
                        session_id=session_id,
                        pid=os.getpid(),
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return removed

    def delete(self, name: str, *, session_id: str | None = None) -> SecretRecord:
        """Remove a secret, returning what was removed."""
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._guard_key_epoch(connection)
                record, _ = self._decode(self._row_for(connection, name))
                connection.execute("DELETE FROM secrets WHERE id = ?", (record.record_id,))
                audit.append(
                    connection,
                    event="delete",
                    ts=now,
                    outcome="ok",
                    secret_id=record.record_id,
                    session_id=session_id,
                    pid=os.getpid(),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return record

    def rotate(self, new_master_key: bytes, *, session_id: str | None = None) -> int:
        """Re-seal every record under ``new_master_key``; returns the count.

        One transaction, not record by record. The blind index is keyed from
        the master key, so a half-rotated store would carry some indexes under
        the old key and some under the new, and no single key could look every
        record up — a different failure from the per-record ``key_generation``
        the design describes, which keeps old ciphertexts readable and does
        work here: each record is opened under its OWN generation before being
        re-sealed under the next one.

        **The caller must have STAGED the new key on disk before calling this**
        (:func:`local_operator.secrets.keys.stage_master_key`), and installs it
        as ``master.key`` after this returns. That ordering is load-bearing and
        the reverse of what an earlier version of this docstring claimed: by
        the time this method commits, the database is re-sealed under
        ``new_master_key``, and the old key opens NOTHING. A crash in the
        window between this COMMIT and the install is survivable only because
        the staged file already holds the new key —
        :func:`local_operator.secrets.access.resolve_master_key` matches it
        against the fingerprint written below and finishes the install. Commit
        first with the key only in memory and an ordinary power cut during a
        rotate loses every secret in the store, unrecoverably.
        """
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # The rotator itself must still be holding the store's current
                # key: two concurrent rotations would otherwise each re-seal
                # from a different starting point and the loser's key would be
                # the one on disk.
                self._guard_key_epoch(connection)
                generation = _read_meta_int(connection, "key_generation", 1) + 1
                rows = connection.execute(f"SELECT {_RECORD_COLUMNS} FROM secrets").fetchall()
                moved = 0
                for row in rows:
                    try:
                        record, value = self._decode(row)
                    except (SecretCorrupt, IncompatibleStore):
                        # A record this key cannot open cannot be re-sealed
                        # under the next one either, and refusing to rotate
                        # until it is gone would mean a single damaged row
                        # blocks the operator's response to a suspected key
                        # compromise — the moment rotation matters most. It is
                        # left exactly as it is, still visible through
                        # `damaged_records()` and still removable by id, and
                        # the count returned reflects what actually moved.
                        continue
                    index = name_index(new_master_key, record.name)
                    metadata = RecordMetadata(
                        record_id=record.record_id,
                        name_index=index,
                        kind=record.kind,
                        key_generation=generation,
                    )
                    nonce, ciphertext = seal(
                        new_master_key,
                        metadata,
                        _payload(record.name, record.description, value),
                    )
                    connection.execute(
                        "UPDATE secrets SET name_index = ?, key_generation = ?, nonce = ?,"
                        " ciphertext = ?, format_version = ? WHERE id = ?",
                        (
                            index,
                            generation,
                            nonce,
                            ciphertext,
                            RECORD_FORMAT_VERSION,
                            record.record_id,
                        ),
                    )
                    moved += 1
                _write_meta_int(connection, "key_generation", generation)
                # The epoch moves in the SAME transaction as the re-seal, so
                # the fingerprint on disk always describes the key the
                # ciphertexts are actually under — including for a process that
                # crashes immediately after this COMMIT, which is how
                # `resolve_master_key` recognises the staged key as the right
                # one.
                _write_meta_blob(connection, KEY_FINGERPRINT_KEY, key_fingerprint(new_master_key))
                audit.append(
                    connection,
                    event="rotate",
                    ts=now,
                    outcome="ok",
                    session_id=session_id,
                    pid=os.getpid(),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        self._master_key = new_master_key
        return moved

    def audit_entries(self, limit: int | None = None) -> list[audit.AuditEntry]:
        """Recent audit entries, oldest first."""
        with closing(self._open(for_write=False)) as connection:
            return audit.entries(connection, limit=limit)

    def verify_audit(self) -> tuple[bool, int | None, str]:
        """Walk the audit hash chain; see :func:`audit.verify`."""
        with closing(self._open(for_write=False)) as connection:
            return audit.verify(connection)


def _validate_description(description: str) -> str:
    """Reject a description that would break the record or the listing.

    Newlines are refused because ``list`` prints one record per line and a
    description carrying one would forge a second row; the length ceiling keeps
    a pathological paste out of every future re-seal of that record.
    """
    if "\n" in description or "\r" in description:
        raise InvalidSecretName("Secret description cannot contain newlines.")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise InvalidSecretName(
            f"Secret description is {len(description)} characters; "
            f"the maximum is {MAX_DESCRIPTION_LENGTH}."
        )
    return description


__all__ = [
    "KINDS",
    "MAX_DESCRIPTION_LENGTH",
    "SCHEMA_VERSION",
    "SecretRecord",
    "SecretStore",
]
