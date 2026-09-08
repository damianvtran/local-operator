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
)
from local_operator.secrets.keys import (
    FILE_MODE,
    check_mode,
    ensure_secrets_dir,
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
    row = connection.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    if row is None:
        return default
    return int(bytes(row[0]).decode("ascii"))


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
        """Every record's metadata, name-sorted. Never returns values.

        A record that fails to authenticate propagates rather than being
        skipped: a store with a corrupt record is something the operator must
        learn the first time they look at it, not on the day they need that
        one secret.
        """
        with closing(self._open(for_write=False)) as connection:
            rows = connection.execute(f"SELECT {_RECORD_COLUMNS} FROM secrets").fetchall()
        return sorted((self._decode(row)[0] for row in rows), key=lambda record: record.name)

    def delete(self, name: str, *, session_id: str | None = None) -> SecretRecord:
        """Remove a secret, returning what was removed."""
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
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

        The caller installs the new key file only after this returns, so a
        crash mid-rotation leaves the old key still matching the untouched
        database.
        """
        now = time.time()
        with closing(self._open(for_write=True)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                generation = _read_meta_int(connection, "key_generation", 1) + 1
                rows = connection.execute(f"SELECT {_RECORD_COLUMNS} FROM secrets").fetchall()
                for row in rows:
                    record, value = self._decode(row)
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
                _write_meta_int(connection, "key_generation", generation)
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
        return len(rows)

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
