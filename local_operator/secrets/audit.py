"""The tamper-evident audit chain (design §12).

Every store, retrieval, update, delete and rename appends a row whose hash
covers the previous row's hash, so an attacker who edits or removes an entry in
the middle breaks the chain at that point and ``lop secret audit --verify``
reports where.

The claim is **tamper-evident, not tamper-proof**, and the design is explicit
that it must not be described as an immutable log: whoever owns the file can
still delete the whole thing, and the ``chflags uappnd`` half of §12 (which
blocks truncation and unlink but which the owner can clear) belongs to the
broker PR along with the mirrored ``audit.log`` file — the broker is the
process that has the peer pid and executable path worth recording. What lands
here is the chain itself and its verifier, over the ``audit`` table.

Never record a value. The columns are deliberately limited to the record's id,
the event, and the peer identity, so a leaked audit log tells an attacker what
happened, not what the secrets are.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

#: The chain's anchor. A fixed non-empty seed rather than ``b""`` so that a row
#: whose ``prev_hash`` was blanked out by an attacker is distinguishable from a
#: genuine first row.
GENESIS = hashlib.sha256(b"local-operator/secret-store/audit/v1").digest()


@dataclass(frozen=True)
class AuditEntry:
    """One row of the audit table, as the verifier sees it."""

    ts: float
    event: str
    secret_id: str | None
    session_id: str | None
    pid: int | None
    exe: str | None
    outcome: str
    prev_hash: bytes
    hash: bytes


def canonical_row(
    ts: float,
    event: str,
    secret_id: str | None,
    session_id: str | None,
    pid: int | None,
    exe: str | None,
    outcome: str,
) -> bytes:
    """Serialise a row for hashing, unambiguously.

    Length-prefixed fields rather than a separator: a session id containing the
    separator byte would otherwise let two different rows serialise identically,
    which is the standard way a naive chain is forged. ``ts`` is formatted with
    a fixed repr so the same row always hashes the same on re-read from SQLite.
    """
    parts: list[bytes] = [
        f"{ts:.6f}".encode("ascii"),
        event.encode("utf-8"),
        (secret_id or "").encode("utf-8"),
        (session_id or "").encode("utf-8"),
        str(pid if pid is not None else "").encode("ascii"),
        (exe or "").encode("utf-8"),
        outcome.encode("utf-8"),
    ]
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def chain_hash(prev_hash: bytes, row: bytes) -> bytes:
    """``SHA256(prev_hash || canonical(row))`` — the link itself."""
    return hashlib.sha256(prev_hash + row).digest()


def last_hash(connection: sqlite3.Connection) -> bytes:
    """The hash the next appended row must chain from.

    Ordered by ``rowid``, not by ``ts``: two appends inside the same
    millisecond, or a clock that steps backwards, must not reorder the chain.
    """
    cursor = connection.execute("SELECT hash FROM audit ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    return GENESIS if row is None else bytes(row[0])


def append(
    connection: sqlite3.Connection,
    *,
    event: str,
    ts: float,
    outcome: str,
    secret_id: str | None = None,
    session_id: str | None = None,
    pid: int | None = None,
    exe: str | None = None,
) -> bytes:
    """Append one entry and return its hash.

    Takes the caller's open connection and does not commit: the audit row and
    the change it describes belong in the same transaction, or a crash between
    them produces a store whose history disagrees with its contents.
    """
    previous = last_hash(connection)
    digest = chain_hash(
        previous, canonical_row(ts, event, secret_id, session_id, pid, exe, outcome)
    )
    connection.execute(
        "INSERT INTO audit(ts, event, secret_id, session_id, pid, exe, outcome, prev_hash, hash)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, event, secret_id, session_id, pid, exe, outcome, previous, digest),
    )
    return digest


def entries(connection: sqlite3.Connection, limit: int | None = None) -> list[AuditEntry]:
    """Read the chain in append order, newest last."""
    sql = (
        "SELECT ts, event, secret_id, session_id, pid, exe, outcome, prev_hash, hash"
        " FROM audit ORDER BY rowid"
    )
    rows = connection.execute(sql).fetchall()
    if limit is not None:
        rows = rows[-limit:]
    return [
        AuditEntry(
            ts=row[0],
            event=row[1],
            secret_id=row[2],
            session_id=row[3],
            pid=row[4],
            exe=row[5],
            outcome=row[6],
            prev_hash=bytes(row[7]),
            hash=bytes(row[8]),
        )
        for row in rows
    ]


def verify(connection: sqlite3.Connection) -> tuple[bool, int | None, str]:
    """Walk the chain; return ``(ok, broken_position, message)``.

    ``broken_position`` is the 1-based index of the first entry that does not
    verify, which is the number the operator needs — it says *where* history
    stopped being trustworthy, and everything before it is still evidence.

    Two distinct breaks are reported separately because they mean different
    things: a wrong ``prev_hash`` means a row was removed or reordered ahead of
    this one, while a wrong ``hash`` over a correct ``prev_hash`` means this
    row's own contents were edited in place.
    """
    expected = GENESIS
    for position, entry in enumerate(entries(connection), start=1):
        if entry.prev_hash != expected:
            return (
                False,
                position,
                f"audit chain broken at entry {position}: an earlier entry was removed "
                "or reordered",
            )
        recomputed = chain_hash(
            expected,
            canonical_row(
                entry.ts,
                entry.event,
                entry.secret_id,
                entry.session_id,
                entry.pid,
                entry.exe,
                entry.outcome,
            ),
        )
        if recomputed != entry.hash:
            return (
                False,
                position,
                f"audit chain broken at entry {position}: this entry was edited after "
                "it was written",
            )
        expected = entry.hash
    return True, None, "audit chain intact"
