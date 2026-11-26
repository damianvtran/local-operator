"""Display-only retention of ADMITTED Team Bridge message bodies.

WHY THIS EXISTS. An inbound bridge card in the journal is metadata only (see
``connector.conversation.build_conversation_card``), and the screened body
reaches the agent exactly once, in a hash-bound ephemeral frame
(``pending_turn``/``request_body``). That kept the model boundary airtight but
left the operator looking at "Message body unavailable" for every message the
agent had in fact read, with no way to review it later. This store keeps the
body the agent was actually given so a surface can render it inline.

WHY A SEPARATE DATABASE. ``state.sqlite3`` carries a contentless invariant that
reviews and docstrings lean on (reservations, holds and receipts hold ids,
digests and closed codes only). Putting third-party text in the same file would
quietly end that guarantee for every table beside it. A separate
``display_bodies.sqlite3`` (0600) keeps the state store body-free, and keeps
this content out of transcripts, forks, exports, transfers, search and
compaction — none of which read this file.

WHO MAY WRITE (closed set, both hash-verified):

* ``source='turn'`` — :meth:`PendingTurn.settle` on a ``completed`` outcome
  only, from the ``FetchedBody`` whose sha256 matched the screened reservation.
  A turn that drifted, aborted or failed writes nothing.
* ``source='hub'`` — the card-body resolver's backfill, only when the hub's
  admitted-only history read returned a body whose digest equals the row's
  ``body_hash``.

WHAT IT IS NEVER. A model input on its own: the one path from here to a model
is the bounded excerpt gate in ``card_projection`` (``source='turn'`` rows, live
owner + consent). Never logged. Never an admission proof: replaying a row from
here does not count as the agent receiving it.

BOUNDS (stated, not tuned): a body over the frame's ``MAX_BODY_BYTES`` (16 KiB,
the most a turn could ever have been given) is refused; rows older than 90 days
are dropped; total body bytes over 64 MiB evict oldest-first. The bounds are
storage hygiene for a cache that the hub can refill, not correctness limits.

REVOCATION. Every row is keyed by the owner namespace (``store._owner_key``),
and a read answers only for the live owner. Purges are EAGER, at the event that
ends the standing (review R3 / QA Q1), not only when a card is next viewed:

* Radient sign-out — ``AuthStore.org_hub_signed_out``, the hook every logout
  host and the desktop's account removal share;
* bridge opt-out — the one-click Turn off (``client._apply_local_opt_out``) and
  every explicit ``team_bridge.enabled`` write, which all reach
  ``connector.install.sync_connector_service`` (:func:`purge_if_bridge_off`);
* consent withdrawn or the bridge off as observed by the excerpt gate
  (``Session._bridge_history_transform``) at the next model request;
* another namespace's enrollment commit (account switch, ``keep_owner_key``).

The resolver still purges when it observes the same states (defence in depth).
The hub backfill can restore what the viewer is still authorised to see.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DisplayBodyStore",
    "MAX_TOTAL_BYTES",
    "TTL_SECONDS",
    "display_db_path",
]

#: Rows older than this are dropped on the next write (90 days).
TTL_SECONDS = 90 * 24 * 60 * 60
#: Total stored body bytes before oldest-first eviction (64 MiB).
MAX_TOTAL_BYTES = 64 * 1024 * 1024
_SOURCES = frozenset({"turn", "hub"})
_SENDER_KINDS = frozenset({"human", "agent", "unknown"})
_FILE_MODE = 0o600
_BUSY_TIMEOUT_S = 5.0

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS display_bodies ("
    "owner_key TEXT NOT NULL, message_id TEXT NOT NULL, "
    "conversation_id TEXT NOT NULL, body TEXT NOT NULL, body_sha256 TEXT NOT NULL, "
    "sender_kind TEXT NOT NULL, admitted_at REAL NOT NULL, source TEXT NOT NULL, "
    "PRIMARY KEY (owner_key, message_id))"
)
#: The per-owner key ``card_projection.ExcerptMemo`` HMACs its body digests
#: with. It lives HERE, beside the bodies, so whatever ends a body's retention
#: ends the key too: ``purge`` deletes it with the rows, and it expires at the
#: same TTL as a row. A memo file left in a session directory then holds
#: digests no one can test a guessed body against (review R1-1 on PR #4629).
_MEMO_KEY_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS excerpt_memo_keys ("
    "owner_key TEXT PRIMARY KEY, key BLOB NOT NULL, created_at REAL NOT NULL)"
)
_MEMO_KEY_BYTES = 32


def display_db_path(config_dir: Path) -> Path:
    """``<config_dir>/team_bridge/display_bodies.sqlite3`` — beside, never inside, state."""

    return Path(config_dir) / "team_bridge" / "display_bodies.sqlite3"


def purge_if_bridge_off(config_dir: Path) -> int:
    """Purge every row when ``team_bridge.enabled`` reads a DEFINITE off. Never raises.

    Only a readable config that says off counts: a torn or missing read is not
    an opt-out, and dropping the cache on one would make a transient read error
    cost the operator every retained body (the hub can refill, but only for
    messages it still serves).
    """

    try:
        from local_operator.config import read_config_values

        from .settings import TeamBridgeSettings

        values = read_config_values(Path(config_dir))
        if values is None or TeamBridgeSettings.from_values(values).enabled:
            return 0
        return DisplayBodyStore(Path(config_dir)).purge()
    except Exception:  # noqa: BLE001 — every caller already did its real work
        logger.debug("bridge display: opt-out purge skipped", exc_info=False)
        return 0


#: Serialises reads of the process-shared ``AuthStore`` connection from this
#: module. The TUI resolves every card as it mounts, each on a worker thread,
#: and concurrent reads through the one shared connection interleaved: measured
#: in the round-2 rig as a random card answering "signed out" (a row read back
#: as ``None`` / a ``TypeError`` from a half-read row) while the operator was
#: signed in.
_AUTH_READ_LOCK = threading.Lock()


def radient_sign_in_state(config_dir: Path, credential_id: Any) -> bool | None:
    """Is the owner's own Radient OAuth row on this machine? ``None`` = unreadable.

    Local read only: no network and no token refresh. ``False`` is a definite
    answer (no row, a foreign or disabled row); ``None`` means the store could
    not be read, which a caller must not report as "signed out".
    """

    try:
        wanted = int(credential_id)
    except (TypeError, ValueError):
        return False
    try:
        from local_operator.providers.auth_store import shared_auth_store

        root = Path(config_dir)
        with _AUTH_READ_LOCK:
            row = shared_auth_store(db_path=root / "auth.db", config_dir=root).get_credential(
                wanted
            )
    except Exception:  # noqa: BLE001 — unreadable is not "signed out"
        return None
    return (
        row is not None
        and row.provider == "radient"
        and row.credential_type == "oauth"
        and not getattr(row, "disabled_cause", None)
    )


def radient_signed_in(config_dir: Path, credential_id: Any) -> bool:
    """The excerpt gate's sign-in check (QA Q1); unreadable fails closed.

    ``authority_fence`` proves the enrollment, consent and designation, but
    those rows outlive a ``lop logout radient`` — so without this a signed-out
    machine kept feeding excerpts to the model. ``False`` only withholds
    excerpts (the ids-only marker).
    """

    return radient_sign_in_state(config_dir, credential_id) is True


class DisplayBodyStore:
    """The one reader/writer of ``display_bodies.sqlite3``."""

    def __init__(self, config_dir: Path) -> None:
        self._path = display_db_path(config_dir)

    @contextmanager
    def _connect(self, *, create: bool) -> Iterator[sqlite3.Connection | None]:
        if not create and not self._path.exists():
            # A read must not create the file: an install that never admitted
            # a body has no display store, and that stays true.
            yield None
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 BEFORE sqlite opens it, so there is no window in
        # which another local account could read a freshly created file.
        if create and not self._path.exists():
            fd = os.open(self._path, os.O_CREAT | os.O_WRONLY, _FILE_MODE)
            os.close(fd)
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_S)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(_SCHEMA)
            conn.execute(_MEMO_KEY_SCHEMA)
            with conn:
                yield conn
        finally:
            conn.close()
            self._restrict_sidecars()

    def _restrict_sidecars(self) -> None:
        # WAL/journal sidecars carry the same bytes; keep them owner-only too.
        for suffix in ("", "-journal", "-wal", "-shm"):
            path = self._path.with_name(self._path.name + suffix)
            try:
                if path.exists():
                    os.chmod(path, _FILE_MODE)
            except OSError:
                logger.debug("bridge display: could not restrict mode")

    def record(
        self,
        owner_key: str,
        *,
        message_id: str,
        conversation_id: str,
        body: str,
        body_sha256: str,
        sender_kind: str,
        source: str,
        now: float | None = None,
    ) -> bool:
        """Insert or refresh one admitted body. False means refused (never raises)."""

        from .body_frame import MAX_BODY_BYTES

        if source not in _SOURCES or not isinstance(body, str):
            return False
        if len(body.encode("utf-8")) > MAX_BODY_BYTES:
            return False
        # The caller already verified the hash against the screened/served
        # digest; re-deriving it here means no caller can store bytes under a
        # digest they do not match, whatever it was handed.
        # Same construction as ``tool.body_digest`` (sha256 hex of UTF-8);
        # spelled inline so this module never imports the tool surface.
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != body_sha256:
            return False
        kind = sender_kind if sender_kind in _SENDER_KINDS else "unknown"
        stamp = time.time() if now is None else now
        try:
            with self._connect(create=True) as conn:
                assert conn is not None
                conn.execute(
                    "INSERT INTO display_bodies (owner_key, message_id, conversation_id, "
                    "body, body_sha256, sender_kind, admitted_at, source) "
                    "VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(owner_key, message_id) DO UPDATE SET "
                    # A LIVE turn-sourced row is the stronger fact (the agent
                    # was given it, and the excerpt gate keys on it); a later
                    # hub backfill must not downgrade it or move its stamp.
                    # Any other existing row (hub-sourced, or a turn row past
                    # its TTL that ``_prune`` has not reached yet) takes the
                    # new source AND the new stamp: keeping the old stamp made
                    # a refill of an expired row get pruned again by the
                    # ``_prune`` below in this very call (review N2), so the
                    # refill only took on the second view. All right-hand
                    # sides read the OLD row, so the two CASEs agree.
                    "source = CASE WHEN display_bodies.source = 'turn' "
                    "AND display_bodies.admitted_at >= ? THEN 'turn' "
                    "ELSE excluded.source END, "
                    "admitted_at = CASE WHEN display_bodies.source = 'turn' "
                    "AND display_bodies.admitted_at >= ? THEN display_bodies.admitted_at "
                    "ELSE excluded.admitted_at END, "
                    "conversation_id = excluded.conversation_id, body = excluded.body, "
                    "body_sha256 = excluded.body_sha256, sender_kind = excluded.sender_kind",
                    (
                        owner_key,
                        message_id,
                        conversation_id,
                        body,
                        body_sha256,
                        kind,
                        stamp,
                        source,
                        stamp - TTL_SECONDS,
                        stamp - TTL_SECONDS,
                    ),
                )
                self._prune(conn, stamp)
        except (sqlite3.Error, OSError) as failure:
            # Display retention is best effort: a turn's settlement never
            # depends on it. A class name only — never the body.
            logger.debug("bridge display: record failed (%s)", type(failure).__name__)
            return False
        return True

    @staticmethod
    def _prune(conn: sqlite3.Connection, now: float) -> None:
        conn.execute("DELETE FROM display_bodies WHERE admitted_at < ?", (now - TTL_SECONDS,))
        conn.execute("DELETE FROM excerpt_memo_keys WHERE created_at < ?", (now - TTL_SECONDS,))
        total = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(body AS BLOB))), 0) FROM display_bodies"
        ).fetchone()[0]
        if total <= MAX_TOTAL_BYTES:
            return
        excess = total - MAX_TOTAL_BYTES
        for row in conn.execute(
            "SELECT rowid, LENGTH(CAST(body AS BLOB)) AS size FROM display_bodies "
            "ORDER BY admitted_at ASC"
        ).fetchall():
            if excess <= 0:
                break
            conn.execute("DELETE FROM display_bodies WHERE rowid = ?", (row["rowid"],))
            excess -= int(row["size"])

    def bodies(self, owner_key: str, message_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """``message_id -> row`` for the given owner. Never creates the file."""

        wanted = sorted({mid for mid in message_ids if isinstance(mid, str)})
        found: dict[str, dict[str, Any]] = {}
        if not wanted:
            return found
        try:
            with self._connect(create=False) as conn:
                if conn is None:
                    return found
                for start in range(0, len(wanted), 400):
                    chunk = wanted[start : start + 400]
                    for row in conn.execute(
                        "SELECT message_id, conversation_id, body, body_sha256, sender_kind, "
                        "admitted_at, source FROM display_bodies WHERE owner_key = ? "
                        f"AND message_id IN ({','.join('?' * len(chunk))}) AND admitted_at >= ?",
                        (owner_key, *chunk, time.time() - TTL_SECONDS),
                    ).fetchall():
                        found[str(row["message_id"])] = dict(row)
        except (sqlite3.Error, OSError) as failure:
            logger.debug("bridge display: read failed (%s)", type(failure).__name__)
            return {}
        return found

    def purge(self, *, keep_owner_key: str | None = None) -> int:
        """Delete every row not owned by ``keep_owner_key`` (all rows when None)."""

        try:
            with self._connect(create=False) as conn:
                if conn is None:
                    return 0
                if keep_owner_key is None:
                    cursor = conn.execute("DELETE FROM display_bodies")
                    conn.execute("DELETE FROM excerpt_memo_keys")
                else:
                    cursor = conn.execute(
                        "DELETE FROM display_bodies WHERE owner_key != ?", (keep_owner_key,)
                    )
                    conn.execute(
                        "DELETE FROM excerpt_memo_keys WHERE owner_key != ?", (keep_owner_key,)
                    )
                return int(cursor.rowcount or 0)
        except (sqlite3.Error, OSError) as failure:
            logger.debug("bridge display: purge failed (%s)", type(failure).__name__)
            return 0

    def memo_key(self, owner_key: str, *, create: bool = False) -> bytes | None:
        """The owner's live excerpt-memo HMAC key, or None. Never creates the file.

        ``create`` mints a key when none is live, but only inside an existing
        store: no store means no retained body, so nothing to digest.
        """

        now = time.time()
        try:
            with self._connect(create=False) as conn:
                if conn is None:
                    return None
                # Read with the TTL applied, as ``bodies`` does: a key ``_prune``
                # has not reached yet is already expired.
                row = conn.execute(
                    "SELECT key FROM excerpt_memo_keys WHERE owner_key = ? AND created_at >= ?",
                    (owner_key, now - TTL_SECONDS),
                ).fetchone()
                if row is not None:
                    return bytes(row["key"])
                if not create:
                    return None
                # Replace only an EXPIRED key, then read back: two runtimes
                # minting at once agree on one key instead of the later one
                # silently invalidating the other's memo file.
                conn.execute(
                    "INSERT INTO excerpt_memo_keys (owner_key, key, created_at) VALUES (?,?,?) "
                    "ON CONFLICT(owner_key) DO UPDATE SET key = excluded.key, "
                    "created_at = excluded.created_at WHERE excerpt_memo_keys.created_at < ?",
                    (owner_key, os.urandom(_MEMO_KEY_BYTES), now, now - TTL_SECONDS),
                )
                row = conn.execute(
                    "SELECT key FROM excerpt_memo_keys WHERE owner_key = ?", (owner_key,)
                ).fetchone()
                return None if row is None else bytes(row["key"])
        except (sqlite3.Error, OSError) as failure:
            logger.debug("bridge display: memo key failed (%s)", type(failure).__name__)
            return None

    def delete(self, owner_key: str, message_ids: Sequence[str]) -> int:
        """Drop specific rows (a hub ``deleted`` answer)."""

        ids = [mid for mid in message_ids if isinstance(mid, str)]
        if not ids:
            return 0
        try:
            with self._connect(create=False) as conn:
                if conn is None:
                    return 0
                cursor = conn.execute(
                    "DELETE FROM display_bodies WHERE owner_key = ? "
                    f"AND message_id IN ({','.join('?' * len(ids))})",
                    (owner_key, *ids),
                )
                return int(cursor.rowcount or 0)
        except (sqlite3.Error, OSError) as failure:
            logger.debug("bridge display: delete failed (%s)", type(failure).__name__)
            return 0
