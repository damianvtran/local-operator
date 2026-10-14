"""Bounded local retain/recall, with a transport-neutral causal operation log.

SQLite/FTS5 is deliberately the entire runtime dependency. Hindsight's structured
retain/recall lifecycle inspired provenance, freshness and bounded candidate pools
(https://github.com/vectorize-io/hindsight), not its inference/database stack or
benchmark claims. Scopes are visibility filters, not a sandbox against shell access.

Validate *before* opening a store: a rejected payload must not create a database,
FTS entry, history row or diagnostic containing the value. The heuristic shape/ID
checks cannot recognise every opaque free-text secret; never ingest transcripts or
environments wholesale. Secret names, rather than values, belong in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlsplit

from local_operator.compaction.tokens import count_text_tokens
from local_operator.redaction_shapes import match_shape_names
from local_operator.supplements.denylist import is_sensitive

SESSION_EXPORT_NAME = "memory-session.jsonl"
SCOPES = ("global", "session", "team", "agent")
ITEM_TOKENS = 256
ITEM_BYTES = 8192
RECALL_TOKENS = 1200
RECALL_HARD_TOKENS = 2000
MAX_DB_BYTES = 64 * 1024 * 1024
MAX_OWNER_OPS = 20000
MAX_REVISIONS = 128
MAX_TOMBSTONES = 5000
MAX_BATCH_BYTES = 256 * 1024
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
# Contextual ID checks complement the shared credential shape table, which does
# not claim to recognise identity documents. Names such as secret PASSPORT_NO
# remain useful and are not values; numeric assignments/documents are refused.
_ID = re.compile(
    r"(?<![\w-])(?:\d{3}[- ]\d{2}[- ]\d{4}|(?:\d[ -]?){13,19})(?![\w-])"
    r"|\b(?:passport|ssn|social.security|driver.?s?.?licen[cs]e|licen[cs]e|national.id|card.number)"
    r"(?:\s*(?:no\.?|number))?\s*[:=#]?\s+(?=[A-Z0-9 -]{0,24}\d)[A-Z0-9][A-Z0-9 -]{4,24}\b",
    re.I,
)
_STOP = set(
    (
        "a an and are as at be by can could did do does for from have how i in is it me my of on "
        "or our please should that the their this to use was we what when where which who will "
        "with would you your create new show find tell get save remember memory work help want "
        "need about"
    ).split()
)


class MemoryError(ValueError):
    """Safe, value-free actionable refusal suitable for user/tool surfaces."""


def safe_segment(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value in (".", ".."):
        raise MemoryError("Memory identity must be one safe segment (1–128 characters).")
    return value


def guard(value: Any) -> None:
    """Inspect every string, including metadata and wire-only fields, without echo."""
    if isinstance(value, dict):
        for key, child in value.items():
            guard(key)
            guard(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            guard(child)
    elif isinstance(value, str):
        if len(value.encode("utf-8")) > ITEM_BYTES:
            raise MemoryError("Memory field exceeds the 8192-byte cap; split into atomic facts.")
        if match_shape_names(value) or _ID.search(value):
            raise MemoryError(
                "Sensitive value refused. Store it with lop secret; memory may hold its NAME only."
            )
        # Check path-like tokens, not the complete prose (which can contain a
        # harmless reference beside a sensitive path). Never persist the latter.
        for word in re.findall(r"(?:~?/|[A-Za-z]:[/\\])[^\s<>\"']+", value):
            if is_sensitive(word):
                raise MemoryError(
                    "Sensitive path refused; retain a safe locator or secret NAME instead."
                )


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _id() -> str:
    # UUIDv7 layout without depending on Python 3.14 (the supported floor is 3.12).
    bits = (int(time.time() * 1000) << 80) | (7 << 76) | (uuid.uuid4().int & ((1 << 76) - 1))
    bits = (bits & ~(3 << 62)) | (2 << 62)
    return str(uuid.UUID(int=bits))


@dataclass(frozen=True)
class MemoryIdentity:
    session_id: str = ""
    team_id: str = ""
    agent_id: str = ""

    def owner(self, scope: str) -> str:
        if scope == "global":
            return "global"
        if scope not in SCOPES:
            raise MemoryError("Unknown scope; use session, team, agent or global.")
        owner = getattr(self, scope + "_id")
        if scope == "agent" and owner in ("", "main"):
            raise MemoryError(
                "Anonymous agent 'main' has no persistent agent scope. "
                "Use session or global memory."
            )
        if not owner:
            raise MemoryError(f"No authorized {scope} identity; use session or global memory.")
        return safe_segment(owner)

    def scopes(self) -> list[tuple[str, str]]:
        result = [("global", "global")]
        for scope in SCOPES[1:]:
            try:
                result.append((scope, self.owner(scope)))
            except MemoryError:
                continue
        return result


def validate_fact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MemoryError("Memory content must be a JSON object; read guide://memory.")
    guard(raw)
    allowed = {
        "title",
        "body",
        "tags",
        "aliases",
        "source",
        "status",
        "verified_at",
        "expires_at",
        "pinned",
        "subject",
        "created_at",
        "source_time",
    }
    if set(raw) - allowed:
        raise MemoryError("Unknown memory field; read guide://memory for the narrow protocol.")
    fact = {
        "title": "",
        "body": "",
        "tags": [],
        "aliases": [],
        "source": {},
        "status": "active",
        "verified_at": None,
        "expires_at": None,
        "pinned": False,
        "subject": "",
        "created_at": None,
        "source_time": None,
        **raw,
    }
    for key in ("title", "body", "subject"):
        if not isinstance(fact[key], str):
            raise MemoryError("Memory title/body/subject must be text.")
    if (
        not fact["title"].strip()
        or not fact["body"].strip()
        or len(fact["title"]) > 160
        or len(fact["subject"]) > 128
    ):
        raise MemoryError(
            "Supply an atomic body and a title (1–160 characters); subject cap is 128."
        )
    for key in ("tags", "aliases"):
        if (
            not isinstance(fact[key], list)
            or len(fact[key]) > 12
            or any(not isinstance(x, str) or not x.strip() or len(x) > 64 for x in fact[key])
        ):
            raise MemoryError(
                "Tags/aliases must contain at most 12 short strings (64 characters each)."
            )
    if (
        fact["status"] not in ("active", "candidate", "disputed", "superseded", "expired")
        or type(fact["pinned"]) is not bool
    ):
        raise MemoryError("Invalid memory status or pinned flag.")
    for key in ("verified_at", "expires_at", "created_at", "source_time"):
        x = fact[key]
        if x is not None and (type(x) not in (int, float) or not 0 <= x <= 32503680000):
            raise MemoryError("Freshness timestamps must be finite Unix seconds or null.")
    source = fact["source"]
    if (
        not isinstance(source, dict)
        or len(source) > 6
        or set(source) - {"session", "entry", "tool", "kind", "uri", "author"}
        or any(not isinstance(x, str) or len(x) > 256 for x in source.values())
    ):
        raise MemoryError(
            "Source allows session/entry/tool/kind/uri/author, each at most 256 characters."
        )
    wire = _dump(fact)
    if len(wire.encode()) > ITEM_BYTES or count_text_tokens(wire) > ITEM_TOKENS:
        raise MemoryError(
            "Memory item exceeds 256 tokens/8192 bytes including metadata. Split or consolidate it."
        )
    return fact


def validate_op(op: Any, *, allow_redacted: bool = False) -> dict[str, Any]:
    if not isinstance(op, dict) or len(_dump(op).encode()) > 16384:
        raise MemoryError("Invalid or oversized memory operation.")
    guard(op)
    required = {
        "v",
        "op_id",
        "entity_id",
        "origin_device",
        "seq",
        "hlc",
        "parents",
        "kind",
        "scope",
        "owner_key",
        "fact",
        "body_sha",
    }
    if set(op) != required or op["v"] != 1:
        raise MemoryError("Unsupported memory operation version/fields; update both peers.")
    for key in ("op_id", "entity_id", "origin_device", "owner_key"):
        safe_segment(op[key])
    if type(op["seq"]) is not int or not 1 <= op["seq"] <= 2**63 - 1:
        raise MemoryError("Invalid origin sequence.")
    if (
        not isinstance(op["hlc"], list)
        or len(op["hlc"]) != 3
        or any(type(x) is not int or not 0 <= x <= 2**63 - 1 for x in op["hlc"][:2])
        or op["hlc"][2] != op["origin_device"]
    ):
        raise MemoryError("Invalid memory logical clock.")
    if not isinstance(op["parents"], list) or len(op["parents"]) > 16:
        raise MemoryError("Memory parents must be a bounded list.")
    for parent in op["parents"]:
        safe_segment(parent)
    if (
        op["scope"] not in SCOPES
        or (op["scope"] == "global" and op["owner_key"] != "global")
        or (op["scope"] == "agent" and op["owner_key"] == "main")
    ):
        raise MemoryError("Invalid memory scope/owner.")
    if not isinstance(op["body_sha"], str) or not re.fullmatch(r"[0-9a-f]{64}", op["body_sha"]):
        raise MemoryError("Invalid memory content hash.")
    if op["kind"] == "put" and op["fact"] is None and allow_redacted:
        return op
    if op["kind"] == "put":
        fact = validate_fact(op["fact"])
        if hashlib.sha256(_dump(fact).encode()).hexdigest() != op["body_sha"]:
            raise MemoryError("Memory content hash mismatch.")
    elif op["kind"] != "tombstone" or op["fact"] is not None:
        raise MemoryError("Tombstones contain no memory body.")
    return op


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ops (
 op_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, origin TEXT NOT NULL,
 seq INTEGER NOT NULL, scope TEXT NOT NULL, owner TEXT NOT NULL,
 kind TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(origin,seq));
CREATE INDEX IF NOT EXISTS ops_entity ON ops(entity_id);
CREATE INDEX IF NOT EXISTS ops_owner ON ops(scope,owner);
CREATE TABLE IF NOT EXISTS current (
 entity_id TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL,
 op_id TEXT NOT NULL, fact TEXT NOT NULL, tokens INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS current_owner ON current(scope,owner);
CREATE VIRTUAL TABLE IF NOT EXISTS recall USING fts5(
 entity_id UNINDEXED,title,tags,body,tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(recall,'row');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS captures(session TEXT,compaction TEXT,status TEXT,n INTEGER,
 PRIMARY KEY(session,compaction));
"""


class MemoryStore:
    def __init__(
        self,
        config: Path,
        identity: MemoryIdentity,
        session_dir: Path | None = None,
        *,
        origin: str | None = None,
    ):
        self.root = Path(config) / "memory"
        self.path = self.root / "memory.db"
        self.identity = identity
        self.session_dir = session_dir
        self.origin = safe_segment(origin) if origin else None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(self.path, timeout=5)
        os.chmod(self.path, 0o600)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA secure_delete=ON")
            db.execute("PRAGMA journal_size_limit=1048576")
            db.execute("PRAGMA wal_autocheckpoint=64")
            db.execute(f"PRAGMA max_page_count={MAX_DB_BYTES // 4096}")
            db.executescript(_SCHEMA)
            yield db
        finally:
            db.close()

    def _origin(self, db: sqlite3.Connection) -> str:
        if self.origin:
            return self.origin
        row = db.execute("SELECT value FROM meta WHERE key='origin'").fetchone()
        if row:
            return str(row[0])
        value = _id()
        db.execute("INSERT INTO meta VALUES ('origin',?)", (value,))
        return value

    @staticmethod
    def _ops(db: sqlite3.Connection, entity: str) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM ops WHERE entity_id=?", (entity,))
        ]

    @staticmethod
    def _heads(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parents = {p for op in ops for p in op["parents"]}
        return sorted(
            (op for op in ops if op["op_id"] not in parents),
            key=lambda op: (op["hlc"], op["op_id"]),
            reverse=True,
        )

    def _reindex(self, db: sqlite3.Connection, entity: str) -> None:
        ops = self._ops(db, entity)
        db.execute("DELETE FROM recall WHERE entity_id=?", (entity,))
        db.execute("DELETE FROM current WHERE entity_id=?", (entity,))
        if any(op["kind"] == "tombstone" for op in ops):
            # Forgotten IDs are permanently dead. Bodies in old puts are removed
            # too; keeping their hash and ancestry still permits convergence.
            for op in ops:
                if op["fact"] is not None:
                    op["fact"] = None
                    db.execute("UPDATE ops SET data=? WHERE op_id=?", (_dump(op), op["op_id"]))
            return
        heads = self._heads(ops)
        if not heads:
            return
        op = heads[0]
        fact = op["fact"]
        db.execute(
            "INSERT INTO current VALUES (?,?,?,?,?,?)",
            (
                entity,
                op["scope"],
                op["owner_key"],
                op["op_id"],
                _dump(fact),
                count_text_tokens(_dump(fact)),
            ),
        )
        if fact["status"] in ("active", "candidate", "disputed"):
            db.execute(
                "INSERT INTO recall VALUES (?,?,?,?)",
                (entity, fact["title"], " ".join(fact["tags"] + fact["aliases"]), fact["body"]),
            )

    @staticmethod
    def _quota(db: sqlite3.Connection, scope: str, owner: str) -> None:
        count = db.execute(
            "SELECT count(*) FROM ops WHERE scope=? AND owner=?", (scope, owner)
        ).fetchone()[0]
        tombs = db.execute(
            "SELECT count(*) FROM ops WHERE scope=? AND owner=? AND kind='tombstone'",
            (scope, owner),
        ).fetchone()[0]
        tokens = db.execute(
            "SELECT coalesce(sum(tokens),0) FROM current WHERE scope=? AND owner=?", (scope, owner)
        ).fetchone()[0]
        if (
            count > MAX_OWNER_OPS
            or tombs > MAX_TOMBSTONES
            or tokens > (16000 if scope == "global" else 50000)
        ):
            raise MemoryError(
                "Memory quota reached (including history/tombstones). Inspect memory://, "
                "consolidate explicitly, or export history; no facts were evicted."
            )

    def put(
        self,
        scope: str,
        fact: dict[str, Any],
        *,
        entity: str | None = None,
        expected_version: str | None = None,
        forget: bool = False,
    ) -> dict[str, Any]:
        owner = self.identity.owner(scope)
        if entity is not None:
            safe_segment(entity)
            if not expected_version:
                raise MemoryError(
                    "Correction/forget needs expected_version from read; no blind overwrite."
                )
            safe_segment(expected_version)
        elif forget:
            raise MemoryError("Forget needs an existing memory ID and expected_version.")
        clean = None
        if not forget:
            # Host-derived provenance is present even when the agent omitted it.
            # A source's external date remains unknown unless explicitly supplied;
            # creation time records retention, never verification of the claim.
            if not isinstance(fact, dict):
                raise MemoryError("Memory content must be a JSON object.")
            guard(fact)
            source = fact.get("source", {})
            if not isinstance(source, dict):
                raise MemoryError("Memory source must be an object.")
            source = {
                "session": self.identity.session_id,
                "kind": "reported",
                "author": self.identity.agent_id or "anonymous",
                **source,
            }
            clean = validate_fact(
                dict(fact, source=source, created_at=fact.get("created_at", time.time()))
            )
        entity = entity or _id()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ops = self._ops(db, entity)
            if ops:
                if any(op["scope"] != scope or op["owner_key"] != owner for op in ops):
                    raise MemoryError("Memory not found in the authorized scope.")
                if any(op["kind"] == "tombstone" for op in ops):
                    raise MemoryError(
                        "Memory was forgotten. Store a new entity instead; resurrection is refused."
                    )
                heads = self._heads(ops)
                if heads[0]["op_id"] != expected_version:
                    raise MemoryError(
                        "Memory changed concurrently. Read its current revision and retry."
                    )
                if len(ops) >= MAX_REVISIONS and not forget:
                    raise MemoryError(
                        "Memory revision cap reached; consolidate into a new "
                        "explicitly linked fact."
                    )
                parents = [op["op_id"] for op in heads]
            elif expected_version:
                raise MemoryError("Memory not found in the authorized scope.")
            else:
                parents = []
            origin = self._origin(db)
            seq = db.execute(
                "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?", (origin,)
            ).fetchone()[0]
            last = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
            now = int(time.time() * 1000)
            clock = max(now, int(last[0]) + 1 if last else 0)
            op = {
                "v": 1,
                "op_id": _id(),
                "entity_id": entity,
                "origin_device": origin,
                "seq": seq,
                "hlc": [clock, 0, origin],
                "parents": parents,
                "kind": "tombstone" if forget else "put",
                "scope": scope,
                "owner_key": owner,
                "fact": clean,
                "body_sha": hashlib.sha256(_dump(clean).encode()).hexdigest(),
            }
            validate_op(op)
            db.execute(
                "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                (op["op_id"], entity, origin, seq, scope, owner, op["kind"], _dump(op)),
            )
            db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
            self._reindex(db, entity)
            self._quota(db, scope, owner)
            # A session's portable log is fsync'd before the index commit. If
            # that commit crashes, lazy import reconstructs it idempotently.
            if scope == "session" and self.session_dir:
                self._export(db, owner)
            db.commit()
            if forget:
                db.execute("INSERT INTO recall(recall) VALUES ('optimize')")
                db.commit()
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "id": entity,
            "version": op["op_id"],
            "scope": scope,
            "status": "forgotten" if forget else clean["status"],
        }

    def _export(self, db: sqlite3.Connection, owner: str) -> None:
        assert self.session_dir is not None
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / SESSION_EXPORT_NAME
        # Snapshot replacement rather than leaving forgotten bodies in an
        # append-only portable log. All causal IDs/hashes remain, bodies do not.
        rows = db.execute(
            "SELECT data FROM ops WHERE scope='session' AND owner=? ORDER BY origin,seq", (owner,)
        ).fetchall()
        data = "".join(row[0] + "\n" for row in rows)
        if len(data.encode()) > 16 * 1024 * 1024:
            raise MemoryError(
                "Session memory export cap reached. Consolidate/export before adding more history."
            )
        temp = path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def apply_ops(
        self,
        batch: list[dict[str, Any]],
        *,
        origins: set[str] | None = None,
        contiguous: bool = True,
    ) -> int:
        """Atomic idempotent merge. Authenticate origins before calling from mesh.

        A redacted put is legal ONLY beside an irreversible tombstone for its
        entity. Keeping its op/hash preserves causal/vector identity without
        reintroducing the forgotten content in export or later replication.
        """
        if (
            not isinstance(batch, list)
            or len(batch) > 256
            or len(_dump(batch).encode()) > MAX_BATCH_BYTES
        ):
            raise MemoryError("Memory batch exceeds 256 operations/256 KiB.")
        tombs = {
            op.get("entity_id")
            for op in batch
            if isinstance(op, dict) and op.get("kind") == "tombstone"
        }
        for op in batch:
            validate_op(op, allow_redacted=True)
            if origins is not None and op["origin_device"] not in origins:
                raise MemoryError("Memory origin is not an active trusted network member.")
        if not batch:
            return 0
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            changed: set[str] = set()
            owners: set[tuple[str, str]] = set()
            for op in sorted(batch, key=lambda x: (x["origin_device"], x["seq"])):
                entity = op["entity_id"]
                old_ops = self._ops(db, entity)
                if any(
                    x["scope"] != op["scope"] or x["owner_key"] != op["owner_key"] for x in old_ops
                ):
                    raise MemoryError("Memory entity scope/owner is immutable.")
                dead = entity in tombs or any(x["kind"] == "tombstone" for x in old_ops)
                if op["kind"] == "put" and op["fact"] is None and not dead:
                    raise MemoryError("Redacted memory put requires its tombstone.")
                old = db.execute("SELECT data FROM ops WHERE op_id=?", (op["op_id"],)).fetchone()
                if old:
                    previous = json.loads(old[0])
                    if dead:
                        previous["fact"] = None
                        comparison = dict(op, fact=None)
                    else:
                        comparison = op
                    if previous != comparison:
                        raise MemoryError(
                            "Memory operation identity cannot be reused with different content."
                        )
                    continue
                if contiguous:
                    seq = db.execute(
                        "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?",
                        (op["origin_device"],),
                    ).fetchone()[0]
                    if op["seq"] != seq:
                        raise MemoryError("Memory sequence gap; refresh peer state and retry.")
                if len(old_ops) >= MAX_REVISIONS:
                    raise MemoryError("Memory revision cap reached; no conflicts discarded.")
                stored = dict(op, fact=None) if dead and op["kind"] == "put" else op
                try:
                    db.execute(
                        "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                        (
                            op["op_id"],
                            entity,
                            op["origin_device"],
                            op["seq"],
                            op["scope"],
                            op["owner_key"],
                            op["kind"],
                            _dump(stored),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise MemoryError("Memory origin sequence reused; refresh peer state.") from exc
                changed.add(entity)
                owners.add((op["scope"], op["owner_key"]))
            for entity in changed:
                self._reindex(db, entity)
            for scope, owner in owners:
                self._quota(db, scope, owner)
                if scope == "session" and owner == self.identity.session_id and self.session_dir:
                    self._export(db, owner)
            db.commit()
            if tombs:
                db.execute("INSERT INTO recall(recall) VALUES ('optimize')")
                db.commit()
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return len(changed)

    def import_session(self) -> None:
        if not self.session_dir or not self.identity.session_id:
            return
        path = self.session_dir / SESSION_EXPORT_NAME
        if not path.exists():
            return
        if path.stat().st_size > 16 * 1024 * 1024:
            raise MemoryError("Portable session memory exceeds its 16 MiB cap.")
        ops = []
        for line in path.read_text().splitlines():
            op = json.loads(line)
            validate_op(op, allow_redacted=True)
            if op["scope"] != "session":
                raise MemoryError("Session memory export contains a non-session operation.")
            if op["owner_key"] != self.identity.session_id:
                # Replica promotion is a new session identity; derive fresh
                # causal/entity IDs deterministically so restart is idempotent.
                prefix = self.identity.session_id + "|"
                op = dict(
                    op,
                    owner_key=self.identity.session_id,
                    entity_id=hashlib.sha256((prefix + op["entity_id"]).encode()).hexdigest(),
                    op_id=hashlib.sha256((prefix + op["op_id"]).encode()).hexdigest(),
                    origin_device=hashlib.sha256(
                        (prefix + op["origin_device"]).encode()
                    ).hexdigest(),
                    parents=[
                        hashlib.sha256((prefix + p).encode()).hexdigest() for p in op["parents"]
                    ],
                )
                op["hlc"] = [*op["hlc"][:2], op["origin_device"]]
            ops.append(op)
        # Portable files are scope-filtered: gaps from other scopes are expected.
        # The complete file was validated before any batch is persisted.
        ops.sort(key=lambda op: op["kind"] != "tombstone")
        for start in range(0, len(ops), 128):
            self.apply_ops(ops[start : start + 128], contiguous=False)

    def _where(self, scope: str | None) -> tuple[str, list[str]]:
        pairs = [(scope, self.identity.owner(scope))] if scope else self.identity.scopes()
        return "(" + " OR ".join("(scope=? AND owner=?)" for _ in pairs) + ")", [
            v for pair in pairs for v in pair
        ]

    def show(self, scope: str, entity: str) -> dict[str, Any]:
        owner = self.identity.owner(scope)
        safe_segment(entity)
        if not self.path.exists():
            raise MemoryError("Memory not found in the authorized scope.")
        with self.connection() as db:
            ops = self._ops(db, entity)
            if not ops or any(op["scope"] != scope or op["owner_key"] != owner for op in ops):
                raise MemoryError("Memory not found in the authorized scope.")
            tombs = [op for op in ops if op["kind"] == "tombstone"]
            heads = self._heads(tombs or ops)
            return {
                "id": entity,
                "version": heads[0]["op_id"],
                "scope": scope,
                "owner": owner,
                "status": "forgotten" if tombs else heads[0]["fact"]["status"],
                "fact": None if tombs else heads[0]["fact"],
                "conflicts": (
                    []
                    if tombs
                    else [{"version": op["op_id"], "fact": op["fact"]} for op in heads[1:]]
                ),
                "revisions": len(ops),
                "untrusted": True,
            }

    def inventory(self, scope: str | None = None, *, limit: int = 20) -> dict[str, Any]:
        where, values = self._where(scope)
        if not 1 <= limit <= 50:
            raise MemoryError("Inventory limit must be 1–50.")
        if not self.path.exists():
            return {
                "items": [],
                "active_tokens": 0,
                "ops": 0,
                "limits": {"global_tokens": 16000, "owner_tokens": 50000, "item_tokens": 256},
            }
        with self.connection() as db:
            rows = db.execute(
                f"SELECT entity_id,scope,op_id,fact FROM current WHERE {where} "
                "ORDER BY entity_id DESC LIMIT ?",
                [*values, limit],
            ).fetchall()
            usage = db.execute(
                f"SELECT coalesce(sum(tokens),0) FROM current WHERE {where}", values
            ).fetchone()[0]
            count = db.execute(f"SELECT count(*) FROM ops WHERE {where}", values).fetchone()[0]
            return {
                "items": [
                    {
                        "id": row[0],
                        "scope": row[1],
                        "version": row[2],
                        "title": json.loads(row[3])["title"],
                        "status": json.loads(row[3])["status"],
                    }
                    for row in rows
                ],
                "active_tokens": usage,
                "ops": count,
                "storage_bytes": sum(
                    p.stat().st_size for p in self.root.glob("memory.db*") if p.is_file()
                ),
                "limits": {
                    "global_tokens": 16000,
                    "owner_tokens": 50000,
                    "item_tokens": 256,
                    "owner_ops": MAX_OWNER_OPS,
                    "revisions": MAX_REVISIONS,
                    "storage_bytes": MAX_DB_BYTES,
                },
                "help": (
                    "Inventory is not the bank. Search ?q=words, read an ID, "
                    "consolidate explicitly; retrieval does not refresh verification."
                ),
            }

    def search(
        self,
        query: str,
        scope: str | None = None,
        *,
        tokens: int = RECALL_TOKENS,
        limit: int = 10,
        suggest: bool = False,
    ) -> list[dict[str, Any]]:
        guard(query)
        if (
            not isinstance(query, str)
            or len(query) > 2048
            or type(tokens) is not int
            or not 1 <= tokens <= RECALL_HARD_TOKENS
            or not 1 <= limit <= 20
        ):
            raise MemoryError("Search accepts ≤2048 characters, 1–20 results and 1–2000 tokens.")
        where, values = self._where(scope)
        terms = list(
            dict.fromkeys(
                t.lower()
                for t in re.findall(r"[\w]+", query)
                if len(t) >= 2 and t.lower() not in _STOP
            )
        )[:24]
        if not terms or not self.path.exists():
            return []
        with self.connection() as db:
            match = " OR ".join('"' + term + '"' for term in terms)
            rows = db.execute(
                "SELECT c.entity_id,c.scope,c.op_id,c.fact FROM recall JOIN current c "
                "ON c.entity_id=recall.entity_id WHERE recall MATCH ? AND "
                f"{where} ORDER BY bm25(recall,0,4,2,1),c.entity_id LIMIT 10",
                [match, *values],
            ).fetchall()
            result: list[dict[str, Any]] = []
            used = 0
            # Document frequency is indexed, not an O(bank) Python pass. Ask FTS
            # itself for per-term matches so porter stemming cannot disagree.
            authorized_count = db.execute(
                f"SELECT count(*) FROM current WHERE {where}", values
            ).fetchone()[0]
            for row in rows:
                fact = json.loads(row[3])
                if fact["expires_at"] is not None and fact["expires_at"] <= time.time():
                    continue
                if suggest and fact["status"] == "disputed":
                    continue
                if suggest:
                    hits = []
                    specific = False
                    for term in terms:
                        hit = db.execute(
                            "SELECT 1 FROM recall WHERE entity_id=? AND recall MATCH ?",
                            (row[0], '"' + term + '"'),
                        ).fetchone()
                        if hit:
                            hits.append(term)
                            freq = db.execute(
                                "SELECT count(*) FROM recall JOIN current c "
                                "ON c.entity_id=recall.entity_id WHERE recall MATCH ? "
                                f"AND {where}",
                                ['"' + term + '"', *values],
                            ).fetchone()[0]
                            specific |= len(term) >= 4 and freq <= max(2, authorized_count * 0.01)
                    coverage = len(hits) / len(terms)
                    if not ((len(hits) >= 2 and coverage >= 0.4) or (specific and coverage >= 0.5)):
                        continue
                item = {
                    "id": row[0],
                    "scope": row[1],
                    "version": row[2],
                    "title": fact["title"],
                    "status": fact["status"],
                    "untrusted": True,
                }
                if not suggest:
                    item.update(
                        body=fact["body"],
                        source=fact["source"],
                        verified_at=fact["verified_at"],
                        expires_at=fact["expires_at"],
                    )
                cost = count_text_tokens(_dump(item))
                if used + cost <= tokens:
                    result.append(item)
                    used += cost
                if len(result) >= min(limit, 3 if suggest else limit):
                    break
            return result


def parse_url(url: str) -> tuple[str | None, str | None, dict[str, str]]:
    parts = urlsplit(url)
    if parts.scheme != "memory" or parts.fragment or parts.username or parts.password or parts.port:
        raise MemoryError(
            "Use memory://, memory://session, or memory://session/ID; owners are never addresses."
        )
    scope = parts.netloc or None
    entity = parts.path.lstrip("/") or None
    if scope is not None and scope not in SCOPES:
        raise MemoryError("Unknown memory scope; never guess another session/team owner.")
    if entity:
        safe_segment(entity)
    query = parse_qs(parts.query, strict_parsing=True)
    if set(query) - {"q", "tokens", "limit"} or any(len(v) != 1 for v in query.values()):
        raise MemoryError("Memory read supports q, tokens and limit only.")
    return scope, entity, {k: v[0] for k, v in query.items()}
