"""Owner-LOCAL, keyless finite-operation fences, not peer-sync or positive authority.

A withheld holder is a projection, not a durable restriction: any legacy positive
writer can restore it after a lost deny compensation. Publish PENDING *before*
mutable policy/projection, and keep it until exact signed completion readbacks.
The sentinel distinguishes untouched legacy owners from a missing/corrupt journal
on an initialized owner. Losing BOTH stores is not recoverable proof of authority.

Lock order: source record -> consent -> journal -> network record -> placement
(the network-record step is what relay pairing admission takes first, so finite
apply must take it ahead of placement too, never first under it — R4-1). Positive producers
already holding placement read the atomic negative snapshot WITHOUT reverse locks;
actual serving serializes consent/journal and refuses unknown durability. Never
sync these records, export a bearer/hash, or hold locks over native/network work.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn

from local_operator.network import store
from local_operator.network.credentials import consent, placement
from local_operator.network.types import MeshRefusal

LIMIT = 262144
MAX_RECORDS = 64
MAX_DENIES = 256
FIELDS = {
    "id",
    "revision",
    "status",
    "recipient_device",
    "public_key_fingerprint",
    "key",
    "scope",
    "approval_id",
    "run_id",
    "request_digest",
    "policy_revision",
    "denies",
}


def _refuse(code: str = "finite_journal_unknown") -> NoReturn:
    raise MeshRefusal(
        code,
        "Finite sharing durability is unconfirmed; future borrows are refused. "
        "Inspect the owner-local finite recovery journal; use a fresh exact signed finite-share "
        "request when its context is readable. Do not re-share, onboard or replay an old approval.",
    )


def _read(path: Path, limit: int) -> Any:
    if path.is_symlink() or not path.is_file():
        _refuse()
    try:
        # O_NOFOLLOW also closes the race between the preceding metadata probe
        # and opening a file replaced by a symlink. No values are logged.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            _refuse()
        return json.loads(raw, object_pairs_hook=consent._pairs, parse_constant=lambda _: _refuse())
    except (OSError, ValueError):
        _refuse()


class Journal:
    def __init__(self, network: str, owner: str, root: Path | None = None):
        store.safe_component(network)
        store.safe_component(owner)
        self.network, self.owner = network, owner
        self.root = placement._config_root(root)
        self.path = placement.placement_path(network, self.root).with_name("finite-journal.json")
        self.sentinel = self.path.with_name("finite-journal.initialized.json")
        if self.path.parent.is_symlink() or self.path.is_symlink() or self.sentinel.is_symlink():
            _refuse()
        self._permit: object | None = None

    @contextmanager
    def locked(self) -> Iterator[None]:
        # A never-initialized legacy owner has no fence to serialize and must
        # not gain new files on an ordinary serve. The consent document lock
        # already serializes a concurrent first initialization: a finite apply
        # holds it for its whole fence-publish/projection critical section.
        if not self.path.exists() and not self.sentinel.exists():
            yield
            return
        placement.credentials_dir_for(self.network, self.root)
        with store._write_lock(self.path):
            with placement._cross_process_lock(self.path.with_name(".finite-journal.lock"), root=self.root):
                try:
                    yield
                finally:
                    self._permit = None

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "network_id": self.network,
            "owner_device": self.owner,
            "revision": 1,
            "records": [],
        }

    def _validate(self, row: Any) -> dict[str, Any]:
        try:
            consent._exact(
                row,
                {"schema", "network_id", "owner_device", "revision", "records"},
                {"schema", "network_id", "owner_device", "revision", "records"},
            )
            if type(row["schema"]) is not int or row["schema"] != 1:
                _refuse()
            if row["network_id"] != self.network or row["owner_device"] != self.owner:
                _refuse()
            consent._revision(row["revision"])
            records = row["records"]
            if not isinstance(records, list) or len(records) > MAX_RECORDS:
                _refuse()
            ids: set[str] = set()
            active: set[tuple[str, str]] = set()
            total = 0
            for rec in records:
                consent._exact(rec, FIELDS, FIELDS)
                for field in ("id", "approval_id", "run_id", "request_digest"):
                    if not isinstance(rec[field], str) or len(rec[field].encode()) > 256:
                        _refuse()
                    consent._ref(rec[field])
                if rec["id"] in ids:
                    _refuse()
                ids.add(rec["id"])
                consent._revision(rec["revision"])
                consent._revision(rec["policy_revision"])
                store.safe_component(rec["recipient_device"])
                if (
                    not isinstance(rec["key"], str)
                    or not rec["key"]
                    or len(rec["key"].encode()) > 200
                ):
                    _refuse()
                if rec["scope"] not in {"device", "session"} or rec["status"] not in {
                    "pending",
                    "failed",
                    "complete",
                }:
                    _refuse()
                fp = rec["public_key_fingerprint"]
                if (
                    not isinstance(fp, str)
                    or len(fp) != 71
                    or not fp.startswith("sha256:")
                    or any(c not in "0123456789abcdef" for c in fp[7:])
                ):
                    _refuse()
                if rec["status"] != "complete":
                    pair = (rec["recipient_device"], rec["key"])
                    if pair in active:
                        _refuse()
                    active.add(pair)
                denies = rec["denies"]
                if not isinstance(denies, list) or not denies:
                    _refuse()
                total += len(denies)
                seen: set[str] = set()
                for denial in denies:
                    consent._exact(
                        denial,
                        {"selector", "revision", "reason", "approval_ref"},
                        {"selector", "revision", "reason", "approval_ref"},
                    )
                    selector = consent._selector(denial["selector"])
                    if selector["credential_key"] != rec["key"]:
                        _refuse()
                    consent._revision(denial["revision"])
                    consent._ref(denial["approval_ref"])
                    if denial["reason"] not in {"withdrawn", "local_only", "logout", "expired"}:
                        _refuse()
                    encoded = json.dumps(
                        {k: denial[k] for k in ("selector", "revision", "approval_ref")},
                        sort_keys=True,
                    )
                    if encoded in seen:
                        _refuse()
                    seen.add(encoded)
            if total > MAX_DENIES or len(json.dumps(row, ensure_ascii=False).encode()) > LIMIT:
                _refuse()
        except (MeshRefusal, TypeError, ValueError):
            _refuse()
        return row

    def load(self) -> dict[str, Any]:
        if self.path.parent.is_symlink() or self.path.is_symlink() or self.sentinel.is_symlink():
            _refuse()
        exists, marked = self.path.exists(), self.sentinel.exists()
        if not exists and not marked:
            return self._empty()  # ONLY a never-initialized legacy owner
        if not exists or not marked:
            _refuse()
        marker = _read(self.sentinel, 1024)
        if (
            not isinstance(marker, dict)
            or type(marker.get("schema")) is not int
            or marker != {"schema": 1, "network_id": self.network, "owner_device": self.owner}
        ):
            _refuse()
        return self._validate(_read(self.path, LIMIT))

    def initialize(self) -> None:
        if self.path.exists() or self.sentinel.exists():
            self.load()
            return
        # Journal first: interrupted sentinel publication fails CLOSED, not as
        # an empty legacy owner. Neither initialization step changes authority.
        self._publish(self.path, self._empty())
        self._publish(
            self.sentinel, {"schema": 1, "network_id": self.network, "owner_device": self.owner}
        )
        self.load()

    def _publish(self, path: Path, row: dict[str, Any]) -> None:
        if path.parent.is_symlink() or path.is_symlink():
            _refuse()
        store._write_private_json(path, row)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if _read(path, LIMIT) != row:
            _refuse()

    def save(self, row: dict[str, Any], expected: int) -> dict[str, Any]:
        current = self.load()
        if current["revision"] != expected:
            _refuse("finite_journal_conflict")
        candidate = copy.deepcopy(row)
        candidate["revision"] = consent._revision(expected + 1)
        self._validate(candidate)
        self._publish(self.path, candidate)
        if self.load() != candidate:
            _refuse()
        return candidate

    def pending(self, recipient: str, key: str) -> dict[str, Any] | None:
        return next(
            (
                copy.deepcopy(r)
                for r in self.load()["records"]
                if r["recipient_device"] == recipient
                and r["key"] == key
                and r["status"] != "complete"
            ),
            None,
        )

    def peek(self, recipient: str, key: str) -> dict[str, Any] | None:
        """Hot-path read: a legacy owner with NO journal files has no fence.

        Every owner borrow/mint/copy consults this, so the uninitialized case
        must not create files or take a lock. Once ANY fence file exists, the
        strict fail-closed ``pending`` reader is used instead.
        """
        if not self.path.exists() and not self.sentinel.exists():
            return None
        return self.pending(recipient, key)

    def _begin(self, data: dict[str, Any], approval_id: str, run_id: str, digest: str) -> object:
        self.initialize()
        row = self.load()
        old = self.pending(data["recipient_device"], data["key"])
        if data["guard"] != ({"id": old["id"], "revision": old["revision"]} if old else None):
            _refuse("finite_journal_conflict")
        rec = {
            "id": old["id"] if old else "tx_" + secrets.token_hex(12),
            "revision": consent._revision(old["revision"] + 1) if old else 1,
            "status": "pending",
            "recipient_device": data["recipient_device"],
            "public_key_fingerprint": data["public_key_fingerprint"],
            "key": data["key"],
            "scope": data["scope"],
            "approval_id": approval_id,
            "run_id": run_id,
            "request_digest": digest,
            "policy_revision": data["policy_revision"],
            "denies": copy.deepcopy(data["denies"]),
        }
        if old:
            row["records"] = [rec if r["id"] == old["id"] else r for r in row["records"]]
        else:
            row["records"].append(rec)
        self.save(row, row["revision"])
        # The ONLY completion handle is local to this HELD journal instance;
        # no public bypass boolean, peer message or generic producer accepts it.
        self._permit = (object(), rec)
        return self._permit

    def _finish(self, permit: object) -> None:
        if permit is not self._permit or not isinstance(permit, tuple):
            _refuse()
        expected = permit[1]
        row = self.load()
        rec = next((r for r in row["records"] if r["id"] == expected["id"]), None)
        if rec is None or rec != expected:
            _refuse("finite_journal_conflict")
        original = copy.deepcopy(row)
        rec["status"] = "complete"
        rec["revision"] = consent._revision(rec["revision"] + 1)
        try:
            self.save(row, row["revision"])
        except (MeshRefusal, OSError, ValueError):
            # An unconfirmed completion acknowledgement retains the negative
            # snapshot; compensation of a projection is NEVER itself authority.
            # Total inability to write BOTH stores is not claimed recoverable.
            latest = self.load()
            original["revision"] = consent._revision(latest["revision"] + 1)
            self._validate(original)
            self._publish(self.path, original)
            _refuse()
        self._permit = None


def check(network: str, owner: str, recipient: str, key: str, root: Path | None = None) -> None:
    # Snapshot read only: safe from positive producers holding placement locks.
    # Owner serving additionally holds consent->journal across its final check.
    guard = Journal(network, owner, root).peek(recipient, key)
    if guard:
        raise MeshRefusal(
            "finite_recovery_pending",
            "A previous finite-sharing transaction is "
            "unconfirmed. Bare sharing and future borrows remain refused; request a fresh exact "
            "finite-share approval to inspect and recover this retained owner-local fence.",
        )
