"""Recipient-local DENY watermarks, never positive permission or grant material.

Diagnostic PlacementState snapshots may be stale, cleared or unreadable. They
must not erase a newer security decision. This separate bounded document uses
ConsentDocument's thread+process lock, locked reread and durable readback pattern.
A cache stamp is meaningful only alongside an owner-issued, still-valid lease.
"""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from local_operator.network import store
from local_operator.network.credentials import placement

LIMIT = 262144
MAX_COUNTER = 2**63 - 1
NONRESTRICTIVE = frozenset(
    {
        "",
        "active",
        "owner_offline",
        "owner_unreachable",
        "internal",
        "rate_limited",
        "quota_blocked",
        "unsupported",
    }
)


class PolicyUnknown(RuntimeError):
    """Missing established, corrupt or unprovable evidence is never permission."""


@dataclass(frozen=True)
class Stamp:
    incarnation: str
    network_generation: int
    owner: str
    key: str
    key_generation: int


class HeldPolicyStore:
    def __init__(self, network_id: str, recipient: str, root: Path | None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", network_id) or not recipient:
            raise PolicyUnknown("invalid held-policy namespace")
        self.network_id = network_id
        self.recipient = recipient
        from local_operator.paths import config_dir

        self.root = (root or config_dir()).resolve()
        self.path = placement.credentials_dir_for(network_id, self.root) / "held-policy.json"
        self.marker = self.path.with_name("held-policy.initialized")

    def absent(self) -> bool:
        """Genuine first use: neither the document nor its marker exists.

        Deliberately narrower than "not established". A marker without its
        document, a document without its marker, unreadable bytes, a bad binding
        or a duplicate row all stay :class:`PolicyUnknown`: those are evidence
        this recipient HAD authority and can no longer prove it. Total absence is
        different in kind — nothing has ever been recorded here, so a reader that
        only needs deny knowledge has nothing to fence against and no deny to
        discover, and must not answer "unknown authority".

        Read outside the store's own locks on purpose: the caller uses it to
        decide whether to open a transaction at all. A race can only make this
        answer stale, and the loser of that race takes the locked path, which
        re-reads both files and stays authoritative.
        """
        return not os.path.lexists(self.path) and not os.path.lexists(self.marker)

    def _read(self, path: Path) -> dict[str, Any]:
        before = path.lstat()
        placement._checked_info(before, directory=False)
        # Match placement's checked-path fallback on Windows; chmod is not a
        # claim about ACLs there. POSIX prefers no-follow descriptor validation.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            placement._checked_info(info, directory=False)
            if (
                not placement._same_node(before, info)
                or not placement._same_node(info, path.lstat())
                or (os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o600)
            ):
                raise PolicyUnknown("unprovable held-policy file")
            raw = stream.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise PolicyUnknown("held-policy bound exceeded")
        doc = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(doc, dict):
            raise PolicyUnknown("invalid held-policy document")
        return doc

    def _validate(self, doc: dict[str, Any]) -> None:
        if set(doc) != {
            "schema",
            "network_id",
            "recipient",
            "incarnation",
            "revision",
            "network_generation",
            "membership_denied",
            "entries",
        }:
            raise PolicyUnknown("invalid held-policy schema")
        if (
            type(doc["schema"]) is not int
            or doc["schema"] != 1
            or doc["network_id"] != self.network_id
            or doc["recipient"] != self.recipient
            or not isinstance(doc["incarnation"], str)
            or not re.fullmatch(r"[a-f0-9]{32}", doc["incarnation"])
            or type(doc["membership_denied"]) is not bool
        ):
            raise PolicyUnknown("invalid held-policy binding")
        for field in ("revision", "network_generation"):
            if type(doc[field]) is not int or not 0 <= doc[field] < MAX_COUNTER:
                raise PolicyUnknown("invalid held-policy counter")
        if not isinstance(doc["entries"], list) or len(doc["entries"]) > 512:
            raise PolicyUnknown("invalid held-policy entries")
        seen = set()
        for row in doc["entries"]:
            if not isinstance(row, dict) or set(row) != {"owner", "key", "generation", "code"}:
                raise PolicyUnknown("invalid held-policy row")
            for field in ("owner", "key", "code"):
                if not isinstance(row[field], str) or not 0 < len(row[field]) <= 256:
                    raise PolicyUnknown("invalid held-policy ref")
            if type(row["generation"]) is not int or not 0 < row["generation"] < MAX_COUNTER:
                raise PolicyUnknown("invalid held-policy generation")
            pair = row["owner"], row["key"]
            if pair in seen:
                raise PolicyUnknown("duplicate held-policy identity")
            seen.add(pair)

    def _publish(self, path: Path, doc: dict[str, Any]) -> None:
        store._write_private_json(path, doc)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if self._read(path) != doc:
            raise PolicyUnknown("held-policy write unconfirmed")

    @contextmanager
    def transaction(
        self, *, initialize: bool | Callable[[], bool] = False
    ) -> Iterator[dict[str, Any]]:
        try:
            with (
                store._write_lock(self.path),
                placement._cross_process_lock(
                    self.path.with_name(".held-policy.lock"), root=self.root
                ),
            ):
                exists, established = os.path.lexists(self.path), os.path.lexists(self.marker)
                if (
                    not exists
                    and not established
                    and (initialize() if callable(initialize) else initialize)
                ):
                    doc = {
                        "schema": 1,
                        "network_id": self.network_id,
                        "recipient": self.recipient,
                        "incarnation": uuid.uuid4().hex,
                        "revision": 0,
                        "network_generation": 0,
                        "membership_denied": False,
                        "entries": [],
                    }
                    # Genesis migrates known legacy restrictions, not its positive
                    # status. An unreadable legacy file cannot establish clean state.
                    legacy = self.path.with_name("placement.state.json")
                    if legacy.exists():
                        payload = self._read(legacy)
                        if (
                            type(payload.get("schema")) is not int
                            or payload["schema"] != 1
                            or payload.get("network_id") != self.network_id
                            or not isinstance(payload.get("observations"), list)
                        ):
                            raise PolicyUnknown("invalid legacy observation evidence")
                        for row in payload["observations"]:
                            code = row.get("held_fence_code") or row.get("status", "")
                            if code not in NONRESTRICTIVE:
                                owner = row.get("held_fence_owner") or row.get("owner_device")
                                if not owner:
                                    entry = placement.PlacementDocument.load(
                                        self.network_id, self.root, self_device=self.recipient
                                    ).entry(row["key"])
                                    owner = entry.owner_device if entry else ""
                                self.deny(doc, owner, row["key"], code)
                    self._validate(doc)
                    self._publish(self.path, doc)
                    self._publish(
                        self.marker,
                        {
                            "schema": 1,
                            "network_id": self.network_id,
                            "recipient": self.recipient,
                            "incarnation": doc["incarnation"],
                        },
                    )
                elif not exists or not established:
                    raise PolicyUnknown("missing established held-policy evidence")
                doc = self._read(self.path)
                marker = self._read(self.marker)
                self._validate(doc)
                if doc["revision"] >= MAX_COUNTER - 1:
                    raise PolicyUnknown("held-policy revision exhausted")
                if type(marker.get("schema")) is not int:
                    raise PolicyUnknown("invalid held-policy marker")
                if marker != {
                    "schema": 1,
                    "network_id": self.network_id,
                    "recipient": self.recipient,
                    "incarnation": doc["incarnation"],
                }:
                    raise PolicyUnknown("held-policy incarnation mismatch")
                before = copy.deepcopy(doc)
                yield doc
                if doc != before:
                    doc["revision"] += 1
                    self._validate(doc)
                    self._publish(self.path, doc)
        except PolicyUnknown:
            raise
        except Exception as exc:
            raise PolicyUnknown("held-policy evidence unavailable") from exc

    def stamp(self, doc: dict[str, Any], owner: str, key: str) -> Stamp:
        row = next((x for x in doc["entries"] if (x["owner"], x["key"]) == (owner, key)), None)
        generation = row["generation"] if row else 0
        if max(doc["network_generation"], generation) >= MAX_COUNTER - 1:
            raise PolicyUnknown("held-policy generation exhausted")
        return Stamp(doc["incarnation"], doc["network_generation"], owner, key, generation)

    def code(self, doc: dict[str, Any], owner: str, key: str) -> str:
        row = next((x for x in doc["entries"] if (x["owner"], x["key"]) == (owner, key)), None)
        return row["code"] if row else ""

    def deny(self, doc: dict[str, Any], owner: str, key: str, code: str) -> None:
        row = next((x for x in doc["entries"] if (x["owner"], x["key"]) == (owner, key)), None)
        if row is None:
            row = {"owner": owner, "key": key, "generation": 0, "code": code}
            doc["entries"].append(row)
        row["generation"] += 1
        row["code"] = code

    def deny_network(self, doc: dict[str, Any]) -> None:
        if not doc["membership_denied"]:
            doc["network_generation"] += 1
            doc["membership_denied"] = True
