"""Idempotency keys for ``net_session_create``: one request, at most one session.

WHY THIS EXISTS. A create on a peer is NOT idempotent by construction: the owner
mints the id (``relay._op_session_create`` refuses a frame that names one, so two
devices can never claim one id), which means the requester cannot say "this is the
session I asked for a moment ago". When the create's REPLY is lost — the link drops,
or the slow lane answers after the requester's hop bound — the requester used to
report ``peer_unreachable: <peer> stopped answering before it replied`` while the
owner HAD created the session (observed live 2026-11-11: 59b1f39ac5ee on
lop-remote-node-2). A caller that believes that sentence retries and gets a SECOND
session running the same first prompt.

THE FIX IS A REQUESTER-MINTED KEY, NOT A REQUESTER-MINTED ID. The requester puts an
opaque ``create_key`` on the create frame; the owner records ``key -> session id`` as
soon as the directory exists. A replay of the same key returns the existing id
instead of minting another, and a separate read-only op
(``net_session_create_lookup``) lets the requester ask, after a lost reply, whether
its key produced a session. The id is still minted by the owner, so the §2 routing
rule is untouched.

CONSTRAINTS.

* Keys are NAMESPACED BY THE REQUESTING DEVICE. A member can only replay or look up
  keys it sent itself, so a key is never an oracle onto another member's sessions.
* ``pending`` is in memory only: it covers the window between the reservation and
  the mkdir inside ONE relay process. A relay that restarts mid-create left either no
  directory (nothing to find; the reservation is gone with the process) or a
  directory that was bound before the crash.
* Bound entries persist under ``<root>/network/create-keys.json`` (0600) so a replay
  that arrives after an owner relay restart still dedups. The file is bounded by
  count and age: a key exists to bridge a retry, which happens in seconds or
  minutes, never days.
* Old peers ignore the field (unknown frame keys are dropped), so the requester
  must never treat "no lookup answer" as "not created" — see
  ``RelayServer._ctl_peer_create``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

#: Where bound keys live, relative to the config root.
INDEX_FILENAME = "create-keys.json"

#: A key is opaque to the owner but bounded in shape, because it is persisted and
#: echoed back in sentences: hex/uuid-ish tokens only, never free text.
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

#: How many bound keys the owner keeps. A retry window is seconds to minutes; this
#: cap only stops a chatty requester from growing the file without bound.
MAX_ENTRIES = 512

#: How long a bound key answers a replay. A day comfortably covers "the operator
#: retried after lunch" while keeping the file small.
MAX_AGE_S = 24 * 3600.0

Reservation = Literal["new", "existing", "pending"]


def normalize_key(value: object) -> str:
    """The key if it is well-formed, else ``""`` (treated as "no key sent").

    A malformed key is ignored rather than refused: the key is a safety net the
    requester adds, and refusing a create over it would turn a dedup aid into a new
    way to fail.
    """
    text = str(value or "").strip() if isinstance(value, str) else ""
    return text if _KEY_PATTERN.match(text) else ""


def _stamped_at(row: dict[str, object]) -> float:
    """A bound row's write time; a damaged value sorts oldest (pruned first)."""
    value = row.get("at")
    return float(value) if isinstance(value, (int, float)) else 0.0


class CreateKeyIndex:
    """``(requesting device, create_key) -> session id`` on the OWNER device."""

    def __init__(self, root: Path | str) -> None:
        self._path = Path(root) / "network" / INDEX_FILENAME
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._bound: dict[str, dict[str, object]] | None = None

    @staticmethod
    def _slot(device_id: str, key: str) -> str:
        return f"{device_id}:{key}"

    # -- persistence (the caller holds ``_lock``) --------------------------------

    def _load(self) -> dict[str, dict[str, object]]:
        if self._bound is not None:
            return self._bound
        bound: dict[str, dict[str, object]] = {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("keys"), dict):
                for slot, row in data["keys"].items():
                    if isinstance(row, dict) and isinstance(row.get("session_id"), str):
                        bound[str(slot)] = row
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            # A damaged index costs dedup for keys already sent, never a create.
            logger.warning("create-key index unreadable at %s; starting empty", self._path)
        self._bound = bound
        return bound

    def _prune(self, bound: dict[str, dict[str, object]], now: float) -> None:
        stale = [slot for slot, row in bound.items() if now - _stamped_at(row) > MAX_AGE_S]
        for slot in stale:
            bound.pop(slot, None)
        if len(bound) > MAX_ENTRIES:
            oldest = sorted(bound, key=lambda slot: _stamped_at(bound[slot]))
            for slot in oldest[: len(bound) - MAX_ENTRIES]:
                bound.pop(slot, None)

    def _save(self, bound: dict[str, dict[str, object]]) -> None:
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".create-keys.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "keys": bound}, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- the three acts ------------------------------------------------------------

    def reserve(self, device_id: str, key: str) -> tuple[Reservation, str]:
        """Claim ``key`` for a new create, or report what it already produced.

        ``("new", "")`` — the caller must later ``bind`` or ``release``.
        ``("existing", sid)`` — a replay: answer with ``sid``, create nothing.
        ``("pending", "")`` — the first create with this key is still before its
        mkdir; the caller must create nothing.
        """
        slot = self._slot(device_id, key)
        with self._lock:
            if slot in self._pending:
                return "pending", ""
            row = self._load().get(slot)
            if row is not None:
                return "existing", str(row.get("session_id") or "")
            self._pending.add(slot)
            return "new", ""

    def bind(self, device_id: str, key: str, session_id: str) -> None:
        """Record that ``key`` produced ``session_id`` (the directory now exists)."""
        slot = self._slot(device_id, key)
        now = time.time()
        with self._lock:
            self._pending.discard(slot)
            bound = self._load()
            bound[slot] = {"session_id": session_id, "at": now}
            self._prune(bound, now)
            try:
                self._save(bound)
            except OSError:
                # The in-memory entry still dedups replays to this relay process;
                # only a restart would forget it. Never fail the create for this.
                logger.warning("create-key index not persisted at %s", self._path, exc_info=True)

    def release(self, device_id: str, key: str) -> None:
        """Drop a reservation whose create was refused before anything was written."""
        with self._lock:
            self._pending.discard(self._slot(device_id, key))

    def lookup(
        self, device_id: str, key: str
    ) -> tuple[Literal["created", "pending", "unknown"], str]:
        """What ``key`` produced, for the requester that lost the create's reply."""
        slot = self._slot(device_id, key)
        with self._lock:
            if slot in self._pending:
                return "pending", ""
            row = self._load().get(slot)
            if row is not None:
                return "created", str(row.get("session_id") or "")
            return "unknown", ""
