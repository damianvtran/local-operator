"""The device-local observation document: what THIS borrower last learned.

``<config>/network/credentials/<network_id>/placement.state.json``, 0600, NEVER
synced. The split from ``placement.json`` is the load-bearing half of the design's
ownership model (design §2.2): placement is FACTS, authored only by the owner of
each row; this file is OBSERVATIONS, authored only by this device. Merging the two
would let last-writer-wins clobber a fact with a rumour — a peer that happened to
be offline for a minute could otherwise publish "damian-mbp does not own openai"
into every other device's copy of the truth.

TWO THINGS THIS FILE DOES, and each is the reason it exists rather than living in
memory:

* **It is the refusal cache.** A borrow that came back ``not_a_holder`` or
  ``owner_offline`` is remembered with a TTL, so the next provider call in the same
  turn — and every provider call for the next minute — answers locally instead of
  asking a device that has already said no. Without it, one refused borrow becomes
  one request per provider call, which against a rate-limiting owner is a retry
  storm this device built.
* **It survives the process.** The borrower is a session runtime that exits; the
  observation belongs to the DEVICE, and re-asking on every launch is exactly the
  storm above, one restart slower.

WHAT IT MUST NEVER HOLD, and the structural reason it cannot: a token. There is no
field for one — the only credential-adjacent values are a grant's opaque
identifier and the time it was issued, which is why
``_FORBIDDEN_PLACEMENT_KEYS`` (``placement.py``) is applied to this document too:
a future field named ``access_token`` here is refused at the writer rather than
noticed at review.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from local_operator.network.credentials.placement import (
    PLACEMENT_SCHEMA,
    _assert_no_material,
    _config_root,
    _ensure_private_dir,
    placement_state_path,
)

#: The states an observation may carry. ``active`` is a POSITIVE observation (a
#: borrow succeeded), and it is stored rather than implied: it clears a stale
#: refusal on the next successful grant, which is how a peer that was offline and
#: came back stops being reported offline before its TTL would have expired.
OBSERVATION_STATES: frozenset[str] = frozenset(
    {
        "active",
        "owner_offline",
        "not_a_holder",
        "revoked",
        "grant_invalid",
        "quota_blocked",
        "no_local_credential",
        "interactive_required",
        "rate_limited",
        "epoch_stale",
        "refresh_failed",
        "unsupported",
        "device_bound",
        "not_authorised",
        "internal",
    }
)

#: The TTL applied to a state the caller stored without one, in milliseconds.
#: Short by default: an observation with no stated lifetime is a rumour, and the
#: cost of re-asking is one request while the cost of believing a stale refusal is
#: a credential the operator cannot use and cannot explain.
DEFAULT_OBSERVATION_TTL_MS = 30_000

FILE_MODE = 0o600


class PlacementState:
    """This device's observations about the credentials it may borrow."""

    def __init__(self, network_id: str, *, root: Path | None = None) -> None:
        self.network_id = network_id
        self.root = _config_root(root)
        self._observations: dict[str, dict[str, Any]] = {}
        #: ``(key, owner_device, credential_id) -> row`` — see :meth:`cool_down`.
        self._cooldowns: dict[tuple[str, str, int], dict[str, Any]] = {}

    # -- reads --------------------------------------------------------------

    @property
    def path(self) -> Path:
        return placement_state_path(self.network_id, self.root)

    def observation(self, key: str, *, now: float | None = None) -> dict[str, Any] | None:
        """The live observation for ``key``, or ``None`` when there is none or it expired."""
        row = self._observations.get(key)
        if row is None:
            return None
        stamp = float(row.get("observed_at") or 0.0)
        ttl_ms = int(row.get("retry_after_ms") or 0)
        if ttl_ms <= 0:
            return None
        current = now if now is not None else time.time()
        if (current - stamp) * 1000.0 >= ttl_ms:
            return None
        return row

    def status(self, key: str, *, now: float | None = None) -> str:
        """The live state for ``key``, or ``""`` when there is nothing to report."""
        row = self.observation(key, now=now)
        if row is None:
            return ""
        return str(row.get("status") or "")

    def active(self, key: str) -> bool:
        return self.status(key) == "active"

    def last_grant_at(self, key: str) -> float | None:
        row = self._observations.get(key)
        if row is None:
            return None
        stamp = row.get("last_grant_at")
        return float(stamp) if stamp else None

    def last_grant_from(self, device: str) -> float | None:
        """When ``device`` last served this device a grant, for any key, or ``None``.

        The owner-offline sentence's "last seen" reads this: a grant is proof the
        owner was up, and it is the only sighting a borrower records durably.
        """
        stamps = [
            float(row.get("last_grant_at") or 0.0)
            for row in self._observations.values()
            if row.get("owner_device") == device
        ]
        latest = max(stamps, default=0.0)
        return latest or None

    # -- writes -------------------------------------------------------------

    def observe(
        self,
        key: str,
        status: str,
        *,
        reason: str = "",
        owner_device: str = "",
        retry_after_ms: int = 0,
        owner_retry_after_ms: int = 0,
    ) -> None:
        """Record an observation, with the TTL that bounds how long it is believed.

        ``retry_after_ms`` BOUNDS THE CACHE; ``owner_retry_after_ms`` IS WHAT THE
        OWNER SAID, and they are not the same number (review round 2, M2). A
        refusal whose owner stated no remainder still caches for this device's own
        default — and a sentence that rendered that default back as "the owner's
        remainder" was claiming a fact about another device's cap that nobody
        stated. The owner's number is recorded separately, absent when none was
        sent, and is the only one a countdown may ever show.
        """
        row: dict[str, Any] = {
            "key": key,
            "owner_device": owner_device,
            "status": status,
            "reason": reason,
            "observed_at": time.time(),
            "retry_after_ms": int(retry_after_ms or DEFAULT_OBSERVATION_TTL_MS),
            "last_grant_id": self._observations.get(key, {}).get("last_grant_id", ""),
            "last_grant_at": self._observations.get(key, {}).get("last_grant_at", 0.0),
        }
        if owner_retry_after_ms:
            row["owner_retry_after_ms"] = int(owner_retry_after_ms)
        self._observations[key] = row

    def note_grant(self, key: str, grant_id: str, *, owner_device: str = "") -> None:
        """Record a successful borrow. Clears any refusal standing against ``key``.

        THE POSITIVE OBSERVATION IS THE POINT: the refusal cache is a cache, and a
        borrow that just succeeded is the freshest possible evidence that the key is
        usable. Leaving a ``not_a_holder`` row in place for the rest of its TTL after
        a successful grant would make the operator's own ``share`` appear not to have
        taken effect for a minute.
        """
        self._observations[key] = {
            "key": key,
            "owner_device": owner_device,
            "status": "active",
            "reason": "",
            "observed_at": time.time(),
            "retry_after_ms": DEFAULT_OBSERVATION_TTL_MS,
            "last_grant_id": grant_id,
            "last_grant_at": time.time(),
        }

    def clear(self, key: str) -> None:
        self._observations.pop(key, None)

    def forget_owner(self, device: str) -> None:
        """Drop every observation about credentials owned by ``device``.

        Used when a device leaves the network: its placement rows go with it, and a
        cached ``active`` for a key nobody owns any more is exactly the observation
        that would serve a stale sentence.
        """
        mine = [k for k, row in self._observations.items() if row.get("owner_device") == device]
        for key in mine:
            del self._observations[key]
        # Its cooldowns too (review round 1, M2): they name rows in THAT device's id
        # space, which mean nothing — or a different account — anywhere else.
        for slot in [s for s in self._cooldowns if s[1] == device]:
            del self._cooldowns[slot]

    # -- borrower-side cooldowns --------------------------------------------

    def cool_down(
        self,
        key: str,
        *,
        owner_device: str,
        credential_id: int,
        block_scope: str,
        until: float,
        label: str = "",
    ) -> None:
        """Remember that the owner's row ``credential_id`` answered a usage limit.

        WHY THE BORROWER KEEPS THIS AND THE OWNER DOES NOT: a local row that
        answers ``429 … retry in 24h`` is blocked for that window by
        ``AuthStore.rotate_sibling`` and the cascade moves to a sibling. A
        BORROWED row's verdict is reported to the owner instead, and the owner
        deliberately writes at most a 60 s, family-scoped block on a peer's word
        (``owner.REMOTE_QUOTA_BLOCK_MAX_MS``, F3: one report must never lock the
        owner out of its own login) — and nothing at all for a report it cannot
        scope. So the next grant re-served the same spent row and the turn ended
        on the same 429. This row is the borrower's OWN routing preference: it
        travels on the grant frame as ``exclude_credential_ids`` (hide these rows
        from THIS resolve only) and never touches the owner's row, so F3 holds.

        The window comes from ``auth_store.failure_block_window`` — the rule a
        local row follows — so the two cannot drift. ``label`` is the account's
        non-secret identity label (email or account id), kept so the refusal can
        name WHICH account is resting; it is never a token.
        """
        from local_operator.providers.auth_store import MAX_CREDENTIAL_BLOCK_MS

        now = time.time()
        if credential_id <= 0 or until <= now:
            return
        # RE-CAPPED ON EVERY WRITE, INCLUDING A LOAD (review round 1, m2): the file is
        # self-authored, but a backward clock step or a garbled row must not rest an
        # account past the one-hour ceiling the shared rule keeps on purpose.
        until = min(float(until), now + MAX_CREDENTIAL_BLOCK_MS / 1000.0)
        self._cooldowns[(key, owner_device, int(credential_id))] = {
            "key": key,
            "owner_device": owner_device,
            "credential_id": int(credential_id),
            "block_scope": block_scope,
            "until": float(until),
            "label": label,
        }

    def cooldowns(
        self,
        key: str,
        model_id: str = "",
        *,
        owner_device: str,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """The live cooldowns ``owner_device`` holds for ``key`` that gate ``model_id``.

        ``owner_device`` IS REQUIRED (review round 1, M2): row ids are integers in
        ONE owner's ``auth.db``, so after the key moves to another owner, id N there
        is a different account — a rest recorded against the old owner must never
        hide (or be named for) a row of the new one. Earliest-ending first.

        The scope test is ``auth_store.block_scope_gates`` — the predicate the
        local store's ``is_blocked_for_model`` applies — so an account-wide
        cooldown gates every model and a ``model:<family>`` one only that family.
        """
        from local_operator.providers.auth_store import block_scope_gates

        stamp = time.time() if now is None else now
        live = [
            dict(row)
            for (row_key, row_owner, _id), row in self._cooldowns.items()
            if row_key == key
            and row_owner == owner_device
            and float(row.get("until") or 0.0) > stamp
            and block_scope_gates(str(row.get("block_scope") or ""), model_id)
        ]
        return sorted(live, key=lambda row: float(row["until"]))

    def _prune_cooldowns(self, now: float | None = None) -> None:
        stamp = time.time() if now is None else now
        for slot in [s for s, row in self._cooldowns.items() if float(row["until"]) <= stamp]:
            del self._cooldowns[slot]

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": PLACEMENT_SCHEMA,
            "network_id": self.network_id,
            "observations": [row for _, row in sorted(self._observations.items())],
        }
        # Omitted when empty, so a device that never cooled a borrowed row writes
        # the file it always wrote; expired rows are dropped rather than carried.
        self._prune_cooldowns()
        if self._cooldowns:
            payload["cooldowns"] = [row for _, row in sorted(self._cooldowns.items())]
        # The same guard the placement document uses, and for the same reason: the
        # cheapest way to prove this file cannot hold a bearer is to refuse one at
        # the writer. See this module's docstring.
        _assert_no_material(payload, where="placement state")
        return payload

    def save(self) -> Path:
        from local_operator.network import store

        payload = self.to_json()
        _ensure_private_dir(self.path.parent, root=self.root)
        with store._write_lock(self.path):
            return store._write_private_json(self.path, payload)

    @classmethod
    def load(cls, network_id: str, root: Path | None = None) -> PlacementState:
        """Read the observations, or an empty set. Never raises for a missing file.

        An unreadable state file yields NO observations, which means the borrower
        asks rather than assumes. The failure mode of a corrupt cache must be a
        request, never a silently-served refusal.
        """
        state = cls(network_id, root=root)
        try:
            raw = state.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return state
        try:
            payload = json.loads(raw)
        except ValueError:
            return state
        if not isinstance(payload, dict):
            return state
        for row in payload.get("observations") or []:
            if isinstance(row, dict) and row.get("key"):
                state._observations[str(row["key"])] = dict(row)
        for row in payload.get("cooldowns") or []:
            # A malformed row is skipped, never guessed at: the failure mode of a
            # bad cooldown must be one more ask of the owner, not a hidden login.
            try:
                state.cool_down(
                    str(row["key"]),
                    owner_device=str(row.get("owner_device") or ""),
                    credential_id=int(row["credential_id"]),
                    block_scope=str(row.get("block_scope") or ""),
                    until=float(row["until"]),
                    label=str(row.get("label") or ""),
                )
            except (KeyError, TypeError, ValueError):
                continue
        return state

    @classmethod
    def for_network(cls, network_id: str, root: Path | None = None) -> PlacementState:
        return cls.load(network_id, root)


def state_file_is_private(path: Path) -> bool:
    """Whether an existing state file carries the mode this module promises.

    A reader for the doctor/CLI surface, and a test hook: a file written by an
    earlier build — or by hand — is worth being able to report on without opening it.
    """
    try:
        return (os.stat(path).st_mode & 0o777) == FILE_MODE
    except OSError:
        return False
