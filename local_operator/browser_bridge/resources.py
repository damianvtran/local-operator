"""Private session browser obligations, never a PID/age-based orphan registry.

The transcript's execution lease owns writes. A process disappearing proves
nothing about task completion: only finish() records terminal intent. Capabilities
stay in a mode-0600 sidecar, and public inventories deliberately omit them.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.browser_bridge.backend import BridgeClient, BridgeError

RESOURCE_NAME = ".browser-resource.json"


class BrowserOwnershipError(RuntimeError):
    """An expected ownership/compatibility refusal, safe to show without a stack."""


@dataclass(frozen=True)
class BrowserCleanupResult:
    state: str
    detail: str = ""


class BrowserResource:
    def __init__(self, directory: Path, session_id: str) -> None:
        self.directory = directory
        self.session_id = session_id
        self.path = directory / RESOURCE_NAME
        self.lock = asyncio.Lock()
        self.record: dict[str, Any] = {}
        self.generation = ""
        self.previous_generation = ""
        self.recovered = False
        self._lease_at_creation = self._lease_generation()
        self._host_generation = self._lease_at_creation or secrets.token_urlsafe(24)

    @property
    def execution_generation(self) -> str:
        """A pure identity lookup; stale owners are rejected INSIDE finalization."""
        return self.generation or self._host_generation

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(value, dict) or value.get("session_id") != self.session_id:
            raise BrowserOwnershipError("browser resource ownership is unresolved")
        return value

    def _lease_generation(self) -> str:
        from local_operator.session_lease import _read_claim

        generation, _pid = _read_claim(self.directory / ".execution-lease")
        return generation or ""

    def initialize(self) -> None:
        if self.generation:
            self.assert_current()
            return
        self.record = self._load()
        self.previous_generation = str(self.record.get("generation", ""))
        # Factory-created sessions use the existing exclusive execution lease.
        # In-process children have unique transcript directories and no lease;
        # their private generation still fences a stale finalizer after resume.
        if self._lease_at_creation and self._lease_generation() != self._lease_at_creation:
            raise BrowserOwnershipError("browser host execution lease changed; no action taken")
        self.generation = self._host_generation
        generations = list(self.record.get("bridge_generations", []))
        for candidate in (self.previous_generation, self.generation):
            if candidate and candidate not in generations:
                generations.append(candidate)
        self.record.update(
            bridge_generations=generations,
            session_id=self.session_id,
            generation=self.generation,
            proof=self.record.get("proof") or secrets.token_urlsafe(32),
            allocation_id=self.record.get("allocation_id") or secrets.token_urlsafe(24),
            state=self.record.get("state", "closed"),
        )
        if self.previous_generation and self.previous_generation != self.generation:
            # A resume is a new execution of the SAME conversation, authorized
            # by the lease. Preserve its tab/retention but retire old run intent.
            self.record.pop("terminal", None)
            if self.record.get("retention") == "paused scope":
                self.record["release_pause"] = True
        self._save()

    def assert_current(self) -> None:
        lease = self._lease_generation()
        current = self._load()
        if (lease and lease != self.generation) or current.get("generation") != self.generation:
            raise BrowserOwnershipError("browser owner generation is stale; no action taken")

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".browser-resource-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def params(self) -> dict[str, Any]:
        self.initialize()
        return {
            "requester": f"session:{self.session_id}",
            "owner_proof": str(self.record["proof"]),
            "owner_generation": self.generation,
            "previous_generation": self.previous_generation,
            "previous_generations": self.record["bridge_generations"],
            "allocation_id": str(self.record["allocation_id"]),
        }

    async def recover(self) -> dict[str, Any]:
        try:
            result = await BridgeClient().call("owner_recover", self.params())
        except BridgeError as exc:
            if "unknown method" in exc.message or "protocol" in exc.message.lower():
                raise BrowserOwnershipError(
                    "Browser ownership recovery requires an updated Local Operator extension. "
                    "Update the extension, reconnect it, then retry; no new tab was allocated."
                ) from exc
            raise
        if result.get("ownership_version") != 1:
            raise BrowserOwnershipError(
                "browser extension needs ownership-recovery support; update it first"
            )
        self.assert_current()
        if self.record.get("release_pause") and result.get("state") != "unresolved":
            await BridgeClient().call("owner_release", self.params())
            self.assert_current()
            self.record.pop("release_pause", None)
            self.record["retention"] = ""
        if result.get("state") == "unresolved" and self.record.get("surface_id"):
            # A full browser restart destroys session-storage authority. Keep
            # the old capability only as private evidence, never adopt by tab ID.
            self.record["unresolved_surface_id"] = self.record["surface_id"]
        self.record["surface_id"] = str(result.get("tab", ""))
        self.record["state"] = (
            "cleanup_pending"
            if self.record.get("terminal")
            else str(result.get("state", "unresolved"))
        )
        if result.get("state") != "unresolved":
            # Acknowledged CAS collapses the crash-replay chain. Before this
            # receipt every attempted generation remains a possible peer state.
            self.record["bridge_generations"] = [self.generation]
        self.recovered = True
        self._save()
        return result

    def remember(self, surface_id: str, *, state: str | None = None) -> None:
        self.assert_current()
        self.record["surface_id"] = surface_id
        self.record["state"] = (
            "cleanup_pending"
            if self.record.get("terminal")
            else state or ("owned" if surface_id else "closed")
        )
        self._save()

    def allocate(self) -> None:
        self.assert_current()
        if self.record.get("terminal"):
            raise BrowserOwnershipError("browser scope has ended; resume it before allocating")
        if self.record.get("state") in ("closed", "unresolved"):
            self.record["allocation_id"] = secrets.token_urlsafe(24)
        self.record["state"] = "allocating"
        self._save()

    async def finish(self, generation: str, outcome: str) -> BrowserCleanupResult:
        try:
            self.initialize()
        except (RuntimeError, OSError, ValueError):
            # A stale child still owes its terminal event. Refusal must be a
            # result, not an exception that bypasses the runner's publication.
            return BrowserCleanupResult(
                "unresolved", "owner changed or unreadable; no action taken"
            )
        if generation != self.generation:
            return BrowserCleanupResult("unresolved", "stale generation; no action taken")
        # Persist intent BEFORE waiting for a possibly in-flight tool. A crash
        # or timeout leaves an exact obligation, not guessed terminal evidence.
        self.assert_current()
        outcome = str(self.record.get("terminal") or outcome)
        self.record["terminal"] = outcome
        self.record["state"] = "cleanup_pending"
        self._save()
        try:
            async with self.lock:
                self.assert_current()
                if not self.recovered:
                    await self.recover()
                if self.record.get("unresolved_surface_id") and not self.record.get("surface_id"):
                    return BrowserCleanupResult(
                        "unresolved",
                        "browser ownership could not be proven after restart; no tab adopted",
                    )
                if self.record.get("retention"):
                    await BridgeClient().call(
                        "owner_retain", {**self.params(), "reason": self.record["retention"]}
                    )
                result = await BridgeClient().call(
                    "owner_finish", {**self.params(), "outcome": outcome}
                )
                self.assert_current()
                state = str(result.get("state", "unresolved"))
                self.record["state"] = state
                if state == "closed":
                    self.record["surface_id"] = ""
                    if self.record.get("unresolved_surface_id"):
                        state = self.record["state"] = "unresolved"
                self._save()
                return BrowserCleanupResult(state)
        except Exception as exc:
            return BrowserCleanupResult("pending", type(exc).__name__)


def read_inventory(directory: Path) -> list[dict[str, Any]]:
    """Redacted local evidence; unknown/live/retained never becomes eligible."""
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*/{RESOURCE_NAME}")):
        try:
            value = json.loads(path.read_text())
            rows.append(
                {
                    "session_id": path.parent.name,
                    "generation": value.get("generation", ""),
                    "state": value.get("state", "unresolved"),
                    "terminal": value.get("terminal", ""),
                    "retention": value.get("retention", ""),
                    "cleanup_candidate": bool(value.get("terminal"))
                    and not value.get("retention")
                    and value.get("state") == "cleanup_pending",
                }
            )
        except (OSError, ValueError, TypeError):
            rows.append(
                {"session_id": path.parent.name, "state": "unresolved", "cleanup_candidate": False}
            )
    return rows


async def cleanup_exact(directory: Path, generation: str) -> BrowserCleanupResult:
    """Explicit operator recovery of ONE terminal generation, never a sweep.

    The execution lease rejects live/uncertain owners. A dead PID only permits
    acquiring that lease; matching durable terminal intent is still mandatory.
    """
    from local_operator.session_lease import acquire_session_lease

    value = BrowserResource(directory, directory.name)._load()
    if value.get("generation") != generation or not value.get("terminal") or value.get("retention"):
        return BrowserCleanupResult("unresolved", "selection stale, nonterminal, or retained")
    lease = acquire_session_lease(directory)
    try:
        resource = BrowserResource(directory, directory.name)
        current = resource._load()
        if current != value:
            return BrowserCleanupResult("unresolved", "ownership changed; no action taken")
        resource.initialize()
        return await asyncio.wait_for(
            resource.finish(resource.generation, str(value["terminal"])), 5.0
        )
    except TimeoutError:
        return BrowserCleanupResult("pending", "browser cleanup deadline exceeded")
    finally:
        lease.release()
