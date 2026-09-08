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
from local_operator.browser_bridge.protocol import ErrorCode

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
        # Whether the execution lease DEFINES this owner's identity. Only then
        # is a lease change a resume that must fence us; an owner whose identity
        # came from the record is authorized by the record plus, for the CLI
        # path, the lease it holds for exclusivity rather than for naming.
        self._identity_from_lease = False

    @property
    def execution_generation(self) -> str:
        """A pure identity lookup; stale owners are rejected INSIDE finalization.

        Never mints: an identity read must not become a write, or merely asking
        who owns the tab would rotate the answer.
        """
        if self.generation:
            return self.generation
        if self._lease_at_creation:
            return self._lease_at_creation
        try:
            return str(self._load().get("generation", ""))
        except (BrowserOwnershipError, OSError, ValueError):
            return ""

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
        if self._lease_at_creation and self._lease_generation() != self._lease_at_creation:
            raise BrowserOwnershipError("browser host execution lease changed; no action taken")
        # Identity is DURABLE PER SESSION, never per BrowserResource instance.
        #
        # Factory-built sessions inherit the exclusive execution lease's
        # generation, so a resume is a genuinely new execution and the old one
        # is correctly fenced. In-process children take neither path: they use
        # `claim_session`, not `acquire_session_lease`, so there is no lease to
        # read. Minting a per-instance token there made a SECOND Session over
        # the same directory revoke the FIRST one's authority over a tab it was
        # still holding — the incumbent's finalizer returned `unresolved` while
        # the newcomer inherited its surface_id, stranding a tab in the pool
        # that nobody could close. That is the exact leak this module exists to
        # remove, so the unleased case ADOPTS the stored identity instead: one
        # session id means one owner, whichever object is asking.
        self.generation = (
            self._lease_at_creation or self.previous_generation or secrets.token_urlsafe(24)
        )
        self._identity_from_lease = bool(self._lease_at_creation)
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
        if self._is_new_execution_over_settled_scope():
            self.record.pop("terminal", None)
            # The extension keeps its OWN copy of the terminal intent and used
            # to clear it only when the generation string changed — which the
            # unleased path deliberately never does, so retiring the record
            # alone moved the refusal one layer down instead of removing it.
            # DURABLE, and cleared only by an acknowledged recover: a run that
            # dies between here and the bridge would otherwise leave the
            # extension holding a terminal no later run could ever retire.
            self.record["resumed_scope"] = True
            if self.record.get("retention") == "paused scope":
                self.record["release_pause"] = True
        self._save()

    def _is_new_execution_over_settled_scope(self) -> bool:
        """May this owner retire the terminal intent the record already carries?

        The question a resume has to answer is "is this a NEW RUN of the
        session", and the honest evidence for that is the record itself, never
        a pid and never a lease.

        Keying this on holding a lease was wrong in both directions, and the
        wrong one shipped: in-process children use ``claim_session``, not
        ``acquire_session_lease``, so for every subagent the branch was dead
        and ``terminal`` became permanent. ``allocate`` refuses on terminal, so
        a finalized child could never browse again on any later run — and
        ``hub op='resume'`` relaunches children on their own directory, which
        made that an ordinary flow, not a corner. Worse, a scope that ended
        holding a tab for a pending approval could no longer be finished, so
        the tab was stranded: the leak class this module exists to remove.

        Two facts make the record sufficient. ``terminal`` is written only by
        ``finish``, so its presence means the previous scope SETTLED and has no
        live owner left to fence. And ``initialize`` runs only on a
        freshly-constructed resource — the instance that called ``finish``
        returns early from it — so reaching here over a settled record IS a
        later execution. That keeps B1 intact: a live incumbent has no terminal
        recorded, so a concurrent second instance still cannot retire anything.

        A STRANDED scope (the close failed, so the tab is still out there) is
        retired too, and that is deliberate rather than an oversight: the
        resumed owner is alive again and is the party responsible for that tab,
        which is precisely the route out that the crash-shape refusal names
        — resume the session and let the owner close it. It costs the operator
        no evidence, because ``cleanup_exact`` reaches a record through
        ``adopt``, never through this method, and a row only stops being a
        cleanup candidate while a live owner is actually holding it.
        """
        return bool(self.record.get("terminal"))

    def adopt(self, generation: str) -> None:
        """Take on an EXISTING record's identity without minting a new one.

        Operator recovery acts on the row the operator selected, so it must not
        renumber it: the CLI holds the lease for exclusivity, and the lease's
        own freshly-minted generation is not this owner's name.
        """
        self.record = self._load()
        if self.record.get("generation") != generation:
            raise BrowserOwnershipError("browser owner generation is stale; no action taken")
        self.generation = generation
        self.previous_generation = generation
        self._identity_from_lease = False

    def assert_current(self) -> None:
        current = self._load()
        lease = self._lease_generation() if self._identity_from_lease else ""
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
            # This owner is a NEW RUN over a scope that already settled. The
            # generation is reused (B1), so it is the only evidence the
            # extension has that a resume happened; without it the two sides
            # disagree about what a resume is and `open` is refused forever.
            "resumed_scope": bool(self.record.get("resumed_scope")),
        }

    async def recover(self) -> dict[str, Any]:
        try:
            result = await BridgeClient().call("owner_recover", self.params())
        except BridgeError as exc:
            # An ownership-AWARE extension refuses with the typed OWNER_REFUSED;
            # anything else from an `owner_*` method means the extension does
            # not implement ownership at all. Keying on the typed code rather
            # than on the old side's wording is what stops a reworded message
            # from silently degrading this into a raw internal error.
            if exc.code in (ErrorCode.INTERNAL, ErrorCode.PROTO_MISMATCH):
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
        # The extension has now retired its copy, so the obligation is
        # discharged. Held until the ACK rather than cleared when it was
        # written: an unacknowledged resume must be replayed, and leaving it
        # set would make a LATER finish's terminal clearable by a recover that
        # is no longer a resume at all.
        self.record.pop("resumed_scope", None)
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
            # The bridge's own message names the command that diagnoses the
            # failure; the class name is the fallback for a type that carries
            # no message, matching how the CLI renders its outer handler.
            return BrowserCleanupResult("pending", str(exc) or type(exc).__name__)


#: States in which a tab is stranded and an operator may recover it. Both are
#: reachable: ``finish`` writes ``cleanup_pending`` as its durable intent, then
#: OVERWRITES it with the bridge's own reply, so a failed ``chrome.tabs.remove``
#: persists ``pending``. Listing one and accepting the other made the single
#: genuinely-stranded state render as "not a candidate" while cleanup would have
#: taken it. One tuple, read by both, so they cannot drift apart again.
_RECOVERABLE_STATES = ("cleanup_pending", "pending")


def cleanup_disposition(value: dict[str, Any]) -> tuple[bool, str]:
    """``(eligible, reason)`` for one record — the ONE eligibility rule.

    The reason names what would make the row actionable, because "you copied
    the wrong generation" and "a human is finishing a login in that tab" are
    opposite situations and the operator cannot otherwise tell which they are
    in. Guessing wrong pushes them to retry against a protected tab, which is
    the behaviour the fence exists to discourage.
    """
    if value.get("retention"):
        return False, f"retained: {value['retention']} — the owning session must release it"
    if not value.get("terminal"):
        # The crash shape: killed before finish_browser_scope, so no terminal
        # intent exists and none may be inferred from the process being gone.
        # Refusing is correct, but refusing SILENTLY dead-ends the operator, so
        # name the route that actually works: the owner closes its own tab.
        return False, (
            "no terminal intent recorded (owner did not finish) — "
            "resume it with 'lop --resume <session>' and let it close the tab"
        )
    if value.get("state") not in _RECOVERABLE_STATES:
        return False, f"state is '{value.get('state', 'unresolved')}'; nothing is stranded"
    return True, ""


def read_inventory(directory: Path) -> list[dict[str, Any]]:
    """Redacted local evidence; unknown/live/retained never becomes eligible."""
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*/{RESOURCE_NAME}")):
        try:
            value = json.loads(path.read_text())
            eligible, reason = cleanup_disposition(value)
            rows.append(
                {
                    "session_id": path.parent.name,
                    "generation": value.get("generation", ""),
                    "state": value.get("state", "unresolved"),
                    "terminal": value.get("terminal", ""),
                    "retention": value.get("retention", ""),
                    "cleanup_candidate": eligible,
                    "blocked_reason": reason,
                }
            )
        except (OSError, ValueError, TypeError):
            rows.append(
                {
                    "session_id": path.parent.name,
                    "state": "unresolved",
                    "cleanup_candidate": False,
                    "blocked_reason": "record is unreadable",
                }
            )
    return rows


async def cleanup_exact(directory: Path, generation: str) -> BrowserCleanupResult:
    """Explicit operator recovery of ONE terminal generation, never a sweep.

    The execution lease rejects live/uncertain owners. A dead PID only permits
    acquiring that lease; matching durable terminal intent is still mandatory.
    """
    from local_operator.session_lease import acquire_session_lease

    value = BrowserResource(directory, directory.name)._load()
    if not value:
        return BrowserCleanupResult("unresolved", "no ownership record for that session")
    if value.get("generation") != generation:
        return BrowserCleanupResult(
            "unresolved",
            "generation does not match the current record — "
            "re-run 'lop browser tabs' and copy the current generation",
        )
    eligible, reason = cleanup_disposition(value)
    if not eligible:
        return BrowserCleanupResult("unresolved", reason)
    lease = acquire_session_lease(directory)
    try:
        # Constructed INSIDE the lease so it reads the generation the lease just
        # established. Building it outside made a CLI-path resource mint its own
        # token and persist it, rotating the very identifier the listing had
        # just told the operator to copy: their retry then failed as "stale"
        # through no fault of theirs, on every attempt.
        resource = BrowserResource(directory, directory.name)
        current = resource._load()
        if current != value:
            return BrowserCleanupResult("unresolved", "ownership changed; no action taken")
        resource.adopt(str(value["generation"]))
        return await asyncio.wait_for(
            resource.finish(resource.generation, str(value["terminal"])), 5.0
        )
    except TimeoutError:
        return BrowserCleanupResult("pending", "browser cleanup deadline exceeded")
    finally:
        lease.release()
