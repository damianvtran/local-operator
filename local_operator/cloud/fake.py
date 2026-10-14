"""``FakeComputeProvider``: the same seam with no cloud behind it.

WHY IT EXISTS, and it is one reason with two uses. The OSWorld adapter's
``providers/fake.py`` is the highest-value test of its own seam because it drives
the CODE PATH the real provider drives rather than a mock standing beside it, and
the ``--mock`` flag Slice 0 proved needs a key-free end-to-end run through the
same verbs. So this class implements the Protocol as written — async, the same
five methods, the same records — and a caller that works against it works against
``FargateProvider``.

WHAT IT DELIBERATELY HAS NO OPINION ABOUT. Timing, cold starts, whether the task
did any work, and what the platform would have charged. It holds a phase per
reference and that is all. ``describe()`` reports one number — how many workloads
this fake has been asked to create — because ``submit`` is the ONLY method that
creates anything, so it is the only method whose call can change what
``describe()`` says. That is the fake's own proof that the seam's create verb is
singular, and ``tests/unit/cloud/test_provider_contract.py`` asserts it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from local_operator.cloud.provider import (
    CloudRefused,
    ComputeStatus,
    DrainOutcome,
    ProviderCapacity,
    ProviderPhase,
)
from local_operator.cloud.types import PoolRequest, Tier

#: The exit code a container reports when the platform kills it for overstaying
#: its stop timeout — 128 + SIGKILL. A drain that hit its deadline ends this way,
#: and reporting 0 there would read as success.
_SIGKILL = 137


@dataclass
class _FakeTask:
    """One workload the fake is pretending to run."""

    ref: str
    request_id: str
    phase: ProviderPhase = "pending"
    exit_code: int | None = None
    stop_reason: str = ""
    created_at: float = 0.0
    started_at: float | None = None
    stopped_at: float | None = None
    detail: dict[str, str] = field(default_factory=dict)


class FakeComputeProvider:
    """The in-memory provider: ``name == "fake"``."""

    name = "fake"

    def __init__(self, *, tier: Tier = "fargate", checkpoint: bool = True) -> None:
        #: The tier this fake claims, so a caller that branches on it can be
        #: tested on both arms without a real backend.
        self.tier: Tier = tier
        #: Whether ``drain`` confirms a checkpoint or hits its deadline. Both are
        #: real provider outcomes and the ledger must handle each, so the fake
        #: makes the choice a constructor argument rather than a test mock.
        self.checkpoint = checkpoint
        self._tasks: dict[str, _FakeTask] = {}
        self._submitted = 0

    # -- the Protocol ------------------------------------------------------

    async def describe(self) -> ProviderCapacity:
        """What this fake can accept, and how much it has been asked for."""
        return ProviderCapacity(
            provider=self.name,
            tier=self.tier,
            ready=True,
            detail={"cluster": "fake", "submitted": str(self._submitted)},
        )

    async def submit(self, request: PoolRequest) -> str:
        """Create the workload and return its reference. The only creator here."""
        self._submitted += 1
        ref = f"fake-{self._submitted:04d}"
        self._tasks[ref] = _FakeTask(
            ref=ref,
            request_id=request.request_id or ref,
            created_at=time.time(),
            detail={"size": request.size, "budget_usd": f"{request.budget_usd:g}"},
        )
        return ref

    async def status(self, provider_ref: str) -> ComputeStatus:
        """One reading. Never mutates, and refuses a reference it does not know."""
        task = self._known(provider_ref)
        return ComputeStatus(
            provider=self.name,
            provider_ref=provider_ref,
            phase=task.phase,
            exit_code=task.exit_code,
            stop_reason=task.stop_reason,
            started_at=task.started_at,
            stopped_at=task.stopped_at,
            detail=dict(task.detail),
        )

    async def cancel(self, provider_ref: str, *, reason: str) -> None:
        """Stop it now. Idempotent: cancelling a stopped task changes nothing."""
        task = self._known(provider_ref)
        if task.phase in ("stopped", "absent"):
            return
        self._stop(task, reason=reason, exit_code=None)

    async def drain(self, provider_ref: str, *, deadline_s: float) -> DrainOutcome:
        """Ask it to stop, wait up to ``deadline_s`` for a checkpoint, then stop it.

        The two arms are the two arms a real drain has (§4.2): the container
        flushed what it had (``checkpointed``, ``drained=True``) or the deadline
        passed and the platform killed it (``deadline``, ``drained=False``). The
        second one is why ``DrainOutcome`` carries a reason at all.
        """
        task = self._known(provider_ref)
        if task.phase in ("stopped", "absent"):
            return DrainOutcome(
                drained=False, waited_s=0.0, reason="already_stopped", checkpoint={}
            )
        if not self.checkpoint:
            self._stop(task, reason="drained_unconfirmed", exit_code=_SIGKILL)
            return DrainOutcome(
                drained=False, waited_s=deadline_s, reason="deadline", checkpoint={}
            )
        self._stop(task, reason="drained", exit_code=0)
        return DrainOutcome(
            drained=True,
            waited_s=0.0,
            reason="checkpointed",
            checkpoint={
                "branch": f"lop/{task.request_id}",
                "commit": f"fake-commit-{task.ref}",
                "session_id": f"fake-session-{task.request_id}",
                "artifacts": "",
            },
        )

    # -- the test/`--mock` knob --------------------------------------------

    def mark(
        self,
        provider_ref: str,
        phase: ProviderPhase,
        *,
        exit_code: int | None = None,
        stop_reason: str = "",
    ) -> None:
        """Drive this fake's own reading.

        NOT part of the Protocol: the seam has no "make it running" call, because
        on a real provider that is the platform's business and not the client's.
        It exists so a test (and ``--mock``) can put a workload into the state it
        wants to reason about.
        """
        task = self._known(provider_ref)
        if phase == "running":
            task.phase = phase
            if task.started_at is None:
                task.started_at = time.time()
            return
        if phase in ("stopped", "absent"):
            self._stop(task, reason=stop_reason, exit_code=exit_code)
            return
        task.phase = phase

    # -- internals ---------------------------------------------------------

    def _known(self, provider_ref: str) -> _FakeTask:
        task = self._tasks.get(provider_ref)
        if task is None:
            raise CloudRefused(
                "provider_ref_unknown",
                f"this provider has no task {provider_ref!r}",
            )
        return task

    def _stop(self, task: _FakeTask, *, reason: str, exit_code: int | None) -> None:
        task.phase = "stopped"
        task.stopped_at = time.time()
        task.stop_reason = reason
        task.exit_code = exit_code
