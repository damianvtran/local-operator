"""The provider protocol: what reset_start/cleanup/score need from a backend.

This is the seam between the adapter (which owns the contract with the
harness) and an environment backend (which owns the cloud or the fake).
``EnvironmentProvider`` is a structural Protocol, not a base class, so the AWS
implementation and the in-process fake are interchangeable
without the adapter knowing which it holds.

Every method is async because the AWS path is I/O-bound and the fake must
drive the same code path. ``allocate`` is the ONLY method that creates a
resource; it is called from ``reset_start``, never from ``prepare``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor


@runtime_checkable
class EnvironmentProvider(Protocol):
    """One environment backend. Implemented by FakeProvider and AwsProvider."""

    async def allocate(self, plan: ProvisioningPlan, task: TaskDescriptor) -> None:
        """Create the environment. The side-effect boundary; reset_start only."""
        ...

    async def observe(self) -> dict[str, Any]:
        """Return OSWorld's raw observation dict (screenshot/a11y/terminal/instruction)."""
        ...

    async def execute(self, statements: list[str]) -> None:
        """Run compiled guest statements, then settle. No observation here."""
        ...

    async def evaluate(self) -> Any:
        """Run OSWorld's evaluator; return its raw result (float or dict)."""
        ...

    async def terminate(self, instance_ref: str) -> str:
        """Terminate the instance named by the tag ref. Returns an evidence code."""
        ...

    async def delete_schedule(self, lease_ref: str) -> str:
        """Delete the TTL schedule named by the ref. Returns an evidence code."""
        ...

    async def describe(self, instance_ref: str) -> dict[str, Any] | None:
        """Resolve a tag ref to live instance state, or None if absent."""
        ...

    async def respond(self, prompt: str) -> str | None:
        """The task's user_simulator answer, or None when the task has none."""
        ...
