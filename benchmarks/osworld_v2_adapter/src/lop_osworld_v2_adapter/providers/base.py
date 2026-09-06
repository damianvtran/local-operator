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

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor

# How long the client waits for ONE guest statement before giving up. Matches
# upstream's own client deadline (python.py:670) and is a hang detector, not a
# throughput budget: a statement that legitimately needs longer than this is a
# statement we should not be sending. It lives here, at the provider seam,
# rather than inline at its single call site because the action compiler must
# derive an admission bound from it (``actions.MAX_TYPE_CHARS``) — the two
# drifting apart is precisely the defect that cost two episodes.
GUEST_COMMAND_TIMEOUT_S = 90.0

# Budgeted cost of delivering ONE character through the guest's X11 synthetic
# key path, with pyautogui's default (zero) inter-key interval.
#
# Regression over 21 real batches measured 4.18 ms/char (R^2 = 0.998) on a
# healthy guest. This is budgeted at ~2x that because the measurement is a
# single sample of one instance type on one AMI, and that AMI is burstable —
# the same credit starvation that has already blinded this adapter's screenshot
# server (see ``providers.fake.blind_observations``) also slows key delivery.
# Budgeting the measured figure exactly would make the bound below a prediction
# of the median guest rather than a limit safe on a slow one.
GUEST_TYPE_MS_PER_CHAR = 8.0

# The largest share of the deadline a single type may be budgeted to consume.
# The remaining 40% is not slack for typing: it absorbs the fixed per-command
# overhead the same regression measured at 3.6 s (transport, the guest server's
# own dispatch, screenshot settle) plus the tail of a guest that is slower than
# the envelope above already assumes.
GUEST_TYPE_DEADLINE_FRACTION = 0.6


@runtime_checkable
class EnvironmentProvider(Protocol):
    """One environment backend. Implemented by FakeProvider and AwsProvider."""

    async def allocate(
        self, plan: ProvisioningPlan, task: TaskDescriptor, *, cache_root: Path
    ) -> None:
        """Create the environment. The side-effect boundary; reset_start only.

        ``cache_root`` is the ABSOLUTE, episode-scoped directory (outside the
        digest-pinned workspace) that any backend which downloads assets must
        write into. See ``adapter._episode_cache_root`` for why a cwd-relative
        cache wedges the rescue sweep.
        """
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
