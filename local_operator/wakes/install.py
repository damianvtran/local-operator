"""Install-on-demand for the wake supervisor — the chokepoint, not yet the
installer.

There is exactly one writer of schedule state, ``Session._persist_wake_schedules``,
so there is exactly one place to ask "does something outside this process
now need to exist to fire these?" That question is this function. It is
called after every non-empty persist, idempotently and best-effort: the
persist has already succeeded by the time it runs, and nothing it does (or
fails to do) may change that.

**This is a stub.** The supervisor it will install — the ~40 MB always-on
process that reads :mod:`local_operator.wakes.store` and engages a runtime
for a cold session whose wake comes due — does not exist yet. The hook lands
first so that the chokepoint is already wired through the persist path when
the supervisor arrives, and so the session-side contract (call shape, the
never-raises rule, where the outcome surfaces) is settled and tested before
the platform-specific installer is written. The fill-in is scheduled as
PR 7 of the detached-session series and is shaped after
``local_operator.mobile.install`` (LaunchAgent on macOS; ``KeepAlive:
{SuccessfulExit: False}`` so the supervisor can self-retire when the index
empties).

Until then: every call reports ``installed=False`` with a reason, and a
session that persists a schedule keeps firing it in-process exactly as it
does today. Nothing is lost that was not already lost — a wake in a closed
session did not fire before this stub either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallOutcome:
    """What the hook did. ``installed`` is "a supervisor is now in place"
    (freshly installed OR already present); ``reason`` explains a False."""

    installed: bool
    reason: str = ""


#: The single reason the stub ever reports. A constant so tests and the wake
#: tool's receipt can match on it rather than on prose.
NOT_AVAILABLE_REASON = "supervisor not yet available"


def ensure_supervisor_installed(config_dir: Path) -> InstallOutcome:
    """Make sure the wake supervisor is installed for ``config_dir``.

    Contract (binding on the real implementation, not just the stub):

    - **Idempotent.** Called after every persist; an already-installed
      supervisor is a cheap check, never a reinstall.
    - **Never raises.** The caller is the wake persist path; an installer
      failure is logged and reported through the outcome, never propagated.
      The persist has already succeeded and must stay succeeded.
    - **Best-effort.** A platform with no installer (Linux without a service
      manager the hook knows) reports ``installed=False`` and the session
      carries on firing its own wakes in-process.
    """
    del config_dir  # the real installer keys the LaunchAgent label off it
    return InstallOutcome(installed=False, reason=NOT_AVAILABLE_REASON)
