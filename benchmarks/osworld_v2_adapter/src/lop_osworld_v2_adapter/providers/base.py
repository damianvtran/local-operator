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

# Transport margin between OUR socket deadline and the GUEST's own subprocess
# deadline, and the reason the two must be ordered rather than merely close.
#
# The guest's ``/execute`` route reads ``timeout`` from the request body and
# defaults it to 120 (osworld-server @ a3cc3f0, ``execute_command``). Sending no
# ``timeout`` therefore put the guest's deadline (120 s) OUTSIDE our socket read
# deadline (90 s) — two deadlines governing one command with nothing relating
# them. In that ordering a slow command always resolves the WRONG way: our
# socket expires first, ``requests`` raises a ReadTimeout, and the command is
# STILL RUNNING in the guest. That branch is unrecoverable by construction — a
# transport error cannot distinguish "never started" from "half-applied", so
# ``execute`` can only report an unknown outcome, and the no-retry policy that
# correctly refuses to replay a possibly-committed batch then ends the episode.
# Those ReadTimeouts are an observed episode-killer, not a hypothesis.
#
# Putting the guest's deadline strictly INSIDE ours inverts the race so it
# resolves the RIGHT way: the guest reaches its own deadline first and ANSWERS.
# ``execute_command`` catches the resulting ``TimeoutExpired`` and returns an
# HTTP 500 error body.
#
# Be precise about what that buys, because it is less than it first appears.
# ``http_post_json`` calls ``raise_for_status()``, so that 500 raises
# ``HTTPError``, and ``execute``'s blanket ``except Exception`` still reports
# "outcome is unknown" — the SAME branch a ``ReadTimeout`` produced, and the
# no-retry policy still ends the episode. What the ordering actually changes is
# WHEN and WHY the answer arrives: the guest gives up first and replies, so the
# failure lands promptly and is attributable to the command outrunning its
# budget, instead of our socket expiring while the command is still running and
# leaving a half-applied batch nobody can characterise. Turning that into a
# genuinely DEFINITE outcome needs the 500's timeout shape distinguished from a
# transport error before the blanket handler; this module does not do that yet,
# and claiming otherwise would misdescribe the code.
#
# The margin pays for the round trip that answer still has to make — request
# transmission, the guest's Flask dispatch, and the error response coming back —
# so it is sized for TRANSPORT, not for command work. 10 s against the same
# regression that measured 3.6 s of fixed per-command overhead (see
# ``GUEST_TYPE_DEADLINE_FRACTION``) is ~2.8x that figure, deliberately generous
# because the AMI is burstable and the margin only has to be big enough to carry
# a small error response home. Too small and a loaded guest's answer misses our
# socket anyway, losing the very determinism this ordering buys.
GUEST_TRANSPORT_MARGIN_S = 10.0

# Floor below which subtracting the full margin stops making sense, and the
# share of an already-tight socket deadline the guest gets instead. Both exist
# only for the squeezed tail: they keep the derived deadline positive AND
# strictly inside the socket deadline when the caller's remaining budget is
# smaller than the margin itself. The fraction is well under 1 so the ordering
# survives; its exact value matters little because a command with under a second
# of budget is failing regardless.
_MIN_GUEST_DEADLINE_S = 1.0
_SQUEEZED_DEADLINE_FRACTION = 0.5

# The strictly positive floor under EVERY derived guest deadline. A body
# ``timeout`` of 0 -- or a negative one -- is not "no deadline", it is
# "kill immediately", which is the exact outcome this whole module exists to
# prevent. The squeezed branch alone does not rule it out: a caller that has
# already exhausted its budget can reach here with ``socket_timeout_s`` at or
# below zero (``guest_disk`` samples its remaining budget at the guard and
# again at the call, so a value positive at the guard can be ~0 by the call),
# and a fraction of zero is still zero. Clamping keeps the guest's deadline
# positive so a doomed-either-way call still gets a real, if tiny, chance to
# run rather than a guaranteed instant kill.
_ABSOLUTE_FLOOR_S = 0.05


def guest_deadline_for(socket_timeout_s: float) -> float:
    """The guest's subprocess deadline for a command we wait ``socket_timeout_s`` on.

    The relationship, not a constant, because callers do not all wait the same
    time: ``guest_disk`` shrinks its socket deadline as its own budget drains,
    and the guest's deadline has to stay inside WHICHEVER deadline actually
    applies. Expressing it as a function is what stops a second literal being
    written next to the first.

    The result is ALWAYS strictly positive, and strictly less than
    ``socket_timeout_s`` for every socket deadline large enough for the ordering
    to mean anything (anything above ``_ABSOLUTE_FLOOR_S``). That ordering is the
    invariant the whole fix rests on — a guest deadline at or beyond ours
    restores the ambiguous ordering this function exists to prevent.

    Below that floor the two properties genuinely conflict: a caller whose budget
    is already spent cannot be given a deadline that is both inside its own and
    positive. Positivity wins, because a non-positive body ``timeout`` is an
    instant kill while a deadline marginally outside an already-exhausted socket
    budget merely returns the pre-existing behaviour.

    Subtracting the margin is the normal case. A caller whose socket deadline is
    already at or under the margin (``guest_disk`` shrinks its own toward zero as
    its budget drains) would derive a non-positive deadline, which the guest
    reads as "kill this immediately". Such a call is close to doomed either way,
    but it must not be converted into a guaranteed instant kill, so it falls back
    to a fraction of the socket deadline: still strictly inside, still positive,
    and it degrades smoothly instead of stepping off a cliff.
    """

    inner = socket_timeout_s - GUEST_TRANSPORT_MARGIN_S
    if inner < _MIN_GUEST_DEADLINE_S:
        squeezed = min(_MIN_GUEST_DEADLINE_S, socket_timeout_s * _SQUEEZED_DEADLINE_FRACTION)
        return max(squeezed, _ABSOLUTE_FLOOR_S)
    return inner


# The deadline sent with a command run at the DEFAULT socket timeout, named so
# the canonical pair is greppable and the ordering invariant is assertable.
# DERIVED, never written as a literal: the invariant this file exists to hold is
# ``GUEST_EXECUTE_TIMEOUT_S < GUEST_COMMAND_TIMEOUT_S``, and a hand-written
# second number is exactly how that invariant was lost the first time.
GUEST_EXECUTE_TIMEOUT_S = guest_deadline_for(GUEST_COMMAND_TIMEOUT_S)

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
