"""The kind-neutral job walk: submit each available rung, in order, fail forward.

Semantics (byte-for-byte what the image cascade's loop did before the
extraction — the refactor's oracle is the unedited image suite):

- per rung: ``remaining = deadline - now``; a spent deadline records a
  ``skipped``/``timeout`` attempt with the exact sentence below; otherwise the
  rung runs under ``min(rung_timeout_s, remaining)``.
- ``JobCancelled`` / ``asyncio.CancelledError`` are a STOP, re-raised with no
  failover — the user asked for the walk to end.
- ``TimeoutError`` records a failed attempt (``reason_class="timeout"``) and
  the walk continues.
- ``RungSkipped`` records a skipped attempt and continues (no failure update —
  a skip is not a refusal).
- a route named in ``pre_attempts`` is recorded as the supplied attempt and
  NEVER dispatched — the caller's own before-the-walk filter (the image
  lane's capability check); it is checked before the budget arm because a
  capability is not time-dependent.
- any other exception goes through :func:`~local_operator.artifacts.errors.
  failure_reason_class`, records a failed attempt, emits the mid-walk failure
  update, and continues.
- success returns a :class:`JobOutcome`; all failed raises
  ``on_exhausted(message, attempts)`` — the kind builds its own exception (the
  image lane raises ``ImageGenerationUnavailable`` carrying its resolution).

The walk itself holds no numbers and no provider names: budgets, labels and
the tool name are PARAMETERS, so the same walk serves every kind and transport
(poll, single request/response, SSE stream).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from local_operator.artifacts import (
    ArtifactKind,
    JobAttempt,
    JobCancelled,
    JobOutcome,
    JobSpec,
)
from local_operator.artifacts.errors import failure_reason_class
from local_operator.artifacts.progress import (
    ProgressFn,
    emit_progress,
    progress_details,
)
from local_operator.artifacts.rung import CancelHandle, RungResult, RungSkipped
from local_operator.clients._http import APIError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AbortSignal

__all__ = [
    "PauseFn",
    "all_failed_message",
    "attempt_message",
    "emit_rung_failure",
    "make_pause",
    "run_job_walk",
    "status_code_of",
]

#: How a rung lets the walk wait abortably: ``await pause(seconds)`` returns
#: early (raising the kind's cancellation class) when the user's abort signal
#: fires, so the rare no-cancellation race becomes a clean receipt instead of a
#: turn that waits out the provider. ``None`` in tests and library use.
PauseFn = Callable[[float], Awaitable[None]]


def make_pause(signal: "AbortSignal | None", cancelled_cls: type[JobCancelled]) -> PauseFn | None:
    """The abort-aware wait every rung's poll loop uses.

    ``None`` without a signal (library use, tests). With one: a zero-length
    wait is a plain abort check, and a real wait races ``signal.wait()``
    against the timeout — the signal winning raises ``cancelled_cls`` (the
    kind's cancellation class), our own task's cancellation propagates
    untouched (that is the loop's path, not ours).
    """
    if signal is None:
        return None

    async def pause(seconds: float) -> None:
        if signal.aborted:
            raise cancelled_cls()
        try:
            await asyncio.wait_for(signal.wait(), timeout=max(0.0, seconds))
        except TimeoutError:
            return
        # The wait returned because the signal fired (not because the timeout
        # elapsed) — the user asked the walk to stop.
        raise cancelled_cls()

    return pause


def attempt_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def status_code_of(exc: BaseException) -> int | None:
    return exc.status_code if isinstance(exc, APIError) else None


def all_failed_message(
    kind: ArtifactKind,
    attempts: Sequence[JobAttempt],
    labels: Mapping[str, str],
    *,
    action: str = "generation",
) -> str:
    """The exhausted-walk sentence: one header plus one line per attempt.

    ``kind.value.capitalize()`` is the display word ("image" → "Image"); the
    kind vocabulary is closed, so capitalize() cannot mangle a name.
    ``action`` is the kind's word for what failed — "generation" → "Image
    generation failed…"; the image lane's edit path passes ``"editing"`` —
    a parameter rather than a branch, so the generic layer stays blind to
    every lane's request shapes.
    """
    lines = [f"{kind.value.capitalize()} {action} failed on every available provider:"]
    for attempt in attempts:
        label = labels.get(attempt.route, str(attempt.route))
        if attempt.outcome == "skipped":
            lines.append(f"- {label}: skipped — {attempt.message}")
        else:
            lines.append(f"- {label}: {attempt.message}")
    return "\n".join(lines)


def emit_rung_failure(
    emit: ProgressFn | None,
    route: str,
    *,
    labels: Mapping[str, str],
    tool: str,
    message: str,
    error_type: str,
    num_images: int,
) -> None:
    """The mid-walk failure update: the pair the surfaces branch on.

    ``error_type`` is the SAME classification the attempt record beside it
    carries as ``reason_class`` — one classifier, so the update and the
    attempt cannot disagree (the Q7 wire scope: "map sensibly alongside
    ``attempts[].reason_class``; don't duplicate/contradict"). ``stage`` is
    ``None``: no canonical stage names a mid-walk failure — the pair is the
    semantics, and the next update (the next rung's ``queued``) replaces it.
    """
    label = labels.get(route, str(route))
    emit_progress(
        emit,
        f"Generating via {label}: failed — {message}",
        **progress_details(
            tool=tool,
            stage=None,
            provider=str(route),
            num_images=num_images,
            error=message,
            error_type=error_type,
        ),
    )


async def run_job_walk(
    *,
    kind: ArtifactKind,
    spec: JobSpec,
    candidates: Sequence[str],
    labels: Mapping[str, str],
    call: Callable[[str], Awaitable[RungResult]],
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    tool: str,
    rung_timeout_s: float,
    overall_timeout_s: float,
    on_exhausted: Callable[[str, tuple[JobAttempt, ...]], BaseException],
    pre_attempts: Mapping[str, JobAttempt] | None = None,
    action: str = "generation",
) -> JobOutcome:
    """Run the walk. See the module docstring for the failure contract.

    ``call`` is the kind's dispatch closure over its concrete params — it
    receives the route STRING and must resolve the credential, run the
    transport and fill ``handle`` itself. ``pause`` and ``handle`` are also
    what the kind's closure receives; the walk carries them for the interface
    (one submit/progress/cancel/steer/restart shape) without branching on
    them in v1.

    ``pre_attempts`` maps a route to the attempt record its caller already
    decided (e.g. the image lane's capability filter): the route is never
    dispatched and the supplied record is appended at the route's position in
    ``candidates``, so the attempt list stays in cascade order. Each
    pre-recorded route still needs a ``candidates`` entry — that list is the
    order. ``action`` is forwarded to :func:`all_failed_message`.
    """
    attempts: list[JobAttempt] = []
    deadline = time.monotonic() + overall_timeout_s
    pre = pre_attempts or {}
    for route in candidates:
        recorded = pre.get(route)
        if recorded is not None:
            # Pre-recorded before the walk (a capability the caller decided):
            # log it in place and never call. Before the budget arm on
            # purpose — see the module docstring.
            attempts.append(recorded)
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            attempts.append(
                JobAttempt(
                    route=route,
                    outcome="skipped",
                    reason_class="timeout",
                    message=(
                        "The overall generation budget was spent before this "
                        "provider was reached."
                    ),
                )
            )
            continue
        budget = min(rung_timeout_s, remaining)
        try:
            result = await asyncio.wait_for(call(route), timeout=budget)
        except (JobCancelled, asyncio.CancelledError):
            # User cancellation is a STOP, not a failure: no failover.
            raise
        except TimeoutError:
            timeout_message = (
                f"{labels.get(route, route)} exceeded its " f"{int(budget)}s generation budget."
            )
            attempts.append(
                JobAttempt(
                    route=route,
                    outcome="failed",
                    reason_class="timeout",
                    message=timeout_message,
                )
            )
            emit_rung_failure(
                emit,
                route,
                labels=labels,
                tool=tool,
                message=timeout_message,
                error_type="timeout",
                num_images=spec.count,
            )
            continue
        except RungSkipped as exc:
            attempts.append(
                JobAttempt(
                    route=route,
                    outcome="skipped",
                    reason_class=exc.reason_class,
                    message=str(exc),
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - every rung failure fails forward
            reason_class = failure_reason_class(exc)
            message = attempt_message(exc)
            attempts.append(
                JobAttempt(
                    route=route,
                    outcome="failed",
                    reason_class=reason_class,
                    message=message,
                    status_code=status_code_of(exc),
                )
            )
            emit_rung_failure(
                emit,
                route,
                labels=labels,
                tool=tool,
                message=message,
                error_type=reason_class,
                num_images=spec.count,
            )
            continue
        attempts.append(JobAttempt(route=route, outcome="ok"))
        return JobOutcome(
            kind=kind,
            assets=tuple(result.assets),
            route=route,
            attempts=tuple(attempts),
            model=result.model,
            prompt=spec.prompt,
            seed=spec.seed,
            generation_id=result.generation_id,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            billing_basis=result.billing_basis,
            cost_provenance=result.cost_provenance,
            usage_record_id=result.usage_record_id,
            strength_ignored=result.strength_ignored,
        )
    raise on_exhausted(all_failed_message(kind, attempts, labels, action=action), tuple(attempts))
