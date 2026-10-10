"""The kind-neutral walk, its pause helper, and the video seam-proof.

These tests drive ``run_job_walk`` with a fake ``call`` — the same shape the
image suite drives the cascade with — so they pin the GENERIC contract the
image refactor reuses: first-wins order, fail-forward per class, recorded
skips, the budget-exhausted sentence, both stop conditions, and the
exhausted-walk factory. One test runs the walk for ``kind=VIDEO`` with no
video code anywhere, which is the seam's proof.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.artifacts import (
    ArtifactKind,
    BillingBasis,
    JobAttempt,
    JobCancelled,
    JobSpec,
    JobUnavailable,
    MediaAsset,
)
from local_operator.artifacts import walk as artifacts_walk
from local_operator.artifacts.errors import failure_reason_class
from local_operator.artifacts.rung import CancelHandle, RungResult, RungSkipped
from local_operator.clients._http import APIError
from local_operator.harness.types import AbortSignal

LABELS = {"alpha": "Alpha", "beta": "Beta"}


def _result(
    model: str = "flux/dev",
    cost: float | None = None,
    basis: BillingBasis | None = None,
    provenance: str | None = None,
) -> RungResult:
    return RungResult(
        assets=[MediaAsset(data=b"png", content_type="image/png", source_url="https://x/1.png")],
        model=model,
        generation_id="req-1",
        cost_usd=cost,
        cost_source="reported" if cost is not None else None,
        billing_basis=basis,
        cost_provenance=provenance,
    )


def _script(script: dict[str, object]):
    """A fake ``call``: value = result, exception instance = raise it."""
    calls: list[str] = []

    async def fake(route: str) -> RungResult:
        calls.append(route)
        outcome = script[route]
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, RungResult)
        return outcome

    return fake, calls


def _exhausted_factory(kind: ArtifactKind = ArtifactKind.IMAGE):
    captured: list[tuple[str, tuple[JobAttempt, ...]]] = []

    def on_exhausted(message: str, attempts: tuple[JobAttempt, ...]) -> BaseException:
        captured.append((message, attempts))
        return JobUnavailable(message, attempts=attempts)

    return on_exhausted, captured


async def _run(
    script: dict[str, object],
    *,
    kind: ArtifactKind = ArtifactKind.IMAGE,
    candidates: tuple[str, ...] = ("alpha", "beta"),
    rung_timeout_s: float = 240.0,
    overall_timeout_s: float = 300.0,
    emit=None,
):
    fake, calls = _script(script)
    handle = CancelHandle()
    on_exhausted, captured = _exhausted_factory(kind)
    outcome = await artifacts_walk.run_job_walk(
        kind=kind,
        spec=JobSpec(kind=kind, prompt="a cat", count=2, seed=5),
        candidates=candidates,
        labels=LABELS,
        call=fake,
        handle=handle,
        emit=emit,
        pause=None,
        tool="generate_image",
        rung_timeout_s=rung_timeout_s,
        overall_timeout_s=overall_timeout_s,
        on_exhausted=on_exhausted,
    )
    return outcome, calls, captured


# ---------------------------------------------------------------------------
# The walk matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_available_rung_wins_and_later_ones_never_run() -> None:
    outcome, calls, _ = await _run(
        {"alpha": _result("default-model", cost=0.08, basis="billed", provenance="doc 2026-10-09")}
    )

    assert calls == ["alpha"]
    assert outcome.route == "alpha"
    assert [attempt.outcome for attempt in outcome.attempts] == ["ok"]
    assert outcome.model == "default-model"
    assert outcome.cost_usd == 0.08
    assert outcome.cost_source == "reported"
    assert outcome.billing_basis == "billed"
    assert outcome.cost_provenance == "doc 2026-10-09"
    assert outcome.generation_id == "req-1"


@pytest.mark.asyncio
async def test_a_failed_rung_fails_forward_with_the_class_and_the_update() -> None:
    failure = APIError("out of credits", status_code=402)
    updates: list[tuple[str, dict[str, Any]]] = []
    outcome, calls, _ = await _run(
        {"alpha": failure, "beta": _result("beta-model")},
        emit=lambda text, details: updates.append((text, details)),
    )

    assert calls == ["alpha", "beta"]
    assert outcome.route == "beta"
    first = outcome.attempts[0]
    assert (first.route, first.outcome) == ("alpha", "failed")
    assert first.reason_class == "insufficient_credits"
    assert first.status_code == 402
    assert outcome.attempts[1].outcome == "ok"
    # The failure update rides the SAME classification as the attempt beside
    # it, carries the canonical key set, and names no stage.
    assert len(updates) == 1
    text, details = updates[0]
    assert text == "Generating via Alpha: failed — out of credits"
    assert details["tool_name"] == "generate_image"
    assert details["error"] == "out of credits"
    assert details["error_type"] == first.reason_class == "insufficient_credits"
    assert details["stage"] is None
    assert details["provider"] == "alpha"
    assert details["num_images"] == 2
    assert details["progress_fraction"] is None


@pytest.mark.asyncio
async def test_a_rung_timeout_records_the_exact_sentence_and_the_update() -> None:
    async def slow(route: str) -> RungResult:
        await asyncio.sleep(5)
        return _result()

    updates: list[tuple[str, dict[str, Any]]] = []
    handle = CancelHandle()
    on_exhausted, _ = _exhausted_factory()
    with pytest.raises(JobUnavailable) as caught:
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(kind=ArtifactKind.IMAGE, prompt="a cat"),
            candidates=("alpha",),
            labels=LABELS,
            call=slow,
            handle=handle,
            emit=lambda text, details: updates.append((text, details)),
            pause=None,
            tool="generate_image",
            rung_timeout_s=0.2,
            overall_timeout_s=300.0,
            on_exhausted=on_exhausted,
        )
    attempt = caught.value.attempts[0]
    assert (attempt.outcome, attempt.reason_class) == ("failed", "timeout")
    assert attempt.message == "Alpha exceeded its 0s generation budget."
    assert updates[0][0] == (
        "Generating via Alpha: failed — Alpha exceeded its 0s generation budget."
    )
    assert updates[0][1]["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_a_skipped_rung_is_recorded_and_walked_past_without_an_update() -> None:
    updates: list[tuple[str, dict[str, Any]]] = []
    skip = RungSkipped("balance cannot fund it", reason_class="insufficient_balance")
    outcome, calls, _ = await _run(
        {"alpha": skip, "beta": _result()},
        emit=lambda text, details: updates.append((text, details)),
    )

    assert calls == ["alpha", "beta"]
    first = outcome.attempts[0]
    assert (first.outcome, first.reason_class) == ("skipped", "insufficient_balance")
    assert updates == [], "a skip is not a refusal; no failure update fires"


@pytest.mark.asyncio
async def test_a_spent_overall_budget_skips_every_remaining_rung() -> None:
    fake, calls = _script({"alpha": _result(), "beta": _result()})

    on_exhausted, _ = _exhausted_factory()
    with pytest.raises(JobUnavailable) as caught:
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(kind=ArtifactKind.IMAGE, prompt="a cat"),
            candidates=("alpha", "beta"),
            labels=LABELS,
            call=fake,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            tool="generate_image",
            rung_timeout_s=240.0,
            overall_timeout_s=0.0,
            on_exhausted=on_exhausted,
        )

    exc = caught.value
    assert calls == [], "nothing runs once the deadline has passed"
    assert [attempt.outcome for attempt in exc.attempts] == ["skipped", "skipped"]
    assert all(
        attempt.message
        == "The overall generation budget was spent before this provider was reached."
        for attempt in exc.attempts
    )


@pytest.mark.asyncio
async def test_user_cancellation_stops_the_walk_without_failover() -> None:
    class _Stop(JobCancelled):
        pass

    fake, calls = _script({"alpha": _Stop(), "beta": _result()})
    on_exhausted, _ = _exhausted_factory()
    with pytest.raises(_Stop):
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(kind=ArtifactKind.IMAGE, prompt="a cat"),
            candidates=("alpha", "beta"),
            labels=LABELS,
            call=fake,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            tool="generate_image",
            rung_timeout_s=240.0,
            overall_timeout_s=300.0,
            on_exhausted=on_exhausted,
        )
    assert calls == ["alpha"], "a stop is not a failure to fail over from"


@pytest.mark.asyncio
async def test_task_cancellation_propagates_untouched() -> None:
    fake, calls = _script({"alpha": asyncio.CancelledError(), "beta": _result()})
    on_exhausted, _ = _exhausted_factory()
    with pytest.raises(asyncio.CancelledError):
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(kind=ArtifactKind.IMAGE, prompt="a cat"),
            candidates=("alpha", "beta"),
            labels=LABELS,
            call=fake,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            tool="generate_image",
            rung_timeout_s=240.0,
            overall_timeout_s=300.0,
            on_exhausted=on_exhausted,
        )
    assert calls == ["alpha"]


@pytest.mark.asyncio
async def test_all_rungs_failed_raises_the_factory_exception_with_every_attempt() -> None:
    fake, calls = _script(
        {
            "alpha": APIError("r down", status_code=500),
            "beta": APIError("f down", status_code=None),
        }
    )
    updates: list[tuple[str, dict[str, Any]]] = []
    on_exhausted, captured = _exhausted_factory()
    with pytest.raises(JobUnavailable) as caught:
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(kind=ArtifactKind.IMAGE, prompt="a cat"),
            candidates=("alpha", "beta"),
            labels=LABELS,
            call=fake,
            handle=CancelHandle(),
            emit=lambda text, details: updates.append((text, details)),
            pause=None,
            tool="generate_image",
            rung_timeout_s=240.0,
            overall_timeout_s=300.0,
            on_exhausted=on_exhausted,
        )

    exc = caught.value
    assert calls == ["alpha", "beta"]
    assert [attempt.outcome for attempt in exc.attempts] == ["failed", "failed"]
    text = str(exc)
    assert text.startswith("Image generation failed on every available provider:")
    assert "Alpha: r down" in text
    assert "Beta: f down" in text
    assert captured[0][0] == text
    assert captured[0][1] == exc.attempts
    assert [(d["error_type"], d["error"]) for _, d in updates] == [
        (attempt.reason_class, attempt.message) for attempt in exc.attempts
    ]


@pytest.mark.asyncio
async def test_an_unknown_exception_classifies_as_unknown_and_walks_on() -> None:
    outcome, calls, _ = await _run({"alpha": ValueError("boom"), "beta": _result()})

    assert calls == ["alpha", "beta"]
    first = outcome.attempts[0]
    assert (first.outcome, first.reason_class) == ("failed", "unknown")
    assert first.message == "boom"
    assert first.status_code is None


# ---------------------------------------------------------------------------
# The video seam: the walk is kind-neutral without any video code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_walk_serves_video_through_the_same_seam() -> None:
    outcome, calls, _ = await _run(
        {"alpha": _result()},
        kind=ArtifactKind.VIDEO,
        candidates=("alpha",),
    )

    assert calls == ["alpha"]
    assert outcome.kind == ArtifactKind.VIDEO
    assert outcome.route == "alpha"

    # And the exhausted sentence switches on the kind, not a pinned word.
    fake, _ = _script({"alpha": ValueError("down")})
    on_exhausted, captured = _exhausted_factory(ArtifactKind.VIDEO)
    with pytest.raises(JobUnavailable) as caught:
        await artifacts_walk.run_job_walk(
            kind=ArtifactKind.VIDEO,
            spec=JobSpec(kind=ArtifactKind.VIDEO, prompt="a clip"),
            candidates=("alpha",),
            labels=LABELS,
            call=fake,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            tool="generate_video",
            rung_timeout_s=240.0,
            overall_timeout_s=300.0,
            on_exhausted=on_exhausted,
        )
    assert str(caught.value).startswith("Video generation failed on every available provider:")


# ---------------------------------------------------------------------------
# make_pause: the abort race helper (mirror of the image suite's)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_returns_on_timeout_and_raises_on_abort() -> None:
    class _Cancelled(JobCancelled):
        pass

    signal = AbortSignal()
    pause = artifacts_walk.make_pause(signal, _Cancelled)
    assert pause is not None

    await pause(0.01)  # timeout path: the wait elapsed, no abort

    signal.abort("stop")
    with pytest.raises(_Cancelled):
        await pause(0.01)

    # The racing form: a long wait interrupted by an abort mid-flight — the
    # signal winning must raise the cancellation class promptly, never wait
    # out the 30 s.
    fresh_signal = AbortSignal()
    fresh = artifacts_walk.make_pause(fresh_signal, _Cancelled)
    assert fresh is not None

    async def fire_soon() -> None:
        await asyncio.sleep(0.01)
        fresh_signal.abort("stop")

    racer = asyncio.ensure_future(fire_soon())
    with pytest.raises(_Cancelled):
        await fresh(30.0)
    await racer


def test_pause_is_none_without_a_signal() -> None:
    class _Cancelled(JobCancelled):
        pass

    assert artifacts_walk.make_pause(None, _Cancelled) is None


def test_failure_reason_class_precedence() -> None:
    assert failure_reason_class(APIError("x", status_code=402, code="insufficient_balance")) == (
        "insufficient_balance"
    )
    assert failure_reason_class(APIError("x", status_code=402)) == "insufficient_credits"
    assert failure_reason_class(APIError("x", status_code=401)) == "unauthorized"
    assert failure_reason_class(APIError("x", status_code=403)) == "unauthorized"
    assert failure_reason_class(APIError("x", status_code=404)) == "not_found"
    assert failure_reason_class(APIError("x", status_code=429)) == "rate_limited"
    assert failure_reason_class(APIError("x", status_code=503)) == "upstream"
    assert failure_reason_class(APIError("x", status_code=None)) == "network"
    assert failure_reason_class(APIError("x", status_code=400)) == "refused"
    assert failure_reason_class(asyncio.TimeoutError()) == "timeout"
    assert failure_reason_class(ValueError("nope")) == "unknown"
