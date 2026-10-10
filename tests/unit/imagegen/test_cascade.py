"""The frozen walk: first credential wins, every failure fails forward.

Unit-level, with the rung executors faked at :func:`_run_route`: what is under
test is the RESOLVER's order and the EXECUTOR's contract (fail-forward,
recorded skips, the two stop conditions, the carried attempt list), not the
wire shapes — those live in ``test_rungs.py`` against ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.harness.types import AbortSignal
from local_operator.imagegen import ImageRoute, MediaAsset
from local_operator.imagegen import availability as image_availability
from local_operator.imagegen import cascade
from local_operator.imagegen import rungs as image_rungs


def _pin_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    radient: bool = False,
    fal: bool = False,
    openai: bool = False,
    openai_sub: bool = False,
    google: bool = False,
    xai: bool = False,
    openrouter: bool = False,
) -> None:
    async def fake_radient(config_dir, base_url, *, store):
        return radient

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", fake_radient)
    monkeypatch.setattr(
        image_availability, "fal_key", lambda config_dir=None: "fk" if fal else None
    )
    monkeypatch.setattr(
        image_availability, "openai_images_key", lambda config_dir=None: "ok" if openai else None
    )
    monkeypatch.setattr(
        image_availability,
        "openai_subscription_grant",
        lambda config_dir=None: openai_sub,
    )
    monkeypatch.setattr(
        image_availability, "google_key", lambda config_dir=None: "gk" if google else None
    )
    monkeypatch.setattr(image_availability, "xai_available", lambda config_dir=None: xai)
    monkeypatch.setattr(
        image_availability, "openrouter_key", lambda config_dir=None: "ork" if openrouter else None
    )


# ---------------------------------------------------------------------------
# resolve_image_route: order, reason, first match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_is_radient_then_fal_then_openai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True, openai=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.RADIENT
    assert resolution.reason == "Signed in to Radient."

    _pin_probes(monkeypatch, fal=True, openai=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.FAL

    _pin_probes(monkeypatch, openai=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.OPENAI


@pytest.mark.asyncio
async def test_the_key_rung_still_wins_over_the_subscription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Manager sign-off #1 (§14.5): v1 is KEY-FIRST — an existing user's path
    # must not change, and openai-sub runs only when earlier rungs refuse or
    # are absent. The order constant is the whole mechanism.
    _pin_probes(monkeypatch, openai=True, openai_sub=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.OPENAI

    _pin_probes(monkeypatch, openai_sub=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.OPENAI_SUB
    assert resolution.reason == "A ChatGPT subscription sign-in is stored."


@pytest.mark.asyncio
async def test_the_google_rung_appends_after_the_subscription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, openai_sub=True, google=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.OPENAI_SUB

    _pin_probes(monkeypatch, google=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.GOOGLE
    assert resolution.reason == "A Google AI Studio key is stored."


@pytest.mark.asyncio
async def test_the_xai_rung_appends_after_google(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, google=True, xai=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.GOOGLE

    _pin_probes(monkeypatch, xai=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.XAI
    assert resolution.reason == "An xAI key or sign-in is stored."


@pytest.mark.asyncio
async def test_the_openrouter_rung_is_the_last_resort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Append-only: everything earlier beats it, and it resolves alone.
    _pin_probes(monkeypatch, radient=True, openrouter=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.RADIENT

    _pin_probes(monkeypatch, xai=True, openrouter=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.XAI

    _pin_probes(monkeypatch, openrouter=True)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.OPENROUTER
    assert resolution.reason == "An OpenRouter key is stored."


@pytest.mark.asyncio
async def test_no_rung_names_every_remedy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _pin_probes(monkeypatch)
    resolution = await cascade.resolve_image_route(tmp_path)
    assert resolution.route == ImageRoute.NONE
    assert "/login radient" in resolution.reason
    assert "lop login fal" in resolution.reason
    assert "openai-key" in resolution.reason
    assert "sign in to a ChatGPT plan" in resolution.reason
    assert [rung.available for rung in resolution.rungs] == [False] * len(cascade.IMAGE_RUNG_ORDER)


# ---------------------------------------------------------------------------
# run_image_cascade: the walk
# ---------------------------------------------------------------------------


def _make_route_script(script: dict[ImageRoute, object]):
    """A ``_run_route`` fake: value = result, exception class = raise it."""
    calls: list[ImageRoute] = []

    async def fake(route, **kwargs):
        calls.append(route)
        outcome = script[route]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return fake, calls


def _result(model: str = "flux/dev", cost: float | None = None) -> image_rungs.RungResult:
    return image_rungs.RungResult(
        assets=[MediaAsset(data=b"png", content_type="image/png", source_url="https://x/1.png")],
        model=model,
        generation_id="req-1",
        cost_usd=cost,
    )


@pytest.mark.asyncio
async def test_first_available_rung_wins_and_later_ones_never_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    fake, calls = _make_route_script({ImageRoute.RADIENT: _result("default-model", cost=0.08)})
    monkeypatch.setattr(cascade, "_run_route", fake)

    outcome = await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [ImageRoute.RADIENT]
    assert outcome.route == ImageRoute.RADIENT
    assert [attempt.outcome for attempt in outcome.attempts] == ["ok"]
    assert outcome.model == "default-model"
    assert outcome.cost_usd == 0.08
    assert outcome.generation_id == "req-1"


@pytest.mark.asyncio
async def test_a_failed_rung_fails_forward(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    failure = APIError("out of credits", status_code=402)
    fake, calls = _make_route_script(
        {ImageRoute.RADIENT: failure, ImageRoute.FAL: _result("fal-ai/flux/dev")}
    )
    monkeypatch.setattr(cascade, "_run_route", fake)
    updates: list[tuple[str, dict[str, Any]]] = []

    outcome = await cascade.run_image_cascade(
        prompt="a cat",
        config_dir=tmp_path,
        emit=lambda text, details: updates.append((text, details)),
    )

    assert calls == [ImageRoute.RADIENT, ImageRoute.FAL]
    assert outcome.route == ImageRoute.FAL
    first = outcome.attempts[0]
    assert (first.route, first.outcome) == (ImageRoute.RADIENT, "failed")
    assert first.reason_class == "insufficient_credits"
    assert first.status_code == 402
    assert outcome.attempts[1].outcome == "ok"
    # The failure update rides the SAME classification as the attempt beside
    # it (Q7: "map sensibly alongside attempts[].reason_class; don't
    # duplicate/contradict") — and no canonical stage names a mid-walk
    # failure, so the pair carries the semantics alone.
    assert len(updates) == 1
    text, details = updates[0]
    assert text == "Generating via Radient: failed — out of credits"
    assert details["error"] == "out of credits"
    assert details["error_type"] == first.reason_class == "insufficient_credits"
    assert details["stage"] is None
    assert details["provider"] == "radient"


@pytest.mark.asyncio
async def test_a_skipped_rung_is_recorded_and_walked_past(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    skip = image_rungs.RungSkipped("balance cannot fund it", reason_class="insufficient_balance")
    fake, calls = _make_route_script({ImageRoute.RADIENT: skip, ImageRoute.FAL: _result()})
    monkeypatch.setattr(cascade, "_run_route", fake)

    outcome = await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [ImageRoute.RADIENT, ImageRoute.FAL]
    first = outcome.attempts[0]
    assert (first.outcome, first.reason_class) == ("skipped", "insufficient_balance")


@pytest.mark.asyncio
async def test_all_rungs_failed_carries_every_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True, openai=True)
    fake, calls = _make_route_script(
        {
            ImageRoute.RADIENT: APIError("r down", status_code=500),
            ImageRoute.FAL: APIError("f down", status_code=None),
            ImageRoute.OPENAI: APIError("o down", status_code=401),
        }
    )
    monkeypatch.setattr(cascade, "_run_route", fake)
    updates: list[tuple[str, dict[str, Any]]] = []

    with pytest.raises(cascade.ImageGenerationUnavailable) as caught:
        await cascade.run_image_cascade(
            prompt="a cat",
            config_dir=tmp_path,
            emit=lambda text, details: updates.append((text, details)),
        )

    exc = caught.value
    assert calls == [ImageRoute.RADIENT, ImageRoute.FAL, ImageRoute.OPENAI]
    assert [attempt.outcome for attempt in exc.attempts] == ["failed", "failed", "failed"]
    text = str(exc)
    assert "Radient: r down" in text
    assert "FAL: f down" in text
    assert "OpenAI: o down" in text
    # One failure update per failed rung, each carrying EXACTLY the attempt's
    # classification and message — the pair can never drift from the record.
    assert [(d["error_type"], d["error"]) for _, d in updates] == [
        (attempt.reason_class, attempt.message) for attempt in exc.attempts
    ]


@pytest.mark.asyncio
async def test_no_available_rungs_raises_the_remedy_not_an_empty_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch)
    with pytest.raises(cascade.ImageGenerationUnavailable) as caught:
        await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)
    assert "/login radient" in str(caught.value)
    assert caught.value.attempts == ()


@pytest.mark.asyncio
async def test_user_cancellation_stops_the_walk_without_failover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    fake, calls = _make_route_script(
        {ImageRoute.RADIENT: cascade.ImageGenerationCancelled(), ImageRoute.FAL: _result()}
    )
    monkeypatch.setattr(cascade, "_run_route", fake)

    with pytest.raises(cascade.ImageGenerationCancelled):
        await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)
    assert calls == [ImageRoute.RADIENT], "a stop is not a failure to fail over from"


@pytest.mark.asyncio
async def test_task_cancellation_propagates_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    fake, calls = _make_route_script(
        {ImageRoute.RADIENT: asyncio.CancelledError(), ImageRoute.FAL: _result()}
    )
    monkeypatch.setattr(cascade, "_run_route", fake)

    with pytest.raises(asyncio.CancelledError):
        await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)
    assert calls == [ImageRoute.RADIENT]


@pytest.mark.asyncio
async def test_the_walk_dispatches_the_subscription_rung(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, openai_sub=True)
    fake, calls = _make_route_script({ImageRoute.OPENAI_SUB: _result("gpt-image-2")})
    monkeypatch.setattr(cascade, "_run_route", fake)

    outcome = await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [ImageRoute.OPENAI_SUB]
    assert outcome.route == ImageRoute.OPENAI_SUB
    assert outcome.model == "gpt-image-2"


@pytest.mark.asyncio
async def test_the_walk_dispatches_the_xai_rung(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, xai=True)
    fake, calls = _make_route_script({ImageRoute.XAI: _result("grok-imagine-image-2.0")})
    monkeypatch.setattr(cascade, "_run_route", fake)

    outcome = await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [ImageRoute.XAI]
    assert outcome.route == ImageRoute.XAI


@pytest.mark.asyncio
async def test_the_walk_dispatches_the_openrouter_rung(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, openrouter=True)
    fake, calls = _make_route_script({ImageRoute.OPENROUTER: _result("seedream-4.5")})
    monkeypatch.setattr(cascade, "_run_route", fake)

    outcome = await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [ImageRoute.OPENROUTER]
    assert outcome.route == ImageRoute.OPENROUTER


@pytest.mark.asyncio
async def test_the_subscription_rung_refuses_a_platform_key_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Reviewer round 1 F1 / QA Q1: ``get_oauth_access`` rotates to a sibling
    # ``api_key`` row when a grant cannot mint a bearer — right for chat, a
    # leak here, because that key must never reach chatgpt.com. An api_key
    # credential therefore reads as unauthorized and fails forward, and the
    # transport (a recorder) must never see a request.
    from local_operator.providers.auth_store import AuthStore

    _pin_probes(monkeypatch, openai_sub=True)
    store = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    store.upsert_credential(
        "openai", {"type": "api_key", "source": "login", "key": "sk-synthetic-not-real"}
    )
    store.close()

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        sent.append(request)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(cascade.ImageGenerationUnavailable) as caught:
        await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path, client=client)
    await client.aclose()

    assert sent == [], "the platform key must never reach the chat backend"
    assert [(str(attempt.route), attempt.reason_class) for attempt in caught.value.attempts] == [
        ("openai-sub", "unauthorized")
    ]


@pytest.mark.asyncio
async def test_a_spent_budget_skips_the_remaining_rungs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin_probes(monkeypatch, radient=True, fal=True)
    fake, calls = _make_route_script({ImageRoute.RADIENT: _result(), ImageRoute.FAL: _result()})
    monkeypatch.setattr(cascade, "_run_route", fake)
    monkeypatch.setattr(cascade, "IMAGE_GENERATION_TIMEOUT_S", 0.0)

    with pytest.raises(cascade.ImageGenerationUnavailable) as caught:
        await cascade.run_image_cascade(prompt="a cat", config_dir=tmp_path)

    assert calls == [], "nothing runs once the deadline has passed"
    assert [attempt.outcome for attempt in caught.value.attempts] == ["skipped", "skipped"]


# ---------------------------------------------------------------------------
# _make_pause: the abort race helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_returns_on_timeout_and_raises_on_abort() -> None:
    signal = AbortSignal()
    pause = cascade._make_pause(signal)
    assert pause is not None

    await pause(0.01)  # timeout path: the wait elapsed, no abort

    signal.abort("stop")
    with pytest.raises(cascade.ImageGenerationCancelled):
        await pause(0.01)

    # The racing form: a long wait interrupted by an abort mid-flight — the
    # signal winning must raise OUR cancellation class promptly, never wait
    # out the 30 s.
    fresh_signal = AbortSignal()
    fresh = cascade._make_pause(fresh_signal)
    assert fresh is not None

    async def fire_soon() -> None:
        await asyncio.sleep(0.01)
        fresh_signal.abort("stop")

    racer = asyncio.ensure_future(fire_soon())
    with pytest.raises(cascade.ImageGenerationCancelled):
        await fresh(30.0)
    await racer
