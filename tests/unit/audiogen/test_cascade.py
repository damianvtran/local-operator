"""The frozen walk: first credential wins, every failure fails forward.

Unit-level, with the rung executors faked at :func:`_run_route`: what is under
test is the RESOLVER's order, the EXECUTOR's contract (fail-forward, recorded
skips, the cancellation stop, the carried attempt list) and the direction
filter (sound effects never reach OpenAI) — not the wire shapes, which live in
``test_rungs.py`` against ``httpx.MockTransport``.
"""

from __future__ import annotations

from typing import Literal

import pytest

from local_operator.artifacts.rung import CancelHandle, RungResult
from local_operator.audiogen import AudioRoute, cascade
from local_operator.audiogen.models import AudioJobSpec
from local_operator.clients._http import APIError


def _pin_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    radient: bool = False,
    elevenlabs: bool = False,
    openai: bool = False,
) -> None:
    availability = cascade.audio_availability
    monkeypatch.setattr(availability, "radient_available", lambda config_dir=None: radient)
    monkeypatch.setattr(
        availability, "elevenlabs_key", lambda config_dir=None: "el" if elevenlabs else None
    )
    monkeypatch.setattr(
        availability, "openai_audio_key", lambda config_dir=None: "oa" if openai else None
    )


def _spec(direction: Literal["voiceover", "sound_effect"] = "voiceover") -> AudioJobSpec:
    return AudioJobSpec(
        tool="generate_voiceover" if direction == "voiceover" else "generate_sound_effect",
        direction=direction,
        text="hello" if direction == "voiceover" else "a clap",
        duration_s=None if direction == "voiceover" else 1.0,
    )


def _rung_ok(route: str) -> RungResult:
    from local_operator.audiogen.models import AudioAsset

    return RungResult(
        assets=[
            AudioAsset(
                data=b"ID3fake",
                content_type="audio/mpeg",
                source_url="",
            )
        ],
        model="m",
        generation_id="g",
    )


# ---------------------------------------------------------------------------
# resolve_audio_route: order, direction filter, reason, first match
# ---------------------------------------------------------------------------


def test_priority_is_radient_then_elevenlabs_then_openai(monkeypatch):
    _pin_probes(monkeypatch, radient=True, elevenlabs=True, openai=True)
    resolution = cascade.resolve_audio_route(direction="voiceover")
    assert resolution.route == AudioRoute.RADIENT
    assert resolution.reason == "Signed in to Radient."

    _pin_probes(monkeypatch, elevenlabs=True, openai=True)
    assert cascade.resolve_audio_route(direction="voiceover").route == AudioRoute.ELEVENLABS

    _pin_probes(monkeypatch, openai=True)
    assert cascade.resolve_audio_route(direction="voiceover").route == AudioRoute.OPENAI


def test_sound_effects_skip_openai_and_say_so(monkeypatch):
    _pin_probes(monkeypatch, openai=True)
    resolution = cascade.resolve_audio_route(direction="sound_effect")
    assert resolution.route == AudioRoute.NONE
    assert not resolution.servable
    assert "sound effects need one" in resolution.reason
    # The rung is still REPORTED available — it is the DIRECTION that cannot
    # ride it, and the reason chain says which credential each rung needs.
    assert [rung.available for rung in resolution.rungs] == [False, False, True]


def test_the_remedy_names_what_voiceover_can_do(monkeypatch):
    _pin_probes(monkeypatch)
    resolution = cascade.resolve_audio_route(direction="voiceover")
    assert resolution.route == AudioRoute.NONE
    assert "ElevenLabs or OpenAI" in resolution.reason


# ---------------------------------------------------------------------------
# run_audio_cascade: fail-forward, skips, cancellation, exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_rung_fails_forward_to_the_next(monkeypatch):
    _pin_probes(monkeypatch, radient=True, openai=True)
    calls: list[str] = []

    async def fake_run(route, **kwargs):
        calls.append(str(route))
        if route == AudioRoute.RADIENT:
            raise APIError("hub refused", status_code=402)
        return _rung_ok(str(route))

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    outcome = await cascade.run_audio_cascade(spec=_spec(), handle=CancelHandle())
    assert calls == ["radient", "openai"]
    assert outcome.route == AudioRoute.OPENAI
    assert [(a.route, a.outcome, a.reason_class) for a in outcome.attempts] == [
        ("radient", "failed", "insufficient_credits"),
        ("openai", "ok", ""),
    ]


@pytest.mark.asyncio
async def test_a_sound_effect_records_the_openai_skip_without_calling_it(monkeypatch):
    """OpenAI is pre-recorded at its cascade position, never dispatched.

    The walk records a pre-recorded route when it REACHES that position — and
    a success returns immediately — so the skip shows on the exhausting path
    (the image lane's own semantics), which is also the path where the note
    matters: the refusal must name the capable providers without having tried
    an OpenAI sound effect.
    """
    _pin_probes(monkeypatch, elevenlabs=True, openai=True)
    calls: list[str] = []

    async def fake_run(route, **kwargs):
        calls.append(str(route))
        raise APIError(f"{route} is down", status_code=503)

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    with pytest.raises(cascade.AudioGenerationUnavailable) as caught:
        await cascade.run_audio_cascade(spec=_spec("sound_effect"), handle=CancelHandle())
    assert calls == ["elevenlabs"], "OpenAI must never be dispatched for a sound effect"
    assert [(a.route, a.outcome, a.reason_class) for a in caught.value.attempts] == [
        ("elevenlabs", "failed", "upstream"),
        ("openai", "skipped", "unsupported"),
    ]
    assert "sound-effect" in caught.value.attempts[1].message
    assert "OpenAI: skipped" in str(caught.value)


@pytest.mark.asyncio
async def test_a_successful_first_rung_returns_before_later_skips(monkeypatch):
    _pin_probes(monkeypatch, elevenlabs=True, openai=True)
    calls: list[str] = []

    async def fake_run(route, **kwargs):
        calls.append(str(route))
        return _rung_ok(str(route))

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    outcome = await cascade.run_audio_cascade(spec=_spec("sound_effect"), handle=CancelHandle())
    assert calls == ["elevenlabs"]
    assert [(a.route, a.outcome) for a in outcome.attempts] == [("elevenlabs", "ok")]


@pytest.mark.asyncio
async def test_cancellation_is_a_stop_with_no_failover(monkeypatch):
    _pin_probes(monkeypatch, radient=True, elevenlabs=True)
    calls: list[str] = []

    async def fake_run(route, **kwargs):
        calls.append(str(route))
        raise cascade.AudioGenerationCancelled()

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    with pytest.raises(cascade.AudioGenerationCancelled):
        await cascade.run_audio_cascade(spec=_spec(), handle=CancelHandle())
    assert calls == ["radient"], "the user asked for the walk to end"


@pytest.mark.asyncio
async def test_exhaustion_carries_every_attempt_and_the_resolution(monkeypatch):
    _pin_probes(monkeypatch, radient=True, openai=True)

    async def fake_run(route, **kwargs):
        raise APIError(f"{route} is down", status_code=503)

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    with pytest.raises(cascade.AudioGenerationUnavailable) as caught:
        await cascade.run_audio_cascade(spec=_spec(), handle=CancelHandle())
    exc = caught.value
    assert "Audio generation failed on every available provider:" in str(exc)
    assert [a.route for a in exc.attempts] == ["radient", "openai"]
    assert all(a.reason_class == "upstream" for a in exc.attempts)
    assert exc.resolution is not None
    assert getattr(exc.resolution, "rungs", None)


@pytest.mark.asyncio
async def test_no_capable_rung_raises_the_remedy_before_any_dispatch(monkeypatch):
    _pin_probes(monkeypatch, openai=True)

    async def fake_run(route, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("nothing may be dispatched")

    monkeypatch.setattr(cascade, "_run_route", fake_run)
    with pytest.raises(cascade.AudioGenerationUnavailable) as caught:
        await cascade.run_audio_cascade(spec=_spec("sound_effect"), handle=CancelHandle())
    assert "sound effects need one" in str(caught.value)
    assert caught.value.attempts == ()


def test_the_rung_declarations_cover_the_wired_executors():
    # The add-a-rung checklist, as a test: every rung in the order has a
    # direction row and a label, and every label names a real route.
    for route in cascade.AUDIO_RUNG_ORDER:
        assert route in cascade.AUDIO_RUNG_DIRECTIONS
        assert route in cascade.AUDIO_RUNG_LABELS
        assert cascade.AUDIO_RUNG_DIRECTIONS[route], "a rung with no directions cannot run"
    assert cascade.AUDIO_RUNG_DIRECTIONS[AudioRoute.OPENAI] == frozenset({"voiceover"})


@pytest.mark.asyncio
async def test_admitted_terminal_rate_limit_never_fails_forward(monkeypatch):
    import httpx

    from local_operator.audiogen import rungs

    _pin_probes(monkeypatch, radient=True, elevenlabs=True)
    seen = []

    def respond(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "admitted",
                    "status": "COMPLETED",
                    "error_type": "audio_rate_limited",
                    "error": "busy after admission",
                },
            )
        raise AssertionError("terminal failure must not poll")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def route(route, **kwargs):
            seen.append(route)
            assert route == AudioRoute.RADIENT
            return await rungs.run_radient(
                kwargs["spec"],
                credential="synthetic",
                base_url="https://fixture.invalid/v1",
                handle=kwargs["handle"],
                emit=kwargs["emit"],
                pause=kwargs["pause"],
                client=client,
            )

        monkeypatch.setattr(cascade, "_run_route", route)
        with pytest.raises(cascade.AudioGenerationUnavailable) as caught:
            await cascade.run_audio_cascade(spec=_spec())
    assert seen == [AudioRoute.RADIENT]
    assert caught.value.attempts[-1].error_type == "audio_rate_limited"
