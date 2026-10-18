"""The central audio rate table: one source, unknown → None, versioned.

What is under test is the CONTRACT the frozen cost rules need: every estimate
carries the table's version label, an unpriceable combination answers ``None``
(never a fabricated zero), and the unpinned-model resolution matches what the
rung would actually run.
"""

from __future__ import annotations

from local_operator.audiogen import pricing


def test_every_row_carries_the_pinned_version_and_source():
    for (provider, model), rate in pricing.VOICEOVER_RATES_PER_1K_CHARS.items():
        estimate = pricing.voiceover_estimate(provider, model, 1000)
        assert estimate is not None, (provider, model)
        assert estimate.amount_usd == rate
        assert estimate.price_version == pricing.PRICE_VERSION
        assert pricing.PRICE_VERSION in estimate.provenance
        assert "estimate" in estimate.provenance
    for (provider, model), rate in pricing.SOUND_EFFECT_RATES_PER_SECOND.items():
        estimate = pricing.sound_effect_estimate(provider, model, 10)
        assert estimate is not None and estimate.amount_usd == rate * 10
        assert estimate.price_version == pricing.PRICE_VERSION


def test_unknown_provider_or_model_is_none_never_zero():
    assert pricing.voiceover_estimate("unknown", "m", 100) is None
    assert pricing.voiceover_estimate("elevenlabs", "no-such-model", 100) is None
    assert pricing.sound_effect_estimate("openai", None, 10) is None  # no OpenAI SFX path
    assert pricing.voiceover_estimate("elevenlabs", None, 0) is None
    assert pricing.sound_effect_estimate("elevenlabs", None, 0.0) is None


def test_an_unpinned_model_prices_on_what_the_rung_would_run():
    # The rung's default resolution and the table's must agree, or the consent
    # card would price one model and the call would run another.
    from local_operator.audiogen.models import (
        ELEVENLABS_VOICEOVER_MODEL,
        OPENAI_VOICEOVER_MODEL,
        default_voiceover_model,
    )

    assert default_voiceover_model("elevenlabs") == ELEVENLABS_VOICEOVER_MODEL
    assert default_voiceover_model("openai") == OPENAI_VOICEOVER_MODEL
    pinned = pricing.voiceover_estimate("elevenlabs", ELEVENLABS_VOICEOVER_MODEL, 1000)
    unpinned = pricing.voiceover_estimate("elevenlabs", None, 1000)
    assert unpinned is not None and pinned is not None
    assert unpinned.amount_usd == pinned.amount_usd
    assert unpinned.units == 1000.0 and unpinned.unit == "chars"


def test_every_byok_model_the_rungs_send_is_priceable():
    """A model the rung sends MUST have a rate row, or every estimate is None.

    The table keys are literals (it is data), so this is the drift pin: the
    ids the rungs default to — voiceover and Flash voiceover per provider, and
    the sound-effect model — are exactly the ids the table prices.
    """
    from local_operator.audiogen.models import (
        ELEVENLABS_SOUND_EFFECT_MODEL,
        ELEVENLABS_VOICEOVER_FLASH_MODEL,
        ELEVENLABS_VOICEOVER_MODEL,
        OPENAI_VOICEOVER_MODEL,
    )

    voiceover_keys = set(pricing.VOICEOVER_RATES_PER_1K_CHARS)
    assert ("elevenlabs", ELEVENLABS_VOICEOVER_MODEL) in voiceover_keys
    assert ("elevenlabs", ELEVENLABS_VOICEOVER_FLASH_MODEL) in voiceover_keys
    assert ("openai", OPENAI_VOICEOVER_MODEL) in voiceover_keys
    assert ("elevenlabs", ELEVENLABS_SOUND_EFFECT_MODEL) in set(
        pricing.SOUND_EFFECT_RATES_PER_SECOND
    )
    # And no row prices a model the lane cannot send (a typo would be dead data).
    assert all(provider in {"elevenlabs", "openai"} for provider, _model in voiceover_keys)
