"""The per-family request mapping: neutral values onto each wire, and refusals.

Every expectation here was read from the vendor's own live schema on
2026-10-10 (captures in the video phase's scratchpad); the point of the table
is that a value a family cannot represent is REFUSED (fail-forward at the
rung), never silently substituted.
"""

from __future__ import annotations

import pytest

from local_operator.video import models as video_models


def test_family_for_model_longest_prefix_wins() -> None:
    assert video_models.family_for_model("fal-ai/veo3.1/image-to-video").name == "veo3.1"
    assert video_models.family_for_model("alibaba/wan-3.0-prime/text-to-video").name == "wan-3.0"
    assert video_models.family_for_model("alibaba/wan-3.0/text-to-video").name == "wan-3.0"
    assert video_models.family_for_model("fal-ai/wan/v2.7/edit-video").name == "wan-3.0"
    assert video_models.family_for_model("minimax/h3-max/text-to-video").name == "h3-max"
    assert (
        video_models.family_for_model("google/gemini-omni-flash/video").name == "gemini-omni-flash"
    )
    # Unknown families take the GENERIC mapping (numeric duration, sources).
    unknown = video_models.family_for_model("vendor/unknown-model")
    assert unknown is video_models.GENERIC_FAMILY
    # NO kling entry exists on purpose: its slugs are view-redacted in every
    # capture, and a pin is never transcribed from a view.
    assert video_models.family_for_model("kling/anything").name == "generic"


def test_veo_string_durations_and_rejection() -> None:
    family = video_models.family_for_model("fal-ai/veo3.1/image-to-video")
    params = video_models.map_family_params(
        family,
        prompt="p",
        duration_s=8,
        resolution="720p",
        aspect_ratio="16:9",
        audio=False,
        seed=1,
    )
    assert params["duration"] == "8s"
    assert params["resolution"] == "720p"
    assert params["generate_audio"] is False
    with pytest.raises(video_models.VideoParamRejected):
        video_models.map_family_params(
            family,
            prompt="p",
            duration_s=5,
            resolution=None,
            aspect_ratio=None,
            audio=None,
            seed=None,
        )
    with pytest.raises(video_models.VideoParamRejected):
        # veo3.1 does not offer 480p: refusal, not substitution.
        video_models.map_family_params(
            family,
            prompt="p",
            duration_s=8,
            resolution="480p",
            aspect_ratio=None,
            audio=None,
            seed=None,
        )


def test_wan_and_h3_mappings() -> None:
    wan = video_models.family_for_model("alibaba/wan-3.0/text-to-video")
    params = video_models.map_family_params(
        wan,
        prompt="p",
        duration_s=5,
        resolution="1080p",
        aspect_ratio="9:16",
        audio=True,
        seed=None,
    )
    assert params["duration"] == 5  # integer seconds on this family
    assert params["resolution"] == "1080p"
    assert params["audio"] is True

    h3 = video_models.family_for_model("minimax/h3-max/text-to-video")
    params = video_models.map_family_params(
        h3,
        prompt="p",
        duration_s=10,
        resolution="720p",
        aspect_ratio=None,
        audio=None,
        seed=None,
    )
    assert params["duration"] == 10
    assert params["resolution"] == "768P"  # the family's own spelling
    # prompt expansion is pinned OFF so a request never silently spends
    # extra time/money on a rewrite.
    assert params["prompt_expansion_mode"] == "disabled"


def test_generic_mapping_keeps_the_request_runnable() -> None:
    generic = video_models.GENERIC_FAMILY
    params = video_models.map_family_params(
        generic,
        prompt="p",
        duration_s=5,
        resolution=None,
        aspect_ratio=None,
        audio=None,
        seed=7,
    )
    assert params == {"prompt": "p", "duration": 5, "seed": 7}
    with pytest.raises(video_models.VideoParamRejected):
        video_models.map_family_params(
            generic,
            prompt="p",
            duration_s=5,
            resolution="720p",
            aspect_ratio=None,
            audio=None,
            seed=None,
        )
    assert video_models.GENERIC_FAMILY.source_fields == {
        "image": "image_url",
        "video": "video_url",
    }


def test_estimate_unit_price_posture() -> None:
    veo = video_models.family_for_model("fal-ai/veo3.1/text-to-video")
    assert video_models.estimate_unit_price_usd(veo, "720p") == pytest.approx(0.20)
    # None quotes the LOWEST listed tier, never a figure the family's own
    # table cannot back (veo lists 0.20 at both HD tiers).
    assert video_models.estimate_unit_price_usd(veo, None) == pytest.approx(0.20)
    assert video_models.estimate_unit_price_usd(video_models.GENERIC_FAMILY, "720p") is None


def test_fal_defaults_never_pin_veo() -> None:
    for mode, model_id in video_models.FAL_DEFAULT_MODELS.items():
        assert "veo" not in model_id, f"{mode} default must not pin a retiring veo family"
    assert set(video_models.FAL_DEFAULT_MODELS) == {"t2v", "i2v", "v2v"}


def test_mode_source_kinds_vocabulary() -> None:
    assert video_models.VIDEO_MODE_SOURCE_KIND == {"t2v": None, "i2v": "image", "v2v": "video"}
    assert video_models.resolve_mode(source_image=False, source_video=False) == "t2v"
    assert video_models.resolve_mode(source_image=True, source_video=False) == "i2v"
    assert video_models.resolve_mode(source_image=False, source_video=True) == "v2v"
