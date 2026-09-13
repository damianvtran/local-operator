"""Observation building and artifact verification.

Asserts the content-addressed identity contract: ``observation_id ==
observation_content_id(...)``, the artifact lands at ``<root>/<sha256>``, and
the REAL ``verify_artifact`` accepts it. Also asserts the coordinate-space
invariant: native is always 1920x1080 and the PNG header must agree or the
build raises (a resized guest would silently miscalibrate every coordinate).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from lop_osworld_v2_adapter.observation import (
    NATIVE_SCREEN,
    NO_FRAME_MESSAGE,
    SCREENSHOT_CAUSE_KEY,
    ObservationBuilder,
    ObservationCauseError,
    ObservationError,
    write_png_rgb,
)

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.protocol import FrameSize


def _frame(shade: int = 1) -> bytes:
    width, height = NATIVE_SCREEN.width, NATIVE_SCREEN.height
    return write_png_rgb(width, height, bytes((shade, shade, shade)) * (width * height))


def _raw(shade: int = 1, with_a11y: bool = False) -> dict[str, object]:
    return {
        "screenshot": _frame(shade),
        "accessibility_tree": "<tree/>" if with_a11y else None,
        "terminal": None,
        "instruction": "do the thing",
    }


def test_observation_id_is_content_derived(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    assert observation.observation_id == observation_content_id(observation)


def test_artifact_lands_at_the_content_address(tmp_path: Path) -> None:
    # The address is that of the BOUNDED frame, not the guest capture: the
    # artifact is what the model saw, so its digest, size and media type all
    # describe the bytes that were sent.
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(shade=7), task_id="t", episode_id="e", sequence=0)
    artifact = observation.frames[0].artifact
    payload = (tmp_path / artifact.sha256).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == artifact.sha256
    assert artifact.byte_count == len(payload)
    assert artifact.media_type in {"image/png", "image/jpeg"}
    # The native capture is no longer a published artifact.
    assert not (tmp_path / hashlib.sha256(_frame(shade=7)).hexdigest()).exists()


def test_the_real_verify_artifact_accepts_the_frame(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    # This is the same call the parent's HostVerifier makes on every frame.
    data = verify_artifact(tmp_path, observation.frames[0].artifact)
    assert hashlib.sha256(data).hexdigest() == observation.frames[0].artifact.sha256


def test_geometry_keeps_native_and_publishes_the_bounded_model_size(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    geometry = observation.frames[0].geometry
    # Native stays the guest's real screen: it is what coordinates convert BACK
    # to, and it is never inferred from the image.
    assert geometry.native == FrameSize(width=1920, height=1080)
    assert geometry.model_visible == FrameSize(width=1280, height=720)


def test_sequence_zero_carries_the_instruction(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    assert observation.text == "do the thing"


def test_later_sequences_drop_the_instruction(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=1)
    assert observation.text is None


def test_a_missing_screenshot_raises(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    raw = _raw()
    raw["screenshot"] = None
    with pytest.raises(ObservationError) as raised:
        builder.build(raw, task_id="t", episode_id="e", sequence=0)
    # Degrades to exactly the pre-cause text, and invents nothing: with no
    # provider cause there is no exception cause either.
    assert str(raised.value) == NO_FRAME_MESSAGE
    assert raised.value.__cause__ is None


def test_a_provider_cause_rides_as_the_error_cause_not_the_message(tmp_path: Path) -> None:
    """The provider's account must not rewrite the harness's own vocabulary.

    ``NO_FRAME_MESSAGE`` is journalled as ``observation-phase-retry`` and is what
    a bundle reader greps for, so a provider cause travels as the error's
    ``__cause__`` -- which the worker bounds and canary-checks as
    ``RpcErrorDetail.causes`` -- and never as the message.
    """

    builder = ObservationBuilder(tmp_path)
    raw = _raw()
    raw["screenshot"] = None
    raw[SCREENSHOT_CAUSE_KEY] = (
        "screenshot unavailable: upstream_failures=3 kinds=status,status,status"
    )
    with pytest.raises(ObservationError) as raised:
        builder.build(raw, task_id="t", episode_id="e", sequence=0)
    assert str(raised.value) == NO_FRAME_MESSAGE
    assert isinstance(raised.value.__cause__, ObservationCauseError)
    assert "upstream_failures=3" in str(raised.value.__cause__)


def test_a_stray_cause_does_not_change_observation_identity(tmp_path: Path) -> None:
    """Capacity facts stay out of ``observation_content_id``.

    The cause rides the RAW dict and is dropped at the builder, so an
    observation built beside a provider cause is content-identical to the same
    frame built without one -- which is the invariant
    ``docs/benchmarks/osworld_2/README.md`` states for environment facts.
    """

    builder = ObservationBuilder(tmp_path)
    plain = builder.build(_raw(shade=5), task_id="t", episode_id="e", sequence=0)
    with_cause = _raw(shade=5)
    with_cause[SCREENSHOT_CAUSE_KEY] = "screenshot unavailable: elapsed_ms=15012"
    caused = builder.build(with_cause, task_id="t", episode_id="e", sequence=0)
    assert caused.observation_id == plain.observation_id
    assert SCREENSHOT_CAUSE_KEY not in (caused.metadata or {})


def test_a_resized_guest_frame_raises(tmp_path: Path) -> None:
    # A 100x100 frame claims to be the screen but is not the native size: the
    # guest resized, and every pointer coordinate would be wrong.
    builder = ObservationBuilder(tmp_path)
    raw = _raw()
    raw["screenshot"] = write_png_rgb(100, 100, b"\x00\x00\x00" * (100 * 100))
    with pytest.raises(ObservationError):
        builder.build(raw, task_id="t", episode_id="e", sequence=0)


def test_identical_screens_give_distinct_ids_across_sequences(tmp_path: Path) -> None:
    # ExecutionReceipt refuses input == output observation id; the sequence is
    # part of the hashed content, so two visually identical screens still
    # advance the episode.
    builder = ObservationBuilder(tmp_path)
    a = builder.build(_raw(shade=3), task_id="t", episode_id="e", sequence=1)
    b = builder.build(_raw(shade=3), task_id="t", episode_id="e", sequence=2)
    assert a.observation_id != b.observation_id


def test_metadata_records_a11y_availability_without_a_frame(tmp_path: Path) -> None:
    # The a11y tree is deliberately NOT a frame (a geometry for XML is a
    # fiction); its presence is metadata only.
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(with_a11y=True), task_id="t", episode_id="e", sequence=0)
    assert len(observation.frames) == 1
    assert observation.metadata["a11y_available"] is True


def test_metadata_excludes_floats(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    for value in observation.metadata.values():
        assert not isinstance(value, float)


def _photographic_frame(size: tuple[int, int]) -> bytes:
    """A native-sized frame that is NOT line art and NOT flat.

    ``write_png_rgb`` with a constant shade compresses to a few KB and reads as
    line art, which takes the ladder's bilevel exemption — so it would prove
    nothing about the screen edge. Real per-pixel content is what makes the
    resize and the format choice meaningful.
    """
    import os

    width, height = size
    return write_png_rgb(width, height, os.urandom(width * height * 3))


def test_frame_artifact_is_the_bounded_image_and_geometry_maps_back_to_native(
    tmp_path: Path,
) -> None:
    """The published frame IS what the model sees, and it still points home.

    Three properties the episode depends on together: the artifact decodes to
    the model-visible size (not the guest's), a coordinate at the far corner of
    that space converts back into native pixels, and an identical guest screen
    yields an identical artifact address — which is what ``_frames_identical``
    reads to tell the model the screen did not move.
    """
    from PIL import Image

    png = _photographic_frame((NATIVE_SCREEN.width, NATIVE_SCREEN.height))
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(
        {"screenshot": png, "accessibility_tree": None, "terminal": None, "instruction": "go"},
        task_id="t",
        episode_id="e",
        sequence=0,
    )

    frame = observation.frames[0]
    payload = (tmp_path / frame.artifact.sha256).read_bytes()
    assert Image.open(io.BytesIO(payload)).size == (1280, 720)
    assert frame.geometry.model_visible == FrameSize(width=1280, height=720)
    assert frame.geometry.native == NATIVE_SCREEN

    # Coordinates the model emits in this space land back on the guest's.
    # The interior scales by 1.5 native px per model px...
    interior = frame.geometry.model_to_native(640, 360)
    assert (interior.x, interior.y) == (960, 540)
    penultimate = frame.geometry.model_to_native(1278, 718)
    assert (penultimate.x, penultimate.y) == (1917, 1077)
    # ...and the last row/column is CLAMPED to the last native pixel rather
    # than scaled (protocol._scale_clamped_axis), so a click on the extreme
    # edge of the frame cannot land one pixel short of the screen edge.
    corner = frame.geometry.model_to_native(1279, 719)
    assert (corner.x, corner.y) == (1919, 1079)

    # Identical native bytes must give an identical artifact address, both
    # within one builder (memoised) and across a fresh one (deterministic).
    again = builder.build(
        {"screenshot": png, "accessibility_tree": None, "terminal": None, "instruction": "go"},
        task_id="t",
        episode_id="e",
        sequence=1,
    )
    assert again.frames[0].artifact.sha256 == frame.artifact.sha256
    fresh = ObservationBuilder(tmp_path).build(
        {"screenshot": png, "accessibility_tree": None, "terminal": None, "instruction": "go"},
        task_id="t",
        episode_id="e",
        sequence=2,
    )
    assert fresh.frames[0].artifact.sha256 == frame.artifact.sha256


def test_native_dimension_check_still_runs_on_the_raw_screenshot(tmp_path: Path) -> None:
    """The guest-resize guard must judge the GUEST's bytes, not our own.

    Bounding happens after this check by construction: if the assertion ever
    moved onto the bounded frame it would compare our resizer's output against
    the native screen and either fire on every observation or never fire at
    all, and the case it exists for — a guest that silently changed resolution,
    miscalibrating every coordinate — would go unnoticed.
    """
    builder = ObservationBuilder(tmp_path)
    raw = {
        "screenshot": _photographic_frame((1600, 900)),
        "accessibility_tree": None,
        "terminal": None,
        "instruction": "go",
    }
    with pytest.raises(ObservationError, match="guest frame is 1600x900"):
        builder.build(raw, task_id="t", episode_id="e", sequence=0)
