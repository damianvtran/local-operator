"""Observation building and artifact verification.

Asserts the content-addressed identity contract: ``observation_id ==
observation_content_id(...)``, the artifact lands at ``<root>/<sha256>``, and
the REAL ``verify_artifact`` accepts it. Also asserts the coordinate-space
invariant: native is always 1920x1080 and the PNG header must agree or the
build raises (a resized guest would silently miscalibrate every coordinate).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from lop_osworld_v2_adapter.observation import (
    NATIVE_SCREEN,
    ObservationBuilder,
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
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(shade=7), task_id="t", episode_id="e", sequence=0)
    artifact = observation.frames[0].artifact
    expected = tmp_path / hashlib.sha256(_frame(shade=7)).hexdigest()
    assert expected.exists()
    assert artifact.sha256 == hashlib.sha256(_frame(shade=7)).hexdigest()
    assert artifact.media_type == "image/png"
    assert artifact.byte_count == len(_frame(shade=7))


def test_the_real_verify_artifact_accepts_the_frame(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    # This is the same call the parent's HostVerifier makes on every frame.
    data = verify_artifact(tmp_path, observation.frames[0].artifact)
    assert data == _frame()


def test_geometry_is_native_and_unresized(tmp_path: Path) -> None:
    builder = ObservationBuilder(tmp_path)
    observation = builder.build(_raw(), task_id="t", episode_id="e", sequence=0)
    geometry = observation.frames[0].geometry
    assert geometry.native == FrameSize(width=1920, height=1080)
    assert geometry.model_visible == FrameSize(width=1920, height=1080)


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
    with pytest.raises(ObservationError):
        builder.build(raw, task_id="t", episode_id="e", sequence=0)


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
