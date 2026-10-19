"""Real RecordSink writes, artifact verification and old-shape reader round trips."""

from __future__ import annotations

import base64
import copy
import errno
import hashlib
import io
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from local_operator.evaluation.adapters.supervisor import SupervisionError
from local_operator.evaluation.record_frames import resolve_record_images
from local_operator.evaluation.record_sink import MAX_RECORD_LINE_BYTES, RecordSink
from local_operator.harness.types import (
    ImageContent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
)


def png_frame() -> bytes:
    """A valid, incompressible 1280x720 PNG, without imaging dependencies."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    # Deterministic noisy captures ensure the test catches accidental inline
    # payloads; a flat-colour PNG would fit under the bound even before the fix.
    pixels = hashlib.shake_256(b"evidence-frame-regression").digest(1280 * 720 * 3)
    rows = b"".join(b"\0" + pixels[start : start + 3840] for start in range(0, len(pixels), 3840))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(scope="module")
def frame() -> bytes:
    return png_frame()


def image(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "data": base64.b64encode(data).decode("ascii"),
        "mime_type": "image/png",
    }


def write_event(
    path: Path, payload: dict[str, Any], artifact_root: Path | None = None
) -> dict[str, Any]:
    sink = RecordSink(path, expected_bytes=0, reserve_bytes=0, artifact_root=artifact_root)
    try:
        sink.write("agent_event", payload)
        assert sink.failure is None, str(sink.failure)
    finally:
        sink.close()
    return json.loads(path.read_text())


def test_reuses_adapter_artifact_and_roundtrips_all_content(tmp_path: Path, frame: bytes) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    digest = hashlib.sha256(frame).hexdigest()
    (artifacts / digest).write_bytes(frame)
    path = tmp_path / "episode" / "events.jsonl"
    payload = {
        "event": {
            "type": "turn_end",
            "tool_results": [
                {
                    "content": [{"type": "text", "text": "observation ✓"}, image(frame)],
                    "id": "result",
                    "is_error": False,
                },
            ],
            "message": {"tool_calls": [{"arguments": image(b"literal tool argument")}]},
            "duration": 1.25,
        }
    }
    original = copy.deepcopy(payload)
    row = write_event(path, payload, artifacts)
    block = row["event"]["tool_results"][0]["content"][1]
    assert block["type"] == "image_reference"
    assert block["artifact"] == {
        "sha256": digest,
        "media_type": "image/png",
        "byte_count": len(frame),
        "path": "../artifacts/" + digest,
    }
    assert not (path.parent / "frames").exists()
    assert resolve_record_images(row, path) == {"kind": "agent_event", **original}
    assert payload == original
    assert (path.parent / block["artifact"]["path"]).read_bytes() == frame


def test_fallback_deduplicates_frame_heavy_turn(tmp_path: Path, frame: bytes) -> None:
    path = tmp_path / "events.jsonl"
    payload = {"event": {"type": "turn_end", "tool_results": [image(frame)] * 12}}
    row = write_event(path, payload)
    assert path.stat().st_size < MAX_RECORD_LINE_BYTES
    assert "record_oversized" not in row
    assert len(list((tmp_path / "frames").iterdir())) == 1
    assert resolve_record_images(row, path) == {"kind": "agent_event", **payload}


def test_oversized_text_is_preserved_and_explicitly_flagged(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    text = "é" * MAX_RECORD_LINE_BYTES
    row = write_event(path, {"event": {"type": "turn_end", "text": text}})
    assert row["event"]["text"] == text
    assert row["record_oversized"]["reason"] == "projected_record_exceeds_annotation_threshold"
    assert row["record_oversized"]["unflagged_bytes"] > MAX_RECORD_LINE_BYTES


def test_resolver_loudly_refuses_missing_and_corrupt_artifacts(
    tmp_path: Path, frame: bytes
) -> None:
    path = tmp_path / "events.jsonl"
    row = write_event(path, {"content": [image(frame)]})
    artifact = tmp_path / row["content"][0]["artifact"]["path"]
    artifact.write_bytes(b"wrong")
    with pytest.raises(SupervisionError, match="byte count differs"):
        resolve_record_images(row, path)
    artifact.unlink()
    with pytest.raises(SupervisionError, match="unsafe or unavailable"):
        resolve_record_images(row, path)


def test_failed_publication_keeps_inline_event_without_dangling_reference(
    tmp_path: Path, frame: bytes
) -> None:
    # A pre-existing truncated CAS entry models an interrupted/full-disk write.
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / hashlib.sha256(frame).hexdigest()).write_bytes(b"")
    sink = RecordSink(tmp_path / "events.jsonl", expected_bytes=0, reserve_bytes=0)
    try:
        sink.write("agent_event", {"content": [image(frame)]})
        assert sink.failure is None
        assert sink.lines_written == 1
        row = json.loads(sink.path.read_text())
        assert row["content"] == [image(frame)]
        assert "byte count differs" in row["record_image_fallbacks"][0]["reason"]
        assert row["record_image_fallbacks"][0]["path"] == "$.content[0]"
        assert "image_reference" not in sink.path.read_text()
    finally:
        sink.close()


@pytest.mark.parametrize(
    "shape",
    [
        "svg",
        "bmp",
        "jpg_alias",
        "uppercase_png",
        "jpeg_trailing",
        "empty",
        "url_details",
        "missing_mime",
        "invalid_base64",
        "non_string_mime",
    ],
)
def test_unprojectable_blocks_preserve_text_and_project_valid_siblings(
    tmp_path: Path, frame: bytes, shape: str
) -> None:
    block: dict[str, Any] = image(frame)
    if shape == "svg":
        block = {**image(b"<svg/>"), "mime_type": "image/svg+xml"}
    elif shape == "bmp":
        block = {**image(b"BM" + b"0" * 64), "mime_type": "image/bmp"}
    elif shape == "jpg_alias":
        block["mime_type"] = "image/jpg"
    elif shape == "uppercase_png":
        block["mime_type"] = "image/PNG"
    elif shape == "jpeg_trailing":
        output = io.BytesIO()
        Image.new("RGB", (2, 2)).save(output, format="JPEG")
        block = {**image(output.getvalue() + b"x" * 64), "mime_type": "image/jpeg"}
    elif shape == "empty":
        block["data"] = ""
    elif shape == "url_details":
        block = {"type": "image", "data": "https://example.invalid/image.png"}
    elif shape == "missing_mime":
        del block["mime_type"]
    elif shape == "invalid_base64":
        block["data"] = "not base64!"
    elif shape == "non_string_mime":
        block["mime_type"] = None
    path = tmp_path / "events.jsonl"
    payload: dict[str, Any] = {
        "event": {
            "type": "turn_end",
            "text": "trajectory must survive",
            "tool_results": [
                {"content": [{"type": "text", "text": "tool text"}, image(frame), block]}
            ],
        }
    }
    if shape == "url_details":
        payload["event"]["details"] = {"preview": block}
    row = write_event(path, payload)
    content = row["event"]["tool_results"][0]["content"]
    assert content[1]["type"] == "image_reference"
    assert content[2] == block
    assert row["record_image_fallbacks"]
    assert resolve_record_images(row, path)["event"] == payload["event"]
    if path.stat().st_size > MAX_RECORD_LINE_BYTES:
        assert "record_oversized" in row


def test_artifact_io_failure_is_inline_fallback_not_record_failure(
    tmp_path: Path, frame: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def storage_failure(*args: Any, **kwargs: Any) -> bytes:
        raise OSError(errno.ENOSPC, "synthetic artifact store is full")

    monkeypatch.setattr("local_operator.evaluation.record_frames.verify_artifact", storage_failure)
    payload = {"text": "survives artifact I/O failure", "content": [image(frame)]}
    row = write_event(tmp_path / "events.jsonl", payload)
    assert row["content"] == payload["content"]
    assert row["text"] == payload["text"]
    assert "synthetic artifact store is full" in row["record_image_fallbacks"][0]["reason"]


def test_large_image_url_is_annotated_without_claiming_a_size_cap(tmp_path: Path) -> None:
    payload = {
        "content": [{"type": "image_url", "image_url": "data:image/png;base64," + "A" * 533600}]
    }
    row = write_event(tmp_path / "events.jsonl", payload)
    assert row["content"] == payload["content"]
    assert row["record_oversized"]["reason"] == "projected_record_exceeds_annotation_threshold"


@pytest.mark.parametrize(
    "mime_type,format_name",
    [
        ("image/png", "PNG"),
        ("image/jpeg", "JPEG"),
        ("image/gif", "GIF"),
        ("image/webp", "WEBP"),
    ],
)
def test_documented_supported_media_project_and_resolve(
    tmp_path: Path, mime_type: str, format_name: str
) -> None:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(output, format=format_name)
    block = {**image(output.getvalue()), "mime_type": mime_type}
    path = tmp_path / "events.jsonl"
    row = write_event(path, {"content": [block]})
    assert row["content"][0]["type"] == "image_reference"
    assert "record_image_fallbacks" not in row
    assert resolve_record_images(row, path)["content"] == [block]


@pytest.mark.parametrize("corruption", ["truncated", "directory", "foreign"])
def test_bad_adapter_artifact_does_not_poison_later_events(
    tmp_path: Path, frame: bytes, corruption: str
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    target = artifacts / hashlib.sha256(frame).hexdigest()
    if corruption == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"" if corruption == "truncated" else b"x" * len(frame))
    path = tmp_path / "episode" / "events.jsonl"
    sink = RecordSink(path, expected_bytes=0, reserve_bytes=0, artifact_root=artifacts)
    try:
        for turn in range(2):
            sink.write("agent_event", {"text": f"turn {turn}", "content": [image(frame)]})
        assert sink.failure is None
        assert sink.lines_written == 2
    finally:
        sink.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for turn, row in enumerate(rows):
        assert row["text"] == f"turn {turn}"
        assert row["content"] == [image(frame)]
        assert len(row["record_image_fallbacks"]) == 1
        assert row["record_image_fallbacks"][0]["path"] == "$.content[0]"
        if corruption == "foreign":
            assert "digest differs" in row["record_image_fallbacks"][0]["reason"]
        assert resolve_record_images(row, path)["content"] == [image(frame)]
    assert not (path.parent / "frames").exists()


def test_legacy_inline_reader_fails_clearly_and_resolver_accepts_old_rows(
    tmp_path: Path, frame: bytes
) -> None:
    payload = {"content": [image(frame)]}
    path = tmp_path / "events.jsonl"
    row = write_event(path, payload)
    with pytest.raises(ValidationError, match="image"):
        ImageContent.model_validate(row["content"][0])
    assert resolve_record_images(payload, path) == payload
    restored = resolve_record_images(row, path)
    assert base64.b64decode(ImageContent.model_validate(restored["content"][0]).data) == frame


def test_tool_execution_start_args_are_not_interpreted_as_content(tmp_path: Path) -> None:
    from local_operator.headless_print import printable_event

    # This is a user argument describing an image, not a published PNG. Trying
    # to verify/persist it would fail and lose an otherwise valid event.
    event = ToolExecutionStartEvent(
        tool_call_id="call-args", tool_name="example", args=image(b"literal argument")
    )
    path = tmp_path / "events.jsonl"
    payload = {"event": printable_event(event)}
    row = write_event(path, payload)
    assert row == {"kind": "agent_event", **payload}
    assert resolve_record_images(row, path) == row
    assert not (tmp_path / "frames").exists()


def test_real_turn_end_projection_roundtrips_typed_tool_results(
    tmp_path: Path, frame: bytes
) -> None:
    from local_operator.headless_print import printable_event

    event = TurnEndEvent(
        tool_results=[
            ToolResult(
                tool_call_id="action-1",
                tool_name="desktop",
                content=[ImageContent(**image(frame))],
                duration_s=0.2,
                is_error=True,
            )
        ]
    )
    path = tmp_path / "events.jsonl"
    payload = {"event": printable_event(event)}
    row = write_event(path, payload)
    with pytest.raises(ValidationError, match="image_reference"):
        TurnEndEvent.model_validate(row["event"])
    restored = resolve_record_images(row, path)
    assert TurnEndEvent.model_validate(restored["event"]) == event
    assert path.stat().st_size < MAX_RECORD_LINE_BYTES
