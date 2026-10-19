"""Real RecordSink writes, artifact verification and old-shape reader round trips."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
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
    assert row["record_oversized"]["reason"] == "non_image_payload_exceeds_limit"
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


def test_failed_publication_records_failure_without_dangling_reference(
    tmp_path: Path, frame: bytes
) -> None:
    # A pre-existing truncated CAS entry models an interrupted/full-disk write.
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / hashlib.sha256(frame).hexdigest()).write_bytes(b"")
    sink = RecordSink(tmp_path / "events.jsonl", expected_bytes=0, reserve_bytes=0)
    try:
        sink.write("agent_event", {"content": [image(frame)]})
        assert sink.failure is not None
        assert "byte count differs" in str(sink.failure)
        assert sink.lines_written == 0
        assert sink.path.read_bytes() == b""
    finally:
        sink.close()


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
