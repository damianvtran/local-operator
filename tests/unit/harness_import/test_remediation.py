"""Round-1 remediation probes: preview bounds, source caps, env roots, zstd."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from local_operator.harness_import import store
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.sources.omp import encode_cwd
from local_operator.harness_import.types import ImportRequest, SourceSession
from local_operator.session.transcript import Transcript

SOURCE = "omp"
SESSION_ID = "omp-synthetic"


def _request(all_sessions: bool = False) -> ImportRequest:
    return ImportRequest(
        sources=[SOURCE],
        session_ids={} if all_sessions else {SOURCE: [SESSION_ID]},
        all=all_sessions,
    )


@pytest.mark.asyncio
async def test_sources_reports_listing_counts_and_availability(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    sources = {info.source: info for info in await service.sources()}
    assert sources["omp"].session_count == 1
    assert sources["omp"].availability == "full"


@pytest.mark.asyncio
async def test_preview_does_not_build_mapped_sessions(omp_store, monkeypatch):
    home, config = omp_store
    service = HarnessImportService(config)

    def fail(*args, **kwargs):
        raise AssertionError("preview must not map a session")

    monkeypatch.setattr("local_operator.harness_import.service.build_mapped_session", fail)
    preview = await service.preview(_request())
    assert preview.total_sessions == 1
    assert preview.total_messages == 8


@pytest.mark.asyncio
async def test_oversized_source_is_localized_not_fatal(omp_store):
    home, config = omp_store
    root = home / ".omp" / "agent" / "sessions" / "--tmp--"
    oversized = root / "oversized.jsonl"
    with oversized.open("wb") as handle:
        handle.truncate(store.MAX_SOURCE_BYTES + 1)
    service = HarnessImportService(config)
    sessions = await service.list_sessions(SOURCE)
    by_id = {row.source_id: row for row in sessions}
    assert by_id[SESSION_ID].availability == "full"
    assert any(row.availability == "unsupported" for row in sessions)
    result = await service.import_sessions(_request(all_sessions=True))
    assert result.state == "done"
    assert any(row.outcome == "imported" for row in result.sessions)
    assert any(row.outcome == "metadata_only" for row in result.sessions)


@pytest.mark.asyncio
async def test_malformed_lines_are_counted_and_warned(omp_store):
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    with path.open("a") as handle:
        handle.write("{not json\n")
    service = HarnessImportService(config)
    result = await service.import_sessions(_request())
    row = result.sessions[0]
    assert row.outcome == "imported"
    assert row.skipped >= 1
    assert any("malformed or oversized source line" in warning for warning in row.warnings)
    transcript = Transcript(config / "sessions" / row.target_session_id)
    manifest = json.loads(
        (config / "sessions" / row.target_session_id / "harness-import.json").read_text()
    )
    assert manifest["counts"]["duplicate_results_skipped"] == 0
    assert transcript.build_llm_history()


@pytest.mark.asyncio
async def test_duplicate_tool_results_are_counted(omp_store):
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "message",
                    "id": "duplicate-result",
                    "parentId": "meta1",
                    "timestamp": "2026-01-03T00:00:30.000Z",
                    "message": {
                        "role": "toolResult",
                        "timestamp": "2026-01-03T00:00:30.000Z",
                        "toolCallId": "call-omp-1",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "duplicate"}],
                    },
                }
            )
            + "\n"
        )
    service = HarnessImportService(config)
    result = await service.import_sessions(_request())
    manifest = json.loads(
        (
            config / "sessions" / result.sessions[0].target_session_id / "harness-import.json"
        ).read_text()
    )
    assert result.sessions[0].skipped >= 1
    assert manifest["counts"]["duplicate_results_skipped"] == 1


@pytest.mark.asyncio
async def test_cancel_during_the_only_session_is_honoured(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    state = {"cancelled": False}

    def progress(update):
        if update.phase == "session" and not state["cancelled"]:
            service.cancel()
            state["cancelled"] = True

    result = await service.import_sessions(_request(), progress=progress)
    assert result.state == "cancelled"
    assert all(row.outcome != "imported" for row in result.sessions)


def test_omp_memory_matches_encoded_cwd(isolated):
    home, config = isolated
    agent = home / ".omp" / "agent"
    memory = agent / "memories" / encode_cwd("/tmp/synthetic-omp")
    memory.mkdir(parents=True)
    (memory / "learned.md").write_text("- Synthetic work note\n")
    (memory / "MEMORY.md").write_text("- Synthetic personal note\n")
    service = HarnessImportService(config)
    session = SourceSession(
        source="omp",
        source_id="cwd-session",
        path=Path("/tmp"),
        cwd="/tmp/synthetic-omp",
    )
    records = service.adapters["omp"].read_memory([session])
    assert records
    assert all(record.cwd == "/tmp/synthetic-omp" for record in records)


@pytest.mark.skipif(sys.version_info < (3, 14), reason="compression.zstd is 3.14+")
def test_zstd_rollout_opens_text_mode(tmp_path):
    from compression import zstd  # type: ignore

    path = tmp_path / "rollout-synthetic.jsonl.zst"
    lines = [
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "type": "session_meta",
                "payload": {"id": "zstd-session", "cwd": "/tmp"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "zstd question"}],
                },
            }
        ),
    ]
    with zstd.open(path, "wt", encoding="utf-8") as handle:  # type: ignore[attr-defined]
        handle.write("\n".join(lines) + "\n")
    from local_operator.harness_import.sources.codex import _records

    records = _records(path)
    assert [record["type"] for record in records] == ["session_meta", "response_item"]


def test_private_use_glyphs_are_stripped_from_titles():
    from local_operator.harness_import.service import _normalise_title

    assert _normalise_title("\U000f0016 Synthetic title") == "Synthetic title"


def test_image_cap_compares_decoded_bytes(monkeypatch):
    import base64

    from local_operator.harness.types import ImageContent
    from local_operator.harness_import import mapping

    monkeypatch.setattr(mapping, "MAX_IMAGE_BYTES", 3)
    warnings: list[str] = []
    kept = mapping._clean_images(
        [ImageContent(data=base64.b64encode(b"1234").decode(), mime_type="image/png")],
        warnings,
    )
    assert kept == []
    assert warnings and "decoded payload" in warnings[0]


@pytest.mark.asyncio
async def test_unsupported_options_are_reported(omp_store):
    from local_operator.harness_import.types import ImportWarning

    home, config = omp_store
    service = HarnessImportService(config)
    request = ImportRequest(
        sources=[SOURCE],
        session_ids={SOURCE: [SESSION_ID]},
        include_cleared_segments=True,
    )
    preview = await service.preview(request)
    assert any(
        warning.code == "option_unsupported"
        for warning in preview.warnings
        if isinstance(warning, ImportWarning)
    )
