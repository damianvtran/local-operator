"""Inline UI uses the original streamed answer and preserves file highlights."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.guides.discovery import (
    discover_guides,
    guides_available,
    make_guide_resolver,
)
from local_operator.harness.types import (
    MessageUpdateEvent,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.inline_ui import INSTRUCTIONS, instructions_for
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements.decision import FILES_QUESTION_ID
from tests.unit.session.test_session import ScriptedStream
from tests.unit.supplements.test_replay_ask_resume import _Vendor
from tests.unit.supplements.test_runner import (
    _await_row,
    _make_session,
    _report_stream,
    _write_tool,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "viewer_capable", "inline_enabled"),
    [
        ("desktop", False, False),
        ("desktop", True, True),
        ("tui", False, True),
        ("mobile", False, False),
        ("mobile", True, True),
    ],
)
async def test_inline_ui_replaces_decoration_but_keeps_file_highlights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    viewer_capable: bool,
    inline_enabled: bool,
) -> None:
    target = "reports/q3.md"
    chunks = [
        'Traffic rose.\n\n```lop-ui\n{"type":"bar",',
        '"title":"Requests","data":[["Q3",42]]}\n```\n',
        "\nSee the report.",
    ]
    reply = "".join(chunks)
    stream: ScriptedStream = _report_stream(target)
    stream.turns[1] = [
        *(StreamTextDelta(delta=chunk) for chunk in chunks),
        StreamEndEvent(stop_reason="stop"),
    ]
    session = _make_session(
        tmp_path / "sessions" / "s1", stream, [_write_tool(tmp_path, [])], tmp_path
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    vendor = _Vendor()
    monkeypatch.setattr(handle._supplements, "_seam", lambda: vendor)
    vendor_calls: list[bool] = []
    monkeypatch.setattr(
        handle._supplements, "_render", lambda *args, **kwargs: vendor_calls.append(True)
    )
    handle.subscribe(lambda: None)
    events: list[object] = []
    session.subscribe(events.append)
    try:
        await session.prompt(
            "chart traffic and write a file", input_surface=surface, inline_ui=viewer_capable
        )
        row = await _await_row(session.transcript)
        if inline_enabled:
            assert all(request.system_blocks[-1] == INSTRUCTIONS for request in stream.requests)
        else:
            assert all(INSTRUCTIONS not in request.system_blocks for request in stream.requests)
        assert all(question == FILES_QUESTION_ID for question in vendor.asked)
        assert not vendor_calls
        assert len(stream.requests) == 2  # the file tool and the answer, no graphics round
        assert INSTRUCTIONS not in session._context.system_blocks
        deltas = [
            event.delta for event in events if isinstance(event, MessageUpdateEvent) and event.delta
        ]
        assert deltas == chunks
        assert row["components"] == []
        assert [file["path"] for file in row["files"]] == [target]
        anchor = next(entry for entry in session.transcript.entries() if entry.id == row["anchor"])
        plain = "".join(block["text"] for block in anchor.payload["content"])
        assert plain == ("Traffic rose.\n\n\n\nSee the report." if inline_enabled else reply)
        expected_version = 1 if inline_enabled else None
        assert anchor.inline_ui_source_version == expected_version
        assert "inline_ui_source_version" not in anchor.payload
        assert all(
            event.message._inline_ui_source_version == expected_version
            for event in events
            if isinstance(event, MessageUpdateEvent) and event.delta
        )
    finally:
        await handle.dispose()


def test_tui_is_default_and_attached_viewers_negotiate() -> None:
    assert instructions_for("tui") == INSTRUCTIONS
    for surface in ("desktop", "mobile"):
        assert instructions_for(surface) is None
        assert instructions_for(surface, viewer_capable=True) == INSTRUCTIONS
    for surface in (None, "remote", "unknown"):
        assert instructions_for(surface, viewer_capable=True) is None


def test_component_schemas_are_discovered_on_demand() -> None:
    guides = discover_guides()
    available = {guide.name: guide for guide in guides_available(guides, None)}
    assert "guide://inline-ui" in INSTRUCTIONS
    assert "scatter plot" in available["inline-ui"].description
    body = make_guide_resolver(available)("guide://inline-ui")
    assert body is not None
    for component in ("pie", "scatter", "histogram", "heatmap", "spreadsheet", "series"):
        assert f"`{component}`" in body
        assert component not in INSTRUCTIONS
    for mode in ("grouped", "stacked", "percent", "dual"):
        assert f"`{mode}`" in body
    assert len(INSTRUCTIONS) < len(body) // 10
