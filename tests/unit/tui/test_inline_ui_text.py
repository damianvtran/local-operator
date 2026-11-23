"""Contract and real Textual assistant-frame coverage for inline UI fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from local_operator.session.inline_ui_text import terminal_inline_ui
from local_operator.tui.widgets.assistant import AssistantBlock, _markdown, flatten
from scripts.visual_capture import save_capture, svg_text_runs_by_row


def _widget(kind: str = "bar", **fields: object) -> str:
    data = {"type": kind, "title": "Requests by shift", **fields}
    return "```lop-ui\n" + json.dumps(data) + "\n```"


@pytest.mark.parametrize(
    ("kind", "data", "expected"),
    [
        ("bar", {"data": [["Early", 8]], "unit": "Requests"}, ("Early", "Requests", "8")),
        ("line", {"data": [["Monday", -2]]}, ("Monday", "-2")),
        ("pie", {"data": [["East", 3]]}, ("East", "3")),
        ("histogram", {"data": [["0–9", 4]]}, ("0–9", "4")),
        (
            "scatter",
            {"xLabel": "Hours", "yLabel": "Items", "data": [[1.5, 10]]},
            ("Hours", "Items", "1.5", "10"),
        ),
        (
            "heatmap",
            {"columns": ["Early", "Late"], "rowLabels": ["Mon"], "rows": [[18, None]]},
            ("Early", "Late", "Mon", "18"),
        ),
        ("table", {"columns": ["Car", "Count"], "rows": [["Blue", 3]]}, ("Car", "Blue")),
        (
            "spreadsheet",
            {"columns": ["Total", "Blank"], "rows": [["=SUM(A1:A2)", None]]},
            ("Total", "=SUM(A1:A2)", "Blank"),
        ),
    ],
)
def test_widget_is_visible_as_data(kind, data, expected) -> None:
    source = f"Before.\n\n{_widget(kind, **data)}\n\nAfter."
    visible = terminal_inline_ui(source, streaming=False)
    text = flatten(_markdown(visible), 80).plain
    assert all(label in text for label in ("Before.", "Requests by shift", "After.", *expected))
    assert "lop-ui" not in visible


@pytest.mark.parametrize("mode", ["grouped", "stacked", "percent", "dual"])
def test_series_preserves_categories_and_units(mode) -> None:
    fields = {
        "mode": mode,
        "labels": ["Early"],
        "unit": "Items",
        "rightUnit": "Dollars",
        "series": [
            {"label": "Red", "values": [1], "axis": "left"},
            {"label": "Blue", "values": [3], "axis": "right"},
        ],
    }
    visible = terminal_inline_ui(_widget("series", **fields), streaming=False)
    text = flatten(_markdown(visible), 80).plain
    assert all(label in text for label in ("Early", "Red", "Blue"))
    if mode == "percent":
        assert "1 (25.0%)" in text and "3 (75.0%)" in text and "Items" in text
    elif mode == "dual":
        assert "Red (Items)" in text and "Blue (Dollars)" in text
    else:
        assert "Red (Items)" in text and "Blue (Items)" in text


def test_wide_layout_and_percent_keep_exact_data_at_60_columns() -> None:
    wide = _widget(
        "spreadsheet",
        columns=[f"Revenue {i}" for i in range(8)],
        rows=[[1234567 + i for i in range(8)]],
    )
    percent = _widget(
        "series",
        mode="percent",
        labels=["Q1", "Q2"],
        unit="Requests",
        series=[{"label": "Red", "values": [4, 4000]}, {"label": "Blue", "values": [9, 9000]}],
    )
    for source, expected in (
        (wide, [f"Revenue {i}: {1234567 + i}" for i in range(8)]),
        (percent, ["4 (30.8%)", "9 (69.2%)", "4000 (30.8%)", "9000 (69.2%)", "Requests"]),
    ):
        projected = terminal_inline_ui(source, streaming=False)
        text = flatten(_markdown(projected), 52).plain
        assert all(token in text for token in expected), text
        assert "…" not in text


@pytest.mark.parametrize(
    "fence",
    [
        "```lop-ui\n{{\n```",
        "```lop-ui extra\n{}\n```",
        _widget("unknown"),
        _widget(data=[["not a number", True]]),
        _widget(data=[["not finite", float("inf")]]),
        _widget(data=[["A", 1]], unused=float("nan")),
        _widget(data=[["A", 1e308]]).replace("1e+308", "1e309"),
        _widget(data=[["A", 1]] * 33),
        _widget("pie", data=[["no share", 0]]),
        _widget("histogram", data=[["negative count", -1]]),
        _widget("table", columns=["A", "B"], rows=[[1]]),
        _widget("heatmap", columns=["A"], rowLabels=["B"], rows=[["text"]]),
        _widget(
            "series", mode="percent", labels=["A"], series=[{"label": "No share", "values": [0]}]
        ),
        _widget("series", mode="dual", labels=["A"], series=[{"label": "One axis", "values": [1]}]),
    ],
)
def test_invalid_fences_are_hidden_with_prose_retained(fence) -> None:
    visible = terminal_inline_ui(f"Before.\n{fence}\nAfter.", streaming=False)
    assert visible == "Before.\n\nAfter."


@pytest.mark.parametrize(
    "tail", ["`", "``", "```", "```l", "```lop-", "```lop-ui", "```lop-ui\n{", "```lop-ui\n{}\n``"]
)
def test_streaming_opener_and_incomplete_payload_are_hidden(tail) -> None:
    assert terminal_inline_ui(f"Before.\n\n{tail}", streaming=True).strip() == "Before."


@pytest.mark.parametrize(
    "source",
    [
        "````md\n```lop-ui\n{}\n```\n````",
        "> ```lop-ui\n> {}\n> ```",
        "- ```lop-ui\n  {}\n  ```",
        "    ```lop-ui\n    {}\n    ```",
        "```python\nprint('```lop-ui')\n```",
    ],
)
def test_nested_and_ordinary_fences_keep_their_source(source) -> None:
    assert terminal_inline_ui(source, streaming=True) == source
    assert terminal_inline_ui(source, streaming=False) == source


def test_crlf_and_multiple_widgets_without_following_blank_lines() -> None:
    widget = _widget(data=[["Early", 8]])
    source = f"Before.\n{widget}\nBetween.\n{widget}\nAfter.".replace("\n", "\r\n")
    visible = terminal_inline_ui(source, streaming=False)
    assert visible.count("Requests by shift") == 2
    assert "Between.\r\n" in visible
    assert visible.endswith("After.")


def test_widget_values_never_become_terminal_links_or_markup() -> None:
    attack = "[click](https://example.test) | *markup* <img> `code`\n\x1b[31m\u202e"
    source = _widget("table", title=attack, columns=["Column | name"], rows=[[attack]])
    visible = terminal_inline_ui(source, streaming=False)
    assert "https://" not in visible
    segments = list(Console(width=100).render(_markdown(visible)))
    assert not any(segment.style and segment.style.link for segment in segments)
    rendered = "".join(segment.text for segment in segments)
    assert "[click](https://example.test) | *markup* <img> `code`" in rendered
    assert "Column | name" in rendered
    assert "\x1b" not in rendered and "\u202e" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (80, 24), (150, 40)])
async def test_real_assistant_stream_and_settled_frame(size, tmp_path: Path) -> None:
    from local_operator.harness.types import (
        Message,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
    )
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.events import AssistantDelta
    from tests.unit.session.test_session import wait_for
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    lead = "Early handled most requests.\n\nSee the breakdown.\n\n"
    source = lead + _widget(data=[["Early", 18], ["Late", 4]], unit="Requests") + "\n\nDone."
    partial = source[: source.index('"title"')]

    async def frame(target_app, pilot, name: str) -> str:
        await pilot.pause()
        target = tmp_path / f"{name}.svg"
        save_capture(target_app, target)
        return "\n".join("".join(row) for row in svg_text_runs_by_row(target.read_text()))

    # Same app and stylesheet, independent baseline. A retained unmarked code
    # sample must not hide a later streaming leak in the positive frame.
    legacy = OperatorApp(lambda: _factory(FakeSession()))
    async with legacy.run_test(size=size) as pilot:
        await await_session_boot(legacy)
        legacy.post_message(AssistantDelta(partial))
        assert '"type": "bar"' in await frame(legacy, pilot, "before")

    async with app.run_test(size=size) as pilot:
        await await_session_boot(app)
        # Use the actual session-event ingress rather than bypassing the bridge
        # with already display-shaped Textual messages.
        message = Message(role="assistant")
        message._inline_ui_source_version = 1
        session.emit(MessageStartEvent(message=message))
        session.emit(MessageUpdateEvent(message=message, delta=partial))
        await wait_for(lambda: app._streaming_block is not None, timeout=10)
        streaming = await frame(app, pilot, "streaming")
        assert "Early handled most requests." in streaming
        assert '"type"' not in streaming and "Requests by shift" not in streaming
        session.emit(MessageUpdateEvent(message=message, delta=source[len(partial) :]))
        await wait_for(
            lambda: app._streaming_block is not None
            and "Requests by shift" in app._streaming_block.text(),
            timeout=10,
        )
        complete = await frame(app, pilot, "closed")
        assert "Requests by shift" in complete and "Done." in complete

        final = Message.assistant(source).model_copy(update={"id": message.id})
        final._inline_ui_source_version = 1
        session.emit(MessageEndEvent(message=final))
        settled = await frame(app, pilot, "settled")
        expected = ("Requests by shift", "Early", "Late", "18", "Done.")
        assert all(label in settled for label in expected)
        blocks = list(app.query(AssistantBlock))
        assert '"type"' not in blocks[-1].text()
        assert settled == await frame(app, pilot, "settled-again")
        geometry = json.loads((tmp_path / "settled.geometry.json").read_text())
        assert geometry["grid"] == list(size) and geometry["css_path"]


@pytest.mark.asyncio
async def test_page_keys_read_a_narrow_grid_from_the_focused_composer(tmp_path: Path) -> None:
    from local_operator.harness.types import (
        Message,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
    )
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.session.test_session import wait_for
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    labels = ("actual", "target", "prior", "net", "gross", "forecast", "high")
    columns = ["Region", *(f"Revenue {label}" for label in labels)]
    rows = [
        ["North", *range(1234567, 1234574)],
        ["South", *range(2234567, 2234574)],
    ]
    source = (
        "Collecting exact values.\n\n"
        + _widget("spreadsheet", title="Quarterly planning values", columns=columns, rows=rows)
        + "\n\n"
        + _widget(
            "series",
            title="Share of requests",
            mode="percent",
            labels=["Q1", "Q2"],
            unit="requests",
            series=[{"label": "Red", "values": [4, 4000]}, {"label": "Blue", "values": [9, 9000]}],
        )
        + "\n"
    )
    late = "\nEnd of report.\n\nAnd a late update."
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await await_session_boot(app)
        editor = app.query_one(Editor)
        view = app.query_one(TranscriptView)
        assert app.focused is editor
        await pilot.press("x")
        caret = editor.selection

        async def frame(name: str) -> str:
            await pilot.pause()
            assert app.focused is editor and editor.text == "x" and editor.selection == caret
            path = tmp_path / f"{name}.svg"
            save_capture(app, path)
            return "\n".join("".join(row) for row in svg_text_runs_by_row(path.read_text()))

        message = Message(role="assistant")
        message._inline_ui_source_version = 1
        session.emit(MessageStartEvent(message=message))
        session.emit(MessageUpdateEvent(message=message, delta=source))
        await wait_for(
            lambda: app._streaming_block is not None
            and "Quarterly planning values" in app._streaming_block.text(),
            timeout=10,
        )
        await pilot.press("ctrl+home")
        await pilot.wait_for_scheduled_animations()
        top = await frame("narrow-top")
        assert view.scroll_offset.y == 0

        await pilot.press("pagedown")
        await pilot.wait_for_scheduled_animations()
        middle = await frame("narrow-middle")
        middle_y = view.scroll_offset.y
        assert 0 < middle_y < view.max_scroll_y
        assert "Revenue forecast: 1234572" not in top
        assert "Revenue forecast: 1234572" in middle

        session.emit(MessageUpdateEvent(message=message, delta=late))
        await wait_for(
            lambda: app._streaming_block is not None
            and "And a late update." in app._streaming_block.text(),
            timeout=10,
        )
        await frame("narrow-late-append")
        assert view.scroll_offset.y == middle_y
        final = Message.assistant(source + late).model_copy(update={"id": message.id})
        final._inline_ui_source_version = 1
        session.emit(MessageEndEvent(message=final))
        await wait_for(lambda: app._streaming_block is None, timeout=10)
        await frame("narrow-late-final")
        assert view.scroll_offset.y == middle_y

        pages = [top, middle]
        # Each press must advance until the measured tail, never jump back to
        # the start or lose the middle of the wrapped grid.
        while view.scroll_offset.y < view.max_scroll_y:
            before = view.scroll_offset.y
            await pilot.press("pagedown")
            await pilot.wait_for_scheduled_animations()
            pages.append(await frame(f"narrow-page-{len(pages)}"))
            assert view.scroll_offset.y > before
        assert "And a late update." in pages[-1]
        for row in rows:
            for column, value in zip(columns, row, strict=True):
                assert any(f"{column}: {value}" in page for page in pages)
        assert not view.show_horizontal_scrollbar
        assert app.screen.virtual_size == app.screen.size

        await pilot.press("pageup")
        await pilot.wait_for_scheduled_animations()
        await frame("narrow-page-up")
        before = view.scroll_offset.y
        assert 0 < before < view.max_scroll_y

        # The same key still navigates the draft if its caret can move.
        await pilot.press("shift+enter", "z", "up", "pagedown")
        assert editor.selection.end[0] == 1 and editor.text == "x\nz"
        assert view.scroll_offset.y == before and app.focused is editor


@pytest.mark.asyncio
async def test_authoritative_empty_final_removes_a_widget_only_terminal_row():
    from local_operator.harness.types import (
        Message,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
    )
    from local_operator.tui.app import OperatorApp
    from tests.unit.session.test_session import wait_for
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)):
        await await_session_boot(app)
        message = Message(role="assistant")
        message._inline_ui_source_version = 1
        session.emit(MessageStartEvent(message=message))
        session.emit(
            MessageUpdateEvent(message=message, delta=_widget(data=[["Early", 18]]) + "\n")
        )
        await wait_for(lambda: app._streaming_block is not None, timeout=10)
        assert "Requests by shift" in app.query_one(AssistantBlock).text()
        session.emit(MessageEndEvent(message=message))
        await wait_for(lambda: not list(app.query(AssistantBlock)), timeout=10)
