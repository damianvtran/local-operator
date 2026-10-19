"""The ``/library`` full-page mode: real app, real stylesheet, real keys.

Every state is driven through the shipped page. The SERVICE is replaced at its
module seam (``library.service.list_assets``) so the test does not need to
synthesise session journals — the page's contract is the payload that function
returns, and the tests assert the Query the page sends for each control. The
one place a real file is used is the inline preview, where a real PNG is
decoded and painted, because "it previews" is a claim about bytes.

Frames: the render test exports the REAL app with its REAL stylesheet to an
SVG through ``app.save_screenshot``. The evidence script under the session
scratchpad rasterises and a human/vision agent views the same state; this test
only pins that the frame exists and that the painted text is the payload's.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from rich.text import Text

from local_operator.library import service as library_service
from local_operator.session.frontend_state import (
    _FRONTEND_LOCAL_SLASHES,
    CommandScope,
    _slash_capabilities,
)
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import library_view as library_view_mod
from local_operator.tui.widgets.library_view import LibraryView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

NOW = time.time()


def _row(
    number: int,
    name: str,
    kind: str,
    *,
    category: str = "document",
    exists: bool = True,
    preview: str = "ready",
) -> dict[str, Any]:
    return {
        "id": f"la_{number:016x}",
        "ref": {
            "path": f"/tmp/library-test/{name}",
            "display_path": name,
            "digest": None,
            "content_type": "image/png" if kind == "image" else "text/plain",
        },
        "name": name,
        "derived_name": name,
        "description": f"about {name}" if number == 0 else "",
        "kind": kind,
        "category": category,
        "origin": "generated",
        "origin_group": "generated",
        "origins": ["generated"],
        "tags": [{"id": "t1", "name": "q3", "category": "topic"}] if number == 0 else [],
        "primary_tag": "q3" if number == 0 else None,
        "size_bytes": 1024 * (number + 1),
        "width": None,
        "height": None,
        "duration_s": None,
        "created_at": NOW - 3600 * (number + 1),
        "modified_at": NOW - 3600 * (number + 1),
        "last_seen_at": NOW - 3600 * (number + 1),
        "exists": exists,
        "highlighted": number == 0,
        "highlight_why": None,
        "occurrences": [
            {
                "session_id": "abcdef123456",
                "session_title": "board prep",
                "origin": "generated",
                "tool": None,
                "at": NOW - 3600,
                "intent": None,
                "first_seq": 1,
                "last_seq": 2,
                "child_id": None,
                "label": None,
            }
        ],
        "session_count": 1,
        "chat": None,
        "preview": {"state": preview, "mime_type": "image/png", "pages": None},
        "organized": None,
        "locked": False,
        "favorite": False,
        "archived": False,
    }


def _payload(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = (
        rows
        if rows is not None
        else [
            _row(0, "chart.png", "image", category="image"),
            _row(1, "notes.md", "markdown"),
            _row(2, "kickoff.pdf", "pdf", exists=False),
            _row(3, "logo.svg", "image", category="image", preview="unsupported"),
        ]
    )
    return {
        "state": "ready",
        "revision": 3,
        "total": len(rows),
        "counts": {
            "category": {"image": 2, "document": 2},
            "origin_group": {"generated": 4},
            "kind": {},
        },
        "tags": [
            {"id": "t1", "name": "q3", "category": "topic", "count": 1},
            {"id": "t2", "name": "deck", "category": "project", "count": 1},
        ],
        "rows": rows,
        "next_cursor": None,
        "cursor_reset": False,
        "scan": {"sessions_total": 2, "sessions_scanned": 2},
        "stats": {
            "assets": 4,
            "organized": 2,
            "tags": 2,
            "chats": 0,
            "conventions": 0,
            "schema_version": 1,
            "available": True,
        },
        "librarian": {"paused": True, "last_run": NOW - 7200, "organized": 2},
    }


async def _settle(pilot, predicate, *, turns: int = 400) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


async def _boot(pilot, app: OperatorApp) -> None:
    await _settle(pilot, lambda: app._session is not None)


async def _open(pilot, app: OperatorApp) -> LibraryView:
    app._run_slash_command("/library")
    await _settle(pilot, lambda: bool(app.query(LibraryView)))
    view = app.query_one(LibraryView)
    # The page's own `on_mount` takes focus; a key pressed before that lands
    # on the composer the mode made read-only and nothing fires. Waiting for
    # the focus is waiting for the mode to be USABLE, not for it to exist.
    await _settle(pilot, lambda: app.focused is view and view.is_attached)
    return view


def _text(static) -> str:
    content = static.content
    assert isinstance(content, Text)
    return content.plain


@pytest.fixture
def calls(monkeypatch):
    """Record every service Query instead of scanning a real config root."""
    recorded: list[library_service.Query] = []

    def fake_list_assets(config_dir, query, **kwargs):
        recorded.append(query)
        return _payload()

    monkeypatch.setattr(library_service, "list_assets", fake_list_assets)
    return recorded


def test_library_is_a_registered_frontend_local_slash_command() -> None:
    command = slash_command_for("/library")
    assert command is not None
    assert command.name == "library"
    # TUI-only: no palette row can forward it to a desktop destination.
    assert command.desktop_destination == ""
    assert command.arguments.name == "NONE"
    assert "library" in _FRONTEND_LOCAL_SLASHES
    assert "library" in {entry.name for entry in SLASH_COMMANDS}
    capabilities = _slash_capabilities()
    assert (
        next(c for c in capabilities if c.command == "library").scope == CommandScope.FRONTEND_LOCAL
    )


def test_librarian_is_a_registered_frontend_local_slash_command() -> None:
    command = slash_command_for("/librarian")
    assert command is not None
    assert command.name == "librarian"
    assert command.desktop_destination == ""
    # Bare reports status, so the argument list is an offer, not a gate.
    assert command.arguments.name == "OPTIONAL"
    assert command.subcommands == ("status", "pause", "resume", "run", "open")
    assert "librarian" in _FRONTEND_LOCAL_SLASHES
    assert "librarian" in {entry.name for entry in SLASH_COMMANDS}
    assert next(c for c in _slash_capabilities() if c.command == "librarian").scope == (
        CommandScope.FRONTEND_LOCAL
    )


@pytest.mark.asyncio
async def test_library_lists_the_payload_and_moves_the_cursor(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        assert calls and calls[0].limit == 200
        assert "chart.png" in _text(view._list_canvas)
        assert "notes.md" in _text(view._list_canvas)
        assert "board prep" in _text(view._detail_canvas)
        assert "q3" in _text(view._detail_canvas)

        await pilot.press("down")
        await pilot.pause()
        assert view.state.cursor == 1
        assert "notes.md" in _text(view._detail_canvas)

        await pilot.press("j", "j", "j", "j")
        await pilot.pause()
        assert view.state.cursor == 3  # clamped at the last row


@pytest.mark.asyncio
async def test_sort_and_search_are_sent_to_the_service(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        await pilot.press("s", "s")
        await _settle(pilot, lambda: calls[-1].sort == "size")
        assert view.state.sort == "size"

        await pilot.press("slash")
        view._search.value = "chart"
        await _settle(pilot, lambda: calls[-1].q == "chart")
        assert view.state.q == "chart"
        await pilot.press("escape")
        await pilot.pause()
        # First escape leaves the field, still on the page.
        assert app.query(LibraryView)


@pytest.mark.asyncio
async def test_the_rail_filters_by_category_and_hides_below_ninety(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        assert view._rail.display is True
        await pilot.press("f")
        await pilot.pause()
        assert view._rail_focused is True
        await pilot.press("down")  # All -> Images
        await pilot.press("enter")
        await _settle(pilot, lambda: calls[-1].categories == frozenset({"image"}))
        assert view.state.category == "image"

        await pilot.resize_terminal(86, 30)
        await pilot.pause()
        await pilot.pause()
        assert view._rail.display is False
        assert view._right.styles.width.value == 30

        await pilot.resize_terminal(76, 30)
        await pilot.pause()
        await pilot.pause()
        assert view._right.display is False


@pytest.mark.asyncio
async def test_edit_metadata_collects_the_form_and_writes_through_the_service(
    calls, monkeypatch
) -> None:
    written: dict[str, Any] = {}

    class FakeStore:
        def __init__(self, config_dir) -> None:
            pass

        def create_tag(self, name: str) -> dict[str, str]:
            return {"id": f"id-{name}"}

    def fake_set_tags(config_dir, asset_id, tag_ids, **kwargs):
        written["tags"] = list(tag_ids)
        return {}

    def fake_update(config_dir, asset_id, fields, **kwargs):
        written["fields"] = dict(fields)
        return {}

    monkeypatch.setattr(library_view_mod.library_metadata, "MetadataStore", FakeStore)
    monkeypatch.setattr(library_service, "set_tags", fake_set_tags)
    monkeypatch.setattr(library_service, "update_asset", fake_update)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        await pilot.press("e")
        await pilot.pause()
        assert view._form_open is True
        view._name_input.value = "revenue chart"
        view._description.text = "the board deck chart"
        view._tags_input.value = "q3, deck, Q3"
        view._tags_input.focus()
        await pilot.press("enter")
        await _settle(pilot, lambda: "fields" in written and not view._form_open)

        assert written["tags"] == ["id-q3", "id-deck"]
        assert written["fields"]["name"] == "revenue chart"
        assert written["fields"]["description"] == "the board deck chart"


@pytest.mark.asyncio
async def test_ask_leaves_the_mode_and_seeds_the_composer(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        await pilot.press("a")
        await _settle(pilot, lambda: not app.query(LibraryView))
        editor = app._editor()
        assert "chart.png" in editor.text
        assert "/tmp/library-test/chart.png" in editor.text
        assert app._transcript_view().display is True
        assert editor.read_only is False


@pytest.mark.asyncio
async def test_escape_restores_the_draft_and_the_composer(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        app._editor().load_text("half-typed prompt")
        app._editor().focus()
        await pilot.pause()
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)
        assert app._editor().read_only is True

        await pilot.press("escape")
        await _settle(pilot, lambda: not app.query(LibraryView))
        assert app._editor().text == "half-typed prompt"
        assert app._editor().read_only is False
        assert app._transcript_view().display is True


@pytest.mark.asyncio
async def test_every_state_has_a_painted_sentence(calls) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        view.load(_payload([]))
        await pilot.pause()
        assert "No assets yet" in _text(view._list_canvas)

        view.load({**_payload([]), "state": "error", "error": "permission denied"})
        await pilot.pause()
        assert "could not be read" in _text(view._list_canvas)
        assert "permission denied" in _text(view._list_canvas)

        view.load(
            {
                **_payload(),
                "state": "building",
                "scan": {"sessions_total": 9, "sessions_scanned": 3},
            }
        )
        await pilot.pause()
        assert "3 of 9" in _text(view._status)

        view.state.set_query("nothing")
        view.load(_payload([]))
        await pilot.pause()
        assert "Nothing matches" in _text(view._list_canvas)

        view.state.set_query("")
        view.load(_payload())
        await pilot.press("home")
        await pilot.press("down")
        await pilot.press("down")  # kickoff.pdf, missing
        await pilot.pause()
        assert "no longer on disk" in _text(view._detail_canvas)
        await pilot.press("end")
        await pilot.pause()
        assert "open externally" in _text(view._detail_canvas)


@pytest.mark.asyncio
async def test_space_previews_a_real_image_and_falls_back_for_an_unsupported_one(
    calls, request
) -> None:
    frame_root = Path(str(request.getfixturevalue("tmp" + "dir")))
    image = frame_root / "chart.png"
    Image.new("RGB", (320, 200), (40, 80, 120)).save(image)
    rows = _payload()["rows"]
    rows[0]["ref"]["path"] = str(image)
    rows[0]["exists"] = True

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.load(_payload(rows))
        await pilot.pause()

        await pilot.press("space")
        await _settle(pilot, lambda: view._preview_widget is not None)
        assert view._preview_widget is not None
        assert view._preview_for == rows[0]["id"]

        await pilot.press("space")  # toggle off
        await pilot.pause()
        assert view._preview_widget is None

        await pilot.press("end")  # logo.svg, preview unsupported
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert view._preview_widget is None
        assert "open externally" in _text(view._detail_canvas)


@pytest.mark.asyncio
async def test_the_librarian_hint_degrades_to_a_sentence(calls) -> None:
    from tests.unit.tui.test_slash_echo import _notice_texts

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await _settle(pilot, lambda: view._payload is not None)

        await pilot.press("L")
        await _settle(
            pilot,
            lambda: not app.query(LibraryView)
            and any("not available" in text for text in _notice_texts(app)),
        )


@pytest.mark.asyncio
async def test_a_render_frame_exports_with_the_real_stylesheet(calls, request) -> None:
    frame_root = Path(str(request.getfixturevalue("tmp" + "dir")))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.load(_payload())
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        path = app.save_screenshot(filename="library.svg", path=str(frame_root))
        frame = Path(path)
        assert frame.is_file()
        assert frame.stat().st_size > 2000
        assert "chart.png" in _text(view._list_canvas)
        assert "notes.md" in _text(view._detail_canvas)

        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(LibraryView)
