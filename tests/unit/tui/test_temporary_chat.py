"""The terminal's temporary chats: verbs, band segment, sidebar group, pin refusal.

Contract §7.1 item 11 and the TUI half of item 7. The pure helpers
(``tui/temporary_chat.py``) are exercised over a real store; the app verbs
(``/new temporary``, ``/temporary on|off|status``) through a real
``OperatorApp`` pilot with a fake session, the way ``test_config_change_notice``
drives ``/new``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.resume import SessionRow
from local_operator.session import temporary
from local_operator.slash_commands import SLASH_COMMANDS
from local_operator.tui import temporary_chat
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import CatalogEntry
from local_operator.tui.sidebar_pins import read_pins, set_pin
from local_operator.tui.widgets import status_line
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SID = "a1b2c3d4e5f6"


def _store(root: Path, session_id: str = SID) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("", "toggle"),
        ("on", "on"),
        ("OFF", "off"),
        ("keep", "off"),
        ("status", "status"),
        ("maybe", None),
    ],
)
def test_parse(arg: str, expected: str | None) -> None:
    assert temporary_chat.parse(arg) == expected


def test_apply_marks_and_keeps(tmp_path: Path) -> None:
    _store(tmp_path)
    ok, sentence = temporary_chat.apply(tmp_path, SID, True)
    assert ok and "30" in sentence
    assert temporary.is_marked(tmp_path, SID)
    assert "deletes in" in temporary_chat.status_sentence(tmp_path, SID)
    ok, sentence = temporary_chat.apply(tmp_path, SID, False)
    assert ok and "kept" in sentence
    assert not temporary.is_marked(tmp_path, SID)
    assert "not temporary" in temporary_chat.status_sentence(tmp_path, SID)


def test_apply_refuses_a_pinned_chat_with_its_remedy(tmp_path: Path) -> None:
    _store(tmp_path)
    set_pin(tmp_path, SID, True)
    ok, sentence = temporary_chat.apply(tmp_path, SID, True)
    assert not ok and "unpin it first" in sentence
    assert not temporary.is_marked(tmp_path, SID)


def test_a_temporary_chat_refuses_a_pin(tmp_path: Path) -> None:
    _store(tmp_path)
    assert temporary_chat.pin_refusal_sentence(tmp_path, SID) == ""
    assert temporary_chat.apply(tmp_path, SID, True)[0]
    assert temporary_chat.pin_refusal_sentence(tmp_path, SID)


def test_the_tombstone_sentence_names_the_way_back(tmp_path: Path) -> None:
    gone = "cccccccccccc"
    assert temporary_chat.deleted_sentence(tmp_path, gone) == ""
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / temporary.TOMBSTONES_NAME).write_text(
        json.dumps([{"id": gone, "deleted_at": time.time()}])
    )
    sentence = temporary_chat.deleted_sentence(tmp_path, gone)
    assert gone in sentence and "lop --temporary" in sentence


def test_every_hold_code_has_its_own_phrase() -> None:
    phrases = {code: temporary_chat.hold_text(code) for code in temporary.HOLD_CODES}
    assert all(phrases.values())
    assert len(set(phrases.values())) == len(phrases)


def test_the_slash_command_is_registered() -> None:
    assert any(command.name == "temporary" for command in SLASH_COMMANDS)


# ---------------------------------------------------------------------------
# Band segment and sidebar group
# ---------------------------------------------------------------------------


def test_the_band_segment_is_dropped_just_before_approvals() -> None:
    for ladder in (status_line._DROP_LADDER, status_line._DROP_LADDER_QUIET):
        assert ladder.index("temporary") == ladder.index("approvals") - 1


def test_the_sidebar_files_temporary_chats_in_their_own_group_first() -> None:
    plain = CatalogEntry(SessionRow("bbbbbbbbbbbb", 2.0, "kept chat"))
    temp = CatalogEntry(SessionRow(SID, 1.0, "scratch", temporary=True))
    sidebar = SessionSidebar()
    sidebar.set_entries([plain, temp])
    rows = sidebar._display_rows()
    headers = [kind for kind, entry in rows if entry is None and kind.startswith("header:")]
    assert headers[0] == "header:temporary"
    ids = [entry.id for _kind, entry in rows if entry is not None]
    assert ids == [SID, "bbbbbbbbbbbb"]


# ---------------------------------------------------------------------------
# The app verbs
# ---------------------------------------------------------------------------


class _HexSession(FakeSession):
    @property
    def session_id(self) -> str:  # type: ignore[override]
        return SID


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


async def _bound(app: OperatorApp, pilot) -> None:  # noqa: ANN001
    for _ in range(200):
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError("the app never bound a session")


@pytest.mark.asyncio
async def test_temporary_on_asks_first_then_marks_and_off_keeps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _store(tmp_path)
    (directory / "transcript.jsonl").write_text('{"type":"message"}\n')
    app = OperatorApp(lambda: _factory(_HexSession()))
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._cmd_temporary("on", lambda *_a, **_k: None)
        await pilot.pause()
        assert not temporary.is_marked(tmp_path, SID), "a chat with messages is asked first"
        assert any("again within 30 seconds" in text for text in _notices(app))
        app._cmd_temporary("on", lambda *_a, **_k: None)
        await pilot.pause()
        assert temporary.is_marked(tmp_path, SID)
        assert app._temporary_flag is True
        assert app._status is not None and app._status._temporary
        app._cmd_temporary("status", lambda *_a, **_k: None)
        app._cmd_temporary("off", lambda *_a, **_k: None)
        await pilot.pause()
        assert not temporary.is_marked(tmp_path, SID)
        assert app._status is not None and app._status._temporary == ""


@pytest.mark.asyncio
async def test_lop_temporary_marks_the_first_bound_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(_HexSession()), start_temporary=True)
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        for _ in range(100):
            if temporary.is_marked(tmp_path, SID):
                break
            await pilot.pause()
    assert temporary.is_marked(tmp_path, SID)
    assert read_pins(tmp_path) == []


@pytest.mark.asyncio
async def test_new_temporary_marks_the_new_session_and_sends_its_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    built: list[_HexSession] = []

    async def resume_factory(_resume_id):  # noqa: ANN001, ANN202
        session = _HexSession()
        built.append(session)
        return session

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._cmd_new("temporary hello there", lambda *_a, **_k: None)
        for _ in range(400):
            if built and built[-1].prompts:
                break
            await pilot.pause()
    assert temporary.is_marked(tmp_path, SID)
    assert built and built[-1].prompts == ["hello there"]
