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


def test_the_band_segment_sheds_late_but_never_outlives_an_alarm() -> None:
    """Just ahead of ``fork`` (the other state rung), behind every reading."""
    for ladder in (status_line._DROP_LADDER, status_line._DROP_LADDER_QUIET):
        assert ladder.index("temporary") == ladder.index("fork") - 1
        assert ladder.index("temporary") > ladder.index("context")
        assert ladder.index("temporary") < ladder.index("approvals")


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


# ---------------------------------------------------------------------------
# Round-1 remediation (core #3832)
# ---------------------------------------------------------------------------


def _routed_session(stale_owner: bool):  # noqa: ANN202
    """An ATTACHED viewer's session: every routed slash lands in ``routed``.

    ``stale_owner`` builds the capability list an owner running the round-1
    code advertises (``temporary`` authoritative), which the local-work
    pullback must still keep on this machine.
    """
    from local_operator.session.frontend_state import (
        CommandScope,
        FrontendSessionState,
        _slash_capabilities,
    )

    routed: list[tuple[str, str]] = []

    class RoutedSession(_HexSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append((command, args))
            return "! /temporary reads this machine's configuration"

    capabilities = _slash_capabilities()
    if stale_owner:
        capabilities = [
            (
                cap.model_copy(
                    update={"scope": CommandScope.AUTHORITATIVE_SESSION, "operation": "slash"}
                )
                if cap.command == "temporary"
                else cap
            )
            for cap in capabilities
        ]
    session = RoutedSession()
    session.frontend_state = FrontendSessionState(
        session_id=SID, epoch="owner", slash_capabilities=capabilities
    )
    return session, routed


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_owner", [False, True])
async def test_slash_temporary_runs_here_on_an_attached_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale_owner: bool
) -> None:
    # C-2 (core QA Q1): through the REAL slash dispatch, not ``_cmd_temporary``.
    # `/temporary` was routed to the owner, which refused it, so it did nothing
    # on every attached viewer.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _store(tmp_path)
    assert temporary.write_mark(tmp_path, SID, True, by="tui").ok
    session, routed = _routed_session(stale_owner)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._run_slash_command("/temporary off")
        for _ in range(50):
            await pilot.pause()
            if not temporary.is_marked(tmp_path, SID):
                break
        assert routed == [], "a machine-local command must not reach the owner"
        assert not temporary.is_marked(tmp_path, SID), "/temporary off kept the chat"
        app._run_slash_command("/temporary status")
        await pilot.pause()
        assert any("not temporary" in text for text in _notices(app))
    assert routed == []


@pytest.mark.asyncio
async def test_a_failed_new_temporary_never_marks_the_next_chat_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C-3 (review R1-M1), the reviewer's repro: the new session never binds,
    # then `/resume <existing>` binds an existing conversation with history,
    # which the leftover pending mark used to make temporary unconfirmed.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    existing = _store(tmp_path)
    (existing / "transcript.jsonl").write_text('{"type":"message"}\n')

    async def resume_factory(resume_id):  # noqa: ANN001, ANN202
        if resume_id is None:
            raise RuntimeError("synthetic construction failure")
        return _HexSession()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._cmd_new("temporary", lambda *_a, **_k: None)
        for _ in range(100):
            await pilot.pause()
            if not getattr(app, "_swapping_session", False):
                break
        assert app._temporary_pending is None, "the failed swap left the mark armed"
        app._cmd_resume(SID, lambda *_a, **_k: None)
        for _ in range(200):
            await pilot.pause()
            if app._session is not None and app._session.session_id == SID:
                break
        for _ in range(20):
            await pilot.pause()
    assert not temporary.is_marked(tmp_path, SID), "an existing chat was marked unconfirmed"


@pytest.mark.asyncio
async def test_a_pending_mark_never_lands_on_a_chat_with_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C-3's backstop: whatever armed it, a pending mark is for a FRESH chat.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    existing = _store(tmp_path)
    (existing / "transcript.jsonl").write_text('{"type":"message"}\n')
    app = OperatorApp(lambda: _factory(_HexSession()))
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._apply_pending_temporary(SID, "keep this question")
        await pilot.pause()
        # Review r2 M-1: the backstop said nothing and dropped the prompt.
        assert app._editor().text == "keep this question"
        assert any("cannot be made temporary" in text for text in _notices(app))
    assert not temporary.is_marked(tmp_path, SID)


@pytest.mark.asyncio
async def test_a_refused_new_temporary_puts_its_prompt_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C-14 (review R1-n2): a refused mark must not swallow the typed prompt.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _store(tmp_path)
    set_pin(tmp_path, SID, True)
    session = _HexSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        app._apply_pending_temporary(SID, "keep this question")
        await pilot.pause()
        assert app._editor().text == "keep this question"
    assert not temporary.is_marked(tmp_path, SID)
    assert session.prompts == []


def test_the_band_word_is_neutral_until_the_last_minutes(tmp_path: Path) -> None:
    # C-4 (design D1 / QA Q2): neutral while live; amber plus the countdown
    # only within five minutes of deletion and with no hold.
    _store(tmp_path)
    now = time.time()
    assert temporary.write_mark(tmp_path, SID, True, by="tui", now=now - 60).ok
    word, alarm = temporary_chat.band_state(tmp_path, SID, now=now)
    assert word == "temporary" and alarm is False
    late = now + 29 * 60
    word, alarm = temporary_chat.band_state(tmp_path, SID, now=late)
    assert alarm is True and word.startswith("deletes in")
    (tmp_path / "sessions" / temporary.STATE_NAME).write_text(
        json.dumps(
            {"v": 1, "evaluated_at": late, "sessions": {SID: {"held_at": late, "hold": "running"}}}
        )
    )
    word, alarm = temporary_chat.band_state(tmp_path, SID, now=late)
    assert (word, alarm) == ("temporary", False), "a held chat is not counting down"
    _store(tmp_path, "bbbbbbbbbbbb")
    assert temporary_chat.band_state(tmp_path, "bbbbbbbbbbbb", now=now) == ("", False)


def test_the_band_paints_muted_at_rest_and_warning_in_the_countdown() -> None:
    from local_operator.tui import theme as theme_mod

    def colour_of(segment_value: str, alarm: bool) -> str:
        from typing import cast

        from rich.style import Style
        from textual.widgets import Static

        from tests.unit.tui.test_effort import _Dock

        line = status_line.StatusLine(cast(Static, _Dock()))
        line.update(temporary=segment_value, temporary_alarm=alarm)
        plain = Style()
        text = line._left_text(set(), set(), plain, plain, plain, plain)
        start = text.plain.index(f"{status_line.ICON_TEMPORARY} {segment_value}") + 2
        for span in text.spans:
            if (
                span.start <= start < span.end
                and isinstance(span.style, Style)
                and span.style.color
            ):
                return str(span.style.color.name)
        raise AssertionError("no styled temporary segment")

    muted = theme_mod.semantic_color("muted")
    warning = theme_mod.semantic_color("warning")
    assert colour_of("temporary", False) == str(muted).lower()
    assert colour_of("deletes in 4 min", True) == str(warning).lower()


def test_the_glyph_meets_the_band_constraints() -> None:
    # C-7: U+25xx, East Asian width N, no emoji presentation, one cell, an
    # outline, and used nowhere else in the band, the sidebar or glyphs.py.
    import inspect
    import unicodedata

    from rich.cells import cell_len

    from local_operator.tui import glyphs
    from local_operator.tui.widgets import session_sidebar

    glyph = status_line.ICON_TEMPORARY
    assert len(glyph) == 1 and 0x25A0 <= ord(glyph) <= 0x25FF
    assert unicodedata.east_asian_width(glyph) == "N"
    assert not 0x25FB <= ord(glyph) <= 0x25FE
    assert cell_len(glyph) == 1
    assert unicodedata.name(glyph).startswith("WHITE ")
    band = inspect.getsource(status_line).replace(f'ICON_TEMPORARY = "{glyph}"', "")
    assert glyph not in band.split("#: A TEMPORARY chat")[0]
    assert glyph not in inspect.getsource(session_sidebar)
    assert glyph not in inspect.getsource(glyphs)


def test_status_reads_as_one_sentence_with_a_countdown_or_a_hold(tmp_path: Path) -> None:
    # C-6 (I18N-1): "temporary chat — deleted deletes in 12 min …" garbled
    # every `/temporary status` in English.
    _store(tmp_path)
    now = 1_900_000_000.0
    assert temporary.write_mark(tmp_path, SID, True, by="tui", now=now).ok
    sentence = temporary_chat.status_sentence(tmp_path, SID, now=now)
    assert sentence == "temporary chat — deletes in 30 min (/temporary off keeps it)"
    (tmp_path / "sessions" / temporary.STATE_NAME).write_text(
        json.dumps(
            {"v": 1, "evaluated_at": now, "sessions": {SID: {"held_at": now, "hold": "running"}}}
        )
    )
    sentence = temporary_chat.status_sentence(tmp_path, SID, now=now)
    assert sentence == "temporary chat — kept while it runs (/temporary off keeps it)"


def test_temporary_sidebar_rows_show_deletes_in_or_the_hold_never_the_age(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C-5 (design D2 / QA Q3): in a group about deletion, "5m" read as time LEFT.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    held = "cccccccccccc"
    _store(tmp_path)
    _store(tmp_path, held)
    now = time.time()
    assert temporary.write_mark(tmp_path, SID, True, by="tui", now=now - 60).ok
    assert temporary.write_mark(tmp_path, held, True, by="tui", now=now - 60).ok
    (tmp_path / "sessions" / temporary.STATE_NAME).write_text(
        json.dumps(
            {"v": 1, "evaluated_at": now, "sessions": {held: {"held_at": now, "hold": "open_ask"}}}
        )
    )
    plain = CatalogEntry(SessionRow("bbbbbbbbbbbb", now - 300, "kept chat"))
    temp = CatalogEntry(SessionRow(SID, now - 300, "scratch", temporary=True))
    waiting = CatalogEntry(SessionRow(held, now - 300, "question", temporary=True))
    sidebar = SessionSidebar()
    sidebar.set_entries([plain, temp, waiting])
    width = 60
    meta, alarm = sidebar._row_meta(temp, width)
    assert meta.startswith("deletes in") and alarm is False
    meta, _alarm = sidebar._row_meta(waiting, width)
    assert meta.startswith("kept while a question")
    assert sidebar._row_meta(plain, width)[0] == sidebar._age(plain)
    # Never more than half the row: the title stays readable.
    narrow, _ = sidebar._row_meta(waiting, 30)
    from rich.cells import cell_len

    assert cell_len(narrow) <= (30 - 6) // 2


# ---------------------------------------------------------------------------
# Round 2: the typed path, and the refusals that must keep the prompt
# ---------------------------------------------------------------------------


async def _type_and_enter(pilot, app: OperatorApp, text: str) -> None:  # noqa: ANN001
    """Type ``text`` key by key and press Enter with the argument list OPEN.

    No Escape first: QA round 2's Q6 repro pressed Enter while the list was
    showing, which is what completed the highlighted (provider) row in place
    of the typed word.
    """
    app._editor().text = ""
    await pilot.pause()
    await pilot.press(*["space" if char == " " else char for char in text])
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause()


@pytest.mark.asyncio
async def test_typed_temporary_on_then_enter_asks_then_marks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Q6 (QA round 2): `/temporary on` + Enter became `/temporary <provider>`
    # because the argument list fell through to the `/login` rows.
    from local_operator.tui.autocomplete import ArgumentChoice

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _store(tmp_path)
    (directory / "transcript.jsonl").write_text('{"type":"message"}\n')
    # A provider controller, so the fall-through reaches the `/login` rows as it
    # does in a real install; one row `on` fuzzy-matches stands in for them.
    monkeypatch.setattr(
        OperatorApp,
        "_login_choices",
        lambda self: [ArgumentChoice(name="openrouter", description="OpenRouter")],
    )
    app = OperatorApp(lambda: _factory(_HexSession()), provider_controller=object())
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        await _type_and_enter(pilot, app, "/temporary on")
        assert any("again within 30 seconds" in text for text in _notices(app))
        assert not temporary.is_marked(tmp_path, SID), "a chat with messages is asked first"
        await _type_and_enter(pilot, app, "/temporary on")
        assert temporary.is_marked(tmp_path, SID)
        assert app._status is not None and app._status._temporary


def test_apply_refuses_when_the_session_dir_cannot_be_created(tmp_path: Path) -> None:
    # Q7 (QA round 2): an unwritable store raised PermissionError out of
    # `apply` and took the TUI down; it is a refusal like any failed write.
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sessions.chmod(0o555)
    try:
        ok, sentence = temporary_chat.apply(tmp_path, SID, True)
    finally:
        sessions.chmod(0o755)
    assert ok is False
    assert sentence.startswith("this chat cannot be made temporary:")
    assert not temporary.is_marked(tmp_path, SID)


@pytest.mark.asyncio
async def test_new_temporary_on_an_unwritable_store_refuses_and_keeps_the_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Q7 through the app: `_apply_pending_temporary` is where the traceback
    # surfaced; the prompt goes back into the composer as on any refusal.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session = _HexSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await _bound(app, pilot)
        sessions.chmod(0o555)
        try:
            app._apply_pending_temporary(SID, "keep this question")
            await pilot.pause()
        finally:
            sessions.chmod(0o755)
        assert app._editor().text == "keep this question"
        assert any("cannot be made temporary" in text for text in _notices(app))
    assert session.prompts == []
    assert not temporary.is_marked(tmp_path, SID)


def test_minutes_left_round_down_and_never_read_zero() -> None:
    # Q8 / D6: the desktop floors, so the terminal and relay floor too. The
    # deadline is stamped a fraction after the mark; a fresh chat reads 30.
    now = 1_800_000_000.0
    info = {"marked_at": now, "idle_minutes": 30, "deletes_at": now + 1800.4, "hold": None}
    assert temporary_chat.deletes_in(info, now) == "deletes in 30 min"
    assert temporary_chat.deletes_in({**info, "deletes_at": now + 200}, now) == "deletes in 3 min"
    assert temporary_chat.deletes_in({**info, "deletes_at": now + 30}, now) == "deletes in 1 min"
    assert temporary_chat.deletes_in({**info, "deletes_at": now - 5}, now) == "deletes in 1 min"
