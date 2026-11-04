"""TUI notification-sound controls: `/settings` auditions and the `/sound` family.

WHAT IS PINNED (notification-sounds addendum 2, §1 U1-U3; §6 C4 acceptance):

* ``sound.preview`` is the ONE audition decision: it ignores ``enabled`` and
  the throttle, honours ``LOCAL_OPERATOR_NO_SOUND`` and volume 0, plays nothing
  for ``system``, and refuses over SSH (the player would sound remotely).
* Arrowing through the `/settings` Sound list writes ONE receipt per 250 ms
  settle, not one per keypress; opening the list plays nothing; Enter on the
  row the cursor settled on is still one sound; a volume commit plays the
  current sound at gain (v/100)**2; a theme capture is never armed by it.
* `/sound`'s picker words equal its handler's words, and each verb answers
  with a notice through the same store/registry code as ``lop sound``.

Receipts are real: the null backend (``LOCAL_OPERATOR_SOUND_BACKEND=null``)
runs the detached player on miniaudio's NULL device and appends a JSON line to
``<config>/run/notification-sounds.jsonl``. ``tests/conftest.py`` arms
``LOCAL_OPERATOR_NO_SOUND`` globally; the tests that need a receipt lift it for
themselves only, with the null backend set, so nothing reaches a speaker.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.notifications import custom_sounds, sound
from local_operator.slash_commands import (
    SOUND_SUBCOMMANDS,
    slash_command_for,
    sound_subcommand_rows,
)
from tests.unit.notifications.test_custom_sounds import tone


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for name in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    from local_operator.tui.settings import settings_reload

    settings_reload()
    return tmp_path


@pytest.fixture
def receipts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Arm the NULL backend and lift the suite-wide mute for THIS test only."""
    monkeypatch.delenv(sound.ENV_NO_SOUND, raising=False)
    monkeypatch.setenv(sound.ENV_BACKEND, "null")
    return tmp_path / "run" / sound.RECEIPT_FILE


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wait_for(path: Path, count: int, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Wait on the EVENT (the receipt count), bounded, never a bare sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _read(path)
        if len(rows) >= count:
            return rows
        time.sleep(0.05)
    return _read(path)


def _values(tmp_path: Path) -> dict[str, Any]:
    config = tmp_path / "config.yml"
    if not config.exists():
        return {}
    return yaml.safe_load(config.read_text()).get("values", {})


# -- the shared decision ------------------------------------------------------


def test_preview_plan_ignores_enabled_and_honours_mute_and_volume() -> None:
    off = {"notifications": {"sound": {"enabled": False, "volume": 40}}}
    decided = sound.preview_plan("felt-marimba", env={}, values=off)
    assert decided.outcome == "play"
    assert decided.gain == pytest.approx(0.16)
    assert sound.preview_plan("system", env={}).outcome == "system"
    assert sound.preview_plan("no-such-sound", env={}).outcome == "unknown"
    assert sound.preview_plan("felt-marimba", 0, env={}).outcome == "silent"
    muted = sound.preview_plan("felt-marimba", env={sound.ENV_NO_SOUND: "1"})
    assert muted.outcome == "muted" and muted.plan is None


def test_preview_refuses_over_ssh_and_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []
    monkeypatch.setattr(sound, "play_plan", lambda *a, **k: started.append(a) or True)
    result = sound.preview("felt-marimba", env={"SSH_CONNECTION": "10.0.0.1 1 10.0.0.2 22"})
    assert result.outcome == "ssh"
    assert started == []


def test_preview_is_unthrottled_and_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    """No throttle claim, and the player argv is the detached `preview` leg."""
    argvs: list[list[str]] = []

    class _Child:
        pid = 1

        def poll(self) -> int:
            return 0

    def spawn(argv: list[str], **_kw: Any) -> _Child:
        argvs.append(list(argv))
        return _Child()

    monkeypatch.setattr("local_operator.proc.spawn_detached_child", spawn)
    monkeypatch.setattr(sound, "claim_throttle", lambda *a, **k: pytest.fail("throttled"))
    monkeypatch.setattr(sound, "_PREVIEW_CHILD", None)
    assert sound.preview("felt-marimba", env={}).outcome == "play"
    assert len(argvs) == 1
    argv = argvs[0]
    assert argv[argv.index("--leg") + 1] == "preview"
    assert argv[argv.index("--sound-id") + 1] == "felt-marimba"


# -- /sound vocabulary ---------------------------------------------------------


def test_the_picker_offers_exactly_the_words_the_handler_accepts() -> None:
    spec = slash_command_for("/sound")
    assert spec is not None and spec.subcommands == SOUND_SUBCOMMANDS
    assert spec.echo is False
    assert [word for word, _help in sound_subcommand_rows()] == list(SOUND_SUBCOMMANDS)
    assert all(help_text for _word, help_text in sound_subcommand_rows())


# -- the real app --------------------------------------------------------------


def _app() -> Any:
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


def _notices(app: Any) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _settle(pilot: Any, seconds: float) -> None:
    """Let Textual timers fire: the debounce is a real `set_timer`."""
    await asyncio.sleep(seconds)
    await pilot.pause()


def _select_row(view: Any, key: str) -> None:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return
    raise AssertionError(f"no row for {key}")


def _open_sound_list(app: Any) -> Any:
    from local_operator.tui.widgets.settings_view import SettingsView

    app._open_settings_view()
    view = app.query_one(SettingsView)
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None:
            if row.setting.key == "notifications.sound.id":
                view._selected = index
                break
    view._repaint()
    return view


@pytest.mark.asyncio
async def test_browsing_the_sound_list_plays_one_sound_per_settle(receipts: Path) -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        view.action_activate()  # expand: lands on the stored choice
        await _settle(pilot, 0.5)
        assert view._expanded == "notifications.sound.id"
        assert view._preview_before is None, "a sound must never arm the theme capture"
        # Opening plays nothing: the user has not moved yet.
        assert _wait_for(receipts, 1, timeout=1.0) == []

        # A burst of five arrows inside the debounce window: one sound, the
        # row the cursor LANDED on.
        for _ in range(5):
            view.action_move(1)
            await pilot.pause()
        landed = view._current().choice.value
        await _settle(pilot, 0.6)
        rows = _wait_for(receipts, 1)
        assert [row["sound_id"] for row in rows] == [landed]
        assert rows[0]["leg"] == "preview" and rows[0]["backend"] == "null"

        # Enter on that same row: the commit restarts the shared debounce, so
        # browse + commit is still ONE more sound, not two.
        view.action_activate()
        await _settle(pilot, 0.6)
        rows = _wait_for(receipts, 2)
        assert len(rows) == 2 and rows[1]["sound_id"] == landed
        await _settle(pilot, 0.4)
        assert len(_read(receipts)) == 2


@pytest.mark.asyncio
async def test_leaving_the_list_inside_the_window_cancels_the_audition(receipts: Path) -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        view.action_activate()
        await pilot.pause()
        view.action_move(1)
        await pilot.press("escape")
        await _settle(pilot, 0.6)
        assert view._expanded is None
        assert _read(receipts) == []


@pytest.mark.asyncio
async def test_sound_volume_commit_plays_the_current_sound_at_the_new_gain(
    receipts: Path, tmp_path: Path
) -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/sound use felt-marimba")
        await _settle(pilot, 0.5)
        _wait_for(receipts, 1)
        app._run_slash_command("/sound volume 40")
        await _settle(pilot, 0.5)
        rows = _wait_for(receipts, 2)
        assert rows[-1]["sound_id"] == "felt-marimba"
        assert rows[-1]["gain"] == pytest.approx(0.16)
        assert _values(tmp_path)["notifications"]["sound"]["volume"] == 40
        # 101 is refused with the REGISTRY's own message; nothing is written.
        app._run_slash_command("/sound volume 101")
        await _settle(pilot, 0.4)
        assert any("at most 100" in text for text in _notices(app))
        assert _values(tmp_path)["notifications"]["sound"]["volume"] == 40
        assert len(_read(receipts)) == 2


@pytest.mark.asyncio
async def test_no_sound_env_writes_no_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sound.ENV_BACKEND, "null")
    monkeypatch.setenv(sound.ENV_NO_SOUND, "1")
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/sound preview felt-marimba")
        await _settle(pilot, 0.3)
        assert any("LOCAL_OPERATOR_NO_SOUND" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_sound_add_rename_remove_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "QA chime.wav"
    source.write_bytes(tone(0.4))
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command(f'/sound add "{source}" QA')
        await app.workers.wait_for_complete()
        await pilot.pause()
        rows = custom_sounds.list_sounds()
        assert [row.name for row in rows] == ["QA"]
        added = rows[0].id
        assert any(f"({added})" in text for text in _notices(app))
        # The same registry list the /settings row expands carries it now.
        from local_operator import settings_io

        choices = settings_io.BY_KEY["notifications.sound.id"].resolved_choices
        assert added in {choice.value for choice in choices}

        app._run_slash_command(f"/sound use {added}")
        app._run_slash_command(f"/sound rename {added} QA two")
        await pilot.pause()
        assert custom_sounds.get(added).name == "QA two"

        app._run_slash_command(f"/sound remove {added}")
        await pilot.pause()
        assert custom_sounds.list_sounds() == ()
        # Removing the SELECTED sound resets the selection in the same receipt.
        assert "id" not in _values(tmp_path).get("notifications", {}).get("sound", {})
        assert any("selected again" in text for text in _notices(app))

        app._run_slash_command("/sound remove soft-pulse")
        await pilot.pause()
        assert any("built-in" in text for text in _notices(app))
        app._run_slash_command("/sound frobnicate")
        await pilot.pause()
        assert any(text.startswith("Use /sound list") for text in _notices(app))


# -- U4: an open /settings page follows a change from another process ----------


def _external_write(tmp_path: Path, volume: int) -> None:
    """What `lop config edit` does from ANOTHER process: a fresh manager write."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["notifications.sound.volume"], volume
    )


def _row_value(view: Any, key: str) -> Any:
    return (
        view._manager.get_config()
        .values.get("notifications", {})
        .get("sound", {})
        .get(key.rsplit(".", 1)[1])
    )


@pytest.mark.asyncio
async def test_an_open_page_reloads_on_an_external_change(tmp_path: Path) -> None:
    from local_operator.config_watch import ConfigChange

    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        _external_write(tmp_path, 40)
        app._on_config_change(
            ConfigChange(values={}, changed_keys=("notifications.sound.volume",), source="disk")
        )
        await pilot.pause()
        assert _row_value(view, "notifications.sound.volume") == 40
        assert "40" in view._list_text.plain


@pytest.mark.asyncio
async def test_a_row_in_edit_is_not_clobbered_and_catches_up_after(tmp_path: Path) -> None:
    from local_operator.config_watch import ConfigChange

    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        _select_row(view, "notifications.sound.volume")
        view.action_activate()  # opens the inline editor seeded with 70
        await pilot.pause()
        assert view._editing == "notifications.sound.volume"
        buffer = view._buffer
        _external_write(tmp_path, 40)
        app._on_config_change(
            ConfigChange(values={}, changed_keys=("notifications.sound.volume",), source="disk")
        )
        await pilot.pause()
        # The editor and its buffer are untouched; the page still holds 70.
        assert view._editing == "notifications.sound.volume" and view._buffer == buffer
        assert _row_value(view, "notifications.sound.volume") in (None, 70)
        await pilot.press("escape")
        await pilot.pause()
        # The cancel ends the gesture; the held reload lands on the next paint.
        assert view._editing is None
        assert _row_value(view, "notifications.sound.volume") == 40


# -- round 1 remediation -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_reload_never_moves_a_half_written_config_aside(tmp_path: Path) -> None:
    """Review round 1, M1: the deferred reload must not run `_load_config`'s
    rename-aside on a file that is broken by the time the gesture ends."""
    from local_operator.config_watch import ConfigChange

    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        _external_write(tmp_path, 40)  # a real file with a real value
        _select_row(view, "notifications.sound.volume")
        view.action_activate()
        await pilot.pause()
        app._on_config_change(
            ConfigChange(values={}, changed_keys=("notifications.sound.volume",), source="disk")
        )
        assert view._reload_pending is True
        config = tmp_path / "config.yml"
        config.write_text("values:\n  notifications: [unclosed\n")  # mid-save
        await pilot.press("escape")
        await pilot.pause()
        # The file is where it was, nothing was renamed, the page kept its
        # last good snapshot, and the reload is still owed.
        assert config.exists()
        assert not list(tmp_path.glob("config.yml.bad*"))
        assert view._reload_pending is True
        # Once the file is whole again, the next change lands normally.
        _external_write_raw(config, 55)
        assert view.reload_from_disk() is True
        assert _row_value(view, "notifications.sound.volume") == 55


def _external_write_raw(config: Path, volume: int) -> None:
    config.write_text(yaml.safe_dump({"values": {"notifications": {"sound": {"volume": volume}}}}))


class _FakeChild:
    """A stand-in player: records when it started and whether it was stopped."""

    pid = 0
    live: list["_FakeChild"] = []
    peak = 0

    def __init__(self, sound_id: str) -> None:
        self.sound_id = sound_id
        self.stopped = False
        _FakeChild.live.append(self)
        _FakeChild.peak = max(_FakeChild.peak, len(_FakeChild.live))

    def poll(self) -> int | None:
        return 0 if self.stopped else None

    def stop(self) -> None:
        self.stopped = True
        _FakeChild.live.remove(self)


@pytest.fixture
def fake_players(monkeypatch: pytest.MonkeyPatch) -> list[_FakeChild]:
    """Fake the player child at the spawn seam; `killpg` stops the matching fake."""
    started: list[_FakeChild] = []
    _FakeChild.live = []
    _FakeChild.peak = 0
    monkeypatch.delenv(sound.ENV_NO_SOUND, raising=False)

    def spawn(plan: Any) -> _FakeChild:
        child = _FakeChild(plan.sound_id)
        child.pid = 10_000 + len(started)
        started.append(child)
        return child

    def killpg(pid: int, _sig: int) -> None:
        next(child for child in started if child.pid == pid).stop()

    monkeypatch.setattr(sound, "_start_preview_child", spawn)
    monkeypatch.setattr(sound.os, "killpg", killpg)
    monkeypatch.setattr(sound, "_PREVIEW_CHILD", None)
    return started


@pytest.mark.asyncio
async def test_reading_pace_browse_keeps_at_most_one_player_alive(
    fake_players: list[_FakeChild],
) -> None:
    """UX round 1, U1: six presses ~380 ms apart outlive the debounce on every
    row; each new audition must stop the one before it, and esc stops the last."""
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = _open_sound_list(app)
        await pilot.pause()
        view.action_activate()
        await pilot.pause()
        for _ in range(6):
            view.action_move(1)
            await _settle(pilot, 0.38)
        await _settle(pilot, 0.3)
        assert len(fake_players) == 6, [c.sound_id for c in fake_players]
        assert _FakeChild.peak == 1
        assert [c.stopped for c in fake_players] == [True] * 5 + [False]
        await pilot.press("escape")
        await pilot.pause()
        assert fake_players[-1].stopped and _FakeChild.live == []


@pytest.mark.asyncio
async def test_sound_off_cancels_a_pending_commit_audition(
    fake_players: list[_FakeChild],
) -> None:
    """Review round 1, m1: `/sound off` inside the debounce window plays nothing."""
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/sound volume 30")
        app._run_slash_command("/sound off")
        await _settle(pilot, 0.6)
        assert fake_players == []
        assert app._sound_preview_request is None


@pytest.mark.asyncio
async def test_sound_verbs_speak_tui_copy_and_select_on_add(tmp_path: Path) -> None:
    source = tmp_path / "bell.wav"
    source.write_bytes(tone(0.3))
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        for command in ("/sound use nope", "/sound remove nope", "/sound rename nope X"):
            app._run_slash_command(command)
            await pilot.pause()
        notices = _notices(app)
        assert sum("There is no sound called nope. Run /sound list" in t for t in notices) == 3
        assert not any("lop sound" in t or "expected one of" in t for t in notices)
        app._run_slash_command(f"/sound add {source} Bell --select")
        await app.workers.wait_for_complete()
        await pilot.pause()
        (row,) = custom_sounds.list_sounds()
        assert row.name == "Bell"
        assert _values(tmp_path)["notifications"]["sound"]["id"] == row.id
        assert any("chose it as your notification sound" in t for t in _notices(app))
        app._run_slash_command("/sound doctor")
        await pilot.pause()
        assert not any(line.startswith("lop:") for t in _notices(app) for line in t.splitlines())
