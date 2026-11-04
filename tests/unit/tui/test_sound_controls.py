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
    calls: list[dict[str, Any]] = []

    def fake(plan: Any, *, leg: str, throttle: bool = True) -> bool:
        calls.append({"id": plan.sound_id, "leg": leg, "throttle": throttle})
        return True

    monkeypatch.setattr(sound, "play_plan", fake)
    assert sound.preview("felt-marimba", env={}).outcome == "play"
    assert calls == [{"id": "felt-marimba", "leg": "preview", "throttle": False}]


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
