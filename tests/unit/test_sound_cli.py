"""``lop sound doctor | list | preview`` output (review round 1 M1, m2-m4; QA Q1, Q2).

``doctor`` is the honesty surface for every case where two sounds play, so
each case here is asserted by the sentence the user reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator import sound_cli
from local_operator.tui import notify

CMUX = {
    "CMUX_SURFACE_ID": "6f1d7a3e-0000-4000-8000-000000000001",
    "CMUX_WORKSPACE_ID": "6f1d7a3e-0000-4000-8000-000000000002",
}
_TERMINAL_MARKERS = (
    "TERM_PROGRAM",
    "TERM",
    "TMUX",
    "ZELLIJ",
    "KITTY_WINDOW_ID",
    "GHOSTTY_RESOURCES_DIR",
    "WEZTERM_PANE",
    "ITERM_SESSION_ID",
    "CMUX_SURFACE_ID",
    "CMUX_WORKSPACE_ID",
    "LOCAL_OPERATOR_NO_SOUND",
    "LOCAL_OPERATOR_NO_NOTIFICATIONS",
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in _TERMINAL_MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    # The rig guard answers False under this redirected HOME; open it so each
    # case reads as the operator's own machine would (the deliberate opt-in
    # tests/unit/tui/test_notify.py documents). The redirected-HOME case below
    # closes it again.
    monkeypatch.setattr(notify, "desktop_belongs_to_this_process", lambda **_k: True)
    return tmp_path


def _config(home: Path, text: str) -> None:
    directory = home / ".local-operator"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yml").write_text(text)


def _doctor(capsys: pytest.CaptureFixture[str]) -> str:
    assert sound_cli.doctor_command() == 0
    return capsys.readouterr().out


def test_apple_terminal_says_lops_sound_replaces_the_bell(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    out = _doctor(capsys)
    assert "replaces the terminal bell" in out
    assert "double sound: none expected" in out


def test_tmux_names_the_kept_bell_and_its_fix(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    out = _doctor(capsys)
    assert "double sound: inside tmux the terminal bell is kept" in out
    assert "lop config edit notifications.sound.id system" in out
    assert "none expected" not in out


def test_tmux_with_system_selected_reports_no_double(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    _config(home, "values:\n  notifications:\n    sound:\n      id: system\n")
    assert "double sound: none expected" in _doctor(capsys)


def test_cmux_without_its_config_reports_cmuxs_default_sound(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in CMUX.items():
        monkeypatch.setenv(name, value)
    out = _doctor(capsys)
    assert "cmux notifications.sound = default" in out
    assert "double sound: cmux plays its own sound too" in out


@pytest.mark.parametrize(
    "payload",
    [{"notifications": []}, ["x"], {"notifications": {"sound": 3}}],
    ids=["list-notifications", "list-root", "int-sound"],
)
def test_a_malformed_cmux_config_never_crashes(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: object,
) -> None:
    for name, value in CMUX.items():
        monkeypatch.setenv(name, value)
    (home / ".config" / "cmux").mkdir(parents=True)
    (home / ".config" / "cmux" / "cmux.json").write_text(json.dumps(payload))
    assert "cmux notifications.sound = default" in _doctor(capsys)


def test_cmux_set_to_none_reports_no_double(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in CMUX.items():
        monkeypatch.setenv(name, value)
    (home / ".config" / "cmux").mkdir(parents=True)
    (home / ".config" / "cmux" / "cmux.json").write_text('{"notifications": {"sound": "none"}}')
    assert "double sound: none expected" in _doctor(capsys)


def test_an_unknown_id_fallback_is_reported(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _config(home, "values:\n  notifications:\n    sound:\n      id: bogus\n")
    out = _doctor(capsys)
    assert "lop: plays soft-pulse" in out
    assert "'bogus' is not a known sound; playing soft-pulse instead" in out


def test_banners_off_leads_the_report(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    out = _doctor(capsys)
    assert out.splitlines()[0].startswith("lop: silent (banners are off")


def test_a_negative_hand_edited_volume_reports_as_clamped(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(home, "values:\n  notifications:\n    sound:\n      volume: -3\n")
    out = _doctor(capsys)
    assert "-3 is outside 0-100; clamped to 0" in out
    assert "volume is 0" not in out


def test_list_marks_nothing_and_notes_an_unknown_id(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(home, "values:\n  notifications:\n    sound:\n      id: bogus\n")
    assert sound_cli.list_command() == 0
    out = capsys.readouterr().out
    assert "note: notifications.sound.id 'bogus' is not a known sound" in out


def test_doctor_under_a_redirected_home_does_not_claim_to_replace_the_bell(
    home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setattr(notify, "desktop_belongs_to_this_process", lambda **_k: False)
    out = _doctor(capsys)
    assert "lop plays no notification sound here" in out
    assert "replaces the terminal bell" not in out
