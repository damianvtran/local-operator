"""``lop aida note``: R23's guarded write path, as a command an agent can run.

Aida has no Python module to import, so the guarded writer is exposed as a
command (her seed documents it). These tests pin the CLI half: the receipt, the
refusal sentences and the exit codes a calling model reads to decide whether to
retry; the write path's own refusals live in ``tests/unit/aida/test_aida_profile.py``.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

from local_operator.cli import aida_note_command


def _args(text: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(text=text)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch HOME with a redirected config dir, as an isolated run needs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    (tmp_path / ".local-operator").mkdir()
    return tmp_path


def _instructions(home: Path) -> Path:
    return home / ".local-operator" / "system_prompt.md"


def test_a_note_is_recorded_and_receipted(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert aida_note_command(_args("Name: Damian")) == 0
    out = capsys.readouterr().out
    assert "recorded" in out
    # The receipt is home-relative (the `_home_relative` rendering), which is
    # the shape every other confirmation in the CLI uses.
    assert "~/.local-operator/system_prompt.md" in out
    assert "- Name: Damian" in _instructions(home).read_text(encoding="utf-8")


def test_text_is_read_from_stdin_when_omitted(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Name: from stdin\nEmail: op@x\n"))
    assert aida_note_command(_args(None)) == 0
    text = _instructions(home).read_text(encoding="utf-8")
    assert "- Name: from stdin" in text
    assert "- Email: op@x" in text


def test_a_blank_note_is_a_refusal_with_a_sentence(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert aida_note_command(_args("   \n")) == 1
    err = capsys.readouterr().err
    assert "not recorded" in err
    assert "blank" in err
    assert not _instructions(home).exists()


def test_a_duplicate_is_success_without_appending(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert aida_note_command(_args("Name: Damian")) == 0
    capsys.readouterr()
    assert aida_note_command(_args("Name: Damian")) == 0
    assert "already recorded" in capsys.readouterr().out
    text = _instructions(home).read_text(encoding="utf-8")
    assert text.count("- Name: Damian") == 1


def test_section_markers_are_refused_with_their_reason(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.aida import profile

    assert aida_note_command(_args(f"x {profile.SECTION_START} y")) == 1
    err = capsys.readouterr().err
    assert "not recorded" in err
    assert "corrupt" in err
