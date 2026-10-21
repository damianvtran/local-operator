"""``lop sessions hold|holds|unhold``: the operator's value holds.

A hold is the one refusal no policy can argue with, so the CLI must resolve a
reference to exactly one conversation, report an unreadable holds file as an
error (not as "no holds"), and record a release in the cleanup log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.cli import (
    sessions_hold_command,
    sessions_holds_command,
    sessions_unhold_command,
)
from local_operator.session.cleanup import CLEANUP_LOG_NAME, HOLDS_NAME, read_holds


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, Any] = dict(session="", reason="because", kind="operator", json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"
    for name in ("aaaa11112222", "aaaa33334444", "bbbb55556666"):
        directory = sessions / name
        directory.mkdir(parents=True)
        (directory / "transcript.jsonl").write_text('{"type":"message"}\n')
    return tmp_path


def test_hold_resolves_a_unique_prefix_and_records_the_reason(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(session="bbbb", reason="source of truth", kind="evidence")
    assert sessions_hold_command(args) == 0

    assert "Held bbbb55556666 (evidence): source of truth" in capsys.readouterr().out
    hold = read_holds(store / "sessions")["bbbb55556666"]
    assert hold["reason"] == "source of truth"
    assert hold["kind"] == "evidence"
    assert hold["by"] == "operator"


def test_hold_refuses_an_ambiguous_prefix_with_the_candidates(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_hold_command(_args(session="aaaa")) == 1

    err = capsys.readouterr().err
    assert "ambiguous" in err and "aaaa11112222" in err and "aaaa33334444" in err
    assert read_holds(store / "sessions") == {}


def test_hold_refuses_a_session_that_is_not_in_the_store(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_hold_command(_args(session="cccc")) == 1
    assert "no session" in capsys.readouterr().err


def test_holds_lists_and_serialises(store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert sessions_hold_command(_args(session="bbbb", reason="keep me")) == 0
    capsys.readouterr()

    assert sessions_holds_command(_args()) == 0
    out = capsys.readouterr().out
    assert "bbbb55556666" in out and "keep me" in out and "1 value hold(s)" in out

    assert sessions_holds_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["holds"]["bbbb55556666"]["reason"] == "keep me"


def test_an_unreadable_holds_file_is_an_error_not_an_empty_list(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (store / "sessions" / HOLDS_NAME).write_text("{ broken", encoding="utf-8")
    assert sessions_holds_command(_args()) == 1
    assert "cannot read the value holds" in capsys.readouterr().err


def test_unhold_releases_the_hold_and_writes_the_decision(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_hold_command(_args(session="bbbb", reason="keep me")) == 0
    capsys.readouterr()

    assert sessions_unhold_command(_args(session="bbbb", reason="case closed")) == 0
    out = capsys.readouterr().out
    assert "Released the hold on bbbb55556666" in out
    assert "Cleanup and retention may now apply" in out
    assert read_holds(store / "sessions") == {}
    rows = [
        json.loads(line)
        for line in (store / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert rows[-1]["policy"] == "hold-release"
    assert rows[-1]["reason"] == "case closed"


def test_a_blank_release_reason_is_refused_and_keeps_the_hold(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """UX round 1, U1: an empty reason used to release an evidence hold."""
    assert sessions_hold_command(_args(session="bbbb", reason="Evidence must remain")) == 0
    capsys.readouterr()

    assert sessions_unhold_command(_args(session="bbbb", reason="")) == 2
    assert "must be non-empty" in capsys.readouterr().err
    assert sessions_unhold_command(_args(session="bbbb", reason="   ")) == 2
    capsys.readouterr()

    assert "bbbb55556666" in read_holds(store / "sessions")
    assert not (store / "sessions" / CLEANUP_LOG_NAME).exists()
    assert sessions_holds_command(_args()) == 0
    assert "bbbb55556666" in capsys.readouterr().out


def test_a_blank_hold_reason_is_refused_before_resolution(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_hold_command(_args(session="bbbb", reason="  ")) == 2
    assert "must be non-empty" in capsys.readouterr().err
    assert read_holds(store / "sessions") == {}


def test_unhold_can_clear_a_hold_whose_session_is_gone(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_hold_command(_args(session="bbbb", reason="keep me")) == 0
    (store / "sessions" / "bbbb55556666").rename(store / "moved-away")
    capsys.readouterr()

    assert sessions_unhold_command(_args(session="bbbb55556666", reason="gone")) == 0
    assert "the session directory is gone" in capsys.readouterr().out
    assert read_holds(store / "sessions") == {}


def test_unhold_with_no_recorded_hold_says_so(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_unhold_command(_args(session="bbbb", reason="nothing")) == 1
    assert "no hold was recorded" in capsys.readouterr().err
