"""``lop sessions cleanup``: the master switch governs the command.

Round 1 of #645 let the bare command run past ``session.cleanup.enabled:
false`` on the theory that typing it was consent. ``/settings`` leaves the
limits in the file when the switch is turned off, so a user who read "off:
nothing is ever removed" and ran the command "to see what it would do" lost
16 of 34 sessions (QA Q1, UX U2, review R1-5). These tests pin the contract
that replaced it: refuse, list-then-confirm-then-remove, ``--force`` as the
only override, JSON always on stdout.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.cli import sessions_cleanup_command
from local_operator.config import ConfigManager
from local_operator.session.cleanup import CLEANUP_LOG_NAME, mark_store


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        dry_run=False,
        force=False,
        yes=False,
        max_sessions=None,
        max_inactive_days=None,
        max_total_bytes=None,
        remove_empty=None,
        json=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """15 real 40-day-old transcripts plus 3 empties, marked, isolated."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"
    mark_store(sessions)
    old = time.time() - 40 * 86400
    for index in range(15):
        directory = sessions / f"s{index:02d}"
        directory.mkdir()
        (directory / "transcript.jsonl").write_text('{"type":"message"}\n')
        os.utime(directory / "transcript.jsonl", (old + index, old + index))
    for index in range(3):
        directory = sessions / f"e{index:02d}"
        directory.mkdir()
        os.utime(directory, (old, old))
    return tmp_path


def _count(root: Path) -> int:
    return sum(1 for p in (root / "sessions").iterdir() if p.is_dir())


def _config(root: Path, **cleanup: object) -> None:
    ConfigManager(root).update_config({"session": {"cleanup": cleanup}})


def test_bare_command_refuses_when_the_switch_is_off(store: Path, capsys: Any) -> None:
    _config(store, enabled=False, max_sessions=3)
    assert sessions_cleanup_command(_args()) == 2
    err = capsys.readouterr().err
    assert "session.cleanup.enabled is off" in err and "--force" in err
    assert _count(store) == 18
    assert not (store / "sessions" / CLEANUP_LOG_NAME).exists()


def test_a_flag_does_not_override_the_switch(store: Path) -> None:
    _config(store, enabled=False)
    assert sessions_cleanup_command(_args(max_sessions=3, yes=True)) == 2
    assert _count(store) == 18


def test_dry_run_lists_with_the_switch_off_and_says_so(store: Path, capsys: Any) -> None:
    _config(store, enabled=False, remove_empty=True)
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "session.cleanup.enabled is off" in out and "preview only" in out
    assert out.count("would remove") == 3 + 1  # three rows plus the summary line
    assert "nothing was removed (dry run)" in out
    assert _count(store) == 18


def test_no_limits_names_the_switch_and_json_is_always_json(store: Path, capsys: Any) -> None:
    assert sessions_cleanup_command(_args()) == 1
    assert "session.cleanup.enabled is off" in capsys.readouterr().err
    assert sessions_cleanup_command(_args(json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "nothing-to-do" and payload["enabled"] is False


def test_enabled_run_lists_first_then_asks_then_removes(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(store, enabled=True, remove_empty=True)
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert sessions_cleanup_command(_args()) == 0
    out = capsys.readouterr().out
    assert out.index("about to remove 3") < out.index("will remove") < out.index("removed 3")
    assert _count(store) == 15
    rows = [
        json.loads(line)
        for line in (store / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert len(rows) == 3 and all(row["actor"] == "cli" for row in rows)


def test_declining_the_prompt_removes_nothing(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(store, enabled=True, remove_empty=True)
    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert sessions_cleanup_command(_args()) == 2
    assert "not confirmed" in capsys.readouterr().out
    assert _count(store) == 18


def test_non_tty_without_yes_refuses(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(store, enabled=True, remove_empty=True)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert sessions_cleanup_command(_args()) == 2
    assert _count(store) == 18


def test_force_with_yes_overrides_the_switch_and_records_it(store: Path, capsys: Any) -> None:
    _config(store, enabled=False, remove_empty=True)
    assert sessions_cleanup_command(_args(force=True, yes=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["forced"] is True and payload["enabled"] is False
    assert payload["outcome"] == "removed" and len(payload["removed"]) == 3
    assert {"session", "policy", "reason", "title", "idle_days", "size_bytes"} <= set(
        payload["removed"][0]
    )
    assert _count(store) == 15


def test_rows_carry_title_age_and_size(store: Path, capsys: Any) -> None:
    _config(store, enabled=True, max_inactive_days=7)
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    # 15 transcripts 40 d idle, the 10 most recent spared -> s00..s04.
    assert out.count("would remove s0") == 5
    assert "40d" in out and "(no title)" in out and "[max_inactive_days]" in out


def test_negative_limits_are_rejected_by_the_parser() -> None:
    from local_operator.cli import _non_negative_int

    with pytest.raises(argparse.ArgumentTypeError):
        _non_negative_int("-3")
    assert _non_negative_int("0") == 0
