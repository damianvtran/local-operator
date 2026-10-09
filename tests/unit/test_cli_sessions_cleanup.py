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
        delegated_max_age_hours=None,
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
    rows = [line for line in out.splitlines() if line.startswith("  would remove ")]
    assert len(rows) == 3 and "would remove 3" in out  # three rows plus the summary line
    assert "nothing was removed (dry run)" in out
    assert _count(store) == 18


def test_no_limits_and_delegated_off_names_the_switch_and_json_is_always_json(
    store: Path, capsys: Any
) -> None:
    _config(store, delegated={"enabled": False})
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
    assert out.index("will remove 3") < out.index("will remove  ") < out.index("removed 3")
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
    assert "40.0d" in out and "(no title)" in out and "[max_inactive_days]" in out
    # U15: every row says whose it is, and U13: every row fits 100 columns.
    rows = [line for line in out.splitlines() if "would remove" in line and "s0" in line]
    assert rows and all(" user " in row for row in rows), rows
    assert max(len(row) for row in rows) <= 100, max(rows, key=len)


def test_negative_limits_are_rejected_by_the_parser() -> None:
    from local_operator.cli import _non_negative_int

    with pytest.raises(argparse.ArgumentTypeError):
        _non_negative_int("-3")
    assert _non_negative_int("0") == 0


# -- the two classes ---------------------------------------------------------


def _delegated(root: Path, name: str, *, age_h: float = 100.0, origin: str = "subagent") -> Path:
    directory = root / "sessions" / name
    directory.mkdir()
    (directory / "origin.json").write_text(json.dumps({"origin": origin}))
    transcript = directory / "transcript.jsonl"
    transcript.write_text('{"type":"message"}\n')
    stamp = time.time() - age_h * 3600
    os.utime(transcript, (stamp, stamp))
    return directory


def test_dry_run_prints_one_section_per_class_with_the_parent_class_off(
    store: Path, capsys: Any
) -> None:
    """Today this exits 1 'no limits configured'; the delegated class needs no limits."""
    _delegated(store, "kid-old-0001")
    _delegated(store, "kid-new-0001", age_h=1)
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert out.index("== Your conversations (parent sessions) ==") < out.index(
        "== Delegated work (subagents and background sessions) =="
    )
    assert "parent class: off" in out
    assert "policy: on, remove delegated sessions idle over 48h" in out
    assert "would remove kid-old-0001" in out and "kept         kid-new-0001" in out
    assert (store / "sessions" / "kid-old-0001").exists(), "a dry run removes nothing"


def test_a_real_run_removes_the_delegated_rows_and_leaves_the_users_sessions(
    store: Path, capsys: Any
) -> None:
    _delegated(store, "kid-old-0001")
    assert sessions_cleanup_command(_args(yes=True)) == 0
    out = capsys.readouterr().out
    assert "removed      kid-old-0001" in out and not (store / "sessions" / "kid-old-0001").exists()
    assert _count(store) == 18, "the 15 transcripts and 3 empties are the parent class, untouched"
    rows = [json.loads(x) for x in (store / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()]
    assert [(r["session"], r["policy"], r["actor"]) for r in rows] == [
        ("kid-old-0001", "delegated_max_age", "cli")
    ]


def test_the_age_override_widens_and_narrows_the_window(store: Path, capsys: Any) -> None:
    _delegated(store, "kid-60h-00001", age_h=60)
    assert sessions_cleanup_command(_args(dry_run=True, delegated_max_age_hours=72)) == 0
    assert "kept         kid-60h-00001" in capsys.readouterr().out
    assert sessions_cleanup_command(_args(dry_run=True, delegated_max_age_hours=24)) == 0
    assert "would remove kid-60h-00001" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["1", "721", "abc", "0", "-5"])
def test_the_age_override_is_validated_like_the_setting(text: str) -> None:
    from local_operator.cli import _delegated_hours

    with pytest.raises(argparse.ArgumentTypeError):
        _delegated_hours(text)
    assert _delegated_hours("2") == 2 and _delegated_hours("720") == 720


def test_json_keeps_its_old_keys_and_gains_per_class_objects(store: Path, capsys: Any) -> None:
    _delegated(store, "kid-old-0001")
    _config(store, enabled=True, remove_empty=True)
    assert sessions_cleanup_command(_args(dry_run=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {
        "outcome",
        "enabled",
        "forced",
        "dry_run",
        "scanned",
        "removed",
        "protected",
        "errors",
        "skipped",
        "confirmed",
        "record",
    } <= set(payload)
    assert {c["session"] for c in payload["removed"]} == {"e00", "e01", "e02", "kid-old-0001"}
    assert payload["parent"]["removed"] and payload["delegated"]["enabled"] is True
    assert [c["session"] for c in payload["delegated"]["removed"]] == ["kid-old-0001"]
    assert payload["delegated"]["max_age_hours"] == 48
    # F3: ONE population, every directory once — the old sum added the parent
    # class to the delegated pass's whole-store count (18 + 19 = 37 here; 15 +
    # 18 = 33 in the review's probe). Asserts the VALUE, not just the key.
    assert payload["scanned"] == _count(store) == 19


def test_parent_limits_with_the_switch_off_refuse_even_though_delegated_is_on(
    store: Path, capsys: Any
) -> None:
    """Delegated being on must not turn a refused parent run into 'nothing to remove'."""
    _config(store, enabled=False, max_sessions=3)
    assert sessions_cleanup_command(_args()) == 2
    assert "session.cleanup.enabled is off" in capsys.readouterr().err
    assert _count(store) == 18


def test_a_force_run_with_the_delegated_switch_off_still_honours_force(store: Path) -> None:
    _config(store, delegated={"enabled": False})
    _delegated(store, "kid-old-0001")
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    assert (store / "sessions" / "kid-old-0001").exists()
    assert sessions_cleanup_command(_args(force=True, yes=True)) == 0
    assert not (store / "sessions" / "kid-old-0001").exists()


# -- a class that will not run reads "would remove" (F1) ----------------------


def test_a_real_run_with_the_parent_switch_off_previews_the_parent_rows(
    store: Path, capsys: Any
) -> None:
    """F1: parent off + delegated on — the parent rows are a PREVIEW, and the
    listing must say so ("would remove" + the off-note), not promise them."""
    _config(store, enabled=False, remove_empty=True)
    _delegated(store, "kid-old-0001")
    assert sessions_cleanup_command(_args(yes=True)) == 0
    out = capsys.readouterr().out
    parent_section = out.split("== Delegated work")[0]
    assert "note: session.cleanup.enabled is off" in parent_section
    assert "this is a preview only" in parent_section
    assert "would remove 3" in parent_section
    assert "will remove" not in parent_section
    assert "removed      kid-old-0001" in out
    assert not (store / "sessions" / "kid-old-0001").exists()
    assert (store / "sessions" / "e00").exists(), "the parent class was not run"


def test_a_real_run_with_the_delegated_switch_off_previews_its_rows(
    store: Path, capsys: Any
) -> None:
    """The mirror: delegated off + parent on."""
    _config(store, enabled=True, remove_empty=True, delegated={"enabled": False})
    _delegated(store, "kid-old-0001")
    assert sessions_cleanup_command(_args(yes=True)) == 0
    out = capsys.readouterr().out
    parent_section, delegated_section = out.split("== Delegated work")
    assert "will remove 3" in parent_section
    assert "note: delegated cleanup is off in config; this is a preview only" in delegated_section
    assert "would remove kid-old-0001" in delegated_section
    assert "will remove kid-old-0001" not in out
    assert (store / "sessions" / "kid-old-0001").exists(), "the delegated class was not run"
    assert not (store / "sessions" / "e00").exists(), "the parent class did run"


def test_a_real_run_with_nothing_to_do_never_promises_a_removal(store: Path, capsys: Any) -> None:
    """F1's probe: delegated off with rows to show and nothing removable in the
    running class used to print "will remove 3" and then "nothing to remove"."""
    _config(store, enabled=True, max_sessions=999, delegated={"enabled": False})
    _delegated(store, "kid-old-0001")
    assert sessions_cleanup_command(_args(yes=True)) == 0
    out = capsys.readouterr().out
    assert "nothing to remove" in out
    assert "would remove kid-old-0001" in out
    assert "note: delegated cleanup is off in config; this is a preview only" in out
    assert "will remove kid-old-0001" not in out
    assert (store / "sessions" / "kid-old-0001").exists()
