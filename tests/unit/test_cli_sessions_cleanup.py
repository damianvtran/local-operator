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

from local_operator.cli import (
    _CLEANUP_TEXT_ROWS,
    _content_row_text,
    sessions_cleanup_command,
)
from local_operator.config import ConfigManager
from local_operator.session import delegated_retention as dr
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
        scratchpads_only=False,
        report_non_git_copies=False,
        reference_repo=None,
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


def test_the_kept_listing_and_json_are_both_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The kept names are CAPPED, so the text must not promise a full ``--json``.

    The pass names at most ``WINDOW_NAMES_CAP`` kept-by-origin sessions (the
    totals line still carries the true count), and the printer shows at most
    ``_CLEANUP_TEXT_ROWS`` of those. Past the cap the old line said ``--json
    lists every row``, which was false: JSON carries the same capped list. One
    600-session store pins the cap and both truncations at once.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    mark_store(tmp_path / "sessions")
    for index in range(600):
        _delegated(tmp_path, f"k{index:03d}", age_h=40 * 24, origin="agent-config")

    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    printed = [line for line in out.splitlines() if line.startswith("  kept ")]
    assert len(printed) == _CLEANUP_TEXT_ROWS
    # The remainder counts against the NAMED list the printer was handed
    # (500), not the store total: 600 kept, 500 named, 200 printed, 300 more.
    assert f"... and {dr.WINDOW_NAMES_CAP - _CLEANUP_TEXT_ROWS} more kept" in out
    assert f"capped at {dr.WINDOW_NAMES_CAP} per group" in out
    assert "--json lists every row" not in out

    assert sessions_cleanup_command(_args(dry_run=True, json=True)) == 0
    delegated = json.loads(capsys.readouterr().out)["delegated"]
    assert delegated["kept_by_origin"] == 600
    assert len(delegated["kept_by_origin_names"]) == dr.WINDOW_NAMES_CAP
    assert len(delegated["protected"]) == dr.WINDOW_NAMES_CAP
    assert {entry["guard"] for entry in delegated["protected"]} == {"origin not reapable"}


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


# -- the scratchpad-content block --------------------------------------------


def _content_pad(store: Path, name: str) -> Path:
    """A scratchpad for a delegated session the RECORD pass keeps.

    The keep comes from an active parent with a roster edge, so the delegated
    rows stay empty and these tests exercise the content carve-out, not a
    record removal.
    """
    sessions = store / "sessions"
    parent = sessions / f"parent-{name}"
    parent.mkdir()
    (parent / "transcript.jsonl").write_text('{"type":"message"}\n')
    (parent / dr.ROSTER_NAME).write_text(
        json.dumps({"version": 1, "records": [{"session_dir": str(sessions / name)}]})
    )
    directory = _delegated(store, name)
    pad = directory / "scratchpad"
    pad.mkdir()
    return pad


def _big_log(path: Path, *, size_mb: int = 11, age_h: float = 24) -> Path:
    """A sparse file of roughly ``size_mb`` MiB, last modified ``age_h`` ago."""
    with open(path, "wb") as handle:
        handle.seek(size_mb * 1024 * 1024 - 1)
        handle.write(b"\0")
    stamp = time.time() - age_h * 3600
    os.utime(path, (stamp, stamp))
    return path


def test_dry_run_shows_the_scratchpad_content_block(store: Path, capsys: Any) -> None:
    pad = _content_pad(store, "content-kid-01")
    big = _big_log(pad / "desktop-suite.log")
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "scratchpad content: would reclaim 1 pad(s)" in out
    assert "(1 stale output)" in out
    assert "content-kid-01" in out and "(stale output; 1 entry)" in out
    assert big.exists(), "a dry run removes nothing"


def test_json_carries_the_scratchpad_content_block(store: Path, capsys: Any) -> None:
    pad = _content_pad(store, "content-kid-02")
    big = _big_log(pad / "desktop-suite.log")
    assert sessions_cleanup_command(_args(dry_run=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    scratch = payload["delegated"]["scratchpad"]
    assert scratch["enabled"] is True
    assert [row["session"] for row in scratch["reclaimed"]] == ["content-kid-02"]
    assert scratch["classes"] == {"stale output": 1}
    assert scratch["bytes"] == big.stat().st_size
    assert scratch["kept"] == [] and scratch["remaining"] == 0
    assert payload["delegated"]["removed"] == []  # no record rows: nothing removed


def test_a_content_only_run_confirms_with_its_own_sentence_and_reclaims(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _content_pad(store, "content-kid-03")
    big = _big_log(pad / "desktop-suite.log")
    big_size = big.stat().st_size
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert sessions_cleanup_command(_args()) == 0
    out = capsys.readouterr().out
    assert (
        "Reclaim build/cache content from 1 session(s)? Transcripts and conversation "
        "records will be kept. Type 'yes' to confirm"
    ) in out
    assert not big.exists()
    assert (
        "Reclaimed scratchpad content from 1 session(s); no transcripts or "
        "conversation records were deleted."
    ) in out
    rows = [
        json.loads(line)
        for line in (store / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["policy"] == "delegated_scratchpad"
    assert rows[0]["session"] == "content-kid-03" and rows[0]["actor"] == "cli"
    # The CLI is a MANUAL trigger, and its row points at a path-level manifest.
    assert rows[0]["trigger"] == "manual" and rows[0]["paths"] == 1
    manifest = json.loads(Path(rows[0]["manifest"]).read_text())
    assert manifest["actor"] == "cli" and manifest["trigger"] == "manual"
    assert manifest["session"] == "content-kid-03"
    assert manifest["completed"] == [
        {"path": str(big), "bytes": big_size, "class": "stale output", "method": "remove"}
    ]


def test_a_recently_touched_pad_is_kept_by_the_content_phase(store: Path, capsys: Any) -> None:
    """The restored age gate, through the CLI: a live-adjacent pad is not ours."""
    pad = _content_pad(store, "content-kid-08")
    big = _big_log(pad / "desktop-suite.log")
    transcript = store / "sessions" / "content-kid-08" / "transcript.jsonl"
    stamp = time.time() - 3600
    os.utime(transcript, (stamp, stamp))
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "scratchpad content: would reclaim 0 pad(s)" in out
    assert "content kept content-kid-08  (inside the retention window)" in out
    assert sessions_cleanup_command(_args(yes=True)) == 0
    assert big.exists(), "a pad touched inside the window keeps its content"
    assert "Reclaimed scratchpad content" not in capsys.readouterr().out


def test_the_scratchpad_switch_off_reclaims_nothing_and_says_so(store: Path, capsys: Any) -> None:
    _config(store, delegated={"scratchpad": {"enabled": False}})
    pad = _content_pad(store, "content-kid-04")
    big = _big_log(pad / "desktop-suite.log")
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "scratchpad content: off (session.cleanup.delegated.scratchpad.enabled is off" in out
    assert sessions_cleanup_command(_args(yes=True)) == 0
    assert big.exists(), "the switch gates only the content phase"
    assert "nothing to remove" in capsys.readouterr().out


def test_force_runs_the_content_phase_when_the_scratchpad_switch_is_off(
    store: Path,
) -> None:
    _config(store, delegated={"scratchpad": {"enabled": False}})
    pad = _content_pad(store, "content-kid-05")
    big = _big_log(pad / "desktop-suite.log")
    assert sessions_cleanup_command(_args(force=True, yes=True)) == 0
    assert not big.exists()


def test_a_worktree_row_says_its_removal_is_a_git_worktree_remove() -> None:
    row = dr.ContentRow(
        session="abcdef123456",
        reason="merged-clean clone past the window",
        bytes=1024,
        entries=3,
        classes=("merged-clean clone",),
        method="worktree-remove",
    )
    assert "; worktree remove" in _content_row_text(row, "would reclaim")
    plain = dr.ContentRow(
        session="abcdef123456",
        reason="stale output past the window",
        bytes=1024,
        entries=1,
        classes=("stale output",),
    )
    assert "worktree remove" not in _content_row_text(plain, "will reclaim")


def test_a_two_bundle_row_lists_both_rescues() -> None:
    row = dr.ContentRow(
        session="abcdef123456",
        reason="merged-clean clone past the window",
        bytes=2048,
        entries=6,
        classes=("merged-clean clone",),
        rescue_bundle="/pads/reap-rescue-clone-a.bundle",
        rescue_bundle_bytes=11,
        rescue_bundles=("/pads/reap-rescue-clone-a.bundle", "/pads/reap-rescue-clone-b.bundle"),
    )
    text = _content_row_text(row, "would reclaim")
    assert "rescue bundles reap-rescue-clone-a.bundle, reap-rescue-clone-b.bundle" in text


def test_apply_confines_content_to_the_previewed_pads(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import cli as cli_module

    pad_x = _content_pad(store, "content-kid-06")
    big_x = _big_log(pad_x / "desktop-suite.log")
    grown: list[Path] = []
    original = cli_module._print_content

    def mutate(delegated: Any, verb: str, policy: Any | None = None) -> None:
        original(delegated, verb, policy)
        # A pad that becomes eligible AFTER the preview (the preview's own rows
        # were computed before this runs). The apply must not ride it in.
        if not grown:
            pad_y = _content_pad(store, "content-kid-07")
            grown.append(_big_log(pad_y / "sibling-fail.log"))

    monkeypatch.setattr(cli_module, "_print_content", mutate)
    assert sessions_cleanup_command(_args(yes=True)) == 0
    out = capsys.readouterr().out
    assert "Reclaimed scratchpad content from 1 session(s)" in out
    assert not big_x.exists(), "the previewed pad is reclaimed"
    assert grown and grown[0].exists(), "a pad that became eligible after the preview is not"


# -- --scratchpads-only: the audit, the content plan, and the switches --------


def _pad_cap(root: Path, monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """A configured pad cap with the reader's floor patched to test scale."""
    from local_operator import scratchpad as scratchpad_module

    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_MAX_BYTES_FLOOR", 1024)
    ConfigManager(root).update_config({"session": {"scratchpad": {"max_bytes": value}}})


def test_scratchpads_only_dry_run_prints_the_audit_and_the_plan_only(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode is AUDIT + CONTENT PLAN, not a record preview (addendum; n4a)."""
    pad = _content_pad(store, "sp-dry")
    big = _big_log(pad / "desktop-suite.log")
    (pad / "build").mkdir()
    (pad / "build" / "x.o").write_bytes(b"x" * 4096)

    assert sessions_cleanup_command(_args(dry_run=True, scratchpads_only=True)) == 0
    out = capsys.readouterr().out

    assert "== Scratchpads ==" in out
    assert "session pad(s)" in out and "shapes: build" in out
    assert "== Scratchpad content ==" in out
    assert "would reclaim" in out and "sp-dry" in out
    assert "Your conversations" not in out and "Delegated work" not in out
    assert big.exists(), "a dry run removes nothing"


def test_scratchpads_only_dry_run_json_carries_the_audit_and_the_flag(
    store: Path, capsys: Any
) -> None:
    pad = _content_pad(store, "sp-json")
    _big_log(pad / "desktop-suite.log")

    assert sessions_cleanup_command(_args(dry_run=True, scratchpads_only=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["scratchpads_only"] is True
    assert payload["outcome"] == "dry-run"
    assert payload["scratchpads"]["count"] >= 1
    row = next(row for row in payload["scratchpads"]["pads"] if row["session"] == "sp-json")
    assert row["path"].endswith("/sp-json/scratchpad")
    assert [shape["path"] for shape in row["shapes"]] == []


def test_the_off_sentence_names_the_switch_that_is_actually_off(store: Path, capsys: Any) -> None:
    """QA Q1, both directions: naming the scratchpad switch while the MASTER
    delegated switch is the one off sends the operator to a row already on."""
    pad = _content_pad(store, "switch-kid")
    big = _big_log(pad / "desktop-suite.log")

    _config(store, delegated={"enabled": False, "scratchpad": {"enabled": True}})
    assert sessions_cleanup_command(_args(dry_run=True, scratchpads_only=True)) == 0
    out = capsys.readouterr().out
    assert "session.cleanup.delegated.enabled is off" in out
    assert "scratchpad.enabled is off" not in out

    _config(store, delegated={"enabled": True, "scratchpad": {"enabled": False}})
    assert sessions_cleanup_command(_args(scratchpads_only=True, yes=True)) == 2
    err = capsys.readouterr().err
    assert "session.cleanup.delegated.scratchpad.enabled is off" in err
    assert big.exists()

    _config(store, delegated={"enabled": False, "scratchpad": {"enabled": False}})
    assert sessions_cleanup_command(_args(scratchpads_only=True, yes=True)) == 2
    err = capsys.readouterr().err
    assert "session.cleanup.delegated.enabled and " in err
    assert "are both off" in err


def test_scratchpads_only_yes_reports_confirmed_true(store: Path, capsys: Any) -> None:
    """QA Q2: ``--yes`` confirms the shown plan; null meant "no confirmation"
    and made an applied run indistinguishable from one that was never asked."""
    pad = _content_pad(store, "confirmed-kid")
    _big_log(pad / "desktop-suite.log")

    assert sessions_cleanup_command(_args(scratchpads_only=True, yes=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["outcome"] == "removed"
    assert payload["confirmed"] is True
    assert payload["scratchpads_only"] is True


def test_an_interactive_confirmation_and_a_decline_both_report_their_answer(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _content_pad(store, "typed-kid")
    _big_log(pad / "desktop-suite.log")

    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert sessions_cleanup_command(_args(json=True)) == 2
    declined = json.loads(capsys.readouterr().out)
    assert declined["confirmed"] is False

    confirmed_stdin = io.StringIO("yes\n")
    monkeypatch.setattr("sys.stdin", confirmed_stdin)
    monkeypatch.setattr(confirmed_stdin, "isatty", lambda: True)
    assert sessions_cleanup_command(_args(json=True)) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["confirmed"] is True


def test_an_over_cap_audit_row_says_where_the_rest_of_the_bytes_are(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX U1: the shapes are a vocabulary, not an accounting. The row names the
    pad and what the shapes do NOT explain, from the totals already measured."""
    pad = _content_pad(store, "over-kid")
    (pad / "bulk.dat").write_bytes(b"x" * 8192)
    (pad / "build").mkdir()
    (pad / "build" / "x.o").write_bytes(b"x" * 4096)
    _pad_cap(store, monkeypatch, 4095)

    assert sessions_cleanup_command(_args(dry_run=True, scratchpads_only=True)) == 0
    out = capsys.readouterr().out

    assert "[OVER CAP" in out
    assert f"other content: inspect {pad} with du; shapes account for" in out
    assert "shapes account for 4.0K" in out


def test_the_ordinary_dry_run_points_at_the_size_view_before_the_kept_list(
    store: Path, capsys: Any
) -> None:
    """UX U4: the kept list can be thousands of rows; the size view must be
    named above it, not after it."""
    _content_pad(store, "pointer-kid")

    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out

    pointer = out.index("size by pad: lop sessions cleanup --scratchpads-only --dry-run")
    kept = out.index("  kept ")
    assert pointer < kept
    assert "Scratchpads section below" in out


# --------------------------------------------------------------------------
# ``--report-non-git-copies``: the read-only class report
# --------------------------------------------------------------------------

_GIT_ENV = {
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(path: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT_ENV},
    )


def _copy_session(
    root: Path, *, size_mb: int = 55, age_hours: float = 40 * 24
) -> tuple[Path, Path]:
    """A published reference repo plus one delegated session holding a copy."""
    import shutil
    import subprocess

    reference = root / "copy-reference"
    reference.mkdir()
    _git(reference, "init", "-q", "-b", "main")
    with open(reference / "payload", "wb") as handle:
        handle.seek(size_mb * 1024 * 1024 - 1)
        handle.write(b"\0")
    origin = root / "copy-origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT_ENV},
    )
    _git(reference, "remote", "add", "origin", str(origin))
    _git(reference, "add", "-A")
    _git(reference, "commit", "-qm", "content")
    _git(reference, "push", "-q", "origin", "HEAD:main")
    _git(reference, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")

    session = root / "sessions" / "copycat"
    session.mkdir()
    (session / "origin.json").write_text('{"origin": "subagent"}')
    transcript = session / "transcript.jsonl"
    transcript.write_text('{"type": "message"}\n')
    old = time.time() - age_hours * 3600
    os.utime(transcript, (old, old))
    copy = session / "scratchpad" / "qa" / "base-repo"
    copy.mkdir(parents=True)
    shutil.copyfile(reference / "payload", copy / "payload")
    return reference, copy


def test_the_copy_report_lists_the_class_and_removes_nothing(store: Path, capsys: Any) -> None:
    reference, copy = _copy_session(store)

    assert (
        sessions_cleanup_command(
            _args(dry_run=True, report_non_git_copies=True, reference_repo=[str(reference)])
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "== Non-git copies (read-only) ==" in out
    assert "1 candidate(s)" in out
    assert "would reclaim now 1 pad(s)" in out
    assert "would reclaim  copycat" in out and "base-repo" in out
    assert (copy / "payload").exists(), "the report removes nothing"


def test_the_copy_report_json_carries_every_row(store: Path, capsys: Any) -> None:
    reference, _copy = _copy_session(store)

    assert (
        sessions_cleanup_command(
            _args(
                dry_run=True,
                report_non_git_copies=True,
                reference_repo=[str(reference)],
                json=True,
            )
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    report = payload["non_git_copies"]
    assert report["count"] == 1 and report["would_reclaim_now"] == 1
    assert report["candidates"][-1]["proved_by"] == [os.path.realpath(reference)]
    assert payload["outcome"] == "report"


def test_the_copy_report_without_a_dry_run_is_refused(store: Path, capsys: Any) -> None:
    assert sessions_cleanup_command(_args(report_non_git_copies=True)) == 2
    assert "--report-non-git-copies only reports" in capsys.readouterr().err
    assert _count(store) == 18


def test_the_normal_dry_run_says_the_copy_arm_is_off(store: Path, capsys: Any) -> None:
    assert sessions_cleanup_command(_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert (
        "non-git copies: off (session.cleanup.delegated.non_git_copies.enabled is off; "
        "--report-non-git-copies previews the class)" in out
    )


def test_the_copy_report_exits_3_when_a_candidate_could_not_be_scanned(
    store: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One broken candidate is loud, but the rest of the report still prints."""
    report = dr.NonGitCopyReport(errors=1)
    report.candidates.append(
        dr.NonGitCopyCandidate(
            session="broken",
            path="broken/scratchpad/copy",
            error=True,
            reason="walk produced zero files from a non-empty tree (4096 bytes)",
        )
    )
    monkeypatch.setattr(dr, "scan_non_git_copies", lambda *args, **kwargs: report)

    assert sessions_cleanup_command(_args(dry_run=True, report_non_git_copies=True)) == 3

    captured = capsys.readouterr()
    assert "1 candidate(s) could not be scanned" in captured.err
    assert "  error" in captured.out and "zero files" in captured.out


def test_the_copy_report_honours_the_delegated_window_override(store: Path, capsys: Any) -> None:
    """A 10 h pad is inside 48 h and past 2 h: the override IS the preview."""
    reference, _copy = _copy_session(store, age_hours=10)

    assert (
        sessions_cleanup_command(
            _args(
                dry_run=True,
                report_non_git_copies=True,
                reference_repo=[str(reference)],
                json=True,
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)["non_git_copies"]
    assert payload["window_hours"] == 48
    assert payload["would_reclaim_now"] == 0
    assert payload["proof_ok_within_window"] == 1

    assert (
        sessions_cleanup_command(
            _args(
                dry_run=True,
                report_non_git_copies=True,
                reference_repo=[str(reference)],
                delegated_max_age_hours=2,
                json=True,
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)["non_git_copies"]
    assert payload["window_hours"] == 2
    assert payload["would_reclaim_now"] == 1
    assert payload["proof_ok_within_window"] == 0
