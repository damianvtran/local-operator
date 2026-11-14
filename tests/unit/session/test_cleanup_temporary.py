"""The temporary-chat pass in ``session/cleanup.py``: who expires, who is held.

The pass is the one automatic path that deletes a conversation the user can see
in a live window, so every cell here states the direction it protects:
expiry happens only for an exact mark, idle past its window, with nothing
holding it — and with ``session.cleanup.enabled`` OFF, because the mark is the
opt-in.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from local_operator.session import cleanup
from local_operator.session import temporary as temp
from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    TEMPORARY_PASS_CAP,
    TEMPORARY_POLICY,
    CleanupPolicy,
    _machine_slept,
    mark_store,
    run_temporary_pass,
)

NOW = 1_000_000_000.0
WINDOW_S = 30 * 60.0
A = "a" * 12
B = "b" * 12

#: Every switch OFF that could matter: the pass must not depend on any of them.
OFF = CleanupPolicy(enabled=False, delegated_enabled=False, temporary_idle_minutes=30)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    mark_store(sessions)
    return tmp_path


def _session(
    root: Path,
    session_id: str,
    *,
    idle_s: float = 0.0,
    origin: str | None = None,
    mark_age_s: float | None = None,
) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / "transcript.jsonl"
    transcript.write_text('{"type":"message"}\n', encoding="utf-8")
    stamp = NOW - idle_s
    os.utime(transcript, (stamp, stamp))
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}), encoding="utf-8")
    if mark_age_s is not None:
        assert temp.write_mark(root, session_id, True, by="cli", now=NOW - mark_age_s).ok
    return directory


def _run(root: Path, **kwargs: object) -> cleanup.TemporaryPassResult:
    return run_temporary_pass(root, OFF, actor="test", now=NOW, **kwargs)  # type: ignore[arg-type]


def _log_rows(root: Path) -> list[dict[str, object]]:
    path = root / "sessions" / CLEANUP_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- expiry --------------------------------------------------------------------


def test_an_idle_marked_chat_expires_with_every_switch_off(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 5, mark_age_s=WINDOW_S + 10)
    result = _run(root)
    assert result.expired == [A]
    assert not (root / "sessions" / A).exists()


def test_recent_keep_does_not_apply(root: Path) -> None:
    """The ONLY session in the store is the most recent one, and it still expires."""
    _session(root, A, idle_s=WINDOW_S + 5, mark_age_s=WINDOW_S + 10)
    assert _run(root).expired == [A]


def test_a_chat_inside_its_window_is_kept(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S - 60, mark_age_s=WINDOW_S + 10)
    assert _run(root).expired == []
    assert (root / "sessions" / A).is_dir()


def test_the_window_runs_from_the_mark_when_it_is_newer(root: Path) -> None:
    _session(root, A, idle_s=10 * WINDOW_S, mark_age_s=60)
    assert _run(root).expired == []


def test_the_configured_window_is_honoured(root: Path) -> None:
    _session(root, A, idle_s=16 * 60, mark_age_s=16 * 60)
    policy = CleanupPolicy(enabled=False, delegated_enabled=False, temporary_idle_minutes=15)
    assert run_temporary_pass(root, policy, actor="t", now=NOW).expired == [A]


def test_a_stub_policy_runs_nothing(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 5, mark_age_s=WINDOW_S + 10)
    policy = cleanup.policy_from_config(object())
    result = run_temporary_pass(root, policy, actor="t", now=NOW)
    assert result.skipped == "disabled" and (root / "sessions" / A).is_dir()


def test_unmarked_kept_and_foreign_sessions_are_never_removed(root: Path) -> None:
    _session(root, A, idle_s=10 * WINDOW_S)  # unmarked
    kept = _session(root, B, idle_s=10 * WINDOW_S, mark_age_s=10 * WINDOW_S)
    assert temp.write_mark(root, B, False, by="cli").ok  # Keep
    child = _session(root, "c" * 12, idle_s=10 * WINDOW_S, origin="subagent")
    (child / temp.MARK_NAME).write_text(
        json.dumps({"v": 1, "temporary": True, "session_id": "c" * 12, "marked_at": 1.0})
    )
    assert _run(root).expired == []
    assert kept.is_dir() and child.is_dir() and (root / "sessions" / A).is_dir()


def test_an_unmarked_store_is_not_written_to(root: Path) -> None:
    for index in range(50):
        _session(root, f"{index:012x}", idle_s=10 * WINDOW_S)
    before = sorted(p.name for p in (root / "sessions").iterdir())
    result = _run(root)
    assert result.expired == [] and result.skipped == "nothing marked"
    assert sorted(p.name for p in (root / "sessions").iterdir()) == before
    assert not (root / cleanup.TEMPORARY_LOCK_NAME).exists()


def test_only_expired_marked_unguarded_ids_go_in_a_disabled_store(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant's companion: with ``enabled: false`` the pass removes ONLY
    the expired, marked, unguarded ids out of a 50-session store."""
    expired = set()
    for index in range(50):
        sid = f"{index:012x}"
        if index % 5 == 0:
            _session(root, sid, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
            expired.add(sid)
        elif index % 5 == 1:
            _session(root, sid, idle_s=60, mark_age_s=WINDOW_S + 60)  # inside window
        else:
            _session(root, sid, idle_s=10 * WINDOW_S)  # unmarked
    guarded = "000000000005"
    monkeypatch.setattr(cleanup, "_has_armed_wake", lambda _root, session: session == guarded)
    result = _run(root)
    assert set(result.expired) == expired - {guarded}
    remaining = {p.name for p in (root / "sessions").iterdir() if p.is_dir()}
    assert remaining == {f"{i:012x}" for i in range(50)} - (expired - {guarded})


def test_the_pass_is_capped(root: Path) -> None:
    for index in range(TEMPORARY_PASS_CAP + 3):
        _session(root, f"{index:012x}", idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    assert len(_run(root).expired) == TEMPORARY_PASS_CAP


def test_should_stop_ends_the_pass(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    _session(root, B, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    calls = iter([False, True])
    result = _run(root, should_stop=lambda: next(calls))
    assert result.expired == [A] and (root / "sessions" / B).is_dir()


def test_dry_run_removes_nothing_and_writes_nothing(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    result = _run(root, dry_run=True)
    assert result.expired == [A]
    assert (root / "sessions" / A).is_dir()
    assert not (root / "sessions" / temp.TOMBSTONES_NAME).exists()
    assert not (root / "sessions" / temp.STATE_NAME).exists()


# -- the record the pass leaves ----------------------------------------------------


def test_the_log_row_has_no_title_and_the_tombstone_lands(root: Path) -> None:
    directory = _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    (directory / "title.json").write_text(json.dumps({"title": "secret plans"}))
    _run(root)
    rows = [row for row in _log_rows(root) if row.get("session") == A]
    assert rows and rows[-1]["policy"] == TEMPORARY_POLICY
    assert "secret plans" not in json.dumps(rows)
    assert not rows[-1].get("title")
    assert temp.deleted_at(root, A, now=NOW) == NOW


def test_the_tombstone_is_written_before_removal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    seen: list[bool] = []
    real = cleanup.remove_session_dir

    def spy(target: Path, **kwargs: object) -> bool:
        seen.append(any(e["id"] == A for e in temp.read_tombstones(root, now=NOW)))
        return real(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cleanup, "remove_session_dir", spy)
    _run(root)
    assert seen == [True]


def test_a_failed_tombstone_keeps_the_chat(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, "_append_temporary_tombstone", lambda *_a: False)
    result = _run(root)
    assert result.expired == [] and result.errors == 1
    assert (root / "sessions" / A).is_dir()


# -- holds ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probe", "code"),
    [
        ("_has_armed_wake", "armed_wake"),
        ("_has_armed_monitor", "armed_monitor"),
    ],
)
def test_guard_holds_map_to_codes(
    root: Path, monkeypatch: pytest.MonkeyPatch, probe: str, code: str
) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, probe, lambda *_a: True)
    result = _run(root)
    assert result.expired == [] and result.held == {A: code}
    state = temp.read_state(root)
    assert state["sessions"][A] == {"held_at": NOW, "hold": code}


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        ("held by the operator", "operator_hold"),
        ("has unread spooled mail", "unread_mail"),
        ("has an open ask", "open_ask"),
        ("operator holds unreadable", "cannot_verify"),
        ("guard failed: OSError", "cannot_verify"),
    ],
)
def test_every_guard_reason_has_a_code(
    root: Path, monkeypatch: pytest.MonkeyPatch, reason: str, code: str
) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, "_guard", lambda *_a, **_k: reason)
    assert _run(root).held == {A: code}


def test_a_desktop_on_screen_chat_is_held(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.session.runtime import presence

    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(presence, "desktop_attending_session", lambda sid, _root=None: sid == A)
    assert _run(root).held == {A: "on_screen"}


def test_a_pinned_chat_is_held_as_linked(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    (root / "sidebar-pins.json").write_text(json.dumps([A]))
    assert _run(root).held == {A: "linked"}


def test_an_unreadable_pin_file_holds_as_cannot_verify(root: Path) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    (root / "sidebar-pins.json").write_text("{not json")
    assert _run(root).held == {A: "cannot_verify"}


def test_an_in_flight_child_holds_and_an_unreadable_roster_cannot_verify(root: Path) -> None:
    directory = _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    roster = directory / "subagent-roster.v1.json"
    roster.write_text(json.dumps({"records": [{"session_dir": "/x/sessions/" + B}]}))
    assert _run(root).held == {A: "running"}
    roster.write_text("{bad")
    assert _run(root).held == {A: "cannot_verify"}


def test_held_at_restarts_the_window(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, "_has_armed_wake", lambda *_a: True)
    assert _run(root).held == {A: "armed_wake"}
    monkeypatch.setattr(cleanup, "_has_armed_wake", lambda *_a: False)
    # The hold just ended: the window restarts from the last held pass.
    later = run_temporary_pass(root, OFF, actor="t", now=NOW + 60)
    assert later.expired == [] and (root / "sessions" / A).is_dir()
    assert temp.read_state(root)["sessions"][A] == {"held_at": NOW, "hold": None}
    expired = run_temporary_pass(root, OFF, actor="t", now=NOW + WINDOW_S + 1)
    assert expired.expired == [A]


def test_state_prunes_ids_no_longer_marked(root: Path) -> None:
    _session(root, A, idle_s=60, mark_age_s=60)
    _run(root)
    assert A in temp.read_state(root)["sessions"]
    temp.write_mark(root, A, False, by="cli")
    _run(root)
    assert temp.read_state(root)["sessions"] == {}


# -- the idle keep-alive resident ------------------------------------------------------


def _resident(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, *, leaves: bool, on_leave=None
):
    """A claim held only by an idle keep-alive runtime, via the real call sites."""
    state = {"gone": False, "asked": 0, "withdrawn": 0}

    def guard(  # type: ignore[no-untyped-def]
        directory: Path, config_dir: Path, now: float, *, live_owner_gone: bool = False
    ):
        if live_owner_gone or state["gone"]:
            return None
        return "claimed by a live process"

    def ask(record: object, directory: Path, *, actor: str) -> bool:
        state["asked"] += 1
        if on_leave is not None:
            on_leave(directory)
        state["gone"] = leaves
        return leaves

    from local_operator.session.runtime import registry

    monkeypatch.setattr(cleanup, "_guard", guard)
    monkeypatch.setattr(cleanup, "_keep_alive_resident", lambda *_a: object())
    monkeypatch.setattr(cleanup, "_ask_keep_alive_to_leave", ask)
    monkeypatch.setattr(
        registry, "remove_exit_request", lambda _d: state.__setitem__("withdrawn", 1)
    )
    return state


def test_an_idle_resident_is_asked_to_leave_once_the_window_elapsed(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    state = _resident(monkeypatch, leaves=True)
    assert _run(root).expired == [A]
    assert state["asked"] == 1


def test_a_warm_resident_inside_the_window_is_never_asked(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(root, A, idle_s=60, mark_age_s=60)
    state = _resident(monkeypatch, leaves=True)
    assert _run(root).expired == []
    assert state["asked"] == 0


def test_a_resident_that_stays_holds_as_running(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    _resident(monkeypatch, leaves=False)
    result = _run(root)
    assert result.expired == [] and result.held == {A: "running"}


def test_a_keep_during_the_preempt_wins(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    _resident(
        monkeypatch,
        leaves=True,
        on_leave=lambda d: temp.write_mark(root, d.name, False, by="desktop"),
    )
    assert _run(root).expired == []
    assert (root / "sessions" / A).is_dir()


def test_a_busy_or_watched_runtime_holds(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, "_guard", lambda *_a, **_k: "leased by a live process")
    monkeypatch.setattr(cleanup, "_keep_alive_resident", lambda *_a: None)
    monkeypatch.setattr(cleanup, "_temporary_live_hold", lambda *_a: "on_screen")
    assert _run(root).held == {A: "on_screen"}


# -- cascade -----------------------------------------------------------------------------


def _roster(directory: Path, *children: tuple[str, bool]) -> None:
    records = [
        {"session_dir": f"/x/sessions/{child}", "settled_at": None if live else 1.0}
        for child, live in children
    ]
    (directory / "subagent-roster.v1.json").write_text(json.dumps({"records": records}))


def test_settled_subagent_children_go_with_the_chat(root: Path) -> None:
    parent = _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    sub = _session(root, "1" * 12, idle_s=WINDOW_S, origin="subagent")
    grandchild = _session(root, "2" * 12, idle_s=WINDOW_S, origin="subagent")
    workstream = _session(root, "3" * 12, idle_s=WINDOW_S, origin="agent-workstream")
    fork = _session(root, "4" * 12, idle_s=WINDOW_S, origin="fork")
    _roster(parent, ("1" * 12, False), ("3" * 12, False), ("4" * 12, False))
    _roster(sub, ("2" * 12, False))
    result = _run(root)
    assert result.expired == [A]
    assert sorted(result.children) == ["1" * 12, "2" * 12]
    assert not sub.exists() and not grandchild.exists()
    assert workstream.is_dir() and fork.is_dir()
    policies = {row["session"]: row["policy"] for row in _log_rows(root)}
    assert policies["1" * 12] == "temporary-child"


def test_a_child_with_its_own_live_child_is_left(root: Path) -> None:
    parent = _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    sub = _session(root, "1" * 12, idle_s=WINDOW_S, origin="subagent")
    _roster(parent, ("1" * 12, False))
    _roster(sub, ("2" * 12, True))
    assert _run(root).children == []
    assert sub.is_dir()


# -- hosting -------------------------------------------------------------------------------


def test_the_suspend_guard() -> None:
    assert not _machine_slept(60.0, 60.0, 60.0)
    assert not _machine_slept(150.0, 60.0, 60.0)
    assert _machine_slept(3600.0, 60.0, 60.0)


def test_the_tick_skips_a_fresh_pass_and_an_empty_store(root: Path) -> None:
    provider = lambda: OFF  # noqa: E731
    assert cleanup._temporary_tick(root, provider, actor="t") is None
    assert not (root / cleanup.TEMPORARY_LOCK_NAME).exists()
    _session(root, A, idle_s=time.time() - NOW + WINDOW_S + 60, mark_age_s=None)
    assert temp.write_mark(root, A, True, by="cli", now=time.time() - WINDOW_S - 60).ok
    first = cleanup._temporary_tick(root, provider, actor="t")
    assert first is not None and first.expired == [A]
    # Fresh state: the next tick in the same minute does nothing.
    _session(root, B, idle_s=time.time() - NOW + WINDOW_S + 60)
    assert temp.write_mark(root, B, True, by="cli", now=time.time() - WINDOW_S - 60).ok
    assert cleanup._temporary_tick(root, provider, actor="t") is None


def test_start_is_idempotent(tmp_path: Path) -> None:
    class Manager:
        def get_nested_value(self, _path: object, default: object) -> object:
            return default

    try:
        assert cleanup.start_temporary_sweeper(Manager(), tmp_path)
        assert not cleanup.start_temporary_sweeper(Manager(), tmp_path)
    finally:
        cleanup.stop_temporary_sweeper()


def test_the_per_test_store_maintenance_reset_stops_the_sweeper(tmp_path: Path) -> None:
    # Session runtimes start the sweeper from `_start_store_maintenance`; the
    # autouse reset must stop it, or it leaks into later tests in the worker
    # (CI gw2: test_start_is_idempotent saw False after a session-factory test).
    from local_operator.session_factory import reset_store_maintenance_for_tests

    class Manager:
        def get_nested_value(self, _path: object, default: object) -> object:
            return default

    try:
        assert cleanup.start_temporary_sweeper(Manager(), tmp_path)
        thread = cleanup._TEMPORARY_SWEEPER["thread"]
        reset_store_maintenance_for_tests()
        assert not thread.is_alive()
        assert cleanup._TEMPORARY_SWEEPER == {}
    finally:
        cleanup.stop_temporary_sweeper()


def test_a_keep_landing_after_the_tombstone_still_wins_and_withdraws_it(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # C-8 (review R1-m3): the Keep re-read is the LAST read before the
    # remover, after the roster and tombstone, and the tombstone written for a
    # chat the user then kept is withdrawn rather than left to be believed.
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    real = cleanup._append_temporary_tombstone

    def tombstone_then_keep(config_dir: Path, session_id: str, now: float) -> bool:
        landed = real(config_dir, session_id, now)
        assert temp.write_mark(root, session_id, False, by="desktop").ok
        return landed

    monkeypatch.setattr(cleanup, "_append_temporary_tombstone", tombstone_then_keep)
    result = _run(root)
    assert result.expired == []
    assert (root / "sessions" / A).is_dir()
    assert all(entry["id"] != A for entry in temp.read_tombstones(root, now=NOW))


def test_a_failed_tombstone_withdraw_leaves_a_debug_trace(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Review r2 N-1: the withdraw's False was ignored, so a failure left no trace.
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    real = cleanup._append_temporary_tombstone

    def tombstone_then_keep(config_dir: Path, session_id: str, now: float) -> bool:
        landed = real(config_dir, session_id, now)
        assert temp.write_mark(root, session_id, False, by="desktop").ok
        return landed

    monkeypatch.setattr(cleanup, "_append_temporary_tombstone", tombstone_then_keep)
    monkeypatch.setattr(cleanup, "_withdraw_temporary_tombstone", lambda *_a: False)
    with caplog.at_level(logging.DEBUG, logger=cleanup.__name__):
        result = _run(root)
    assert result.expired == []
    assert f"tombstone for kept {A} not withdrawn" in caplog.text


def test_an_open_ask_outranks_the_wake_it_armed(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # C-12 (QA Q5): the ``ask`` tool arms a hidden deadline wake, and the guard
    # probes wakes first, so an ask-held chat said "kept while a wake is set".
    _session(root, A, idle_s=WINDOW_S + 60, mark_age_s=WINDOW_S + 60)
    monkeypatch.setattr(cleanup, "_has_armed_wake", lambda *_a: True)
    monkeypatch.setattr(cleanup, "_has_open_asks", lambda _d: True)
    result = _run(root)
    assert result.held == {A: "open_ask"}
    # Control: a wake with no open ask keeps its own word.
    monkeypatch.setattr(cleanup, "_has_open_asks", lambda _d: False)
    assert _run(root).held == {A: "armed_wake"}
