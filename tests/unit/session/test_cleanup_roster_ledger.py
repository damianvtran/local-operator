"""The roster is a LEDGER, not a resumption set; pins are never reaped; the dry run streams.

Evidence for the three changes in this slice, all over ``cleanup_store_fixture``'s
synthetic store (the real on-disk shapes, one directory per class):

* rule 3 protects a roster child only while its record is IN FLIGHT or PAUSED
  (``settled_at`` absent/unreadable, or ``paused`` truthy) — a settled child is
  history, so the 48 h window can finally reach old subagents;
* a pinned session is never reaped by either layer of the delegated pass;
* ``run_delegated_pass`` / ``run_cleanup`` announce each decision as it is made
  and ``audit_scratchpads`` honours a budget, without changing a decision.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import delegated_retention as dr
from local_operator.session.cleanup import CLEANUP_LOG_NAME, CleanupPolicy, run_cleanup
from local_operator.session.scratchpad_audit import audit_scratchpads
from tests.unit.session.cleanup_store_fixture import (
    HOUR,
    Store,
    _record,
    _roster,
    _session,
    build_store,
)

ON = CleanupPolicy(delegated_enabled=True, delegated_max_age_hours=48)


def _provider():
    return lambda: ON


def _pass(store: Store, **kwargs: Any) -> dr.DelegatedResult:
    kwargs.setdefault("now", store.now)
    kwargs.setdefault("removal_pause_s", 0)
    dr._FILE_CACHE.clear()
    return dr.run_delegated_pass(store.root, _provider(), **kwargs)


@pytest.fixture
def store(tmp_path: Path) -> Any:
    built = build_store(tmp_path)
    try:
        yield built
    finally:
        built.restore()


def _dirs(store: Store) -> set[str]:
    # Dot entries are the store's own sidecars (`.cleanup-manifests/`), not sessions.
    return {p.name for p in store.sessions.iterdir() if p.is_dir() and not p.name.startswith(".")}


def _old_rule_children(store: Store) -> set[str]:
    """The pre-change predicate: EVERY child any roster names, settled or not."""
    names: set[str] = set()
    for roster in store.sessions.glob(f"*/{dr.ROSTER_NAME}"):
        for record in json.loads(roster.read_text())["records"]:
            names.add(os.path.basename(record["session_dir"]))
    return names


# --------------------------------------------------------------------------
# Rule 3: the ledger
# --------------------------------------------------------------------------


def test_the_old_predicate_keeps_every_settled_child_and_the_new_one_frees_them(
    store: Store,
) -> None:
    old = _old_rule_children(store)
    # Pre-change behaviour, derived from the roster itself: every settled child
    # the fixture expects the fix to free WAS named by the roster and so kept.
    assert store.freed_by_the_roster_fix <= old
    assert len(store.freed_by_the_roster_fix) >= 100

    result = _pass(store, dry_run=True)

    removed = {row.session for row in result.removed}
    assert removed == store.removable
    kept = dict(result.protected)
    assert not removed & set(kept)
    # None of the freed children is still reported as "parent session still active".
    assert not [n for n in store.freed_by_the_roster_fix if n in kept]


def test_every_protected_class_survives_with_the_reason_of_its_real_guard(store: Store) -> None:
    result = _pass(store, dry_run=True)
    kept = dict(result.protected)
    for name, expected in store.kept.items():
        assert name in kept, f"{name} was not reported as kept"
        assert expected in kept[name] or (
            expected.startswith("inside the") and "window" in kept[name]
        ), (name, kept[name], expected)


def test_in_flight_and_unreadable_roster_fields_are_kept_without_exception(store: Store) -> None:
    result = _pass(store)
    still = _dirs(store)
    # The one error is the pad nothing can read (`unreadablepad`): kept and counted.
    assert result.errors == 1
    for name in (
        "inflight",
        "nofield",
        "paused",
        "settled-str",
        "settled-bool",
        "settled-nan",
        "twice",
        "inflight-parent",
        "gc-inflight",
        "padheavy",
    ):
        assert name in still, f"{name} (in flight / unreadable ledger row) was removed"


def test_a_paused_child_that_carries_settled_at_is_kept(store: Store) -> None:
    """``paused`` is orthogonal to ``settled_at``: ``hub`` resumes a paused child."""
    record = [r for r in _ledger(store) if r["session_dir"].endswith("/paused")][0]
    assert record["paused"] is True and record["settled_at"] is not None

    result = _pass(store)

    assert (store.sessions / "paused").is_dir()
    assert dict(_pass(store, dry_run=True).protected)["paused"] == "parent session still active"
    assert "paused" not in {row.session for row in result.removed}


def _ledger(store: Store) -> list[dict[str, Any]]:
    return json.loads((store.sessions / "parent" / dr.ROSTER_NAME).read_text())["records"]


def test_a_child_named_by_several_records_is_protected_if_any_is_in_flight(store: Store) -> None:
    kept = dict(_pass(store, dry_run=True).protected)
    assert kept["twice"] == "parent session still active"


@pytest.mark.parametrize(
    ("fields", "protected"),
    [
        ({"settled_at": 1.0}, False),
        ({"settled_at": 1}, False),
        ({"settled_at": 1.0, "paused": False}, False),
        ({"settled_at": 1.0, "paused": True}, True),
        ({"settled_at": None}, True),
        ({}, True),
        ({"settled_at": "1.0"}, True),
        ({"settled_at": True}, True),
        ({"settled_at": float("inf")}, True),
        ({"settled_at": 1.0, "paused": "false"}, True),
        ({"settled_at": 1.0, "paused": 1}, True),
    ],
)
def test_roster_children_protects_exactly_the_actionable_records(
    tmp_path: Path, fields: dict[str, object], protected: bool
) -> None:
    dr._FILE_CACHE.clear()
    roster = tmp_path / dr.ROSTER_NAME
    roster.write_text(json.dumps({"version": 1, "records": [{"session_dir": "/x/kid", **fields}]}))
    assert dr.roster_children(roster) == ({"kid"} if protected else set())


def test_a_non_object_record_or_roster_makes_the_set_unknowable(tmp_path: Path) -> None:
    dr._FILE_CACHE.clear()
    roster = tmp_path / dr.ROSTER_NAME
    for payload in (
        {"version": 1, "records": ["not-a-dict"]},
        {"version": 1, "records": {}},
        [],
        "x",
    ):
        roster.write_text(json.dumps(payload))
        dr._FILE_CACHE.clear()
        with pytest.raises(dr._Unknowable):
            dr.roster_children(roster)
    roster.write_text("{truncated")
    dr._FILE_CACHE.clear()
    with pytest.raises(dr._Unknowable):
        dr.roster_children(roster)


def test_an_unreadable_roster_of_a_live_parent_still_skips_the_whole_pass(store: Store) -> None:
    (store.sessions / "parent" / dr.ROSTER_NAME).write_text('{"records": [1]}')
    result = _pass(store)
    assert result.skipped and result.skipped.startswith("skipped:")
    assert result.removed == []
    assert _dirs(store) >= store.removable  # nothing was removed


def test_compression_asks_the_same_question(store: Store) -> None:
    """``active_parent_sessions`` shares the narrowed source of truth."""
    dr._FILE_CACHE.clear()
    active = dr.active_parent_sessions(store.root, store.now)
    assert not active & store.freed_by_the_roster_fix
    assert {"inflight", "paused", "nofield", "twice"} <= active


def test_transitivity_follows_only_in_flight_children(store: Store) -> None:
    """A settled child's own roster is not read, an in-flight child's is."""
    assert (store.sessions / "gc-settled").is_dir()
    kept = dict(_pass(store, dry_run=True).protected)
    assert kept["gc-inflight"] == "parent session still active"
    assert "gc-settled" not in kept


# --------------------------------------------------------------------------
# Pins
# --------------------------------------------------------------------------


def test_a_pinned_session_is_never_reaped_and_names_the_pin(store: Store) -> None:
    result = _pass(store)
    assert (store.sessions / "pinned").is_dir()
    assert "pinned" not in {row.session for row in result.removed}
    assert dict(_pass(store, dry_run=True).protected)["pinned"] == dr.PINNED_REASON


def test_a_pinned_session_outranks_a_settled_ledger_row(store: Store) -> None:
    """Pinned AND settled-in-a-roster: the pin still wins (and names itself)."""
    _session(store, "pinned-settled")
    ledger = _ledger(store) + [_record(store, "pinned-settled")]
    _roster(store.sessions / "parent", ledger)
    (store.root / "sidebar-pins.json").write_text(json.dumps(["pinned", "pinned-settled"]))
    result = _pass(store)
    assert (store.sessions / "pinned-settled").is_dir()
    assert dict(_pass(store, dry_run=True).protected)["pinned-settled"] == dr.PINNED_REASON
    assert "pinned-settled" not in {row.session for row in result.removed}


def test_a_pinned_pad_keeps_its_content_too(store: Store) -> None:
    pad = _session(store, "pinned")  # re-open the existing pinned directory
    big = dr_stale(pad / "scratchpad" / "desktop-suite.log", store.now)
    result = _pass(store)
    assert big.exists()
    assert ("pinned", dr.PINNED_REASON) in result.content_kept


def dr_stale(path: Path, now: float) -> Path:
    from tests.unit.session.cleanup_store_fixture import _stale_file

    return _stale_file(path, now=now)


@pytest.mark.parametrize("content", ["{truncated", '{"a": 1}', '"pinned"'])
def test_an_unreadable_pin_file_skips_the_pass_and_removes_nothing(
    store: Store, content: str
) -> None:
    (store.root / "sidebar-pins.json").write_text(content)
    before = _dirs(store)
    result = _pass(store)
    assert result.skipped and "sidebar-pins.json" in result.skipped
    assert result.removed == [] and _dirs(store) == before


def test_an_absent_pin_file_means_nothing_is_pinned(store: Store) -> None:
    (store.root / "sidebar-pins.json").unlink()
    result = _pass(store, dry_run=True)
    assert "pinned" in {row.session for row in result.removed}


def test_the_pin_file_is_read_once_per_pass(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tui import sidebar_pins

    calls: list[Path] = []
    real = sidebar_pins.load_pin_ids
    monkeypatch.setattr(
        sidebar_pins, "load_pin_ids", lambda root: (calls.append(root), real(root))[1]
    )
    _pass(store, dry_run=True)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Dry run == real run, receipts, no half-removed directory
# --------------------------------------------------------------------------


def test_the_dry_run_removal_set_is_the_set_the_real_run_removes_on_the_same_copy(
    store: Store, tmp_path_factory: pytest.TempPathFactory
) -> None:
    dry = _pass(store, dry_run=True)
    before = _dirs(store)
    # A dry run writes nothing: no receipt, no manifest, no directory gone.
    assert not (store.sessions / CLEANUP_LOG_NAME).exists()
    assert not (store.sessions / ".cleanup-manifests").exists()
    assert {row.session for row in dry.content} == store.content

    # The same store, copied, then really run. (The unreadable pad is made
    # readable for the copy and locked again, so both runs see the same guard.)
    copy_root = tmp_path_factory.mktemp("copy")
    store.restore()
    shutil.copytree(store.root, copy_root, symlinks=True, dirs_exist_ok=True)
    for locked in store.locked:
        os.chmod(locked, 0o000)
        os.chmod(copy_root / locked.relative_to(store.root), 0o000)
    real_store = Store(root=copy_root, now=store.now)
    try:
        real = _pass(real_store, actor="cli", trigger="manual")
    finally:
        os.chmod(copy_root / store.locked[0].relative_to(store.root), 0o755)

    assert {row.session for row in dry.removed} == {row.session for row in real.removed}
    assert {row.session for row in real.removed} == store.removable
    # Exactly the removal set disappeared: no more, no less, and no husk.
    after = _dirs(real_store)
    assert before - after == store.removable
    assert after - before == set()
    assert store.survives <= after
    # The pad-heavy child's RECORD stays; only its stale output went (content layer).
    assert (copy_root / "sessions" / "padheavy").is_dir()
    assert not (copy_root / "sessions" / "padheavy" / "scratchpad" / "desktop-suite.log").exists()
    assert {row.session for row in real.content} == store.content
    assert not [p for p in (copy_root / "sessions").iterdir() if p.name.endswith(".removing")]

    # One receipt per removal, with actor, age, bytes.
    rows = [
        json.loads(line)
        for line in (copy_root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
        if line
    ]
    session_rows = [r for r in rows if r.get("policy") == dr.DELEGATED_POLICY]
    assert sorted(r["session"] for r in session_rows) == sorted(store.removable)
    assert all(r["actor"] == "cli" and "bytes" in r and r["reason"] for r in session_rows)
    assert all(r.get("last_activity_at") or r.get("at") for r in session_rows)


def test_the_freed_bytes_are_reported_and_a_second_run_removes_nothing(store: Store) -> None:
    first = _pass(store)
    assert {row.session for row in first.removed} == store.removable
    assert first.freed_bytes > 0
    second = _pass(store)
    assert second.removed == []


# --------------------------------------------------------------------------
# Streaming observers
# --------------------------------------------------------------------------


def test_the_observers_see_exactly_the_decisions_the_result_lists_hold(store: Store) -> None:
    planned: list[str] = []
    kept: list[tuple[str, str]] = []
    beats: list[tuple[str, int, int | None, float]] = []
    result = _pass(
        store,
        dry_run=True,
        on_plan=lambda row: planned.append(row.session),
        on_kept=lambda name, reason: kept.append((name, reason)),
        on_progress=lambda *beat: beats.append(beat),
    )
    assert planned == [row.session for row in result.removed]
    assert kept == result.protected
    assert len(planned) == len(set(planned)) and len({n for n, _ in kept}) == len(kept)
    assert any(phase == "scan" for phase, *_ in beats)


def test_a_failing_observer_never_changes_a_decision(store: Store) -> None:
    def boom(*_args: object) -> None:
        raise BrokenPipeError

    plain = _pass(store, dry_run=True)
    noisy = _pass(store, dry_run=True, on_plan=boom, on_kept=boom, on_progress=boom)
    assert [r.session for r in noisy.removed] == [r.session for r in plain.removed]
    assert noisy.protected == plain.protected


def test_the_scan_heartbeat_fires_every_256_directories_not_per_directory(tmp_path: Path) -> None:
    store = Store(root=tmp_path, now=time.time())
    from local_operator.session.cleanup import mark_store

    mark_store(store.sessions)
    for index in range(700):
        _session(store, f"d{index:04d}", age_h=1)
    beats: list[int] = []
    dr._scan(
        store.sessions,
        store.now - 48 * HOUR,
        store.now,
        None,
        None,
        lambda phase, done, total, elapsed: beats.append(done),
    )
    assert beats == [256, 512]


def test_run_cleanup_announces_its_plan_as_it_decides(tmp_path: Path) -> None:
    store = Store(root=tmp_path, now=time.time())
    from local_operator.session.cleanup import mark_store

    mark_store(store.sessions)
    for index in range(5):
        (store.sessions / f"e{index}").mkdir()
        os.utime(store.sessions / f"e{index}", (store.now - 99 * HOUR, store.now - 99 * HOUR))
    held: list[str] = []
    policy = CleanupPolicy(enabled=True, remove_empty=True)
    result = run_cleanup(
        store.root,
        policy,
        dry_run=True,
        now=store.now,
        on_plan=lambda row: held.append(row.session),
    )
    assert held == [row.session for row in result.removed] and len(held) == 5


# --------------------------------------------------------------------------
# The bounded audit
# --------------------------------------------------------------------------


def _pads(root: Path, count: int) -> None:
    for index in range(count):
        pad = root / "sessions" / f"s{index:03d}" / "scratchpad"
        pad.mkdir(parents=True)
        (pad / "f").write_bytes(b"x" * 100)


def test_an_unbudgeted_audit_is_complete_and_its_json_is_unchanged(tmp_path: Path) -> None:
    _pads(tmp_path, 12)
    seen: list[tuple[int, int]] = []
    audit = audit_scratchpads(tmp_path / "sessions", on_pad=lambda _p, d, t: seen.append((d, t)))
    assert len(audit.pads) == 12 and not audit.stopped_early
    assert seen[0] == (1, 12) and seen[-1] == (12, 12)
    assert set(audit.as_dict()) == {"cap", "count", "bytes", "over_cap", "truncated", "pads"}


def test_a_spent_budget_stops_between_pads_and_says_how_many_were_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session import scratchpad_audit as module

    _pads(tmp_path, 12)
    clock = iter(range(1000))
    monkeypatch.setattr(module.time, "monotonic", lambda: float(next(clock)))
    audit = audit_scratchpads(tmp_path / "sessions", budget_s=3.5)
    assert audit.stopped_early and audit.total_pads == 12
    assert 1 <= len(audit.pads) < 12
    payload = audit.as_dict()
    assert payload["budget_exhausted"] is True and payload["total_pads"] == 12
    assert payload["count"] == len(audit.pads)


def test_a_tiny_budget_still_audits_one_pad(tmp_path: Path) -> None:
    _pads(tmp_path, 4)
    audit = audit_scratchpads(tmp_path / "sessions", budget_s=0.0)
    assert len(audit.pads) == 1 and audit.stopped_early
