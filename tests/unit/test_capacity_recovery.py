"""Capacity recovery: the trigger, the safe-GC predicate, and the controller.

Driven over tiny owned roots (real generation directories built the way a prune
sees them) with SIMULATED capacity — never a host disk fill. The subject is the
DECISION: what is provably unreferenced is reclaimed, and everything unknown is
retained.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import capacity_recovery, install_transaction, update as update_mod

DIST = update_mod.DISTRIBUTION_NAME


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated ``$HOME`` so ``stable_root()`` relocates the whole layout."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LOP_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(update_mod, "_fresh_publication", lambda _t, _m: None)
    monkeypatch.setattr(update_mod, "_verify_signed_candidate", lambda _g, _m: None)
    return tmp_path


def _generation(name: str, *, marker: bool = True, age: float = 10_000.0) -> Path:
    """A generation directory with the marker a real build carries.

    The name is ``<stamp>-<token>``-shaped so name order is creation order, as the
    real layout's is; ``marker=False`` reproduces a crashed/in-flight tree.
    """
    import os
    import time

    generation = update_mod.generations_dir() / name
    venv = generation / "tools" / DIST
    venv.mkdir(parents=True)
    if marker:
        (venv / ".lop-source").write_text("pypi 0.0.1\n", encoding="utf-8")
    # Age the tree so an unmarked one is not read as "an install in flight".
    stamp = time.time() - age
    os.utime(generation, (stamp, stamp))
    return generation


def _session_record(config_root: Path, generation: Path, pid: int = 4242) -> None:
    """Publish one live session record naming ``generation``'s venv, as a runtime does.

    ``install_root`` is the VENV (``<gen>/tools/<dist>``), not the generation
    directory — that is the wire shape the prune normalises two levels up from, and
    a fixture that stored the generation here would not exercise that agreement.
    """
    import json

    from local_operator.session.runtime.types import RUN_DIRNAME

    directory = config_root / RUN_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pid}.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "kind": "tui",
                "session_id": f"s{pid}",
                "conversation_name": "c",
                "cwd": "/tmp",
                "model_label": "m",
                "control_port": 0,
                "control_key": "",
                "install_root": str(generation / "tools" / DIST),
                "build": "0.0.1",
            }
        ),
        encoding="utf-8",
    )


# -- the trigger ---------------------------------------------------------------


def test_measure_reports_no_crossing_under_the_floor(home: Path) -> None:
    _generation("20260101T000000Z-a")
    reading = capacity_recovery.measure_capacity(max_generations=8, min_free_bytes=1)
    assert reading.generations == 1
    assert reading.threshold == ""
    assert reading.over is False


def test_the_count_threshold_crosses(home: Path) -> None:
    for index in range(3):
        _generation(f"2026010{index + 1}T000000Z-{index}")
    reading = capacity_recovery.measure_capacity(max_generations=2, min_free_bytes=1)
    assert reading.threshold == "count"
    assert reading.over is True


def test_the_free_threshold_crosses_on_simulated_capacity(home: Path) -> None:
    _generation("20260101T000000Z-a")
    reading = capacity_recovery.measure_capacity(max_generations=100, min_free_bytes=10**18)
    assert reading.threshold == "free"


def test_the_request_is_idempotent_by_record(home: Path) -> None:
    # Two generations with a max of one, so the count threshold genuinely crosses
    # (free space is left out of it by a tiny floor).
    for index in range(2):
        _generation(f"2026010{index + 1}T000000Z-{index}")
    operation = install_transaction.start_operation(
        update_mod.stable_root(), baseline="a", target="b", route="latest"
    )
    first = capacity_recovery.capacity_request(
        update_mod.stable_root(), operation.identifier, max_generations=1, min_free_bytes=1
    )
    assert first is not None and first["state"] == "requested"
    second = capacity_recovery.capacity_request(
        update_mod.stable_root(), operation.identifier, max_generations=1, min_free_bytes=1
    )
    assert second is not None and second["state"] == "requested"
    assert second["requested_at"] == first["requested_at"]
    assert second == first, "a re-entry while the request is open must not write a second"


def test_no_request_when_under_both_thresholds(home: Path) -> None:
    operation = install_transaction.start_operation(
        update_mod.stable_root(), baseline="a", target="b", route="latest"
    )
    assert (
        capacity_recovery.capacity_request(
            update_mod.stable_root(), operation.identifier, max_generations=100, min_free_bytes=1
        )
        is None
    )


# -- the safe-GC predicate -----------------------------------------------------


def test_an_unreferenced_generation_is_reclaimable(home: Path) -> None:
    for index in range(4):
        _generation(f"2026010{index + 1}T000000Z-{index}")
    plan = capacity_recovery.reclaimable_now(keep=0)
    assert len(plan.reclaimable) == 4
    # DRY RUN: nothing was actually removed.
    assert all(path.is_dir() for path in plan.reclaimable)


def test_a_live_recorded_root_is_kept(home: Path) -> None:
    live = _generation("20260101T000000Z-live")
    for index in range(3):
        _generation(f"2026010{index + 2}T000000Z-{index}")
    _session_record(home / ".local-operator", live)
    plan = capacity_recovery.reclaimable_now(keep=0)
    assert live.resolve() not in {path.resolve() for path in plan.reclaimable}


def test_the_pointer_target_is_kept(home: Path) -> None:
    target = _generation("20260101T000000Z-target")
    update_mod.pointer_path().parent.mkdir(parents=True, exist_ok=True)
    update_mod.pointer_path().symlink_to(target)
    for index in range(3):
        _generation(f"2026010{index + 2}T000000Z-{index}")
    plan = capacity_recovery.reclaimable_now(keep=0)
    assert target.resolve() not in {path.resolve() for path in plan.reclaimable}


def test_an_in_flight_tree_without_a_marker_is_kept(home: Path) -> None:
    in_flight = _generation("20260101T000000Z-building", marker=False, age=1.0)
    plan = capacity_recovery.reclaimable_now(keep=0)
    assert in_flight.resolve() not in {path.resolve() for path in plan.reclaimable}


# -- the controller ------------------------------------------------------------


def test_the_controller_reclaims_a_moved_cohort(home: Path) -> None:
    cohort_root = _generation("20260101T000000Z-cohort")
    for index in range(2):
        _generation(f"2026010{index + 2}T000000Z-x{index}")
    _session_record(home / ".local-operator", cohort_root)
    requested: list[capacity_recovery.Cohort] = []

    # The move is simulated by removing the record the way a successful checkpoint
    # would (the runtime left the root); the controller must RE-CENSUS, not trust
    # the request's say-so. The reclaim callback is the caller's real prune.
    def request_move(cohort: capacity_recovery.Cohort) -> bool:
        requested.append(cohort)
        (home / ".local-operator" / "run" / "mobile" / "4242.json").unlink()
        return True

    state = capacity_recovery.run_capacity_recovery(
        request_move=request_move,
        wait_move=lambda _c: True,
        reclaim=lambda _paths: update_mod.prune_generations(keep=0, dry_run=False),
    )
    assert requested and requested[0].root.resolve() == cohort_root.resolve()
    assert state == "reclaimed"
    # The reclaim ran through the predicate for real: the tree is gone.
    assert not cohort_root.exists()


def test_the_controller_reports_partial_when_a_cohort_will_not_move(home: Path) -> None:
    cohort_root = _generation("20260101T000000Z-cohort")
    _session_record(home / ".local-operator", cohort_root)
    state = capacity_recovery.run_capacity_recovery(
        request_move=lambda _c: False, reclaim=lambda _p: None
    )
    assert state == "partial"
    assert cohort_root.exists(), "nothing may be deleted when the cohort did not confirm"


def test_the_controller_moves_no_cohort_when_none_is_eligible(home: Path) -> None:
    # Generations with no live record: nothing to move, and the controller says so
    # rather than idling a fleet.
    for index in range(2):
        _generation(f"2026010{index + 1}T000000Z-x{index}")
    assert capacity_recovery.run_capacity_recovery() == "nothing-eligible"


def test_the_promotion_tail_records_pressure_on_the_candidates_operation(home: Path) -> None:
    """The automatic action is a RECORD, not a deletion.

    Drives the real tail helper over a real generation (built by the fake uv) with
    the measurement steered over the threshold, and asserts the durable capacity
    request lands on THIS candidate's operation — the tie ``operation_for_generation``
    exists to make, so recovery is not handed to a concurrent newer operation.
    """
    from tests.unit.test_install_generations import FakeUv

    generation = update_mod.install_into_generation(
        runner=FakeUv(version="0.52.0"), version="0.52.0"
    )
    import local_operator.capacity_recovery as cr

    original = cr.measure_capacity
    try:
        cr.measure_capacity = lambda **_k: cr.CapacityMeasurement(99, 0, "both")
        update_mod._note_capacity_after_promotion(generation)
    finally:
        cr.measure_capacity = original
    operation = install_transaction.operation_for_generation(generation)
    assert operation is not None
    entry = install_transaction.capacity_status(
        update_mod.stable_root(), operation.identifier
    )
    assert entry is not None and entry["state"] == "requested"
    # NOTHING WAS RECLAIMED: the generation is still there.
    assert generation.exists()


def test_the_promotion_tail_is_silent_without_pressure(home: Path) -> None:
    from tests.unit.test_install_generations import FakeUv

    generation = update_mod.install_into_generation(
        runner=FakeUv(version="0.52.0"), version="0.52.0"
    )
    operation = install_transaction.operation_for_generation(generation)
    assert operation is not None
    # Steer the measurement under both thresholds via the settings the helper reads.
    import local_operator.capacity_recovery as cr

    original = cr.measure_capacity
    try:
        cr.measure_capacity = lambda **_k: cr.CapacityMeasurement(0, 10**18, "")
        update_mod._note_capacity_after_promotion(generation)
    finally:
        cr.measure_capacity = original
    assert (
        install_transaction.capacity_status(update_mod.stable_root(), operation.identifier)
        is None
    )


def test_an_unresolvable_free_reading_is_treated_as_over(home: Path, monkeypatch) -> None:
    """An unreadable free reading is not 'plenty of room' — take the safe path."""

    def _boom(_p):
        raise OSError("no stat")

    monkeypatch.setattr(capacity_recovery.shutil, "disk_usage", _boom)
    reading = capacity_recovery.measure_capacity(max_generations=100, min_free_bytes=1)
    assert reading.free_bytes == -1
    assert reading.threshold == "free"
