"""Capacity recovery: the trigger, the safe-GC predicate, and the controller.

Driven over tiny owned roots (real generation directories built the way a prune
sees them) with SIMULATED capacity — never a host disk fill. The subject is the
DECISION: what is provably unreferenced is reclaimed, and everything unknown is
retained.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from local_operator import capacity_recovery, install_transaction
from local_operator import update as update_mod

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


def test_the_controller_reports_moved_unreclaimed_when_no_reclaim_was_passed(
    home: Path,
) -> None:
    """A decision-only call must NOT claim a deletion it did not make."""
    cohort_root = _generation("20260101T000000Z-cohort")
    for index in range(2):
        _generation(f"2026010{index + 2}T000000Z-x{index}")
    _session_record(home / ".local-operator", cohort_root)

    def request_move(cohort: capacity_recovery.Cohort) -> bool:
        (home / ".local-operator" / "run" / "mobile" / "4242.json").unlink()
        return True

    # No ``reclaim`` argument: the cohort moves and the roots are provably
    # unreferenced, but nothing is deleted — so the state is moved-unreclaimed.
    state = capacity_recovery.run_capacity_recovery(
        request_move=request_move, wait_move=lambda _c: True
    )
    assert state == "moved-unreclaimed"
    assert cohort_root.exists()


def test_dry_run_does_not_sweep_staging_links(home: Path, monkeypatch) -> None:
    """A dry run makes the DECISION only — the staging sweep is a mutation."""
    calls: list[float] = []
    monkeypatch.setattr(update_mod, "_sweep_staging_links", lambda moment: calls.append(moment))
    for index in range(3):
        _generation(f"2026010{index + 1}T000000Z-{index}")
    update_mod.prune_generations(keep=0, dry_run=True)
    assert calls == [], "a dry run must not sweep staging links"
    # The normal path still sweeps, so the guard is scoped to dry_run, not the
    # function.
    update_mod.prune_generations(keep=0, dry_run=False)
    assert len(calls) == 1


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
    entry = install_transaction.capacity_status(update_mod.stable_root(), operation.identifier)
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
        install_transaction.capacity_status(update_mod.stable_root(), operation.identifier) is None
    )


def test_an_unresolvable_free_reading_is_treated_as_over(home: Path, monkeypatch) -> None:
    """An unreadable free reading is not 'plenty of room' — take the safe path."""

    def _boom(_p):
        raise OSError("no stat")

    monkeypatch.setattr(capacity_recovery.shutil, "disk_usage", _boom)
    reading = capacity_recovery.measure_capacity(max_generations=100, min_free_bytes=1)
    assert reading.free_bytes == -1
    assert reading.threshold == "free"


# -- the automatic prune after promotion (operator decision, 2026-11-16) --------
#
# A fleet node reached 113 generations (~8.6 GB) because the promotion tail only
# RECORDED capacity pressure and nothing consumed the record. The tail now runs
# ``lop install prune``'s own policy (``update.reclaim_after_promotion``). These
# drive that real tail over real generations built by the fake uv, and pin the
# properties the operator asked for: the count is bounded, current/held/running
# trees survive, every doubt keeps trees, the switch works, and a failing
# reclaim never fails the update.


def _write_update_settings(home: Path, values: dict[str, object] | str) -> None:
    """Write ``values.update`` into THIS test's ``config.yml`` (never a real one).

    ``config_dir()`` follows ``$HOME`` here because conftest scrubs
    ``LOCAL_OPERATOR_CONFIG_DIR``; the assert makes a regression in that isolation
    fail loudly instead of writing the developer's config.
    """
    import yaml

    from local_operator.paths import config_dir

    root = config_dir()
    assert home in root.parents, root
    root.mkdir(parents=True, exist_ok=True)
    body = (
        values
        if isinstance(values, str)
        else yaml.safe_dump({"values": {"update": values}}, sort_keys=True)
    )
    path = root / "config.yml"
    path.write_text(body, encoding="utf-8")
    # ``read_config_values`` memoises a parse per (inode, size, mtime_ns, ...),
    # and two rewrites inside one coarse clock tick can share all of them. A
    # strictly increasing explicit mtime makes every rewrite a new key.
    _CONFIG_WRITES.append(None)
    stamp = time.time_ns() + len(_CONFIG_WRITES) * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


_CONFIG_WRITES: list[None] = []


def _promote(version: str) -> Path:
    """One real promotion (fake uv build + pointer flip), then the shared tail."""
    from tests.unit.test_install_generations import _install

    generation = _install(version)
    update_mod.reclaim_after_promotion(generation)
    return generation


def _on_disk() -> list[Path]:
    return sorted(
        path
        for path in update_mod.generations_dir().iterdir()
        if path.is_dir() and not path.is_symlink()
    )


def test_n_plus_five_updates_leave_at_most_n_plus_refs_generations(home: Path) -> None:
    """(a) The count is bounded by keep + referenced + current, update after update."""
    keep = update_mod.DEFAULT_KEEP_GENERATIONS
    held = _promote("0.52.0")
    _session_record(home / ".local-operator", held)
    promoted = [held]
    for index in range(1, keep + 6):
        promoted.append(_promote(f"0.52.{index}"))
        # Bounded after EVERY update, not only at the end: one held root, the
        # pointer's target, and at most ``keep`` unreferenced survivors.
        assert len(_on_disk()) <= keep + 1 + 1, [path.name for path in _on_disk()]
    survivors = _on_disk()
    assert len(survivors) == keep + 2, [path.name for path in survivors]
    assert update_mod.current_generation() == promoted[-1].resolve()
    assert promoted[-1].is_dir(), "the build just promoted is never removed"
    assert held.is_dir(), "a session-referenced generation is never removed"


def test_current_held_and_running_generations_are_never_removed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) keep=0 is the harshest setting; the three protected trees still survive."""
    _write_update_settings(home, {"auto_prune": True, "keep_generations": 0})
    running = _install_only("0.52.0")
    # The updater imports from the tree it is replacing, which no record names.
    monkeypatch.setattr(update_mod, "_running_generation", lambda: running.resolve())
    held = _install_only("0.52.1")
    _session_record(home / ".local-operator", held)
    doomed = _promote("0.52.2")
    current = _promote("0.52.3")
    assert running.is_dir(), "the tree this process runs from was removed"
    assert held.is_dir(), "a session-referenced generation was removed"
    assert current.is_dir() and update_mod.current_generation() == current.resolve()
    # The control that proves keep=0 was really in force: the unprotected one went.
    assert not doomed.exists()


def test_an_unreadable_census_retains_everything(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) An incomplete record read proves nothing by omission: nothing goes."""
    _write_update_settings(home, {"keep_generations": 0})
    generations = [_install_only(f"0.52.{index}") for index in range(3)]
    incomplete = update_mod.ReferencedTrees(
        roots=(), complete=False, gaps=("a session record could not be read",)
    )
    monkeypatch.setattr(update_mod, "referenced_install_roots", lambda: incomplete)
    newest = _install_only("0.52.9")
    plan = update_mod.reclaim_after_promotion(newest)
    assert plan is not None and plan.references_complete is False
    assert plan.removed == ()
    assert all(path.is_dir() for path in [*generations, newest])


def test_an_unreadable_config_or_a_foreign_pointer_retains_everything(home: Path) -> None:
    """(c) Unknown policy or a pointer this run does not own: no prune at all."""
    _write_update_settings(home, {"keep_generations": 0})
    generations = [_install_only(f"0.52.{index}") for index in range(3)]
    # A torn/unparseable config: the switch may be off, so nothing is removed.
    _write_update_settings(home, "values: [unterminated\n")
    newest = _install_only("0.52.8")
    assert update_mod.reclaim_after_promotion(newest) is None
    # A pointer that no longer names the promoted build (another actor flipped
    # it): this run does not own the decision.
    _write_update_settings(home, {"keep_generations": 0})
    assert update_mod.reclaim_after_promotion(generations[0]) is None
    assert all(path.is_dir() for path in [*generations, newest])


def test_auto_prune_off_removes_nothing(home: Path) -> None:
    """(d) Negative control: the same updates with the switch off keep every tree."""
    _write_update_settings(home, {"auto_prune": False, "keep_generations": 0})
    generations = [_promote(f"0.52.{index}") for index in range(4)]
    assert all(path.is_dir() for path in generations)
    assert len(_on_disk()) == 4
    # A hand-edited non-boolean is someone trying to stop it: it reads as OFF.
    _write_update_settings(home, {"auto_prune": "no", "keep_generations": 0})
    _promote("0.52.9")
    assert len(_on_disk()) == 5


def test_a_reclaim_failure_does_not_fail_the_update(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) The build is already current; a failing prune is logged, not raised."""

    def _boom(**_kwargs: object) -> object:
        raise update_mod.UpdateError("another install mutation owns the root")

    monkeypatch.setattr(update_mod, "prune_generations", _boom)
    reported: list[object] = []
    generation = _install_only("0.52.0")
    assert update_mod.reclaim_after_promotion(generation, on_prune=reported.append) is None
    assert reported == []
    assert generation.is_dir()

    # And through the real PyPI tail: perform_upgrade still returns the target.
    monkeypatch.setattr(update_mod, "install_into_generation", lambda *_a, **_k: generation)
    assert (
        update_mod.perform_upgrade(target="0.52.0", kind=update_mod.InstallKind.UV_TOOL) == "0.52.0"
    )


def test_the_policy_reader_defaults_and_skips_on_bad_values(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing keys read the defaults; a bad keep or block SKIPS, never falls back (R1)."""
    default = update_mod.AutoPrunePolicy(enabled=True, keep=update_mod.DEFAULT_KEEP_GENERATIONS)
    _write_update_settings(home, {})
    assert update_mod.auto_prune_policy() == default
    for bad in (-1, True, "3", 1.5, [2]):
        caplog.clear()
        _write_update_settings(home, {"keep_generations": bad})
        with caplog.at_level("WARNING", logger=update_mod.logger.name):
            assert update_mod.auto_prune_policy() is None, bad
        assert "keep_generations" in caplog.text, bad
    # A non-mapping ``update`` block is the same doubt one level up.
    caplog.clear()
    _write_update_settings(home, "values:\n  update: 3\n")
    with caplog.at_level("WARNING", logger=update_mod.logger.name):
        assert update_mod.auto_prune_policy() is None
    assert "not a mapping" in caplog.text
    # An explicit ``null`` is UNSET, not bad: it reads the defaults.
    _write_update_settings(home, {"keep_generations": None, "auto_prune": None})
    assert update_mod.auto_prune_policy() == default
    _write_update_settings(home, {"keep_generations": 5})
    assert update_mod.auto_prune_policy() == update_mod.AutoPrunePolicy(enabled=True, keep=5)


def test_a_bad_keep_value_removes_nothing(home: Path) -> None:
    """R1 end to end: ``keep_generations: "0"`` is not a licence to prune at 2."""
    generations = [_install_only(f"0.52.{index}") for index in range(5)]
    _write_update_settings(home, {"keep_generations": "0"})
    assert update_mod.reclaim_after_promotion(generations[-1]) is None
    assert all(path.is_dir() for path in generations)


def test_a_non_boolean_auto_prune_warns_and_reads_off(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """N2: the off reading is said out loud, naming the value."""
    _write_update_settings(home, {"auto_prune": "yes"})
    with caplog.at_level("WARNING", logger=update_mod.logger.name):
        policy = update_mod.auto_prune_policy()
    assert policy is not None and policy.enabled is False
    assert "update.auto_prune is 'yes'" in caplog.text


def test_the_build_current_before_the_flip_survives_keep_zero(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2 / QA Q2: the pre-flip ``current`` survives keep=0 even when the
    updater runs from an OLDER build — the running tree is not the previous one.
    """
    _write_update_settings(home, {"keep_generations": 0})
    running = _install_only("0.51.9")
    monkeypatch.setattr(update_mod, "_running_generation", lambda: running.resolve())
    older = _install_only("0.52.0")
    previous = _install_only("0.52.1")
    captured = update_mod.previous_generation_for_reclaim()
    assert captured == previous.resolve()
    promoted = _install_only("0.52.2")
    plan = update_mod.reclaim_after_promotion(promoted, previous=captured)
    assert plan is not None
    assert previous.is_dir(), "the build replaced by this update was removed"
    assert running.is_dir(), "the tree the updater runs from was removed"
    # The control: keep=0 really ran, so the older unprotected build went.
    assert not older.exists()
    assert promoted.is_dir()


def test_the_previous_root_keeps_an_incomplete_census_incomplete() -> None:
    """Adding the protection must never turn an incomplete read into a complete one."""
    incomplete = update_mod.ReferencedTrees(roots=(), complete=False, gaps=("torn",))
    merged = update_mod._with_named_generation(incomplete, Path("/g/prev"))
    assert merged.complete is False and merged.gaps == ("torn",)
    assert merged.roots == (Path("/g/prev") / "tools" / DIST,)
    plain = update_mod._with_named_generation([Path("/g/a/tools/x")], Path("/g/prev"))
    assert plain.complete is True and len(plain.roots) == 2


def _install_only(version: str) -> Path:
    """A promotion WITHOUT the tail, for tests that call the tail themselves."""
    from tests.unit.test_install_generations import _install

    return _install(version)
