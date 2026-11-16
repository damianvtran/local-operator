"""Future-only fenced (v2) operations over synthetic roots.

Each test maps to one proof obligation from the last-writer-authority boundary:
legacy negative control, address fence, late-write fence, live last writer,
dynamic enrollment, unfinished domains, legacy sidecar re-entry and honest
capacity. Every root is a pytest tmp directory; no live install root is read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import fenced_operation as v2
from local_operator import install_transaction as tx


def _start(root: Path) -> v2.WriterContext:
    return v2.start_operation(root, baseline="A", target="B", route="default-cli")


def _record_path(root: Path, context: v2.WriterContext) -> Path:
    return tx.canonical_root(root) / v2.V2_OPERATION_DIR / f"{context.operation_id}.json"


def _raw(root: Path, context: v2.WriterContext) -> dict[str, Any]:
    return json.loads(_record_path(root, context).read_bytes())


def _status(root: Path, context: v2.WriterContext) -> dict[str, Any]:
    row = v2.operation_status(root, context.operation_id)
    assert row is not None
    return row


def _generation(root: Path, name: str = "g1") -> Path:
    generation = tx.canonical_root(root) / "generations" / name
    generation.mkdir(parents=True)
    return generation


# 1. Legacy negative control -------------------------------------------------


def test_legacy_writer_rewrites_a_naive_same_path_seal(tmp_path):
    """Why v2 needs its own address: a seal key in the legacy JSON fences nothing."""
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    path = tmp_path / tx.OPERATION_DIR / f"{operation.identifier}.json"
    record = json.loads(path.read_bytes())
    record["terminal_seal"] = {"sealed": True}
    path.write_text(json.dumps(record))
    operation.advance("partial", receipts=[{"late": True}])
    after = json.loads(path.read_bytes())
    assert after["terminal_seal"] == {"sealed": True}
    assert after["phase"] == "partial" and after["receipts"] == [{"late": True}]


# 2. Address fence ------------------------------------------------------------


def test_legacy_apis_cannot_address_a_v2_record(tmp_path):
    context = _start(tmp_path)
    before = _record_path(tmp_path, context).read_bytes()
    identifier = context.operation_id
    assert tx.operation_status(tmp_path, identifier) is None
    with pytest.raises(tx.InstallTransactionError):
        tx.UpdateOperation(tx.canonical_root(tmp_path), identifier).advance("partial")
    with pytest.raises(tx.InstallTransactionError):
        tx.record_occurrence(tmp_path, identifier, "o1", kind="scheduler", state="running")
    with pytest.raises(tx.InstallTransactionError):
        tx.note_capacity(tmp_path, identifier, state="requested", threshold="count")
    with pytest.raises(tx.InstallTransactionError):
        tx.advance_handoff(tmp_path, identifier, "ready")
    assert _record_path(tmp_path, context).read_bytes() == before
    assert not (tmp_path / tx.OPERATION_DIR / f"{identifier}.json").exists()


def test_v2_locator_is_invisible_to_legacy_and_yields_no_writer(tmp_path):
    context = _start(tmp_path)
    generation = _generation(tmp_path)
    v2.link_generation(context, generation)
    assert not (generation / v2.LEGACY_SIDECAR_NAME).exists()
    assert tx.operation_for_generation(generation) is None
    observed = v2.observe_generation(generation)
    assert observed is not None
    assert observed["writer"] is None and observed["operation_id"] == context.operation_id
    assert not any(isinstance(value, v2.WriterContext) for value in observed.values())


def test_status_is_read_only_and_creates_nothing(tmp_path):
    root = tmp_path / "absent"
    assert v2.operation_status(root, "0" * 32) is None
    assert not root.exists()


# 3. Late-write fence ---------------------------------------------------------

_MUTATIONS: list[tuple[str, Callable[[v2.WriterContext], object]]] = [
    ("advance", lambda c: v2.advance(c, "partial")),
    ("occurrence", lambda c: v2.record_occurrence(c, "o9", kind="scheduler", state="running")),
    ("capacity", lambda c: v2.note_capacity(c, state="requested", threshold="count")),
    ("handoff", lambda c: v2.advance_handoff(c, "ready")),
    ("continuation", lambda c: v2.declare_continuation(c, "late")),
    ("enroll", lambda c: v2.enroll_writer(c, "reporter")),
    ("link", lambda c: v2.link_generation(c, c.root / "generations" / "late")),
    ("finish", lambda c: v2.finish(c, "completed")),
]


@pytest.mark.parametrize("label,mutate", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_every_mutator_rejects_a_finished_writer(tmp_path, label, mutate):
    context = _start(tmp_path)
    v2.finish(context, "completed")
    before = _record_path(tmp_path, context).read_bytes()
    with pytest.raises(v2.FencedOperationError):
        mutate(context)
    assert _record_path(tmp_path, context).read_bytes() == before, label


@pytest.mark.parametrize("label,mutate", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_every_mutator_rejects_after_seal_even_for_a_still_open_slot(tmp_path, label, mutate):
    """Seal is checked BEFORE slot state: a slot forged open after sealing is inert."""
    context = _start(tmp_path)
    assert v2.finish(context, "completed", seal=True)["sealed"] is True
    record = _raw(tmp_path, context)
    record["writers"][context.slot]["state"] = "open"
    _record_path(tmp_path, context).write_text(json.dumps(record))
    before = _record_path(tmp_path, context).read_bytes()
    with pytest.raises(v2.FencedOperationError, match="sealed"):
        mutate(context)
    assert _record_path(tmp_path, context).read_bytes() == before, label


def test_generic_advance_cannot_replace_domain_maps(tmp_path):
    context = _start(tmp_path)
    for field in ("handoff", "capacity", "continuations", "occurrences", "obligations"):
        with pytest.raises(v2.FencedOperationError):
            v2.advance(context, "partial", **{field: {}})


def test_second_api_cannot_reopen_a_finished_writers_authority(tmp_path):
    coordinator = _start(tmp_path)
    reporter = v2.enroll_writer(coordinator, "reporter")
    v2.finish(reporter, "completed")
    with pytest.raises(v2.FencedOperationError):
        v2.note_capacity(reporter, state="requested", threshold="count")
    with pytest.raises(v2.FencedOperationError):
        v2.finish(reporter, "failed")
    assert _raw(tmp_path, coordinator)["writers"][reporter.slot]["state"] == "finished"


def test_forged_or_wrong_epoch_contexts_are_refused(tmp_path):
    context = _start(tmp_path)
    forged = v2.WriterContext(context.root, context.operation_id, "f" * 32, "coordinator")
    with pytest.raises(v2.FencedOperationError, match="not admitted"):
        v2.advance(forged, "partial")
    with pytest.raises(v2.FencedOperationError, match="not admitted"):
        v2.try_seal(forged)
    wrong_role = v2.WriterContext(context.root, context.operation_id, context.slot, "reporter")
    with pytest.raises(v2.FencedOperationError, match="not admitted"):
        v2.advance(wrong_role, "partial")
    wrong_epoch = v2.WriterContext(
        context.root, context.operation_id, context.slot, "coordinator", epoch=2
    )
    with pytest.raises(v2.FencedOperationError, match="epoch"):
        v2.advance(wrong_epoch, "partial")


def test_role_permissions_are_closed(tmp_path):
    coordinator = _start(tmp_path)
    scheduler = v2.enroll_writer(coordinator, "scheduler")
    with pytest.raises(v2.FencedOperationError, match="may not"):
        v2.advance(scheduler, "partial")
    with pytest.raises(v2.FencedOperationError, match="may not"):
        v2.enroll_writer(scheduler, "reporter")
    with pytest.raises(v2.FencedOperationError):
        v2.enroll_writer(coordinator, "coordinator")


# 4. Live last writer ---------------------------------------------------------


def test_live_reporter_seals_by_finishing_and_later_writes_reject(tmp_path):
    """No PID shortcut: this very process stays alive after its own seal."""
    context = _start(tmp_path)
    v2.advance(context, "partial", receipts=[{"unit": "serve", "state": "current"}])
    result = v2.finish(context, "completed", seal=True)
    assert result == {"finished": True, "sealed": True, "blockers": []}
    sealed = _raw(tmp_path, context)["sealed"]
    assert sealed["sealed_by"] == context.slot and sealed["writers"] == {context.slot: "completed"}
    assert v2.try_seal(context)["already"] is True
    with pytest.raises(v2.FencedOperationError):
        v2.advance(context, "failed")


def test_last_delegated_writer_triggers_the_seal(tmp_path):
    coordinator = _start(tmp_path)
    reporter = v2.enroll_writer(coordinator, "reporter")
    first = v2.finish(coordinator, "completed", seal=True)
    assert first["sealed"] is False and f"writer_open:{reporter.slot}" in first["blockers"]
    assert v2.finish(reporter, "completed", seal=True)["sealed"] is True


def test_try_seal_by_a_finished_writer_seals_once_conditions_hold(tmp_path):
    context = _start(tmp_path)
    assert v2.finish(context, "completed")["sealed"] is False
    assert v2.try_seal(context) == {"sealed": True, "already": False, "blockers": []}


# 5. Dynamic enrollment -------------------------------------------------------


def test_registration_before_effect_and_closed_enrollment(tmp_path):
    coordinator = _start(tmp_path)
    tail = v2.enroll_writer(coordinator, "service_tail")
    assert _raw(tmp_path, coordinator)["writers"][tail.slot]["state"] == "open"
    v2.advance(tail, "services")
    v2.close_enrollment(coordinator)
    writers_before = set(_raw(tmp_path, coordinator)["writers"])
    with pytest.raises(v2.FencedOperationError, match="enrollment is closed"):
        v2.enroll_writer(coordinator, "reporter")
    assert set(_raw(tmp_path, coordinator)["writers"]) == writers_before


def test_finishing_coordinator_closes_enrollment_irrevocably(tmp_path):
    coordinator = _start(tmp_path)
    v2.enroll_writer(coordinator, "reporter")
    v2.finish(coordinator, "completed")
    assert _raw(tmp_path, coordinator)["enrollment"] == "closed"


def test_lost_final_ack_stays_protected(tmp_path):
    """A delegated writer that never finishes keeps the operation unsealed."""
    coordinator = _start(tmp_path)
    child = v2.enroll_writer(coordinator, "service_tail")
    v2.advance(child, "services")
    result = v2.finish(coordinator, "completed", seal=True)
    assert result["sealed"] is False
    assert _status(tmp_path, coordinator)["seal_blockers"] == [f"writer_open:{child.slot}"]


# 6. Unfinished domains -------------------------------------------------------


@pytest.mark.parametrize(
    "setup,blocker",
    [
        (lambda c: v2.note_capacity(c, state="requested", threshold="count"), "capacity"),
        (lambda c: v2.note_capacity(c, state="partial", threshold="count"), "capacity"),
        (
            lambda c: v2.record_occurrence(c, "o1", kind="scheduler", state="running"),
            "occurrence:o1",
        ),
        (lambda c: v2.advance_handoff(c, "ready"), "handoff"),
        (lambda c: v2.declare_continuation(c, "restart"), "continuation:restart"),
    ],
    ids=["capacity-requested", "capacity-partial", "occurrence", "handoff", "continuation"],
)
def test_pending_obligation_prevents_seal(tmp_path, setup, blocker):
    context = _start(tmp_path)
    v2.advance(context, "services")
    setup(context)
    result = v2.finish(context, "completed", seal=True)
    assert result["sealed"] is False
    assert f"obligation_pending:{blocker}" in result["blockers"]
    assert _raw(tmp_path, context)["sealed"] is None


def test_resolved_obligations_allow_seal(tmp_path):
    context = _start(tmp_path)
    v2.note_capacity(context, state="requested", threshold="count")
    v2.note_capacity(context, state="moved", threshold="count")
    v2.record_occurrence(context, "o1", kind="scheduler", state="running")
    v2.record_occurrence(context, "o1", kind="scheduler", state="terminal", terminal="done")
    v2.advance_handoff(context, "ready")
    v2.advance_handoff(context, "commit")
    v2.advance_handoff(context, "ack")
    v2.declare_continuation(context, "restart")
    v2.resolve_continuation(context, "restart", outcome="done")
    assert v2.finish(context, "completed", seal=True)["sealed"] is True


def test_only_the_owner_may_end_its_obligation(tmp_path):
    coordinator = _start(tmp_path)
    scheduler = v2.enroll_writer(coordinator, "scheduler")
    v2.record_occurrence(scheduler, "o1", kind="scheduler", state="running")
    with pytest.raises(v2.FencedOperationError, match="owner"):
        v2.record_occurrence(coordinator, "o1", kind="scheduler", state="terminal", terminal="done")
    v2.declare_continuation(coordinator, "c1")
    reporter = v2.enroll_writer(coordinator, "reporter")
    with pytest.raises(v2.FencedOperationError, match="declaring writer"):
        v2.resolve_continuation(reporter, "c1", outcome="done")


@pytest.mark.parametrize(
    "mutation,blocker",
    [
        (
            lambda r: r.update(unknown_consumer={"pending": True}),
            "unknown_field:unknown_consumer",
        ),
        (lambda r: r["obligations"].update(x={"state": "mystery"}), "obligation_unknown:x"),
        (
            lambda r: r["writers"].update(w={"role": "daemon", "state": "finished"}),
            "writer_unknown:w",
        ),
    ],
    ids=["unknown-field", "unknown-obligation", "unknown-writer-role"],
)
def test_unsupported_shapes_block_seal_rather_than_discharge(tmp_path, mutation, blocker):
    context = _start(tmp_path)
    record = _raw(tmp_path, context)
    mutation(record)
    _record_path(tmp_path, context).write_text(json.dumps(record))
    result = v2.finish(context, "completed", seal=True)
    assert result["sealed"] is False and blocker in result["blockers"]


def test_corrupt_v2_record_fails_closed(tmp_path):
    context = _start(tmp_path)
    _record_path(tmp_path, context).write_text("{not json")
    with pytest.raises(v2.FencedOperationError):
        v2.advance(context, "partial")
    with pytest.raises(v2.FencedOperationError):
        v2.operation_status(tmp_path, context.operation_id)
    _record_path(tmp_path, context).write_text(
        json.dumps({"operation_id": context.operation_id, "phase": "partial"})
    )
    with pytest.raises(v2.FencedOperationError, match="untrusted shape"):
        v2.try_seal(context)


# 7. Legacy sidecar re-entry --------------------------------------------------


def test_legacy_sidecar_generation_is_never_linked_or_observed_as_v2(tmp_path):
    legacy = tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    generation = _generation(tmp_path)
    tx.link_operation(legacy, generation)
    context = _start(tmp_path)
    with pytest.raises(v2.FencedOperationError, match="legacy recovery sidecar"):
        v2.link_generation(context, generation)
    assert not (generation / v2.V2_SIDECAR_NAME).exists()
    assert v2.observe_generation(generation) is None


def test_dual_published_generation_fails_closed(tmp_path):
    context = _start(tmp_path)
    generation = _generation(tmp_path)
    v2.link_generation(context, generation)
    (generation / v2.LEGACY_SIDECAR_NAME).write_text(
        json.dumps({"operation_id": context.operation_id})
    )
    with pytest.raises(v2.FencedOperationError, match="both recovery protocols"):
        v2.observe_generation(generation)


def test_legacy_alias_of_the_same_id_blocks_seal(tmp_path):
    context = _start(tmp_path)
    legacy_dir = tx.canonical_root(tmp_path) / tx.OPERATION_DIR
    legacy_dir.mkdir(mode=0o700)
    (legacy_dir / f"{context.operation_id}.json").write_text(
        json.dumps({"operation_id": context.operation_id, "phase": "partial"})
    )
    result = v2.finish(context, "completed", seal=True)
    assert result["sealed"] is False and "legacy_alias" in result["blockers"]


def test_forged_locator_yields_no_status(tmp_path):
    context = _start(tmp_path)
    generation = _generation(tmp_path)
    (generation / v2.V2_SIDECAR_NAME).write_text(
        json.dumps(
            {"protocol": v2.PROTOCOL, "operation_id": context.operation_id, "access": "write"}
        )
    )
    with pytest.raises(v2.FencedOperationError, match="untrusted"):
        v2.observe_generation(generation)
    (generation / v2.V2_SIDECAR_NAME).write_text(
        json.dumps(
            {"protocol": v2.PROTOCOL, "operation_id": context.operation_id, "access": "observe"}
        )
    )
    with pytest.raises(v2.FencedOperationError, match="disagrees"):
        v2.observe_generation(generation)


# 8. Capacity stays one honest pool -------------------------------------------


def _fill_legacy(root: Path, count: int) -> None:
    directory = tx.canonical_root(root) / tx.OPERATION_DIR
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:032x}.json").write_text("{}")


def test_full_legacy_namespace_refuses_v2_admission_without_creating_it(tmp_path):
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit)
    with pytest.raises(tx.RecoveryCapacityError) as caught:
        _start(tmp_path)
    assert caught.value.code == "update_recovery_capacity"
    assert not (tx.canonical_root(tmp_path) / v2.V2_OPERATION_DIR).exists()


def test_v2_entries_share_the_legacy_budget(tmp_path):
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit - 1)
    _start(tmp_path)
    with pytest.raises(tx.RecoveryCapacityError):
        _start(tmp_path)
    assert v2.active_recovery_entries(tmp_path) == tx.RecoveryCapacityError.active_limit


_CAPACITY_SENTENCE = "update recovery records are at capacity; inspect pending operations"


def test_255_legacy_plus_one_v2_refuses_both_admission_paths(tmp_path):
    """ONE pool: the old ``start_operation`` must see v2 entries too."""
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit - 1)
    _start(tmp_path)
    legacy_dir = tx.canonical_root(tmp_path) / tx.OPERATION_DIR
    before = sorted(path.name for path in legacy_dir.iterdir())
    with pytest.raises(tx.RecoveryCapacityError) as legacy_refusal:
        tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    with pytest.raises(tx.RecoveryCapacityError) as v2_refusal:
        _start(tmp_path)
    for refusal in (legacy_refusal.value, v2_refusal.value):
        assert isinstance(refusal, tx.InstallTransactionError)
        assert str(refusal) == _CAPACITY_SENTENCE
        assert refusal.code == "update_recovery_capacity"
    assert sorted(path.name for path in legacy_dir.iterdir()) == before
    assert tx.active_recovery_entries(tmp_path) == tx.RecoveryCapacityError.active_limit


def _forbid_lock_and_liveness(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("capacity status acquired a transaction or probed owner liveness")

    monkeypatch.setattr(tx, "transaction", forbidden)
    monkeypatch.setattr(tx, "owner_alive", forbidden)


def test_status_counts_both_namespaces_consistently_with_admission(tmp_path, monkeypatch):
    """255 v1 + 1 v2: the dry run says 256/at-limit exactly when admission refuses."""
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit - 1)
    _start(tmp_path)
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    with monkeypatch.context() as patched:
        _forbid_lock_and_liveness(patched)
        report = tx.recovery_capacity_status(tmp_path)
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before
    assert report["observed_entries"] == report["protected_entries"] == 256
    assert report["namespace_entries"] == {tx.OPERATION_DIR: 255, tx.V2_OPERATION_DIR: 1}
    assert report["at_limit"] is True and report["scan_complete"] is True
    assert report["retirable_entries"] == 0
    assert report["protected_reasons"] == {
        "unreadable_or_unsafe_record": 255,
        "new_protocol_open": 1,
    }
    assert report["observed_entries"] == tx.active_recovery_entries(tmp_path)
    with pytest.raises(tx.RecoveryCapacityError):
        tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    with pytest.raises(tx.RecoveryCapacityError):
        _start(tmp_path)


def test_status_below_limit_matches_admission_admitting(tmp_path):
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit - 2)
    context = _start(tmp_path)
    v2.finish(context, "completed", seal=True)
    report = tx.recovery_capacity_status(tmp_path)
    assert report["observed_entries"] == 255 and report["at_limit"] is False
    assert report["protected_reasons"]["new_protocol_sealed"] == 1
    tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    assert tx.recovery_capacity_status(tmp_path)["at_limit"] is True


def test_status_bound_spans_both_namespaces(tmp_path, monkeypatch):
    """256 inspected in TOTAL plus one sentinel, never 256 per namespace."""
    _fill_legacy(tmp_path, 200)
    v2_dir = tx.canonical_root(tmp_path) / tx.V2_OPERATION_DIR
    v2_dir.mkdir(mode=0o700)
    for index in range(100):
        (v2_dir / f"{index + 1000:032x}.json").write_text("{}")
    original = tx.read_record
    reads: list[Path] = []

    def counted(path: Path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(tx, "read_record", counted)
    report = tx.recovery_capacity_status(tmp_path)
    assert report["observed_entries"] == 257 and report["scan_complete"] is False
    assert report["namespace_entries"] == {tx.OPERATION_DIR: 200, tx.V2_OPERATION_DIR: 57}
    assert report["at_limit"] is True and len(reads) == 256
    assert report["protected_reasons"]["uninspected_overflow"] == 1


def test_status_reports_unsafe_v2_directory_without_following_it(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / f"{1:032x}.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(elsewhere, root / tx.V2_OPERATION_DIR)
    report = tx.recovery_capacity_status(root)
    assert report["scan_complete"] is False
    assert report["protected_reasons"] == {"unsafe_namespace_directory": 1}
    assert report["namespace_entries"][tx.V2_OPERATION_DIR] == 0
    with pytest.raises(tx.InstallTransactionError, match="unsafe"):
        tx.start_operation(root, baseline="A", target="B", route="default-cli")
    # Q1: counted before the legacy directory is created, so nothing is left behind.
    assert not (root / tx.OPERATION_DIR).exists()


def test_legacy_only_boundary_is_unchanged(tmp_path):
    """255 legacy entries still admit the 256th through the old path."""
    _fill_legacy(tmp_path, tx.RecoveryCapacityError.active_limit - 1)
    tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")
    with pytest.raises(tx.RecoveryCapacityError, match="at capacity"):
        tx.start_operation(tmp_path, baseline="A", target="B", route="default-cli")


def test_sealed_v2_records_still_occupy_their_slot(tmp_path):
    """Sealing is not retirement: nothing here frees capacity."""
    context = _start(tmp_path)
    v2.finish(context, "completed", seal=True)
    assert v2.active_recovery_entries(tmp_path) == 1
    assert _record_path(tmp_path, context).exists()


def test_v2_record_mode_is_owner_only(tmp_path):
    context = _start(tmp_path)
    assert os.stat(_record_path(tmp_path, context)).st_mode & 0o077 == 0
