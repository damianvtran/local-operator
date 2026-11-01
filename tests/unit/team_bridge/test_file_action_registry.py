"""Synthetic registry mechanics only: no authenticated response or route authority."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from local_operator.team_bridge import file_action_registry as actions


def binding(**changes):
    return replace(actions.InertActionBinding(
        credential_id=1, account_id="acc_synthetic", login_incarnation="a" * 32,
        tenant_id="tenant_synthetic", window_id="window_synthetic", view_generation=4,
        conversation_id="conversation_synthetic", message_id="message_synthetic",
        file_ids=frozenset({"file_one", "file_two"}),
    ), **changes)


def issue(registry, candidate, *, ttl_seconds):
    return registry.commission(candidate, ttl_seconds=ttl_seconds, expected_epoch=registry.current_epoch())


def test_claim_once_and_registry_reader_tracks_cancel():
    registry = actions.InertFileActionRegistry()
    context = issue(registry, binding(), ttl_seconds=30)
    with pytest.raises(actions.ActionUnavailable):
        registry.start(context, "file_other")
    operation = registry.start(context, "file_one")
    with pytest.raises(actions.ActionUnavailable):
        registry.current(operation)
    view = registry.claim(operation)
    assert view.file_id == "file_one" and view.binding == binding()
    assert registry.current(operation) == view
    with pytest.raises(actions.ActionUnavailable):
        registry.claim(operation)
    registry.cancel(operation)
    registry.cancel(operation)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(operation)
    assert context not in repr(registry) and operation not in repr(view)


def test_expiry_revoke_and_view_invalidation_are_scoped():
    clock = [100.0]
    registry = actions.InertFileActionRegistry(clock=lambda: clock[0])
    first = issue(registry, binding(), ttl_seconds=10)
    other = issue(registry, binding(view_generation=5), ttl_seconds=20)
    op = registry.start(first, "file_one")
    other_op = registry.start(other, "file_two")
    registry.claim(op)
    registry.claim(other_op)
    registry.invalidate_view("window_synthetic", 4)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(op)
    assert registry.current(other_op).file_id == "file_two"
    clock[0] = 120.0
    with pytest.raises(actions.ActionUnavailable):
        registry.current(other_op)
    registry.revoke(other)


def test_one_use_claim_is_serialized():
    registry = actions.InertFileActionRegistry()
    operation = registry.start(issue(registry, binding(), ttl_seconds=30), "file_one")

    def claim():
        try:
            registry.claim(operation)
            return True
        except actions.ActionUnavailable:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(lambda _: claim(), range(32))) == 1


@pytest.mark.parametrize("change", [
    {"credential_id": 0}, {"credential_id": True}, {"account_id": ""},
    {"login_incarnation": "not-a-login"}, {"view_generation": -1},
    {"file_ids": frozenset()}, {"file_ids": frozenset(str(i) for i in range(17))},
    {"file_ids": frozenset({"bad\nfile"})},
])
def test_invalid_candidate_fails_closed(change):
    with pytest.raises(actions.ActionUnavailable):
        issue(actions.InertFileActionRegistry(), binding(**change), ttl_seconds=30)


@pytest.mark.parametrize("ttl", [0, -1, 61, float("nan"), float("inf"), True])
def test_invalid_ttl_fails_closed(ttl):
    with pytest.raises(actions.ActionUnavailable):
        issue(actions.InertFileActionRegistry(), binding(), ttl_seconds=ttl)


def test_capacity_is_bounded_and_expiry_reclaims_space(monkeypatch):
    monkeypatch.setattr(actions, "_MAX_CONTEXTS", 2)
    monkeypatch.setattr(actions, "_MAX_OPERATIONS", 1)
    clock = [1.0]
    registry = actions.InertFileActionRegistry(clock=lambda: clock[0])
    context = issue(registry, binding(), ttl_seconds=1)
    issue(registry, binding(), ttl_seconds=1)
    with pytest.raises(actions.ActionUnavailable):
        issue(registry, binding(), ttl_seconds=1)
    registry.start(context, "file_one")
    with pytest.raises(actions.ActionUnavailable):
        registry.start(context, "file_two")
    clock[0] = 2.0
    assert issue(registry, binding(), ttl_seconds=1)


def test_process_local_namespace_and_unknown_clock_fail_closed():
    first = actions.InertFileActionRegistry()
    second = actions.InertFileActionRegistry()
    context = issue(first, binding(), ttl_seconds=10)
    with pytest.raises(actions.ActionUnavailable):
        second.start(context, "file_one")
    with pytest.raises(actions.ActionUnavailable):
        issue(actions.InertFileActionRegistry(clock=lambda: float("nan")), binding(), ttl_seconds=10)


def test_retired_generation_cannot_resurrect_and_new_generation_can():
    registry = actions.InertFileActionRegistry()
    old = issue(registry, binding(), ttl_seconds=30)
    old_op = registry.start(old, "file_one")
    registry.claim(old_op)
    registry.invalidate_view("window_synthetic", 4)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(old_op)
    for generation in (0, 4):
        with pytest.raises(actions.ActionUnavailable):
            issue(registry, binding(view_generation=generation), ttl_seconds=30)
    new = issue(registry, binding(view_generation=5), ttl_seconds=30)
    operation = registry.start(new, "file_two")
    registry.claim(operation)
    registry.revoke(new)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(operation)


def test_watermark_capacity_rotates_and_recovers_without_reviving_old_handles(monkeypatch):
    monkeypatch.setattr(actions, "_MAX_WINDOW_TOMBSTONES", 1)
    registry = actions.InertFileActionRegistry()
    old_epoch = registry.current_epoch()
    context = issue(registry, binding(view_generation=5), ttl_seconds=30)
    operation = registry.start(context, "file_one")
    registry.claim(operation)
    registry.invalidate_view("window_other", 1)
    registry.invalidate_view("window_third", 1)
    assert registry.current_epoch() != old_epoch
    with pytest.raises(actions.ActionUnavailable):
        registry.current(operation)
    with pytest.raises(actions.ActionUnavailable):
        registry.start(context, "file_one")
    with pytest.raises(actions.ActionUnavailable):
        registry.commission(binding(view_generation=6), ttl_seconds=30, expected_epoch=old_epoch)
    fresh = issue(registry, binding(view_generation=6), ttl_seconds=30)
    new_operation = registry.start(fresh, "file_two")
    registry.claim(new_operation)
    registry.revoke(fresh)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(new_operation)


def test_concurrent_revoke_prevents_subsequent_reads():
    from threading import Barrier
    registry = actions.InertFileActionRegistry()
    context = issue(registry, binding(), ttl_seconds=30)
    operation = registry.start(context, "file_one")
    registry.claim(operation)
    barrier = Barrier(2)

    def revoke():
        barrier.wait(timeout=3)
        registry.revoke(context)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(revoke)
        barrier.wait(timeout=3)
        # A read can legitimately win before the revoke acquires the lock.
        try:
            registry.current(operation)
        except actions.ActionUnavailable:
            pass
        worker.result(timeout=3)
    with pytest.raises(actions.ActionUnavailable):
        registry.current(operation)
