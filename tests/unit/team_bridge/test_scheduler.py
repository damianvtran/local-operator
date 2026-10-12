"""H3 A5: serialized RR selector — fairness, request priority, no amplifier."""

from __future__ import annotations

from typing import Any, Mapping

from local_operator.team_bridge.scheduler import select_next_screening_target


def _item(sender: str, kind: str, ident: int) -> dict[str, Any]:
    # Canonical H2 DTO: sender a SIBLING of message; type nested in message.
    return {"message": {"type": kind, "id": ident}, "sender": {"account_id": sender}}


def test_rotation_requests_first_and_single_target() -> None:
    items = [_item("a", "note", 1), _item("a", "request", 2), _item("b", "note", 3)]
    first = select_next_screening_target(items)
    assert first is not None and first["message"]["id"] == 2  # requests first within a
    second = select_next_screening_target(items, last_sender="a")
    assert second is not None and second["message"]["id"] == 3
    third = select_next_screening_target(items, last_sender="b")
    assert third is not None and third["message"]["id"] == 2


def test_no_starvation_and_empty() -> None:
    items = [_item("a", "note", 1), _item("b", "note", 2)]
    seen: list[Any] = []
    for last in (None, "a", "b"):
        nxt = select_next_screening_target(items, last_sender=last)
        assert nxt is not None
        seen.append(nxt["message"]["id"])
    assert seen[0] != seen[1], "one sender can never drain the rotation"
    assert select_next_screening_target([]) is None


def test_malformed_sender_forms_own_slot() -> None:
    items: list[dict[str, Any]] = [
        {"message": {"type": "note", "id": 9}, "sender": "not-a-mapping"},
        _item("a", "note", 1),
    ]
    first = select_next_screening_target(items, last_sender="a")
    assert first is not None and first["message"]["id"] == 9  # unknown sender never merges


def test_account_cycle_serializes_competing_holders() -> None:
    import asyncio
    import tempfile
    from pathlib import Path

    from local_operator.team_bridge.scheduler import AccountScreeningCycle
    from local_operator.team_bridge.store import OwnerReceipt, Store
    from local_operator.team_bridge.types import OwnerSnapshot

    root = Path(tempfile.mkdtemp())
    store = Store(root)
    owner = OwnerSnapshot(7, "acc_a", "tnt_1", "bep_1", "https://qa.invalid/v1", 1).validated()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "r"))

    first = AccountScreeningCycle(store=store, owner=owner, holder="w1")
    second = AccountScreeningCycle(store=store, owner=owner, holder="w2")
    order: list[str] = []

    async def drive() -> None:
        gate = asyncio.Event()

        async def slow_screen(target: Mapping[str, Any]) -> str:
            order.append(f"w1:{target['message']['id']}")
            await gate.wait()
            return "done"

        run = asyncio.create_task(first.run_once([_item("a", "note", 1)], slow_screen))
        await asyncio.sleep(0.05)
        # A competing holder gets NOTHING while the lease is live.
        assert await second.run_once([_item("a", "note", 1)], slow_screen) is None
        gate.set()
        assert await run == "done"

    asyncio.run(drive())
    assert order == ["w1:1"]
    # The cursor advanced durably; the lease is released afterwards.
    schedule = store.screen_schedule(owner)
    assert schedule is not None and schedule["last_sender"] == "a"
    assert schedule["lease_holder"] is None


def test_lost_ownership_refuses_a_swallowed_cancellation_result(tmp_path) -> None:
    """R9/Q3-6: a callback that SUPPRESSES the cancellation can return a
    value while ownership is already lost — run_once must refuse that
    result (None) rather than record it under the successor's lease."""

    import asyncio as _asyncio

    from local_operator.team_bridge.scheduler import AccountScreeningCycle
    from tests.unit.team_bridge.test_screening import _setup

    store, owner = _setup(tmp_path)
    cycle = AccountScreeningCycle(store=store, owner=owner, holder="w1")

    async def swallowed(items, screen, turn):
        try:
            await _asyncio.sleep(5)
        except _asyncio.CancelledError:
            # Suppressed on purpose: the OLD code returned this value.
            return "delivered-anyway"
        return "delivered"

    cycle._run_body = swallowed  # type: ignore[method-assign]

    async def keeper(turn, work, lost):
        await _asyncio.sleep(0.05)
        lost["flag"] = True
        work.cancel()
        try:
            await work
        except BaseException:
            pass

    cycle._keep_turn = keeper  # type: ignore[method-assign]

    async def forbidden_screen(_target):
        raise AssertionError("screen must not run in this control")

    result = _asyncio.run(cycle.run_once([], forbidden_screen))
    assert result is None
