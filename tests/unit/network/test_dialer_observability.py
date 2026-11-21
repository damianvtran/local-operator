"""Dial-only relays must explain successful contact and bounded failed redials.

These use the existing serve-shaped two-relay fixture, not the live mesh. The
redial scheduler's clock is synthetic; its connection failures are real TCP
refusals from a reserved, non-listening loopback port, with no real-time sleeps.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import redial, relay
from tests.unit.network.conftest import wait_for
from tests.unit.network.test_relay_e2e import _pair_settled, devices  # noqa: F401


def test_a_real_member_dial_records_the_dialer_and_keeps_the_listener_row(
    devices: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    owner, borrower, host, port = devices
    record, _, _ = _pair_settled(devices, monkeypatch)
    before = borrower.audit.recorded_through
    listener_before = owner.audit.recorded_through
    link, reason = borrower.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    assert wait_for(
        lambda: any(
            row["seq"] > listener_before
            and row["event"] == "link_opened"
            and row["detail"].get("role") == "listener"
            for row in owner.audit.tail()
        )
    ), "the accepting relay never published its listener row"
    dialer = [
        row
        for row in borrower.audit.tail()
        if row["seq"] > before
        and row["event"] == "link_opened"
        and row["detail"].get("role") == "dialer"
    ]
    assert len(dialer) == 1, "the successful real dial left no dialer audit row"
    assert dialer[0]["actor"] == owner.identity.device_id
    assert dialer[0]["detail"] == {"role": "dialer", "epoch": record.epoch, "phase": "member"}
    # Fence on the listener's own row above, not on how long its writer takes.
    listener = [
        row
        for row in owner.audit.tail()
        if row["seq"] > listener_before
        and row["event"] == "link_opened"
        and row["detail"].get("role") == "listener"
    ]
    assert len(listener) == 1
    assert listener[0]["actor"] == borrower.identity.device_id
    assert listener[0]["detail"] == {"role": "listener", "epoch": record.epoch, "phase": "member"}


def test_real_closed_port_redials_are_coalesced_without_losing_the_retry_schedule(
    devices: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path  # noqa: F811
) -> None:
    owner, borrower, _, _ = devices
    record, _, _ = _pair_settled(devices, monkeypatch)
    now = [1_000_000.0]
    tracker = redial.RedialTracker(clock=lambda: now[0], rand=lambda: 0.5)
    tracker.seed(owner.identity.device_id)

    class Ticks:
        gaps = iter((0.0, 61.0, 61.0, 301.0))

        def wait(self, _timeout: float) -> bool:
            gap = next(self.gaps, None)
            if gap is None:
                return True
            now[0] += gap
            return False

        def is_set(self) -> bool:
            return False

    # Only the scheduler is reduced. Its attempts still enter the real dial()
    # implementation with the paired identity/network and hit a real closed port.
    scheduler: Any = object.__new__(relay.RelayServer)
    scheduler.root = tmp_path / "scheduler-only"
    scheduler.redial = tracker
    scheduler.audit = borrower.audit
    scheduler.identity = borrower.identity
    scheduler._refusal_rows = redial.RefusalCoalescer(clock=lambda: now[0])
    scheduler._stop = Ticks()
    scheduler._link_for = lambda _device: None
    scheduler._members_named = lambda _device: [object()]
    scheduler._redial_candidates = lambda _member: ["loopback-test-only"]
    causes: list[str] = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]

        def dial(_device: str, **_kwargs: Any) -> tuple[Any, str]:
            link, reason = borrower.dial(
                record.network_id, host=f"127.0.0.1:{port}", epoch=record.epoch, timeout_s=0.5
            )
            assert link is None, "the reserved test port unexpectedly accepted a connection"
            causes.append(reason)
            return link, reason

        scheduler._ensure_link_with_reason = dial
        scheduler._redial_loop()

    assert len(causes) == 4 and causes[0] and len(set(causes)) == 1
    assert tracker.snapshot(owner.identity.device_id)["attempts"] == 4
    rows = [row for row in borrower.audit.tail() if row["event"] == "redial_failed"]
    assert len(rows) == 2, "repeated failures need a coalesced, not absent or per-tick, audit"
    assert [row["detail"]["attempt"] for row in rows] == [1, 4]
    assert rows[0]["detail"]["code"] == causes[0]
    assert rows[0]["detail"]["device"] == owner.identity.device_id
    assert rows[0]["detail"]["next_retry_at"] > 1_000_000.0
    assert rows[1]["detail"]["repeats"] == 2
