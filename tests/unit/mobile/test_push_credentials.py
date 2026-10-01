"""The coalesced credential report: its bounds, its batching, its one event.

Push/ack-sync S4c part 1; ADR 0006 §4 rule 2 ("recompute on every authenticated
request, SEND coalesced") and §3.2's report block.

What this file pins is the COALESCING, not the transport. The transport is the
injected seam — a recorder here, the cloud POST in part 2 — so every cell below
exercises the state machine the ADR bounds and nothing else: change-triggered,
at most once per device per five minutes, batched for the whole computer, one
credential-change event per rotation, and a fifteen-minute heartbeat that reports
regardless because it is the call that catches a device whose credential lapsed
while its app was shut.

The clock is injected and the loader is a mutable dict, so no cell here sleeps,
touches a network, or depends on how long the suite took to reach it (the repo's
"wait on the event, never on the clock" rule applies to the tests of a timing
rule too).
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.mobile import push_credentials
from local_operator.mobile.push_credentials import (
    CARRIER_EMIT,
    CARRIER_HEARTBEAT,
    CARRIER_REGISTER,
    HEARTBEAT_INTERVAL_S,
    MIN_REPORT_INTERVAL_S,
    CredentialFact,
    CredentialReport,
)

#: A fixed instant every cell reasons at, so nothing depends on wall time.
NOW = 1_780_000_000


class Rig:
    """One report over a mutable fact set, with the transport recorded.

    ``sends`` is what a test asserts on: a list of ``(carrier, block)`` pairs, so
    "nothing was sent" and "one report rode the register" are both one assertion
    and the carrier the seam received is visible.
    """

    def __init__(self) -> None:
        self.facts: dict[str, CredentialFact] = {}
        self.sends: list[tuple[str, dict[str, Any]]] = []
        #: The instant the report believes it is. Advanced explicitly so no cell
        #: here sleeps, and so a cell that means to test the five-minute bound
        #: moves the clock it is testing rather than waiting on a real one.
        self.now = float(NOW)
        self.report = CredentialReport(self._transport, loader=self._load, clock=lambda: self.now)

    def _transport(self, block: dict[str, Any], *, carrier: str) -> None:
        self.sends.append((carrier, block))

    def _load(self, now: float) -> Sequence[CredentialFact]:
        return list(self.facts.values())

    def device(
        self,
        device_id: str,
        *,
        expires_at: int | None,
        live: bool = True,
        expired_at: int | None = None,
        authenticated_at: int | None = None,
    ) -> None:
        """Put one device in the registry the loader reads."""
        self.facts[device_id] = CredentialFact(
            device_id=device_id,
            state="live",
            credential_live=live,
            credential_expires_at=expires_at,
            last_authenticated_at=authenticated_at,
            expired_at=expired_at,
        )

    @property
    def carriers(self) -> list[str]:
        return [carrier for carrier, _block in self.sends]

    @property
    def rows(self) -> list[str]:
        """The device ids in the LAST report sent — the one a cell just asserted."""
        return [row["device_id"] for row in self.sent_rows]

    @property
    def sent_rows(self) -> list[dict[str, Any]]:
        """The device rows of the last report, read back through the wire seam."""
        if not self.sends:
            return []
        return push_credentials.block_devices(self.sends[-1][1])

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_changed_fact_rides_the_next_carrier_and_an_unchanged_one_does_not() -> None:
    """The change trigger, which is what keeps an idle computer silent."""
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000)

    # The first opportunity reports the device: to a process that has just
    # started it is new, and there is no "already reported" state to inherit.
    first = rig.report.piggyback(CARRIER_REGISTER)
    assert first is not None and rig.carriers == [CARRIER_REGISTER]
    assert rig.rows == ["device-a"]

    # The same facts again: unchanged, so the next opportunity sends nothing.
    # This is the half that makes the report O(devices), not O(requests).
    assert rig.report.refresh() == ()
    assert rig.report.due() is False
    assert rig.report.piggyback(CARRIER_EMIT) is None
    assert rig.carriers == [CARRIER_REGISTER]

    # A fact MOVES (the phone authenticated again) and the report rides the next
    # carrier — no timer decided it, and the five-minute bound has passed.
    rig.advance(MIN_REPORT_INTERVAL_S)
    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW)
    assert rig.report.refresh() == ("device-a",)
    assert rig.report.piggyback(CARRIER_EMIT) is not None
    assert rig.carriers == [CARRIER_REGISTER, CARRIER_EMIT]


def test_one_device_is_reported_at_most_once_per_five_minutes() -> None:
    """The per-device ceiling, checked on both sides of the boundary.

    A chatty phone changes ``last_authenticated_at`` on every request, so without
    this bound the change trigger alone would report at the request rate.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW)
    rig.report.refresh()
    assert rig.report.piggyback(CARRIER_EMIT) is not None

    # A change inside the window is HELD, not dropped: it is still reportable, it
    # is just not reportable yet.
    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW + 1)
    assert rig.report.refresh() == ("device-a",)
    assert rig.report.due() is False
    assert rig.report.piggyback(CARRIER_EMIT) is None

    # ...and at the boundary it goes, carrying the newer fact.
    rig.advance(MIN_REPORT_INTERVAL_S)
    assert rig.report.due() is True
    block = rig.report.piggyback(CARRIER_EMIT)
    assert block is not None
    assert push_credentials.block_devices(block)[0]["last_authenticated_at"] == NOW + 1
    assert rig.carriers == [CARRIER_EMIT, CARRIER_EMIT]


def test_every_device_of_the_computer_travels_in_one_report() -> None:
    """Batched per computer: one call names every device that has something to say."""
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000)
    rig.device("device-b", expires_at=NOW + 1000)
    rig.device("device-c", expires_at=NOW + 1000)
    rig.report.refresh()

    block = rig.report.piggyback(CARRIER_REGISTER)

    assert rig.carriers == [CARRIER_REGISTER], "three devices, ONE call"
    assert block is not None
    assert [row["device_id"] for row in push_credentials.block_devices(block)] == [
        "device-a",
        "device-b",
        "device-c",
    ]


def test_the_heartbeat_reports_an_unchanged_fact_once_the_window_has_passed() -> None:
    """The ceiling on silence, on both sides of it."""
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 10_000_000)
    rig.report.refresh()
    assert rig.report.piggyback(CARRIER_REGISTER) is not None

    rig.advance(HEARTBEAT_INTERVAL_S - 1)
    assert rig.report.heartbeat_due() is False
    assert rig.report.heartbeat() is None

    rig.advance(HEARTBEAT_INTERVAL_S)
    block = rig.report.heartbeat()
    assert block is not None
    assert rig.carriers == [CARRIER_REGISTER, CARRIER_HEARTBEAT]
    # The row's own value: the block names the device once per report, and this is
    # the SECOND report, so a row count is not what the bound is about.
    assert rig.rows == ["device-a"]
    assert len(push_credentials.block_devices(block)) == 1


def test_a_fresh_process_reports_rather_than_assume_the_cloud_already_knows() -> None:
    """An unknown last-send counts as due: silence is the failure being closed."""
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000)
    assert rig.report.heartbeat_due() is True


def test_the_heartbeat_carries_a_device_whose_credential_lapsed_while_it_was_shut() -> None:
    """The case the heartbeat exists for, and the reason the lapse is DERIVED.

    No request arrives while the app is shut, so nothing observes the lapse. The
    stored expiry still says when the cookie died, and the heartbeat re-reads it —
    so the report says not-live at the report instant. A coalescer that sent the
    fact it last recorded would keep telling the cloud to deliver.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 60)
    rig.report.refresh()
    opening = rig.report.piggyback(CARRIER_REGISTER)
    assert opening is not None
    assert push_credentials.block_devices(opening)[0]["credential_live"] is True

    # A week passes with the phone shut. The STORED fact is unchanged...
    rig.advance(7 * 24 * 3600)
    assert rig.report.refresh() == ()
    # ...and the derived answer is not, so the heartbeat carries the correction.
    block = rig.report.heartbeat()
    assert block is not None
    assert push_credentials.block_devices(block)[0]["credential_live"] is False


def test_a_standing_marker_outranks_a_future_expiry() -> None:
    """Rotation first: the cookie died even though its nominal expiry has not.

    A rotation kills every cookie while each ``credential_expires_at`` is still
    days away, so a report that read the arithmetic first would tell the cloud to
    keep delivering to a device the machine has already cut off.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 10_000, expired_at=NOW)

    block = rig.report.heartbeat()

    assert block is not None
    assert push_credentials.block_devices(block)[0]["credential_live"] is False


def test_a_rotation_is_exactly_one_event_however_many_devices_it_lapses() -> None:
    """ADR §4 rule 2's own bound, and the reason the API takes the whole set."""
    rig = Rig()
    for name in ("device-a", "device-b", "device-c"):
        rig.device(name, expires_at=NOW + 10_000, expired_at=NOW)

    event = rig.report.note_rotation(("device-a", "device-b", "device-c"))

    assert event is not None
    assert len(rig.report.pending_events()) == 1
    assert rig.report.pending_events()[0].devices == ("device-a", "device-b", "device-c")
    assert rig.report.pending_events()[0].kind == push_credentials.EVENT_ROTATION
    assert rig.report.pending_events()[0].at == NOW


def test_a_rotation_of_a_computer_with_no_devices_records_no_event() -> None:
    """Nothing changed, so there is nothing for the cloud to act on."""
    rig = Rig()
    assert rig.report.note_rotation(()) is None
    assert rig.report.pending_events() == ()
    assert rig.report.due() is False


def test_the_pending_events_clear_once_they_have_been_reported() -> None:
    """An event is a pending reason to report, not a log that grows."""
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 10_000, expired_at=NOW)
    rig.report.note_rotation(("device-a",))
    assert len(rig.report.pending_events()) == 1

    assert rig.report.piggyback(CARRIER_EMIT) is not None

    assert rig.report.pending_events() == ()


def test_the_report_block_is_spelled_in_exactly_one_place() -> None:
    """The wire seam, pinned: the field names come from the module's constants.

    This is the cell that fails if a caller hardcodes a name beside the constants
    — the failure mode the TODO's rebinding is designed around, because a rebind
    that moves the constants but not a literal would otherwise be silent.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW)

    block = rig.report.heartbeat()

    assert block is not None
    assert set(block) == {push_credentials.REPORT_DEVICES_FIELD}
    row = push_credentials.block_devices(block)[0]
    assert set(row) == set(push_credentials.REPORT_DEVICE_FIELDS)
    assert row == {
        "device_id": "device-a",
        "credential_live": True,
        "credential_expires_at": NOW + 1000,
        "last_authenticated_at": NOW,
    }
    # The route the heartbeat call would use is named here and nowhere else.
    assert push_credentials.REPORT_ROUTE == "/v1/push/credentials"


def test_the_recorded_fact_is_the_one_that_was_sent_not_the_one_that_exists() -> None:
    """A fact that changed AFTER a report is still reportable at the next window.

    The coalescer compares against what it SENT, not against the current value:
    recording the current value at send time would mark the newer change as
    already reported and silently drop it.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW)
    rig.report.refresh()
    assert rig.report.piggyback(CARRIER_EMIT) is not None

    rig.device("device-a", expires_at=NOW + 1000, authenticated_at=NOW + 5)
    rig.advance(MIN_REPORT_INTERVAL_S)
    assert rig.report.refresh() == ("device-a",)
    block = rig.report.piggyback(CARRIER_EMIT)
    assert block is not None
    assert push_credentials.block_devices(block)[0]["last_authenticated_at"] == NOW + 5


def test_the_loader_is_the_only_source_of_facts() -> None:
    """A device the registry no longer names cannot ride a report.

    The loader is re-read on every opportunity, so a revoked-and-dropped row stops
    being reported without the coalescer being told: there is no second copy of
    the device list here to go stale.
    """
    rig = Rig()
    rig.device("device-a", expires_at=NOW + 1000)
    rig.device("device-b", expires_at=NOW + 1000)
    rig.report.refresh()

    del rig.facts["device-b"]
    rig.advance(MIN_REPORT_INTERVAL_S)
    block = rig.report.piggyback(CARRIER_EMIT)

    assert block is not None
    assert rig.rows == ["device-a"]


def test_the_bounds_are_the_adrs_numbers() -> None:
    """The ceilings are the contract's, and a test that changed one would be a
    silent renegotiation of what the cloud is promised."""
    assert MIN_REPORT_INTERVAL_S == 300
    assert HEARTBEAT_INTERVAL_S == 900
    assert {CARRIER_REGISTER, CARRIER_EMIT, CARRIER_HEARTBEAT} == {
        "register",
        "emit",
        "heartbeat",
    }


def test_a_device_with_no_credential_fact_is_reported_not_live() -> None:
    """Absence is not evidence, at the report's own level."""
    rig = Rig()
    rig.device("device-a", expires_at=None, live=False)
    block = rig.report.heartbeat()

    assert block is not None
    row = push_credentials.block_devices(block)[0]
    assert row["credential_live"] is False
    # The absent fields stay ABSENT rather than becoming a null or a zero the
    # cloud could read as a value (the repo's absence rule, §3.1's own precedent).
    assert "credential_expires_at" not in row
    assert "last_authenticated_at" not in row


def test_the_report_takes_its_facts_from_the_registry_loaders_shape() -> None:
    """``from_facts`` reads the reduced row ``push_devices`` hands out."""
    fact = CredentialFact.from_facts(
        {
            "device_id": "device-a",
            "state": "expired",
            "credential_live": False,
            "credential_expires_at": NOW,
            "last_authenticated_at": NOW - 10,
            "expired_at": NOW,
        }
    )
    assert fact == CredentialFact(
        device_id="device-a",
        state="expired",
        credential_live=False,
        credential_expires_at=NOW,
        last_authenticated_at=NOW - 10,
        expired_at=NOW,
    )
    assert fact.live_at(NOW - 1) is False
    # A row an earlier build wrote: no expiry, only the flag.
    earlier = CredentialFact.from_facts(
        {"device_id": "device-b", "state": "live", "credential_live": True}
    )
    assert earlier.live_at(NOW) is True
    assert earlier.credential_expires_at is None


def test_the_lapse_rule_is_not_restated_here() -> None:
    """One rule, in ``push_devices``: this module asks, it does not decide.

    Asserted by behaviour rather than by import: the fact delegates, so a marker
    the fact carries still outranks an arithmetic answer — which is only true if
    the delegation reaches the registry's rule and not a private copy of it.
    """
    lapsed = CredentialFact(
        device_id="device-a",
        state="expired",
        credential_live=True,
        credential_expires_at=NOW + 10_000,
        expired_at=NOW,
    )
    assert lapsed.live_at(NOW) is False

    # Through the rig's own registry, so the heartbeat re-reads the loaded fact
    # rather than a value handed straight to ``observe``.
    rig = Rig()
    rig.facts["device-a"] = lapsed
    assert rig.report.heartbeat() is not None
    assert rig.sent_rows[0]["credential_live"] is False
