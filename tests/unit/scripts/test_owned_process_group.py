"""Strict signal authority controls. Synthetic identities NEVER reach OS probes."""

from __future__ import annotations

import signal

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample


class Fake:
    def __init__(self):
        #: birth is `str | None` on purpose: an unreadable birth is a fact the
        #: capability has to refuse, so the fixture must be able to express it.
        self.rows: dict[int, tuple[int, int, int, str | None, bool]] = {
            1: (0, 1, 1, "wrapper", False),
            10: (1, 10, 10, "leader", False),
        }
        self.calls = []
        self.now = 0.0
        self.scheme = "fixture"

    def clock(self):
        return self.now

    def census(self, pgid, deadline):
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        return {p: ProcessSample(self.rows[p][4], self.rows[p][3]) for p in pids if p in self.rows}

    def identity(self, pid):
        return self.rows[pid][:3]

    def cap(self):
        return owned.OwnedGroup.capture(
            10,
            parent_pid=1,
            census=self.census,
            samples=self.samples,
            identity=self.identity,
            scheme=lambda: self.scheme,
            clock=self.clock,
            signal_group=lambda pg, sig: self.calls.append((pg, sig)),
            deadline=5,
        )


@pytest.mark.parametrize("case", ["mismatch", "mixed", "unreadable", "reaped", "expired"])
def test_refuses_when_the_reservation_is_not_provably_ours(monkeypatch, case):
    """The pin, not the membership, is what authority rests on.

    Each case destroys the RESERVATION rather than a member's ancestry: a birth
    that is not the receipt's, a group/session the member does not share, an
    unreadable birth, a reaped leader (number released), and an expired budget.
    """
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    if case == "mismatch":
        fake.rows[10] = (1, 10, 10, "foreign", False)
    if case == "mixed":
        fake.rows[10] = (1, 10, 99, "leader", False)
    if case == "unreadable":
        fake.rows[10] = (1, 10, 10, None, False)
    if case == "reaped":
        del fake.rows[10]
    if case == "expired":
        fake.now = 6
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []


def test_a_zombie_leader_holds_the_group_and_its_members_are_ours(monkeypatch):
    """The inversion: a corpse leader keeps the number, so members need no chain."""
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    fake.rows[10] = (1, 10, 10, "leader", True)
    fake.rows[20] = (10, 10, 10, "grandchild", False)
    verdict = cap.signal(signal.SIGTERM, deadline=5)
    assert verdict.status == "SIGNALED", verdict.reason
    assert verdict.anchor == "leader-reservation"
    assert fake.calls == [(10, signal.SIGTERM)]


def test_orphan_census_cannot_mint_survivor(monkeypatch):
    """A member with no reserved number above it is a stranger, not a survivor.

    The old reason was "no ancestry chain"; the reason now is the reservation
    itself: the leader is gone, so the number was released and nothing in it can
    be called ours.
    """
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    fake.rows = {20: (1, 10, 10, "orphan", False)}
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []


@pytest.mark.parametrize("failure", ["missing", "permission", "sid", "parent", "scheme", "census"])
def test_unknown_facts_cannot_be_authority(monkeypatch, failure):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: fake.scheme)
    cap = fake.cap()
    if failure == "missing":
        cap.samples = lambda _: {}
    elif failure == "permission":

        def denied(_):
            raise PermissionError("fake kernel denial")

        cap.identity = denied
    elif failure == "sid":
        fake.rows[10] = (1, 10, 0, "leader", False)
    elif failure == "parent":
        fake.rows[20] = (10, 10, 10, "child", False)
        fake.rows[10] = (1, 10, 10, None, False)
    elif failure == "scheme":
        fake.scheme = "changed"
    else:
        cap.census = lambda *args: None
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []


def test_empty_is_observation_not_a_signal(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    del fake.rows[10]
    assert cap.signal(signal.SIGKILL, deadline=5).status == "OBSERVED_EMPTY"
    assert fake.calls == []


def test_capture_requires_live_creator_not_numeric_group(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    fake.rows[10] = (99, 10, 10, "leader", False)
    cap = fake.cap()
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []
