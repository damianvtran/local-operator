"""Strict signal authority controls. Synthetic identities NEVER reach OS probes."""

from __future__ import annotations

import os
import signal

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample


class Fake:
    def __init__(self):
        #: birth is `str | None` on purpose: an unreadable birth is a fact the
        #: capability has to refuse, so the fixture must be able to express it.
        self.rows: dict[int, tuple[int, int, int, str | None, bool]] = {
            os.getpid(): (0, os.getpid(), os.getpid(), "wrapper", False),
            10: (os.getpid(), 10, 10, "leader", False),
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
            parent_pid=os.getpid(),
            census=self.census,
            samples=self.samples,
            identity=self.identity,
            scheme=lambda: self.scheme,
            clock=self.clock,
            signal_group=lambda pg, sig: self.calls.append((pg, sig)),
            deadline=5,
        )


@pytest.mark.parametrize(
    "case,refusal_class",
    [
        # CONTRACT: a readable FOREIGN birth is reuse -- the world changed.
        ("mismatch", "pin_lost"),
        # Everything else is the READER failing to read, and must not be spelled
        # as `pin_lost`: a missing row is not proof of a reap, and `is_birth(...)
        # is None` is not False.
        ("missing-row", "unproven"),
        ("mixed", "unproven"),
        ("unreadable", "unproven"),
        ("expired", "unproven"),
    ],
)
def test_refuses_with_the_class_the_contract_assigns(monkeypatch, case, refusal_class):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    if case == "mismatch":
        fake.rows[10] = (os.getpid(), 10, 10, "foreign", False)
    if case == "missing-row":
        del fake.rows[10]
    if case == "mixed":
        fake.rows[10] = (os.getpid(), 10, 99, "leader", False)
    if case == "unreadable":
        fake.rows[10] = (os.getpid(), 10, 10, None, False)
    if case == "expired":
        fake.now = 6
    verdict = cap.signal(signal.SIGKILL, deadline=5)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == refusal_class, verdict.reason
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


def test_an_empty_complete_census_under_a_held_witness_is_absence(monkeypatch):
    """CONTRACT: held (zombie) + empty-complete -> OBSERVED_ABSENT.

    The witness must still be READABLE: the leader is a retained corpse, the census
    completed and listed no live member. Deleting the leader row instead is the
    unreadable column (next test), and the two must never collapse.
    """
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    fake.rows[10] = (os.getpid(), 10, 10, "leader", True)  # retained corpse
    fake.rows.pop(20, None)
    verdict = cap.signal(signal.SIGKILL, deadline=5)
    assert verdict.status == "OBSERVED_EMPTY", verdict.reason
    assert verdict.refusal_class is None
    assert fake.calls == []


def test_a_missing_witness_row_refuses_rather_than_reporting_absence(monkeypatch):
    """CONTRACT: unreadable column. A missing entry is not proof of a reap."""
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    del fake.rows[10]
    verdict = cap.signal(signal.SIGKILL, deadline=5)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "unproven", verdict.reason
    assert fake.calls == []


def test_an_unreadable_census_still_refuses(monkeypatch):
    """The other half of the pair: unreadable is not empty."""
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()

    def unreadable(*args):
        raise owned.Unknown("unreadable /proc row")

    cap.census = unreadable
    verdict = cap.signal(signal.SIGKILL, deadline=5)
    assert verdict.status == "REFUSED", verdict.reason
    assert fake.calls == []


def test_capture_requires_live_creator_not_numeric_group(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    fake.rows[10] = (99, 10, 10, "leader", False)
    cap = fake.cap()
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []
