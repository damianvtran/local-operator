"""Strict signal authority controls. Synthetic identities NEVER reach OS probes."""
from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample


class Fake:
    def __init__(self):
        self.rows = {1: (0, 1, 1, "wrapper", False), 10: (1, 10, 10, "leader", False)}
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
        return owned.OwnedGroup.capture(10, parent_pid=1, census=self.census, samples=self.samples,
            identity=self.identity, scheme=lambda: self.scheme, clock=self.clock,
            signal_group=lambda pg, sig: self.calls.append((pg, sig)), deadline=5)


@pytest.mark.parametrize("case", ["reuse", "unknown", "mixed", "unreadable", "terminal", "expired"])
def test_refuses_group_without_fresh_complete_enrolled_live_identity(monkeypatch, case):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    if case == "reuse": fake.rows[10] = (1, 10, 10, "foreign", False)
    if case == "unknown": fake.rows[20] = (1, 10, 10, "new", False)
    if case == "mixed": fake.rows[10] = (1, 10, 99, "leader", False)
    if case == "unreadable": fake.rows[10] = (1, 10, 10, None, False)
    if case == "terminal": fake.rows[10] = (1, 10, 10, "leader", True)
    if case == "expired": fake.now = 6
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == []


def test_enrols_only_live_ancestry_before_leader_exit(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    cap = fake.cap()
    fake.rows[20] = (10, 10, 10, "survivor", False)
    assert cap.refresh(deadline=5).status == "VERIFIED"
    del fake.rows[10]
    assert cap.signal(signal.SIGTERM, deadline=5).status == "SIGNALED"
    fake.rows[20] = (1, 10, 10, "reused", False)
    assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
    assert fake.calls == [(10, signal.SIGTERM)]


def test_orphan_census_cannot_mint_survivor(monkeypatch):
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
    if failure == "missing": cap.samples = lambda _: {}
    elif failure == "permission":
        def denied(_): raise PermissionError("fake kernel denial")
        cap.identity = denied
    elif failure == "sid": fake.rows[10] = (1, 10, 0, "leader", False)
    elif failure == "parent":
        fake.rows[20] = (10, 10, 10, "child", False)
        fake.rows[10] = (1, 10, 10, None, False)
    elif failure == "scheme": fake.scheme = "changed"
    else: cap.census = lambda *args: None
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
