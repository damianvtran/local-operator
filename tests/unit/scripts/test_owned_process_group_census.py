"""R5 strict census controls: no real process listing, ctypes call or signal."""
from __future__ import annotations

import ctypes

import pytest

from local_operator import owned_process_group as o


@pytest.mark.parametrize("bad", [-1, 3, 257, None])
def test_malformed_darwin_size_refuses(bad):
    with pytest.raises(o.Unknown):
        o.strict_darwin(10, 5, listpids=lambda *args: bad, clock=lambda: 0)


@pytest.mark.parametrize("bad", [-1, 3, 1025, None])
def test_malformed_darwin_fetch_refuses(bad):
    def call(kind, group, buffer, capacity):
        return 4 if buffer is None else bad
    with pytest.raises(o.Unknown):
        o.strict_darwin(10, 5, listpids=call, clock=lambda: 0)


def test_growth_beyond_slack_is_not_silently_truncated():
    sizes = []
    def call(kind, group, buffer, capacity):
        if buffer is None: return 4
        sizes.append(capacity)
        if len(sizes) == 1: return capacity
        for n in range(70): buffer[n] = n + 1
        return 70 * ctypes.sizeof(ctypes.c_int)
    assert len(o.strict_darwin(10, 5, listpids=call, clock=lambda: 0)) == 70
    assert len(sizes) == 2 and sizes[1] > sizes[0]


def test_permanent_saturation_is_unknown_not_partial_or_empty():
    def call(kind, group, buffer, capacity):
        return 4 if buffer is None else capacity
    with pytest.raises(o.Unknown, match="saturated"):
        o.strict_darwin(10, 5, listpids=call, clock=lambda: 0)


def test_zero_size_query_still_fetches_for_arrival_race():
    def call(kind, group, buffer, capacity):
        if buffer is None: return 0
        buffer[0] = 10
        return 4
    assert o.strict_darwin(10, 5, listpids=call, clock=lambda: 0) == [10]


@pytest.mark.parametrize("row", ["broken", "22 (mismatch) S 1 10 10", "10 (name) S bad 10 10"])
def test_linux_unparseable_row_refuses(row):
    with pytest.raises(o.Unknown):
        o.strict_linux(10, 5, listdir=lambda _: ["10"], read=lambda _: row, clock=lambda: 0)


def test_linux_unreadable_row_refuses():
    def read(_): raise PermissionError("fake")
    with pytest.raises(o.Unknown, match="unreadable"):
        o.strict_linux(10, 5, listdir=lambda _: ["10"], read=read, clock=lambda: 0)


def test_linux_disappeared_requires_directory_absence():
    snapshots = iter([["10"], []])
    def absent(_): raise FileNotFoundError("fake")
    assert o.strict_linux(10, 5, listdir=lambda _: next(snapshots), read=absent,
                          stat=absent, clock=lambda: 0) == []
    with pytest.raises(o.Unknown, match="existing"):
        o.strict_linux(10, 5, listdir=lambda _: ["10"], read=absent,
                       stat=lambda _: object(), clock=lambda: 0)


def test_linux_disappeared_row_reappearing_cannot_be_omitted():
    def absent(_): raise FileNotFoundError("fake")
    with pytest.raises(o.Unknown, match="unstable /proc census"):
        o.strict_linux(10, 5, listdir=lambda _: ["10"], read=absent,
                       stat=absent, clock=lambda: 0)


def test_fallback_never_spends_unavailable_one_second(monkeypatch):
    monkeypatch.setattr(o.sys, "platform", "darwin")
    monkeypatch.setattr(o.procstate, "_sysctl_samples", lambda _: None)
    monkeypatch.setattr(o.procstate, "process_samples", lambda _: pytest.fail("unbudgeted fallback"))
    with pytest.raises(o.Unknown, match="fallback"):
        o.bounded_samples([10], 0.25, clock=lambda: 0)
