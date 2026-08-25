"""Resource-usage probing for `lop sessions`, with an injected subprocess.

The whole point of the helper is graceful degradation: a probe that fails, a
pid that vanished, or an unparseable line must yield ``None`` for that number
and never fail the listing. These drive the pure parser + the injected runner
seam so no real ``ps``/``top`` is spawned.
"""

from __future__ import annotations

import sys

from local_operator.mobile.resources import (
    ResourceUsage,
    _parse_mem_size,
    session_resource_usage,
)


def test_parses_batched_ps_rss() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            # `ps -o pid=,rss=` output: pid then rss in KiB.
            return 0, "111 2048\n222 4096\n"
        return 1, ""  # top/footprint unavailable here

    usage = session_resource_usage([111, 222], runner=runner)
    assert usage[111].rss_bytes == 2048 * 1024
    assert usage[222].rss_bytes == 4096 * 1024
    # Footprint degrades to None when the platform probe returns nothing.
    if sys.platform == "darwin":
        assert usage[111].footprint_bytes is None


def test_darwin_footprint_from_top(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    def runner(argv: list[str]) -> tuple[int, str]:
        if argv[0] == "ps":
            return 0, "111 1024\n"
        if argv[0] == "top":
            # `top -l1 -stats pid,mem`: a header block then PID / MEM rows.
            return 0, "PID    MEM\n111    197M\n999    5M\n"
        return 1, ""

    usage = session_resource_usage([111], runner=runner)
    assert usage[111].rss_bytes == 1024 * 1024
    assert usage[111].footprint_bytes == 197 * 1024 * 1024


def test_degrades_to_none_when_every_probe_fails() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        return 1, ""  # ps and top both unavailable / non-zero

    usage = session_resource_usage([111, 222], runner=runner)
    # Every requested pid still gets an entry — a live session is never dropped.
    assert set(usage) == {111, 222}
    assert usage[111] == ResourceUsage(rss_bytes=None, footprint_bytes=None)


def test_runner_that_raises_is_swallowed() -> None:
    def runner(argv: list[str]) -> tuple[int, str]:
        raise OSError("boom")

    # A raising runner must not propagate — the listing degrades, not crashes.
    # (session_resource_usage guards each call; a raising injected runner is
    # caught by the callers the same way a non-zero return is.)
    try:
        usage = session_resource_usage([111], runner=runner)
    except OSError:  # pragma: no cover - this is the failure we assert against
        raise AssertionError("a failing probe must not raise")
    assert usage[111].rss_bytes is None


def test_empty_pid_list_returns_empty() -> None:
    assert session_resource_usage([]) == {}


def test_parse_mem_size_units() -> None:
    assert _parse_mem_size("512B") == 512
    assert _parse_mem_size("2K") == 2048
    assert _parse_mem_size("3M") == 3 * 1024 * 1024
    assert _parse_mem_size("1G") == 1024**3
    assert _parse_mem_size("197M+") == 197 * 1024 * 1024  # top marks growth with '+'
    assert _parse_mem_size("1234") == 1234  # bare bytes
    assert _parse_mem_size("garbage") is None
