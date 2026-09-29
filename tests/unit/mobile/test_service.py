"""The first-run refusal has to name a command that works on THIS host (A23).

``lop mobile serve`` with no password exits 2 and tells the operator what to do
about it. That sentence used to be the same one line everywhere — "Run `lop
mobile install`" — which is a command that cannot work on a host with no user
service supervisor at all (a container, OpenRC/Alpine, a Linux without systemd):
``mobile.install`` refuses there with ``supervisors.no_supervisor_error`` and
names the foreground command instead. The store in the sentence was
macOS-shaped for the same reason: "the Keychain" is only one of the three
stores ``mobile.auth`` now chooses between, and ``store_description()`` is the
single place that knows which.

The last test pins a different seam: the refusal path returns 2 *before* the
daemon is constructed, so nothing in this file (or the propagation suite, which
calls the method directly) executes ``amain``'s display-settings wiring.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator import supervisors
from local_operator.mobile import service


def _first_run_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    supervisor: str | None,
    store: str,
) -> str:
    """Run ``amain`` with no password and return what it printed."""
    monkeypatch.setattr(service, "load_password", lambda: None)
    monkeypatch.setattr(service, "store_description", lambda: store)
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisor)

    assert asyncio.run(service.amain()) == 2, "an unauthenticated daemon must not bind"
    return capsys.readouterr().out


def test_a_host_with_a_supervisor_is_told_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _first_run_message(monkeypatch, capsys, supervisor="launchctl", store="the test store")

    assert "`lop mobile install`" in out
    assert "the test store" in out
    assert "LOP_MOBILE_PASSWORD" in out


def test_a_host_without_a_supervisor_is_not_sent_to_the_installer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _first_run_message(monkeypatch, capsys, supervisor=None, store="a 0600 file (/tmp/pw)")

    assert "`lop mobile install` cannot run here" in out
    assert "`lop mobile password`" in out
    assert "`lop mobile serve`" in out
    assert "a 0600 file (/tmp/pw)" in out


def test_the_store_in_the_message_is_the_platforms_own(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read from ``auth`` rather than spelled here, so it cannot drift again."""
    calls: list[int] = []
    real = service.store_description

    def counted() -> str:
        calls.append(1)
        return real()

    monkeypatch.setattr(service, "load_password", lambda: None)
    monkeypatch.setattr(service, "store_description", counted)
    monkeypatch.setattr(supervisors, "supervisor", lambda: "launchctl")

    asyncio.run(service.amain())

    assert calls == [1]
    assert real() in capsys.readouterr().out


def test_amain_wires_the_daemon_into_the_display_settings_follower() -> None:
    """The daemon boot must subscribe the cross-process ``display.*`` follower.

    ``service.amain``'s ``daemon.watch_display_settings()`` is the ONE seam no
    other test executes: the propagation tests call the method directly, and
    every test above returns 2 at the no-password refusal *before* the daemon
    is constructed. Deleting the call re-opens the daemon half of design
    review round 1's D1 with the suite green (agent review round 2, M1), so pin
    it the way this repo pins entry-point wiring — against ``amain``'s own
    source.
    """
    import inspect

    source = inspect.getsource(service.amain)
    assert "daemon.watch_display_settings()" in source, (
        "service.amain must call daemon.watch_display_settings(): without it a "
        "running relay keeps the display value from its first read until a "
        "restart, which is the D1 defect the follower exists to fix."
    )
