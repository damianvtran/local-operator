"""A supervised receiver running other code than the installed ``lop`` is restarted.

Reproduces the 2026-11-17 Team Bridge incident shape: a launchd receiver kept a
pre-fix build after ``lop`` was upgraded, and nothing restarted it. The tests
seed a runtime record (or none, for a receiver older than the record) and run
the real surface-start path (``install.sync_connector_service``) with only the
supervisor seams replaced.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import launchd, supervisors, update
from local_operator.team_bridge.connector import install, reap
from local_operator.team_bridge.connector import runtime_record as rr
from local_operator.team_bridge.connector import stale_restart as sr

NOW = 1_800_000_000.0

#: Captured at import, before any fixture replaces it, for the negative control.
_REAL_KICKSTART = launchd.kickstart
_REAL_RECONCILE = install._reconcile


def _obs(**overrides: Any) -> sr.Observation:
    base: dict[str, Any] = {
        "restart_needed": True,
        "reason": "version_mismatch",
        "supervised_pid": 4242,
        "recorded_pid": 4242,
        "uptime_s": sr.MIN_UPTIME_S + 1,
        "installed_version": "0.106.7",
        "last_kick": None,
        "recorded_version": "0.101.42",
    }
    base.update(overrides)
    return sr.Observation(**base)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, sr.RESTARTED),
        ({"reason": "no_record", "recorded_pid": None}, sr.RESTARTED),
        ({"restart_needed": False, "reason": None}, sr.NOT_NEEDED),
        ({"reason": "replaced_build"}, sr.OTHER_REPAIR),
        # Review M1: only a receiver BEHIND the installed build is stale.
        ({"recorded_version": "0.107.0"}, sr.NOT_OLDER),
        ({"recorded_version": "0.106.8.dev1"}, sr.NOT_OLDER),
        ({"recorded_version": None}, sr.NOT_OLDER),
        ({"recorded_version": "0.106.6"}, sr.RESTARTED),
        ({"restart_needed": None}, sr.UNPROVEN),
        ({"supervised_pid": None}, sr.UNPROVEN),
        # The record names a foreground receiver: a kick would bounce the wrong one.
        ({"recorded_pid": 777}, sr.OTHER_PROCESS),
        ({"uptime_s": sr.MIN_UPTIME_S - 1}, sr.TOO_YOUNG),
        ({"uptime_s": None}, sr.TOO_YOUNG),
        (
            {
                "last_kick": {
                    "at": NOW - 10 * sr.MIN_INTERVAL_S,
                    "pid": 4242,
                    "installed": "0.106.7",
                }
            },
            sr.ALREADY_KICKED,
        ),
        # Review m3: a refused attempt backs off for MIN_INTERVAL_S, then retries.
        (
            {"last_kick": {"at": NOW - 60, "pid": 4242, "installed": "0.106.7", "refused": True}},
            sr.RATE_LIMITED,
        ),
        (
            {
                "last_kick": {
                    "at": NOW - sr.MIN_INTERVAL_S - 1,
                    "pid": 4242,
                    "installed": "0.106.7",
                    "refused": True,
                }
            },
            sr.RESTARTED,
        ),
        (
            {"last_kick": {"at": NOW - 60, "pid": 1, "installed": "0.106.6"}},
            sr.RATE_LIMITED,
        ),
        (
            {"last_kick": {"at": NOW - sr.MIN_INTERVAL_S - 1, "pid": 1, "installed": "0.106.6"}},
            sr.RESTARTED,
        ),
    ],
)
def test_decide(overrides: dict[str, Any], expected: str) -> None:
    assert sr.decide(_obs(**overrides), now=NOW) == expected


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A launchd-supervised receiver at THIS test's pid, with the kick recorded.

    Only the supervisor seams are replaced; the record, the verdict
    (``cli._restart_verdict``), the decision and the marker are the real code.
    """

    state: dict[str, Any] = {"installed": "0.106.7", "kicks": [], "pid": os.getpid()}
    monkeypatch.setattr(update, "installed_version", lambda: state["installed"])
    monkeypatch.setattr(update, "generation_staleness_of_process", lambda pid: None)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: pid == state["pid"])
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    monkeypatch.setattr(install, "connector_state", lambda root: install.STATE_LOADED)
    monkeypatch.setattr(launchd, "job_pid", lambda **kw: state["pid"])

    def kickstart(**kw: Any) -> bool:
        state["kicks"].append(kw["label"])
        return True

    monkeypatch.setattr(launchd, "kickstart", kickstart)
    # The tmp root is an isolated store, which `_reconcile` refuses before any
    # supervisor call; the unit itself being current is not what is under test.
    monkeypatch.setattr(
        install, "_reconcile", lambda root, force_off: install.ServiceOutcome(install.CODE_CURRENT)
    )
    state["root"] = tmp_path
    return state


def _seed_record(root: Path, *, pid: int, version: str, started_at: float) -> None:
    path = rr.record_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "pid": pid,
                "version": version,
                "generation": None,
                "consent_versions": [],
                "started_at": started_at,
                "singleton": rr.SINGLETON_HELD,
                "recorded_at": started_at,
            }
        ),
        encoding="utf-8",
    )


def test_surface_start_restarts_a_receiver_on_an_older_version(
    rig: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    root = rig["root"]
    _seed_record(root, pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)

    with caplog.at_level(logging.INFO, logger=sr.__name__):
        outcome = install.sync_connector_service(root)

    assert outcome.code == install.CODE_CURRENT
    assert rig["kicks"] == [install.LABEL]
    marker = json.loads(sr.marker_path(root).read_text(encoding="utf-8"))
    assert marker["pid"] == rig["pid"]
    assert marker["installed"] == "0.106.7"
    assert marker["reason"] == "version_mismatch"
    lines = [r.getMessage() for r in caplog.records if r.name == sr.__name__]
    assert lines == [
        "bridge connector stale restart: trigger=surface code=restarted reason=version_mismatch"
    ]
    # Content-free: no version, pid or path in the line.
    assert "0.10" not in lines[0] and str(rig["pid"]) not in lines[0] and str(root) not in lines[0]

    # Idempotent: the same process is still there (the kick did not take) —
    # asking again is a loop, not a repair.
    install.sync_connector_service(root)
    assert rig["kicks"] == [install.LABEL]


def test_a_receiver_older_than_the_record_is_restarted(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident's exact shape: a pre-v0.106.6 receiver writes NO record."""

    old = time.localtime(time.time() - 4 * 86400)
    monkeypatch.setattr(
        reap,
        "_read_process",
        lambda pid: reap._Process(
            pid=pid,
            uid=os.getuid(),
            start=time.strftime("%a %b %d %H:%M:%S %Y", old),
            cpu_s=1.0,
            argv="python -m local_operator.team_bridge.connector",
        ),
    )

    install.sync_connector_service(rig["root"])

    assert rig["kicks"] == [install.LABEL]
    marker = json.loads(sr.marker_path(rig["root"]).read_text(encoding="utf-8"))
    assert marker["reason"] == "no_record"


def test_current_or_young_receivers_are_left_alone(rig: dict[str, Any]) -> None:
    root = rig["root"]
    _seed_record(root, pid=rig["pid"], version="0.106.7", started_at=time.time() - 3600)
    install.sync_connector_service(root)
    # Just started on old code (the mixed window of an upgrade): not yet.
    _seed_record(root, pid=rig["pid"], version="0.101.42", started_at=time.time() - 5)
    install.sync_connector_service(root)
    assert rig["kicks"] == []
    assert not sr.marker_path(root).exists()


def test_a_sandboxed_home_never_reaches_launchd(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: the real ``kickstart`` refuses a plist outside the real home."""

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(launchd, "kickstart", _REAL_KICKSTART)
    monkeypatch.setattr(install, "_launchctl", lambda *a: calls.append(a))
    monkeypatch.setattr(install, "plist_path", lambda: tmp_path / "x.plist")
    _seed_record(rig["root"], pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)

    assert sr.restart_if_stale(rig["root"], trigger="surface") == sr.REFUSED
    assert calls == []
    # The refused ATTEMPT is stamped (review m3), so it backs off, not loops.
    assert json.loads(sr.marker_path(rig["root"]).read_text(encoding="utf-8"))["refused"] is True


def test_a_kick_is_never_repeated_and_the_healthy_path_never_probes(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotent, and the surface-start cost contract holds once settled."""

    root = rig["root"]
    _seed_record(root, pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)
    install.sync_connector_service(root)
    install.sync_connector_service(root)  # same pid, same installed version
    assert rig["kicks"] == [install.LABEL]

    # The restarted receiver records the installed build: no supervisor query,
    # no ``ps`` at all on the next start.
    _seed_record(root, pid=rig["pid"], version="0.106.7", started_at=time.time())

    def no_probe(*a: Any, **kw: Any) -> Any:
        raise AssertionError("the healthy path must not probe")

    monkeypatch.setattr(launchd, "job_pid", no_probe)
    monkeypatch.setattr(install, "connector_state", no_probe)
    monkeypatch.setattr(rr, "_receiver_alive", no_probe)
    install.sync_connector_service(root)
    assert rig["kicks"] == [install.LABEL]


def test_a_fresh_install_settles_the_no_record_probe(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``installed``/``registered`` started the receiver on this build: do not probe."""

    root = rig["root"]
    # The REAL ``_reconcile`` this time (review m4 moved the marker write under
    # its lock), with only the platform/enrolment/unit seams replaced.
    monkeypatch.setattr(install, "_reconcile", _REAL_RECONCILE)
    monkeypatch.setattr(install, "_platform_refusal", lambda r: None)
    monkeypatch.setattr(install, "_wanted", lambda r: True)
    monkeypatch.setattr(install, "_source_checkout_refusal", lambda: None)
    monkeypatch.setattr(
        install, "_install", lambda r: install.ServiceOutcome(install.CODE_INSTALLED)
    )
    assert install.sync_connector_service(root).code == install.CODE_INSTALLED
    assert json.loads(sr.marker_path(root).read_text(encoding="utf-8"))["probed"] == "0.106.7"
    monkeypatch.setattr(
        install, "_reconcile", lambda r, force_off: install.ServiceOutcome(install.CODE_CURRENT)
    )
    monkeypatch.setattr(launchd, "job_pid", lambda **kw: pytest.fail("probed after install"))
    install.sync_connector_service(root)
    assert rig["kicks"] == []


def test_an_older_surface_never_restarts_a_newer_receiver(rig: dict[str, Any]) -> None:
    """Review M1's repro: record 0.107.0, this surface's installed 0.106.7 -> no kick.

    A busy surface still on the previous build answers ``app_start`` after the
    receiver has moved on; restarting it would only land it back on 0.107.0.
    Settled after one probe, so later starts neither probe nor log.
    """

    root = rig["root"]
    _seed_record(root, pid=rig["pid"], version="0.107.0", started_at=time.time() - 3600)
    install.sync_connector_service(root)
    assert rig["kicks"] == []
    assert sr.needs_probe(root) is False
    install.sync_connector_service(root)
    assert rig["kicks"] == []


def test_a_stale_generation_surface_declines(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review M1: a process on a non-current generation proves nothing about the receiver."""

    monkeypatch.setattr(update, "generation_staleness_of_process", lambda pid: pid == os.getpid())
    _seed_record(rig["root"], pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)
    assert sr.restart_if_stale(rig["root"], trigger="surface") == sr.SELF_STALE
    assert rig["kicks"] == []
    # Not settled: the current build's own surface must still be able to act.
    assert not sr.marker_path(rig["root"]).exists()


def test_an_unsupported_supervisor_settles_without_probing(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Review m1: Windows declines once, then neither probes nor logs on later starts."""

    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SCHTASKS)
    monkeypatch.setattr(install, "connector_state", lambda r: pytest.fail("probed on schtasks"))
    _seed_record(rig["root"], pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)
    with caplog.at_level(logging.INFO, logger=sr.__name__):
        install.sync_connector_service(rig["root"])
        install.sync_connector_service(rig["root"])
    lines = [r.getMessage() for r in caplog.records if r.name == sr.__name__]
    assert lines == ["bridge connector stale restart: trigger=surface code=unsupported reason=none"]
    assert rig["kicks"] == []


def test_a_refused_kick_backs_off_instead_of_retrying_every_start(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m3: the attempt is stamped, so the next start is rate-limited."""

    attempts: list[str] = []

    def refuse(**kw: Any) -> bool:
        attempts.append(kw["label"])
        return False

    monkeypatch.setattr(launchd, "kickstart", refuse)
    _seed_record(rig["root"], pid=rig["pid"], version="0.101.42", started_at=time.time() - 3600)
    assert sr.restart_if_stale(rig["root"], trigger="surface") == sr.REFUSED
    assert sr.restart_if_stale(rig["root"], trigger="surface") == sr.RATE_LIMITED
    assert attempts == [install.LABEL]
    marker = json.loads(sr.marker_path(rig["root"]).read_text(encoding="utf-8"))
    assert marker["refused"] is True and marker["pid"] == rig["pid"]


def test_the_healthy_path_does_not_take_the_sync_lock(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m2: a current record answers before ``_sync_exclusive``."""

    _seed_record(rig["root"], pid=rig["pid"], version="0.106.7", started_at=time.time() - 3600)
    monkeypatch.setattr(
        install, "_sync_exclusive", lambda r: pytest.fail("locked on the healthy path")
    )
    assert install._restart_if_stale_locked(rig["root"], trigger="surface") == sr.NOT_NEEDED
