"""The connector-state display cache (``install.cached_connector_state``).

The status route used to run ``launchctl print`` on every status poll; it now
reuses the answer for ``CONNECTOR_STATE_TTL_S`` and every in-process placement
change invalidates it. Monotonic time is patched, never slept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.team_bridge.connector import install


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch):
    install.invalidate_connector_state_cache()
    calls: list[Path] = []
    answers = iter([install.STATE_MISSING, install.STATE_LOADED, install.STATE_INSTALLED])
    clock = {"now": 1000.0}

    def fake_state(config_dir):
        calls.append(Path(config_dir))
        return next(answers)

    monkeypatch.setattr(install, "connector_state", fake_state)
    monkeypatch.setattr(install.time, "monotonic", lambda: clock["now"])
    yield calls, clock
    install.invalidate_connector_state_cache()


def test_within_the_ttl_the_probe_runs_once(probe, tmp_path):
    calls, clock = probe
    assert install.cached_connector_state(tmp_path) == install.STATE_MISSING
    clock["now"] += install.CONNECTOR_STATE_TTL_S - 0.1
    assert install.cached_connector_state(tmp_path) == install.STATE_MISSING
    assert len(calls) == 1


def test_after_the_ttl_it_reads_again(probe, tmp_path):
    calls, clock = probe
    install.cached_connector_state(tmp_path)
    clock["now"] += install.CONNECTOR_STATE_TTL_S
    assert install.cached_connector_state(tmp_path) == install.STATE_LOADED
    assert len(calls) == 2


def test_each_config_dir_has_its_own_entry(probe, tmp_path):
    calls, _clock = probe
    install.cached_connector_state(tmp_path / "a")
    install.cached_connector_state(tmp_path / "b")
    assert len(calls) == 2


@pytest.mark.parametrize("mutator", ["sync_connector_service", "remove_connector_service"])
def test_a_placement_change_invalidates(probe, tmp_path, monkeypatch, mutator):
    calls, _clock = probe
    monkeypatch.setattr(install, "_reconcile", lambda config_dir, force_off: "outcome")
    install.cached_connector_state(tmp_path)
    assert getattr(install, mutator)(tmp_path) == "outcome"
    assert install.cached_connector_state(tmp_path) == install.STATE_LOADED
    assert len(calls) == 2


def test_a_failed_placement_still_invalidates(probe, tmp_path, monkeypatch):
    calls, _clock = probe

    def boom(config_dir, force_off):
        raise RuntimeError("supervisor went away")

    monkeypatch.setattr(install, "_reconcile", boom)
    install.cached_connector_state(tmp_path)
    with pytest.raises(RuntimeError):
        install.sync_connector_service(tmp_path)
    install.cached_connector_state(tmp_path)
    assert len(calls) == 2


def test_the_plist_repair_invalidates(probe, tmp_path, monkeypatch):
    calls, _clock = probe
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: "none")
    install.cached_connector_state(tmp_path)
    install.refresh_plist_if_stale()
    install.cached_connector_state(tmp_path)
    assert len(calls) == 2


def test_a_read_racing_an_invalidation_does_not_store_its_stale_answer(
    probe, tmp_path, monkeypatch
):
    """A probe that started before a placement change must not be cached after it."""

    calls, _clock = probe

    def racing(config_dir):
        calls.append(Path(config_dir))
        install.invalidate_connector_state_cache()  # a sync lands mid-probe
        return install.STATE_MISSING

    monkeypatch.setattr(install, "connector_state", racing)
    install.cached_connector_state(tmp_path)
    install.cached_connector_state(tmp_path)
    assert len(calls) == 2
