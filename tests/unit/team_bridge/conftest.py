"""Per-test isolation for state the bridge client keeps for the WHOLE process."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from local_operator.team_bridge import client as tb_client


@pytest.fixture(autouse=True)
def _empty_directory_cache() -> Iterator[None]:
    """Every test starts and ends with no cached directory answer.

    ``TeamBridgeClient.directory_cached`` holds answers process-wide (the tool
    builds a client per call, so a per-instance cache would never be hit), and
    whether an answer came from that cache decides whether a MISS re-reads the
    directory AND which refusal is recorded. A test that leaves an entry behind
    — or finds one — therefore changes both the request count and the verdict
    the NEXT test observes, which is exactly how the h5d name-miss test became
    order-dependent.

    Modules that already define a fixture of this name (``test_h5d_tool``,
    ``test_rate_limit_refusals``) keep their own, which takes precedence there;
    this one covers the rest of the package.
    """

    tb_client._DIRECTORY_CACHE.clear()
    yield
    tb_client._DIRECTORY_CACHE.clear()


@pytest.fixture(autouse=True)
def _pre_triage_wakes(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``team_bridge.ambient_triage = off`` unless a test opts in to triage.

    WHY: the suites here pin the WAKE MECHANICS that predate channel-wake triage
    (cards, coalescing, routing, replay) with ambient channel traffic as their
    fixture, and ``off`` is defined as exactly that pre-triage behaviour — so they
    keep proving it unchanged. The triage itself (default ``on``) is pinned by
    ``test_wake_triage.py``, which is exempt here and reads the real config-driven
    settings. Settings beyond the mode keep their defaults.
    """

    if request.module.__name__.rsplit(".", 1)[-1] == "test_wake_triage":
        return
    from dataclasses import replace

    from local_operator.team_bridge.connector import conversation
    from local_operator.team_bridge.settings import TeamBridgeSettings

    monkeypatch.setattr(
        conversation,
        "_triage_settings",
        lambda _config_dir: replace(TeamBridgeSettings(), ambient_triage="off"),
    )
