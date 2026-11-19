"""The directory-wide belt teardown really disarms a borrower self-revoke timer.

Regression for the merge-gate flake where a belt armed by one network cell fired
~600 s later inside ``remote_compute/test_cli_remote.py`` and was counted as that
cell's real ``DELETE https://api.github.com/installation/token`` (see
``conftest._no_leaked_self_revoke_belt``). The teardown is only worth having if it
leaves no live timer thread and no scheduled token behind, so that is what this
pins — with the real ``_schedule_self_revoke``, not a stub of it.
"""

from __future__ import annotations

import time

import pytest

from local_operator.network.credentials import github as github_mod
from tests.unit.network.conftest import cancel_leaked_self_revoke_belts


def test_a_far_belt_is_cancelled_and_reaped_without_firing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: list[str] = []
    belt = "synthetic-belt-isolation-value"
    monkeypatch.setattr(
        github_mod, "revoke_installation_token", lambda value, **_: fired.append(value) or True
    )
    github_mod._schedule_self_revoke(belt, int(time.time() * 1000) + 600_000)
    with github_mod._SELF_REVOKE_LOCK:
        timer = github_mod._SELF_REVOKE_TIMERS[belt]
    assert timer.is_alive()

    assert cancel_leaked_self_revoke_belts() >= 1

    assert not timer.is_alive()
    with github_mod._SELF_REVOKE_LOCK:
        assert belt not in github_mod._SELF_REVOKE_TIMERS
        assert belt not in github_mod._SELF_REVOKE_SCHEDULED
    assert fired == []
