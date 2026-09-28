"""The greeting baseline: idempotence, the no-provider refusal, pause respect."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.aida import onboarding, state
from local_operator.wakes import store as wake_store
from tests.unit.aida.conftest import write_config

SESSION_ID = "greet1234567"


def _root_with_session(root: Path) -> None:
    (root / "sessions" / SESSION_ID).mkdir(parents=True)
    state.update_state(root, session_id=SESSION_ID)


@pytest.mark.asyncio
async def test_greet_refuses_without_a_provider_and_stamps_nothing(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    outcome = await onboarding.greet(isolated_root, SESSION_ID)
    assert outcome == "no-provider"
    assert onboarding.greeted_at(isolated_root) is None
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None


@pytest.mark.asyncio
async def test_greet_arms_once_then_reports_already(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a provider configured the greeting arms as a due-now one-shot.

    ``provider_configured`` is monkeypatched rather than configured: the real
    predicate resolves the boot path's hosting/model machinery, and what this
    test pins is the greeting's own behaviour, not that resolution (which the
    route tests exercise for real through the 409).
    """
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    assert await onboarding.greet(isolated_root, SESSION_ID) == "greeted"
    stamp = onboarding.greeted_at(isolated_root)
    assert isinstance(stamp, int)
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    rows = {row["id"]: row for row in entry.get("schedules") or []}
    assert onboarding.GREETING_WAKE_ID in rows

    # Idempotent: a second call (a second device, a retry) does not re-arm.
    assert await onboarding.greet(isolated_root, SESSION_ID) == "already"


@pytest.mark.asyncio
async def test_greet_respects_pause_and_disable(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert await onboarding.greet(isolated_root, SESSION_ID) == "paused"
    assert onboarding.greeted_at(isolated_root) is None

    write_config(isolated_root, {"aida": {"enabled": False}})
    assert await onboarding.greet(isolated_root, SESSION_ID) == "disabled"
    assert onboarding.greeted_at(isolated_root) is None
