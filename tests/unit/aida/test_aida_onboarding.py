"""The greeting baseline: idempotence, the no-provider refusal, pause respect."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.aida import onboarding, proactive, state
from local_operator.wakes import store as wake_store
from tests.unit.aida.conftest import write_config

SESSION_ID = "greet1234567"


def _root_with_session(root: Path) -> None:
    (root / "sessions" / SESSION_ID).mkdir(parents=True)
    state.update_state(root, session_id=SESSION_ID)


def test_the_greeting_addresses_her_by_the_configured_name(isolated_root: Path) -> None:
    """The first-run instruction must name the operator's name for her.

    The greeting's wake line is user-visible in the transcript, so a hard-coded
    "Introduce yourself as Aida" after a rename is exactly the stale reference
    the renameable-chief-of-staff work removes.
    """
    assert "as Aida," in onboarding.greeting_message(isolated_root)
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})
    renamed = onboarding.greeting_message(isolated_root)
    assert "as Sovereign," in renamed
    assert "as Aida" not in renamed


@pytest.mark.asyncio
async def test_greet_arms_with_the_configured_name(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})

    assert await onboarding.greet(isolated_root, SESSION_ID) == "greeted"
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    row = next(
        row for row in entry.get("schedules") or [] if row["id"] == onboarding.GREETING_WAKE_ID
    )
    assert "as Sovereign," in row["message"]


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


@pytest.mark.asyncio
async def test_pause_unstamps_an_undelivered_greeting_and_resume_rearms_it(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1: the paused receipt's promise is true, and the greeting survives.

    ``greet`` while paused used to answer "paused" while the route said the
    greeting "is being held; /aida resume delivers it" — nothing held it and
    resume never armed it. Same seam, second half: a greeting row already
    armed when a pause landed was cancelled as an ``aida-*`` row while
    ``greeted_at`` stayed stamped, so the one-time greeting was lost with the
    ledger claiming delivery. Now the stamp is the ledger of an OWED greeting:
    a pause that cancels the row un-stamps it, and the resume arms it again.
    """
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    # Paused first: refused, nothing stamped, nothing armed.
    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert await onboarding.greet(isolated_root, SESSION_ID) == "paused"
    assert onboarding.greeted_at(isolated_root) is None
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None

    # Resume delivers it: the greeting row is armed and the ledger moves.
    await proactive.resume(isolated_root, SESSION_ID)
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    ids = [row["id"] for row in entry.get("schedules") or []]
    assert onboarding.GREETING_WAKE_ID in ids, entry
    assert onboarding.greeted_at(isolated_root) is not None

    # A pause landing on the armed row cancels it and un-stamps it, so the
    # next resume arms it again rather than losing it forever.
    outcome = await proactive.pause(isolated_root, SESSION_ID)
    assert onboarding.GREETING_WAKE_ID in outcome.cancelled, outcome
    assert onboarding.greeted_at(isolated_root) is None
    await proactive.resume(isolated_root, SESSION_ID)
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    ids = [row["id"] for row in entry.get("schedules") or []]
    assert onboarding.GREETING_WAKE_ID in ids, entry
