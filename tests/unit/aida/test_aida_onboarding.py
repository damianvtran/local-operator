"""The greeting baseline: idempotence, the no-provider refusal, pause respect."""

from __future__ import annotations

import json
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


# --------------------------------------------------------------------------- #
# Slice B: the fresh-install predicate (R21/R22/R26) and the nudge ledger (R25)
# --------------------------------------------------------------------------- #


def _user_session(root: Path, session_id: str) -> None:
    """A session with a conversation HAD — a real turn in its transcript.

    The transcript row is the engagement marker the scan reads (see
    ``onboarding._counts_as_operator_conversation``); a bare directory is the
    boot's own materialisation and deliberately does not count (UX round 2,
    U4).
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        '{"type": "message", "payload": {"kind": "message"}}\n', encoding="utf-8"
    )


def test_the_predicate_is_no_human_conversations_plus_a_provider(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    assert onboarding.fresh_install(isolated_root) is True

    _user_session(isolated_root, "aaaaaaaabbbb")
    assert onboarding.fresh_install(isolated_root) is False
    assert onboarding.other_user_sessions(isolated_root) == ["aaaaaaaabbbb"]


def test_a_boot_materialised_session_directory_does_not_make_an_install_look_used(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boot's own directory — lease + pid, no transcript — is not a conversation.

    Reproduces the shipped shape: every normal launch materialises one session
    directory before any first contact, and counting it as "the operator has
    conversations" turned the predicate false on a fresh install's FIRST
    boot, so a provider-present first contact never greeted (UX review round
    2, U4).
    """
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    booted = isolated_root / "sessions" / "0b7011cccbee"
    booted.mkdir(parents=True)
    (booted / ".execution-lease").write_text('{"pid": 1}', encoding="utf-8")
    (booted / ".session.pid").write_text("1", encoding="utf-8")

    assert onboarding.other_user_sessions(isolated_root) == []
    assert onboarding.fresh_install(isolated_root) is True
    assert onboarding.first_run_pending(isolated_root) is True


def test_an_empty_transcript_file_is_also_not_a_conversation(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A touched-but-unused session's size-0 journal says nothing was said."""
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    touched = isolated_root / "sessions" / "ffff00001111"
    touched.mkdir(parents=True)
    (touched / "transcript.jsonl").write_text("", encoding="utf-8")

    assert onboarding.other_user_sessions(isolated_root) == []
    assert onboarding.fresh_install(isolated_root) is True


def test_a_transcript_with_any_unreadable_content_fails_closed(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything on disk the reader will not call history counts as the operator's.

    R22's direction: "never re-route somebody with conversations" outranks a
    missed first-run greeting. Pinned with the custom-row shape (wake prompts
    and quiet-dial peer notes persist without a turn) so the conservative arm
    cannot be narrowed to "only message rows" by a later refactor.
    """
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    custom_only = isolated_root / "sessions" / "222233334444"
    custom_only.mkdir(parents=True)
    (custom_only / "transcript.jsonl").write_text(
        '{"type": "message", "payload": {"kind": "custom"}}\n', encoding="utf-8"
    )

    assert onboarding.other_user_sessions(isolated_root) == ["222233334444"]
    assert onboarding.fresh_install(isolated_root) is False


def test_a_subagent_run_does_not_make_an_install_look_used(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    machine = isolated_root / "sessions" / "ccccccccdddd"
    machine.mkdir(parents=True)
    (machine / "origin.json").write_text('{"origin": "subagent"}', encoding="utf-8")
    assert onboarding.fresh_install(isolated_root) is True


def test_her_own_session_does_not_count_as_a_human_conversation(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    (isolated_root / "sessions" / SESSION_ID).mkdir(parents=True)
    state.update_state(isolated_root, session_id=SESSION_ID)

    assert onboarding.other_user_sessions(isolated_root) == []
    assert onboarding.fresh_install(isolated_root) is True


def test_an_unusable_session_store_fails_closed(isolated_root: Path) -> None:
    """A store that cannot be read must never re-route somebody (R22)."""
    (isolated_root / "sessions").write_text("not a directory", encoding="utf-8")
    assert onboarding.other_user_sessions(isolated_root) is None
    assert onboarding.fresh_install(isolated_root) is False
    assert onboarding.first_run_pending(isolated_root) is False


def test_first_run_pending_requires_the_greeting_to_still_be_owed(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    assert onboarding.first_run_pending(isolated_root) is True

    onboarding.mark_greeted(isolated_root, 1234)
    assert onboarding.first_run_pending(isolated_root) is False

    # A pause that cancels the unfired greeting re-owes it (m1's clear), and
    # the routing must follow the ledger back to "owed".
    onboarding.clear_greeted(isolated_root)
    assert onboarding.first_run_pending(isolated_root) is True


def test_the_nudge_window_opens_once_then_closes_for_the_configured_span(
    isolated_root: Path,
) -> None:
    t0 = 1_800_000_000_000
    assert onboarding.nudge_offer(isolated_root, now_ms=t0) == onboarding.NUDGE_CLAUSE
    data = json.loads((isolated_root / "aida" / "onboarding.json").read_text(encoding="utf-8"))
    assert data["nudge_offered_at"] == t0
    assert data["nudge_offers"] == 1

    # Inside the window: closed, and nothing re-stamped.
    day = 86_400_000
    assert onboarding.nudge_offer(isolated_root, now_ms=t0 + day) is None
    data = json.loads((isolated_root / "aida" / "onboarding.json").read_text(encoding="utf-8"))
    assert data["nudge_offers"] == 1

    # Past it: open again, and the ledger counts the window.
    later = t0 + onboarding.DEFAULT_NUDGE_DAYS * day
    assert onboarding.nudge_offer(isolated_root, now_ms=later) == onboarding.NUDGE_CLAUSE
    data = json.loads((isolated_root / "aida" / "onboarding.json").read_text(encoding="utf-8"))
    assert data["nudge_offers"] == 2
    assert data["nudge_offered_at"] == later


def test_the_configured_nudge_days_bounds_the_window(isolated_root: Path) -> None:
    write_config(isolated_root, {"aida": {"onboarding": {"nudge_days": 1}}})
    day = 86_400_000
    t0 = 1_800_000_000_000
    assert onboarding.nudge_offer(isolated_root, now_ms=t0) == onboarding.NUDGE_CLAUSE
    assert onboarding.nudge_offer(isolated_root, now_ms=t0 + day - 1) is None
    assert onboarding.nudge_offer(isolated_root, now_ms=t0 + day) == onboarding.NUDGE_CLAUSE


def test_a_hostile_nudge_days_falls_back_to_the_default(isolated_root: Path) -> None:
    """A hand-edited 0 must not turn the bound into every-day nagging."""
    write_config(isolated_root, {"aida": {"onboarding": {"nudge_days": 0}}})
    t0 = 1_800_000_000_000
    assert onboarding.nudge_offer(isolated_root, now_ms=t0) == onboarding.NUDGE_CLAUSE
    assert onboarding.nudge_offer(isolated_root, now_ms=t0 + 1_000) is None


def test_the_cadence_message_carries_the_clause_only_while_the_window_is_open(
    isolated_root: Path,
) -> None:
    """The ROW carries the permission; a closed window is the bare prompt."""
    t0 = 1_800_000_000_000
    message = proactive.cadence_message(isolated_root, t0)
    assert proactive.CADENCE_MESSAGE in message
    assert onboarding.NUDGE_CLAUSE in message

    closed = proactive.cadence_message(isolated_root, t0 + 1_000)
    assert closed == proactive.CADENCE_MESSAGE
