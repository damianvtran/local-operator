"""The greeting baseline: idempotence, the no-provider refusal, pause respect."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.aida import onboarding, proactive, state
from local_operator.wakes import store as wake_store
from tests.unit.aida.conftest import mark_met, write_config

SESSION_ID = "greet1234567"


def _root_with_session(root: Path) -> None:
    # The ATTACHMENT is part of a real creation: ``aida.bootstrap`` writes
    # ``agent="aida"`` beside the directory, and the effective action class is
    # read through it (``action_class.session_action_class``). A helper that
    # skips it builds a session no production path can produce — a classless
    # one, which reads reactive and (correctly) arms nothing.
    from local_operator.resume import write_session_attachment

    session_dir = root / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    write_session_attachment(session_dir, team="", agent="aida", goal="")
    state.update_state(root, session_id=SESSION_ID)


def test_the_trigger_is_a_fact_line_naming_her_and_the_surface(isolated_root: Path) -> None:
    """The hidden trigger carries facts, never playbook prose (audit A7).

    Her instructions for first contact live in her seed (the stable prefix); a
    rename must still reach the line the model reads on the first turn.
    """
    line = onboarding.greeting_message(isolated_root, surface="tui")
    assert line.startswith("[first-run] ")
    assert "surface=tui" in line
    assert "signed_in_with=none" in line
    assert "assistant_name=Aida" in line
    assert "Introduce yourself" not in line
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})
    assert "assistant_name=Sovereign" in onboarding.greeting_message(isolated_root)


def test_the_trigger_names_a_radient_identity_when_one_is_stored(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed in with Radient: she confirms the name, never asks for the email."""
    monkeypatch.setattr(
        onboarding, "radient_identity", lambda root: {"name": "Jane Doe", "email": "jane@x.com"}
    )
    line = onboarding.greeting_message(isolated_root, surface="desktop")
    assert "signed_in_with=radient" in line
    assert "identity=Jane Doe <jane@x.com>" in line


@pytest.mark.asyncio
async def test_greet_arms_a_hidden_row_from_an_attended_surface(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """owed → requested → armed, and the row is HIDDEN (audit A3/A4)."""
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    assert await onboarding.greet(isolated_root, SESSION_ID, surface="tui") == "greeted"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_ARMED
    record = onboarding.greeting_record(isolated_root)
    assert record["surface"] == "tui"
    assert isinstance(record["requested_at"], int) and isinstance(record["armed_at"], int)
    # ARMED is not DELIVERED: nothing claims the user saw a greeting yet.
    assert onboarding.greeted_at(isolated_root) is None
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    row = next(
        row for row in entry.get("schedules") or [] if row["id"] == onboarding.GREETING_WAKE_ID
    )
    assert row["hidden"] is True
    assert row["message"].startswith("[first-run] ")
    # Idempotent: a second call (a second window, a retry) does not re-arm.
    assert await onboarding.greet(isolated_root, SESSION_ID, surface="desktop") == "already"


@pytest.mark.asyncio
async def test_an_unattended_greet_never_starts_the_greeting(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE HEADLESS-RUNTIME FIX (audit A1): no surface, no greeting.

    ``proactive.resume`` and every engine path call ``greet`` without a
    surface; on an owed ledger that must arm nothing and move nothing.
    """
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    assert await onboarding.greet(isolated_root, SESSION_ID) == "not-requested"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_OWED
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None


def test_reconcile_never_arms_an_owed_greeting(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headless runtime of her session (supervisor fire, `lop exec`, mobile)
    reconciles with an owed ledger: no greeting row, and no cadence either —
    she has not met the user yet (audit A9)."""
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    result = proactive.reconcile(
        [], config_dir=isolated_root, session_id=SESSION_ID, class_reactive=False
    )
    ids = [row.id for row in result.schedules]
    assert onboarding.GREETING_WAKE_ID not in ids
    assert proactive.CADENCE_ID not in ids
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_OWED


def test_reconcile_arms_a_requested_greeting_hidden_and_holds_the_cadence(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live-owner path: a REQUESTED greeting is armed by the reconcile."""
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    assert onboarding.request_greeting(isolated_root, "tui") is True

    result = proactive.reconcile(
        [], config_dir=isolated_root, session_id=SESSION_ID, class_reactive=False
    )
    rows = {row.id: row for row in result.schedules}
    assert rows[onboarding.GREETING_WAKE_ID].hidden is True
    assert proactive.CADENCE_ID not in rows, "cadence must wait for the delivery"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_ARMED

    # Delivery moves it to delivered; the NEXT reconcile arms the cadence.
    onboarding.mark_delivered(isolated_root)
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_DELIVERED
    after = proactive.reconcile(
        [], config_dir=isolated_root, session_id=SESSION_ID, class_reactive=False
    )
    ids = [row.id for row in after.schedules]
    assert proactive.CADENCE_ID in ids
    assert onboarding.GREETING_WAKE_ID not in ids, "a delivered greeting never re-arms"


@pytest.mark.asyncio
async def test_ensure_armed_waits_for_the_greeting_on_a_first_run_install(
    isolated_root: Path,
) -> None:
    """The boot armer: no headless 08:30 check-in before she has met the user."""
    _root_with_session(isolated_root)
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "waiting"
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None
    onboarding.mark_delivered(isolated_root)
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "armed"


@pytest.mark.asyncio
async def test_an_existing_install_is_never_greeted_and_keeps_its_cadence(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R22 made durable: human conversations ⇒ ``skipped``, forever."""
    _root_with_session(isolated_root)
    _user_session(isolated_root, "aaaaaaaabbbb")
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    assert await onboarding.greet(isolated_root, SESSION_ID, surface="desktop") == "skipped"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_SKIPPED
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None
    # Its cadence behaves exactly as before the ledger existed.
    assert onboarding.cadence_allowed(isolated_root) is True
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "armed"


def test_an_existing_install_with_an_untouched_ledger_keeps_its_cadence(
    isolated_root: Path,
) -> None:
    """The upgrade path: no ``onboarding.json`` at all, conversations on disk.

    The cadence gate itself migrates the ledger to ``skipped``, so an upgrade
    never silently stops an existing user's check-in.
    """
    _root_with_session(isolated_root)
    _user_session(isolated_root, "aaaaaaaabbbb")
    assert onboarding.cadence_allowed(isolated_root) is True
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_SKIPPED


@pytest.mark.asyncio
async def test_a_born_journal_does_not_settle_the_greeting_or_arm_the_cadence(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session materialised but never typed in is not "an install with conversations".

    The SCAN keeps counting it — R22's fail-closed arm, pinned above — but the
    ledger stamp is permanent and the cadence follows it, and that pair is what
    a fresh install lost to: a boot that materialised a session before the
    greeting hook ran read as "used", so the ledger went ``skipped`` and the
    check-in armed (tips and all) in the same millisecond as her session's
    ``created_at`` (CI ``tui-e2e``, ubuntu-latest, 2026-10-09). Built through
    ``bootstrap._create_session_dir`` so the journal is the SHIPPED born shape —
    her title and birth rows, no message.
    """
    from local_operator.aida import bootstrap

    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    await bootstrap._create_session_dir(isolated_root, "555566667777")
    assert onboarding.other_user_sessions(isolated_root) == ["555566667777"]

    assert onboarding.cadence_allowed(isolated_root) is False
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_OWED
    # Nothing was spent: read the STAMP off the ledger, not ``greeting_record``
    # — that projection exposes only {state, surface, requested_at, armed_at,
    # delivered_at} for the routes, so a `.get("skipped_at")` there is None
    # whatever the file says and the assertion could never fail (review round
    # 2, found by reverting the skip and watching this cell stay green).
    ledger = isolated_root / "aida" / "onboarding.json"
    greeting = (
        json.loads(ledger.read_text(encoding="utf-8")).get("greeting", {})
        if ledger.exists()
        else {}
    )
    assert "skipped_at" not in greeting, greeting
    assert greeting.get("state") in (None, onboarding.GREETING_OWED), greeting


def test_a_torn_transcript_still_settles_the_greeting(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed for what cannot be READ: a torn row is not "nothing was said".

    The settlement's evidence rule discounts only the shape that positively
    says nothing happened — a journal that parses cleanly and holds no message.
    A half-written line is the opposite: the reader cannot tell, so the
    conservative direction wins and the existing user keeps her cadence.
    """
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    torn = isolated_root / "sessions" / "888899990000"
    torn.mkdir(parents=True)
    (torn / "transcript.jsonl").write_text(
        '{"type": "message", "payload": {"kind": "mess', encoding="utf-8"
    )

    assert onboarding.cadence_allowed(isolated_root) is True
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_SKIPPED


def test_her_own_conversation_is_read_through_the_real_path(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_her_conversation_had`` must resolve HER transcript through the config dir.

    The reader it delegates to (``session_has_durable_history``) appends
    ``sessions/<id>/`` itself, so passing the SESSIONS root — which this call
    did — asked for ``<config>/sessions/sessions/<id>/``: a path that never
    exists, so every install answered False and the "met her before the ledger
    existed" branch was dead code. Pinned with a real message row in her
    transcript; revert the argument and this cell goes red, which nothing else
    in the suite did (review round 2).
    """
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    _root_with_session(isolated_root)
    state.update_state(isolated_root, session_id=SESSION_ID)
    (isolated_root / "sessions" / SESSION_ID / "transcript.jsonl").write_text(
        '{"type": "message", "payload": {"kind": "message"}}\n', encoding="utf-8"
    )

    assert onboarding._her_conversation_had(isolated_root) is True
    # And the direction of the other predicate is pinned with it: HER
    # engagement is not the operator's, so the install still counts as fresh
    # (``other_user_sessions`` excludes the session her own state names) —
    # which is why ``_her_conversation_had`` exists as its own signal rather
    # than being inferred from ``first_run_pending``.
    assert onboarding.first_run_pending(isolated_root) is True


def test_a_legacy_greeted_at_stamp_reads_as_delivered(isolated_root: Path) -> None:
    """MIGRATION: the pre-state-machine file carried only an arm-time stamp."""
    path = isolated_root / "aida" / "onboarding.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"greeted_at": 1234, "nudge_offers": 1}), encoding="utf-8")

    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_DELIVERED
    assert onboarding.greeted_at(isolated_root) == 1234
    assert onboarding.greeting_record(isolated_root)["delivered_at"] == 1234
    assert onboarding.cadence_allowed(isolated_root) is True
    assert onboarding.first_run_pending(isolated_root) is False


@pytest.mark.asyncio
async def test_greet_refuses_without_a_provider_and_moves_nothing(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    outcome = await onboarding.greet(isolated_root, SESSION_ID, surface="tui")
    assert outcome == "no-provider"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_OWED
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None


@pytest.mark.asyncio
async def test_greet_respects_pause_and_disable(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    # Paused: the REQUEST is recorded (a person asked) but nothing is armed.
    assert await onboarding.greet(isolated_root, SESSION_ID, surface="tui") == "paused"
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED
    assert wake_store.read_entry(isolated_root, SESSION_ID) is None

    write_config(isolated_root, {"aida": {"enabled": False}})
    assert await onboarding.greet(isolated_root, SESSION_ID, surface="tui") == "disabled"


@pytest.mark.asyncio
async def test_pause_returns_an_armed_greeting_to_requested_and_resume_rearms_it(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1 under the state machine: a cancelled row is re-owed, never lost.

    A pause landing on the armed row cancels it and moves the ledger back to
    ``requested`` (the user did ask), and the resume — an UNATTENDED caller —
    may arm it because the request already exists.
    """
    _root_with_session(isolated_root)
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)

    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert await onboarding.greet(isolated_root, SESSION_ID, surface="tui") == "paused"
    await proactive.resume(isolated_root, SESSION_ID)
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    ids = [row["id"] for row in entry.get("schedules") or []]
    assert onboarding.GREETING_WAKE_ID in ids, entry
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_ARMED

    outcome = await proactive.pause(isolated_root, SESSION_ID)
    assert onboarding.GREETING_WAKE_ID in outcome.cancelled, outcome
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED
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


def test_first_run_pending_holds_until_the_greeting_settles(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requested/armed still route to her (a user who quit before the fire is
    routed again); delivered ends the first-run experience for good."""
    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    assert onboarding.first_run_pending(isolated_root) is True

    onboarding.request_greeting(isolated_root, "tui")
    onboarding.mark_greeted(isolated_root, 1234)
    assert onboarding.first_run_pending(isolated_root) is True

    onboarding.mark_delivered(isolated_root, 5678)
    assert onboarding.first_run_pending(isolated_root) is False
    assert onboarding.greeted_at(isolated_root) == 5678


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


# --------------------------------------------------------------------------- #
# The quiet-day tip ledger (audit A10/U15/D14)
# --------------------------------------------------------------------------- #


def test_no_tip_before_she_has_met_the_operator(isolated_root: Path) -> None:
    assert onboarding.tip_offer(isolated_root, now_ms=1_800_000_000_000) is None


def test_tips_are_fact_backed_once_each_and_one_a_day(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first tip is mobile via Radient when signed in; never repeated."""
    mark_met(isolated_root)
    monkeypatch.setattr(
        onboarding, "radient_identity", lambda root: {"name": "Jane", "email": "j@x.com"}
    )
    t0 = 1_800_000_000_000
    day = 86_400_000
    first = onboarding.tip_offer(isolated_root, now_ms=t0)
    assert first is not None and first.startswith(onboarding.TIP_CLAUSE_PREFIX)
    assert "Radient relay" in first and "essentially free" in first
    # Same day: nothing (a reconcile may rebuild the cadence row twice).
    assert onboarding.tip_offer(isolated_root, now_ms=t0 + 3_600_000) is None
    data = json.loads((isolated_root / "aida" / "onboarding.json").read_text(encoding="utf-8"))
    # Both mobile variants are spent by one offer, so signing out later does
    # not repeat the same suggestion in other words.
    assert set(data["tips_given"]) == {onboarding.TIP_MOBILE_RADIENT, onboarding.TIP_MOBILE_SIGNIN}
    assert data["tip_offered_at"] == t0

    second = onboarding.tip_offer(isolated_root, now_ms=t0 + day)
    assert second is not None and "first one" in second  # the first-team tip
    seen = {first, second}
    for n in range(2, 8):
        tip = onboarding.tip_offer(isolated_root, now_ms=t0 + n * day)
        assert tip not in seen
        if tip is None:
            break
        seen.add(tip)
    # The pool drains: no tip is ever offered twice.
    assert onboarding.tip_offer(isolated_root, now_ms=t0 + 30 * day) is None


def test_the_signed_out_mobile_tip_offers_radient_or_cloudflare(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark_met(isolated_root)
    monkeypatch.setattr(onboarding, "radient_identity", lambda root: None)
    monkeypatch.setattr(onboarding, "_radient_oauth_row", lambda root: False)
    tip = onboarding.tip_offer(isolated_root, now_ms=1_800_000_000_000) or ""
    assert "/login radient" in tip and "Cloudflare" in tip


def test_a_configured_tunnel_skips_the_mobile_tip(isolated_root: Path) -> None:
    mark_met(isolated_root)
    (isolated_root / "tunnel").mkdir()
    (isolated_root / "tunnel" / "config.json").write_text("{}", encoding="utf-8")
    tip = onboarding.tip_offer(isolated_root, now_ms=1_800_000_000_000) or ""
    assert "Phone access" not in tip


def test_the_cadence_message_carries_one_tip_clause(isolated_root: Path) -> None:
    mark_met(isolated_root)
    message = proactive.cadence_message(isolated_root, 1_800_000_000_000)
    assert message.count(onboarding.TIP_CLAUSE_PREFIX) == 1
    assert "quiet-day tip" in proactive.CADENCE_MESSAGE
