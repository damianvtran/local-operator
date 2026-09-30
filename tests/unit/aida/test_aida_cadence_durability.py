"""Her check-in cadence must not drop by accident — the operator's requirement.

Reported from the LIVE install on 2026-09-30 (found by looking at disk, not by a
test): the chief-of-staff session had NO wake file at all, three one-shot
``aida-cadence`` rows sat parked on dead session ids, and
``aida/escalate.json`` held an entry with its day marker stuck at the previous
day. The requirement, in the reporter's words: *she should never just
accidentally drop her wake unless she is specifically asked to*.

The cells below are the acceptance matrix, one cause each. They are written
against the seams production uses — ``bootstrap.ensure_session`` for the boot
paths, ``proactive.reconcile`` for the live session's own writes, and
``run_startup_migrations`` for the upgrade — rather than against private
helpers, so a test that passes here describes a path that is actually reachable.

WHY EVERY CELL IS A FILE-LEVEL ASSERTION: the defect was invisible to the
in-process state (her session believed it was fine; ``state.json`` even named the
right session), and only the WAKE INDEX — the file every out-of-process reader
consults — showed the loss. So the assertions read the index and the escalation
ledger, not the return value of the call that was supposed to write them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from local_operator import aida
from local_operator.agent_profiles import (
    SEED_ORIGIN_PREFIX,
    SEED_SHA256_PREFIX,
    backfill_seed_action_class,
    load_seed,
    seed_fingerprint,
)
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.aida import proactive, state
from local_operator.wakes import store as wake_store

DEAD_ONE = "0827221cce46"
#: A well-formed install fingerprint that is NOT the packaged starter's — the
#: pair the backfill's last narrowing step reads (see ``_pre_class_tags``).
STALE_SHA = "a" * 64
#: A second well-formed-but-foreign fingerprint, for the rows that must not move.
OTHER_SHA = "b" * 64
DEAD_TWO = "ce526344aa86"


def _fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out (strict mode)."""
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


async def _her_session(root: Path) -> str:
    """``ensure_session``'s id, with its ``None`` case asserted away.

    ``None`` means "Aida is disabled in this process", which every cell here
    would fail on anyway; narrowing it once keeps the type honest instead of
    repeating ``assert session_id`` at twenty call sites.
    """
    session_id = await aida.ensure_session(root)
    assert session_id, "ensure_session answered None — Aida is disabled in this rig"
    return session_id


def _cadence_row(root: Path, session_id: str) -> dict[str, Any] | None:
    """The cadence row from the wake INDEX, or ``None`` when nothing is armed.

    The index rather than the transcript: it is the file ``lop wake status``,
    the supervisor and the picker read, and the reported defect was precisely an
    index with no row in it while everything in-process looked healthy.
    """
    entry = wake_store.read_entry(root, session_id)
    if not entry:
        return None
    for row in entry.get("schedules", []):
        if row.get("id") == proactive.CADENCE_ID:
            return row
    return None


def _install_aida_row(root: Path, tags: list[str]) -> AgentRegistry:
    """A registry carrying ONE ``aida`` role row with ``tags``.

    The shape the live install had (and the shape ``install_seed`` produces),
    used directly because the point of these cells is the ROW, not the installer.
    """
    registry = AgentRegistry(root)
    registry.create_agent(_fields(name="aida", tags=tags, categories=["role"]))
    return registry


def _pre_class_tags() -> list[str]:
    """The exact tags the operator's live row carried: role, install record, no class.

    The fingerprint is a WELL-FORMED sha256 that is not the packaged starter's,
    because that pair is the whole signal: ``_installed_fingerprint`` refuses a
    malformed one, and a row with no usable record is deliberately left alone
    (the same refusal ``_sync_one_seed`` makes). The live row's own value was
    ``bfe22544…`` against a packaged ``7f3a1355…`` on the day of the report.
    """
    return [
        "role",
        "delegate:yes",
        f"{SEED_ORIGIN_PREFIX}aida",
        f"{SEED_SHA256_PREFIX}{STALE_SHA}",
    ]


# -- the root cause: a row installed before the class feature -------------


def test_a_pre_class_row_reads_reactive_and_that_is_what_dropped_her_cadence(
    isolated_root: Path,
) -> None:
    """THE REPRODUCTION. The live install's failure, in one assertion pair.

    ``class:`` and the readers that gate on it shipped in ONE release (v0.64.9,
    installed on this machine at 03:55 on the day of the report), and the install
    path is the only writer of a row's tags — so every role row installed by an
    earlier release carries no class tag, and every reader normalizes an absent
    tag to ``reactive``. Her cadence is gated on that read, so an install that
    was working the day before goes silent on upgrade with no user action.
    """
    from local_operator.action_class import PROACTIVE, session_action_class

    session_id = "439818272d84"
    session_dir = isolated_root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    from local_operator.resume import write_session_attachment

    write_session_attachment(session_dir, team="", agent="aida", goal="")

    stale = _install_aida_row(isolated_root, _pre_class_tags())
    assert session_action_class(session_dir, registry=stale) == "reactive"

    flipped = backfill_seed_action_class(isolated_root, registry=stale)
    assert flipped == ("aida",)
    assert session_action_class(session_dir, registry=stale) == PROACTIVE


def test_the_backfill_touches_only_the_rows_it_can_classify(isolated_root: Path) -> None:
    """Every narrowing step of the predicate, one row each.

    The repair writes to the operator's agent registry, so the rows it must NOT
    touch matter as much as the one it must: a deliberate ``reactive`` switch, a
    row whose recorded starter is current (so the tag was removed by hand, not
    by an upgrade), a row the operator authored, and a row from a starter that
    declares no class at all.
    """
    registry = AgentRegistry(isolated_root)
    packaged = load_seed("aida")
    assert packaged is not None
    current = seed_fingerprint(packaged)
    registry.create_agent(_fields(name="aida", tags=_pre_class_tags(), categories=["role"]))
    registry.create_agent(
        _fields(
            name="switched-off",
            tags=[
                "role",
                f"{SEED_ORIGIN_PREFIX}aida",
                f"{SEED_SHA256_PREFIX}{OTHER_SHA}",
                "class:reactive",
            ],
            categories=["role"],
        )
    )
    registry.create_agent(
        _fields(
            name="up-to-date",
            tags=["role", f"{SEED_ORIGIN_PREFIX}aida", f"{SEED_SHA256_PREFIX}{current}"],
            categories=["role"],
        )
    )
    registry.create_agent(_fields(name="authored", tags=["role"], categories=["role"]))
    registry.create_agent(
        _fields(
            name="reviewer",
            tags=["role", f"{SEED_ORIGIN_PREFIX}reviewer", f"{SEED_SHA256_PREFIX}{OTHER_SHA}"],
            categories=["role"],
        )
    )

    assert backfill_seed_action_class(isolated_root, registry=registry) == ("aida",)
    # IDEMPOTENT: the repaired row now carries the tag, so the second pass has
    # nothing to classify — which is what lets the seam run on every launch
    # without a stamp file (see ``config_migrations``).
    assert backfill_seed_action_class(isolated_root, registry=registry) == ()
    # Read through a FRESH registry: the calling instance keeps the row object it
    # loaded (the write lands on disk), so asserting on the stale copy would test
    # the cache rather than the repair.
    after = AgentRegistry(isolated_root)
    for name in ("up-to-date", "authored", "reviewer"):
        untouched_row = after.get_agent_by_name(name)
        assert untouched_row is not None, name
        tags = list(untouched_row.tags or [])
        assert not any(tag.startswith("class:") for tag in tags), name
    # The deliberate switch is left exactly as it was — this is the row a repair
    # without an explicit ``reactive`` encoding would have re-armed.
    switched_row = after.get_agent_by_name("switched-off")
    assert switched_row is not None
    switched = list(switched_row.tags or [])
    assert "class:reactive" in switched and "class:proactive" not in switched
    repaired = after.get_agent_by_name("aida")
    assert repaired is not None
    assert "class:proactive" in list(repaired.tags or [])


def test_the_startup_seam_runs_the_backfill(isolated_root: Path) -> None:
    """One seam, and it is the one ``cli.main`` calls — not a second copy.

    Asserted through the seam rather than the function so a future migration
    that forgets to wire it up fails here, which is the failure the module's own
    docstring is built around ("a migration runs from
    ``run_startup_migrations`` and nowhere else").
    """
    from local_operator import config_migrations

    _install_aida_row(isolated_root, _pre_class_tags())
    config_migrations.run_startup_migrations(isolated_root)

    seam_row = AgentRegistry(isolated_root).get_agent_by_name("aida")
    assert seam_row is not None
    assert "class:proactive" in list(seam_row.tags or [])


# -- the engine's teeth: an existing session is armed on every boot -------


@pytest.mark.asyncio
async def test_a_restart_re_arms_a_session_whose_index_entry_was_lost(
    isolated_root: Path,
) -> None:
    """The restart / update-drain cell: the row survives in the TRANSCRIPT.

    Every external arm writes transcript first and index second, so a crash, a
    kill or a drain between the two — and any hand-deleted index file — leaves a
    schedule the index has never seen. The supervisor can only fire what the
    index shows, so without a repair the cadence exists and nothing can deliver
    it until the operator happens to open the conversation.
    """
    session_id = await _her_session(isolated_root)
    assert session_id and _cadence_row(isolated_root, session_id) is not None

    wake_store.entry_path(isolated_root, session_id).unlink()
    assert _cadence_row(isolated_root, session_id) is None

    again = await _her_session(isolated_root)
    assert again == session_id
    row = _cadence_row(isolated_root, session_id)
    assert row is not None, "the boot arm must repair the index from the transcript"
    # The repair re-projects the TRANSCRIPT's row rather than arming beside it,
    # so the row the index now carries is the one the transcript holds — same
    # due instant, and exactly one row there.
    from_transcript = _transcript_rows(isolated_root, session_id)
    assert [r.get("id") for r in from_transcript].count(proactive.CADENCE_ID) == 1
    assert row["next_due_at"] == from_transcript[0]["next_due_at"]


@pytest.mark.asyncio
async def test_a_session_whose_entry_lost_its_rows_is_healed(isolated_root: Path) -> None:
    """The zero-schedule cell: the entry exists and is empty.

    Distinct from the cell above — an entry with an empty schedule list is
    removed outright by ``store.write_entry``, so "no file" is the shape the
    reader sees; this pins the other spelling (a well-formed entry carrying
    nothing) so neither can drift into being un-healable.
    """
    session_id = await _her_session(isolated_root)
    entry = wake_store.read_entry(isolated_root, session_id)
    assert entry is not None
    entry["schedules"] = []
    wake_store.write_entry(
        isolated_root, session_id, cwd=entry["cwd"], schedules=[], preserve=entry
    )
    assert wake_store.read_entry(isolated_root, session_id) is None

    await _her_session(isolated_root)
    assert _cadence_row(isolated_root, session_id) is not None


@pytest.mark.asyncio
async def test_session_id_churn_migrates_the_cadence_and_reaps_the_ghost(
    isolated_root: Path,
) -> None:
    """THE CHURN CELL. Arm for A, lose A's directory, ensure again.

    A re-creation mints a new id, and the previous incarnation's armed row is one
    no runtime can ever be started for. Nothing migrated it and nothing removed
    it: the operator's install had three such rows, each carrying a full check-in
    message, cleared by hand.
    """
    first = await _her_session(isolated_root)
    assert first and _cadence_row(isolated_root, first) is not None

    # What a churn IS: the directory ``state.json`` names is gone (an early
    # create that failed and was discarded, a cleanup, a hand-deleted store),
    # so the next ensure mints rather than returns the existing id.
    import shutil

    shutil.rmtree(isolated_root / "sessions" / first)

    second = await _her_session(isolated_root)
    assert second and second != first
    assert state.session_id_of(isolated_root) == second
    assert _cadence_row(isolated_root, second) is not None, "the new id must be armed"
    assert wake_store.read_entry(isolated_root, first) is None, "the ghost must be reaped"


@pytest.mark.asyncio
async def test_the_reap_leaves_live_sessions_and_other_peoples_rows_alone(
    isolated_root: Path,
) -> None:
    """Reaping is scoped to the engine's own rows on sessions that are GONE.

    A wake row is not litter just because its session is not hers: another
    session's armed reminder is that session's, and deleting it here would be
    this engine reaching outside its own schedule list — the exact boundary the
    one-writer rules exist to keep.
    """
    hers = await _her_session(isolated_root)
    # A live session that is NOT her, carrying one engine row and one of its own.
    other = isolated_root / "sessions" / "aaaaaaaaaaaa"
    other.mkdir(parents=True)
    (other / "transcript.jsonl").write_text("", encoding="utf-8")
    wake_store.write_entry(
        isolated_root,
        "aaaaaaaaaaaa",
        cwd=str(isolated_root),
        schedules=[
            {"id": "w1", "kind": "scheduled", "message": "mine", "next_due_at": 1, "every_ms": 0},
            {"id": proactive.CADENCE_ID, "kind": "scheduled", "message": "hers", "next_due_at": 2},
        ],
    )
    # A GHOST carrying both kinds: only the engine's row goes.
    wake_store.write_entry(
        isolated_root,
        DEAD_ONE,
        cwd=str(isolated_root),
        schedules=[
            {"id": "w1", "kind": "scheduled", "message": "someone's", "next_due_at": 3},
            {"id": proactive.CADENCE_ID, "kind": "scheduled", "message": "hers", "next_due_at": 3},
        ],
    )
    assert proactive.reap_orphan_rows(isolated_root, keep=hers) == (DEAD_ONE,)

    # The ghost keeps the row that was never the engine's...
    kept = wake_store.read_entry(isolated_root, DEAD_ONE)
    assert kept is not None
    assert [row["id"] for row in kept["schedules"]] == ["w1"]
    # ...and a LIVE session is not swept at all: it keeps both rows, engine or
    # not, because its transcript is present and that is the predicate.
    live_other = wake_store.read_entry(isolated_root, "aaaaaaaaaaaa")
    assert live_other is not None
    assert [row["id"] for row in live_other["schedules"]] == ["w1", proactive.CADENCE_ID]
    assert _cadence_row(isolated_root, hers) is not None


@pytest.mark.asyncio
async def test_pause_is_the_one_thing_that_may_disarm_it(isolated_root: Path) -> None:
    """The pause cell, and the boot arm must respect it.

    The requirement names the exception exactly: she may lose the check-in when
    the operator ASKS. A self-heal that re-armed through ``/aida pause`` would be
    a worse bug than the one this file is about — it would restart the messaging
    the operator stopped.
    """
    session_id = await _her_session(isolated_root)
    assert _cadence_row(isolated_root, session_id) is not None

    await proactive.pause(isolated_root, session_id)
    assert _cadence_row(isolated_root, session_id) is None

    # A boot AFTER the pause (the every-boot arm) must not put it back.
    assert await _her_session(isolated_root) == session_id
    assert _cadence_row(isolated_root, session_id) is None

    armed = await proactive.resume(isolated_root, session_id)
    assert armed == "armed"
    assert _cadence_row(isolated_root, session_id) is not None


@pytest.mark.asyncio
async def test_a_paused_session_stays_dark_across_a_churn(isolated_root: Path) -> None:
    """Pause outranks re-creation: the new id must not be armed either.

    The two mechanisms interact — a churn mints and arms, a pause forbids arming
    — and the operator's own install exercised both in the same window. The
    pause wins, because it is the one that was asked for.
    """
    import shutil

    first = await _her_session(isolated_root)
    await proactive.pause(isolated_root, first)
    shutil.rmtree(isolated_root / "sessions" / first)

    second = await _her_session(isolated_root)
    assert second and second != first
    assert _cadence_row(isolated_root, second) is None


# -- the restart / update-drain cells ------------------------------------


@pytest.mark.asyncio
async def test_a_restart_over_an_intact_index_is_a_no_op(isolated_root: Path) -> None:
    """The ordinary case, and the one that must stay boring.

    An update restarts the runtime; her session comes back with its index entry
    intact and its transcript holding the row. The every-boot arm has to be a
    no-op here — arming a SECOND ``aida-cadence`` row on top of a live one would
    make her check in twice and is the failure mode a naive "ensure" invites.
    """
    session_id = await _her_session(isolated_root)
    before = json.loads(wake_store.entry_path(isolated_root, session_id).read_text())

    assert await _her_session(isolated_root) == session_id
    assert await _her_session(isolated_root) == session_id

    after = json.loads(wake_store.entry_path(isolated_root, session_id).read_text())
    assert after["schedules"] == before["schedules"]
    rows = _transcript_rows(isolated_root, session_id)
    assert [r.get("id") for r in rows].count(proactive.CADENCE_ID) == 1
    assert await proactive.ensure_armed(isolated_root, session_id) == "present"


@pytest.mark.asyncio
async def test_a_stopped_session_is_left_dormant_by_the_boot_arm(isolated_root: Path) -> None:
    """``/stop`` is a user lever, so the boot arm must not walk past it.

    The stop marker lives on the index entry (``stopped_at``, stamped by the
    TUI's stop lever) and both other readers already honour it — the supervisor
    skips such an entry and the trigger gate answers "held". The arm has to
    agree with them, or a stopped session would come back with its check-in
    restored the next time anything booted: the same class of accident as the
    reported defect, with the sign flipped.
    """
    from local_operator.harness.wake_types import WakeSchedule
    from local_operator.resume import write_session_attachment

    session_id = "0123456789ab"
    session_dir = isolated_root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    write_session_attachment(session_dir, team="", agent="aida", goal="")
    state.update_state(isolated_root, session_id=session_id)
    user_row = WakeSchedule(
        id="w1",
        kind="scheduled",
        message="a user's own reminder",
        every_ms=None,
        next_due_at=int(datetime(2026, 9, 30, 9, 0).timestamp() * 1000),
    )
    wake_store.write_entry(
        isolated_root,
        session_id,
        cwd=str(isolated_root),
        schedules=[user_row],
        preserve={"stopped_at": 1},
    )
    assert wake_store.is_held(wake_store.read_entry(isolated_root, session_id))

    assert await proactive.ensure_armed(isolated_root, session_id) == "held"
    assert _cadence_row(isolated_root, session_id) is None
    # ...and the user's own row is untouched by the pass either way.
    stopped = wake_store.read_entry(isolated_root, session_id)
    assert stopped is not None
    assert [row["id"] for row in stopped["schedules"]] == ["w1"]
    assert wake_store.is_held(stopped)


# -- the escalation tray's own symptom -----------------------------------


def _escalate(root: Path, message: str = "Follow up on the manifest churn") -> None:
    state.escalate_path(root).write_text(
        json.dumps({"wakes": [{"in": "10h", "message": message}]}), encoding="utf-8"
    )


def test_the_stale_day_marker_and_unconsumed_tray_share_the_one_root_cause(
    isolated_root: Path,
) -> None:
    """THE DAY-MARKER SYMPTOM, answered: same cause, not a second defect.

    The live install's ``aida/state.json`` read ``extras.day = 2026-09-29`` with
    an entry sitting unconsumed in ``escalate.json``. Both follow from the class
    read alone: ``reconcile``'s reactive branch returns before the tray drain
    (a stop the user asked for must not queue work for its own resume), so with
    her class reading reactive by accident the tray was never consumed and the
    day rolled over only in the file that no longer wrote. Fixing the class
    read fixes both — asserted here in the two directions, so a future change
    that consumes the tray while she is reactive fails rather than passing
    quietly.
    """
    session_id = "0123456789ab"
    (isolated_root / "sessions" / session_id).mkdir(parents=True)
    state.update_state(
        isolated_root,
        session_id=session_id,
        extras={"armed": 2, "day": "2026-09-29"},
    )
    _escalate(isolated_root)
    now = int(datetime(2026, 9, 30, 9, 0).timestamp() * 1000)

    # Reactive (the accident): the tray is left exactly where it was.
    reactive = proactive.reconcile(
        [], config_dir=isolated_root, session_id=session_id, class_reactive=True, now_ms=now
    )
    assert reactive.schedules == []
    assert state.escalate_path(isolated_root).exists()
    held = state.read_state(isolated_root)
    assert held is not None and held["extras"]["day"] == "2026-09-29"

    # Proactive (after the repair): the tray is consumed, the day rolls over in
    # the same write, and the request becomes a row.
    settled = proactive.reconcile(
        [], config_dir=isolated_root, session_id=session_id, class_reactive=False, now_ms=now
    )
    # The order inside the list is reconcile's own business (the ledger's rows
    # are merged before the cadence is ensured); what the invariant needs is
    # that BOTH are there — the cadence restored and the request turned into a
    # bounded one-shot.
    ids = sorted(row.id for row in settled.schedules)
    assert ids == sorted([proactive.CADENCE_ID, "aida-extra-1"])
    assert not state.escalate_path(isolated_root).exists()
    ledger = state.read_state(isolated_root)
    assert ledger is not None
    extras = ledger["extras"]
    assert extras["day"] == "2026-09-30" and extras["armed"] == 1


def _transcript_rows(root: Path, session_id: str) -> list[dict[str, Any]]:
    """The latest ``wake_schedules`` snapshot the session wrote."""
    from local_operator.wakes.arm import (  # noqa: PLC2701 — the reader the arm path uses
        _read_rows,
    )

    rows = _read_rows(root / "sessions" / session_id)
    return [row.model_dump() for row in rows]
