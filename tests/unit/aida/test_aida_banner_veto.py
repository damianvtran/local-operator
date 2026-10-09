"""The check-in BANNER VETO: her cadence may notify, and only when it should.

WHAT THESE TESTS PIN (design 2026-10-09 §1/§3/§7, PR-1). With no surface
attached, Aida's shipped cadence was SILENT whether her reply was actionable or
exactly the quiet sentinel (``Nothing needs your attention today.``; the legacy
``(no action needed)`` spelling is still recognised): ``WakeSchedule.notify`` defaults False and no
Aida row set it, so the completion ladder never had permission to speak. The
fix has two halves and both are pinned here:

* the cadence and the escalation extras DECLARE ``notify=True`` on their rows
  (armed through ``proactive`` — asserted on reconcile arms; the delivery-side
  cells drive rows with the flag set, exactly as the engine arms them);
* the settle-time VETO (``Session._aida_cadence_banner_veto``) turns that True
  back into a False when her ACTUAL REPLY says there is nothing to say — the
  quiet sentinel (normalised), a ``Tip:`` reply, an unsettled greeting ledger,
  or a spent 24 h banner budget. A blanket ``notify=True`` would banner
  the quiet sentinel every day, which is how a chief of staff gets muted.

THE RIG IS THE REAL ONE (pattern: ``tests/unit/session/test_attention_notify``):
a real ``Session`` over her attached session directory, real deliveries through
``_deliver_wake``, the real ``AttentionStore``, and — for the runtime arm — the
real ``ServingSessionHandle`` with only the OS spawn stubbed. ``detached_notify``
itself is doubled at the module seam the ladder uses, so nothing here can post
a real banner.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import pytest

import local_operator.tui.notify as notify_module
from local_operator.aida import onboarding, proactive, state
from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
from local_operator.harness.wake import DueWake
from local_operator.harness.wake_types import WakeSchedule
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream
from tests.unit.aida.conftest import mark_met
from tests.unit.session.test_runtime_completion_announce import (
    _ladder_exhausted,
    _until,
)
from tests.unit.session.test_session import wait_for

MODEL = ModelSpec(provider="test", model_id="aida-veto-model", context_window=100_000)
SESSION_ID = "feedc0de1111"


def _reply(text: str) -> ScriptedStream:
    """One model turn that answers ``text`` and stops."""
    return ScriptedStream([[StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]])


def _errored() -> ScriptedStream:
    """One model turn that ends in a provider error (the loop surfaces it)."""
    return ScriptedStream(
        [[StreamEndEvent(stop_reason="error", error="rig: simulated provider failure")]]
    )


def make_aida_session(root: Path, session_id: str, stream: ScriptedStream) -> Session:
    """A REAL session over ``root/sessions/<id>``, carrying her attachment.

    The same construction ``test_aida_session_hooks`` uses: the attachment
    (``agent="aida"``) is what ``bootstrap`` writes, and ``state.json`` naming
    the id is what makes ``Session._aida_duty`` True — without either, the
    veto short-circuits and these cells would pass vacuously.
    """
    from local_operator.resume import write_session_attachment

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    write_session_attachment(directory, team="", agent="aida", goal="")
    return Session(
        model=MODEL,
        model_source="config",
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        cwd=str(directory.parent),
    )


def _claim_spent(session_id: str, token: str | None) -> bool:
    """Whether the delivery watermark has passed this completion's row.

    Read straight from the store (the announce suite's pattern): asking
    ``claim_delivery`` would SPEND the claim it is inspecting, so this opens the
    database read-only.
    """
    import sqlite3

    from local_operator.paths import config_dir

    path = config_dir() / "attention.db"
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT sequence FROM completions WHERE conversation=? AND token=?",
            (f"session/{session_id}", token),
        ).fetchone()
        assert row is not None, "no completion row for this token"
        delivered = conn.execute(
            "SELECT delivered FROM deliveries WHERE conversation=?",
            (f"session/{session_id}",),
        ).fetchone()
    return bool(delivered) and int(delivered[0]) >= int(row[0])


async def _await_announcement(handle: ServingSessionHandle) -> None:
    """Wait until the turn-settled announcer has RUN and its ladder stopped.

    The handle is subscribed to the session's turn-settled seam at
    construction, so in these cells the completion is announced by the arm
    production uses (``_schedule_completion_announce``) — nothing calls
    ``_announce_completion`` by hand. Waiting on the SLOT rather than on the
    banner count is what makes the quiet cells non-vacuous: a settle and a
    delivery both end the ladder, and the assertions below separate them.
    """
    await _until(lambda: handle._completion_task is not None, timeout_s=10.0)
    await _ladder_exhausted(handle)


def _armed_root(root: Path, *, state_name: str = onboarding.GREETING_DELIVERED) -> None:
    """An install she has met, with the greeting ledger in a chosen state."""
    state.update_state(root, session_id=SESSION_ID)
    if state_name == onboarding.GREETING_DELIVERED:
        mark_met(root)
    else:
        state.write_json(state.onboarding_path(root), {"greeting": {"state": state_name}})


async def _deliver_checkin(
    session: Session,
    *,
    wake_id: str = proactive.CADENCE_ID,
    notify: bool = True,
) -> dict[str, Any]:
    """Deliver one check-in row and wait for the turn's completion row.

    Mirrors production: the row is what the engine arms (``notify=True`` for
    the cadence and the extras), the delivery spawns the turn
    (``_deliver_wake`` -> ``_prompt_messages``) and the settle publishes ONE
    value. Reads the store through the session's own refreshed state; the
    stream is the one the session was built with.
    """
    stream = cast(ScriptedStream, session._stream_fn)
    before = session._attention.get("completion_token")
    schedule = WakeSchedule(
        id=wake_id,
        message="Daily proactive check-in.",
        next_due_at=0,
        created_at=0,
        notify=notify,
    )
    await session._deliver_wake(
        DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    )
    await wait_for(lambda: bool(stream.requests))
    await wait_for(
        lambda: session._attention_run_settled
        and session._attention.get("completion_token") not in (None, before)
    )
    return await session.refresh_attention()


def _seed_banner_stamps(root: Path, stamps: list[int]) -> None:
    data = state.read_json(state.onboarding_path(root), what="onboarding") or {}
    data["banners"] = stamps
    state.write_json(state.onboarding_path(root), data)


def _banner_stamps(root: Path) -> list[int]:
    data = state.read_json(state.onboarding_path(root), what="onboarding") or {}
    return list(data.get("banners") or [])


# ---------------------------------------------------------------------------
# T2 — one constant, three writers.
# ---------------------------------------------------------------------------


def test_quiet_reply_is_one_constant_shared_by_prompt_extras_and_veto() -> None:
    """T2: the sentinel the veto matches is the one the prompts ASK for.

    Mutation that must turn this red: change the string in only one place
    (e.g. reword ``CADENCE_MESSAGE`` without moving ``QUIET_REPLY``) — the
    veto would then silently never match what the model was asked to send.
    """
    assert proactive.QUIET_REPLY in proactive.CADENCE_MESSAGE
    assert proactive.QUIET_REPLY in proactive.DEFAULT_EXTRA_MESSAGE
    assert proactive.QUIET_REPLY in proactive.with_quiet_clause("custom prose")
    assert proactive.reply_is_quiet(proactive.QUIET_REPLY) is True
    # The clause is idempotent: a message that already carries the sentinel is
    # returned unchanged (the cadence prompt itself goes through it on no path,
    # but extras reuse DEFAULT_EXTRA_MESSAGE).
    assert proactive.with_quiet_clause(proactive.DEFAULT_EXTRA_MESSAGE) == (
        proactive.DEFAULT_EXTRA_MESSAGE
    )


# ---------------------------------------------------------------------------
# T1 — the reply decides: sentinel variants, actionable, tip, error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "(no action needed)",
        "(No action needed).",
        "`(no action needed)`",
        "  (no action needed)  ",
        "NO ACTION NEEDED",
        "No action needed",
        "Nothing needs your attention today.",
        "NOTHING NEEDS YOUR ATTENTION TODAY",
        "`Nothing needs your attention today.`",
        "  Nothing needs your attention today.  ",
    ],
)
async def test_a_quiet_reply_and_its_variants_are_vetoed(isolated_root: Path, reply: str) -> None:
    """A quiet day must not banner — including the shapes the model actually writes.

    Mutation: delete the veto (every cell red), or drop the normalisation
    (every variant but the exact sentinel red — a raw ``==`` is the "fails
    loud on any paraphrase-less variant" failure the design rejected).
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _reply(reply))
    try:
        assert session._aida_duty is True
        result = await _deliver_checkin(session)
        assert result["kind"] == "complete"
        assert (
            result["notify"] is False
        ), f"a quiet reply ({reply!r}) banner-ed — the veto did not match it"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_actionable_reply_still_notifies(isolated_root: Path) -> None:
    """The other half of the requirement: actionable check-ins MUST reach out.

    Mutation: hardcode the veto to True (this cell, the tip cell below and
    every budget cell go red together).
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root,
        SESSION_ID,
        _reply("Build 42 is red — restart the renderer (session 121212121212)."),
    )
    try:
        result = await _deliver_checkin(session)
        assert result["kind"] == "complete"
        assert result["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_tip_reply_is_silent_by_default(isolated_root: Path) -> None:
    """A tip is a SILENT row by decision: it keeps the unseen mark, no banner.

    Mutation: invert the tip rule (veto only quiet replies, let tips through)
    — red. The one-line reversal switch is ``TIP_REPLY_NOTIFIES``; flipping it
    in the product is a deliberate operator decision and this cell would then
    require updating, which is the point of pinning it.
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root,
        SESSION_ID,
        _reply("Tip: phone access is not set up — offer the Radient relay."),
    )
    try:
        result = await _deliver_checkin(session)
        assert result["kind"] == "complete"
        assert result["notify"] is False
        assert result["unseen"] is True, "the tip must still land unread"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_errored_checkin_notifies_whatever_the_reply(isolated_root: Path) -> None:
    """Errors outrank the reply rule: a failed check-in is never silent.

    The veto is only ever applied to ``kind != "error"``; with an empty reply
    list the veto WOULD match (empty reply ⇒ quiet), so this cell goes red if
    the error exemption is dropped.
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _errored())
    try:
        result = await _deliver_checkin(session)
        assert result["kind"] == "error"
        assert result["notify"] is True, "an errored check-in must not be vetoed"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_user_armed_wake_is_never_vetoed(isolated_root: Path) -> None:
    """The ids gate: user intent outranks the sentinel rule.

    A ``wN`` wake the user armed with ``notify=True`` whose reply happens to be
    the sentinel still notifies: the veto applies only when EVERY wake id in
    the run is cadence-family. Mutation: treat any wake id (or no wake id) as
    cadence-family — red.
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _reply("(no action needed)"))
    try:
        result = await _deliver_checkin(session, wake_id="w1")
        assert result["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_trigger_checkin_is_never_vetoed(isolated_root: Path) -> None:
    """Trigger rows are outside the cadence family even when they notify.

    Trigger check-ins ask her to ACT, not to report, so the sentinel rule does
    not describe them and the veto must not reach them (their rows stay quiet
    in production by NOT setting notify — but if a future change sets it, this
    pin keeps the veto from silently re-silencing them for the wrong reason).
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _reply("(no action needed)"))
    try:
        result = await _deliver_checkin(session, wake_id="aida-trigger-1234abcd")
        assert result["notify"] is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# T3 — the greeting ledger, fail CLOSED; and a fresh install arms nothing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ledger_state",
    [onboarding.GREETING_OWED, onboarding.GREETING_REQUESTED, onboarding.GREETING_ARMED],
)
async def test_an_unsettled_greeting_ledger_vetoes_even_actionable_replies(
    isolated_root: Path, ledger_state: str
) -> None:
    """No pings before first engagement — and a ledger read error fails CLOSED.

    Mutation: remove the ledger clause from the veto — every cell here red
    (an actionable reply would banner for someone she has never met).
    """
    _armed_root(isolated_root, state_name=ledger_state)
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Something needs you: restart the worker.")
    )
    try:
        result = await _deliver_checkin(session)
        assert result["notify"] is False, f"ledger {ledger_state!r} must fail closed"
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ledger_state", [onboarding.GREETING_DELIVERED, onboarding.GREETING_SKIPPED]
)
async def test_a_settled_greeting_ledger_lets_actionable_replies_through(
    isolated_root: Path, ledger_state: str
) -> None:
    """The control arm for the cell above: delivered/skipped is the green path."""
    state.update_state(isolated_root, session_id=SESSION_ID)
    if ledger_state == onboarding.GREETING_DELIVERED:
        mark_met(isolated_root)
    else:
        state.write_json(
            state.onboarding_path(isolated_root),
            {"greeting": {"state": onboarding.GREETING_SKIPPED, "skipped_at": 1}},
        )
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Something needs you: restart the worker.")
    )
    try:
        result = await _deliver_checkin(session)
        assert result["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_install_gets_no_cadence_row_at_all(tmp_path: Path) -> None:
    """The never-contacted install arms nothing (measured in the design §3).

    This is the outer half of fail-closed: the veto would silence a check-in
    anyway, but a fresh root should not even have the row.
    """
    from local_operator import aida
    from local_operator.wakes import store as wake_store

    root = tmp_path / "config"
    root.mkdir()
    her_id = await aida.ensure_session(root)
    assert her_id
    entry = wake_store.read_entry(root, her_id) or {}
    assert [row["id"] for row in entry.get("schedules") or []] == []


# ---------------------------------------------------------------------------
# T4 — the rolling banner budget.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_fourth_banner_inside_a_day_is_vetoed(isolated_root: Path) -> None:
    """Three stamps in the window veto the next qualifying check-in.

    Mutation: cap to infinity — this cell red. The vetoed run must NOT consume
    budget (it never banners), so the stamps stay at three.
    """
    _armed_root(isolated_root)
    now = int(time.time() * 1000)
    _seed_banner_stamps(isolated_root, [now - 3_600_000, now - 7_200_000, now - 10_800_000])
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Still needs you: rotate the token.")
    )
    try:
        result = await _deliver_checkin(session)
        assert result["notify"] is False, "the 4th banner inside 24 h must be vetoed"
        assert len(_banner_stamps(isolated_root)) == 3, "a vetoed run must not stamp"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_true_publish_stamps_the_budget_exactly_once(isolated_root: Path) -> None:
    """One True publish, one stamp; a vetoed run stamps nothing.

    Mutation: stamp on every publish (the sentinel run below would then leave a
    stamp) or never stamp (the actionable run below leaves none) — either
    direction red.
    """
    _armed_root(isolated_root)
    actionable = make_aida_session(
        isolated_root, SESSION_ID, _reply("Needs you: restart the indexer.")
    )
    try:
        assert _banner_stamps(isolated_root) == []
        result = await _deliver_checkin(actionable)
        assert result["notify"] is True
        assert len(_banner_stamps(isolated_root)) == 1
    finally:
        await actionable.dispose()
    quiet = make_aida_session(isolated_root, SESSION_ID, _reply("(no action needed)"))
    try:
        result = await _deliver_checkin(quiet)
        assert result["notify"] is False
        assert len(_banner_stamps(isolated_root)) == 1, "a vetoed run must not stamp"
    finally:
        await quiet.dispose()


@pytest.mark.asyncio
async def test_the_budget_read_never_takes_the_aida_lock(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settle-time read is LOCK-FREE, on the event loop where the lock would park.

    Seeded at the cap, so a read that worked answers "spent" ⇒ veto. The lock
    is refused for the WHOLE run (the realistic refusal — ``WakeLockBusy``,
    which every other caller on this seam is documented to absorb), so a
    settle-time read that took it would come back unspent and the actionable
    reply would banner: mutation "move the lock into the read" ⇒ red. Nothing
    else in this run is harmed by the refusal either: the only other lock
    taker is the reconcile tray, whose refusal keeps the tray exactly where it
    was by design.
    """
    from local_operator.wakes.lock import WakeLockBusy

    _armed_root(isolated_root)
    now = int(time.time() * 1000)
    _seed_banner_stamps(isolated_root, [now - 1_000, now - 2_000, now - 3_000])

    def _refuse(*args: object, **kwargs: object) -> object:
        raise WakeLockBusy("rig: the aida lock is held for this cell")

    monkeypatch.setattr(state, "locked", _refuse)
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Needs you: renew the certificate.")
    )
    try:
        result = await _deliver_checkin(session)
        assert result["notify"] is False, "the cap must veto without the lock"
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# T5 — the runtime arm end to end: sentinel ⇒ 0 calls, actionable ⇒ 1.
# ---------------------------------------------------------------------------


@pytest.fixture
def detached_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Stub ``detached_notify`` at the module seam the ladder imports from."""
    calls: list[dict[str, str]] = []

    def fake(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        calls.append({"title": title, "body": body, "session_id": session_id})
        return True

    monkeypatch.setattr(notify_module, "detached_notify", fake)
    return calls


@pytest.fixture
def notification_path_on() -> object:
    """Clear the kill switch for the cells that drive the runtime arm.

    ``tests/conftest`` arms ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` for every test
    at import time; the runtime arm's first gate reads it. Set/restore by hand
    (the announce suite's pattern) so a test calling ``monkeypatch.undo()``
    cannot re-arm it mid-cell.
    """
    import os

    prior = os.environ.pop("LOCAL_OPERATOR_NO_NOTIFICATIONS", None)
    yield
    if prior is not None:
        os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = prior


@pytest.fixture(autouse=True)
def _this_process_is_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The identity gate must answer "the user's own home" in this rig.

    These cells run under pytest's autouse HOME redirect; without this patch
    the runtime arm's quiet-home check (``desktop_belongs_to_this_process(
    report=False)``) would settle before the ladder and the actionable cell
    could never reach ``detached_notify``. Patching the INPUT keeps the chain
    real; the redirected-home case has its own cell in
    ``tests/unit/session/test_runtime_completion_announce.py``.
    """
    monkeypatch.setattr("local_operator.supervisors.real_home", lambda: Path.home().resolve())


async def _rig_handle(session: Session, monkeypatch: pytest.MonkeyPatch) -> ServingSessionHandle:
    directory = session.transcript.directory
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    # \"No surface attached\": the per-session live-connection table is the
    # runtime server's own, with its own suite; doubling it here keeps this
    # file about the ladder.
    monkeypatch.setattr(handle, "_watching_surfaces", lambda: frozenset())
    return handle


@pytest.mark.asyncio
async def test_the_runtime_arm_stays_silent_on_a_sentinel_completion(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    detached_calls: list[dict[str, str]],
    notification_path_on: object,
) -> None:
    """T5, quiet half: a vetoed completion settles without a banner call.

    Mutation: remove the veto — the sentinel row reads notify=1 and the ladder
    raises one call.
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _reply("(no action needed)"))
    try:
        handle = await _rig_handle(session, monkeypatch)
        result = await _deliver_checkin(session)
        assert result["notify"] is False
        await _await_announcement(handle)
        assert detached_calls == [], "a sentinel completion must not raise a banner"
        assert (
            _claim_spent(SESSION_ID, result["completion_token"]) is False
        ), "the settle must leave the watermark for the user's own surfaces"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_runtime_arm_delivers_an_actionable_checkin_exactly_once(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    detached_calls: list[dict[str, str]],
    notification_path_on: object,
) -> None:
    """T5, actionable half: exactly one call, carrying HER session id.

    THE OPERATOR'S REQUIREMENT, at the ladder. Mutation: none of the fix — the
    row reads notify=0 and the call count is zero.
    """
    _armed_root(isolated_root)
    session = make_aida_session(isolated_root, SESSION_ID, _reply("Needs you: approve the deploy."))
    try:
        handle = await _rig_handle(session, monkeypatch)
        result = await _deliver_checkin(session)
        assert result["notify"] is True
        await _await_announcement(handle)
        assert len(detached_calls) == 1, detached_calls
        assert detached_calls[0]["session_id"] == SESSION_ID
        assert (
            _claim_spent(SESSION_ID, result["completion_token"]) is True
        ), "the delivered banner must have spent the claim"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_runtime_arm_stays_silent_when_notifications_are_off(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    detached_calls: list[dict[str, str]],
    notification_path_on: object,
) -> None:
    """ "Display notifications off" silences even an actionable check-in.

    The veto can only ever QUIET a run; the kill switch refuses the banner on
    the other side, and both must hold. Mutation: remove the
    ``notifications_enabled`` guard — red.
    """
    from local_operator.tui.settings import settings_reload
    from tests.unit.aida.conftest import write_config

    _armed_root(isolated_root)
    # ``display.notifications`` is a LITERAL dotted top-level key in ``values``
    # (see ``tui/settings.py``'s module note), which is the shape
    # ``write_config`` writes and the settings reader looks up. The reader
    # caches, so reload after the write.
    write_config(isolated_root, {"display.notifications": False})
    settings_reload()
    session = make_aida_session(isolated_root, SESSION_ID, _reply("Needs you: approve the deploy."))
    try:
        handle = await _rig_handle(session, monkeypatch)
        result = await _deliver_checkin(session)
        assert result["notify"] is True, "the row may still say notifiable"
        await _await_announcement(handle)
        assert detached_calls == [], "notifications off must refuse the banner"
        assert (
            _claim_spent(SESSION_ID, result["completion_token"]) is False
        ), "a refused banner must not spend the watermark"
    finally:
        await session.dispose()
        settings_reload()


# ---------------------------------------------------------------------------
# The extras: armed rows carry notify=True and the sentinel clause.
# ---------------------------------------------------------------------------


def test_reconcile_arms_notifying_rows_and_appends_the_quiet_clause(
    isolated_root: Path,
) -> None:
    """The cadence and a custom-message extra are armed with intent + clause.

    Mutation: drop ``notify=True`` from either arm site (the corresponding row
    reads False) or drop ``with_quiet_clause`` (the custom message has no
    sentinel to obey — red).
    """
    from local_operator.resume import write_session_attachment
    from tests.unit.aida.test_aida_proactive import _pin_cadence_away_from_now

    directory = isolated_root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    write_session_attachment(directory, team="", agent="aida", goal="")
    state.update_state(isolated_root, session_id=SESSION_ID)
    mark_met(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    state.write_json(
        state.escalate_path(isolated_root),
        {"wakes": [{"in": "4h", "message": "check the deploy"}]},
    )
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID)
    by_id = {row.id: row for row in result.schedules}
    assert by_id[proactive.CADENCE_ID].notify is True
    extra = next(row for row in result.schedules if row.id.startswith(proactive.EXTRA_ID_PREFIX))
    assert extra.notify is True
    assert proactive.QUIET_REPLY in extra.message, "the custom message lacks the sentinel clause"


# ---------------------------------------------------------------------------
# T12 (remediation round 1) — the catch-up fold, mixed runs, the budget
# stamp and the legacy upgrade. Each cell names its own mutation.
# ---------------------------------------------------------------------------


def _replies(count: int, text: str) -> ScriptedStream:
    """``count`` identical model turns — a multi-prompt cell's tape."""
    return ScriptedStream(
        [[StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")] for _ in range(count)]
    )


def _pending_catchup(session: Session, *, ids: set[str]) -> None:
    """Put the session in the pending-catch-up state ``load`` would build.

    ``_prepare_missed_wake_catchup`` is what sets these in production (an
    overdue row found at load); the cells here build the same state directly
    so the FIRE / TAKE / delivery / veto machinery under test is the real
    one. ``_resume_catchup_notify`` mirrors ``_prepare``'s own computation
    (``any(row.notify for the folded rows)``) — the cadence row's notify is
    True by this change, so the fold asks to notify.
    """
    session._resume_catchup_ids = set(ids)
    session._resume_catchup_text = "While you were away: " + ", ".join(sorted(ids))
    session._resume_catchup_sent = False
    session._resume_catchup_notify = True
    session._resume_grace_ends_ms = float("inf")  # grace NOT passed yet


async def _settle_catchup_turn(session: Session) -> None:
    stream = cast(ScriptedStream, session._stream_fn)
    await wait_for(lambda: bool(stream.requests))
    await wait_for(
        lambda: session._attention_run_settled and bool(session._attention.get("completion_token"))
    )


@pytest.mark.asyncio
async def test_a_pre_grace_fire_keeps_the_fold_for_the_catchup(isolated_root: Path) -> None:
    """QA Q1: a fire landing before the grace deadline must not spend its id.

    The race: the wake layer's re-armed due and ``_resume_grace_ends_ms`` are
    stamped from two different clocks, so a fire can beat the deadline by a
    millisecond — the take returns None and the grace timer still owes the
    delivery. If the fire drops the id anyway, the timer builds the message
    from an EMPTY fold (``wake_ids: []``) and a quiet day banner-ed (QA
    reproduced 2 of 3 sentinel runs through the real supervisor).

    Mutation: reinstate the unconditional ``discard`` in
    ``_deliver_wake_catchup`` → the retention assert goes red.
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Nothing needs your attention today.")
    )
    try:
        _pending_catchup(session, ids={proactive.CADENCE_ID})
        row = WakeSchedule(
            id=proactive.CADENCE_ID,
            message="Daily proactive check-in.",
            next_due_at=0,
            created_at=0,
            notify=True,
        )
        # The fire that beats the deadline: take returns None, nothing delivered.
        await session._deliver_wake_catchup(
            DueWake(schedule=row, occurrence=1, planned_total=1, final=True)
        )
        assert (
            proactive.CADENCE_ID in session._resume_catchup_ids
        ), "a pre-grace fire spent the fold id — the catch-up would build empty"
        # The grace timer now delivers with the fold intact, and the sentinel
        # still vetoes on the catch-up path.
        session._resume_grace_ends_ms = 0
        message = session._take_resume_catchup()
        assert message is not None
        assert message.details["wake_ids"] == [proactive.CADENCE_ID]
        session._deliver_resume_catchup(message)
        await _settle_catchup_turn(session)
        result = await session.refresh_attention()
        assert result["kind"] == "complete"
        assert result["notify"] is False, "the quiet sentinel must still veto on the catch-up path"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_catchup_with_a_lost_fold_fails_closed(isolated_root: Path) -> None:
    """QA Q1's raced shape, deterministically: ``wake_ids: []`` must NOT read
    as "no Aida rows".

    The take is handed an EMPTY fold set — exactly the message the race
    produced. The belt makes the run Aida-bearing, so the sentinel still
    vetoes; without it the empty ids short-circuit the veto and the banner
    ships.

    Mutation: drop the ``_run_catchup_unidentified`` branch (return False on
    empty ids, as before) → this cell's notify assertion goes red.
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Nothing needs your attention today.")
    )
    try:
        _pending_catchup(session, ids=set())
        session._resume_grace_ends_ms = 0
        message = session._take_resume_catchup()
        assert message is not None
        assert message.details["wake_ids"] == []
        session._deliver_resume_catchup(message)
        await _settle_catchup_turn(session)
        assert session._run_catchup_unidentified is True, "the belt must record the lost fold"
        result = await session.refresh_attention()
        assert result["notify"] is False
    finally:
        await session.dispose()


def test_a_legacy_cadence_row_is_upgraded_in_place(isolated_root: Path) -> None:
    """Review MAJOR-1: the upgrade must reach an ESTABLISHED install.

    A row armed before ``notify`` existed (``notify=False``) sits in the
    resident list; reconcile keeps it and must flip the bit ONCE, preserving
    ``created_at`` — otherwise the headline fix ships only to fresh installs
    (the first cut's bug: the upgrade lived inside the create-only branch,
    which never runs while a row exists).

    Mutation: delete the upgrade block in ``reconcile`` → the notify assert
    goes red.
    """
    _armed_root(isolated_root)
    legacy = WakeSchedule(
        id=proactive.CADENCE_ID,
        message="Daily proactive check-in.",
        next_due_at=123,
        created_at=456,
        notify=False,
    )
    first = proactive.reconcile(
        [legacy], config_dir=isolated_root, session_id=SESSION_ID, now_ms=789_000
    )
    kept = next(row for row in first.schedules if row.id == proactive.CADENCE_ID)
    assert (
        kept.notify is True
    ), "the legacy row was not upgraded — established installs stay silent forever"
    assert kept.created_at == 456, "the upgrade must not remint created_at"
    assert first.changed is True
    second = proactive.reconcile(
        first.schedules, config_dir=isolated_root, session_id=SESSION_ID, now_ms=789_001
    )
    assert second.changed is False, "the upgrade must settle — one flip, then stable"


@pytest.mark.asyncio
async def test_ordinary_publishes_do_not_spend_the_banner_budget(isolated_root: Path) -> None:
    """Review MAJOR-2: only her check-in banners stamp the budget.

    Three ordinary turns of hers (each a True publish by user semantics) used
    to stamp three timestamps and veto the next actionable check-in. The
    budget bounds BANNERS, not publishes.

    Mutation: stamp for every True publish of hers (the old gate,
    ``if notify and self._aida_duty``) → the final assert goes red (budget
    spent → the actionable check-in is vetoed).
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root,
        SESSION_ID,
        _replies(4, "Build 42 is red — restart the renderer (session 121212121212)."),
    )
    try:
        for _ in range(3):
            before = session._attention.get("completion_token")
            await session.prompt("Status please")
            await wait_for(
                lambda: session._attention_run_settled
                and session._attention.get("completion_token") not in (None, before)
            )
        assert _banner_stamps(isolated_root) == [], "ordinary publishes must not spend the budget"
        result = await _deliver_checkin(session)
        assert result["notify"] is True, "the actionable check-in must still banner"
        assert len(_banner_stamps(isolated_root)) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_mixed_user_and_cadence_run_keeps_its_banner(isolated_root: Path) -> None:
    """Review MAJOR-3: user intent is never silenced — the veto is exclusive-scope.

    A user prompt that runs while a catch-up is pending INLINES the fold into
    the SAME turn (``prompt``). The reply is the quiet sentinel, but the run
    carries a person's message, so notification semantics stay the user's —
    and the mixed run must not spend the banner budget either.

    Mutation: drop the trigger-classes clause from ``_aida_checkin_run`` → the
    veto scopes a mixed run and this cell goes red (notify False).
    """
    _armed_root(isolated_root)
    session = make_aida_session(
        isolated_root, SESSION_ID, _reply("Nothing needs your attention today.")
    )
    try:
        _pending_catchup(session, ids={proactive.CADENCE_ID})
        session._resume_grace_ends_ms = 0
        await session.prompt("Good morning — anything for me?")
        await wait_for(
            lambda: session._attention_run_settled
            and bool(session._attention.get("completion_token"))
        )
        assert "user" in session._run_triggers
        assert session._run_wake_ids == {proactive.CADENCE_ID}
        result = await session.refresh_attention()
        assert result["notify"] is True, "a mixed run must not be silenced by her sentinel"
        assert _banner_stamps(isolated_root) == [], "a mixed publish is not her banner"
    finally:
        await session.dispose()
