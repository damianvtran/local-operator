"""Auto-naming scheduling: when the title call fires, and what a failure costs.

The naming errand is decoration with one hard rule — it must never make the
session it decorates worse. Three regressions shaped it:

* Launched concurrently with the opening turn, it was a second simultaneous
  request that could rotate the turn's credential, pin the turn to a fallback
  model, or spend the turn's effort boundary. Naming was deferred until the
  turn settled to avoid that, which cured the symptom and created the next
  one. The safety now lives in the SHAPE of the request instead
  (``ChatRequest.isolated``, covered in ``tests/unit/providers/test_failover.py``
  and ``tests/unit/model/test_configure.py``), so the call is concurrent again.
* Deferred, the title landed after the turn. Measured against a real provider:
  a 29.7-second opening turn, the ``lo › <cwd>`` fallback on the tab for all
  29.7 of those seconds, and the title stored 1.8 seconds after the answer was
  already on screen. On a product whose first turn runs for minutes, "the tab
  never becomes a title" was an accurate description of what a user saw.
* A failed attempt used to spend the conversation's one naming request, so a
  minute-zero failure left the session (and the terminal title, which names
  the surface) on the cwd fallback forever. The latch now releases on
  failure and the next substantive message retries.

The re-title half is here too: a conversation drifts, and the MODEL — not a
keyword rule — judges whether a new message moved the subject.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session import naming
from local_operator.tui.app import RETITLE_MIN_GAP_S, OperatorApp
from local_operator.tui.terminal_title import TerminalTitle
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text


class _GatedSession(FakeSession):
    """A session whose ``prompt`` blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.title = ""
        self.name_gate: asyncio.Event | None = None
        # The two "this worker got here" barriers. `_settle` yields for a fixed
        # number of ticks, which is a guess about the scheduler rather than a
        # fact about it, so a test asserting something a WORKER produces waits
        # for the worker to say it got there instead of counting ticks. (The
        # flake that started this was not the scheduler at all — see `_ready`.)
        self.name_started = asyncio.Event()
        self.prompt_started = asyncio.Event()
        self.timeline: list[str] = []

    async def prompt(self, text: str, images: Any = None) -> None:  # noqa: ANN401
        self.prompts.append(text)
        self.timeline.append(f"prompt:{text}")
        self.prompt_started.set()
        await self.gate.wait()

    async def complete_once(self, system: str, prompt: str) -> str:
        self.completions.append((system, prompt))
        self.timeline.append("name:start")
        self.name_started.set()
        if self.name_gate is not None:
            try:
                await self.name_gate.wait()
            except asyncio.CancelledError:
                self.timeline.append("name:cancel")
                raise
        return self.title

    def grow_transcript(self, turns: int) -> None:
        """Stand in for a conversation that has accumulated ``turns`` messages.

        The growth-gated re-title schedule counts user/assistant turns off
        ``session.history()``, so a test that wants the gate OPEN has to present
        a transcript long enough to clear it. The real ``prompt`` on this fake
        never builds history (it only records the text), so this seeds it
        directly with alternating user/assistant entries carrying the role and
        text the theme sampler reads — enough shape for both the turn count and
        the ``<chat>`` sample, nothing more.
        """
        self._history = [
            SimpleNamespace(
                role="user" if index % 2 == 0 else "assistant",
                text=f"message {index}",
            )
            for index in range(turns)
        ]


async def _settle() -> None:
    """Let the app's workers run to a standstill.

    Twelve short yields rather than four: a turn worker, a naming worker and a
    re-title worker can be scheduled by one submit, and each of them has to get
    from `run_worker` to its first await. Bounded ticks are only sound AFTER a
    submit that actually dispatched something — see :func:`_ready` for the race
    that no number of ticks here could ever have fixed.
    """
    for _ in range(12):
        await asyncio.sleep(0.02)


async def _ready(pilot: Any, app: OperatorApp) -> None:  # noqa: ANN401
    """Pump the app until the session has been adopted.

    THE flake this file kept hitting, and it was never about naming.
    ``_submit_prompt`` early-returns a "session is still starting…" notice while
    ``app._session`` is None: no turn, no naming call, no ``session.prompts`` —
    which is the bare `+ []` / `- ['fix the login redirect loop']` diff the suite
    failed with beside its neighbours. Instrumented under 16 CPU hogs, 3 of 3
    failures reported ``prompts=[] workers=[]`` with that notice in the
    transcript, so the submit had beaten adoption rather than the workers being
    slow afterwards.

    One ``pilot.pause()`` is enough on an idle machine and a coin flip on a
    loaded one, because adoption takes an unknown NUMBER of pumps, not an
    interval. So this waits on boot progress and is bounded by pumps rather than
    by the clock — a starved machine takes longer instead of failing, while a
    session that genuinely never arrives still fails instead of hanging.
    """
    for _ in range(200):
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError("the session was never adopted")


async def _boot(title: str = "") -> tuple[OperatorApp, _GatedSession]:
    session = _GatedSession()
    session.title = title
    app = OperatorApp(lambda: _factory(session))
    # Held open by the caller's `async with app.run_test(...)`; returning both
    # handles from inside the context keeps every test one flat block.
    return app, session


@pytest.mark.asyncio
async def test_the_title_is_generated_DURING_the_live_turn() -> None:
    """THE latency regression, pinned at the scheduler.

    The errand used to await ``_turn_settled`` before asking for a title, and a
    first turn on this product runs for minutes. Here the turn is parked
    indefinitely on its gate and the title must still arrive: no provider call
    at all before the gate opens is the bug, not the contract.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("fix the login redirect loop")
        # `_ready` above is what fixed this test's flake (the submit was beating
        # session adoption); these two barriers are what make the ASSERTIONS
        # below independent of tick counts, since both facts they check are
        # produced by workers rather than by the submit. The naming one is
        # suppressed rather than awaited outright so a real regression — no title
        # call at all — still fails on the assertion that names it instead of on
        # a timeout that does not.
        await asyncio.wait_for(session.prompt_started.wait(), timeout=5)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(session.name_started.wait(), timeout=2)
        await _settle()

        # The turn has not finished and never will until the gate opens.
        assert session.prompts == ["fix the login redirect loop"]
        assert not session.gate.is_set()
        # ...and the title is already stored and painted.
        assert len(session.completions) == 1, "the title call waited for the turn"
        assert session.conversation_name == "Fix the login flow"
        assert app._status is not None
        assert app._status._conversation_name == "Fix the login flow"
        session.gate.set()


@pytest.mark.asyncio
async def test_a_follow_up_turn_leaves_an_inflight_naming_call_alone() -> None:
    """A second prompt neither waits for the title nor throws it away.

    Turn start used to cancel naming, because the title call took the same
    provider lane and a user request had to be able to evict it. An isolated
    call is not in that lane, and cancelling one would re-derive the title from
    the SECOND message — naming the conversation after its follow-up.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    session.name_gate = asyncio.Event()  # hold the title call open
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("fix the login redirect loop")
        await asyncio.wait_for(session.name_started.wait(), timeout=1)

        session.gate.set()  # let the opening turn finish
        app._submit_prompt("also add the assignee email column")
        await _settle()

        # The follow-up ran without waiting for the title...
        assert session.prompts == [
            "fix the login redirect loop",
            "also add the assignee email column",
        ]
        # ...and did not evict it.
        assert "name:cancel" not in session.timeline
        assert [item for item in session.timeline if item == "name:start"] == ["name:start"]

        session.name_gate.set()
        await _settle()
        # The title names the OPENER, which is what the conversation is about.
        assert session.conversation_name == "Fix the login flow"
        assert session.completions[0][1] == "fix the login redirect loop"


@pytest.mark.asyncio
async def test_reload_cancels_and_drains_naming_before_adoption() -> None:
    """A replacement session never inherits the old title call's provider lock."""
    first = _GatedSession()
    first.title = "<title>Old session</title>"
    first.name_gate = asyncio.Event()
    second = _GatedSession()
    sessions = [first, second]
    calls = 0
    app: OperatorApp

    async def factory() -> _GatedSession:
        nonlocal calls
        session = sessions[calls]
        calls += 1
        if calls == 2:
            # The old naming worker must have unwound before construction and
            # adoption of the replacement.
            assert not app._turn_provider_lock.locked()
        return session

    app = OperatorApp(factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._maybe_name_conversation("name the old session")
        await asyncio.wait_for(first.name_started.wait(), timeout=1)

        await app._reload_session()
        await _settle()

        assert first.disposed
        assert "name:cancel" in first.timeline
        assert app._session is second
        assert not app._turn_provider_lock.locked()


@pytest.mark.asyncio
async def test_failed_attempt_releases_the_latch_for_a_retry() -> None:
    """A dead naming call costs a delayed title, not a nameless session."""
    app, session = await _boot(title="")  # complete_once answers nothing usable
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._maybe_name_conversation("please review the export columns")
        await _settle()
        assert session.completions, "attempt never ran"
        assert app._name_requested is False, "failure kept the once-only latch"

        session.title = "<title>Bulk export columns</title>"
        app._maybe_name_conversation("also add the assignee email column")
        await _settle()
        assert len(session.completions) == 2, "retry did not fire on the next message"
        assert session.conversation_name == "Bulk export columns"


@pytest.mark.asyncio
async def test_landed_title_paints_the_band_and_keeps_the_latch() -> None:
    """A successful attempt stores the name, paints the band, and stays spent."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._maybe_name_conversation("fix the login flow")
        await _settle()
        assert session.conversation_name == "Fix the login flow"
        assert app._name_requested is True
        assert app._status is not None
        assert app._status._conversation_name == "Fix the login flow"

        # A second substantive message inside the re-title window must not
        # spend a call: the title was decided seconds ago, and a tab label that
        # can change on every message is one nobody can read. (Past the window,
        # the model gets asked — see the re-title tests below.)
        session.title = "<title>A different title</title>"
        app._maybe_name_conversation("now something else entirely")
        await _settle()
        assert len(session.completions) == 1
        assert session.conversation_name == "Fix the login flow"


@pytest.mark.asyncio
async def test_the_band_is_named_from_the_opener_before_any_provider_call() -> None:
    """The instant label: the user's own words, no network, same frame.

    The generated title now lands a second or two later rather than after the
    turn, but "a second or two" is still visible on a tab bar, and the excerpt
    costs nothing. So the band is asserted SYNCHRONOUSLY after submit, before
    the naming worker has had a scheduler tick.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("summarise how compaction picks a cut point in this repo")

        # No await between the submit and here: nothing asynchronous has run.
        assert session.completions == []
        assert app._status is not None
        assert app._status._conversation_name == "Summarise how compaction picks a cut point in…"
        # The STORE is untouched: every gate in the errand reads it to mean
        # "something already named this", so a stand-in written there would
        # cancel the call it is standing in for.
        assert session.conversation_name == ""

        await _settle()
        # The generated title supersedes the excerpt on both the store and the
        # band — and while the turn is still parked on its gate.
        assert not session.gate.is_set()
        assert session.conversation_name == "Fix the login flow"
        assert app._status._conversation_name == "Fix the login flow"
        assert app._provisional_name == ""
        session.gate.set()


@pytest.mark.asyncio
async def test_a_dead_naming_call_leaves_the_opener_on_the_band() -> None:
    """A provider failure costs the better title, not the label.

    Before, this case fell all the way back to the working directory: the one
    naming attempt had been spent on a call that returned nothing. The excerpt
    is strictly better than `lo › tmp` and it is already on screen, so it stays.
    """
    app, session = await _boot(title="")  # empty reply: parse_title returns None
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("fix the login redirect loop")
        session.gate.set()
        await _settle()

        assert session.completions, "the naming call never ran"
        assert session.conversation_name == ""
        assert app._status is not None
        assert app._status._conversation_name == "Fix the login redirect loop"
        # The latch released, so a later substantive message may still retry...
        assert app._name_requested is False
        # ...but the label does NOT churn to the newer message: a conversation is
        # identified by what it was opened for.
        session.title = "<title>Bulk export columns</title>"
        app._submit_prompt("now add a csv export")
        session.gate.set()
        await _settle()
        assert session.conversation_name == "Bulk export columns"
        assert app._status._conversation_name == "Bulk export columns"


@pytest.mark.asyncio
async def test_a_low_signal_opener_names_nothing_at_all() -> None:
    """ "hi" is not a title in either half of the module.

    The deterministic filter gates the stand-in as well as the provider call, so
    a greeting leaves the band on its directory fallback rather than putting the
    word "Hi" on the user's tab bar.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("hi")
        session.gate.set()
        await _settle()
        assert session.completions == []
        assert app._provisional_name == ""
        assert app._status is not None
        assert app._status._conversation_name == ""


@pytest.mark.asyncio
async def test_a_dead_naming_call_retries_when_a_fallback_pins() -> None:
    """Isolated naming 429s on the dead primary; the route edge re-fires it.

    The first turn's naming call races the fallback pin and loses. Without a
    retry on ``EffectiveModelChanged`` the conversation stays untitled for
    the whole session — the phone list and header never update.
    """
    from local_operator.tui.events import EffectiveModelChanged

    app, session = await _boot(title="")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("review what regressed in mobile titles")
        session.gate.set()
        await _settle()
        assert session.conversation_name == ""
        assert app._pending_name_text == "review what regressed in mobile titles"

        session.title = "<title>Mobile title sync</title>"
        app.on_effective_model_changed(
            EffectiveModelChanged(
                provider="xai",
                model_id="grok-4.6",
                effort=None,
                reason="quota exhausted",
                is_fallback=True,
            )
        )
        await _settle()
        assert session.conversation_name == "Mobile title sync"
        assert app._pending_name_text == ""
        assert app._status is not None
        assert app._status._conversation_name == "Mobile title sync"


@pytest.mark.asyncio
async def test_resume_of_a_live_session_does_not_open_a_second_writer(
    tmp_path, monkeypatch
) -> None:
    """A phone-started session is already live; /resume must not fork it.

    Two writers on one transcript is how a TUI resume of a mobile session
    painted the splash: the second process claimed the directory and
    replayed a mid-write journal.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = tmp_path / "sessions" / "live00000001"
    live.mkdir(parents=True)
    (live / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (live / ".session.pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr("os.kill", lambda pid, sig: None)

    rebuilt = {"called": False}

    async def resume_factory(_resume_id: str | None):
        rebuilt["called"] = True
        return FakeSession()

    app, session = await _boot()
    app._resume_factory = resume_factory
    notices: list[str] = []

    def notice(body: str, kind: str = "info") -> None:
        notices.append(body)

    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._resume_session("live00000001", notice)
        await _settle()
        assert rebuilt["called"] is False
        # The refuse is a system notice (splash stays), not the slash
        # receipt callback — look at the transcript, not ``notices``.
        text = _transcript_text(app)
        assert "already open" in text
        assert "pid 4242" in text
        assert "second writer" not in text
        assert app._session is session
        # Splash survives a refused navigation (D1).
        from local_operator.tui.widgets.welcome import WelcomeView

        assert app.query_one(WelcomeView).display is True

        # F1: ``@latest`` must resolve then refuse, not skip the owner check.
        rebuilt["called"] = False
        app._resume_session("@latest", notice)
        await _settle()
        assert rebuilt["called"] is False
        assert app._session is session


# -- re-titling a conversation that has drifted -------------------------------


def _fake_clock(app: OperatorApp) -> list[float]:
    """Replace the app's monotonic seam with a one-cell list the test drives.

    Installed BEFORE the first title lands: ``_store_title`` stamps the
    re-title window from this same clock, so a fake swapped in afterwards
    would leave the window stamped at real monotonic time and no amount of
    advancing would ever open it.

    Starts well away from zero, because zero is also ``_retitle_checked_at``'s
    initial value: a clock starting at 0 makes "the window was never stamped"
    and "the window was stamped just now" indistinguishable, which is exactly
    the bug these tests have to be able to see. Real monotonic time is a large
    number for the same reason this one is.
    """
    now = [10_000.0]
    app._clock = lambda: now[0]  # type: ignore[method-assign]
    return now


async def _named(app: OperatorApp, session: _GatedSession, opener: str) -> list[float]:
    """Drive ``app`` to a stored, model-generated title on a controlled clock.

    Leaves the re-title window CLOSED, which is where a freshly named
    conversation genuinely is: a caller that wants to exercise the re-title
    path advances the returned clock itself, so no test is quietly relying on
    a helper to decide the throttle for it.
    """
    now = _fake_clock(app)
    app._submit_prompt(opener)
    session.gate.set()
    # The naming barrier, because this helper is a PRECONDITION for seven tests:
    # its own `assert` below is what a slipped tick surfaces as, and "the first
    # title never landed" describes the scheduler rather than anything about
    # re-titling. (The boot race that made it fail in the first place is handled
    # by `_ready` in each test, before this runs.)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(session.name_started.wait(), timeout=5)
    await _settle()
    assert session.conversation_name, "the first title never landed"
    # A fresh gate, already open, so the follow-up turns below run through
    # instead of parking on the opener's spent one.
    session.gate = asyncio.Event()
    session.gate.set()
    return now


async def _named_a_while_ago(app: OperatorApp, session: _GatedSession, opener: str) -> list[float]:
    """:func:`_named`, then into a state where the next message may re-title.

    Two gates guard the re-title now, and this opens both. It advances the fake
    clock past ``RETITLE_MIN_GAP_S`` (the secondary churn floor) AND grows the
    transcript past the growth gate: the first title seeds
    ``_last_titled_turn_count`` from the transcript, which the fake leaves at 0,
    so ``should_refresh_theme`` needs the history to reach ``THEME_REFRESH_MIN_TURNS``
    (4) before a re-title is eligible. Six turns clears it with margin and is the
    length the geometric schedule's first refresh lands at anyway.
    """
    now = await _named(app, session, opener)
    now[0] += RETITLE_MIN_GAP_S + 1
    session.grow_transcript(6)
    return now


@pytest.mark.asyncio
async def test_a_material_change_of_subject_retitles_both_surfaces() -> None:
    """The whole point of the re-title: a drifted session names its subject.

    The model — not this code — decides that the subject moved; the fake stands
    in for it by answering with a replacement title instead of the sentinel.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        assert app._status is not None
        # A real title writer over a capture sink: the terminal tab is the
        # surface the whole feature is named after, and asserting the band
        # alone would not notice the two drifting apart.
        written: list[str] = []
        title = TerminalTitle(written.append)
        title.start()
        app._status.set_terminal_title(title)

        await _named_a_while_ago(app, session, "fix the login redirect loop")
        assert "Fix the login flow" in title.current

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()

        assert len(session.completions) == 2, "no re-title check was made"
        # The re-title call now carries the theme prompt, and the current title
        # rides the DATA as the `<current-title>` anchor rather than the system
        # block — the whole point of titling the trajectory, not the message.
        retitle_system, retitle_data = session.completions[1]
        assert retitle_system == naming.THEME_SYSTEM_PROMPT
        assert "<current-title>\nFix the login flow\n</current-title>" in retitle_data
        assert retitle_data.startswith("<chat>")
        # The newest message reaches the model as the tail of the trajectory.
        assert "forget that, rewrite the billing importer instead" in retitle_data
        assert session.conversation_name == "Billing importer rewrite"
        assert app._status._conversation_name == "Billing importer rewrite"
        assert "Billing importer rewrite" in title.current
        assert any("Billing importer rewrite" in chunk for chunk in written)


@pytest.mark.asyncio
async def test_the_sentinel_answer_leaves_the_title_and_the_band_alone() -> None:
    """A "same subject" answer must not repaint. A band that rewrites itself
    mid-turn with the same words is a flicker the user reads as a glitch."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named_a_while_ago(app, session, "fix the login redirect loop")
        assert app._status is not None
        painted: list[str] = []
        real_update = app._status.update

        def spy_update(**kwargs: Any) -> None:
            if kwargs.get("conversation_name") is not None:
                painted.append(kwargs["conversation_name"])
            real_update(**kwargs)

        app._status.update = spy_update  # type: ignore[method-assign]

        session.title = "<title/>"  # the model says: unchanged
        app._submit_prompt("and make the redirect preserve the query string")
        await _settle()

        assert len(session.completions) == 2, "the model was never asked"
        assert session.conversation_name == "Fix the login flow"
        assert painted == [], "the band repainted on a no-change answer"


@pytest.mark.asyncio
async def test_a_human_rename_is_never_overwritten_by_a_retitle() -> None:
    """``user_set`` wins permanently, and costs no call to defend."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named_a_while_ago(app, session, "fix the login redirect loop")
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation"
        assert len(session.completions) == 1, "a renamed conversation spent a re-title call"


@pytest.mark.asyncio
async def test_a_settled_session_does_not_even_spend_a_check_on_an_in_goal_step() -> None:
    """The growth gate at the host level: an in-goal step on a long, settled
    session spends NO provider call at all.

    This is the drift the whole change exists to stop, defended one layer up
    from the sampler. The conversation was named at turn 1 and has since grown —
    but not doubled — so ``should_refresh_theme`` declines before any call is
    dispatched. Under the old wall-time-only throttle this same message an hour
    in would have spent a check and, shown only itself, pivoted the title.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        now = await _named(app, session, "fix the login redirect loop")
        # Past the time floor, so only the growth gate can decline this. The
        # title landed with the transcript at length 0, so the baseline is 0 and
        # the floor is four turns; three turns is short of it.
        now[0] += RETITLE_MIN_GAP_S + 1
        session.grow_transcript(3)

        session.title = "<title>Find Port Credit restaurants</title>"
        app._submit_prompt("now find me some good Port Credit restaurants to try it out")
        await _settle()

        assert len(session.completions) == 1, "a settled session spent a re-title check"
        assert session.conversation_name == "Fix the login flow"


@pytest.mark.asyncio
async def test_the_refresh_budget_stops_re_titling_after_the_cap() -> None:
    """Growth alone would keep re-titling a long session; the cap makes it stop.

    Five automatic re-titles are the ceiling. Here the budget is driven to the
    cap directly, then a transcript grown far past any growth threshold is shown
    an unmistakable change of subject — and it is declined without a call,
    because a session with an established identity must stop tracking the cursor.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        now = await _named(app, session, "fix the login redirect loop")
        now[0] += RETITLE_MIN_GAP_S + 1
        # The budget is spent: five refreshes already charged.
        app._theme_refresh_count = 5
        app._last_titled_turn_count = 1
        session.grow_transcript(500)

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget all that, rewrite the billing importer instead")
        await _settle()

        assert len(session.completions) == 1, "a capped session spent a re-title check"
        assert session.conversation_name == "Fix the login flow"


# -- `/rename`: the human's title ---------------------------------------------


async def _type_command(pilot: Any, app: OperatorApp, text: str) -> None:  # noqa: ANN401
    """Type a slash command into the real editor and press Enter.

    The reported path rather than ``_run_slash_command``: the submit handler is
    what decides a line beginning with `/` is a command at all, and a rename
    that only works when its handler is called by hand is not one a user has.

    Esc first when the picker is open — Enter on an open command picker
    COMPLETES the highlighted row and submits THAT, so a bare ``/rename`` would
    never reach the branch under test.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    """Notice bodies, unwrapped — the receipt a rename leaves behind."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_a_typed_rename_names_both_surfaces_and_asks_no_provider() -> None:
    """``/rename <title>`` is the entry point ``user_set`` exists for.

    Both surfaces, in the frame it was typed: the band and the terminal tab are
    what a name IS to the user, and a rename that needed a turn to take effect
    would read as a command that did nothing. No provider call either — the
    words are already the user's, so there is nothing to ask anyone.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        assert app._status is not None
        written: list[str] = []
        title = TerminalTitle(written.append)
        title.start()
        app._status.set_terminal_title(title)

        await _type_command(pilot, app, "/rename Ledger reconciliation")

        assert session.conversation_name == "Ledger reconciliation"
        assert session.conversation_name_state.user_set is True
        assert app._status._conversation_name == "Ledger reconciliation"
        assert "Ledger reconciliation" in title.current
        assert any("Ledger reconciliation" in chunk for chunk in written)
        assert session.completions == [], "the rename asked a provider for a title"
        assert session.prompts == [], "the rename was sent to the model as a turn"


@pytest.mark.asyncio
async def test_a_typed_rename_outranks_a_later_material_change() -> None:
    """The precedence gate, reached the way the product reaches it.

    ``test_a_human_rename_is_never_overwritten_by_a_retitle`` sets ``user_set``
    by hand; this drives the only writer of it the shipped TUI has, so the gate
    is exercised end to end — a genuine change of subject arrives afterwards and
    must neither repaint the name nor spend a call to be refused.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named_a_while_ago(app, session, "fix the login redirect loop")
        await _type_command(pilot, app, "/rename Ledger reconciliation")
        assert session.conversation_name == "Ledger reconciliation"

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation"
        assert app._status is not None
        assert app._status._conversation_name == "Ledger reconciliation"
        assert len(session.completions) == 1, "a renamed conversation spent a re-title call"


@pytest.mark.asyncio
async def test_a_rename_before_the_first_prompt_spends_no_naming_call() -> None:
    """A conversation the user has already named has nothing left to name.

    ``_maybe_name_conversation`` reads the stored name, so the opener takes the
    re-title path and is refused there by ``user_set`` — and the excerpt never
    goes up, because a stand-in for a name that exists would be a downgrade.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _type_command(pilot, app, "/rename Ledger reconciliation")

        app._submit_prompt("fix the login redirect loop")
        await asyncio.wait_for(session.prompt_started.wait(), timeout=5)
        await _settle()

        assert session.completions == [], "a named conversation asked for a title"
        assert session.conversation_name == "Ledger reconciliation"
        assert app._status is not None
        assert app._status._conversation_name == "Ledger reconciliation"
        assert app._provisional_name == "", "an opener excerpt displaced the user's name"
        session.gate.set()


@pytest.mark.asyncio
async def test_a_bare_rename_reports_the_title_and_changes_nothing() -> None:
    """The empty-argument form answers; it does not clear the name.

    ``ConversationName.set("")`` would store an empty string as a USER-SET
    title, permanently un-naming the conversation and locking out every
    generated one — so the bare form has to be a report and nothing else.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _type_command(pilot, app, "/rename")
        assert session.conversation_name == ""
        assert session.conversation_name_state.user_set is False
        assert any("unnamed" in text for text in _notices(app)), _notices(app)

        await _type_command(pilot, app, "/rename Ledger reconciliation")
        await _type_command(pilot, app, "/rename")

        assert session.conversation_name == "Ledger reconciliation"
        assert any("Ledger reconciliation" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_chatty_follow_up_makes_no_call_at_all() -> None:
    """ "thanks" cannot have moved a subject, and the filter is free."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named_a_while_ago(app, session, "fix the login redirect loop")

        for chatter in ("thanks", "looks good", "ok"):
            app._submit_prompt(chatter)
        await _settle()

        assert len(session.completions) == 1, "a pleasantry spent a provider call"
        assert session.conversation_name == "Fix the login flow"


@pytest.mark.asyncio
async def test_a_burst_of_substantive_follow_ups_costs_exactly_one_check() -> None:
    """A burst on one subject costs exactly one check.

    Two gates cooperate to make this so. The FIRST follow-up clears the growth
    gate (the transcript grew to six turns) and the secondary time floor, so it
    spends a check; the model answers "no change", which charges one refresh and
    advances ``_last_titled_turn_count`` to six. The next two follow-ups arrive
    against an unchanged transcript, so the growth gate now demands sixteen
    turns and declines them without a call. Unthrottled this was one provider
    call per message.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named_a_while_ago(app, session, "fix the login redirect loop")

        session.title = "<title/>"
        for follow_up in (
            "also handle the oauth callback path",
            "and the logout redirect too",
            "check the session cookie flags while you are there",
        ):
            app._submit_prompt(follow_up)
            await _settle()

        assert len(session.completions) == 2, "the burst was not throttled to one check"


@pytest.mark.asyncio
async def test_the_time_floor_defends_against_a_same_breath_burst() -> None:
    """The SECONDARY churn guard, isolated with the growth gate held open.

    With the transcript already past the growth threshold, growth alone would
    admit a re-title the instant a message arrives. The time floor is what stops
    two checks landing within seconds of each other: the first title stamps
    ``_retitle_checked_at`` at the store, so a message one second later is
    declined, and only once the clock crosses ``RETITLE_MIN_GAP_S`` does the
    model get asked. A tab label that changes twice in one breath is unreadable.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        now = await _named(app, session, "fix the login redirect loop")
        # Growth gate held OPEN so the only guard left is the time floor.
        session.grow_transcript(6)

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()
        assert len(session.completions) == 1, "a one-second-old title was up for replacement"
        assert session.conversation_name == "Fix the login flow"

        # Past the floor the model does get asked, and its answer is taken.
        now[0] += RETITLE_MIN_GAP_S + 1
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()
        assert len(session.completions) == 2
        assert session.conversation_name == "Billing importer rewrite"


# -- the re-title call itself -------------------------------------------------


@pytest.mark.asyncio
async def test_generate_retitle_reads_the_sentinel_as_no_change() -> None:
    async def answer(system: str, prompt: str) -> str:
        return "<title/>"

    assert await naming.generate_retitle("Fix the login flow", "and the logout too", answer) is None


@pytest.mark.asyncio
async def test_generate_retitle_treats_a_restatement_as_no_change() -> None:
    """A model that "changes" the title to the one already in force has said
    "no change" in the expensive spelling; repainting it would be a flicker."""

    async def answer(system: str, prompt: str) -> str:
        return "<title>fix the LOGIN flow</title>"

    assert await naming.generate_retitle("Fix the login flow", "and the logout too", answer) is None


@pytest.mark.asyncio
async def test_generate_retitle_never_asks_without_a_title_or_about_chatter() -> None:
    calls: list[str] = []

    async def answer(system: str, prompt: str) -> str:
        calls.append(prompt)
        return "<title>Never reached</title>"

    assert await naming.generate_retitle("", "rewrite the billing importer", answer) is None
    assert await naming.generate_retitle("Fix the login flow", "thanks!", answer) is None
    assert calls == []


@pytest.mark.asyncio
async def test_generate_retitle_swallows_a_provider_failure_as_no_change() -> None:
    """``None`` means both "unchanged" and "the call died", because they are the
    same instruction to the caller: leave the title alone."""

    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("429 rate limited")

    assert await naming.generate_retitle("Fix the login flow", "rewrite the importer", boom) is None


# -- the theme sampler and the drift regression it fixes ----------------------


def _turn(role: str, text: str) -> SimpleNamespace:
    """A minimal history entry — the theme sampler only reads .role and .text."""
    return SimpleNamespace(role=role, text=text)


@pytest.mark.asyncio
async def test_an_in_goal_step_does_not_pivot_the_title() -> None:
    """THE regression: an in-goal step must not read as a new subject.

    The reported failure — a "build a web fetch tool" session renamed to "Find
    Port Credit restaurants" the moment the user exercised the tool — was caused
    by the model seeing ONLY the newest message. Here a model that titles from
    what it is shown keeps the theme, because the whole trajectory (opener plus
    tail, anchored on the current title) is what it now receives: the in-goal
    step is plainly a step inside the same body of work, not a pivot.
    """
    seen: list[str] = []

    async def titles_from_context(system: str, prompt: str) -> str:
        seen.append(prompt)
        # The opener is in the trajectory, so the theme is obvious: keep it.
        if "web fetch" in prompt.lower():
            return "<title/>"
        # Shown only the in-goal step, the same model would pivot — the bug.
        return "<title>Find Port Credit restaurants</title>"

    trajectory = [
        _turn("user", "help me build a web fetch tool for the agent"),
        _turn("assistant", "I'll add a fetch tool that GETs a URL."),
        _turn("user", "wire it into the tool registry"),
        _turn("assistant", "Done — registered and callable."),
    ]
    result = await naming.generate_retitle(
        "Build a web fetch tool",
        "now find me some good Port Credit restaurants to try it out",
        titles_from_context,
        turns=trajectory,
    )
    assert result is None, "an in-goal step pivoted the title"
    # And the reason it held: the model saw the opener, not just the newest line.
    assert "web fetch tool" in seen[0]


def test_build_theme_context_samples_head_and_tail_with_an_elision_marker() -> None:
    """The whole trajectory reaches the model: opener + tail, gap marked.

    A tail-only window is why the old design chased the newest message; the
    sampler keeps the opening turns (which state the subject) and a recent tail,
    and marks the turns dropped between them so two fragments never read as an
    abrupt switch.
    """
    turns = [_turn("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(12)]
    context = naming.build_theme_context(turns, "the newest thing", current_title="Old title")

    # The anchor leads the envelope.
    assert context.startswith("<chat>\n<current-title>\nOld title\n</current-title>")
    # The opening turns are present (the head states the subject)...
    assert "turn 0" in context and "turn 1" in context and "turn 2" in context
    # ...a middle turn is dropped...
    assert "turn 6" not in context
    # ...the gap is marked...
    assert "<elided/>" in context
    # ...and the newest message rides the tail.
    assert "the newest thing" in context.rsplit("<elided/>", 1)[-1]


def test_build_theme_context_appends_the_newest_message_as_the_tail() -> None:
    """The retitle fires at SUBMIT, so history lacks the newest message yet.

    The sampler appends it as a trailing user turn — without it the tail, where
    a genuine change of subject shows up, would be a message stale by one.
    """
    turns = [_turn("user", "opener"), _turn("assistant", "reply")]
    context = naming.build_theme_context(turns, "brand new subject", current_title="T")
    assert context.rstrip().endswith("brand new subject\n</user>\n</chat>")


def test_build_theme_context_ignores_non_conversational_entries() -> None:
    """Tool results and role-less custom entries are noise for a THEME."""
    turns = [
        _turn("user", "opener"),
        _turn("tool", "a large tool result"),
        SimpleNamespace(text="a custom entry with no role"),  # no .role
        _turn("assistant", "reply"),
    ]
    context = naming.build_theme_context(turns, "", current_title="T")
    assert "a large tool result" not in context
    assert "a custom entry with no role" not in context
    assert "opener" in context and "reply" in context


def test_build_theme_context_is_empty_when_there_is_nothing_to_title() -> None:
    """No turns and no newest message ⇒ no context ⇒ the caller spends no call."""
    assert naming.build_theme_context([], "", current_title="T") == ""


def test_should_refresh_theme_follows_the_geometric_schedule() -> None:
    """Titled at turn 1, then eligible at >=6, >=16, >=36, capped at five."""
    # Never titled (baseline 0): the gate reduces to the four-turn floor.
    assert not naming.should_refresh_theme(3, last_titled_turn_count=0, refresh_count=0)
    assert naming.should_refresh_theme(4, last_titled_turn_count=0, refresh_count=0)
    # Titled at turn 1: next eligibility at 1*2 + 4 = 6.
    assert not naming.should_refresh_theme(5, last_titled_turn_count=1, refresh_count=0)
    assert naming.should_refresh_theme(6, last_titled_turn_count=1, refresh_count=0)
    # Titled again at turn 6: next at 6*2 + 4 = 16.
    assert not naming.should_refresh_theme(15, last_titled_turn_count=6, refresh_count=1)
    assert naming.should_refresh_theme(16, last_titled_turn_count=6, refresh_count=1)
    # The cap is final regardless of how much the transcript has grown.
    assert not naming.should_refresh_theme(10_000, last_titled_turn_count=1, refresh_count=5)


# -- the opener-derived label itself -----------------------------------------


def test_a_short_opener_is_quoted_whole_and_sentence_cased() -> None:
    assert naming.provisional_title("fix the login redirect loop") == "Fix the login redirect loop"
    # The model's own casing survives, exactly as in `parse_title`: a stand-in
    # that rewrote "macOS" would be a worse label than the sentence it quotes.
    assert naming.provisional_title("macOS build fails") == "macOS build fails"


def test_a_long_opener_is_cut_on_a_word_boundary_with_an_ellipsis() -> None:
    """Truncation, not rejection — the one place this module differs.

    An over-long ANSWER from the model is evidence it ignored the format, so
    ``parse_title`` throws it away. An over-long opener is just a long request,
    and an excerpt of it is the point.
    """
    label = naming.provisional_title(
        "please go through the compaction module and explain how it picks a cut point"
    )
    assert label == "Please go through the compaction module and…"
    assert len(label) <= naming.MAX_PROVISIONAL_CHARS + 1  # the ellipsis rides outside the cap
    # Both caps bind: eight words is the ceiling, and the character cap cut this
    # one at seven.
    assert len(label.split()) <= naming.MAX_PROVISIONAL_WORDS
    assert (
        len(naming.provisional_title("one two three four five six seven eight nine ten").split())
        == naming.MAX_PROVISIONAL_WORDS
    )


def test_the_cut_does_not_leave_punctuation_stranded_before_the_ellipsis() -> None:
    """`endpoint,…` reads as a typo; `endpoint…` reads as a quote."""
    label = naming.provisional_title(
        "add pagination to the search results endpoint, then update the docs"
    )
    assert label == "Add pagination to the search results endpoint…"
    assert ",…" not in label


def test_an_opener_leading_with_a_url_or_path_is_not_sentence_cased() -> None:
    """`Https://example.com/…` reads as a rendering bug, not as a quotation."""
    assert naming.provisional_title("https://example.com/a/b fails to load").startswith("https://")
    assert naming.provisional_title("src/main.py raises on empty input").startswith("src/main.py")
    # A real word still gets its capital, apostrophes and hyphens included.
    assert naming.provisional_title("don't let the parser crash") == "Don't let the parser crash"


def test_one_enormous_word_is_cut_rather_than_dropped() -> None:
    """A pasted URL or stack frame still names the session.

    Returning "" here would put the cwd fallback back on the tab for exactly the
    paste-heavy openers this feature is aimed at.
    """
    label = naming.provisional_title("https://example.com/" + "x" * 200)
    assert label.endswith("…")
    assert len(label) == naming.MAX_PROVISIONAL_CHARS + 1


def test_low_signal_openers_get_no_label() -> None:
    """Both halves of the module answer "no title" the same way."""
    for opener in ("hi", "thanks!", "", "   ", "???"):
        assert naming.provisional_title(opener) == ""


def test_a_multi_line_opener_is_collapsed_onto_one_row() -> None:
    """The band is one row and the terminal title is one string."""
    label = naming.provisional_title("fix the parser\n\nit crashes on empty input")
    assert "\n" not in label
    assert label.startswith("Fix the parser it crashes")
