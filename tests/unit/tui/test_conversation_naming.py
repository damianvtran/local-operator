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
from typing import Any

import pytest

from local_operator.session import naming
from local_operator.tui.app import RETITLE_MIN_GAP_S, OperatorApp
from local_operator.tui.terminal_title import TerminalTitle
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class _GatedSession(FakeSession):
    """A session whose ``prompt`` blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.title = ""
        self.name_gate: asyncio.Event | None = None
        self.name_started = asyncio.Event()
        self.timeline: list[str] = []

    async def prompt(self, text: str, images: Any = None) -> None:  # noqa: ANN401
        self.prompts.append(text)
        self.timeline.append(f"prompt:{text}")
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


async def _settle() -> None:
    """Let app-thread workers run to a standstill.

    Twelve short yields rather than four: a turn worker, a naming worker and a
    re-title worker can now be scheduled in one submit, and on a loaded machine
    four ticks was enough to make the ORDER of those observable — this suite
    started failing only when run alongside its neighbours.
    """
    for _ in range(12):
        await asyncio.sleep(0.02)


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
        await pilot.pause()
        app._submit_prompt("fix the login redirect loop")
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
        await pilot.pause()
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
        await pilot.pause()
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
        await pilot.pause()
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
        await pilot.pause()
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
        await pilot.pause()
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
    app, session = await _boot(title="")  # a reply with no <title> at all
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
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
        await pilot.pause()
        app._submit_prompt("hi")
        session.gate.set()
        await _settle()
        assert session.completions == []
        assert app._provisional_name == ""
        assert app._status is not None
        assert app._status._conversation_name == ""


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
    await _settle()
    assert session.conversation_name, "the first title never landed"
    # A fresh gate, already open, so the follow-up turns below run through
    # instead of parking on the opener's spent one.
    session.gate = asyncio.Event()
    session.gate.set()
    return now


async def _named_a_while_ago(app: OperatorApp, session: _GatedSession, opener: str) -> list[float]:
    """:func:`_named`, then far enough past ``RETITLE_MIN_GAP_S`` to ask again."""
    now = await _named(app, session, opener)
    now[0] += RETITLE_MIN_GAP_S + 1
    return now


@pytest.mark.asyncio
async def test_a_material_change_of_subject_retitles_both_surfaces() -> None:
    """The whole point of the re-title: a drifted session names its subject.

    The model — not this code — decides that the subject moved; the fake stands
    in for it by answering with a replacement title instead of the sentinel.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
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
        assert session.completions[1][0].startswith("A conversation is titled: Fix the login flow")
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
        await pilot.pause()
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
        await pilot.pause()
        await _named_a_while_ago(app, session, "fix the login redirect loop")
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation"
        assert len(session.completions) == 1, "a renamed conversation spent a re-title call"


@pytest.mark.asyncio
async def test_a_chatty_follow_up_makes_no_call_at_all() -> None:
    """ "thanks" cannot have moved a subject, and the filter is free."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _named_a_while_ago(app, session, "fix the login redirect loop")

        for chatter in ("thanks", "looks good", "ok"):
            app._submit_prompt(chatter)
        await _settle()

        assert len(session.completions) == 1, "a pleasantry spent a provider call"
        assert session.conversation_name == "Fix the login flow"


@pytest.mark.asyncio
async def test_a_burst_of_substantive_follow_ups_costs_exactly_one_check() -> None:
    """The throttle. Unthrottled this is one provider call per message."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
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
async def test_a_landed_title_starts_the_re_title_window() -> None:
    """The window is measured from when the title took EFFECT.

    The first title comes from the naming path, which stamps nothing on its way
    out — only ``_store_title`` does, and only when a title actually lands.
    Without that stamp the window is still sitting at its initial zero when the
    conversation gets its name, so the very next message would be allowed to
    replace a title the user has had on screen for one second.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        now = await _named(app, session, "fix the login redirect loop")

        session.title = "<title>Billing importer rewrite</title>"
        app._submit_prompt("forget that, rewrite the billing importer instead")
        await _settle()
        assert len(session.completions) == 1, "a one-second-old title was up for replacement"
        assert session.conversation_name == "Fix the login flow"

        # Past the window the model does get asked, and its answer is taken.
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
