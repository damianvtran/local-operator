"""Auto-naming scheduling: when the title call fires, and what a failure costs.

The naming errand is decoration with one hard rule — it must never make the
session it decorates worse. Two regressions motivated this suite:

* Launched concurrently with the opening turn, it was a second simultaneous
  request on the same OAuth account at the account's busiest moment, and the
  concurrency ceilings made BOTH calls fail: the turn surfaced early
  provider-failure notices and the title died. The errand now waits for the
  turn to settle.
* A failed attempt used to spend the conversation's one naming request, so a
  minute-zero failure left the session (and the terminal title, which names
  the surface) on the cwd fallback forever. The latch now releases on
  failure and the next substantive message retries.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.session import naming
from local_operator.tui.app import OperatorApp
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
    """Let app-thread workers run; two pauses is what the suite's timing needs."""
    for _ in range(4):
        await asyncio.sleep(0.02)


async def _boot(title: str = "") -> tuple[OperatorApp, _GatedSession]:
    session = _GatedSession()
    session.title = title
    app = OperatorApp(lambda: _factory(session))
    # Held open by the caller's `async with app.run_test(...)`; returning both
    # handles from inside the context keeps every test one flat block.
    return app, session


@pytest.mark.asyncio
async def test_naming_waits_for_the_live_turn_to_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No second provider request, even beyond the old courtesy timeout."""
    # The old implementation fell through after 2 × TITLE_TIMEOUT_S and
    # called the provider while the turn was still live. Shrink that clock so
    # the regression is deterministic without making the test wait 40s.
    monkeypatch.setattr(naming, "TITLE_TIMEOUT_S", 0.01)
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._start_turn("fix the login flow")
        app._maybe_name_conversation("fix the login flow")
        await _settle()
        assert session.completions == [], "naming raced the live turn"

        session.gate.set()  # the turn settles
        await _settle()
        assert len(session.completions) == 1, "naming did not fire once the turn settled"


@pytest.mark.asyncio
async def test_follow_up_preempts_an_inflight_naming_call() -> None:
    """A follow-up provider call starts only after naming cancellation unwinds."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    session.name_gate = asyncio.Event()  # hold complete_once inside the provider lane
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._start_turn("opening turn")
        app._maybe_name_conversation("fix the login flow")
        await _settle()
        assert not session.name_started.is_set()

        session.gate.set()  # opening turn settles; naming takes the provider lane
        await asyncio.wait_for(session.name_started.wait(), timeout=1)

        app._start_turn("follow-up turn")
        # Mirrors `_submit_prompt`: latch ownership is transferred
        # synchronously, so this follow-up schedules the replacement naming
        # worker before the canceled one has finished unwinding.
        app._maybe_name_conversation("follow-up turn")
        await _settle()

        assert "name:cancel" in session.timeline
        assert session.timeline.index("name:cancel") < session.timeline.index(
            "prompt:follow-up turn"
        )
        starts = [index for index, item in enumerate(session.timeline) if item == "name:start"]
        assert len(starts) == 2
        assert session.timeline.index("prompt:follow-up turn") < starts[1]

        session.name_gate.set()
        await _settle()
        assert session.conversation_name == "Fix the login flow"


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

        # A second substantive message must NOT rename what the user is reading.
        session.title = "<title>A different title</title>"
        app._maybe_name_conversation("now something else entirely")
        await _settle()
        assert len(session.completions) == 1
        assert session.conversation_name == "Fix the login flow"


@pytest.mark.asyncio
async def test_the_band_is_named_from_the_opener_before_the_turn_settles() -> None:
    """THE regression. The title used to arrive only after the whole turn.

    Measured against a real provider on the pre-fix code: a 29.7-second opening
    turn during which the band and the terminal tab read `lo › <cwd>`, and a
    generated title stored 1.8 seconds after the answer was already on screen.
    Since a first turn on this product routinely runs for minutes, "the tab
    never becomes a title" was an accurate description of what a user saw.

    The errand still waits for the turn — it shares the provider lane, which is
    what the rest of this suite is about — so the fix is that the OPENER names
    the conversation immediately, for free, and the generated title upgrades it.
    This test therefore asserts the band while the turn is deliberately parked.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._submit_prompt("summarise how compaction picks a cut point in this repo")
        await _settle()

        # The turn is still live and the naming call has not been made.
        assert session.prompts == ["summarise how compaction picks a cut point in this repo"]
        assert not app._turn_settled.is_set()
        assert session.completions == []
        # ...and the band is already named, from the user's own words.
        assert app._status is not None
        assert app._status._conversation_name == "Summarise how compaction picks a cut point in…"
        # The STORE is untouched: every gate in the errand reads it to mean
        # "something already named this", so a stand-in written there would
        # cancel the call it is standing in for.
        assert session.conversation_name == ""

        session.gate.set()
        await _settle()
        # The generated title supersedes the excerpt on both the store and the band.
        assert session.conversation_name == "Fix the login flow"
        assert app._status._conversation_name == "Fix the login flow"
        assert app._provisional_name == ""


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
    """"hi" is not a title in either half of the module.

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
