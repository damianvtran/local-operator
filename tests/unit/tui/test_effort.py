"""Seeing and steering reasoning effort from the app: the band, the key, the command.

The report behind this was that there was "no way to see the thinking effort or
control/change that". Both halves are tested here against the REAL app — a
Pilot pressing real keys and submitting real lines through the editor — because
the two failure modes are both invisible to a unit test of the handler: a
keybinding another widget eats, and a band that agrees with the app's own
variable while disagreeing with the spec the request is built from.

Every assertion about "what is in force" therefore reads the SPEC the session
would send, not the app's remembered choice.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from textual.geometry import Size
from textual.widgets import Static

from local_operator.model.configure import build_model_spec
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import (
    FakeProviderController,
    FakeSession,
    _band,
    _factory,
    _FakeDef,
)


class EffortSession(FakeSession):
    """A session carrying a REAL ``ModelSpec``, so the ladder is the shipped one.

    ``FakeSession.model`` is ``None`` and its ``set_model`` a no-op, which would
    make every assertion here vacuous: the level would live only in the app's
    own variable and nothing would prove the request changed.
    """

    def __init__(self, provider: str = "anthropic", model_id: str = "claude-opus-5") -> None:
        super().__init__()
        self._spec = build_model_spec(provider, model_id)

    @property
    def model_label(self) -> str:
        return f"{self._spec.provider}/{self._spec.model_id}"

    @property
    def model(self) -> Any:
        return self._spec

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        self._spec = model


class EffortProviderController(FakeProviderController):
    """A controller that resolves a selector to a REAL, shipped ``ModelSpec``.

    ``FakeProviderController.resolve_model`` answers with a bare ``FakeModel``
    that carries no ladder at all, which would make every clamp assertion in
    this file vacuous: a level would be cleared because there was nothing to
    clamp INTO, not because the target lacks a knob. ``build_model_spec`` is
    the same offline derivation the real controller lands on.
    """

    def login_providers(self):
        return [
            _FakeDef("anthropic", "Anthropic", None, ("claude",)),
            _FakeDef("openai", "OpenAI", None, ("gpt",)),
        ]

    def provider(self, pid):
        for definition in self.login_providers():
            if definition.id == pid:
                return definition
        return None

    def has_any_credential(self, provider):
        return True

    def is_usable(self, provider):
        return True

    # ``FakeProviderController.resolve_model`` is annotated to return a bare
    # ``FakeModel``; a controller that answers with a REAL spec is the point of
    # this subclass, so the return is widened to ``Any`` rather than narrowed
    # to a type the base cannot promise.
    def resolve_model(self, provider, model_id) -> Any:
        return build_model_spec(provider.lower(), model_id)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type into the real editor and press Enter — the reported path.

    The command picker is dismissed first: Enter on an open list completes the
    highlighted row instead of submitting what was typed, so a test that skipped
    this would exercise the completion, not the command.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _rows(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


def _level(app: OperatorApp) -> str | None:
    """The level the session would SEND on its next request, or None if unset.

    Reads the SPEC, never the app's remembered choice: the two agreeing is the
    property most of these tests exist to check, so a helper that read the
    convenient one would make them vacuous.

    The assert is the precondition, not a type-checker appeasement: every test
    in this file is meaningless before the session has booted, and a failure
    here means `_boot` returned early rather than that effort is broken.
    """
    assert app._session is not None, "the session must be up before reading its spec"
    return app._session.model.reasoning_effort


@pytest.mark.asyncio
async def test_the_band_states_the_level_beside_the_model() -> None:
    """The report's first half: the effort was nowhere on screen. It is on the
    band from the first frame, next to the model, without anyone typing
    anything — `claude-opus-5` runs at Anthropic's documented `high`.

    The model's own TEXT is deliberately not asserted here: how an id becomes a
    label is a separate concern with its own tests, and pinning it in both
    places is how two owners end up editing each other's expectations. What
    matters to this segment is that the level is present and sits beside the
    model rather than adrift among the counters on the right.
    """
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await pilot.pause()
        band = _band(app)
    assert "▴ high" in band
    assert 0 < band.index("▴") < band.index("⌂")


@pytest.mark.asyncio
async def test_shift_tab_cycles_the_level_and_the_band_follows() -> None:
    """The keybinding, pressed for real.

    Textual's own ``Screen`` binds ``shift+tab`` to ``focus_previous``, so this
    is the test that the app's binding actually WINS the key rather than merely
    existing in the table — and that what moved is the spec the next request is
    built from, not just a label.
    """
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        before = _level(app)
        await pilot.press("shift+tab")
        await pilot.pause()
        after = _level(app)
        band = _band(app)
        focused = app.focused
    assert (before, after) == ("high", "xhigh")
    assert "xhigh" in band
    # The key was consumed here, not spent moving focus out of the composer.
    assert isinstance(focused, Editor)


@pytest.mark.asyncio
async def test_cycling_wraps_around_the_top_of_the_ladder() -> None:
    """Five presses on a five-rung ladder come back to where they started, so
    every level is reachable from every other one with the key alone."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        seen = [_level(app)]
        for _ in range(5):
            await pilot.press("shift+tab")
            await pilot.pause()
            seen.append(_level(app))
    assert seen == ["high", "xhigh", "max", "low", "medium", "high"]


@pytest.mark.asyncio
async def test_cycling_leaves_no_rows_in_the_transcript() -> None:
    """The band is the receipt. A key a user may press four times to get round
    the ladder must not write four rows into the reading record — the same test
    the slash-command echo policy applies."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for _ in range(4):
            await pilot.press("shift+tab")
            await pilot.pause()
        notices, rows = _notices(app), _rows(app)
    assert rows == []
    assert [n for n in notices if "effort" in n] == []


@pytest.mark.asyncio
async def test_effort_with_a_level_sets_it_and_says_what_changed() -> None:
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        level = _level(app)
        notices = _notices(app)
        band = _band(app)
    assert level == "low"
    assert "low" in band
    receipt = [n for n in notices if "reasoning effort" in n]
    assert receipt, notices
    # Names the old level and the new one, and how long the choice lasts: it is
    # session-scoped by design and nothing else on screen would say so.
    assert "high" in receipt[-1] and "low" in receipt[-1]
    assert "this session" in receipt[-1]


@pytest.mark.asyncio
async def test_effort_with_an_unknown_level_changes_nothing_and_lists_the_real_ones() -> None:
    """A rejection has to be actionable: the levels are per model, so "invalid"
    without the set leaves the user guessing at a five-word vocabulary."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort turbo")
        level = _level(app)
        notices = _notices(app)
    assert level == "high"  # untouched
    warning = [n for n in notices if "turbo" in n]
    assert warning, notices
    for name in ("low", "medium", "high", "xhigh", "max"):
        assert name in warning[-1]


@pytest.mark.asyncio
async def test_bare_effort_lists_the_ladder_and_marks_the_current_level() -> None:
    """It LISTS rather than opening a picker: the answer is five words that fit
    on the row it is printed on, and `shift+tab` is already the one-keystroke
    picker — so the listing names that key instead of competing with it."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort")
        listing = [n for n in _notices(app) if "reasoning effort" in n][-1]
        level = _level(app)
    assert level == "high"  # a listing changes nothing
    assert "●high" in listing
    for name in ("auto", "low", "medium", "xhigh", "max"):
        assert name in listing
    assert "shift+tab" in listing
    # The scope statement is this command's answer to `PERSIST_HINT`, which a
    # bare `/model` prints in the same slot: /model volunteers that a pick CAN
    # be persisted (`/model default`, or the settings page), so /effort has to
    # volunteer that a level cannot.
    assert "this session only" in listing


@pytest.mark.asyncio
async def test_effort_auto_returns_to_the_models_own_default() -> None:
    """The way back. Cycling can never reach "unset", so without this an
    explicit level would be permanent for the session once one was chosen."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/effort auto")
        level = _level(app)
        remembered = app._effort_choice
    assert level == "high"
    assert remembered is None


@pytest.mark.asyncio
async def test_a_non_reasoning_model_offers_nothing_and_says_so() -> None:
    """It must not silently accept a level it would ignore: the band shows no
    effort at all, the command refuses, and the key explains itself rather than
    doing nothing."""
    app = OperatorApp(lambda: _factory(EffortSession("openai", "gpt-4.1")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await pilot.pause()
        band = _band(app)
        await _submit(pilot, app, "/effort high")
        refusal = _notices(app)[-1]
        await pilot.press("shift+tab")
        await pilot.pause()
        key_answer = _notices(app)[-1]
        level = _level(app)
    # No effort segment at all — not "reasoning", not a level. Its ABSENCE is
    # what makes the segment informative when it is there.
    assert "▴" not in band
    assert "reasoning" not in band and "high" not in band
    assert "not adjustable" in refusal
    assert "not adjustable" in key_answer
    assert level is None


@pytest.mark.asyncio
async def test_the_chosen_level_survives_a_session_rebuild() -> None:
    """`/new`, `/reload` and `/resume` all build a fresh session whose spec
    carries the MODEL's default. A choice that silently reverted there would be
    a band asserting a level that is not in force one command later."""
    sessions = [EffortSession(), EffortSession()]
    app = OperatorApp(lambda: _factory(sessions.pop(0)))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        # In-process rebuild — ``/reload`` now re-execs the process.
        app._session_factory = lambda: _factory(sessions.pop(0))  # type: ignore[assignment]
        await app._reload_session(keep_context=True)
        for _ in range(40):
            await pilot.pause()
            if app._session is not None and not sessions:
                break
        level = _level(app)
        band = _band(app)
    assert level == "low"
    assert "low" in band


@pytest.mark.asyncio
async def test_a_level_the_next_model_cannot_take_is_forgotten_not_hidden(
    monkeypatch, tmp_path
) -> None:
    """Carrying `max` onto a model whose ladder stops at `high` would be a 400
    on the next turn; keeping it in the wings to reappear two switches later
    would be spookier than making the user re-pick."""
    # The status band includes cwd; a checkout named context-max must not be
    # mistaken for a remembered effort level.
    monkeypatch.chdir(tmp_path)
    sessions = [EffortSession(), EffortSession("openai", "gpt-4.1")]
    app = OperatorApp(lambda: _factory(sessions.pop(0)))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort max")
        app._session_factory = lambda: _factory(sessions.pop(0))  # type: ignore[assignment]
        await app._reload_session(keep_context=True)
        for _ in range(40):
            await pilot.pause()
            if app._session is not None and not sessions:
                break
        level = _level(app)
        remembered = app._effort_choice
        band = _band(app)
    assert level is None
    assert remembered is None
    assert "max" not in band


@pytest.mark.asyncio
async def test_setting_the_level_it_is_already_on_says_so_rather_than_drawing_an_arrow() -> None:
    """`high → high` reads as a change that did not happen. The no-op branch
    exists so the receipt never describes movement there was none of."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort high")
        answer = _notices(app)[-1]
        level = _level(app)
    assert level == "high"
    assert "already" in answer
    assert "→" not in answer


@pytest.mark.asyncio
async def test_a_model_with_a_ladder_and_no_level_reads_auto_on_both_surfaces() -> None:
    """OpenAI boots unset by design, so this is the first frame most OpenAI
    users see. It used to read `▴ reasoning` — a word on no ladder — while the
    listing marked nothing at all, leaving no surface answering "what is it
    running at".
    """
    app = OperatorApp(lambda: _factory(EffortSession("openai", "gpt-5.4")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await pilot.pause()
        band = _band(app)
        await _submit(pilot, app, "/effort")
        listing = [n for n in _notices(app) if "reasoning effort" in n][-1]
    assert "▴ auto" in band
    assert "●auto" in listing


@pytest.mark.asyncio
async def test_a_model_that_reasons_without_a_ladder_keeps_the_word_reasoning() -> None:
    """`deepseek-reasoner` reasons at a depth the API exposes no name for. The
    band and the command have to agree about that: the refusal used to say the
    model "has no reasoning-effort levels" one row under a band asserting
    `reasoning`."""
    app = OperatorApp(lambda: _factory(EffortSession("deepseek", "deepseek-reasoner")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await pilot.pause()
        band = _band(app)
        await _submit(pilot, app, "/effort")
        answer = _notices(app)[-1]
    assert "▴ reasoning" in band
    assert answer == "reasoning effort: not adjustable on deepseek/deepseek-reasoner"


@pytest.mark.asyncio
async def test_the_key_answers_once_per_model_not_once_per_press() -> None:
    """A user probing an unfamiliar key four times got four rows in the loudest
    ink — the exact transcript noise the key's silence exists to avoid."""
    app = OperatorApp(lambda: _factory(EffortSession("openai", "gpt-4.1")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for _ in range(4):
            await pilot.press("shift+tab")
            await pilot.pause()
        refusals = [n for n in _notices(app) if "not adjustable" in n]
    assert len(refusals) == 1, refusals


@pytest.mark.asyncio
async def test_cycling_falls_back_to_the_transcript_when_the_band_dropped_the_segment() -> None:
    """ "The band is the receipt" only holds while the band still HAS the
    segment. On the shipped drop ladder the effort rung is shed at ordinary
    widths once a cost figure is on the row — so at 60 columns the key was
    changing a billable setting and moving nothing on screen at all."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        assert app._status is not None  # the band exists from on_mount
        app._status.update(cost="$0.045", context_tokens=400_000)
        await pilot.pause()
        await pilot.press("shift+tab")
        await pilot.pause()
        band = _band(app)
        receipts = [n for n in _notices(app) if "reasoning effort" in n]
        level = _level(app)
    assert level == "xhigh"
    assert "xhigh" not in band  # the ladder shed the segment at this width
    assert receipts and "high" in receipts[-1] and "xhigh" in receipts[-1]


@pytest.mark.asyncio
async def test_cycling_is_inert_while_a_completion_list_is_open() -> None:
    """Shift+Tab is "previous suggestion" in every completion UI, so it is the
    natural probe with a list open — and cycling there pins a level onto the
    model the user is in the middle of choosing."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "/eff"
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        assert editor._picker.is_open()
        await pilot.press("shift+tab")
        await pilot.pause()
        level = _level(app)
    assert level == "high"


@pytest.mark.asyncio
async def test_the_choice_rides_a_model_switch_and_is_dropped_by_one_that_cannot_take_it() -> None:
    """Decision 7, on the `/model` path rather than the reload path — the pilot
    factory has no provider controller, so `_cmd_model` itself bails early and
    the carry rule has to be exercised where it lives."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        carried = app._spec_with_chosen_effort(build_model_spec("openai", "gpt-5.4"))
        dropped = app._spec_with_chosen_effort(build_model_spec("openai", "gpt-4.1"))
        remembered_after_drop = app._effort_choice
    assert carried.reasoning_effort == "low"
    assert dropped.reasoning_effort is None
    assert remembered_after_drop is None


@pytest.mark.asyncio
async def test_effort_auto_on_a_model_with_no_documented_default_sends_nothing() -> None:
    """The OpenAI half of `/effort auto`: there is no level to restore, so the
    key leaves the wire entirely and the receipt names the state the band shows
    — `auto` — rather than a level.

    `auto` replaced the longer `the provider's default` in the receipt (U2), for
    the cells the session scope and the durable pointer now take: at 80 columns a
    notice row holds 70 (design review D2), and origin + scope + pointer do not
    fit. What is left is the band's own word for this state, which is the half
    the user can see."""
    app = OperatorApp(lambda: _factory(EffortSession("openai", "gpt-5.4")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort high")
        await _submit(pilot, app, "/effort auto")
        level = _level(app)
        receipt = [n for n in _notices(app) if "reasoning effort" in n][-1]
        band = _band(app)
    assert level is None
    assert receipt == "reasoning effort: high → auto (this session) — /settings sets auto", receipt
    assert "▴ auto" in band


@pytest.mark.asyncio
async def test_effort_auto_says_the_withdrawal_is_for_this_session_only() -> None:
    """U2: `/effort auto` withdraws the level for THIS conversation, and the
    stored key still governs the next one — so the receipt has to say both, and
    name the one route that changes the standing default (`/settings`; a stored
    level survives a bare `/model default` by design, D6, so that command is not
    that route)."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort xhigh")
        await _submit(pilot, app, "/effort auto")
        receipt = [n for n in _notices(app) if "reasoning effort" in n][-1]
    assert receipt == "reasoning effort: xhigh → high (this session) — /settings sets auto", receipt
    assert len(receipt) <= 70, receipt


@pytest.mark.asyncio
async def test_the_effort_set_receipt_holds_one_row_at_eighty_columns() -> None:
    """D2: the pointer is worth keeping and the row still has to fit. A notice
    row holds `width - 10` cells, so at 80 columns the budget is 70 — the base
    receipt was 45 and the added pointer took it to 73, which wrapped the row and
    widowed the word `it`. Measured here as cells, because a wrapped row and a
    fitting one render identically in a transcript block list."""
    app = OperatorApp(lambda: _factory(EffortSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort high")
        await _submit(pilot, app, "/effort xhigh")
        receipt = [n for n in _notices(app) if "reasoning effort" in n][-1]
    assert (
        receipt == "reasoning effort: high → xhigh (this session); /model default keeps it"
    ), receipt
    assert len(receipt) <= 70, receipt


# ---------------------------------------------------------------------------
# The child's page: whose level is the band naming?
# ---------------------------------------------------------------------------


class _Dock:
    """The band widget a ``StatusLine`` paints into.

    Not a real ``Static``: constructing one needs a running app, and these
    three are the fast band tests that deliberately do not start a Pilot. Same
    stand-in ``tests/unit/tui/test_subagent_stats.py`` uses, for the same
    reason — the two members ``StatusLine`` touches, and nothing else.
    """

    def __init__(self, width: int = 120) -> None:
        self.size = Size(width, 1)
        self.content: Any = ""
        #: The ``layout`` flag of the last paint — see ``StatusLine.refresh``.
        self.layout: bool = True

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        """Mirrors ``Static.update`` parameter for parameter.

        Not ``**kwargs``: a double that swallows anything stops standing for
        the thing it is named after, which is how this one missed ``layout``.
        """
        self.content = content
        self.layout = layout


def _overlay_band() -> Any:
    """A band on the parent's model at `high`, ready for an overlay."""
    from local_operator.tui.widgets.status_line import StatusLine

    status = StatusLine(cast(Static, _Dock()))
    status.update(model_label="anthropic/claude-opus-5", effort="high", cwd="/tmp")
    return status


def test_a_child_on_another_model_is_not_given_the_parents_effort() -> None:
    """Effort is a property of the MODEL, which is the one thing the child's
    page exists because the child can change. The parent's level beside the
    child's model name is not merely stale: `gpt-4.1` has no ladder at all, so
    `high` there is a level that model cannot be running."""
    from local_operator.tui.widgets.status_line import SubagentBand

    status = _overlay_band()
    assert "high" in status.render_text(120).plain
    status.set_subagent(SubagentBand(model_label="openai/gpt-4.1"))
    child = status.render_text(120).plain
    # The model text itself belongs to another owner's tests; what this one
    # pins is that the parent's level did not follow its model out of the frame.
    assert "claude" not in child.lower()
    assert "high" not in child
    assert "▴" not in child


def test_a_child_on_the_parents_model_keeps_the_level_it_is_actually_running() -> None:
    """The normal path — a child built on the parent's spec inherits the level
    with it — so blanking unconditionally would hide a true reading on the
    common case. Same rule `_shown_model_name` already applies to the label."""
    from local_operator.tui.widgets.status_line import SubagentBand

    status = _overlay_band()
    status.set_subagent(SubagentBand(model_label="anthropic/claude-opus-5"))
    assert "high" in status.render_text(120).plain


def test_closing_the_page_reveals_the_parents_level_again() -> None:
    from local_operator.tui.widgets.status_line import SubagentBand

    status = _overlay_band()
    before = status.render_text(120).plain
    status.set_subagent(SubagentBand(model_label="openai/gpt-4.1"))
    status.set_subagent(None)
    assert status.render_text(120).plain == before


def test_a_child_on_another_model_shows_its_OWN_recorded_effort() -> None:
    """The tier is recorded at the child's launch (``AsyncJob.effort``) and is
    true of whatever model the child runs, so a different-model child now names
    its own level instead of the blank it used to show. `hi` here is the
    CHILD's tier, not the parent's `high`, so the overlay names the level the
    child is actually running rather than dropping the segment for safety."""
    from local_operator.tui.widgets.status_line import SubagentBand

    status = _overlay_band()
    status.set_subagent(SubagentBand(model_label="openai/gpt-4.1", effort="hi"))
    child = status.render_text(120).plain
    # The child's own tier, beside its own model — not the parent's `high`.
    assert "hi" in child
    assert "high" not in child
    assert "claude" not in child.lower()


def test_the_overlays_own_effort_wins_even_on_the_parents_model() -> None:
    """When the overlay carries a tier it is authoritative — it was recorded at
    launch — so it is used even for a child on the parent's model rather than
    reading the parent's live level, which can drift as the operator cycles
    ``/effort`` while the page is open."""
    from local_operator.tui.widgets.status_line import SubagentBand

    status = _overlay_band()  # parent is on `high`
    status.set_subagent(SubagentBand(model_label="anthropic/claude-opus-5", effort="lo"))
    shown = status.render_text(120).plain
    assert "lo" in shown
    # The parent's `high` must not leak through the overlay's own reading.
    assert "high" not in shown


@pytest.mark.asyncio
async def test_apply_frontend_state_preserves_auto_effort_label() -> None:
    """Frontend state sync must derive the effort label via ``_effort_label`` rather
    than a raw string read of ``reasoning_effort``. Models booting unset with a ladder
    (every OpenRouter reasoning model) carry ``reasoning_effort=None`` and must render
    ``auto`` on the status band rather than having the segment clobbered to empty string.
    """
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )

    session = EffortSession("openrouter", "google/gemini-3.8-flash")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot, app)
        assert app._status is not None
        assert app._status._effort == "auto"

        # Simulate the periodic or event-driven frontend state update
        state = FrontendSessionState(
            session_id="session-1",
            epoch="1",
            effective_model=FrontendModelSpec(**session.model.model_dump()),
        )
        app._apply_frontend_state(state)
        await pilot.pause()

        assert app._status._effort == "auto"
        assert "auto" in _band(app)


# ---------------------------------------------------------------------------
# `/model default` persists the level, and `/model saved` adopts it back
# ---------------------------------------------------------------------------


def _config_file(tmp_path) -> None:
    """A file on disk, so ConfigManager builds its own Config rather than
    mutating the module-level DEFAULT_CONFIG singleton other tests share."""
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
    )


def _written(tmp_path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((tmp_path / "config.yml").read_text())["values"]


@pytest.mark.asyncio
async def test_model_default_saves_the_level_alongside_the_model(tmp_path, monkeypatch) -> None:
    """`/model default` used to persist only the model pair, so the operator
    re-ran `/effort` every launch. The remembered level now rides into the
    birth default, written LAST so a mid-loop failure cannot leave the effort
    pointing at a model the pair never reached (nothing else has moved yet)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model default")
        notices = _notices(app)
    written = _written(tmp_path)
    assert written["model_name"] == "claude-opus-5", written
    assert written["model_effort"] == "low", written
    # The receipt names the third key, in the registry's own vocabulary
    # (`auto` for the empty value), so a user can see what was made durable.
    assert any("model_effort low (used by new sessions)" in n for n in notices), notices


@pytest.mark.asyncio
async def test_model_default_with_no_chosen_level_saves_no_opinion(tmp_path, monkeypatch) -> None:
    """The model's SEEDED default is not a deliberate choice (D6). Persisting
    `high` here — which `build_model_spec` merely seeds for Anthropic — would
    freeze an INFERENCE into every future launch and, on a later
    `/model default <other>`, silently deepen that model's reasoning."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/model default")
        notices = _notices(app)
    written = _written(tmp_path)
    assert written["model_effort"] == "", written
    assert any("model_effort auto (used by new sessions)" in n for n in notices), notices


@pytest.mark.asyncio
async def test_model_default_clamps_the_saved_rung_and_runs_it(tmp_path, monkeypatch) -> None:
    """D7: on the persist path the SAVED rung and the RUNNING rung must agree.

    `/effort xhigh` on `claude-opus-5`, then `/model default
    anthropic/claude-opus-4-6` — whose ladder stops at `high`, one rung below
    `xhigh` on both sides, so the tie goes down. Without the live clamp the key
    would say `xhigh` while the band read the model's own default, and the next
    launch would make that disagreement real.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort xhigh")
        await _submit(pilot, app, "/model default anthropic/claude-opus-4-6")
        level = _level(app)
        remembered = app._effort_choice
        band = _band(app)
    written = _written(tmp_path)
    assert written["model_name"] == "claude-opus-4-6", written
    assert written["model_effort"] == "high", written
    assert level == "high"
    assert remembered == "high"
    assert "high" in band


@pytest.mark.asyncio
async def test_model_default_on_a_model_with_no_ladder_clears_the_level(
    tmp_path, monkeypatch
) -> None:
    """A model that takes no effort knob has nothing to save: the key is cleared
    rather than left holding a level the route would silently drop, and the
    receipt says `auto`. No 400, and no band lying about a level in force."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model default openai/gpt-4.1")
        level = _level(app)
    written = _written(tmp_path)
    assert written["model_name"] == "gpt-4.1", written
    assert written["model_effort"] == "", written
    assert level is None


@pytest.mark.asyncio
async def test_model_saved_adopts_the_configured_effort(tmp_path, monkeypatch) -> None:
    """`/model saved` means "put me back on my configured baseline", and effort
    is part of that baseline now (D8). Restored CLAMPED, so a configured level
    the saved model cannot express lands on its nearest rung rather than
    vanishing."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
        "  model_effort: medium\n"
    )
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model saved")
        level = _level(app)
        remembered = app._effort_choice
    assert level == "medium"
    assert remembered == "medium"


@pytest.mark.asyncio
async def test_model_saved_clears_the_level_when_nothing_is_configured(
    tmp_path, monkeypatch
) -> None:
    """The key unset means "the model's own default", so adopting the baseline
    CLEARS a level chosen here — the opposite of leaving the pick in force, and
    the reason the override is tri-state rather than a plain value."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model saved")
        level = _level(app)
        remembered = app._effort_choice
    assert level == "high"  # the model's own documented default
    assert remembered is None


@pytest.mark.asyncio
async def test_model_saved_receipt_names_the_effort_it_adopts(tmp_path, monkeypatch) -> None:
    """U1 / design D7: `/model saved` on the model already in force moves the
    EFFORT dial, and its receipt used to be an `X → X` model line that said
    nothing about the level — while pointing at `/model default saves this for
    new sessions`, the command the user did NOT run.

    The configured baseline is the only thing that moved here, so the receipt
    names that move and drops the model line entirely."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
        "  model_effort: medium\n"
    )
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model saved")
        notices = _notices(app)
    assert any(
        "reasoning effort: low → medium (the configured default)" in n for n in notices
    ), notices
    assert not any(n.startswith("model: ") for n in notices), notices


@pytest.mark.asyncio
async def test_model_saved_with_no_configured_effort_names_the_model_default(
    tmp_path, monkeypatch
) -> None:
    """The sibling clause when the key is UNSET: adopting a baseline with no
    stored level clears the session's pick, so the receipt says the level in
    force is the model's own rather than claiming a configured default that does
    not exist."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _config_file(tmp_path)
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        await _submit(pilot, app, "/model saved")
        notices = _notices(app)
    assert any(
        "reasoning effort: low → high (the model's own default)" in n for n in notices
    ), notices
    assert not any(n.startswith("model: ") for n in notices), notices


@pytest.mark.asyncio
async def test_model_saved_that_is_refused_does_not_arm_the_next_switch(
    tmp_path, monkeypatch
) -> None:
    """B2, the exact repro: a saved default naming a provider that is no longer
    known makes `/model saved` refuse — and the armed effort override used to
    survive that refusal, landing on the next, unrelated `/model` switch. The
    user asked for `high`, got a refusal notice, then switched models for an
    unrelated reason and silently landed on `none`: reasoning OFF, with the band
    as the only signal."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: deadprovider\n  model_name: ghost\n"
        "  model_effort: none\n"
    )
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort high")
        await _submit(pilot, app, "/model saved")
        refused = _notices(app)
        armed_after_refusal = app._pending_effort_override
        # The unrelated switch the leak used to land on.
        await _submit(pilot, app, "/model openai/gpt-5.1")
        level = _level(app)
        remembered = app._effort_choice
        band = _band(app)
    assert any("unknown provider: deadprovider" in n for n in refused), refused
    assert armed_after_refusal == (False, None), armed_after_refusal
    # `gpt-5.1` seeds no level, so the correct answer is `auto` — not the `none`
    # the leaked override would have clamped onto its ladder.
    assert level is None, band
    assert remembered is None
    assert "▴ auto" in band


@pytest.mark.asyncio
async def test_a_write_only_default_leaves_the_session_level_alone(tmp_path, monkeypatch) -> None:
    """M1: the write-only `/model default` (the model already in force) switches
    NOTHING — it writes the birth default for new sessions. It used to
    `_effort_choice = saved_effort` anyway, fabricating a preference the live
    spec never carried, so the next, unrelated `/model <other>` resurrected a
    level the user had explicitly withdrawn with `/effort auto`."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
        "  model_effort: high\n"
    )
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort auto")
        await _submit(pilot, app, "/model default")
        choice_after_write = app._effort_choice
        written = _written(tmp_path)
        # The unrelated switch the resurrected choice used to land on.
        await _submit(pilot, app, "/model openai/gpt-5.1")
        level = _level(app)
    # The stored level is PRESERVED (D6 clause 2) — the write-only form still
    # writes the config default for new sessions.
    assert written["model_effort"] == "high", written
    # ...and the session is untouched: no remembered choice, so the switch below
    # carries nothing.
    assert choice_after_write is None
    assert level is None


@pytest.mark.asyncio
async def test_the_default_receipt_names_a_clamped_rung(tmp_path, monkeypatch) -> None:
    """U4: `/model default <p>/<id>` clamps the stored level into the target's
    ladder and SAVES the clamped value — the receipt has to say that a rung was
    dropped, because the user asked for `xhigh` and the file now holds `high`.

    D3's budget holds on both rows: the receipt is one row at 120 columns (110
    cells) and the clamp note is on its own row rather than inside it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: anthropic\n  model_name: claude-opus-5\n"
        "  model_effort: xhigh\n"
    )
    app = OperatorApp(
        lambda: _factory(EffortSession()), provider_controller=EffortProviderController()
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/model default anthropic/claude-opus-4-6")
        notices = _notices(app)
    saved = [n for n in notices if n.startswith("boot default: ")]
    assert len(saved) == 1, notices
    assert "model_effort high (used by new sessions)" in saved[0], saved[0]
    # The 110-cell budget is measured with the SHORT home-relative path the
    # receipt prints in practice (`~/config/config.yml`), so the test's absolute
    # temp path is normalised the same way before measuring.
    normalised = saved[0].replace(str(tmp_path / "config.yml"), "~/config/config.yml")
    # Measured WITHOUT the access suffix: that clause (` · anthropic logged in`)
    # rides only when there is an access note to give, and it pushed this row
    # over the 120-column budget before this PR too — D3's finding is about the
    # receipt proper, which is the form the designer measured.
    assert len(normalised.split(" · ")[0]) <= 110, normalised
    assert any(
        n == "model_effort high (xhigh clamps to this model's nearest rung)" for n in notices
    ), notices
