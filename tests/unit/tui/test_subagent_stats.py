"""Per-child stats: the panel's rows and the band the full-page view is under.

Three things are pinned here, and each of them is a thing that was WRONG
before this file existed:

* the status band described the PARENT while the reader was inside a child's
  page, and a child frequently runs a different model;
* a row said ``tool: bash done`` — the mechanism — where the main
  conversation's working line had already been taught to say the model's own
  intent;
* nothing on a row said what the child had spent or how full its context was.

The width ladder gets its own section, because "the numbers must not push the
intent off the row" is a requirement and not a preference: this panel is the
surface most often read at half a terminal.
"""

from __future__ import annotations

import time
from typing import Any, cast

import pytest
from rich.cells import cell_len
from textual.geometry import Size
from textual.widgets import Static

from local_operator.harness.types import Usage
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.status_line import StatusLine, SubagentBand
from local_operator.tui.widgets.subagent_panel import (
    ACTIVITY_FLOOR,
    LABEL_FLOOR,
    JobStats,
    SubagentPanel,
    SubagentRow,
    compose_row,
    job_stats,
    panel_layout,
    row_facts,
)
from tests.unit.tui.test_band_panels import FakeSession, _async_factory, _fake_jobs

#: A model the registry prices AND sizes, so the numbers under test are the
#: renderer's arithmetic rather than the catalogue's availability.
PARENT_MODEL = "anthropic/claude-sonnet-4"
#: Deliberately a DIFFERENT model with a different window and price. The whole
#: point of the band re-pointing is a child that is not on its parent's model,
#: so every frame here makes the two differ on purpose.
CHILD_MODEL = "anthropic/claude-opus-5"


def _window_of(model_label: str) -> int:
    """The registry's window for a model — resolved in the TEST, never in a row."""
    from local_operator.model.configure import resolve_model_info

    provider, _, model_id = model_label.partition("/")
    return int(resolve_model_info(provider, model_id).context_window or 0)


def _model_shown(label: str, *, short: bool = False) -> str:
    """How the band spells a model label — it renders a display name, not an id."""
    from local_operator.tui.widgets.status_line import format_model_label

    return format_model_label(label, short=short)


class Job:
    """An ``AsyncJob`` as the panel reads one — duck-typed, like the widget."""

    def __init__(
        self,
        job_id: str = "j1",
        label: str = "MR review agent round 2",
        status: str = "running",
        *,
        progress: str = "",
        age: float = 469.0,
        settled: bool = False,
        queued: bool = False,
        model_label: str | None = None,
        context_window: int | None = None,
        usage: Usage | None = None,
        result_text: str | None = None,
        error_text: str | None = None,
    ) -> None:
        self.id = job_id
        self.type = "task"
        self.status = status
        self.label = label
        self.start_time = time.time() - age
        self.settled_at = self.start_time + age if settled else None
        self.result_text = result_text
        self.error_text = error_text
        self.latest_details = {"progress": progress} if progress else None
        self.queued = queued
        self.trajectory: list[dict[str, Any]] | None = None
        self.prompt: str | None = None
        self.model_label = model_label
        # Recorded on the job at launch, exactly as the engine records it: the
        # panel must never resolve a model on the paint path (a memo miss is a
        # provider listing fetch), so a window the job does not carry is a
        # window the row does not show.
        self.context_window = (
            context_window
            if context_window is not None
            else _window_of(model_label or PARENT_MODEL)
        )
        self.usage = usage


def _plain(job: Job, width: int, *, parent: str = PARENT_MODEL, current: bool = False) -> str:
    """One row as the string a reader sees, laid out alone at ``width``."""
    facts = row_facts(job, fallback_id=job.id, current=current)
    stats = job_stats(job, default_model_label=parent)
    rung, column, clock = panel_layout([(facts, stats)], width)
    return compose_row(
        facts=facts,
        stats=stats,
        spinner_glyph="⣾",
        width=width,
        rung=rung,
        column=column,
        clock=clock,
    ).plain


def _column(jobs: list[Job], width: int) -> list[str]:
    """A whole panel's worth of rows, reduced together the way the panel does."""
    measured = [
        (
            row_facts(job, fallback_id=job.id, current=False),
            job_stats(job, default_model_label=PARENT_MODEL),
        )
        for job in jobs
    ]
    rung, column, clock = panel_layout(measured, width)
    return [
        compose_row(
            facts=f,
            stats=s,
            spinner_glyph="⣾",
            width=width,
            rung=rung,
            column=column,
            clock=clock,
        ).plain
        for f, s in measured
    ]


# -- what a row is made of ---------------------------------------------------
def test_a_running_row_carries_duration_context_cost_and_the_models_intent() -> None:
    """The four facts the owner asked for, on one row, from one child's job."""
    job = Job(
        progress="Auditing merged MRs.",
        usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
    )
    row = _plain(job, 120)
    assert "7m49s" in row  # duration, as the ledger already spells it
    assert "24.1%/200k" in row  # context, against the PARENT's window (inherited)
    assert "$0." in row  # cost, a real number rather than a dash
    # The intent, normalised the way a tool card normalises one: this row sits
    # in a frame full of lowercase micro-labels.
    assert "auditing merged MRs" in row
    assert "Auditing merged MRs." not in row


def test_a_row_says_what_the_child_is_doing_not_which_tool_is_moving() -> None:
    """The regression the intent field exists for, on the subagent surface.

    ``tool: bash done`` is the mechanism. The main conversation's working line
    was taught to stop saying that; a row four cells below it saying it again
    would be two vocabularies for one state.
    """
    assert "running 3 tools" in _plain(Job(progress="running 3 tools"), 120)
    assert "thinking" in _plain(Job(progress="thinking"), 120)
    assert "responding" in _plain(Job(progress="responding"), 120)


def test_a_child_on_its_own_model_is_priced_and_sized_by_THAT_model() -> None:
    """A ``model_spec`` override is exactly what these numbers exist to show."""
    usage = Usage(input_tokens=100_000, output_tokens=10_000, context_tokens=100_000)
    inherited = job_stats(Job(usage=usage), default_model_label=PARENT_MODEL)
    overridden = job_stats(
        Job(model_label=CHILD_MODEL, usage=usage), default_model_label=PARENT_MODEL
    )
    assert inherited.model_label == PARENT_MODEL
    assert overridden.model_label == CHILD_MODEL
    # Different window and different price, so neither number may be shared.
    assert inherited.context_window != overridden.context_window
    assert inherited.cost is not None and overridden.cost is not None
    assert inherited.cost != overridden.cost


def test_a_child_that_has_reported_nothing_shows_no_number_rather_than_a_zero() -> None:
    """ "Nobody has told us" and "this was free" are different facts.

    A confident ``$0.0000`` on a child whose provider has simply not reported
    yet is the lie this whole change was opened against.
    """
    stats = job_stats(Job(progress="thinking"), default_model_label=PARENT_MODEL)
    assert stats.cost is None
    assert stats.context_tokens == 0
    row = _plain(Job(progress="thinking"), 120)
    assert "$" not in row
    assert "%" not in row
    assert "thinking" in row


def test_a_settled_row_keeps_its_outcome_and_still_carries_its_numbers() -> None:
    """A finished child's spend is the number a reader came back for."""
    row = _plain(
        Job(
            status="completed",
            settled=True,
            age=134,
            result_text="root caused: the fixture leaks a global",
            usage=Usage(input_tokens=91_000, output_tokens=4_100, context_tokens=92_000),
        ),
        120,
    )
    assert "✓" in row and "2m14s" in row
    assert "%" in row and "$" in row
    assert "root caused: the fixture leaks a global" in row


def test_the_row_the_page_is_showing_keeps_only_what_the_page_does_not_say() -> None:
    """One fact, one surface, within the twenty-five rows a reader can see.

    The page's title already states the child's state and age three rows
    above this row, and its body states the outcome at length. What the title
    does NOT carry is the spend and the window occupancy, so those are what
    the row keeps — along with the name, which is the row's whole job in this
    mode. Settled with the page's owner rather than each surface trimming
    independently.
    """
    job = Job(
        # Named HERE rather than leaned on from the class default, because the
        # name is what this test is about: an assertion that reaches 170 lines
        # up for the label it names is how this one came to name a child the
        # fixture never built.
        label="schema migration",
        status="failed",
        settled=True,
        error_text="provider error: 429 rate limited",
        usage=Usage(input_tokens=3_100, output_tokens=200, context_tokens=3_100),
    )
    unmarked = _plain(job, 120)
    assert "429 rate limited" in unmarked and "✗" in unmarked and "7m49s" in unmarked
    marked = _plain(job, 120, current=True)
    assert "429 rate limited" not in marked  # the page's body says it, at length
    assert "✗" not in marked and "7m49s" not in marked  # the page's title says both
    assert "schema migration" in marked  # which subagent: the row's whole job
    assert "$" in marked and "%" in marked  # the two facts the title lacks


# -- the width ladder --------------------------------------------------------
def test_at_sixty_columns_the_intent_survives_and_the_money_is_what_goes() -> None:
    """The stated requirement: three more fields may not cost the sentence."""
    job = Job(
        progress="auditing merged MRs",
        usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
    )
    wide, narrow = _plain(job, 120), _plain(job, 60)
    assert "$" in wide and "24.1%/200k" in wide
    assert "$" not in narrow, narrow
    assert "auditing merged" in narrow, narrow
    assert cell_len(narrow) <= 60


def test_the_ladder_sheds_cost_then_shortens_context_then_drops_it() -> None:
    """The reduction ORDER, walked one width at a time.

    Cost before context is the status band's own order (context predicts the
    next compaction; spend does not move inside a second), and context
    shortens TWICE before it drops: the decimal goes first and the
    denominator only after that, because a bare `24%` beside a bare `31%`
    invites a comparison between two children on different windows.
    """
    job = Job(
        label="MR review agent round 2",
        progress="auditing merged MRs",
        usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
    )
    seen: list[tuple[bool, str]] = []
    # Stops at 54, which is where this label's budget (a third of the width)
    # reaches LABEL_FLOOR. Below that the label stops shrinking with the row
    # while the row keeps narrowing, so a shed segment can briefly become
    # affordable again — documented in `_RUNGS`, and below any width a dock is
    # read at.
    for width in range(120, 53, -1):
        row = _plain(job, width)
        has_cost = "$" in row
        if "24.1%/200k" in row:
            context = "full"
        elif "24%/200k" in row:
            context = "nodec"
        elif "24%" in row:
            context = "short"
        else:
            context = "none"
        if not seen or seen[-1] != (has_cost, context):
            seen.append((has_cost, context))
    assert seen == [
        (True, "full"),
        (False, "full"),
        (False, "nodec"),
        (False, "short"),
        (False, "none"),
    ], seen


def test_the_activity_is_never_traded_for_a_number() -> None:
    """No width shows a number while withholding the sentence beside it."""
    job = Job(
        progress="auditing merged MRs",
        usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
    )
    for width in range(120, 39, -1):
        row = _plain(job, width)
        has_number = "$" in row or "%" in row
        has_activity = "auditing" in row
        assert has_activity or not has_number, f"{width}: {row!r}"


def test_the_label_is_the_last_thing_to_yield_and_stops_at_its_floor() -> None:
    """Two rows truncated to the same prefix are two rows nobody can tell apart."""
    job = Job(label="MR review agent round 2", progress="auditing merged MRs")
    # A label may never take more than a third of the row, so it is already
    # squeezed at 60 — and still says which child it is.
    assert "MR review agent rou" in _plain(job, 60)
    assert "MR review agent round 2" in _plain(job, 120)  # whole when there is room
    narrow = _plain(job, 30)
    assert "MR review a" in narrow  # squeezed to the floor, not dropped
    assert cell_len(narrow) <= 30


def test_a_row_never_overruns_the_width_it_was_given() -> None:
    """Every rung, every state, measured — a row that overflows is a broken dock."""
    jobs = [
        Job(progress="auditing merged MRs", usage=Usage(input_tokens=48_000, output_tokens=9_000)),
        Job("j2", "a" * 90, progress="b" * 90),
        Job("j3", "x", status="failed", settled=True, error_text="y" * 90),
        Job("j4", "queued crawler", queued=True),
    ]
    for width in range(20, 141):
        for row in _column(jobs, width):
            assert cell_len(row) <= width, f"{width}: {row!r}"


def test_the_whole_column_sheds_together_so_a_blank_cell_means_one_thing() -> None:
    """Rows are read as a column; per-row fitting punched holes in it.

    With each row fitting itself, a short-labelled child kept its dollar figure
    at a width where its neighbour had dropped one — so a blank cost meant
    either "unpriced" or "did not fit", with nothing on screen to say which.
    """
    usage = Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200)
    short = Job("j1", "docs", progress="auditing merged MRs", usage=usage)
    long = Job("j2", "MR review agent round 2", progress="auditing merged MRs", usage=usage)
    rows = _column([short, long], 60)
    assert ("$" in rows[0]) == ("$" in rows[1]), rows
    # ...and the long row alone is what forced it: on its own the short one
    # could have afforded more.
    assert "$" in _plain(short, 60)


def test_the_floors_are_what_the_ladder_stops_at() -> None:
    """The two constants are load-bearing, so they are asserted, not assumed."""
    assert ACTIVITY_FLOOR >= len("running 3 tools")
    assert LABEL_FLOOR >= len("MR review a")


# -- the band under the page -------------------------------------------------
class _Dock:
    """The band widget a ``StatusLine`` paints into, small enough to assert on.

    Not a real ``Static``: constructing one needs a running app, and these
    band tests are the fast ones that deliberately do not start a Pilot. The
    two members ``StatusLine`` actually touches are implemented here and the
    cast at the construction site is where that claim is stated — a mounted
    ``StatusLine`` is exercised through the app in the wiring tests below.
    """

    def __init__(self, width: int = 120) -> None:
        self.size = Size(width, 1)
        self.content: Any = ""
        #: The ``layout`` flag of the last paint. Recorded because the band
        #: asking for a layout pass is a real regression — see
        #: ``StatusLine.refresh`` — and a stub that dropped the argument could
        #: not say so.
        self.layout: bool = True

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        """Mirrors ``Static.update``, parameter for parameter.

        Deliberately NOT ``**kwargs``: a double that accepts anything is how
        this one drifted from the real signature in the first place, and the
        drift only surfaced when production started passing ``layout``. Spelled
        out, a future argument fails here loudly instead of being swallowed.
        """
        self.content = content
        self.layout = layout


def _band(width: int = 120) -> StatusLine:
    status = StatusLine(cast(Static, _Dock(width)))
    status.update(
        model_label=PARENT_MODEL,
        context_tokens=120_000,
        context_window=200_000,
        cost="$1.20",
        cwd="/tmp",
    )
    return status


def test_the_band_describes_the_child_while_its_page_is_open() -> None:
    """Model, context and cost all move to the child, and all four differ."""
    status = _band()
    parent = status.render_text(120).plain
    assert _model_shown(PARENT_MODEL) in parent
    assert "60.0%/200k" in parent and "$1.20" in parent

    status.set_subagent(
        SubagentBand(
            model_label=CHILD_MODEL,
            context_tokens=482_000,
            context_window=1_000_000,
            cost="$0.47",
            duration=469.0,
        )
    )
    child = status.render_text(120).plain
    assert _model_shown(CHILD_MODEL) in child, child
    assert "48.2%/1M" in child, child
    assert "$0.47" in child, child
    assert "7m49s" in child, child
    # Not one of the parent's numbers survives into the child's frame.
    assert _model_shown(PARENT_MODEL) not in child
    assert "60.0%/200k" not in child
    assert "$1.20" not in child


def test_leaving_the_page_restores_the_parents_band_exactly() -> None:
    status = _band()
    before = status.render_text(120).plain
    status.set_subagent(SubagentBand(model_label=CHILD_MODEL, cost="$0.47"))
    assert status.render_text(120).plain != before
    status.set_subagent(None)
    assert status.render_text(120).plain == before


def test_the_parents_numbers_keep_moving_underneath_and_are_revealed_current() -> None:
    """Why this is an overlay and not a save-and-restore.

    The parent does not stop while the page is open: its turns keep ending and
    each one writes a fresh cost. A snapshot taken on the way in would hand
    back a number that went stale in the reader's hand.
    """
    status = _band()
    status.set_subagent(SubagentBand(model_label=CHILD_MODEL, cost="$0.47"))
    status.update(cost="$9.99", context_tokens=180_000)
    assert "$9.99" not in status.render_text(120).plain
    status.set_subagent(None)
    revealed = status.render_text(120).plain
    assert "$9.99" in revealed
    assert "90.0%/200k" in revealed


def test_a_fact_the_child_lacks_is_omitted_rather_than_borrowed_from_the_parent() -> None:
    """A missing number is a smaller lie than somebody else's number."""
    status = _band()
    status.set_subagent(SubagentBand(model_label=CHILD_MODEL))
    child = status.render_text(120).plain
    assert _model_shown(CHILD_MODEL) in child
    assert "%" not in child, child  # no context reading, so no context segment
    assert "$" not in child, child  # no spend recorded, so no cost segment


def test_the_child_band_survives_a_narrow_terminal() -> None:
    """The overlay rides the band's own ladder, so it sheds the same way."""
    status = _band(60)
    status.set_subagent(
        SubagentBand(
            model_label=CHILD_MODEL,
            context_tokens=482_000,
            context_window=1_000_000,
            cost="$0.47",
            duration=469.0,
        )
    )
    narrow = status.render_text(60).plain
    assert cell_len(narrow) <= 60, narrow
    # Sheds no earlier than the parent's model would; the one rung where it DOES
    # yield is the irreducible row under an overlay, which is pinned below.
    assert _model_shown(CHILD_MODEL, short=True) in narrow, narrow


def test_no_width_paints_a_childs_model_with_nothing_saying_whose_it_is() -> None:
    """D9, as an invariant over the whole ladder rather than at one width.

    The irreducible row used to emit the CHILD's model under the PARENT's
    session with nothing naming the child — `◆ Gemini 2.5 Pro Preview` while
    the session itself was on Opus 5, a model on screen attributed to nobody.
    So the NAME is the segment that never drops and the model is what yields
    to it, which inverts the ladder's order everywhere else and is why it is
    written down on :attr:`SubagentBand.model_label`.

    Stated over every width rather than at the one that happened to break,
    because the ladder is re-tuned often and the misattribution is silent:
    the frame stays plausible, it just credits the wrong session.
    """
    from local_operator.tui.widgets.status_line import ICON_AGENTS, ICON_MODEL

    name = "IngestAuditor"

    def frame_at(width: int) -> str:
        status = _band(width)
        status.set_subagent(
            SubagentBand(
                model_label=CHILD_MODEL,
                label=name,
                context_tokens=48_200,
                context_window=200_000,
                cost="$0.465",
                duration=485.0,
            )
        )
        return status.render_text(width).plain

    for width in range(120, 19, -1):
        frame = frame_at(width)
        assert cell_len(frame) <= width, (width, frame)
        # The owner rides every rung: whatever else the band has given up by
        # here, a reader can still tell whose numbers these are.
        assert ICON_AGENTS in frame and name in frame, (width, frame)
    # And at the narrow end the MODEL is what went — not the row overflowing,
    # and not the name going with it.
    assert ICON_MODEL not in frame_at(20), frame_at(20)


# -- the two ends wired together ---------------------------------------------
@pytest.mark.asyncio
async def test_opening_and_leaving_a_page_repoints_and_restores_the_live_band() -> None:
    """End to end through the app: open the page, read the band, leave."""
    job = Job(
        model_label=CHILD_MODEL,
        progress="auditing merged MRs",
        usage=Usage(input_tokens=200_000, output_tokens=20_000, context_tokens=482_000),
    )
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        # The session boots in a worker, and the band re-points off the LIVE
        # session's ledger — asserting before it lands reads an app that has
        # no jobs to point at yet.
        for _ in range(50):
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is not None
        assert app._status is not None
        app._status.update(model_label="test/model", cost="$1.20")
        before = app._status.render_text(120).plain
        assert "test/model" in before

        app._open_subagent_view(job.id)
        # The band reads the panel's reading rather than deriving one, and that
        # reading is taken off-thread — a repaint may not price a child. Pump
        # until it lands, which is what a user experiences as the numbers
        # appearing a frame or two after the page does.
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        for _ in range(60):
            await pilot.pause()
            if panel.stats_for(job.id).cost is not None:
                break
        app._refresh_subagent_view()
        child = app._status.render_text(120).plain
        assert _model_shown(CHILD_MODEL) in child, child
        assert "test/model" not in child, child
        assert "48.2%/1M" in child, child
        assert "$" in child, child

        app._close_subagent_view()
        await pilot.pause()
        assert app._status.render_text(120).plain == before


@pytest.mark.asyncio
async def test_a_relayed_event_costs_no_repaint_until_the_next_tick() -> None:
    """The coalescing, asserted rather than described.

    A child emits an event per tool start, per tool end and per message
    boundary; painting every row on each of them is work quadratic in the
    fan-out, on the thread that reads the keyboard.
    """
    jobs = [
        Job("j1", "MR review agent round 2", progress="auditing merged MRs"),
        Job("j2", "docs sweep", progress="running 3 tools"),
        Job("j3", "flaky test triage", progress="thinking"),
    ]
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._paint_all()
        await pilot.pause()

        painted = _watch(panel)

        for index in range(12):  # four relayed events per child
            jobs[index % 3].latest_details = {"progress": f"auditing merged MRs {index}"}
            panel.sync(session)
        assert painted == [], "a relayed event painted a row before the tick"

        panel._tick()
        assert sorted(set(painted)) == ["j1", "j2", "j3"]
        # ...and the burst collapsed into ONE repaint per row, not twelve.
        assert len(painted) == 3


@pytest.mark.asyncio
async def test_a_mounted_panel_lays_out_against_the_screen_not_its_own_width() -> None:
    """The band is content-sized, so the panel can be WIDER than the terminal.

    ``#band`` is ``width: auto`` on purpose (a full-width sibling defeated the
    input shell's centering) and the slot inside it is ``1fr``, so one long
    row grows the panel past the dock — measured 65 against a 58-cell dock.
    Handed that number the ladder keeps a rung the terminal cannot show, and
    the row loses the end of its sentence to a hard cut rather than shedding a
    figure: the ladder defeated by exactly the row it exists for.
    """
    jobs = [
        Job(
            "j1",
            "MR review agent round 2",
            progress="auditing merged MRs",
            usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
        ),
        Job(
            "j2",
            "docs sweep",
            progress="running 3 tools",
            usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
        ),
    ]

    # A model the registry actually prices and sizes: the panel reads the
    # PARENT's label off the session for every child that recorded none of
    # its own, and `test/model` would leave the rows with nothing to shed.
    class PricedSession(FakeSession):
        @property
        def model_label(self) -> str:
            return PARENT_MODEL

    session = PricedSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(60, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        await pilot.pause()
        panel._tick()  # the arrange has happened; this is the settling repaint
        await pilot.pause()

        screen = app.screen.size.width
        assert panel._row_width() <= screen
        rows = [str(getattr(panel._rows[job.id], "content", "")) for job in jobs]
        for row in rows:
            assert cell_len(row) <= screen, row
            assert "$" not in row, row  # the money is what went
            assert "%" in row, row  # the context stayed, shortened
        # Both rows shed the same field, so a blank cost means one thing.
        assert "auditing merged" in rows[0] and "running 3 tools" in rows[1]


@pytest.mark.asyncio
async def test_a_child_appearing_paints_at_once_rather_than_waiting_for_a_tick() -> None:
    """A row's arrival changes the dock's height; a delayed height is a jump."""
    session = FakeSession()
    session.jobs = _fake_jobs()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        session.jobs = _fake_jobs(Job("j1", "docs sweep", progress="auditing merged MRs"))
        panel.sync(session)
        await pilot.pause()
        row = panel._rows["j1"]
        assert "docs sweep" in str(getattr(row, "content", ""))


@pytest.mark.asyncio
async def test_an_open_page_does_not_reinstate_the_per_event_repaint() -> None:
    """The coalescing has to hold in the mode where a reader is watching.

    Every relayed child event reaches `_refresh_subagent_view`, which marks
    the panel's current row — and that used to repaint the whole list and
    re-resolve every child's numbers, immediately. So the one configuration
    where the dock is being watched closely was the one where the per-event,
    per-row repaint came back.
    """
    jobs = [
        Job("j1", "MR review agent round 2", progress="auditing merged MRs"),
        Job("j2", "docs sweep", progress="running 3 tools"),
        Job("j3", "flaky test triage", progress="thinking"),
    ]
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(50):
            await pilot.pause()
            if app._session is not None:
                break
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._tick()
        app._open_subagent_view("j1")
        await pilot.pause()
        panel._tick()

        painted = _watch(panel)
        for index in range(12):
            jobs[index % 3].latest_details = {"progress": f"auditing merged MRs {index}"}
            app._refresh_band()  # what on_subagent_progress does
        assert painted == [], "an open page repainted rows per relayed event"
        panel._tick()
        # All three, once each — the burst of twelve collapsed into one paint
        # per row. j1 is the current row and repaints too, because a RUNNING
        # child keeps its activity even while its page is open.
        assert sorted(painted) == ["j1", "j2", "j3"]


@pytest.mark.asyncio
async def test_leaving_the_page_gives_the_row_its_outcome_back_on_the_next_tick() -> None:
    """The current row suppresses a SETTLED outcome; Esc must restore it."""
    job = Job(
        "j1",
        "docs sweep",
        status="completed",
        settled=True,
        age=10,
        result_text="root caused: the fixture leaks a global",
    )
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._tick()
        row = panel._rows["j1"]
        assert "root caused" in str(getattr(row, "content", ""))
        panel.mark_current("j1")
        panel._tick()
        assert "root caused" not in str(getattr(row, "content", ""))
        panel.mark_current(None)  # Esc
        panel._tick()
        assert "root caused" in str(getattr(row, "content", ""))


@pytest.mark.asyncio
async def test_a_queued_child_starting_keeps_the_spinner_alive() -> None:
    """The tick that brings a row to life must not be the one that kills the timer."""
    job = Job("j1", "docs sweep", queued=True)
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._tick()
        panel._tick()
        assert panel._rows["j1"].running is False
        job.queued = False  # the manager promotes it
        panel.sync(session)
        panel._tick()
        assert panel._rows["j1"].running is True
        assert panel._spinner_timer is not None, "the spinner stopped under a live row"


@pytest.mark.asyncio
async def test_a_job_that_refuses_to_be_read_does_not_reach_the_timer() -> None:
    """`sync` runs on the 1 Hz poll and the Subagent* handlers.

    An exception there is an unhandled Textual message-handler exception —
    the whole app, for a status row — and `getattr`'s default only covers a
    MISSING attribute, not a property that raises.
    """

    class ReplayedJob:
        id = "j1"
        type = "task"
        label = "docs sweep"
        start_time = 0.0
        settled_at = None
        queued = False
        latest_details: Any = ["not", "a", "mapping"]
        model_label = None
        context_window = 0
        usage = None

        @property
        def status(self) -> str:
            raise RuntimeError("this ledger row is not an AsyncJob")

    facts = row_facts(ReplayedJob(), fallback_id="j1", current=False)
    assert facts.label == "j1"  # degraded, not raised

    session = FakeSession()
    session.jobs = _fake_jobs(ReplayedJob())
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)  # must not raise
        panel._tick()
        assert panel.display is True


def _watch(panel: SubagentPanel) -> list[str]:
    """Record every row repaint the panel performs from now on.

    The job id is captured as a STRING rather than by holding the row: the
    recorder replaces ``row.update``, so a closure over the row would be
    reading an attribute off a widget it has just monkeypatched.
    """
    painted: list[str] = []

    def watch(row: SubagentRow) -> None:
        job_id = row.job_id
        original = row.update

        def update(renderable: Any, *args: Any, **kwargs: Any) -> Any:
            painted.append(job_id)
            return original(renderable, *args, **kwargs)

        row.update = update  # type: ignore[method-assign]

    for subagent_row in panel._rows.values():
        watch(subagent_row)
    return painted


def test_a_job_the_panel_cannot_read_degrades_instead_of_crashing() -> None:
    """Observability may not be able to take the app down."""

    class Hostile:
        @property
        def usage(self) -> Any:
            raise RuntimeError("no")

        @property
        def model_label(self) -> Any:
            raise RuntimeError("no")

    stats = job_stats(Hostile(), default_model_label=PARENT_MODEL)
    assert stats == JobStats(model_label=PARENT_MODEL)


def test_the_stats_line_up_down_the_column() -> None:
    """Same fields is half a scannable column; same POSITIONS is the other half.

    With the label inline and variable-length, one child's cost sat at column
    63 and its neighbour's at 17, so comparing two children's spend meant
    reading across both rows hunting for a ``$``.
    """
    usage = Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200)
    rows = _column(
        [
            Job("j1", "MR review agent round 2", progress="auditing merged MRs", usage=usage),
            Job("j2", "docs", progress="running 3 tools", usage=usage),
            Job("j3", "fix", progress="thinking", usage=usage),
        ],
        120,
    )
    assert len({row.index("$") for row in rows}) == 1, rows
    assert len({row.index("%") for row in rows}) == 1, rows


def test_the_clock_column_holds_still_under_a_shorter_duration() -> None:
    """`22s` is two cells narrower than `7m49s`, and everything right of the
    clock inherited that. Durations right-align into one panel-wide column."""
    usage = Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200)
    rows = _column(
        [
            Job("j1", "alpha", progress="auditing merged MRs", age=469, usage=usage),
            Job(
                "j2",
                "beta",
                status="failed",
                settled=True,
                age=22,
                error_text="provider error: 429",
                usage=usage,
            ),
        ],
        120,
    )
    assert len({row.index(" · ") for row in rows}) == 1, rows


@pytest.mark.asyncio
async def test_pricing_a_child_never_blocks_the_paint(monkeypatch) -> None:
    """A repaint may do no I/O at any budget.

    `job_cost` resolves the child's model to price it, and for a model the
    shipped registry does not describe that is a 10 s provider listing plus a
    3 s aggregator catalogue — and a child on a `model_spec` override is the
    case these stats exist to surface, so the slow path is the interesting
    one. The reading is taken off-thread; the row paints without its numbers
    for a frame rather than the app pausing to earn them.
    """
    import local_operator.model.configure as cfg

    real = cfg.resolve_model_info

    def slow(provider: str, model_id: str):
        time.sleep(0.4)
        return real(provider, model_id)

    monkeypatch.setattr(cfg, "resolve_model_info", slow)

    job = Job(
        "j1",
        "docs sweep",
        progress="auditing merged MRs",
        model_label="someprovider/some-unlisted-model",
        usage=Usage(input_tokens=48_000, output_tokens=9_000, context_tokens=48_200),
    )
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        started = time.perf_counter()
        panel.sync(session)
        panel._tick()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.2, f"the paint path blocked for {elapsed * 1000:.0f} ms"
        assert "docs sweep" in str(getattr(panel._rows["j1"], "content", ""))


@pytest.mark.asyncio
async def test_a_swept_child_is_dropped_from_the_stats_cache() -> None:
    """The cache is pruned with the rows, not only when the ledger empties."""
    jobs = [Job(f"j{index}", f"child {index}", progress="thinking") for index in range(5)]
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._tick()
        for _ in range(20):
            await pilot.pause()
            if len(panel._stats) == 5:
                break
        assert len(panel._stats) == 5
        session.jobs = _fake_jobs(jobs[0])  # retention sweeps the rest
        panel.sync(session)
        assert len(panel._rows) == 1
        assert set(panel._stats) == {"j0"}



@pytest.mark.asyncio
async def test_a_spinning_row_repaints_without_a_layout_pass() -> None:
    """The sheet fixes a row at ``height: 1`` and the row is built no-wrap to
    the measured width, so its content can never move the box.

    Textual's ``Static.update`` reflows the WHOLE screen by default, and this
    runs 12.5 times a second per running child for as long as the child lives.
    Measured with three running rows behind a 161-block transcript: 5.2% of a
    core with the default against 2.1% without it.
    """
    session = FakeSession()
    session.jobs = _fake_jobs(Job("j1", "docs sweep", status="running"))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        panel.sync(session)
        panel._tick()
        await pilot.pause()

        row = panel._rows["j1"]
        seen: list[bool] = []
        original = row.update

        def update(content: Any, *, layout: bool = True) -> Any:
            seen.append(layout)
            return original(content, layout=layout)

        row.update = update  # type: ignore[method-assign]
        # Force a real repaint: the fingerprint guard would otherwise swallow
        # a tick that changed nothing, and then the test would assert on an
        # empty list and pass for the wrong reason.
        row._fingerprint = None
        panel._tick()

        assert seen == [False]