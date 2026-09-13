"""Search spend reaches the band total and both diagnostics panels.

Driven through the REAL app and the REAL screens. The ledger write, the band's
fold in ``_spend_total``/``_apply_frontend_state``, the ``/session`` capture and
the arguments ``/analytics`` hands its screen are the wiring; a test that called
the formatter directly would pass while the band stayed short and the panels
printed nothing.

The ledger is PROCESS-WIDE by design (see ``local_operator.web_search.cost``),
so the autouse fixture below is not hygiene: without it one test's searches are
the next test's band total.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import Usage
from local_operator.session.frontend_state import CostKnowledge, FrontendSessionState
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen
from local_operator.tui.widgets.session_panel import SessionScreen
from local_operator.web_search.cost import SEARCH_SPEND
from local_operator.web_search.models import SearchCost
from tests.unit.analytics.test_store import _snap
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_band_panels import _async_factory
from tests.unit.tui.test_cost_aggregation import (
    _band_cost,
    _resolving,
    _Session,
    _settle_boot,
)
from tests.unit.tui.test_slash_echo import _submit


@pytest.fixture(autouse=True)
def _isolated_ledger():
    SEARCH_SPEND.reset()
    yield
    SEARCH_SPEND.reset()


def _record(session: str = "sess", provider: str = "deepseek", usd: float | None = 0.0031) -> None:
    SEARCH_SPEND.record(
        session,
        provider,
        SearchCost(usd=usd, basis="token estimate", priced_from_usage=False),
    )


def flowed(text: str) -> str:
    """The report's text with wrapping collapsed, for asserting on PROSE.

    The panel wraps its footnotes at build time with an indent on every line
    (design D2), so a sentence spans several lines and a test about WORDING must
    not depend on where the break landed. Same helper as ``test_session_panel``.
    """
    return " ".join(text.split())


class _ResumedSession(FakeSession):
    """A session whose transcript carried the rows a real one would return.

    ``rows`` defaults to ``None`` so the subclass stays assignable wherever a
    ``FakeSession`` is expected: a required extra argument is a Liskov violation,
    which pyright (rightly) reports at every use site.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._rows = rows or []

    def restored_search_spend(self) -> list[dict[str, Any]]:
        return self._rows


# -- the band ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_band_total_folds_in_search_spend() -> None:
    """The reported shape of the bug: model money on the band, search money not.

    A tenth of a dollar of model spend against a third of a cent of search spend
    so the two are separable in the rendered figure -- at $10 the band's two
    decimals would hide the search half entirely and the assertion would pass on
    a band that never folded anything in.
    """
    session = _Session("anthropic/opus")
    app = OperatorApp(_async_factory(cast(Any, session)))
    async with app.run_test(size=(100, 28)) as pilot:
        await _settle_boot(pilot, app, session)
        with _resolving():
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=10_000))
            )
            await pilot.pause()
            assert _band_cost(app) == "$0.100"

            _record()
            # The SAME usage again, so the turn's own figure cannot be what moved
            # the cell: only the search half differs between the two reads.
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=10_000))
            )
            await pilot.pause()
            # The band must MOVE, not merely the accessor: this is the cell a user
            # reads, and a fold that never reached it would leave the test above
            # passing on a number nobody sees.
            assert _band_cost(app) == "$0.203"

    # Two turns of $0.10 each, plus the third of a cent of search spend that only
    # a folded-in ledger can account for.
    assert app._total_cost == pytest.approx(0.20)
    assert app._spend_total() == pytest.approx(0.2031)


@pytest.mark.asyncio
async def test_the_band_marks_a_figure_that_an_unpriced_search_makes_a_floor() -> None:
    """A combined total is a LOWER BOUND when one of its halves is unpriced.

    The band printed `◆ $0.104` unmarked while `/session` for the same session
    printed `$0.0040+` next to ``future-engine  $—  1 search · no published
    price``: one figure, two spellings, and the band's was the dishonest one.
    """
    session = _Session("anthropic/opus")
    app = OperatorApp(_async_factory(cast(Any, session)))
    async with app.run_test(size=(100, 28)) as pilot:
        await _settle_boot(pilot, app, session)
        with _resolving():
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=10_000))
            )
            await pilot.pause()
            assert _band_cost(app) == "$0.100"

            # A priced search keeps the figure exact...
            _record()
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=10_000))
            )
            await pilot.pause()
            assert _band_cost(app) == "$0.203"

            # ...and an unpriced one turns the same figure into a floor.
            _record(provider="future-engine", usd=None)
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=10_000))
            )
            await pilot.pause()
            assert _band_cost(app) == "≥$0.303"


@pytest.mark.asyncio
async def test_the_canonical_band_marks_a_search_only_figure_as_partial() -> None:
    """Unpriceable model money plus priced search money is HALF a session.

    The canonical branch falls back to the search figure when the store reports
    no model cost, and it did so unmarked -- presenting the retrieval half as the
    session total, which is the failure mode the money cell's own docstring calls
    the more expensive lie.
    """
    session = _Session("anthropic/opus")
    app = OperatorApp(_async_factory(cast(Any, session)))
    async with app.run_test(size=(100, 28)) as pilot:
        await _settle_boot(pilot, app, session)
        _record()
        state = FrontendSessionState(
            session_id="sess",
            epoch="e",
            cumulative_parent_cost=None,
            cost_knowledge=CostKnowledge.UNKNOWN,
        )
        app._apply_frontend_state(state)
        await pilot.pause()
    assert _band_cost(app) == "≥$0.0031"


@pytest.mark.asyncio
async def test_frontend_snapshot_band_folds_in_search_spend() -> None:
    """The canonical path renders the STORE's figure, so it needs the fold too.

    Production sessions take this branch, and it paints ``cumulative_cost``
    rather than ``_spend_total()``: folding search spend into the accessor alone
    would leave every real session's band short while the tests above passed.
    """
    session = _Session("anthropic/opus")
    app = OperatorApp(_async_factory(cast(Any, session)))
    async with app.run_test(size=(100, 28)) as pilot:
        await _settle_boot(pilot, app, session)
        _record()
        state = FrontendSessionState(
            session_id="sess",
            epoch="e",
            cumulative_parent_cost=0.10,
            cost_knowledge=CostKnowledge.EXACT,
        )
        app._apply_frontend_state(state)
        await pilot.pause()
    assert _band_cost(app) == "$0.103"


@pytest.mark.asyncio
async def test_a_turn_that_priced_nothing_still_shows_its_search_spend() -> None:
    """The gate that repaints the cell has to know search spend exists.

    A turn whose usage is unpriceable (or absent) reaches the band with no model
    figure at all, and the cell is only written when there is something to show.
    A search spent on that turn is real money, so the segment must appear rather
    than hold the previous turn's number -- or nothing.
    """
    session = _Session("test/mock")
    app = OperatorApp(_async_factory(cast(Any, session)))
    async with app.run_test(size=(100, 28)) as pilot:
        await _settle_boot(pilot, app, session)
        assert _band_cost(app) == ""
        _record()
        app.post_message(TurnEnded(False, None, context_tokens=0))
        await pilot.pause()
        assert _band_cost(app) == "$0.0031"


# -- the panels -------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_panel_prints_search_spend_and_each_provider(tmp_path, monkeypatch) -> None:
    """``/session`` shows the total, the count, and a row per provider."""
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    _record(provider="deepseek", usd=0.0031)
    SEARCH_SPEND.record("sess", "deepseek", SearchCost(usd=0.0020, basis="token estimate"))
    _record(provider="tavily", usd=0.0080)
    # An unpriced search must read as a COUNT, never as $0.
    SEARCH_SPEND.record("sess", "tavily", SearchCost(usd=None, basis=""))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    assert "Search spend" in text
    assert "Total spend" in text and "$0.013+" in text
    assert "4 searches · 1 unpriced" in text
    assert " ├ tavily" in text and " └ deepseek" in text
    assert "$0.0080+" in text and "2 searches · 1 unpriced" in text
    assert "$0.0051" in text
    # The footnote follows the HEADLINE rather than contradicting it. It used to
    # say this screen kept the two halves apart; the moment ``Est. cost`` above
    # started including the search money, that sentence described the opposite of
    # what the row showed -- a note explaining a figure it disagrees with.
    assert "includes this money and names it" in flowed(text)


@pytest.mark.asyncio
async def test_session_panel_omits_the_section_when_nothing_was_searched(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    assert "Search spend" not in text


@pytest.mark.asyncio
async def test_analytics_prints_process_search_spend_and_session_share(
    tmp_path, monkeypatch
) -> None:
    """``/analytics`` shows the process total, the providers, and this session's share."""
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    # This session: one priced search. Another session: three times as much, so
    # the share is a real fraction rather than the 100% a single-session run gives.
    _record(provider="deepseek", usd=0.0031)
    SEARCH_SPEND.record("other", "tavily", SearchCost(usd=0.0093, basis="tavily credits"))
    SEARCH_SPEND.record("other", "tavily", SearchCost(usd=None, basis=""))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics /usage")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_lines(app)

    assert "Search spend" in text
    assert "process-wide" in text
    assert "$0.012+" in text  # the process total: 0.0031 priced + 0.0093 priced
    assert "3 searches · 1 unpriced" in text
    assert " ├ tavily" in text and " └ deepseek" in text
    assert "This session" in text and "1 search · 25% of search spend" in text


# -- resume -----------------------------------------------------------------


def test_recovered_search_spend_is_seeded_once_and_never_doubled() -> None:
    """Adoption seeds the ledger; a second adoption of the same session must not.

    ``/reload`` adopts a session this process has already watched, so a replay
    that did not check first would double every recovered search.
    """
    rows: list[dict[str, Any]] = [
        {"provider": "deepseek", "usd": 0.0031, "basis": "token estimate"},
        {"provider": "tavily", "usd": None, "basis": ""},
        {"provider": "tavily", "usd": 0.0080, "basis": "tavily credits"},
    ]
    session = _ResumedSession(rows)
    app = OperatorApp(_async_factory(cast(Any, session)))
    app._session = session  # type: ignore[assignment] -- the app's facade slot is a protocol

    app._restore_search_spend(session)
    totals = SEARCH_SPEND.session("sess")
    assert totals.searches == 3
    assert totals.usd == pytest.approx(0.0111)
    assert totals.unpriced_searches == 1

    app._restore_search_spend(session)
    assert SEARCH_SPEND.session("sess").searches == 3
    assert app._spend_total() == pytest.approx(0.0111)


def test_a_host_without_a_transcript_seeds_nothing() -> None:
    """The reduced/attached hosts have no ``restored_search_spend``; say so quietly."""
    app = OperatorApp(_async_factory(cast(Any, FakeSession())))
    app._session = FakeSession()  # type: ignore[assignment] -- see above
    app._restore_search_spend(app._session)
    assert SEARCH_SPEND.session("sess").searches == 0


@pytest.mark.asyncio
async def test_a_read_row_is_labelled_as_a_read(tmp_path, monkeypatch) -> None:
    """A read is not a search, on screen as well as in the ledger.

    ``web_read`` records under ``<provider>:read`` precisely so its money lands
    in the total without inflating the search count -- and then the row rendered
    "1 search", which is the one thing the row exists to deny.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    _record()
    SEARCH_SPEND.record(
        "sess",
        "deepseek:read",
        SearchCost(usd=0.0020, basis="token estimate", priced_from_usage=True),
        kind="read",
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    # The total names the read apart from the searches...
    assert "1 search · 1 read" in text or "1 search · 1 unpriced" in text
    # ...and the row says read.
    assert " └ deepseek:read" in text
    assert "1 read" in text
    # The search count is untouched: a read is not counted as a search.
    assert "2 searches" not in text


@pytest.mark.asyncio
async def test_the_search_block_carries_the_cost_legend(tmp_path, monkeypatch) -> None:
    """A marked search figure owes the footnote that explains the mark.

    The legend predicate named only the model scopes, so `$0.015+` and `$—`
    appeared with nothing on screen to define them -- the failure the predicate's
    own comment calls out as the thing it exists to prevent.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    _record(provider="future-engine", usd=None)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    assert "$—" in text
    assert "+ lower bound" in text and "$— no published price" in text


@pytest.mark.asyncio
async def test_a_narrow_frame_keeps_the_search_count(tmp_path, monkeypatch) -> None:
    """Below the shortest rung the qualifier wraps; it is never dropped.

    At a 60-column terminal the headline row printed `Total spend  $0.015+` with
    no count at all and `└ future-engine  $—` with no words, because the ladder
    fell through to a silent crop.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    _record()
    SEARCH_SPEND.record("sess", "tavily", SearchCost(usd=0.0080, basis="per-search rate"))
    SEARCH_SPEND.record("sess", "future-engine", SearchCost(usd=None, basis=""))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    assert "Search spend" in text
    # The count survives, even where the row had to give up its qualifier.
    assert "3 searches" in text
    # The unpriced engine still says why, on its own line if need be.
    assert "unpriced" in text or "no published price" in text


def test_a_reads_known_price_is_rendered_not_hidden() -> None:
    """A priced read is priced, even though it is not a search.

    ``priced_searches`` counted only searches, so a read-only row -- or a
    session whose only spend was reads -- reported ``cost_is_known`` False and
    rendered ``$—`` for money the ledger had recorded exactly. Caught in a
    rendered frame, not by a test: the fixture's read had a price and the panel
    printed the unknown-price mark over it.
    """
    from local_operator.tui.costs import SearchSpendSnapshot

    SEARCH_SPEND.reset()
    SEARCH_SPEND.record(
        "sess",
        "deepseek:read",
        SearchCost(usd=0.002, basis="token estimate", priced_from_usage=True),
        kind="read",
    )
    snapshot = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))
    row = snapshot.rows[0]

    assert row.cost_is_known is True
    assert row.cost_usd == pytest.approx(0.002)
    assert snapshot.cost_is_known is True
    # ...and a read with an UNKNOWN price still reads as unknown, not free.
    SEARCH_SPEND.record("sess", "deepseek:read", None, kind="read")
    partial = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))
    assert partial.cost_is_partial is True
    SEARCH_SPEND.reset()


@pytest.mark.asyncio
async def test_a_read_only_conversation_still_gets_its_section(tmp_path, monkeypatch) -> None:
    """Read-only spend is spend: the section must not vanish while the band shows it.

    The guard was ``snapshot.searches``, so a conversation whose only retrieval
    money came from reads drew no Search spend heading at all on either panel --
    while the band kept the figure, leaving the user with a number and nowhere to
    read it.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record(
        "sess",
        "deepseek:read",
        SearchCost(usd=0.0020, basis="token estimate", priced_from_usage=True),
        kind="read",
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_text(app)

    assert "Search spend" in text
    assert "1 read" in text
    assert "0 searches" not in text


def _panel_lines(app: OperatorApp) -> str:
    """The ANALYTICS screen's rendered body lines, with the screen narrowed.

    ``_report_lines`` is the analytics screen's; the diagnostics screen renders
    through ``_report_text`` (see ``_panel_text``).
    """
    screen = app.screen
    assert isinstance(screen, AnalyticsScreen)
    return "\n".join(line.plain for line in screen._report_lines())


def _panel_text(app: OperatorApp) -> str:
    """The diagnostics screen's rendered body text, with the screen narrowed.

    ``app.screen`` is ``Screen[object]`` to the type checker, and the report
    methods live on the diagnostics screen; this is the same
    ``assert isinstance`` narrowing ``test_session_panel`` uses, in one place.
    """
    screen = app.screen
    assert isinstance(screen, SessionScreen)
    return screen._report_text().plain


@pytest.mark.asyncio
async def test_the_legend_is_printed_once_per_screen(tmp_path, monkeypatch) -> None:
    """One footnote, one print, however many vocabularies carry a mark.

    The same string explains the model rows' `+`/`$—` and the search block's, so
    a frame with an unpriced model call AND an unpriced search engine could show
    it twice with nothing telling the reader the two are the same legend.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record("sess", "future-engine", SearchCost(usd=None, basis=""))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_lines(app)

    # The legend LINE, exactly once -- the phrase itself also appears on the
    # unpriced total's note and its provider row, which is not the legend.
    assert text.count("+ lower bound (some calls unpriced)") == 1


@pytest.mark.asyncio
async def test_a_read_only_session_still_shows_its_share(tmp_path, monkeypatch) -> None:
    """The share row is about this session's spend, which a read has.

    The guard was ``session.searches``, so the one row that says how much of the
    process total belongs to the conversation you are looking at disappeared for
    a read-only conversation -- while its money was in the total above.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record(
        "sess",
        "deepseek:read",
        SearchCost(usd=0.0020, basis="token estimate", priced_from_usage=True),
        kind="read",
    )
    SEARCH_SPEND.record("other", "brave", SearchCost(usd=0.0080, basis="rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_lines(app)

    assert "This session" in text
    assert "1 read" in text
    # The share row's noun follows the kind: `0 searches · 100%` would be the
    # guard's fix producing a worse sentence than the bug it closed.
    assert "0 searches" not in text


@pytest.mark.asyncio
async def test_a_mixed_sessions_share_row_names_both_kinds(tmp_path, monkeypatch) -> None:
    """The share covers both kinds' money, so it names both counts.

    `5 searches · 100% of search spend` beside a share whose dollars include a
    read's reads as if the read were not part of the session's spend.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    _record()
    SEARCH_SPEND.record(
        "sess",
        "deepseek:read",
        SearchCost(usd=0.0020, basis="token estimate", priced_from_usage=True),
        kind="read",
    )
    SEARCH_SPEND.record("other", "brave", SearchCost(usd=0.0080, basis="rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _panel_lines(app)

    assert "This session" in text
    assert "1 search · 1 read" in text


# -- usage metrics: free vs paid, and search money in the headline ----------


def test_the_ledger_counts_free_and_paid_operations_exactly() -> None:
    """Counts and money per half, at the WRITE.

    The stored ``usd`` is a sum, so a provider that served one free and one paid
    call could not be split correctly at render time; the split has to be taken
    where each call's own price is still in hand. Six free searches and no
    searches at all both total $0.0000, which is why the counts exist.
    """
    from local_operator.tui.costs import SearchSpendSnapshot
    from local_operator.web_search.cost import SEARCH_SPEND
    from local_operator.web_search.models import SearchCost

    SEARCH_SPEND.reset()
    SEARCH_SPEND.record("sess", "duckduckgo", SearchCost(usd=0.0, basis="free"))
    SEARCH_SPEND.record("sess", "duckduckgo", SearchCost(usd=0.0, basis="free"))
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="per-search rate"))
    snapshot = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))

    assert snapshot.free_operations == 2
    assert snapshot.paid_operations == 1
    assert snapshot.free_usd == pytest.approx(0.0)
    assert snapshot.paid_usd == pytest.approx(0.004)
    SEARCH_SPEND.reset()


def test_combined_spend_is_one_rule_for_every_surface() -> None:
    """Model plus search, with the flags each surface has to render.

    The band folded search in while both panels' headline said model-only, so one
    session reported two different costs. This function is the single answer, and
    these are the four shapes it has to get right.
    """
    from local_operator.tui.costs import SearchSpendSnapshot, combined_spend
    from local_operator.web_search.cost import SEARCH_SPEND
    from local_operator.web_search.models import SearchCost

    SEARCH_SPEND.reset()
    empty = SearchSpendSnapshot()
    # A priced model with no searches: the combined figure IS the model figure.
    plain = combined_spend(1.20, empty)
    assert plain.total_usd == pytest.approx(1.20)
    assert (plain.is_unknown, plain.is_floor) == (False, False)

    # A model figure that is itself a floor stays a floor once search is added.
    floored = combined_spend(1.20, empty, model_is_partial=True)
    assert floored.is_floor is True

    # Real search money beside an UNPRICEABLE model: a floor, never "unknown".
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="rate"))
    search = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))
    mixed = combined_spend(None, search)
    assert mixed.total_usd == pytest.approx(0.004)
    assert (mixed.is_unknown, mixed.is_floor) == (False, True)

    # Nothing priceable at all: unknown, and not a floor (there is no known part
    # for it to be a floor of).
    nothing = combined_spend(None, empty)
    assert (nothing.is_unknown, nothing.is_floor) == (True, False)
    SEARCH_SPEND.reset()


@pytest.mark.asyncio
async def test_the_session_headline_includes_search_and_names_it(tmp_path, monkeypatch) -> None:
    """``Est. cost`` is the session's WHOLE money, with the search half named.

    The band already folded search in; this row did not, so the figure a person
    reads first was the one that omitted retrieval.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="per-search rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.screen._report_text().plain

    assert "Est. cost" in text
    assert "incl. $0.0040 search" in text
    # The usage half. This ledger is one paid search and no free ones, so the
    # line prints the paid half alone (D5: a zero for the other half is a
    # statement about nothing) -- and the money is the point at this width.
    assert "1 paid ($0.0040)" in text


@pytest.mark.asyncio
async def test_a_session_that_never_searched_claims_no_search_spend(tmp_path, monkeypatch) -> None:
    """No searches must not read as ``incl. $0.0000 search``.

    A zero there is a claim about retrieval rather than a report of none, and it
    would also imply the combined figure had a search component to name.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.screen._report_text().plain

    assert "Est. cost" in text
    assert "search" not in text.split("Est. cost")[1].split("\n")[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 100, 120])
async def test_the_headline_names_its_search_component_at_every_width(
    tmp_path, monkeypatch, width
) -> None:
    """MAJOR-2: the note was cropped mid-note, losing the fact it exists to state.

    At 80 columns the single wide note rendered ``≈ list price × tokens · incl``:
    the search component, on the row whose whole purpose is to name it.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="per-search rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.screen._report_text().plain

    assert "Est. cost" in text
    assert "$0.0040 search" in flowed(
        text
    ), f"the search component is not on screen at {width} columns"


def test_the_free_paid_line_reconciles_with_the_unpriced_tally() -> None:
    """MINOR-1: three numbers that look like a breakdown have to BE one.

    Without the unpriced clause, ``3 free · 2 paid`` sat under
    ``4 searches · 1 unpriced`` and the reader was left recomputing by hand.
    """
    from local_operator.tui.costs import SearchSpendSnapshot
    from local_operator.tui.widgets.analytics_panel import search_spend_section

    SEARCH_SPEND.reset()
    SEARCH_SPEND.record("sess", "duckduckgo", SearchCost(usd=0.0, basis="free"))
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="rate"))
    SEARCH_SPEND.record("sess", "serpapi", None)  # no published rate
    snapshot = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))
    text = "\n".join(
        line.plain if hasattr(line, "plain") else str(line)
        for line in search_spend_section(snapshot, 120, meta="this session · live")
    )

    # D5: "N of M priced free" states the denominator, and the unpriced clause
    # closes the gap to the count on the row above -- 1 free + 1 paid = 2 priced,
    # + 1 unpriced = the 3 searches the block's total row reports.
    assert "1 of 2 priced free" in text and "$0.0040 paid" in text
    assert "1 unpriced" in text
    assert (
        snapshot.free_operations + snapshot.paid_operations + snapshot.unpriced_searches
        == snapshot.count
    )
    SEARCH_SPEND.reset()


def test_an_all_search_figure_does_not_wear_a_token_priced_note() -> None:
    """MINOR-3: the note described a model half the figure does not contain."""
    from local_operator.tui.costs import (
        SearchSpendSnapshot,
        combined_spend,
        cost_note_rungs,
    )

    SEARCH_SPEND.reset()
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="rate"))
    snapshot = SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))
    spend = combined_spend(None, snapshot)

    rungs = cost_note_rungs(spend, search_component="incl. $0.0040 search")
    assert spend.model_usd is None
    assert all("list price × tokens" not in rung for rung in rungs)
    assert any("search only" in rung for rung in rungs)
    SEARCH_SPEND.reset()


@pytest.mark.asyncio
async def test_a_narrow_frame_drops_the_note_column_rather_than_cropping_it(
    tmp_path, monkeypatch
) -> None:
    """At 60 columns the whole note column goes, and that is the DESIGN.

    ``session_panel._NOTE_MIN`` drops the trailing note below 60 columns (the
    same rule that drops the ``$x subagent`` split), so the honest assertion here
    is the documented behaviour and not a component the frame cannot hold. Pinned
    so a later change to the ladder cannot quietly start cropping mid-note
    instead -- the failure MAJOR-2 was about.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.004, basis="per-search rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = app.screen._report_text().plain

    assert "Est. cost" in text
    # Nothing HALF-written: no fragment of the note survives into the frame.
    assert "incl" not in text.split("Est. cost")[1].split("\n")[0]
    assert "≈ list price × tokens" not in flowed(text)


# ---------------------------------------------------------------------------
# Round-2 findings: the label, the bar policy, unpriced-is-not-free, D9
# ---------------------------------------------------------------------------


def _ledger(rows):
    from local_operator.tui.costs import SearchSpendSnapshot

    SEARCH_SPEND.reset()
    for provider, cost, *rest in rows:
        SEARCH_SPEND.record("sess", provider, cost, kind=(rest[0] if rest else "search"))
    return SearchSpendSnapshot.of(SEARCH_SPEND.session("sess"))


def test_unpriced_searches_are_not_called_free() -> None:
    """R2-MAJOR-1: the headline branched on money, and ``usd`` is zero when unpriced.

    The ledger counts free/paid/unpriced AT THE WRITE precisely because a summed
    ``usd`` cannot tell them apart; inferring "free" from "no money" at render time
    put two contradicting claims on one screen.
    """
    from local_operator.tui.widgets.analytics_panel import search_component_text

    unpriced_only = _ledger([("future-engine", None), ("future-engine", None)])
    assert search_component_text(unpriced_only) == "incl. 2 unpriced searches"

    # Free AND unpriced: the unpriced part is what cannot be claimed, so it is
    # what the headline names.
    mixed = _ledger([("duckduckgo", SearchCost(usd=0.0, basis="free")), ("future-engine", None)])
    assert "free" not in search_component_text(mixed)

    assert search_component_text(_ledger([("duckduckgo", SearchCost(usd=0.0, basis="free"))])) == (
        "incl. 1 free search"
    )
    # R2-MINOR-2: a page read is not a search.
    assert search_component_text(
        _ledger([("deepseek:read", SearchCost(usd=0.0, basis="free"), "read")])
    ) == ("incl. 1 free read")


def test_the_bar_is_drawn_only_where_this_ledgers_note_still_fits() -> None:
    """R2-MAJOR-2: a fixed threshold drew the bar where it CAUSED the orphan.

    Whether the bar pays for itself depends on the note it shares the row with,
    so the decision is taken from this ledger's own widest note: at 81 cells the
    30-cell note fits beside it, at 70 it does not and there is no bar. Pin both
    ends, and that the count never lands on a line of its own.
    """
    from local_operator.tui.widgets.analytics_panel import search_spend_section

    snapshot = _ledger([("future-engine", None), ("future-engine", None)])

    def render(width: int) -> str:
        return "\n".join(
            line.plain if hasattr(line, "plain") else str(line)
            for line in search_spend_section(snapshot, width, meta="this session · live")
        )

    narrow, wide = render(70), render(90)
    assert "█" not in narrow and "bars:" not in narrow, "no bar, and nothing promising one"
    assert "2 searches · no published price" in narrow
    assert "█" in wide and "bars: operations" in wide
    # The count is never orphaned onto its own line, at either end.
    for text in (narrow, wide):
        assert not any(
            line.strip().startswith(("1 search", "2 searches")) and line.startswith("    ")
            for line in text.splitlines()
        )


@pytest.mark.asyncio
async def test_analytics_never_paints_a_cropped_money_fragment(tmp_path, monkeypatch) -> None:
    """R2-MAJOR-3 / D9: the shortest rung was drawn anyway, then cropped.

    At 60 columns that painted ``incl. $0.0`` -- a well-formed FALSE figure, two
    rows above a block showing the real total on the same frame.
    """
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: tmp_path / "l.db")
    store = AnalyticsStore(tmp_path / "l.db")
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.0069, basis="per-rate"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 44)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        rows = [line.plain for line in app.screen._report_lines() if "Est. cost" in line.plain]

    assert rows, "the headline row is on screen"
    row = rows[0]
    assert "Est. cost · search" in row
    # No fragment of any width: the row ends at the FIGURE, with nothing but
    # padding after it, so no half-token can be mistaken for money.
    assert "incl" not in row
    assert row.split()[-1].startswith("$"), f"something follows the figure: {row!r}"
