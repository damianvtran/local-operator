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

import pytest

from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import Usage
from local_operator.session.frontend_state import CostKnowledge, FrontendSessionState
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded
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
    """A session whose transcript carried the rows a real one would return."""

    def __init__(self, rows) -> None:
        super().__init__()
        self._rows = rows

    def restored_search_spend(self):
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
    app = OperatorApp(_async_factory(session))
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
async def test_frontend_snapshot_band_folds_in_search_spend() -> None:
    """The canonical path renders the STORE's figure, so it needs the fold too.

    Production sessions take this branch, and it paints ``cumulative_cost``
    rather than ``_spend_total()``: folding search spend into the accessor alone
    would leave every real session's band short while the tests above passed.
    """
    session = _Session("anthropic/opus")
    app = OperatorApp(_async_factory(session))
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
    app = OperatorApp(_async_factory(session))
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
        text = app.screen._report_text().plain

    assert "Search spend" in text
    assert "Total spend" in text and "$0.013+" in text
    assert "4 searches · 1 unpriced" in text
    assert " ├ tavily" in text and " └ deepseek" in text
    assert "$0.0080+" in text and "2 searches · 1 unpriced" in text
    assert "$0.0051" in text
    # The split, in prose: the band's headline covers both, this screen does not.
    assert "added to the model estimate" in flowed(text)


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
        text = app.screen._report_text().plain

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
        text = "\n".join(line.plain for line in app.screen._report_lines())

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
    rows = (
        {"provider": "deepseek", "usd": 0.0031, "basis": "token estimate"},
        {"provider": "tavily", "usd": None, "basis": ""},
        {"provider": "tavily", "usd": 0.0080, "basis": "tavily credits"},
    )
    session = _ResumedSession(rows)
    app = OperatorApp(_async_factory(session))
    app._session = session

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
    app = OperatorApp(_async_factory(FakeSession()))
    app._session = FakeSession()
    app._restore_search_spend(app._session)
    assert SEARCH_SPEND.session("sess").searches == 0
