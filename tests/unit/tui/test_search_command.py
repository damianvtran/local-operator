from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text


@pytest.mark.asyncio
async def test_search_command_shows_status_and_applies_provider_toggle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search")
        await pilot.pause()
        first = _transcript_text(app)
        assert "Web search" in first
        assert "Round Robin" in first
        # One state word per row, with no readiness column beside it: a provider
        # that cannot serve is never in the chain, so the two columns always said
        # the same thing (round-1 D6).
        assert "DuckDuckGo" in first and "enabled ·" in first
        # The header names the CHAIN, in try order, and the exclusions -- the two
        # header facts the CLI had and this surface did not (round-1 D1/U6).
        assert "chain" in first and "→" in first
        assert "excluded" in first
        assert "/search enable|disable <provider>" in first
        assert "/search balance round_robin|ordered" in first
        assert "search setup tavily --oauth|--api-key" in first
        assert "search setup searxng --endpoint <url>" in first

        app._run_slash_command("/search setup tavily")
        app._run_slash_command("/search disable tavily")
        app._run_slash_command("/search")
        await pilot.pause()
        after = _transcript_text(app)
        assert "run in a shell: local-operator search setup tavily --oauth" in after
        assert "tavily excluded; it will not be used by any search until" in after
        assert "Tavily" in after and "excluded ·" in after

    assert session.prompts == []


@pytest.mark.asyncio
async def test_search_listing_prints_the_state_legend_and_the_ring_rule(
    monkeypatch, tmp_path
) -> None:
    """The three new labels are defined where they are printed (round-1 U5).

    `auto free`, `auto best-effort` and `auto paid` used to appear with nothing
    saying what they mean, and the ordering rule (`paid` is tried after the free
    legs, never before one) was stated only by `search enable`.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search")
        await pilot.pause()
        text = _transcript_text(app)

    # The legend defines the words this listing paints, with their meanings, and it
    # does not define words the install does not have (round-3 D3-3).
    assert "states" in text
    for state in ("enabled", "auto free", "auto best-effort", "needs setup"):
        assert state in text, state
    assert "enabled = in your priority order" in text
    assert "auto free = with the free providers" in text
    assert "auto paid" not in text  # no DeepSeek credential in this fixture
    assert "excluded = " not in text  # nothing is excluded in this fixture


@pytest.mark.asyncio
async def test_search_order_notice_is_word_for_word_the_cli_sentence(monkeypatch, tmp_path) -> None:
    """Round-1 U6: the TUI dropped "tried first", the one word that says prefix.

    The named list is a priority PREFIX, not the whole chain, and the CLI's
    sentence is the one both surfaces print.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search order duckduckgo tavily")
        await pilot.pause()
        text = _transcript_text(app)

    sentence = "search order: duckduckgo, tavily (tried first; any exclusion named here"
    assert sentence in text


@pytest.mark.asyncio
async def test_search_setup_names_the_route_for_every_provider_that_has_one(
    monkeypatch, tmp_path
) -> None:
    """Round-1 N1: `/search setup perplexity` fell through to the DuckDuckGo line.

    The setup tuple lost `perplexity` when `parallel` joined it, and `deepseek`
    -- the provider this change makes auto-join -- answered with "DuckDuckGo needs
    no setup" too.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search setup perplexity")
        app._run_slash_command("/search setup deepseek")
        app._run_slash_command("/search setup exa")
        await pilot.pause()
        text = _transcript_text(app)

    assert "local-operator search setup perplexity --api-key" in text
    assert "login deepseek" in text
    assert "exa works keyless (free MCP tier)" in text
    assert "DuckDuckGo needs no setup" not in text


def test_chain_markers_carry_the_shared_ink(tmp_path) -> None:
    """Round-2 D2-3: `(paid)` was plain dim on the chain row, amber two rows below.

    The TUI reads the provider module's marker table and paints each marker in the
    token that table names, so one fact cannot have two weights.
    """
    from rich.style import Style

    from local_operator.credentials import CredentialManager

    # A provider-class store ROW (PR2a): the plaintext file leg is gone, so the
    # row is what the resolver and the status rows read.
    from local_operator.providers.registry import store_provider_key
    from local_operator.tui import theme as theme_mod
    from local_operator.tui.app import _search_chain_text
    from local_operator.web_search.models import WebSearchSettings
    from local_operator.web_search.providers import (
        CHAIN_MARKER_TOKENS,
        chain_label,
        chain_leg_marker,
        provider_statuses,
    )

    credentials = CredentialManager.readonly(tmp_path / "config")
    store_provider_key("DEEPSEEK_API_KEY", "stored", base=tmp_path / "config")
    settings = WebSearchSettings(providers=["duckduckgo", "brave", "deepseek"])
    statuses = provider_statuses(settings, credentials)

    markers = {chain_leg_marker(status) for status in statuses if status.enabled}
    assert "" in markers and "(paid)" in markers and "(setup needed)" in markers
    # "" is the unmarked case (a free, ready leg), so only the markers must be covered.
    assert markers - {""} <= set(CHAIN_MARKER_TOKENS)
    assert CHAIN_MARKER_TOKENS["(paid)"] == "warning"

    # The TUI row is the CLI line with the markers in their own ink.
    row = _search_chain_text(statuses)
    assert row.plain == chain_label(statuses)
    paid = next(span for span in row.spans if "(paid)" in row.plain[span.start : span.end])
    # `Text.spans[].style` is `str | Style` in Rich's own typing, so narrow before
    # reading the colour out of it.
    assert isinstance(paid.style, Style)
    assert paid.style.color == Style(color=theme_mod.semantic_color("warning")).color


def test_every_state_word_has_its_own_ink() -> None:
    """Round-1 D5: the spending words painted exactly like the prose beside them.

    All six states shared the dim detail tint, so the word that says a leg spends
    money -- the design's own mitigation for surprise spend -- was indistinguishable
    from the sentence describing the provider. The ink table is keyed by the shared
    vocabulary, and this pins that it covers every word that vocabulary can print:
    a new state cannot render unstyled by omission.
    """
    from local_operator.tui.app import _SEARCH_STATE_TOKENS
    from local_operator.web_search.providers import STATE_MEANINGS

    assert set(_SEARCH_STATE_TOKENS) == set(STATE_MEANINGS)
    # The two spending words carry the caution ink; every other word is at least the
    # prose weight -- round 2 measured `faint` at 3.89:1, below AA and quieter than
    # the description beside it, for the two words that tell a reader to act.
    assert _SEARCH_STATE_TOKENS["auto paid"] == "warning"
    assert _SEARCH_STATE_TOKENS["enabled (paid)"] == "warning"
    assert _SEARCH_STATE_TOKENS["excluded"] == "muted"
    assert _SEARCH_STATE_TOKENS["needs setup"] == "muted"
    assert _SEARCH_STATE_TOKENS["auto free"] == "muted"


@pytest.mark.asyncio
async def test_search_order_notice_names_the_paid_landing(monkeypatch, tmp_path) -> None:
    """Round-2 U2-1 on this surface: the receipt is derived, not hardcoded."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.providers.registry import store_provider_key

    store_provider_key("DEEPSEEK_API_KEY", "stored")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search order deepseek")
        await pilot.pause()
        text = _transcript_text(app)

    assert "search order: deepseek (deepseek is paid and runs after the free legs;" in text
    # The false half of the old sentence is gone: nothing in this notice promises the
    # named paid leg is tried first.
    notice = text.split("search order: deepseek (", 1)[1].split(")")[0]
    assert "tried first" not in notice


@pytest.mark.asyncio
async def test_search_listing_labels_the_stored_order_and_defines_the_legend(
    monkeypatch, tmp_path
) -> None:
    """Round-2 U2-2/D2-4/U2-5: two arrow-lists, one labelled; a legend with meanings."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search")
        await pilot.pause()
        text = _transcript_text(app)

    # The stored prefix is named, so the `chain` row reads as the correction rather
    # than as a rival list.
    assert "order:" in text
    # The legend carries the meanings, not just the words.
    assert "auto free = with the free providers" in text
    assert "needs setup = cannot serve yet" in text


@pytest.mark.asyncio
async def test_needs_setup_notice_does_not_claim_it_applies_now(monkeypatch, tmp_path) -> None:
    """Round-2 D2-5: `; applies now` belongs to a sentence about a change."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search enable brave")
        await pilot.pause()
        text = _transcript_text(app)

    assert "brave is allowed, but no search can use it yet" in text
    assert (
        "no search can use it yet: run `local-operator search setup brave` (BRAVE_API_KEY)" in text
    )
    assert (
        "use it yet: run `local-operator search setup brave` (BRAVE_API_KEY); applies now"
        not in text
    )
