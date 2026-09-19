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

    assert "states" in text
    for state in ("enabled", "auto free", "auto best-effort", "auto paid", "excluded"):
        assert state in text, state
    assert "needs setup" in text


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
    # The two spending words carry the caution ink; the two out-of-play words are
    # dimmer than the prose they sit in; the free in-play words stay a step above it.
    assert _SEARCH_STATE_TOKENS["auto paid"] == "warning"
    assert _SEARCH_STATE_TOKENS["enabled (paid)"] == "warning"
    assert _SEARCH_STATE_TOKENS["excluded"] == "faint"
    assert _SEARCH_STATE_TOKENS["needs setup"] == "faint"
    assert _SEARCH_STATE_TOKENS["auto free"] == "muted"
