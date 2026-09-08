"""The MCP startup toast announces once per process, per distinct outcome.

Reported: "MCP ready: 12 servers, 425 tools" fired on EVERY session attach,
including every click in the session sidebar. ``session.mcp_startup`` is a
frozen BOOT SNAPSHOT and ``_report_mcp_startup`` runs on every adoption, so a
sidebar switch re-announced a round that happened minutes ago — and, through
``RemoteSession``'s rehydration of the owner's outcome, sometimes a round this
process never ran at all.

These tests pin the rule the fix implements: announce at most once per app
process per DISTINCT outcome (content, not session identity), with the durable
failure notice deduped per session instead. The status band is asserted
alongside, because the suppression is only safe while the band keeps
re-stating live MCP state on every attach.
"""

from __future__ import annotations

import os

import pytest

from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import (
    FakeMcpManager,
    McpSession,
    _band,
    _factory,
    _transcript_text,
)


@pytest.fixture(autouse=True)
def isolate_sources(tmp_path, monkeypatch):
    # A headless pilot that inherits the operator's CMUX_* variables can rename
    # their real cmux workspaces; HOME is redirected too because the config dir
    # alone leaves the cache pointed at the real home (AGENTS.md, "Isolating a
    # run").
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def _outcome(**overrides) -> McpStartupOutcome:
    """The reported shape, shrunk: several servers up, a tool tally, no failure."""
    fields = {
        "configured": ("github", "linear", "slack"),
        "connected": ("github", "linear", "slack"),
        "tool_count": 425,
    }
    fields.update(overrides)
    return McpStartupOutcome(**fields)  # type: ignore[arg-type]


class _IdentifiedMcpSession(McpSession):
    """``McpSession`` with a settable id.

    ``FakeSession.session_id`` is a fixed property returning ``"sess"``, so two
    fakes are indistinguishable to the per-session notice key — which is
    exactly the distinction these tests are about.
    """

    def __init__(self, manager, startup, session_id: str) -> None:
        super().__init__(manager, startup)
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id


def _session(outcome: McpStartupOutcome, *, session_id: str) -> _IdentifiedMcpSession:
    manager = FakeMcpManager(list(outcome.configured), list(outcome.connected))
    return _IdentifiedMcpSession(manager, outcome, session_id)


@pytest.mark.asyncio
async def test_boot_announces_the_startup_outcome() -> None:
    """First loadup is exactly what the operator wants announced."""
    app = OperatorApp(lambda: _factory(_session(_outcome(), session_id="a")))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        assert toast.display is True
        assert "MCP ready: 3 servers, 425 tools" in toast.message


@pytest.mark.asyncio
async def test_attaching_a_second_session_with_the_same_outcome_stays_silent() -> None:
    """THE REPORTED DEFECT. MCP servers are process-wide and shared, so every
    sidebar click re-announced a tally the user was already told. The band still
    carries the live count, which is what makes the silence safe."""
    first = _session(_outcome(), session_id="a")
    second = _session(_outcome(), session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        assert toast.display is True
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert toast.display is False, "a re-attach re-announced a boot snapshot"
        # The surface that legitimately re-states MCP state per attach.
        assert "⊙ 3 MCP" in _band(app)


@pytest.mark.asyncio
async def test_re_adopting_the_same_session_stays_silent() -> None:
    """A sidebar click back onto the session already attached is the same
    snapshot a second time; nothing about it is news."""
    session = _session(_outcome(), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(session, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_changed_tally_announces_again() -> None:
    """A session in another cwd with a genuinely different server set IS news,
    and the fingerprint is content — so it announces once of its own."""
    first = _session(_outcome(), session_id="a")
    second = _session(
        _outcome(
            configured=("github", "linear", "slack", "notion"),
            connected=("github", "linear", "slack", "notion"),
            tool_count=511,
        ),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert toast.display is True
        assert "MCP ready: 4 servers, 511 tools" in toast.message


@pytest.mark.asyncio
async def test_connect_order_alone_does_not_re_announce() -> None:
    """``connected`` is a set with an incidental order — the connect race
    reorders it run to run. An order-sensitive key would re-toast an outcome
    whose content is identical, which is why the fingerprint sorts."""
    first = _session(_outcome(), session_id="a")
    second = _session(
        _outcome(connected=("slack", "github", "linear")),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert toast.display is False


@pytest.mark.asyncio
async def test_the_settled_outcome_still_announces_after_a_silent_gate() -> None:
    """The 250 ms gate snapshot is ``settling`` and unreportable, so it records
    nothing; the settled round that follows carries its own fingerprint and is
    the one the user actually sees. The fix must not consume the gate pass."""
    settling = McpStartupOutcome(
        configured=("github", "linear", "slack"),
        settling=True,
    )
    session = _session(settling, session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        assert toast.display is False, "a settling snapshot must stay quiet"

        # What the manager's settle callback does: the factory rebuilds
        # ``session.mcp_startup`` with the final tally, then the app re-reports.
        session.mcp_startup = _outcome()
        app._report_mcp_startup(session)
        await pilot.pause()
        assert toast.display is True
        assert "MCP ready: 3 servers, 425 tools" in toast.message


@pytest.mark.asyncio
async def test_a_failure_notice_is_written_once_per_session() -> None:
    """A durable failure record belongs in the transcript of the session it
    describes — so a SECOND session gets its own copy — but the repeat on the
    same session is the duplicate the re-attach used to append."""
    outcome = _outcome(
        connected=("github", "linear"),
        failures={"slack": "command not found: slack-mcp"},
        tool_count=310,
    )
    first = _session(outcome, session_id="a")
    second = _session(outcome, session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert _transcript_text(app).count("MCP slack failed") == 1

        # Same session again: the transcript already carries the record.
        app._adopt_session(first, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert _transcript_text(app).count("MCP slack failed") == 1

        # A different session with the same failure: its own transcript has
        # never carried it, so it gets the record even though the toast — a
        # process-wide interruption — stays silent.
        toast = app.query_one(Toast)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        for _ in range(6):
            await pilot.pause()
        assert _transcript_text(app).count("MCP slack failed") == 2
        assert toast.display is False
