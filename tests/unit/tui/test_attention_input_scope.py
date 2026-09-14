"""Input witnesses one observed result, never a future completion (real store)."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual import events

from local_operator.harness.types import Message, TextContent
from local_operator.resume import SessionRow
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import CatalogEntry
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_attention import ReceiptSession, _settle
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote

#: Wall-clock bound on a real switch, so a switch that never commits fails this
#: file in seconds instead of sitting out the navigation's own retry budget.
#: A BACKSTOP, not an assertion: the pump ends the moment the switch binds.
_SWITCH_BACKSTOP_S = 5.0


async def mouse_down(app):
    # Pilot.click posts below App.on_event; use the actual driver's entry point
    # so this exercises the latch, not merely Textual's downstream widget route.
    await app.on_event(events.MouseDown(None, 1, 1, 0, 0, 1, False, False, False))


@pytest.mark.asyncio
@pytest.mark.parametrize("blur", [False, True])
@pytest.mark.parametrize("edge", ["key", "mouse"])
async def test_input_cannot_acknowledge_future_token(tmp_path, monkeypatch, blur, edge):
    monkeypatch.setattr("local_operator.tui.attention.focus_is_measurable", lambda: False)
    session = ReceiptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 0]
        if edge == "mouse":
            await mouse_down(app)
        else:
            await pilot.press("a")
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 1]
        if blur:
            app.on_app_blur(events.AppBlur())
        # Same visible row is deliberately the strongest adversarial case: the
        # token identity, not just an anchor still being on screen, must fence B.
        session.store.publish("session/sess", str(uuid.uuid4()), session.result.id, "complete")
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [2, 1]
        if edge == "mouse":
            await mouse_down(app)
        else:
            await pilot.press("b")
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [2, 2]


@pytest.mark.asyncio
async def test_blur_revokes_unconsumed_edge(tmp_path, monkeypatch):
    monkeypatch.setattr("local_operator.tui.attention.focus_is_measurable", lambda: False)
    session = ReceiptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        await mouse_down(app)
        app.on_app_blur(events.AppBlur())
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 0]


@pytest.mark.asyncio
async def test_hidden_result_is_not_witnessed_by_input(tmp_path, monkeypatch):
    monkeypatch.setattr("local_operator.tui.attention.focus_is_measurable", lambda: False)
    session = ReceiptSession(tmp_path / "attention.db", long=True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        app._transcript_view().scroll_home(animate=False)
        await pilot.pause()
        assert not app._completion_anchor_visible(session.result.id)
        await mouse_down(app)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 0]
        app._transcript_view().scroll_end(animate=False)
        await _settle(app, pilot, session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 0]


@pytest.mark.asyncio
async def test_switch_transfers_only_the_observed_catalogue_token(tmp_path, monkeypatch):
    monkeypatch.setattr("local_operator.tui.attention.focus_is_measurable", lambda: False)
    session = ReceiptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        # Exercise the driver's input -> selected-message -> async-bind seam.
        # The target result is not visible until after navigation, but its exact
        # token is already displayed by the catalogue at the input edge.
        app._transcript_view().display = False
        app._session_sidebar.set_entries(
            [
                CatalogEntry(
                    SessionRow(session.session_id, 0.0, "Target"),
                    True,
                    "complete",
                    session.token,
                    session.result.id,
                )
            ]
        )

        class Outgoing(FakeSession):
            @property
            def session_id(self):
                return "outgoing"

        app._session = Outgoing()
        monkeypatch.setattr(app, "_select_sidebar_session", lambda *args, **kwargs: None)
        await mouse_down(app)
        app.on_session_sidebar_selected(SessionSidebar.Selected(session.session_id))
        assert app._attention_navigation_receipt is not None, "the click carried no token"
        assert app._attention_navigation_receipt[:3] == (
            session.session_id,
            session.token,
            session.result.id,
        )
        # Binding is deliberately later than the click, as on a cold load.
        app._bind_viewer(session)
        app._transcript_view().display = True
        await _settle(app, pilot, session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [1, 1]
        app._bind_viewer(session)
        session.store.publish("session/sess", str(uuid.uuid4()), session.result.id, "complete")
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["revision"] == [2, 1]


class ReceiptRemote(SidebarRemote):
    """A sidebar VIEWER that carries the real attention surface.

    ``SidebarRemote`` is the suite's owner-backed viewer double (``owns_runtime
    = False``, a real frontend store, a real history window), so a switch to it
    runs the production prepare/commit/adopt path instead of a hand-bound
    session. This subclass adds only the two attention calls
    ``_poll_completion_attention`` reaches for, exactly as ``ReceiptSession``
    adds them to the owner fake.
    """

    def __init__(self, session_id: str, path: Path) -> None:
        # Set before ``super().__init__`` because the history the parent stores
        # has to be the result the attention store names.
        self.store = AttentionStore(path)
        self.token = str(uuid.uuid4())
        self.result = Message(
            role="assistant", content=[TextContent(text="The final answer is visible.")]
        )
        super().__init__(session_id, history=[self.result])
        self.store.publish("session/sess", self.token, self.result.id, "complete")

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, "session/sess")

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, "session/sess", token)


@pytest.mark.asyncio
async def test_the_real_switch_lands_the_clicked_token_on_the_incoming_session(
    tmp_path, monkeypatch
) -> None:
    """The seam the fix lives in, driven through the app's own switch.

    The case above stops ``_select_sidebar_session``, so nothing between the
    click and the receipt ever runs and it binds the viewer by hand. The chain
    this fix exists for -- ``SessionNavigation.select`` ->
    ``_prepare``/``_commit_sidebar_session`` -> ``_adopt_session`` ->
    ``_bind_viewer`` -- is therefore unpinned, and a ``_bind_viewer(None)``
    added anywhere in it would restore the operator's original symptom (the
    check mark never clearing on a click) with that case still green here.
    This case starts the REAL navigation from the click and reads the receipt
    the switch itself produced, then the mark: the incoming completion is only
    receipted if the clicked token survived the whole chain.
    """
    monkeypatch.setattr("local_operator.tui.attention.focus_is_measurable", lambda: False)
    # Both halves are VIEWERS: `_commit_sidebar_session` refuses to switch away
    # from a current session that is not one, so a boot on the owner fake cannot
    # reach the chain under test at all.
    outgoing = ReceiptRemote("outgoing-session", tmp_path / "outgoing.db")
    incoming = ReceiptRemote("incoming-session", tmp_path / "incoming.db")
    app = OperatorApp(lambda: _factory(outgoing))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, outgoing.result.id)
        # Only external discovery is redirected: the leased source is what the
        # production prepare/commit path then reads, unchanged.
        source = SessionInteraction(incoming)
        app._sidebar_sources[incoming.session_id] = source
        app._interactions[id(incoming)] = source

        async def lease(_session_id, *, speculative=False):
            source.preparations += 1
            return source

        app._lease_sidebar_source = lease  # type: ignore[method-assign]
        app._session_sidebar.set_entries(
            [
                CatalogEntry(
                    SessionRow(incoming.session_id, 0.0, "Target"),
                    True,
                    "complete",
                    incoming.token,
                    incoming.result.id,
                )
            ]
        )
        # The click, in the two halves the app itself splits it into: the
        # driver's input edge witnesses the catalogue token, then the handler
        # starts the switch.
        await mouse_down(app)
        app.on_session_sidebar_selected(SessionSidebar.Selected(incoming.session_id))
        deadline = asyncio.get_running_loop().time() + _SWITCH_BACKSTOP_S
        while app._session is not incoming and asyncio.get_running_loop().time() < deadline:
            await pilot.pause()
        assert app._session is incoming, (
            "the switch never bound the clicked conversation, so nothing below "
            "would be measuring the production chain"
        )
        assert app._attention_input_receipt == (
            incoming,
            incoming.token,
            incoming.result.id,
        ), "the token the click carried did not survive the switch's own bind"
        await _settle(app, pilot, incoming.result.id)
        await app._poll_completion_attention()
        assert not incoming.store.state("session/sess")[
            "unseen"
        ], "the click's token did not clear the incoming session's unseen completion"
