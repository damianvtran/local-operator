"""Input witnesses one observed result, never a future completion (real store)."""

import uuid

import pytest
from textual import events

from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_attention import ReceiptSession, _settle


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
    from local_operator.resume import SessionRow
    from local_operator.session.catalog import CatalogEntry
    from local_operator.tui.widgets.session_sidebar import SessionSidebar

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
