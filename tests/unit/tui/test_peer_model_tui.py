"""A peer switch on a TUI-hosted session, driven through a REAL ``OperatorApp``.

The architect's open risk (design "Risks"): ``_run_slash_command`` might SCHEDULE
``/model`` rather than run it, in which case a read-back in the same hop would
see the old label and report a successful switch as a failure — or, worse, the
reverse. These drive ``TuiSessionHandle.receive_peer_model`` against the app's
own ``/model`` path, through the real ``_on_app`` hop, with the real
``ProviderController`` over an isolated store, and read the answer the sender
gets.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.mobile import peer_model
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class _SwitchableSession(FakeSession):
    """A FakeSession whose label follows ``set_model``, like a real Session."""

    def __init__(self) -> None:
        super().__init__()
        self._label = "test/mock"
        self.applied: list[tuple[Any, bool]] = []
        self.peer_cards: list[tuple[str, dict[str, Any]]] = []

    @property
    def model_label(self) -> str:
        return self._label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        self.applied.append((model, explicit))
        self._label = f"{model.provider}/{model.model_id}"

    async def receive_peer_message(
        self, text, *, mode="mailbox", wake=False, sender=None
    ):  # noqa: ANN001
        assert (mode, wake) == ("mailbox", False), "the audit card must never open a turn"
        self.peer_cards.append((text, sender or {}))
        return "delivered to the mailbox (will be read on the next turn)"


async def _app(session: _SwitchableSession, tmp_path, monkeypatch):
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController
    from local_operator.tui.app import OperatorApp

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # Every provider counts as usable: this file is about the TUI hop, and the
    # credential probe has its own tests (test_peer_model.py).
    monkeypatch.setattr(peer_model, "provider_usable_here", lambda _p: True)
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    controller = ProviderController(store, tmp_path)
    return OperatorApp(lambda: _factory(session), provider_controller=controller), store


async def _handle(app, pilot):
    for _ in range(50):
        if app._mobile_handle is not None:
            return app._mobile_handle
        await pilot.pause(0.1)
    raise AssertionError("the TUI never built its control handle")


@pytest.mark.asyncio
async def test_an_idle_tui_switch_is_read_back_in_the_same_hop(tmp_path, monkeypatch) -> None:
    session = _SwitchableSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            sender = {"pid": 4242, "conversation_name": "fleet boss"}
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender=sender)
            await pilot.pause()
            assert detail == (
                "switched to deepseek/deepseek-flash (was test/mock); its next turn runs on it"
            )
            # The app's own /model ran: one explicit switch, through set_model.
            assert [(m.provider, m.model_id, e) for m, e in session.applied] == [
                ("deepseek", "deepseek-flash", True)
            ]
            assert session.peer_cards == [
                (
                    "[remote model switch] switched this session from test/mock to "
                    "deepseek/deepseek-flash",
                    sender,
                )
            ]
            assert handle._projection.model_label == "deepseek/deepseek-flash"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_busy_tui_switch_names_the_call_in_flight(tmp_path, monkeypatch) -> None:
    session = _SwitchableSession()
    session.streaming = True
    session.running_children = 1
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert detail == (
                "switched to deepseek/deepseek-flash (was test/mock) mid-turn; the call in "
                "flight finishes on test/mock, every later call uses deepseek/deepseek-flash; "
                "1 running subagent keeps its current model — new and resumed ones use "
                "deepseek/deepseek-flash"
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_tui_refusal_never_reaches_model(tmp_path, monkeypatch) -> None:
    session = _SwitchableSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            with pytest.raises(ValueError) as caught:
                await handle.receive_peer_model("deepseek", "not-a-model", sender={})
            assert str(caught.value) == (
                "refused: 'not-a-model' is not a model deepseek serves; still on test/mock"
            )
            assert session.applied == [] and session.peer_cards == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_tui_switch_that_did_not_take_is_a_refusal(tmp_path, monkeypatch) -> None:
    """``/model`` refuses only with a notice on screen; the read-back catches it."""
    session = _SwitchableSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            # A host that has lost its controller cannot run /model at all.
            app._providers = None
            with pytest.raises(ValueError) as caught:
                await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert "did not take effect" in str(caught.value)
            assert "still on test/mock" in str(caught.value)
            assert session.peer_cards == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_tui_same_pair_is_a_no_op(tmp_path, monkeypatch) -> None:
    session = _SwitchableSession()
    session._label = "deepseek/deepseek-flash"
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert detail == "already on deepseek/deepseek-flash; nothing changed"
            assert session.applied == []
    finally:
        store.close()
