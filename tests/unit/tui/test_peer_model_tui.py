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
            assert detail.splitlines() == [
                "switched to deepseek/deepseek-flash (was test/mock)",
                "its next turn runs on it",
            ]
            # The app's own /model ran: one explicit switch, through set_model.
            assert [(m.provider, m.model_id, e) for m, e in session.applied] == [
                ("deepseek", "deepseek-flash", True)
            ]
            assert session.peer_cards == [
                (
                    "[remote model switch] now on deepseek/deepseek-flash (was test/mock)",
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
            assert detail.splitlines() == [
                "switched to deepseek/deepseek-flash (was test/mock)",
                "mid-turn: the call in flight finishes on the old model",
                "1 running subagent stays on the old model; new and resumed ones switch",
            ]
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
            assert detail == "already on deepseek/deepseek-flash\nnothing changed"
            assert session.applied == []
    finally:
        store.close()


class _FallbackSession(_SwitchableSession):
    """Selection on Claude while a pinned fallback serves deepseek (M1)."""

    def __init__(self) -> None:
        super().__init__()
        self._label = "anthropic/claude-opus-5"
        self.active_fallback: Any = object()

    @property
    def effective_model_label(self) -> str:
        return "deepseek/deepseek-flash" if self.active_fallback is not None else self._label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        super().set_model(model, explicit=explicit)
        if explicit:
            self.active_fallback = None  # an explicit re-selection drops the pin


@pytest.mark.asyncio
async def test_a_tui_switch_onto_the_active_fallback_is_a_real_switch(
    tmp_path, monkeypatch
) -> None:
    """M1 on the TUI host: the "already on" test reads the SELECTION."""
    session = _FallbackSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert detail.splitlines()[0] == (
                "switched to deepseek/deepseek-flash (was anthropic/claude-opus-5)"
            )
            assert [(m.model_id, e) for m, e in session.applied] == [("deepseek-flash", True)]
            assert session.model_label == "deepseek/deepseek-flash"
            assert session.active_fallback is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_tui_raise_after_the_switch_took_is_reported_as_switched(
    tmp_path, monkeypatch
) -> None:
    """N1 on the TUI host: the read-back runs in `finally`, in the same hop."""
    session = _SwitchableSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            real = app._run_slash_command

            def raising(text, *args, **kwargs):  # noqa: ANN001
                real(text, *args, **kwargs)
                raise RuntimeError("receipt paint failed")

            monkeypatch.setattr(app, "_run_slash_command", raising)
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert detail.splitlines() == [
                "switched to deepseek/deepseek-flash (was test/mock)",
                "with an error after the switch: RuntimeError: receipt paint failed",
            ]
            assert len(session.peer_cards) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_an_accepted_local_switch_still_records_who_asked(tmp_path, monkeypatch) -> None:
    """Q2: a local-setup provider's capacity probe runs AFTER the hop, so the
    answer is `pending`; the card is written now, worded as a request."""
    session = _SwitchableSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)

            def pending_only(text, *args, **kwargs):  # noqa: ANN001
                # What `_cmd_model` does for a local-setup provider: park the
                # activation behind a probe and return with nothing switched.
                app._model_activation_pending = 1

            monkeypatch.setattr(app, "_run_slash_command", pending_only)
            detail = await handle.receive_peer_model(
                "deepseek", "deepseek-flash", sender={"conversation_name": "fleet boss"}
            )
            assert detail.startswith("pending: switch to deepseek/deepseek-flash accepted\n")
            assert session.applied == []
            assert [text for text, _ in session.peer_cards] == [
                "[remote model switch] switch to deepseek/deepseek-flash requested (on "
                "test/mock until it applies)"
            ]
    finally:
        app._model_activation_pending = None
        store.close()


@pytest.mark.asyncio
async def test_a_tui_pinned_on_the_request_whose_model_command_never_ran_is_a_refusal(
    tmp_path, monkeypatch
) -> None:
    """M2 on the TUI host: `/model` refused by the app's own gate (nothing applied)
    while a fallback already serves the requested model."""
    session = _FallbackSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            monkeypatch.setattr(app, "_run_slash_command", lambda *_a, **_k: None)
            with pytest.raises(ValueError) as caught:
                await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert "did not take effect" in str(caught.value)
            assert session.applied == [] and session.peer_cards == []
            assert session.model_label == "anthropic/claude-opus-5"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_tui_pending_local_switch_onto_the_pinned_fallback_stays_pending(
    tmp_path, monkeypatch
) -> None:
    """M2's second face: a local-setup activation whose provider IS the pinned
    fallback must answer `pending`, not `switched`."""
    session = _FallbackSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)

            def pending_only(*_a, **_k):  # noqa: ANN002, ANN003
                app._model_activation_pending = 1

            monkeypatch.setattr(app, "_run_slash_command", pending_only)
            detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
            assert detail.startswith("pending: "), detail
            assert [text for text, _ in session.peer_cards] == [
                "[remote model switch] switch to deepseek/deepseek-flash requested (on "
                "anthropic/claude-opus-5 until it applies)"
            ]
    finally:
        app._model_activation_pending = None
        store.close()


@pytest.mark.asyncio
async def test_a_tui_reclaiming_the_displaced_selection_says_back_on(tmp_path, monkeypatch) -> None:
    """N5 on the TUI host: asking for the selection a fallback displaced."""
    session = _FallbackSession()
    app, store = await _app(session, tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            handle = await _handle(app, pilot)
            detail = await handle.receive_peer_model("anthropic", "claude-opus-5", sender={})
            assert detail.splitlines()[0] == (
                "back on anthropic/claude-opus-5 (was on fallback deepseek/deepseek-flash)"
            )
            assert session.active_fallback is None
            assert [text for text, _ in session.peer_cards] == [
                "[remote model switch] back on anthropic/claude-opus-5 "
                "(was on fallback deepseek/deepseek-flash)"
            ]
    finally:
        store.close()
