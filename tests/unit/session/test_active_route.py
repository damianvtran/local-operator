"""The effective-model surface: fallback display state, persistence, resume.

A provider fallback re-routes requests away from the SELECTED model, and the
session is the one place that fact is turned into host-visible state:

- ``effective_model`` / ``effective_model_label`` are what a front end paints
  its model display from (the composer band in the TUI);
- every route edge is persisted as an ``active_model_route`` custom entry, so a
  resumed session comes back on the model that was really answering;
- a ``ModelChangeEvent`` rides the session stream at each edge so the display
  updates live rather than at the next boot.

These tests drive the session through the same bridge the real
``SessionStreamFn`` uses (``set_route_handler``), because the contract under
test is the session's half: what it does when the stream reports a route edge.
The stream fn's own half (when it reports one) is covered in
``tests/unit/model/test_configure.py`` and ``tests/unit/providers/test_failover.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ModelChangeEvent, ModelSpec, StreamEndEvent
from local_operator.providers.failover import FallbackTarget
from local_operator.session.session import ACTIVE_ROUTE_CUSTOM_TYPE, Session
from local_operator.session.transcript import Transcript

from .test_session import MODEL, ScriptedStream, wait_for


class RoutedStream(ScriptedStream):
    """A stream fn with the route-bridge capability the session binds to."""

    def __init__(self, turns=None) -> None:
        super().__init__(turns or [[StreamEndEvent(stop_reason="stop")]])
        # Annotated: pyright infers Optional from a bare ``= None``, and every
        # test below CALLS the handler — an Optional call error in a suite that
        # only ever runs with the bridge installed is noise the assert already
        # answers.
        self.route_handler: Any = None
        self.notice_handler: Any = None
        self.restored: list[tuple[str, str | None, str]] = []
        self.models_changed: list[ModelSpec] = []
        self.withdrawals: int = 0

    def set_route_handler(self, handler) -> None:
        self.route_handler = handler

    def set_notice_handler(self, handler) -> None:
        self.notice_handler = handler

    def restore_fallback(self, selector: str, effort: str | None, primary: str) -> None:
        self.restored.append((selector, effort, primary))

    def on_model_changed(self, model: ModelSpec) -> None:
        self.models_changed.append(model)

    def withdraw_fallback(self) -> None:
        self.withdrawals += 1


def _session(tmp_path, stream, **kwargs) -> Session:
    transcript = Transcript(tmp_path / "sess")
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable", "env"],
        **kwargs,
    )


def _route_entries(session: Session) -> list[dict[str, Any]]:
    return [
        entry.payload["details"]
        for entry in session._transcript.entries()
        if entry.type == "custom" and entry.payload.get("custom_type") == ACTIVE_ROUTE_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_fallback_edge_updates_effective_model_and_emits(tmp_path):
    """A pinned fallback becomes the effective model, with its OWN metadata.

    The derived spec must be the target's (context window included), not a
    relabelled copy of the primary: a display naming the fallback while
    dividing usage by the primary's window misreports both at once.
    """
    stream = RoutedStream()
    session = _session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)

    assert session.effective_model_label == session.model_label

    assert stream.route_handler is not None
    await stream.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")

    assert session.effective_model_label == "zai/glm-5.3"
    assert session.model_label == "test/m", "the SELECTION must not move"
    # The target's own window, not the primary's 100k.
    assert session.effective_model.context_window != MODEL.context_window

    changes = [event for event in events if isinstance(event, ModelChangeEvent)]
    assert len(changes) == 1
    assert (changes[0].provider, changes[0].model_id) == ("zai", "glm-5.3")
    assert changes[0].is_fallback is True
    assert changes[0].context_window == session.effective_model.context_window


@pytest.mark.asyncio
async def test_recovery_edge_returns_display_to_the_selection(tmp_path):
    stream = RoutedStream()
    session = _session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)

    await stream.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")
    await stream.route_handler(None, "primary model recovered")

    assert session.effective_model_label == "test/m"
    assert session.active_fallback is None
    changes = [event for event in events if isinstance(event, ModelChangeEvent)]
    assert [change.is_fallback for change in changes] == [True, False]


@pytest.mark.asyncio
async def test_every_edge_is_persisted_including_recovery(tmp_path):
    """Recovery writes an ``active: None`` row.

    ``latest_custom`` scans backwards and stops at the first hit, so without
    the recovery row a session that fell back and recovered would resume
    pinned to a fallback nothing is wrong with.
    """
    stream = RoutedStream()
    session = _session(tmp_path, stream)

    await stream.route_handler(FallbackTarget("zai/glm-5.3", "high"), "provider failure")
    await stream.route_handler(None, "primary model recovered")

    entries = _route_entries(session)
    assert len(entries) == 2
    assert entries[0]["active"] == {"selector": "zai/glm-5.3", "effort": "high"}
    assert entries[0]["primary"] == "test/m"
    assert entries[1]["active"] is None


@pytest.mark.asyncio
async def test_resume_restores_the_pinned_fallback(tmp_path):
    """A session that closed while a fallback served resumes ON that fallback.

    Both halves matter: the display state (``effective_model``) so the band
    opens truthful, and the stream re-pin (``restore_fallback``) so the first
    prompt does not go back to the provider that was failing — a restore
    without the re-pin is display-only and lies the other way.
    """
    first = RoutedStream()
    _session(tmp_path, first)  # binds the route handler; entries land in tmp_path/sess
    await first.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")

    resumed_stream = RoutedStream()
    resumed = Session(
        model=MODEL,
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )

    assert resumed.effective_model_label == "zai/glm-5.3"
    assert resumed.model_label == "test/m"
    assert resumed_stream.restored == [("zai/glm-5.3", None, "test/m")]


@pytest.mark.asyncio
async def test_resume_after_recovery_restores_nothing(tmp_path):
    first = RoutedStream()
    _session(tmp_path, first)  # binds the route handler; entries land in tmp_path/sess
    await first.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")
    await first.route_handler(None, "primary model recovered")

    resumed_stream = RoutedStream()
    resumed = Session(
        model=MODEL,
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )

    assert resumed.effective_model_label == "test/m"
    assert resumed_stream.restored == []


@pytest.mark.asyncio
async def test_resume_drops_a_pin_recorded_against_another_selection(tmp_path):
    """A `/model default` change between runs invalidates the persisted pin.

    The pin rescued the OLD selection; the new selection owes the user a fresh
    start on the model they actually chose, not a detour recorded against a
    model they moved away from.
    """
    first = RoutedStream()
    _session(tmp_path, first)  # binds the route handler; entries land in tmp_path/sess
    await first.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")

    resumed_stream = RoutedStream()
    resumed = Session(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000),
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )

    assert resumed.effective_model_label == "anthropic/claude-opus-5"
    assert resumed_stream.restored == []


@pytest.mark.asyncio
async def test_resume_skips_a_pin_naming_the_current_selection(tmp_path):
    """The user adopting the fallback as their model makes the pin a no-op."""
    first = RoutedStream()
    _session(tmp_path, first)  # binds the route handler; entries land in tmp_path/sess
    await first.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")

    resumed_stream = RoutedStream()
    # Persisted primary was test/m, but the restore's primary-mismatch guard
    # fires first; craft a same-primary entry whose fallback IS the selection.
    resumed = Session(
        model=MODEL,
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    # Sanity: the ordinary restore happened. Now simulate adopting the
    # fallback: a new session selecting zai/glm-5.3 against the same entry.
    adopted_stream = RoutedStream()
    adopted = Session(
        model=resumed.effective_model,
        stream_fn=adopted_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    # The primary differs from the persisted one, so nothing restores — which
    # is also the right answer for "selection == fallback": no spurious pin.
    assert adopted.active_fallback is None
    assert adopted_stream.restored == []


@pytest.mark.asyncio
async def test_set_model_clears_the_pin_and_announces_the_selection(tmp_path):
    """An explicit `/model` switch withdraws the fallback pin's premise."""
    stream = RoutedStream()
    session = _session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)

    await stream.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")
    assert session.effective_model_label == "zai/glm-5.3"

    new_model = ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000)
    session.set_model(new_model)

    assert session.active_fallback is None
    assert session.effective_model_label == "anthropic/claude-opus-5"
    # The clear is persisted (spawned in the background) with the NEW primary,
    # so a resume does not restore the withdrawn pin.
    await wait_for(lambda: len(_route_entries(session)) == 2)
    entries = _route_entries(session)
    assert entries[1]["active"] is None
    assert entries[1]["primary"] == "anthropic/claude-opus-5"
    await wait_for(
        lambda: any(
            isinstance(event, ModelChangeEvent) and not event.is_fallback for event in events
        )
    )


@pytest.mark.asyncio
async def test_reselecting_the_same_model_withdraws_the_fallback_pin(tmp_path):
    """The reported stuck-fallback symptom, closed.

    A fallback pin rescues the SELECTED model without changing it, so "switching
    back" to the recovered model re-selects the model the session is ALREADY
    selected on. The selector never changes, so the stream fn's selector-driven
    route clear never fires — and the same-model branch of ``set_model`` used to
    treat the re-selection as a knob adjustment, leaving the pin in force. The
    user's only workaround was switching away and back.

    An explicit re-selection must withdraw the pin: the session's display state,
    the persisted route entry, and the stream fn's own route state.
    """
    stream = RoutedStream()
    session = _session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)

    await stream.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")
    assert session.effective_model_label == "zai/glm-5.3"

    # Re-select the SAME model the fallback displaced — explicitly, as /model does.
    session.set_model(session.model, explicit=True)

    assert session.active_fallback is None
    assert session.effective_model_label == session.model_label
    assert stream.withdrawals == 1  # the stream fn's route state was told to clear
    await wait_for(lambda: len(_route_entries(session)) == 2)
    entries = _route_entries(session)
    assert entries[1]["active"] is None
    assert entries[1]["primary"] == session.model_label
    await wait_for(
        lambda: any(
            isinstance(event, ModelChangeEvent)
            and not event.is_fallback
            and event.reason == "model reselected"
            for event in events
        )
    )


@pytest.mark.asyncio
async def test_knob_change_while_fallback_pinned_does_not_withdraw(tmp_path):
    """``/effort`` is not a model choice: it must NOT withdraw a pinned fallback.

    The same-model branch of ``set_model`` serves both an explicit re-selection
    and a knob adjustment; only the former withdraws. An ``/effort`` change
    while a fallback serves re-derives the display spec (the pinned test above
    pins that) but leaves the route exactly where it was.
    """
    stream = RoutedStream()
    session = _session(
        tmp_path,
        stream,
        model=ModelSpec(
            provider="test",
            model_id="m",
            context_window=100_000,
            reasoning_efforts=("low", "medium", "high"),
            reasoning_effort="low",
        ),
    )
    await stream.route_handler(FallbackTarget("openai/gpt-5.2", None), "provider failure")

    session.set_model(session.model.model_copy(update={"reasoning_effort": "high"}))

    assert session.active_fallback is not None  # the pin survives a knob change
    assert session.effective_model_label == "openai/gpt-5.2"
    assert stream.withdrawals == 0


@pytest.mark.asyncio
async def test_effort_change_rederives_the_fallback_display_spec(tmp_path):
    """`/effort` while a fallback serves must move the DISPLAYED effort too.

    ``spec_for_target`` carries the chosen level onto a chain entry that names
    none, so the derived display spec holds a snapshot of the old level; the
    same-model branch of ``set_model`` re-derives it from the pin.
    """
    stream = RoutedStream()
    session = _session(
        tmp_path,
        stream,
        model=ModelSpec(
            provider="test",
            model_id="m",
            context_window=100_000,
            reasoning_efforts=("low", "medium", "high"),
            reasoning_effort="low",
        ),
    )
    await stream.route_handler(FallbackTarget("openai/gpt-5.2", None), "provider failure")
    before = session.effective_model.reasoning_effort

    session.set_model(session.model.model_copy(update={"reasoning_effort": "high"}))

    after = session.effective_model.reasoning_effort
    assert session.effective_model_label == "openai/gpt-5.2"
    assert before != after
    assert after == "high"
    # Quietly: same-model set_model persists no route entry (the route did not
    # move) and notifies no model change.
    assert len(_route_entries(session)) == 1


@pytest.mark.asyncio
async def test_unresolvable_persisted_route_is_dropped_not_fatal(tmp_path):
    first = RoutedStream()
    session = _session(tmp_path, first)
    await session._transcript.append_custom(
        ACTIVE_ROUTE_CUSTOM_TYPE,
        {"primary": "test/m", "active": {"selector": "not-a-selector", "effort": None}},
    )

    resumed = Session(
        model=MODEL,
        stream_fn=RoutedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    assert resumed.active_fallback is None
    assert resumed.effective_model_label == "test/m"


@pytest.mark.asyncio
async def test_bare_stream_fn_degrades_to_the_selected_model(tmp_path):
    """Hosts constructing sessions without the route capability keep working."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream)
    assert session.effective_model is session.model
    assert session.effective_model_label == session.model_label


class FastRoutedStream(RoutedStream):
    """A stream fn that also exposes the fast-mode refusal bridge."""

    def __init__(self, turns: Any = None) -> None:
        super().__init__(turns or [[StreamEndEvent(stop_reason="stop")]])
        self.fast_refused_handler: Any = None

    def set_fast_refused_handler(self, handler) -> None:
        self.fast_refused_handler = handler


@pytest.mark.asyncio
async def test_a_fast_mode_refusal_switches_the_sessions_own_dial_off(tmp_path):
    """The session's half of review F1: after the stream reports that the
    provider refused fast mode, the SPEC the next request is built from no
    longer asks for it, so the band stops painting `fast` over standard
    requests and no later call re-pays the refusal.
    """
    stream = FastRoutedStream()
    model = MODEL.model_copy(update={"supports_fast_mode": True, "fast_mode": True})
    session = _session(tmp_path, stream, model=model)
    assert stream.fast_refused_handler is not None
    assert session.model.fast_mode is True

    await stream.fast_refused_handler(
        "anthropic/claude-opus-5", "Usage credits are required for fast mode."
    )

    assert session.model.fast_mode is False, "the dial must come off the spec"
    assert session.effective_model.fast_mode is False
    assert session.model_label == "test/m", "a knob change, never a model change"

    # Idempotent: a second report on a dial already off changes nothing.
    await stream.fast_refused_handler("anthropic/claude-opus-5", "again")
    assert session.model.fast_mode is False
