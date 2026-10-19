"""The conversation's CHOSEN reasoning level survives a restart.

The reported defect: a level chosen with ``/effort`` on a session whose model is
the BOOT model was never written to the ``selected_model`` row — the
re-journal had an early return for the boot selector — so the next
``--resume`` restored the previous row's level. On a session that restarts
often, an explicit "Max" looks like it decays to the model's seeded "high"
after some time, with no keystroke and nothing on screen explaining it.

These tests drive the real ``Session`` (no model traffic: construction and
``set_model`` only), which is the same shape ``/effort`` takes.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.model_selection import read_model_selection
from local_operator.session.session import SELECTED_MODEL_CUSTOM_TYPE, Session
from local_operator.session.transcript import Transcript

from .test_session import ScriptedStream

#: A real registry row with a full ladder that includes ``max`` and seeds
#: ``high`` — the shape of the operator's session (deepseek-flash is also
#: ``('none','low','high','max')`` seeded ``high``).
BOOT = ModelSpec(
    provider="anthropic",
    model_id="claude-opus-5",
    context_window=200_000,
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    reasoning_effort="high",
    reasoning_default_effort="high",
)

#: A target whose ladder STOPS at ``xhigh``, so a chosen ``max`` genuinely
#: cannot be honoured across a fallback hop.
NARROW = ModelSpec(provider="openai", model_id="gpt-5", context_window=400_000)


class NotifyingStream(ScriptedStream):
    def __init__(self) -> None:
        super().__init__([[StreamEndEvent(stop_reason="stop")]])
        self.models_changed: list[ModelSpec] = []

    def on_model_changed(self, model: ModelSpec) -> None:
        self.models_changed.append(model)


def _session(tmp_path, stream, model=BOOT, **kwargs) -> Session:
    return Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
        **kwargs,
    )


def _rows(session: Session) -> list[dict[str, Any]]:
    return [
        entry.payload["details"]
        for entry in session._transcript.entries()
        if entry.type == "custom"
        and entry.payload.get("custom_type") == SELECTED_MODEL_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_a_chosen_level_on_the_boot_model_survives_a_restart(tmp_path):
    """``/effort max`` then restart: the conversation comes back on max.

    This is the exact reported shape. ``set_model`` on the same pair is what
    ``/effort`` does; the boot model's guard must not swallow the write.
    """
    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model.reasoning_effort == "max"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_chosen_level_on_a_switched_model_still_survives(tmp_path):
    """The control: the non-boot path this already worked for."""
    from local_operator.model.configure import build_model_spec

    switched = build_model_spec("anthropic", "claude-sonnet-5-5")
    session = _session(tmp_path, NotifyingStream())
    session.set_model(switched, explicit=True)
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model.model_id == "claude-sonnet-5-5"
    assert resumed.model.reasoning_effort == "max"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_the_journal_row_keeps_the_choice_and_the_boot_selector(tmp_path):
    """A row written on the birth model is a normal v2 row, not a special case."""
    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    await session.dispose()

    rows = _rows(session)
    assert rows, "a chosen level on the birth model must be journalled"
    assert rows[-1] == {
        "version": 2,
        "selector": "anthropic/claude-opus-5",
        "effort": "max",
        "boot": "anthropic/claude-opus-5",
    }
    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.effort == "max"


@pytest.mark.asyncio
async def test_max_survives_a_fallback_that_cannot_express_it(tmp_path):
    """A hop cannot take the level away — neither the choice nor the journal.

    The fallback target's ladder stops at ``xhigh``, so the level IN FORCE on
    that model is the clamp. The conversation's chosen level and the durable
    row must both still say ``max``: a rescue route is not a decision to
    forget what the user asked for. And the clamp is STATED — the canonical
    state carries the divergence and a notice words it — rather than letting
    the band's correct ``xhigh`` read as the user's own choice.
    """
    from local_operator.harness.types import NoticeEvent
    from local_operator.providers.failover import FallbackTarget

    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    await session.dispose()
    resumed = _session(tmp_path, NotifyingStream())
    events: list[Any] = []
    resumed.subscribe(events.append)
    assert resumed.model.reasoning_effort == "max"

    await resumed._on_route_settled(
        FallbackTarget("openai/gpt-5"), "provider failure: quota"
    )

    # What is serving is the clamped level; what the conversation CHOSE is max.
    assert resumed.effective_model.reasoning_effort == "xhigh"
    assert resumed.model.reasoning_effort == "max"
    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.effort == "max"

    # The divergence is published, with one sentence every surface can render.
    displacement = resumed.frontend_state.effort_displacement
    assert displacement["chosen"] == "max"
    assert displacement["effective"] == "xhigh"
    assert displacement["serving"] == "openai/gpt-5"
    assert displacement["selected"] == "anthropic/claude-opus-5"
    assert displacement["note"] == (
        "Falls back to xhigh while on openai/gpt-5 "
        "(chosen: max; returns with anthropic/claude-opus-5)."
    )
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert [event.text for event in notices] == [displacement["note"]]
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_restart_during_a_fallback_restores_the_chosen_level(tmp_path):
    """The fallback pin may outlive the process; the CHOICE it displaced must too.

    ``_restore_active_route`` re-adopts the rescue route recorded on the last
    edge, which is deliberate (the next turn should not be re-sent to the
    provider that was failing). What it must not do is turn the fallback's
    clamped level into the conversation's level: the selection row is still the
    chosen ``max``, and the re-derived serving spec clamps to ``xhigh``.
    """
    from local_operator.providers.failover import FallbackTarget

    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    await session._on_route_settled(
        FallbackTarget("openai/gpt-5"), "provider failure: quota"
    )
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model.reasoning_effort == "max"
    assert resumed.effective_model.reasoning_effort == "xhigh"
    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.effort == "max"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_switch_through_a_ladderless_route_keeps_the_choice(tmp_path):
    """The observed path: max -> a route with no effort ladder -> restart.

    The aggregator route cannot express any level, so its spec carries
    ``None``. A journal that recorded only the spec's level erased ``max`` at
    that moment — and the next switch back to the birth model fell to that
    model's seed ``high``. The choice is a property of the CONVERSATION, so it
    is held beside the spec and written to the row.
    """
    aggregator = ModelSpec(
        provider="test",
        model_id="ladderless",
        context_window=1_000_000,
        reasoning_efforts=(),
        reasoning_effort=None,
        reasoning_default_effort=None,
    )
    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    session.set_model(aggregator, explicit=True)
    assert session.model.reasoning_effort is None  # the route cannot hold it
    assert session.chosen_effort == "max"  # but the conversation still chose it
    await session.dispose()

    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.effort == "max"
    # A restart lands on the aggregator (the user's last selection) and still
    # knows the level the conversation chose.
    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model.model_id == "ladderless"
    assert resumed.chosen_effort == "max"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_effort_auto_withdraws_the_held_choice(tmp_path):
    """``/effort auto`` is a withdrawal; the choice must not come back later.

    ``/effort auto`` puts the model's own seed on the spec
    (``reasoning_effort == reasoning_default_effort``), which is how this
    codebase distinguishes "the user withdrew the level" from "the route could
    not express it". Without that distinction the held ``max`` would ride onto
    the next model the user switched to, resurrecting a preference they had
    explicitly dropped.
    """
    session = _session(tmp_path, NotifyingStream())
    session.set_model(session.model.model_copy(update={"reasoning_effort": "max"}))
    assert session.chosen_effort == "max"

    session.set_model(session.model.model_copy(update={"reasoning_effort": "high"}))
    assert session.model.reasoning_effort == BOOT.reasoning_default_effort
    assert session.chosen_effort is None
    await session.dispose()

    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.effort == "high"


def test_a_serving_spec_on_the_chosen_level_has_nothing_to_report():
    """No displacement key at all when the level in force IS the choice."""
    from local_operator.session.frontend_state import effort_displacement

    same = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        reasoning_effort="max",
        reasoning_default_effort="high",
    )
    assert effort_displacement(same, same.model_copy()) == {}


def test_a_model_seed_is_not_a_choice_so_it_cannot_be_displaced():
    """A level nobody chose is not reported as if it had been taken away.

    ``reasoning_effort`` carries a SEED on a freshly built spec; only a level
    that differs from ``reasoning_default_effort`` was chosen (the same
    predicate ``_carried_effort`` uses). Reporting a seed as "chosen" would
    invent a preference and then mourn it.
    """
    from local_operator.session.frontend_state import effort_displacement

    selected = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        reasoning_effort="high",
        reasoning_default_effort="high",
    )
    serving = ModelSpec(
        provider="openai",
        model_id="gpt-5",
        reasoning_efforts=("none", "low", "medium", "high", "xhigh"),
        reasoning_effort="medium",
        reasoning_default_effort=None,
    )
    # ``high`` equals the selected model's own seed, so nothing was CHOSEN.
    assert effort_displacement(selected, serving) == {}
