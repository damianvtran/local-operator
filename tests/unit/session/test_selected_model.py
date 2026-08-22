"""The mid-session model selection survives quitting and resuming.

``/model <provider>/<id>`` switches the SESSION, and before this the switch
lived only in memory: quitting (ctrl+c) and coming back with ``--resume``
replayed the whole conversation onto the boot default, contradicting what the
transcript itself showed the user choosing.

The mechanism is a ``selected_model`` custom transcript entry, the sibling of
``active_model_route``: that one records where a provider FALLBACK routed
requests, this one records where the USER did. Both are journalled on the edge
and read back at construction.

Each row carries the boot selector the session was constructed with, which is
what lets the restore tell a switch that still applies from one stranded by a
changed boot selection — a ``/model default`` write, an edited agent profile,
or an explicit ``--hosting``/``--model`` flag on the resume itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.session import SELECTED_MODEL_CUSTOM_TYPE, Session
from local_operator.session.transcript import Transcript

from .test_session import MODEL, ScriptedStream, wait_for

#: What the user switches TO in these tests. A different provider as well as a
#: different id, so a restore that carried the boot spec's transport identity
#: across (the bug ``spec_for_target`` exists to prevent) shows up as a wrong
#: provider rather than only a wrong label. A REAL registry entry, because the
#: restore re-derives the spec through the registry rather than replaying a
#: persisted copy — see :func:`test_resume_comes_back_on_the_switched_model`.
SWITCHED = ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000)


class NotifyingStream(ScriptedStream):
    """A stream fn carrying the model-changed capability the session calls.

    The restore is expected to drive it for the same reason ``set_model``
    does: a stream fn that keys caches by selector must start keyed to the
    model that will actually serve, not to the boot default.
    """

    def __init__(self, turns=None) -> None:
        super().__init__(turns or [[StreamEndEvent(stop_reason="stop")]])
        self.models_changed: list[ModelSpec] = []

    def on_model_changed(self, model: ModelSpec) -> None:
        self.models_changed.append(model)


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


def _selection_entries(session: Session) -> list[dict[str, Any]]:
    return [
        entry.payload["details"]
        for entry in session._transcript.entries()
        if entry.type == "custom" and entry.payload.get("custom_type") == SELECTED_MODEL_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_a_switch_is_journalled_with_the_boot_selector(tmp_path):
    """``set_model`` records the selection, the effort, and what it booted on."""
    stream = NotifyingStream()
    session = _session(tmp_path, stream)

    session.set_model(SWITCHED)
    await wait_for(lambda: bool(_selection_entries(session)))

    assert _selection_entries(session) == [
        {"selector": "anthropic/claude-opus-5", "effort": None, "boot": "test/m"}
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_resume_comes_back_on_the_switched_model(tmp_path):
    """The reported bug: switch, quit, resume — and stay on the new model.

    Asserted through the spec rather than the label alone: the restore
    re-derives the target's OWN metadata through the registry, so a resumed
    session carries the switched model's transport identity and window — not
    the boot model's, and not a stale copy of what was journalled.

    The window assertion is deliberately against the REGISTRY's figure rather
    than the 200k the journalling session was constructed with: they differ,
    which is what makes this able to tell re-derivation from replay. A test
    pinning 200k would pass on a restore that simply replayed the persisted
    spec, which is the implementation this one exists to rule out.
    """
    from local_operator.model.configure import build_model_spec

    registry_window = build_model_spec("anthropic", "claude-opus-5").context_window
    assert registry_window != SWITCHED.context_window  # the control for the assert below

    stream = NotifyingStream()
    session = _session(tmp_path, stream)
    session.set_model(SWITCHED)
    await session.dispose()  # what ctrl+c does

    resumed_stream = NotifyingStream()
    resumed = _session(tmp_path, resumed_stream)

    assert resumed.model_label == "anthropic/claude-opus-5"
    assert resumed.model.base_url == "https://api.anthropic.com"
    assert resumed.model.context_window == registry_window
    # The stream fn is told, or the restore is display-only and the first
    # request goes back to the boot model.
    assert [m.model_id for m in resumed_stream.models_changed] == ["claude-opus-5"]
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_switch_made_moments_before_quitting_still_lands(tmp_path):
    """The switch-then-ctrl+c case, which is how the bug is actually hit.

    ``dispose`` cancels background tasks, so without the dispose flush the
    journal write for a switch made in the closing moments of a session is
    cancelled in flight and the next ``--resume`` opens on the boot default.
    Disposing with no intervening await is what reproduces it: the spawned
    write never gets a turn of the event loop.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)

    session.set_model(SWITCHED)
    await session.dispose()  # no await between the switch and the teardown

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model_label == "anthropic/claude-opus-5"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_switching_back_leaves_the_resume_on_the_boot_model(tmp_path):
    """Newest row wins: a switch and a switch back resume on the boot model."""
    stream = NotifyingStream()
    session = _session(tmp_path, stream)

    session.set_model(SWITCHED)
    session.set_model(MODEL)
    await session.dispose()

    resumed_stream = NotifyingStream()
    resumed = _session(tmp_path, resumed_stream)

    assert resumed.model_label == "test/m"
    # The no-op restore is skipped outright rather than re-derived, so nothing
    # is announced to a stream fn that was already keyed to this model.
    assert resumed_stream.models_changed == []
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_changed_boot_selection_outranks_the_journalled_switch(tmp_path):
    """``/model default`` (or ``--model``) between runs wins over the journal.

    The changed boot selection is the newer, more deliberate choice — and a
    restore that overrode it would make the flag the user just typed silently
    not work.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)
    session.set_model(SWITCHED)
    await session.dispose()

    rebooted_stream = NotifyingStream()
    rebooted = Session(
        model=ModelSpec(provider="openai", model_id="gpt-6", context_window=400_000),
        stream_fn=rebooted_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )

    assert rebooted.model_label == "openai/gpt-6"
    assert rebooted_stream.models_changed == []
    await rebooted.dispose()


@pytest.mark.asyncio
async def test_an_effort_change_is_not_a_switch(tmp_path):
    """``/effort`` copies the spec in place; it must not journal a selection.

    Same-model knob changes take ``set_model``'s early return, so the journal
    stays one row per real switch rather than one per keystroke.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)

    session.set_model(MODEL.model_copy(update={"reasoning_effort": "high"}))
    await session.dispose()

    assert _selection_entries(session) == []
    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model_label == "test/m"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_the_switched_effort_rides_the_restore(tmp_path):
    """A switch carrying an effort level resumes on that level.

    The level is part of what the user chose: ``/model`` carries the chosen
    effort onto the new model when it accepts one, so dropping it on resume
    would silently move them off it.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)

    session.set_model(SWITCHED.model_copy(update={"reasoning_effort": "high"}))
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model_label == "anthropic/claude-opus-5"
    assert resumed.model.reasoning_effort == "high"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_an_unresolvable_selection_falls_back_to_the_boot_model(tmp_path):
    """A journalled provider that no longer resolves must not break the resume.

    Losing a switch is survivable; refusing to open the conversation is not.
    And the row must be DROPPED rather than adopted: ``spec_for_target`` does
    not raise on an unknown provider, it returns a spec with ``base_url=None``,
    so a restore that skipped the registry check would resume the session onto
    a model that cannot serve requests and fail on the first prompt as a
    network error instead.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)
    await session._transcript.append_custom(
        SELECTED_MODEL_CUSTOM_TYPE,
        {"selector": "no-such-provider/no-such-model", "effort": None, "boot": "test/m"},
    )
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model_label == "test/m"
    await resumed.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "details",
    [
        {},
        {"selector": ""},
        {"selector": "no-slash", "boot": "test/m"},
        {"selector": 17, "boot": "test/m"},
        {"selector": "anthropic/claude-opus-5"},  # no boot recorded
    ],
)
async def test_a_malformed_row_is_tolerated(tmp_path, details):
    """A resume is never refused because one bookkeeping row is unreadable."""
    stream = NotifyingStream()
    session = _session(tmp_path, stream)
    await session._transcript.append_custom(SELECTED_MODEL_CUSTOM_TYPE, details)
    await session.dispose()

    resumed = _session(tmp_path, NotifyingStream())
    assert resumed.model_label == "test/m"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_resume_does_not_regrow_the_journal(tmp_path):
    """Restoring reads the row; it must not write another one.

    A resume that re-journalled what it just read would append a row per
    launch, so a long-lived conversation's transcript would grow with the
    number of times it has been opened rather than with its content.
    """
    stream = NotifyingStream()
    session = _session(tmp_path, stream)
    session.set_model(SWITCHED)
    await session.dispose()

    for _ in range(3):
        resumed = _session(tmp_path, NotifyingStream())
        await resumed.dispose()

    final = _session(tmp_path, NotifyingStream())
    assert len(_selection_entries(final)) == 1
    await final.dispose()


@pytest.mark.asyncio
async def test_a_bare_stream_fn_still_restores(tmp_path):
    """Hosts constructing sessions with plain stream functions degrade.

    ``on_model_changed`` is an optional capability, so its absence must cost
    the cache re-fit and nothing else — the session still resumes on the
    switched model.
    """
    session = _session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    session.set_model(SWITCHED)
    await session.dispose()

    resumed = _session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    assert resumed.model_label == "anthropic/claude-opus-5"
    await resumed.dispose()
