"""The conversation's FAST dial survives a restart, and a refusal says why.

Two reported defects share this file because they share one piece of state.

* ``damianvtran/local-operator-ui#1266`` — the desktop's fast toggle came back
  OFF after a reopen with no keystroke behind it. Fast mode is a SAME-PAIR knob
  change (``model_copy(update={"fast_mode": ...})``), so ``Session.set_model``
  takes its same-pair early return, and that path journalled nothing: the
  ``selected_model`` row carried the pair and the effort level, never the dial.
  The effort level survives for exactly this reason (see
  ``_journal_effort_if_selection_changed``); the dial had no equivalent.
* the same issue's second half — a provider that refuses fast mode turns the
  dial off from ``_on_fast_refused``, and the host that repaints the toggle was
  told nothing about why its saved preference stopped applying.

Driven on the real ``Session`` (construction, ``set_model``, dispose), the shape
``test_effort_survives_restart.py`` uses: the journal and the restore are what is
under test here, not provider traffic.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, NoticeEvent, StreamEndEvent
from local_operator.session.model_selection import read_model_selection
from local_operator.session.session import SELECTED_MODEL_CUSTOM_TYPE, Session
from local_operator.session.transcript import Transcript

from .test_session import ScriptedStream

#: A real registry row that SELLS the fast tier (``model/speed.py``'s Anthropic
#: list), with the support flag ``build_model_spec`` seeds on it.
BOOT = ModelSpec(
    provider="anthropic",
    model_id="claude-opus-5",
    context_window=200_000,
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    reasoning_effort="high",
    reasoning_default_effort="high",
    supports_fast_mode=True,
)

#: A real row with NO fast tier: the table keys the tier on the pair, and
#: neither Anthropic's explicit list nor OpenAI's family rule names this one.
NO_TIER = ModelSpec(provider="anthropic", model_id="claude-sonnet-5-5", context_window=200_000)


class RecordingStream(ScriptedStream):
    def __init__(self) -> None:
        super().__init__([[StreamEndEvent(stop_reason="stop")]])
        self.models_changed: list[ModelSpec] = []
        self.notices: list[tuple[str, str]] = []

    def on_model_changed(self, model: ModelSpec) -> None:
        self.models_changed.append(model)

    def set_notice_handler(self, handler: Any) -> None:
        self.notice_handler = handler

    # The refusal narration is asserted through the SESSION (what a host
    # receives), so this stream deliberately records nothing of its own.


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
        if entry.type == "custom" and entry.payload.get("custom_type") == SELECTED_MODEL_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_a_fast_dial_on_the_boot_model_survives_a_restart(tmp_path):
    """``/fast on`` then restart: the conversation comes back fast.

    The reported shape. Enabling fast is a same-pair knob change, which is the
    path that used to journal nothing at all.
    """
    session = _session(tmp_path, RecordingStream())
    session.set_model(session.model.model_copy(update={"fast_mode": True}))
    await session.dispose()

    resumed = _session(tmp_path, RecordingStream())
    assert resumed.model.fast_mode is True
    await resumed.dispose()


@pytest.mark.asyncio
async def test_the_row_carries_the_dial_beside_the_selection(tmp_path):
    """One row per change, and the dial is part of the snapshot.

    A SNAPSHOT, not a delta: a row that has no ``fast`` key cannot say whether
    the conversation had the dial off or never had one, and the backward scan
    would then resurrect an older row's ``fast``.
    """
    session = _session(tmp_path, RecordingStream())
    session.set_model(session.model.model_copy(update={"fast_mode": True}))
    await session.dispose()

    rows = _rows(session)
    assert rows, "a dial change on the birth model must be journalled"
    assert rows[-1]["fast"] is True
    assert rows[-1]["selector"] == "anthropic/claude-opus-5"
    assert rows[-1]["version"] == 2
    saved = read_model_selection(tmp_path / "sess")
    assert saved is not None and saved.fast is True


@pytest.mark.asyncio
async def test_turning_the_dial_off_is_journalled_too(tmp_path):
    """The OFF edge must reach disk, or the previous row comes back.

    ``latest_custom`` stops at the newest row, so an unjournalled OFF leaves the
    earlier ``fast: true`` snapshot as the newest evidence on disk.
    """
    session = _session(tmp_path, RecordingStream())
    session.set_model(session.model.model_copy(update={"fast_mode": True}))
    await session.dispose()
    again = _session(tmp_path, RecordingStream())
    assert again.model.fast_mode is True
    again.set_model(again.model.model_copy(update={"fast_mode": False}))
    await again.dispose()

    assert _rows(again)[-1]["fast"] is False
    resumed = _session(tmp_path, RecordingStream())
    assert resumed.model.fast_mode is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_model_without_the_tier_keeps_nothing(tmp_path):
    """The clamp: a route that does not report the tier cannot come back fast.

    A row can be older than the model table (a hand edit, a build whose table
    named a different pair), and sending the key on a route that does not sell
    the tier is a 400 on the turn this restore exists to open.
    """
    session = _session(tmp_path, RecordingStream(), model=NO_TIER)
    session.set_model(session.model.model_copy(update={"fast_mode": True}))
    await session.dispose()
    assert _rows(session)[-1]["fast"] is True, "the row records the wish it was given"

    resumed = _session(tmp_path, RecordingStream(), model=NO_TIER)
    assert resumed.model.supports_fast_mode is False
    assert resumed.model.fast_mode is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_provider_refusal_explains_the_reset_to_the_host(tmp_path):
    """The dial going off is STATED, not silent.

    ``_on_fast_refused`` is the authority on the dial, so the sentence about the
    reset is emitted from there: catalogue-backed (a code and its parameters) so
    a locale-aware host re-renders it instead of parsing English, and emitted
    exactly when a dial actually moved.
    """
    events: list[Any] = []
    session = _session(tmp_path, RecordingStream())
    session.subscribe(events.append)
    session.set_model(session.model.model_copy(update={"fast_mode": True}))

    await session._on_fast_refused(
        "anthropic/claude-opus-5", "Usage credits are required for fast mode."
    )

    assert session.model.fast_mode is False, "the dial must stop claiming the tier"
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert len(notices) == 1, notices
    assert notices[0].code == "tui.notices.fast_mode_refused"
    assert notices[0].params["selector"] == "anthropic/claude-opus-5"
    assert "standard speed" in notices[0].text
    await session.dispose()


@pytest.mark.asyncio
async def test_a_refusal_with_the_dial_already_off_says_nothing(tmp_path):
    """A refusal can only answer a request that ASKED for fast; no dial, no news."""
    events: list[Any] = []
    session = _session(tmp_path, RecordingStream())
    session.subscribe(events.append)

    await session._on_fast_refused("anthropic/claude-opus-5", "Usage credits are required.")

    assert [event for event in events if isinstance(event, NoticeEvent)] == []
    await session.dispose()
