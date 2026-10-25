"""The desktop's FIRST FRAME, eager versus deferred, over the real bridge.

This is the wire contract the renderer depends on. The deferred cold facade
changes where a few durable facts are read from — a durable metadata record
instead of a journal parse — and it must not change the frame the client
receives for the same fixture.

THE ORACLE IS CANONICAL, NOT RAW. Two independent opens legitimately differ in
fields nothing derives from the journal: the bridge mints a fresh ``epoch``, the
frame carries a serve timestamp, and per-page cursors are opaque tokens minted
per read. ``_canonical`` drops exactly those, by name, from one list — anything
NOT on that list differing is a real regression and fails.

Shapes covered, because they are the inputs condensation and grouping read: a
settled conversation with a checkpoint, a checkpointless (pre-ledger) journal, a
journal whose last tool call has no result yet, and a compacted journal whose
head was cut.

The journals are built through the real writer (``Transcript``), so every row is
a row this build can parse; the compaction marker is appended as a raw line
because its id-referencing payload is exactly what the index reads for
``head_cut``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.types import Message, TextContent, ToolCall, Usage
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import cold_metadata, replay_cache
from local_operator.session import transcript_index as ti
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.transcript import TRANSCRIPT_FILENAME, Transcript

SESSION_ID = "defeed012345"

#: Frame keys two independent opens cannot make equal, and why. ``epoch`` is
#: minted per facade (it names one attachment's lifetime), the serve clocks are
#: wall time, and cursors/tokens are opaque positions a reader mints per call.
#: Nothing on this list is derived from history.
_VOLATILE_KEYS = frozenset(
    {
        "epoch",
        "served_at",
        "now",
        "built_at",
        "generated_at",
        "ts",
        "ts_source",
        "cursor",
        "before_token",
        "snapshot_token",
        "subscription_id",
        "took_ms",
        "elapsed_ms",
    }
)


def _canonical(value: Any) -> Any:
    """``value`` with every documented-volatile field removed, recursively."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items() if key not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


async def _seed(
    root: Path,
    *,
    turns: int = 3,
    checkpoint: bool = True,
    spend: bool = True,
    unsettled_tail: bool = False,
    head_cut: bool = False,
) -> Path:
    """One conversation in the four shapes the first frame has to handle."""
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    kept_from = ""
    for turn in range(turns):
        question = Message(role="user", content=[TextContent(text=f"question {turn}")])
        if turn == turns - 1:
            kept_from = question.id
        await transcript.append_message(question)
        if spend:
            await transcript.append_custom(
                SESSION_SPEND_CUSTOM_TYPE,
                {"usd": 0.5, "calls": turn + 1},
            )
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text=f"answer {turn}")],
                tool_calls=[ToolCall(id=f"call{turn}", name="bash", arguments={"command": "true"})],
                usage=Usage(input_tokens=11 * (turn + 1), output_tokens=7, context_tokens=500),
            )
        )
        if unsettled_tail and turn == turns - 1:
            # The last call has no result: the shape the renderer paints as
            # still running, which a settled-only row list cannot show.
            continue
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id=f"call{turn}",
                tool_name="bash",
                content=[TextContent(text=f"output {turn}")],
            )
        )
    if checkpoint:
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {
                "checkpoint_id": "c1",
                "state": {
                    "session_id": SESSION_ID,
                    "epoch": "previous-owner",
                    "sequence": 3,
                    "conversation_title": "Article search rollout",
                    "cumulative_parent_cost": 4.5,
                    "context_tokens": 900,
                    "context_window": 1_000_000,
                },
            },
        )
    transcript.flush()
    if head_cut:
        # A compaction marker whose kept window starts ABOVE row zero is what the
        # index reports as ``head_cut``; the rows above it stay on disk, which is
        # the state the renderer's "older rows exist" affordance is built on.
        with (directory / TRANSCRIPT_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(
                '{"id":"compacted","ts":9999.0,"type":"compaction","payload":'
                '{"first_kept_entry_id":"' + kept_from + '","summary":"earlier work"}}\n'
            )
    return directory


#: Journal parses counted for the current test, and the hook that resets it.
#:
#: Counted rather than timed: a parse's duration moves by seconds under fleet
#: load, and the claim is a COUNT — a deferred open performs no journal parse.
_replay_calls = 0


def _reset_replay_calls() -> None:
    global _replay_calls
    _replay_calls = 0


@pytest.fixture(autouse=True)
def count_replay_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_replay_calls()
    original = replay_cache.read_replay_suffix

    def counted(*args: Any, **kwargs: Any) -> Any:
        global _replay_calls
        _replay_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(replay_cache, "read_replay_suffix", counted)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    replay_cache.replay_cache().clear()
    yield
    replay_cache.replay_cache().clear()
    ti._reset_for_tests()


async def _open(root: Path, sid: str) -> dict[str, Any]:
    """One cold desktop open of ``sid``, and the exposure it created.

    A FRESH POOL PER CALL, which is what makes the second open a second process
    as far as anything in memory can tell: no resident bridge, no replay-cache
    entry, no facade. Everything returned is read INSIDE the handover, because
    leaving the block detaches and disposes the facade it describes, and the
    parse count is sampled BEFORE the hydration so the two costs stay separable.
    """
    pool = DesktopSessions(root)
    facts: dict[str, Any] = {"session_id": sid, "parses_at_entry": _replay_calls}
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        opened = await anext(stream)
        snapshot = await anext(stream)
        await stream.aclose()
        assert opened["type"] == "open" and snapshot["type"] == "snapshot"
        facade = bridge.remote
        assert facade is not None, "an open builds the facade this test is about"
        facts["frame"] = snapshot
        facts["deferred"] = facade._deferred_history is not None
        facts["parses_before_hydration"] = _replay_calls
        facts["after_hydration"] = None
        if facts["deferred"]:
            await facade.hydrate_deferred_history()
            assert facade._deferred_history is None, "hydration settles the deferral"
            facts["after_hydration"] = await bridge.snapshot()
        facts["rows"] = len(facade.history())
    return facts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        {"turns": 3},
        {"turns": 3, "checkpoint": False, "spend": False},
        {"turns": 3, "unsettled_tail": True},
        {"turns": 4, "head_cut": True},
    ],
    ids=["settled", "checkpointless", "unsettled-tail", "head-cut"],
)
async def test_the_first_events_frame_is_the_same_frame_either_way(
    tmp_path: Path, shape: dict[str, Any]
) -> None:
    """Wire equality, on the frame rather than on the rows.

    The deferred run must have ACTUALLY deferred and must have skipped the journal
    parse entirely, and the eager run must have paid it — both checked beside the
    equality, because two eager opens would otherwise satisfy it.
    """
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    await _seed(root, **shape)

    eager = await _open(root, SESSION_ID)
    assert eager["deferred"] is False, "the first open of a journal is eager"
    assert cold_metadata.cold_metadata_path(
        root, SESSION_ID
    ).exists(), "the eager open is the write"
    assert (
        eager["parses_before_hydration"] > eager["parses_at_entry"]
    ), "the eager open parses the journal"

    # A SECOND PROCESS: nothing in memory survives, so only the durable record
    # can explain what the next open does.
    replay_cache.replay_cache().clear()
    _reset_replay_calls()
    deferred = await _open(root, SESSION_ID)
    assert deferred["deferred"] is True, "a matching record defers the message list"
    assert deferred["parses_before_hydration"] == 0, "a deferred open parses no journal"
    assert _replay_calls > 0, "and the hydration it defers does parse, once"

    assert _canonical(deferred["frame"]) == _canonical(eager["frame"])
    # The read it deferred produces the rows the eager open already had.
    assert deferred["rows"] == eager["rows"] > 0
    # And the frame does not shift once that read has landed.
    assert _canonical(deferred["after_hydration"]) == _canonical(eager["frame"])


@pytest.mark.asyncio
async def test_an_open_after_the_journal_changed_is_eager_and_paints_the_new_row(
    tmp_path: Path,
) -> None:
    """The record is for the journal version it describes, and no other.

    The row appended here must be painted, which is the visible consequence of
    the eager path being taken rather than served from a record that predates it.
    """
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    directory = await _seed(root, turns=2)

    first = await _open(root, SESSION_ID)
    assert first["deferred"] is False

    transcript = Transcript(directory)
    await transcript.append_message(
        Message(role="user", content=[TextContent(text="written later")])
    )
    transcript.flush()

    replay_cache.replay_cache().clear()
    second = await _open(root, SESSION_ID)
    assert second["deferred"] is False, "a moved journal must not be served from a record"
    assert second["rows"] == first["rows"] + 1
