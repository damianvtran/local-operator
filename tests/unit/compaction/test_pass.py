"""The host-independent compaction pass and the frame prune it composes.

Two properties are load-bearing here and the rest is coverage:

* ``keep_recent_frames=None`` (the default) must leave an ordinary session's
  pass byte-identical to a pass that predates the frame prune. The proof
  drives the whole pass with default settings over a history carrying images
  and asserts the output message-by-message against an independent
  construction of the pre-feature pipeline.
* A prompt-cache-aware caller must be able to tell an untouched message from
  a rewritten one by identity, so ``prune_stale_frames`` reuses untouched
  messages ``is``-identically and copies only the victims.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.compaction.marker import COMPACTION_MARKER_TYPE
from local_operator.compaction.pass_ import CompactionPassResult, run_compaction_pass
from local_operator.compaction.pruning import (
    STALE_FRAME_NOTICE,
    count_frame_messages,
    prune_stale_frames,
    prune_tool_outputs,
)
from local_operator.compaction.thresholds import CompactionSettings
from local_operator.compaction.tokens import estimate_messages_tokens
from local_operator.harness.types import ImageContent, Message, ModelSpec, TextContent

NOW = 10_000_000
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")


def _frame_turn(text: str, frames: int = 1) -> Message:
    return Message.user(text, [ImageContent(data=PNG) for _ in range(frames)])


def _texts(message: Message) -> list[str]:
    return [block.text for block in message.content if isinstance(block, TextContent)]


def _frames(message: Message) -> int:
    return sum(1 for block in message.content if isinstance(block, ImageContent))


# ---------------------------------------------------------------------------
# prune_stale_frames
# ---------------------------------------------------------------------------


def test_prune_stale_frames_keeps_newest_k_and_collapses_notices() -> None:
    messages = [
        _frame_turn("t0", frames=3),
        Message.assistant("a0"),
        _frame_turn("t1"),
        Message.assistant("a1"),
        _frame_turn("t2"),
        Message.assistant("a2"),
        _frame_turn("t3"),
    ]
    out, dropped = prune_stale_frames(messages, keep_recent_frames=2)

    # t0 (3 frames) and t1 (1 frame) lost their images; t2 and t3 kept theirs.
    assert dropped == 4
    assert [_frames(m) for m in out] == [0, 0, 0, 0, 1, 0, 1]
    # Three consecutive frames became ONE notice, not three.
    assert _texts(out[0]) == ["t0", STALE_FRAME_NOTICE]
    assert _texts(out[2]) == ["t1", STALE_FRAME_NOTICE]
    assert count_frame_messages(out) == 2


def test_prune_stale_frames_never_removes_messages_and_reuses_untouched_by_identity() -> None:
    messages = [_frame_turn("t0"), Message.assistant("a0"), _frame_turn("t1"), _frame_turn("t2")]
    out, dropped = prune_stale_frames(messages, keep_recent_frames=2)

    assert len(out) == len(messages)
    assert dropped == 1
    # The victim is a COPY (the caller's original is untouched)...
    assert out[0] is not messages[0]
    assert _frames(messages[0]) == 1
    assert out[0].id == messages[0].id
    # ...and everything else is the same object, which is how a sent-prefix
    # caller proves it did not rewrite them.
    assert out[1] is messages[1]
    assert out[2] is messages[2]
    assert out[3] is messages[3]


def test_prune_stale_frames_zero_keeps_nothing_and_none_to_prune_is_identity() -> None:
    messages = [_frame_turn("t0"), _frame_turn("t1")]
    out, dropped = prune_stale_frames(messages, keep_recent_frames=0)
    assert dropped == 2
    assert all(_frames(m) == 0 for m in out)

    text_only = [Message.user("hi"), Message.assistant("yo")]
    out, dropped = prune_stale_frames(text_only, keep_recent_frames=0)
    assert dropped == 0
    assert [a is b for a, b in zip(out, text_only)] == [True, True]


def test_prune_stale_frames_rejects_negative_budget() -> None:
    with pytest.raises(ValueError):
        prune_stale_frames([_frame_turn("t0")], keep_recent_frames=-1)


# ---------------------------------------------------------------------------
# run_compaction_pass
# ---------------------------------------------------------------------------


def _model(*, window: int = 128_000, images: bool = False) -> ModelSpec:
    return ModelSpec(provider="p", model_id="m", context_window=window, supports_images=images)


def _history(turns: int, *, words: int = 40) -> list[Message]:
    out: list[Message] = []
    for index in range(turns):
        out.append(_frame_turn(f"observation {index} " + "state " * words))
        out.append(Message.assistant(json.dumps({"actions": [{"kind": "wait", "n": index}]})))
    return out


class _Summarizer:
    def __init__(self, text: str = "SUMMARY") -> None:
        self.text = text
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


@pytest.mark.asyncio
async def test_run_compaction_pass_refuses_below_threshold_but_prunes_frames() -> None:
    history = _history(6)
    settings = CompactionSettings(keep_recent_frames=2)
    summarizer = _Summarizer()

    result = await run_compaction_pass(
        history,
        model=_model(),
        settings=settings,
        summarize=summarizer,
        now_ms=NOW,
        last_activity_ms=NOW,
    )

    assert isinstance(result, CompactionPassResult)
    assert result.ran is False
    assert result.reason == "below-threshold"
    assert result.frames_dropped == 4
    assert count_frame_messages(result.messages) == 2
    assert len(result.messages) == len(history)
    assert summarizer.prompts == []
    # Refused passes still hand back a usable list, with the same objects for
    # every message that was not pruned.
    assert result.messages[-1] is history[-1]
    assert result.messages[-2] is history[-2]


@pytest.mark.asyncio
async def test_run_compaction_pass_context_full_uses_injected_summarizer() -> None:
    history = _history(8, words=200)
    settings = CompactionSettings(keep_recent_tokens=300, keep_recent_frames=2)
    # The frame prune runs BEFORE the trigger (a prune that alone brings the
    # context under the line must never buy a summary), so the window is sized
    # against the post-prune history to make the threshold genuinely bite.
    pruned, _ = prune_stale_frames(history, keep_recent_frames=2)
    tokens = estimate_messages_tokens(pruned)
    summarizer = _Summarizer("Everything so far.")

    result = await run_compaction_pass(
        history,
        model=_model(window=int(tokens / 0.9)),
        settings=settings,
        summarize=summarizer,
        now_ms=NOW,
        last_activity_ms=NOW,
    )

    assert result.ran is True
    assert result.strategy == "context-full"
    assert result.summary_text == "Everything so far."
    assert len(summarizer.prompts) == 1
    assert "<conversation>" in summarizer.prompts[0]
    marker = result.messages[0]
    assert marker.role == "user"
    assert "<previous-context-summary>" in marker.text
    assert "Everything so far." in marker.text
    assert marker.provider_payload is not None
    assert marker.provider_payload[COMPACTION_MARKER_TYPE]["summary"] == "Everything so far."
    assert result.tokens_after < result.tokens_before
    # The kept tail is the same objects the caller passed in.
    assert result.messages[-1] is history[-1]
    assert len(result.messages) < len(history)


@pytest.mark.asyncio
async def test_run_compaction_pass_snapcompact_makes_no_provider_call() -> None:
    history = _history(8, words=200)
    settings = CompactionSettings(strategy="snapcompact", keep_recent_tokens=300)
    tokens = estimate_messages_tokens(history)

    async def never(_prompt: str) -> str:
        raise AssertionError("snapcompact must not call the summarizer")

    result = await run_compaction_pass(
        history,
        model=_model(window=int(tokens / 0.9), images=True),
        settings=settings,
        summarize=never,
        now_ms=NOW,
        last_activity_ms=NOW,
    )

    assert result.ran is True
    assert result.strategy == "snapcompact"
    assert result.preserve_data is not None and "snapcompact" in result.preserve_data
    assert result.messages[0].role == "user"


@pytest.mark.asyncio
async def test_run_compaction_pass_without_summarizer_refuses_rather_than_inventing() -> None:
    history = _history(8, words=200)
    tokens = estimate_messages_tokens(history)

    result = await run_compaction_pass(
        history,
        model=_model(window=int(tokens / 0.9)),
        settings=CompactionSettings(keep_recent_tokens=300),
        summarize=None,
        now_ms=NOW,
        last_activity_ms=NOW,
    )

    assert result.ran is False
    assert result.reason == "summarizer-failed"
    assert len(result.messages) == len(history)


@pytest.mark.asyncio
async def test_run_compaction_pass_disabled_is_a_no_op() -> None:
    history = _history(2)
    result = await run_compaction_pass(
        history,
        model=_model(),
        settings=CompactionSettings(enabled=False, keep_recent_frames=0),
        summarize=None,
        now_ms=NOW,
        last_activity_ms=NOW,
    )
    assert result.reason == "disabled"
    assert [a is b for a, b in zip(result.messages, history)] == [True] * len(history)


@pytest.mark.asyncio
async def test_a_second_pass_does_not_resummarize_the_marker() -> None:
    """A rendered marker is lifted back for the cut, so it is not "history"."""

    history = _history(8, words=200)
    tokens = estimate_messages_tokens(history)
    window = int(tokens / 0.9)
    settings = CompactionSettings(keep_recent_tokens=300)
    first = await run_compaction_pass(
        history,
        model=_model(window=window),
        settings=settings,
        summarize=_Summarizer("first"),
        now_ms=NOW,
        last_activity_ms=NOW,
    )
    assert first.ran

    second = await run_compaction_pass(
        first.messages,
        model=_model(window=window),
        settings=settings,
        summarize=_Summarizer("second"),
        now_ms=NOW,
        last_activity_ms=NOW,
        respect_threshold=False,
    )
    # With a marker plus a short kept tail there is nothing worth another
    # summary: the walker excludes the marker from what counts as history.
    assert second.ran is False
    assert second.reason == "nothing-to-summarize"


@pytest.mark.asyncio
async def test_defaults_leave_an_ordinary_session_pass_byte_identical() -> None:
    """THE compatibility proof for ``keep_recent_frames=None``.

    An ordinary session pastes images as distinct attachments, and the frame
    prune must never touch them unless a surface opts in. This drives the
    full pass with DEFAULT settings over a frame-bearing history and asserts
    the result equals, message for message, what the pre-feature pipeline
    (tool-output prune → threshold) produces on its own: same objects, same
    content, same image blocks, same token figure.
    """

    history = _history(6)
    for index in (1, 3):
        # A pair of tool results so the tool-output prune has real work.
        result = Message(role="tool", tool_call_id=f"call-{index}", tool_name="read")
        result.content = Message.user("content " * 200).content
        result.provider_payload = {"details": {"path": "/same.py"}}
        history.insert(index * 2, result)
    snapshot = [m.model_dump() for m in history]
    ids = [m.id for m in history]

    # Independent reference: the pipeline as it was before frames existed.
    reference = [Message.model_validate(dump) for dump in snapshot]
    reference, _ = prune_tool_outputs(reference, NOW, NOW)
    reference_dump = [m.model_dump(exclude={"id"}) for m in reference]

    result = await run_compaction_pass(
        history,
        model=_model(),
        settings=CompactionSettings(),
        summarize=_Summarizer(),
        now_ms=NOW,
        last_activity_ms=NOW,
    )

    assert result.ran is False
    assert result.reason == "below-threshold"
    assert result.frames_dropped == 0
    assert [m.id for m in result.messages] == ids
    assert [m is original for m, original in zip(result.messages, history)] == [True] * len(history)
    assert [m.model_dump(exclude={"id"}) for m in result.messages] == reference_dump
    assert count_frame_messages(result.messages) == 6
    assert result.tokens_before == estimate_messages_tokens(reference)


REPO = Path(__file__).resolve().parents[3]


def test_pass_module_is_import_isolated() -> None:
    """``pass_`` is consumed by the evaluation runner, which is forbidden the
    session, model and provider packages. A fresh-interpreter probe keeps a
    convenience import from re-coupling it."""

    probe = (
        "import json,sys;"
        "import local_operator.compaction.pass_;"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    imported = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    forbidden = (
        "local_operator.session",
        "local_operator.model",
        "local_operator.providers",
        "local_operator.config",
    )
    leaked = {
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    }
    assert not leaked, sorted(leaked)


# ---------------------------------------------------------------------------
# shed_stale_frames
# ---------------------------------------------------------------------------


def test_shed_stale_frames_removes_oldest_turns_never_the_current_observation() -> None:
    from local_operator.compaction.pruning import shed_stale_frames

    messages = [
        _frame_turn("t0"),
        Message.assistant("a0"),
        _frame_turn("t1"),
        Message.assistant("a1"),
        _frame_turn("t2"),
        Message.assistant("a2"),
        _frame_turn("t3"),
    ]
    out, removed = shed_stale_frames(messages, limit=2)

    # The two oldest turns (observation + reply) went; the newest two frames
    # and the current observation stayed, by identity.
    assert removed == 4
    assert [_texts(m)[0] for m in out] == ["t2", "a2", "t3"]
    assert out[0] is messages[4] and out[-1] is messages[-1]
    assert count_frame_messages(out) == 2

    # limit=0 keeps only the current observation: it is never shed.
    out, removed = shed_stale_frames(messages, limit=0)
    assert [_texts(m)[0] for m in out] == ["t3"]
    assert out[0] is messages[-1]
    assert removed == 6


def test_shed_stale_frames_stops_at_a_compaction_marker_and_at_a_frameless_prefix() -> None:
    from local_operator.compaction.marker import COMPACTION_MARKER_TYPE
    from local_operator.compaction.pruning import shed_stale_frames

    marker = Message.user("<previous-context-summary>...</previous-context-summary>")
    marker.provider_payload = {COMPACTION_MARKER_TYPE: {"summary": "..."}}
    messages = [marker, _frame_turn("t1"), Message.assistant("a1"), _frame_turn("t2")]
    out, removed = shed_stale_frames(messages, limit=1)
    # The marker is never shed: content already summarised is not deleted twice.
    assert out[0] is marker
    assert removed == 2
    assert [_texts(m)[0] for m in out[1:]] == ["t2"]

    # A frameless prefix (text-only benchmark) has nothing to shed: identity out.
    text_only = [Message.user("s0"), Message.assistant("a0"), Message.user("s1")]
    out, removed = shed_stale_frames(text_only, limit=0)
    assert removed == 0
    assert [a is b for a, b in zip(out, text_only)] == [True, True, True]

    with pytest.raises(ValueError):
        shed_stale_frames(messages, limit=-1)
