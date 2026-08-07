"""End-to-end footprint behaviour of a running Session.

The unit tests in ``test_transcript_footprint`` prove the store does the
right thing when told to. These prove the session actually tells it: a real
turn, the real compaction pruning pass, and then a real resume off the same
directory. The failure this guards against is silent — pruning would still
work, the transcript would still replay, and resume would simply cost more
every time without anything looking broken.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.thresholds import CompactionSettings
from local_operator.compaction.tokens import estimate_tokens, invalidate_message_cache
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import ENTRY_PRUNE, Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: Comfortably over compaction's MIN_PRUNE_TOKENS floor, so the pass has a
#: reason to blank it rather than deciding the placeholder costs more.
BIG_OUTPUT = "\n".join(f"line {i}: " + "content " * 8 for i in range(200))


class ScriptedStream:
    def __init__(self, turns):
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def read_tool() -> AgentTool:
    """A read tool whose result carries the ``path`` detail that compaction's
    supersede pass keys on — the ordinary shape, not a special case."""

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            content=[TextContent(text=BIG_OUTPUT)],
            details={"path": args["path"]},
        )

    return AgentTool(
        name="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


def _read_turn(call_id: str, path: str = "engine.py"):
    return [
        StreamToolCallDelta(
            index=0, id=call_id, name="read", argument_delta='{"path":"%s"}' % path
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _final_turn(text: str):
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


@pytest.mark.asyncio
async def test_superseded_read_is_journalled_and_shrinks_the_resume(tmp_path):
    """Read the same file twice, then resume: the resumed prompt must not
    carry the stale copy the live session already blanked."""
    directory = tmp_path / "sess"
    stream = ScriptedStream(
        [
            _read_turn("c1"),
            _final_turn("read once"),
            _read_turn("c2"),
            _final_turn("read twice"),
        ]
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[read_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    await session.prompt("read engine.py")
    await session.prompt("read engine.py again")

    # ``estimate_tokens`` memoizes on message.id, and a replayed message
    # reuses the id it was persisted under — so without invalidation every
    # measurement below would be served the estimate of whichever copy was
    # seen first. A real resume is a fresh process with a cold cache.
    def _tokens(messages) -> int:
        total = 0
        for message in messages:
            invalidate_message_cache(message)
            total += estimate_tokens(message)
        return total

    live_tokens = _tokens(session._context.messages)

    entries = Transcript(directory).entries()
    prunes = [entry for entry in entries if entry.type == ENTRY_PRUNE]
    assert prunes, "the superseded read was blanked in memory but never persisted"

    resumed = Transcript(directory).build_llm_history()

    # What a resume cost before the journal existed: the same transcript with
    # the journal rows stripped out, which is exactly the old file.
    naive_dir = tmp_path / "naive"
    naive_dir.mkdir()
    kept_lines = [
        line
        for line in (directory / "transcript.jsonl").read_text().splitlines()
        if f'"type":"{ENTRY_PRUNE}"' not in line
    ]
    (naive_dir / "transcript.jsonl").write_text("\n".join(kept_lines) + "\n")
    naive_tokens = _tokens(Transcript(naive_dir).build_llm_history())
    resumed_tokens = _tokens(resumed)

    # Measured 2,639 against 5,227 on this fixture: the resume drops the whole
    # superseded read. 0.55 leaves room for tokenizer drift without letting a
    # regression that resurrects the output slip through.
    assert resumed_tokens < naive_tokens * 0.55
    stale = [m for m in resumed if isinstance(m, Message) and BIG_OUTPUT in m.text]
    assert len(stale) == 1, "both copies of the superseded read came back on resume"
    # And a resume now costs what the live session cost, not more. The few
    # tokens of slack are the whitespace difference between the provider's
    # verbatim argument string and the ``json.dumps`` the encoder falls back
    # to once the redundant copy is dropped — not a resurrected tool output.
    assert resumed_tokens <= live_tokens + 16

    # The conversation itself is intact: same messages, same order, same ids,
    # so compaction can still reference any of them as a cut point.
    assert [m.role for m in resumed if isinstance(m, Message)] == [
        m.role for m in session._context.messages if isinstance(m, Message)
    ]
    assert [m.id for m in resumed] == [m.id for m in session._context.messages]


@pytest.mark.asyncio
async def test_resumed_session_continues_the_conversation(tmp_path):
    """Resume off the smaller representation and take another turn: the
    provider must receive the earlier history, not a bare prompt."""
    directory = tmp_path / "sess"
    first = Session(
        model=MODEL,
        stream_fn=ScriptedStream([_read_turn("c1"), _final_turn("done")]),
        tools=[read_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    await first.prompt("read engine.py")

    resumed_stream = ScriptedStream([_final_turn("still here")])
    resumed = Session(
        model=MODEL,
        stream_fn=resumed_stream,
        tools=[read_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    await resumed.prompt("what did you read?")

    sent = resumed_stream.requests[0].messages
    assert [m.role for m in sent[:4]] == ["user", "assistant", "tool", "assistant"]
    assert sent[0].text == "read engine.py"
    assert sent[-1].text == "what did you read?"
    # The tool result is still paired with its call, which every provider
    # requires and which a naive "drop the big message" optimisation breaks.
    assert sent[2].tool_call_id == sent[1].tool_calls[0].id


@pytest.mark.asyncio
async def test_journalling_never_duplicates_across_turns(tmp_path):
    """A prune pass runs every turn; re-journalling an already-blanked result
    would grow the transcript faster than the blanking shrinks it."""
    directory = tmp_path / "sess"
    stream = ScriptedStream(
        [
            _read_turn("c1"),
            _final_turn("one"),
            _read_turn("c2"),
            _final_turn("two"),
            _read_turn("c3"),
            _final_turn("three"),
        ]
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[read_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    for prompt in ("a", "b", "c"):
        await session.prompt(prompt)

    prunes = [e for e in Transcript(directory).entries() if e.type == ENTRY_PRUNE]
    targets = [e.payload["target"] for e in prunes]
    assert len(targets) == len(set(targets)), f"duplicate prune entries: {targets}"


@pytest.mark.asyncio
async def test_real_compaction_then_resume_replays_the_kept_window(tmp_path):
    """The invariant most at risk, exercised through the real compaction
    engine rather than a mock: a session that actually compacts, then a fresh
    Session over the same directory.

    The failure mode this catches is silent. Replay's documented fallback for
    an unresolvable ``first_kept_entry_id`` is "replay the full history", so a
    footprint change that broke the cut reference would look like a working
    resume while quietly restoring everything compaction just paid a
    summarization call to remove.
    """
    directory = tmp_path / "sess"
    # A context window small enough that the default threshold
    # (min(0.8 * window, 600k)) is crossed by a couple of big tool outputs.
    # DISTINCT paths on purpose: reads of the SAME path are superseded and
    # blanked by the prune pass, which drops the estimate back under the
    # threshold and means the fixture never reaches compaction at all.
    small_model = ModelSpec(provider="test", model_id="m", context_window=2_000)
    # The summarization call is routed by REQUEST SHAPE rather than by
    # position in a script: compaction fires whenever the estimate crosses
    # the threshold, which is not a fixed turn index, and a positional script
    # silently hands the summarizer whichever turn happens to be next.
    # ``_one_shot_complete`` is the only call with no tools.
    script = iter(
        [
            _read_turn("c1", "alpha.py"),
            _final_turn("one"),
            _read_turn("c2", "beta.py"),
            _final_turn("two"),
        ]
    )

    def stream(request: ChatRequest, signal: AbortSignal | None):
        events = (
            _final_turn("SUMMARY: the agent read two modules.")
            if not request.tools
            else next(script, _final_turn("done"))
        )

        async def gen():
            for event in events:
                yield event

        return gen()

    session = Session(
        model=small_model,
        stream_fn=stream,
        tools=[read_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        # keep_recent_tokens defaults to 20k, which on a fixture this size
        # keeps the whole history and makes find_cut_point return None.
        compaction_settings=CompactionSettings(
            threshold_tokens=1_500, keep_recent_tokens=500, auto_continue=False
        ),
    )
    await session.prompt("read alpha.py")
    await session.prompt("read beta.py")

    entries = Transcript(directory).entries()
    compactions = [entry for entry in entries if entry.type == "compaction"]
    assert compactions, "the fixture did not actually trigger a compaction"
    cut_id = compactions[-1].payload["first_kept_entry_id"]
    assert any(entry.id == cut_id for entry in entries), (
        "first_kept_entry_id no longer resolves; replay would silently fall "
        "back to the full history"
    )

    resumed = Transcript(directory).build_llm_history()
    marker = resumed[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    assert "SUMMARY:" in marker.details["summary"]
    # Exactly the kept window followed the marker — nothing from before the
    # cut came back.
    ids_after_marker = [m.id for m in resumed[1:]]
    entry_ids = [entry.id for entry in entries]
    assert ids_after_marker == entry_ids[entry_ids.index(cut_id) :][: len(ids_after_marker)]
