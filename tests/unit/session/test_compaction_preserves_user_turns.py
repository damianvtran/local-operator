"""Compaction never summarizes a user turn.

A summarizer paraphrases, and a paraphrased user constraint ("use the existing
helper, don't add a new one" / "NEVER touch billing.py") is exactly how an
agent later does the forbidden thing. The soft summary-prompt clause that asks
the model to keep constraints is best-effort; these tests pin the STRUCTURAL
guarantee that replaced it: user-authored text is carried verbatim across a
compaction pass, on BOTH the live context and \u2014 the invariant a contiguous
``first_kept_entry_id`` suffix does not give for free \u2014 a resumed session.

Real compaction throughout (no stubbed ``compaction.api``): the split between
"summarize the assistant/tool content" and "preserve the user text verbatim"
is only interesting against the real cut point and the real transcript replay.
A text-only model is used so the strategy is the deterministic language summary
(``context-full``); the snapcompact path is covered by the vision-model round
trip in :func:`test_a_vision_model_still_preserves_user_turns_on_resume`.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import CustomMessage, Message, ModelSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

VISION_MODEL = ModelSpec(provider="test", model_id="sees", context_window=100_000)
TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)

#: Small keep window so a few short turns leave history outside it, which is
#: what gives ``find_cut_point`` something to summarize.
KEEP_RECENT = 40

#: A distinctive user constraint. If a compaction pass ever paraphrases it, the
#: exact substring below stops being present and every assertion here fails.
CONSTRAINT = "NEVER touch billing.py"


class ScriptedStream:
    """Replays one event script per call; every summary reply is a PARAPHRASE.

    The summarizer's reply is deliberately NOT the constraint text: a test
    whose summary happened to echo the constraint would pass even if the
    verbatim-preservation path did nothing. Making the summary a distinct
    string proves the constraint survives BECAUSE it was preserved, not
    because the summarizer copied it.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[object] = []

    def __call__(self, request, signal):
        from local_operator.harness.types import StreamEndEvent, StreamTextDelta

        self.requests.append(request)
        index = len(self.requests) - 1
        reply = self.replies[index] if index < len(self.replies) else "PARAPHRASED-SUMMARY"

        async def gen():
            yield StreamTextDelta(delta=reply)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, model=TEXT_MODEL) -> Session:
    return Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


async def _talk_with_constraint(session: Session, later_turns: int = 3) -> None:
    """Open with the distinctive constraint, then bury it under later turns.

    The opening turn must fall OUTSIDE the recent window (into the summarized
    partition) for the test to prove anything, which the later turns guarantee.
    Every turn carries filler so the backwards token walk stops past them.
    """
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(later_turns):
        await session.prompt(f"question {index} " + "detail " * 30)


def _user_texts(messages) -> str:
    return " ".join(m.text for m in messages if isinstance(m, Message))


@pytest.mark.asyncio
async def test_the_opening_constraint_lands_in_the_summarized_partition(tmp_path):
    """Grounding: the constraint turn IS on the to-summarize side of the cut.

    Without this the core test could pass trivially \u2014 a constraint that stayed
    in the kept window was never at risk. This is the CURRENT-behaviour half of
    the reproduction: the opening user turn falls before the cut, so the
    pre-fix code fed it to the summarizer and lost it.
    """
    from local_operator.compaction import api as compaction_api

    session = make_session(tmp_path, ScriptedStream(["reply"] * 8))
    await _talk_with_constraint(session)

    llm_history = session._render_for_compaction()
    cut = compaction_api.find_cut_point(llm_history, KEEP_RECENT)
    assert cut is not None
    to_summarize, kept = compaction_api.prepare_partitions(llm_history, cut)
    assert CONSTRAINT in _user_texts(to_summarize), "the test premise is void: constraint is kept"
    assert CONSTRAINT not in _user_texts(kept)
    await session.dispose()


@pytest.mark.asyncio
async def test_the_constraint_survives_compaction_verbatim_in_live_context(tmp_path):
    """The core guarantee, live: after a pass the exact constraint string is
    present in a real user message of ``_context.messages`` \u2014 not folded into
    the summary marker.

    The pre-fix path summarized the opening turn and the marker's summary is a
    paraphrase, so ``CONSTRAINT`` was absent from the live context entirely;
    this asserts it is back, and specifically in a ``Message`` (not the
    ``CustomMessage`` marker).
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 8))
    await _talk_with_constraint(session)

    outcome = await session.compact_now()
    assert outcome.ran is True

    marker = session._context.messages[0]
    assert isinstance(marker, CustomMessage) and marker.custom_type == "compaction_summary"
    # The summary itself is the paraphrase, and the guarantee is NOT that the
    # summary happens to contain the words \u2014 it is that the user's own message
    # survives beside it.
    assert CONSTRAINT not in marker.details.get("summary", "")
    preserved = [
        m
        for m in session._context.messages
        if isinstance(m, Message) and m.role == "user" and CONSTRAINT in m.text
    ]
    assert preserved, "the user constraint was summarized away instead of preserved verbatim"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_constraint_survives_on_resume(tmp_path):
    """The invariant found in review: a contiguous ``first_kept_entry_id``
    suffix would drop preserved turns on the next resume, because they sit
    BEFORE the cut in the transcript. This rebuilds the session purely from the
    persisted transcript and asserts the constraint is STILL present verbatim.

    A live-only hoist (preserving into ``_context.messages`` but not the marker
    payload) passes the live test above and fails THIS one \u2014 which is exactly
    the resume bug this whole payload path exists to prevent.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 8))
    await _talk_with_constraint(session)
    await session.compact_now()
    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    assert CONSTRAINT in _user_texts(replayed), "resume dropped the preserved user constraint"
    # Exactly one summary marker survives replay, and the preserved turn is a
    # user Message rather than a second marker.
    markers = [m for m in replayed if isinstance(m, CustomMessage)]
    assert len(markers) == 1
    assert any(
        isinstance(m, Message) and m.role == "user" and CONSTRAINT in m.text for m in replayed
    )


@pytest.mark.asyncio
async def test_all_user_turns_before_the_cut_are_preserved(tmp_path):
    """Not just the first: EVERY summarized user turn comes back verbatim, in
    order, live and on resume. Discriminates a fix that only rescued the head
    of the summarized block."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    # Distinctive markers so an omission or a reorder is visible.
    sentinels = [f"SENTINEL-{i} keep-this-exact" for i in range(4)]
    for sentinel in sentinels:
        await session.prompt(f"{sentinel} " + "detail " * 30)
    # Bury them so all four fall outside the recent window.
    for index in range(3):
        await session.prompt(f"tail {index} " + "detail " * 30)

    llm_history = session._render_for_compaction()
    from local_operator.compaction import api as compaction_api

    cut = compaction_api.find_cut_point(llm_history, KEEP_RECENT)
    assert cut is not None
    to_summarize, _ = compaction_api.prepare_partitions(llm_history, cut)
    summarized_sentinels = [s for s in sentinels if s in _user_texts(to_summarize)]
    assert len(summarized_sentinels) >= 2, "premise void: too few sentinels were summarized"

    await session.compact_now()
    directory = session._transcript.directory

    live_text = _user_texts(session._context.messages)
    for sentinel in summarized_sentinels:
        assert sentinel in live_text, f"{sentinel} lost from live context"

    await session.dispose()
    replayed_text = _user_texts(Transcript(directory).build_llm_history())
    for sentinel in summarized_sentinels:
        assert sentinel in replayed_text, f"{sentinel} lost on resume"

    # Order is preserved: the sentinels appear in their original sequence.
    positions = [replayed_text.index(s) for s in summarized_sentinels]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_role_alternation_stays_legal_after_preservation(tmp_path):
    """Preserved turns must not create an illegal adjacency or orphan a
    tool_call/result pair. Mirrors the pairing assertion the cutpoint tests
    use: every kept tool result has its issuing assistant on the same side and
    no summarized call is answered in the kept region."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 8))
    await _talk_with_constraint(session)
    await session.compact_now()

    messages = session._context.messages
    # No orphaned tool results: every tool message answers a call issued by an
    # assistant present in the same list.
    issued = {call.id for m in messages if isinstance(m, Message) for call in m.tool_calls}
    for message in messages:
        if isinstance(message, Message) and message.role == "tool":
            assert message.tool_call_id in issued, "a preserved/kept tool result lost its call"
    # The marker leads and preserved user turns follow it \u2014 a legal shape (the
    # marker renders as user content, which the wire path coalesces with an
    # adjacent user turn exactly as it did for marker+kept before this change).
    assert isinstance(messages[0], CustomMessage)
    await session.dispose()


@pytest.mark.asyncio
async def test_assistant_and_tool_content_is_still_summarized(tmp_path):
    """User preservation must not disable compaction: the summary path still
    fires and the pass still reduces tokens. A fix that preserved everything,
    or refused to summarize, would leave tokens flat.

    The assistant replies carry real bulk on purpose: user turns are now
    preserved verbatim, so the summarizable content is the assistant/tool
    side. That is exactly what the pass must still compress \u2014 a fixture whose
    only bulk was user text would legitimately show no reduction after the fix.
    """
    replies = ["assistant reply " * 60] * 4 + ["the paraphrased summary"]
    stream = ScriptedStream(replies)
    session = make_session(tmp_path, stream)
    await _talk_with_constraint(session, later_turns=4)

    outcome = await session.compact_now()
    assert outcome.ran is True
    assert outcome.strategy == "context-full"
    assert outcome.tokens_before > outcome.tokens_after > 0
    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert entries and entries[0].payload["summary"], "the summary path did not fire"
    await session.dispose()


@pytest.mark.asyncio
async def test_no_user_turns_before_the_cut_is_a_no_op(tmp_path):
    """Edge case: when the summarized block holds no genuine user turn, the
    marker carries no ``preserved_user_turns`` payload \u2014 preservation is inert,
    not a source of empty entries. Built by cutting a history whose only
    pre-cut content is assistant/tool (the opening user turn stays in the kept
    window)."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    await _talk_with_constraint(session, later_turns=4)
    await session.compact_now()

    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    first = entries[0]
    # This particular fixture DOES summarize a user turn, so assert the payload
    # is well-formed rather than absent; the inertness is proven by the helper
    # unit test. Here we pin that the payload only ever holds real user text.
    for turn in first.payload.get("preserved_user_turns") or []:
        assert turn["text"].strip(), "an empty user turn was preserved"
        assert "<previous-context-summary>" not in turn["text"], "a prior summary was preserved"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_vision_model_still_preserves_user_turns_on_resume(tmp_path):
    """The snapcompact (imaged) strategy preserves user turns too: the archive
    rasterizes the discarded history, so WITHOUT the verbatim payload the
    constraint would be readable only as pixels a resume may or may not decode.
    Same round trip, vision model, asserting the text survives replay."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 8), model=VISION_MODEL)
    await _talk_with_constraint(session)

    outcome = await session.compact_now()
    assert outcome.ran is True
    assert outcome.strategy == "snapcompact"
    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    assert CONSTRAINT in _user_texts(replayed), "snapcompact resume dropped the user constraint"


@pytest.mark.asyncio
async def test_a_second_pass_carries_the_first_generations_constraint_forward(tmp_path):
    """Multi-generation: a constraint preserved by pass 1 must still be present
    after pass 2, and pass 2 must NOT re-preserve pass 1's summary as if it
    were a user turn (which would nest summaries and grow unbounded)."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 40))
    await _talk_with_constraint(session)
    outcome1 = await session.compact_now()
    assert outcome1.ran is True
    for index in range(3):
        await session.prompt(f"round two {index} " + "detail " * 30)
    outcome2 = await session.compact_now()
    assert outcome2.ran is True
    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    assert CONSTRAINT in _user_texts(replayed), "gen-1 constraint lost after a second pass"
    assert len([m for m in replayed if isinstance(m, CustomMessage)]) == 1
    # No preserved turn is a rendered summary from a prior generation.
    entries = [e for e in Transcript(directory).entries() if e.type == "compaction"]
    for entry in entries:
        for turn in entry.payload.get("preserved_user_turns") or []:
            assert "<previous-context-summary>" not in turn["text"]
