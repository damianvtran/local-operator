"""``compaction.resume_replay = "card"`` (#3448): a short resume card on replay.

What these pin, and why each matters:

* the DEFAULT stays the historical replay — no card on the record, and a
  resumed context equal to the live one message for message;
* card mode changes ONLY what ``replay_entries(mode="context")`` rebuilds: the
  live context after a pass still holds the verbatim preserved turns, while the
  resumed one is ``[marker, card, (carried), *tail]``;
* the card is stamped ``harness_injected`` and is never re-harvested as an
  operator turn, across a resume and a second pass (the ratchet the carried
  tier's stamp exists to prevent);
* the card actually bounds the reprint on a realistic fixture, by a measured
  margin rather than a claim;
* snapcompact records are scoped off, and the replay tail never splits a tool
  call from its result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import (
    PRESERVED_USER_TURN_KEY,
    RESUME_CARD_ID,
    extract_preserved_user_turns,
    is_carried_tier,
    is_rendered_injection,
    is_resume_card,
    resume_tail_start,
)
from local_operator.compaction.resume_card import build_resume_card, render_resume_card
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import (
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)
KEEP_RECENT = 40
CARD_HEADER = "[Resume card"


class ScriptedStream:
    def __call__(self, request, signal):
        async def gen():
            yield StreamTextDelta(delta="SUMMARY-OR-REPLY " * 10)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _session(directory: Path, mode: str | None) -> Session:
    settings = (
        CompactionSettings(keep_recent_tokens=KEEP_RECENT)
        if mode is None
        else CompactionSettings(keep_recent_tokens=KEEP_RECENT, resume_replay=mode)
    )
    return Session(
        model=TEXT_MODEL,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings,
    )


async def _round(session: Session, tag: str) -> None:
    for turn in range(6):
        await session.prompt(f"{tag} turn {turn} constraint-{tag}-{turn} " + "detail " * 30)


def _payload(directory: Path) -> dict[str, Any]:
    entries = [e for e in Transcript(directory).entries() if e.type == "compaction"]
    assert entries
    return entries[-1].payload


def _dump(messages) -> list[str]:
    return [m.model_dump_json() for m in messages]


@pytest.mark.asyncio
async def test_the_default_writes_no_card_and_resume_equals_live(tmp_path):
    directory = tmp_path / "sess"
    session = _session(directory, None)
    assert session._compaction_settings.resume_replay == "full"
    await _round(session, "a")
    assert (await session.compact_now()).ran is True
    payload = _payload(directory)
    assert "resume_card" not in payload
    assert payload.get("preserved_user_turns"), "fixture must exercise the preserved block"
    replayed = Transcript(directory).build_llm_history()
    assert not any(is_resume_card(m) for m in replayed)
    # The historical live/resume parity, unchanged by this feature.
    assert _dump(replayed) == _dump(_session(directory, None)._context.messages)
    await session.dispose()


@pytest.mark.asyncio
async def test_explicit_full_is_byte_identical_to_the_default(tmp_path):
    """``"full"`` spelled out and the absent key replay the same bytes."""
    dumps = []
    for name, mode in (("default", None), ("full", "full")):
        directory = tmp_path / name
        session = _session(directory, mode)
        await _round(session, "a")
        assert (await session.compact_now()).ran is True
        payload = _payload(directory)
        dumps.append(
            (
                sorted(payload),
                [
                    (
                        type(m).__name__,
                        getattr(m, "role", None),
                        m.text if isinstance(m, Message) else "",
                    )
                    for m in Transcript(directory).build_llm_history()
                ],
            )
        )
        await session.dispose()
    assert dumps[0] == dumps[1]


@pytest.mark.asyncio
async def test_card_mode_changes_replay_only(tmp_path):
    directory = tmp_path / "sess"
    session = _session(directory, "card")
    await _round(session, "a")
    assert (await session.compact_now()).ran is True
    payload = _payload(directory)
    assert isinstance(payload.get("resume_card"), dict)

    # LIVE context: untouched — verbatim preserved turns, no card.
    live = session._context.messages
    assert any(
        (getattr(m, "provider_payload", None) or {}).get(PRESERVED_USER_TURN_KEY) for m in live
    )
    assert not any(is_resume_card(m) for m in live)

    replayed = Transcript(directory).build_llm_history()
    assert (
        isinstance(replayed[0], CustomMessage) and replayed[0].custom_type == "compaction_summary"
    )
    assert is_resume_card(replayed[1]) and replayed[1].id == RESUME_CARD_ID
    assert replayed[1].text.startswith(CARD_HEADER)
    assert is_rendered_injection(replayed[1]), "card must carry the harness_injected stamp"
    # No verbatim preserved turn survives a card resume — the documented price.
    assert not any(
        (getattr(m, "provider_payload", None) or {}).get(PRESERVED_USER_TURN_KEY) for m in replayed
    )
    assert sum(1 for m in replayed if is_resume_card(m)) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_the_card_is_never_harvested_across_a_resume_and_a_second_pass(tmp_path):
    directory = tmp_path / "sess"
    first = _session(directory, "card")
    await _round(first, "a")
    assert (await first.compact_now()).ran is True
    await first.dispose()

    resumed = _session(directory, "card")
    assert sum(1 for m in resumed._context.messages if is_resume_card(m)) == 1
    # The pure function the pass uses refuses it even with every id "genuine".
    context = resumed._context.messages
    ids = {m.id for m in context if isinstance(m, Message)}
    assert not any(
        CARD_HEADER in str(turn.get("text", ""))
        for turn in extract_preserved_user_turns(context, ids)
    )

    await _round(resumed, "b")
    assert (await resumed.compact_now()).ran is True
    payload = _payload(directory)
    harvested = [str(t.get("text", "")) for t in payload.get("preserved_user_turns") or ()]
    assert harvested, "fixture must exercise the preserved block"
    assert not any(CARD_HEADER in text for text in harvested)
    # The live commit dropped the replay-only card from its kept window.
    assert not any(is_resume_card(m) for m in resumed._context.messages)
    # Second card covers round b, and still exactly one on replay.
    replayed = Transcript(directory).build_llm_history()
    assert sum(1 for m in replayed if is_resume_card(m)) == 1
    assert "constraint-b-" in replayed[1].text
    await resumed.dispose()


def _realistic_rows(turns: int) -> list[Message]:
    """Operator turns, each answered by a tool call, its result and a reply."""
    rows: list[Message] = []
    for index in range(turns):
        rows.append(
            Message.user(
                f"Turn {index}: please update module_{index}.py so the retry loop honours"
                f" the backoff ceiling; keep the public signature and do not touch tests. "
                + "Context detail. " * 12
            )
        )
        call = ToolCall(
            id=f"call-{index}", name="bash", arguments={"command": f"pytest -k m{index}"}
        )
        rows.append(Message(role="assistant", content=[], tool_calls=[call]))
        rows.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                tool_name="bash",
                content=[TextContent(text=f"{index} passed in 0.4s\n" + "log line\n" * 120)],
            )
        )
        rows.append(
            Message(
                role="assistant",
                content=[TextContent(text=f"Updated module_{index}.py; tests pass. " * 4)],
            )
        )
    return rows


@pytest.mark.asyncio
async def test_the_card_bounds_the_reprint_on_a_realistic_fixture(tmp_path):
    rows = _realistic_rows(120)
    cut = len(rows) - 4 * 12  # last twelve exchanges are the kept window
    summarized, kept = rows[:cut], rows[cut:]
    preserved = [
        {"id": str(t["id"]), "text": str(t["text"])}
        for t in extract_preserved_user_turns(summarized)
    ]
    totals = {}
    for mode in ("full", "card"):
        transcript = Transcript(tmp_path / mode)
        await transcript.append_messages(rows)
        await transcript.append_compaction(
            "Summary of the refactor so far. " * 20,
            kept[0].id,
            100_000,
            preserved_user_turns=preserved,
            resume_card=build_resume_card(rows) if mode == "card" else None,
        )
        replayed = Transcript(tmp_path / mode).build_llm_history()
        assert all(not (isinstance(m, Message) and m.role == "tool") for m in replayed[:3])
        totals[mode] = sum(estimate_tokens(m) for m in replayed if isinstance(m, Message))
    assert totals["card"] * 4 < totals["full"], totals


def test_the_tail_never_opens_on_a_tool_row_or_splits_a_pair():
    rows = _realistic_rows(6)
    for budget in (1, 50, 400, 2_000, 10**9):
        start = resume_tail_start(rows, budget)
        if start == len(rows):
            continue
        assert rows[start].role != "tool"
        tail_ids = {m.tool_call_id for m in rows[start:] if m.role == "tool"}
        calls = {c.id for m in rows[start:] for c in m.tool_calls or ()}
        assert tail_ids <= calls
    assert resume_tail_start(rows, 10**9) == 0
    assert resume_tail_start([], 100) == 0


@pytest.mark.asyncio
async def test_a_snapcompact_record_is_scoped_off(tmp_path):
    rows = _realistic_rows(4)
    transcript = Transcript(tmp_path / "snap")
    await transcript.append_messages(rows)
    await transcript.append_compaction(
        "s",
        rows[-4].id,
        1,
        preserve_data={"snapcompact": {"text_head": "h", "text_tail": "t"}},
        preserved_user_turns=[{"id": rows[0].id, "text": "verbatim constraint"}],
        resume_card=build_resume_card(rows),
    )
    replayed = Transcript(tmp_path / "snap").build_llm_history()
    assert not any(is_resume_card(m) for m in replayed)
    assert any(isinstance(m, Message) and m.text == "verbatim constraint" for m in replayed)


def test_the_card_renders_actions_open_items_and_next_step():
    card = build_resume_card(_realistic_rows(3))
    message = render_resume_card(card, [{"ask_id": "a-1", "questions": []}])
    assert message is not None and is_carried_tier(message) and is_resume_card(message)
    text = message.text
    assert "Turn 2:" in text and "bash(" in text and "2 passed in 0.4s" in text
    assert "Open items: 1 unresolved ask(s)" in text
    assert "Next step — last assistant message: Updated module_2.py" in text
    assert render_resume_card(None) is None
