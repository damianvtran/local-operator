"""The carried tier holds across a long run of compactions, on both paths.

A fix that holds for two passes and degrades on the twenty-fifth is the same
cliff in a different hat. These tests therefore drive a synthetic session
through **thirty successive compactions** and assert the tier AFTER EVERY PASS:
the queued ask is still present and intact, the decisions are still present, an
answered ask never reappears, an expired ask never resurrects, and exactly one
carried row exists (not one more per pass). The real session that motivated
this is a 224.7 MB trace, far too large to assert against directly, so the
fixture is synthetic by design and says so.

Both paths are asserted every pass: the LIVE context the commit rebuilt, the
marker payload it wrote, and the history ``build_llm_history`` replays from the
journal. A tier written but not re-injected changes nothing; a tier that works
live but not on replay regresses silently on resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.asks import store as ask_store
from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.carry import (
    CARRIED_ASKS_KEY,
    CARRIED_ASKS_LIMIT,
    CARRIED_DECISIONS_KEY,
    CARRIED_DECISIONS_LIMIT,
    CARRIED_TIER_ID_PREFIX,
    read_recent_decisions,
    read_unresolved_asks,
    render_carried_tier,
)
from local_operator.harness.types import Message, ModelSpec, StreamEndEvent, StreamTextDelta
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)

#: Small keep window so each round of turns leaves history outside it, which is
#: what gives ``find_cut_point`` something to summarize.
KEEP_RECENT = 40

#: The operator's number for this fix. Not a smoke test.
PASSES = 31

_MINUTE_MS = 60_000
_BASE_MS = 1_790_000_000_000


class ScriptedStream:
    """Every summary and reply is a PARAPHRASE, never the carried text."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="FILLER assistant reply " * 20)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _make_session(tmp_path: Path, monkeypatch, stream) -> Session:
    # ``Session._config_dir`` resolves in ``__init__`` and the ask log lives
    # under it, so the override must be in place BEFORE the session exists.
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return Session(
        model=TEXT_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


def _queued(ask_id: str, question: str, *, created_at: int, expires_at: int) -> dict:
    return {
        "ask_id": ask_id,
        "at": created_at,
        "created_at": created_at,
        "expires_at": expires_at,
        "kind": "queued",
        "questions": [{"id": f"q-{ask_id}", "question": question, "multi": False}],
    }


def _answered(ask_id: str, *, at: int) -> dict:
    return {
        "ask_id": ask_id,
        "at": at,
        "kind": "answered",
        "answers": {f"q-{ask_id}": ["yes"]},
    }


def _seed_asks(config_dir: Path, session_id: str, events: list[dict]) -> None:
    directory = ask_store.session_dir(config_dir, session_id)
    for event in events:
        assert ask_store.append_event(directory, event) is True


def _seed_decision(store: MemoryStore, body: str, created_at: float) -> None:
    store.put(
        "session",
        {
            "title": body[:80],
            "body": body,
            "tags": ["decision"],
            "created_at": created_at,
        },
    )


def _carried_rows(messages) -> list[Message]:
    return [
        message
        for message in messages
        if isinstance(message, Message) and str(message.id).startswith(CARRIED_TIER_ID_PREFIX)
    ]


def _latest_compaction_payload(session: Session) -> dict:
    entries = [entry for entry in session._transcript.entries() if entry.type == "compaction"]
    assert entries, "the pass did not write a compaction marker"
    return entries[-1].payload


async def _add_round(session: Session, index: int) -> None:
    """Six turns: enough real history for the next pass to have work."""
    for turn in range(6):
        await session.prompt(f"round {index} turn {turn} " + "detail " * 30)


@pytest.mark.asyncio
async def test_the_carried_tier_holds_across_thirty_one_passes(tmp_path, monkeypatch):
    """THE ACCEPTANCE TEST: assert the LAST pass, not the second.

    A queued ask is open for the whole run; a second ask is answered at pass 12
    and must never reappear; a third expires at pass 8 and must never
    resurrect. Decisions are read live from the memory store, so one written
    mid-run appears from the pass after it was written.
    """
    clock = {"now": _BASE_MS}
    monkeypatch.setattr(ask_store, "now_ms", lambda: clock["now"])

    stream = ScriptedStream()
    session = _make_session(tmp_path, monkeypatch, stream)
    session_id = session._session_id
    config_dir = session._config_dir

    open_question = "May I spend the nonprod budget on the fixture?"
    answered_question = "Do you still want the nightly launchd sampler?"
    expiring_question = "Approve the temporary access grant?"
    _seed_asks(
        config_dir,
        session_id,
        [
            _queued(
                "a-aaa1",
                open_question,
                created_at=_BASE_MS,
                expires_at=_BASE_MS + 10_000 * _MINUTE_MS,
            ),
            _queued(
                "a-bbb2",
                answered_question,
                created_at=_BASE_MS,
                expires_at=_BASE_MS + 10_000 * _MINUTE_MS,
            ),
            _queued(
                "a-ccc3",
                expiring_question,
                created_at=_BASE_MS,
                expires_at=_BASE_MS + 8 * _MINUTE_MS,
            ),
        ],
    )

    decision_store = MemoryStore(
        config_dir, MemoryIdentity(session_id, "", ""), origin="device-test"
    )
    first_decision = "DECISION-ONE keep the existing helper"
    second_decision = "DECISION-TWO ship the carried tier"
    _seed_decision(decision_store, first_decision, created_at=1_700_000_000)

    answered_at = _BASE_MS + 12 * _MINUTE_MS
    answered = False
    second_written = False
    for index in range(PASSES):
        clock["now"] = _BASE_MS + index * _MINUTE_MS
        if index == 12 and not answered:
            _seed_asks(config_dir, session_id, [_answered("a-bbb2", at=answered_at)])
            answered = True
        if index == 5 and not second_written:
            _seed_decision(decision_store, second_decision, created_at=1_700_000_100)
            second_written = True

        await _add_round(session, index)
        outcome = await session.compact_now()
        assert outcome.ran is True, f"pass {index} did not run: {outcome.reason}"

        payload = _latest_compaction_payload(session)
        live_blocks = _carried_rows(session._context.messages)
        replayed_blocks = _carried_rows(
            Transcript(session._transcript.directory).build_llm_history()
        )

        # EXACTLY ONE on each path. A second copy is the accumulation defect
        # (one more row per pass), and it is what "does not grow unboundedly"
        # means here.
        assert len(live_blocks) == 1, f"pass {index}: live carried rows {len(live_blocks)}"
        assert len(replayed_blocks) == 1, (
            f"pass {index}: replay carried rows {len(replayed_blocks)}"
        )
        # Not journalled: a stored copy replays twice on resume (the elision
        # notice's 6 -> 7,167 defect).
        assert not any(
            entry.id.startswith(CARRIED_TIER_ID_PREFIX)
            for entry in session._transcript.entries()
        ), f"pass {index}: the carried row was persisted"
        live_text = live_blocks[0].text
        replay_text = replayed_blocks[0].text
        assert live_text == replay_text, f"pass {index}: live and resumed tiers diverged"

        # THE ASK THAT IS STILL OPEN: present and intact on every pass.
        assert "a-aaa1" in live_text and open_question in live_text, f"pass {index}: open ask lost"

        # THE ANSWERED ASK: present before the answer, gone after, forever.
        after_answer = clock["now"] >= answered_at
        if after_answer:
            assert "a-bbb2" not in live_text, f"pass {index}: answered ask reappeared"
            assert answered_question not in live_text, f"pass {index}: answered text reappeared"
        else:
            assert "a-bbb2" in live_text, f"pass {index}: ask not carried before its answer"

        # THE EXPIRING ASK: present while open, gone from its deadline on, and
        # never resurrected.
        if clock["now"] >= _BASE_MS + 8 * _MINUTE_MS:
            assert "a-ccc3" not in live_text, f"pass {index}: expired ask resurrected"
            assert expiring_question not in live_text, f"pass {index}: expired text resurrected"
        else:
            assert "a-ccc3" in live_text, f"pass {index}: open ask not carried"

        # DECISIONS: the first is always there; the second joins after pass 5.
        assert first_decision in live_text, f"pass {index}: decision evicted"
        if second_written:
            assert second_decision in live_text, f"pass {index}: new decision not carried"

        # THE PAYLOAD is the record replay reads, and it is COUNT-bounded.
        assert len(payload.get(CARRIED_ASKS_KEY, ())) <= CARRIED_ASKS_LIMIT
        assert len(payload.get(CARRIED_DECISIONS_KEY, ())) <= CARRIED_DECISIONS_LIMIT
        assert "a-aaa1" in {ask["id"] for ask in payload.get(CARRIED_ASKS_KEY, ())}
        if after_answer:
            assert "a-bbb2" not in {ask["id"] for ask in payload.get(CARRIED_ASKS_KEY, ())}
        if clock["now"] >= _BASE_MS + 8 * _MINUTE_MS:
            assert "a-ccc3" not in {ask["id"] for ask in payload.get(CARRIED_ASKS_KEY, ())}

    # The final pass is the one that matters: repeat the load-bearing claims
    # there so a regression late in the run cannot hide behind early greens.
    final = _carried_rows(session._context.messages)[0].text
    assert open_question in final and first_decision in final and second_decision in final
    assert "a-bbb2" not in final and "a-ccc3" not in final
    await session.dispose()


def test_the_ask_reader_is_live_and_count_bounded(tmp_path, monkeypatch):
    """Direct reader contract: open only, answered/expired gone, 50 cap."""
    clock = {"now": _BASE_MS}
    monkeypatch.setattr(ask_store, "now_ms", lambda: clock["now"])
    config = tmp_path / "config"
    directory = ask_store.session_dir(config, "sess")
    events: list[dict] = []
    for index in range(CARRIED_ASKS_LIMIT + 5):
        events.append(
            _queued(
                f"a-open{index:03d}",
                f"OPEN-QUESTION-{index}",
                created_at=_BASE_MS,
                expires_at=_BASE_MS + 10_000 * _MINUTE_MS,
            )
        )
    events.append(
        _queued(
            "a-answered",
            "ANSWERED-QUESTION",
            created_at=_BASE_MS,
            expires_at=_BASE_MS + 10_000 * _MINUTE_MS,
        )
    )
    events.append(_answered("a-answered", at=_BASE_MS + _MINUTE_MS))
    events.append(
        _queued(
            "a-expired",
            "EXPIRED-QUESTION",
            created_at=_BASE_MS,
            expires_at=_BASE_MS + 500,
        )
    )
    for event in events:
        assert ask_store.append_event(directory, event) is True

    clock["now"] = _BASE_MS + 1 * _MINUTE_MS
    carried = read_unresolved_asks(directory)
    assert len(carried) == CARRIED_ASKS_LIMIT
    ids = {ask["id"] for ask in carried}
    assert "a-answered" not in ids
    assert "a-expired" not in ids
    # Newest 50 are kept; the five oldest fall off the backstop.
    assert "a-open000" not in ids and "a-open005" in ids

    clock["now"] = _BASE_MS + 4 * _MINUTE_MS
    ids = {ask["id"] for ask in read_unresolved_asks(directory)}
    assert "a-expired" not in ids, "an expired ask resurrected on a later read"
    assert "a-open010" in ids


def test_the_decision_reader_is_newest_first_and_count_bounded(tmp_path):
    config = tmp_path / "config"
    store = MemoryStore(config, MemoryIdentity("sess", "", ""), origin="device-test")
    for index in range(CARRIED_DECISIONS_LIMIT + 5):
        store.put(
            "session",
            {
                "title": f"decision {index}",
                "body": f"DECISION-BODY-{index:02d}",
                "tags": ["decision"],
                "created_at": 1_700_000_000 + index,
            },
        )
    store.put(
        "session",
        {
            "title": "not a decision",
            "body": "UNRELATED-FACT",
            "tags": ["note"],
            "created_at": 1_700_000_999,
        },
    )

    carried = read_recent_decisions(store)
    assert len(carried) == CARRIED_DECISIONS_LIMIT
    bodies = [item["body"] for item in carried]
    assert bodies[0] == f"DECISION-BODY-{CARRIED_DECISIONS_LIMIT + 4:02d}"
    assert f"DECISION-BODY-{CARRIED_DECISIONS_LIMIT + 4:02d}" in bodies
    assert "DECISION-BODY-00" not in bodies
    assert all(item["tag"] == "decision" for item in carried)
    assert "UNRELATED-FACT" not in bodies


def test_an_unreadable_ask_log_carries_nothing(tmp_path, monkeypatch):
    """A missing file degrades to no tier, never to a failed pass."""
    clock = {"now": _BASE_MS}
    monkeypatch.setattr(ask_store, "now_ms", lambda: clock["now"])
    missing = ask_store.session_dir(tmp_path / "config", "absent")
    assert read_unresolved_asks(missing) == []


def test_the_cut_walker_never_selects_the_carried_row():
    """A cut on the synthesized row names an entry the journal does not have,
    and the session refuses the whole pass (``cut_not_replayable``)."""
    from local_operator.compaction.cutpoint import find_cut_point, is_carried_tier

    carried = render_carried_tier(
        [{"id": "a-1", "question": "Proceed?", "created_at": _BASE_MS}],
        [{"body": "DECISION", "tag": "decision"}],
    )
    assert carried is not None
    messages = [
        Message.user("older user turn " * 60),
        Message.assistant("older assistant turn " * 60),
        carried,
        Message.user("newer user turn " * 60),
        Message.assistant("newer assistant turn " * 60),
    ]
    for keep in (10, 40, 100, 200, 400, 800):
        cut = find_cut_point(messages, keep)
        assert cut is None or not is_carried_tier(messages[cut]), (
            f"keep={keep}: the cut landed on the carried row at {cut}"
        )


def test_the_renderer_marks_the_row_as_harness_output():
    """The row must not read as the operator's words, and must carry the stamp
    ``extract_preserved_user_turns`` refuses to re-harvest."""
    from local_operator.compaction.cutpoint import (
        RENDERED_INJECTION_KEY,
        extract_preserved_user_turns,
        is_rendered_injection,
    )

    row = render_carried_tier(
        [{"id": "a-1", "question": "Proceed?", "created_at": _BASE_MS}],
        [{"body": "Keep the helper", "tag": "decision"}],
    )
    assert row is not None
    assert row.role == "user"
    assert is_rendered_injection(row)
    assert row.provider_payload.get(RENDERED_INJECTION_KEY) is True
    assert "[compaction carry]" in row.text
    assert "not the user's words" in row.text
    assert extract_preserved_user_turns([row], {row.id}) == []
