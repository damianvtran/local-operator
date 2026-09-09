"""The preserved-user-turn block must not become a runaway ratchet.

Regression cover for a real incident: session ``2f95e374dd22`` fired 25
compaction passes in seven minutes, each reclaiming ~nothing, and grew a
107 MB transcript that filled the operator's disk. Two independent defects
produced it, and both are pinned here.

1. **Provenance.** ``extract_preserved_user_turns`` identified a "genuine"
   user turn by asking whether its id was one of the user ``Message`` entries
   in the LIVE context — justified by the claim that a harness injection is a
   ``CustomMessage`` there. Compaction's own output falsifies that from the
   SECOND pass on: a pass commits ``[marker, *preserved, *kept]`` built from
   the RENDERED history, so every injection inside the kept window is already
   a plain ``Message(role="user")`` in the live context and therefore passes
   the very test written to exclude it. On the real session, 160 of 171
   preserved turns were injections (79 of them repeats of the same
   ``session_state`` system/team brief) and 11 were genuine prompts.

2. **No bound.** ``find_cut_point`` skips already-preserved turns, so the
   block is never re-summarized and never dropped: measured over the last
   eight passes it went 166→171 carried forward with ZERO removed, reaching
   362,780 tokens against a ~640k trigger — 57% of the context permanently
   unshrinkable, which is why every pass produced no usable headroom.

The guarantee these must not weaken is pinned next door in
``test_compaction_preserves_user_turns.py``: a genuine user-authored
constraint still survives a pass verbatim.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import (
    DEFAULT_PRESERVED_TURN_CAP,
    PRESERVED_TURN_ELISION_ID,
    PRESERVED_TURN_ELISION_ID_PREFIX,
    RENDERED_INJECTION_KEY,
    _encode_len,
    cap_preserved_user_turns,
    elision_notice_text,
    extract_preserved_user_turns,
)
from local_operator.harness.types import Message, ModelSpec, TextContent
from local_operator.session.session import Session, _default_convert_to_llm
from local_operator.session.transcript import Transcript

TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)

KEEP_RECENT = 40

CONSTRAINT = "NEVER touch billing.py"

#: Stands in for the repeated system/team brief that dominated the real
#: session's preserved block (79 of 171 turns, 1.27 MB of 1.45 MB).
STATE_TEXT = "[session-state]\n## Available tools\nbash, read, write"


class ScriptedStream:
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


def make_session(tmp_path, stream) -> Session:
    return Session(
        model=TEXT_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


def _preserved(session) -> list[Message]:
    """Preserved turns in the live context: user messages carrying the flag."""
    from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

    return [
        message
        for message in session._context.messages
        if isinstance(message, Message)
        and (message.provider_payload or {}).get(PRESERVED_USER_TURN_KEY)
    ]


def _texts(messages) -> str:
    return " ".join(m.text for m in messages if isinstance(m, Message))


async def _inject_session_state(session: Session) -> None:
    """Append one ``session_state`` delivery exactly as ``_publish_state`` does.

    Built through ``_system_state_message`` rather than hand-rolled so the test
    breaks if the real injection stops being a ``CustomMessage``, which is the
    property the whole provenance argument rests on.
    """
    message = session._system_state_message({"1": STATE_TEXT})
    await session._transcript.append_message(message)
    session._context.messages.append(message)


# ---------------------------------------------------------------------------
# Defect 1 — provenance
# ---------------------------------------------------------------------------


def test_the_renderer_stamps_an_injection_and_not_a_real_prompt():
    """The provenance marker is set at mint time on injected deliveries only.

    This is the mechanism the rest of the fix depends on: after rendering, an
    injection and a prompt are both a plain user ``Message`` and nothing about
    their SHAPE distinguishes them.
    """
    from local_operator.harness.types import CustomMessage

    rendered = _default_convert_to_llm(
        [
            Message(role="user", content=[TextContent(text=CONSTRAINT)]),
            CustomMessage(
                custom_type="session_state",
                attribution="system",
                details={"text": STATE_TEXT},
            ),
        ]
    )

    assert len(rendered) == 2
    prompt, injection = rendered
    # Structurally identical — this is precisely why shape cannot decide it.
    assert prompt.role == injection.role == "user"
    assert not (prompt.provider_payload or {}).get(RENDERED_INJECTION_KEY)
    assert (injection.provider_payload or {}).get(RENDERED_INJECTION_KEY) is True


def test_a_rendered_injection_is_not_preserved_even_when_its_id_is_genuine():
    """The exact bug, at the unit boundary.

    The id set is what pass 2 hands in — and it CONTAINS the injection's id,
    because by then the injection is a plain user message in the live context.
    Before the fix that id-set membership was the whole test and the injection
    was preserved; now provenance excludes it while the real prompt survives.
    """
    prompt = Message(role="user", content=[TextContent(text=CONSTRAINT)])
    injection = Message(role="user", content=[TextContent(text=STATE_TEXT)])
    injection.provider_payload = {RENDERED_INJECTION_KEY: True}

    preserved = extract_preserved_user_turns(
        [prompt, injection],
        {prompt.id, injection.id},
    )

    assert [turn["text"] for turn in preserved] == [CONSTRAINT]


@pytest.mark.asyncio
async def test_a_session_state_delivery_is_not_preserved_on_the_second_pass(tmp_path):
    """End to end, against the real cut point and the real commit.

    The ORDERING here is the whole test, and getting it wrong makes the test
    vacuous. The injection must land in pass 1's KEPT window — i.e. be the last
    thing before pass 1 — so that pass 1's commit rebuilds the context from the
    RENDERED history and bakes it in as a plain ``Message(role="user")``. Only
    then does pass 2 see the generational state the bug lives in, where the
    injection's id is genuinely inside ``genuine_user_ids``.

    An earlier version of this test injected AFTER pass 1. The delivery was
    therefore still a ``CustomMessage`` when pass 2 built its id set, the id-set
    test excluded it for the wrong reason, and the test passed against a mutant
    with the provenance stamp removed — asserting the implementation back to
    itself. Verified under mutation: with ``_injected_user_message``'s stamp
    deleted this now fails, and the pre-fix tree leaks the brief here.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 40))
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)

    # Last before the pass: this is what puts it in the kept window.
    await _inject_session_state(session)
    assert (await session.compact_now()).ran is True

    # Pass 1 has now rewritten it into a plain user Message — the state the
    # defeated discriminator could not see. Assert that rather than trust it.
    baked = [
        message
        for message in session._context.messages
        if isinstance(message, Message) and STATE_TEXT in (message.text or "")
    ]
    assert baked, "test premise void: the injection is not a plain Message after pass 1"
    live_user_ids = {
        message.id
        for message in session._context.messages
        if isinstance(message, Message) and message.role == "user"
    }
    assert baked[0].id in live_user_ids, "test premise void: the old id-set test would exclude it"

    # Bury it so it falls into pass 2's SUMMARIZED partition.
    for index in range(4):
        await session.prompt(f"round two {index} " + "detail " * 30)

    assert (await session.compact_now()).ran is True

    preserved_texts = [m.text for m in _preserved(session)]
    assert any(CONSTRAINT in text for text in preserved_texts), "the real constraint was dropped"
    assert not any(
        STATE_TEXT in text for text in preserved_texts
    ), "a session_state injection was preserved as if the user had written it"
    await session.dispose()


@pytest.mark.asyncio
async def test_repeated_injections_do_not_ratchet_the_preserved_block(tmp_path, monkeypatch):
    """The runaway itself: many passes, many injections, bounded block.

    The pre-fix shape is +1 preserved turn per pass with ZERO removed, forever.
    Here two properties hold instead: no injection is ever preserved (defect 1)
    and the block stops growing once the cap binds (defect 2). Genuine prompts
    do still accumulate up to the bound — that is the guarantee working, not a
    leak — and beyond it the oldest are evicted behind a cumulative notice.

    The cap is forced to a tiny value so it binds within a handful of short
    synthetic turns; a real session's is ``threshold // 4`` (160,000 on the
    ~640k trigger this fix was diagnosed against). Patching the cap rather
    than inflating the turns keeps the test fast and keeps it testing the
    eviction rule rather than the tokenizer.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 200))
    monkeypatch.setattr(Session, "_preserved_turns_cap", lambda self, settings: 200)
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)

    counts: list[int] = []
    for round_index in range(4):
        await _inject_session_state(session)
        for index in range(2):
            await session.prompt(f"r{round_index} q{index} " + "detail " * 30)
        assert (await session.compact_now()).ran is True
        preserved = _preserved(session)
        counts.append(len(preserved))
        assert not any(
            STATE_TEXT in message.text for message in preserved
        ), f"round {round_index} preserved a session_state injection"

    # Bounded, not monotonic: the pre-fix series rose without limit.
    assert max(counts) <= 6, f"preserved block ratcheted: {counts}"
    assert counts[-1] <= counts[1], f"preserved block still growing: {counts}"

    # Eviction stays honest: exactly one notice, reporting the cumulative total.
    notices = [m for m in _preserved(session) if m.id.startswith(PRESERVED_TURN_ELISION_ID_PREFIX)]
    assert len(notices) == 1
    assert "older user message(s)" in notices[0].text
    await session.dispose()


def test_the_cap_is_capacity_shaped_not_a_keep_recent_multiple(tmp_path):
    """The preserved block is a session-long accumulation, not a recency window.

    Sizing it as ``keep_recent_tokens * _TASK_FLOOR_KEEP_MULTIPLE`` (the task
    floor's term) made the bound 200 tokens on a 40-token keep window — less
    than one user turn — so every constraint but the newest was evicted. Two
    advisor round-trip tests caught it. The bound must stay large enough to
    hold real constraints at a small keep window.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 4))
    settings = CompactionSettings(keep_recent_tokens=KEEP_RECENT)

    from local_operator.compaction.api import resolve_threshold_tokens

    cap = session._preserved_turns_cap(settings)

    assert cap >= 1_000, f"cap collapsed to {cap}: smaller than a single user turn"
    # And it tracks CAPACITY, not the keep window.
    threshold = resolve_threshold_tokens(TEXT_MODEL.context_window, settings)
    assert cap == max(KEEP_RECENT, threshold // 4)


def test_the_task_floor_does_not_anchor_on_an_injection():
    """``task_boundary_floor`` shares the discriminator and so shared the bug.

    Its docstring made the same false claim about injections being a
    ``CustomMessage`` in the live context. Anchoring the floor on a
    ``session_state`` delivery that arrives every turn would re-measure the
    "active task" from the injection, widening the preserve window on every
    pass — the same runaway through the other consumer.
    """
    from local_operator.compaction.cutpoint import task_boundary_floor

    prompt = Message(role="user", content=[TextContent(text=CONSTRAINT)])
    filler = [
        Message(role="assistant", content=[TextContent(text="work " * 200)]) for _ in range(3)
    ]
    injection = Message(role="user", content=[TextContent(text=STATE_TEXT)])
    injection.provider_payload = {RENDERED_INJECTION_KEY: True}
    messages = [prompt, *filler, injection]
    ids = {prompt.id, injection.id}

    # Anchored on the real prompt, the span covers the filler; anchored on the
    # injection it would be nearly nothing.
    span = task_boundary_floor(messages, ids, cap=1_000_000)
    from_injection = task_boundary_floor([injection], ids, cap=1_000_000)
    assert span > from_injection * 5


def test_subagent_and_peer_deliveries_are_excluded_too():
    """Hub/peer/job deliveries are not operator-authored constraints.

    They accounted for 80 of the real session's 171 preserved turns. A
    subagent's status report is ordinary history the summarizer may compress;
    preserving it buys no protection and costs the headroom a pass exists for.
    """
    from local_operator.harness.comms import HUB_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    rendered = _default_convert_to_llm(
        [
            CustomMessage(
                custom_type=HUB_MESSAGE_TYPE,
                attribution="system",
                details={"text": "subagent says it finished"},
            ),
        ]
    )
    assert extract_preserved_user_turns(rendered, {m.id for m in rendered}) == []


# ---------------------------------------------------------------------------
# Defect 2 — the cap
# ---------------------------------------------------------------------------


def test_the_cap_evicts_oldest_first_and_reports_it_as_a_genuine_drop():
    """Recent constraints are the live ones, so the OLDEST genuine turns go
    first — and the elision is counted rather than silent, because a silent
    drop is how an agent concludes a constraint was never given."""
    turns = [{"id": f"t{i}", "text": "word " * 100} for i in range(10)]

    capped = cap_preserved_user_turns(turns, cap=300)

    assert capped.turns, "the cap dropped everything"
    assert capped.genuine_dropped == len(turns) - len(capped.turns)
    assert capped.injections_dropped == 0
    surviving = [turn["id"] for turn in capped.turns]
    # A suffix of the original order: newest kept, oldest evicted.
    assert surviving == [turn["id"] for turn in turns][-len(surviving) :]
    # The notice names the operator's own words explicitly.
    notice = elision_notice_text(capped.genuine_dropped, capped.injections_dropped)
    assert notice is not None and "you wrote" in notice


def test_an_injection_is_shed_before_any_genuine_turn_is_evicted():
    """Provenance beats age. Shedding a harness injection costs the operator
    nothing, so it must never compete with a genuine turn for the budget.

    The previous revision relied on oldest-first eviction to remove injections,
    asserting they were "the oldest". They are typically the NEWEST
    (``session_state`` arrives every turn), so age-based eviction preferentially
    dropped the operator's constraints and kept the injections.
    """
    turns = [
        {"id": "genuine-old", "text": CONSTRAINT},
        {"id": "inj-1", "text": STATE_TEXT},
        {"id": "inj-2", "text": STATE_TEXT},
    ]

    capped = cap_preserved_user_turns(turns, cap=1_000_000, injection_ids={"inj-1", "inj-2"})

    assert [t["id"] for t in capped.turns] == ["genuine-old"]
    assert capped.injections_dropped == 2
    # No genuine turn was touched, so the notice must not claim one was.
    assert capped.genuine_dropped == 0
    notice = elision_notice_text(capped.genuine_dropped, capped.injections_dropped)
    assert notice is not None
    assert "harness-injected" in notice and "you wrote" not in notice


def test_the_cap_is_a_token_budget_not_a_turn_count():
    """One oversized paste and 300 short turns are the same problem, and only a
    token budget sees both."""
    small = [{"id": f"s{i}", "text": "hi"} for i in range(50)]
    assert cap_preserved_user_turns(small, cap=10_000).turns == small

    big = [{"id": "a", "text": "word " * 5_000}, {"id": "b", "text": "tiny"}]
    capped = cap_preserved_user_turns(big, cap=1_000)
    assert [turn["id"] for turn in capped.turns] == ["b"]


def test_the_newest_turn_survives_even_when_it_alone_exceeds_the_cap():
    """Returning an empty block for one oversized turn would discard the LIVE
    constraint — the exact failure preservation exists to prevent."""
    turns = [{"id": "old", "text": "word " * 100}, {"id": "new", "text": "word " * 10_000}]

    capped = cap_preserved_user_turns(turns, cap=50)

    assert [t["id"] for t in capped.turns] == ["new"]


def test_a_non_positive_cap_fails_closed():
    """A cap of 0 used to return the block UNBOUNDED — restoring the exact
    ratchet this function exists to prevent, on the one input a caller would
    plausibly pass to mean "nothing survives". It now falls back to the shipped
    default, so the invariant holds for every caller rather than the careful
    one."""
    turns = [{"id": f"t{i}", "text": "word " * 5_000} for i in range(40)]

    for degenerate in (0, -5):
        capped = cap_preserved_user_turns(turns, cap=degenerate)
        assert len(capped.turns) < len(turns), f"cap={degenerate} degraded open"
        total = sum(_encode_len(t["text"]) for t in capped.turns)
        assert total <= DEFAULT_PRESERVED_TURN_CAP


def test_the_running_count_is_carried_structurally_not_reparsed():
    """The count must survive a round trip without existing as content.

    Encoding it in a synthetic turn's id meant the persist path journalled that
    turn and every later pass re-counted it: the reported figure doubled on
    each resume (6 -> 7,167 over ten cycles, a 231x overstatement). Carried as
    arguments, re-capping an already-capped block neither inflates nor loses it.
    """
    turns = [{"id": f"t{i}", "text": "word " * 100} for i in range(10)]

    once = cap_preserved_user_turns(turns, cap=300)
    twice = cap_preserved_user_turns(
        once.turns,
        cap=300,
        already_dropped_genuine=once.genuine_dropped,
        already_dropped_injections=once.injections_dropped,
    )

    # Idempotent at the same cap: nothing further dropped, count unchanged.
    assert [t["id"] for t in twice.turns] == [t["id"] for t in once.turns]
    assert twice.genuine_dropped == once.genuine_dropped
    assert twice.total_dropped == once.total_dropped


def test_a_notice_journalled_by_the_previous_revision_is_not_replayed():
    """Forward compatibility with transcripts the broken revision wrote.

    Those notices were journalled as ordinary turns with
    ``compaction-elision-<count>`` ids. Replaying one would re-inject a
    synthetic message as if the user had written it, so they are dropped.
    """
    turns = [
        {"id": "compaction-elision-6", "text": "[6 older user message(s) ...]"},
        {"id": "real", "text": CONSTRAINT},
    ]

    capped = cap_preserved_user_turns(turns, cap=1_000_000)

    assert [t["id"] for t in capped.turns] == ["real"]


# ---------------------------------------------------------------------------
# Defect 3 — the read path heals, and stays equivalent to the live context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_matches_the_live_context_after_the_filter_and_cap(tmp_path):
    """Live and resumed contexts stay byte-identical.

    The equivalence other tests pin, re-asserted across the new filter AND the
    new cap: replay re-applies the cap the WRITING pass recorded rather than
    recomputing one it has no settings for.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 40))
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)
    await session.compact_now()
    await _inject_session_state(session)
    for index in range(3):
        await session.prompt(f"round two {index} " + "detail " * 30)
    await session.compact_now()

    live = [m.text for m in _preserved(session)]
    directory = session._transcript.directory
    await session.dispose()

    from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

    replayed = Transcript(directory).build_llm_history()
    resumed = [
        m.text
        for m in replayed
        if isinstance(m, Message) and (m.provider_payload or {}).get(PRESERVED_USER_TURN_KEY)
    ]
    assert resumed == live


@pytest.mark.asyncio
async def test_live_and_resume_match_when_the_cap_BINDS_and_a_turn_follows(tmp_path, monkeypatch):
    """The regression the pinned equivalence test structurally cannot catch.

    Three conditions have to hold at once for the defect to appear, and the
    older test satisfies none of them: the cap must actually BIND (so a notice
    is minted at all), a turn must FOLLOW the pass (so the turn-end persist
    pass runs), and the comparison must cover the WHOLE context rather than
    only the preserved block.

    What went wrong: the notice was minted as an ordinary ``Message`` in the
    live context, ``_is_persistable_message`` admitted every plain ``Message``,
    so the next boundary journalled it AFTER ``first_kept_entry_id`` — inside
    the replayed suffix — while replay ALSO synthesized it from the payload.
    Measured live 2 rows vs resumed 3. The journalled copy carried
    ``compaction_preserved`` too, so ``find_cut_point`` skipped it and the next
    pass folded its count into its own: the reported figure doubled on every
    resume, 6 -> 7,167 over ten cycles.
    """
    # A cap small enough that a handful of ordinary turns overflow it.
    monkeypatch.setattr(Session, "_preserved_turns_cap", lambda self, settings: 120)
    session = make_session(tmp_path, ScriptedStream(["reply"] * 60))
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(5):
        await session.prompt(f"question {index} " + "detail " * 30)

    assert (await session.compact_now()).ran is True

    notices = [m for m in _preserved(session) if m.id.startswith(PRESERVED_TURN_ELISION_ID_PREFIX)]
    assert notices, "test premise void: the cap did not bind, so no notice was minted"

    # The turn AFTER the pass is what triggers the persist that journalled it.
    await session.prompt("after the pass " + "detail " * 30)

    live = [
        (type(m).__name__, getattr(m, "role", None), (m.text or "")[:60])
        for m in session._context.messages
        if isinstance(m, Message)
    ]
    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    resumed = [
        (type(m).__name__, getattr(m, "role", None), (m.text or "")[:60])
        for m in replayed
        if isinstance(m, Message)
    ]
    assert resumed == live, "live and resumed contexts diverged once the cap bound"

    # And the notice exists exactly once on each side, not duplicated on resume.
    assert sum(1 for row in resumed if row[2].startswith("[")) == sum(
        1 for row in live if row[2].startswith("[")
    )

    # The journal must not contain the notice at all: it is synthesized from
    # the payload counts on both paths, so a stored copy is a duplicate by
    # construction.
    raw = (directory / "transcript.jsonl").read_text()
    assert PRESERVED_TURN_ELISION_ID not in raw, "the elision notice was journalled"


@pytest.mark.asyncio
async def test_the_reported_elision_count_does_not_inflate_across_resumes(tmp_path, monkeypatch):
    """The count must describe reality after N round trips.

    QA measured it doubling every resume — 6 -> 7,167 over ten cycles, a 231x
    overstatement — in the one message whose entire purpose is telling the
    model the truth about what it lost. The cause was the count living in a
    turn's id: that turn got journalled, replayed as history, and re-counted.
    """
    monkeypatch.setattr(Session, "_preserved_turns_cap", lambda self, settings: 120)
    session = make_session(tmp_path, ScriptedStream(["reply"] * 60))
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(5):
        await session.prompt(f"question {index} " + "detail " * 30)
    await session.compact_now()
    await session.prompt("after the pass " + "detail " * 30)
    directory = session._transcript.directory
    await session.dispose()

    def reported() -> list[str]:
        return [
            m.text
            for m in Transcript(directory).build_llm_history()
            if isinstance(m, Message) and (m.text or "").startswith("[")
        ]

    first = reported()
    assert first, "test premise void: no elision notice was rendered"
    # Ten replays of the same transcript must all report the same figure.
    for _ in range(10):
        assert reported() == first, "the reported elision count changed across resumes"


def test_the_heal_sheds_stored_injections_by_journal_provenance(tmp_path):
    """The read path must shed injections regardless of whether the cap binds.

    The previous revision applied only the cap on read and justified it with
    "injections are the oldest, so oldest-first eviction removes them". They
    are typically the NEWEST — ``session_state`` arrives every turn — so
    age-based eviction preferentially dropped the operator's constraints. And
    because most poisoned blocks sit BELOW the default cap, it never engaged at
    all: measured on the real fleet, 12% of stored injections were reached.
    After this change, 398 of 398 across the five named sessions.

    The block here is deliberately far below any cap, so only provenance can
    shed anything.
    """
    import json
    import time

    directory = tmp_path / "provenance-heal"
    directory.mkdir()
    kept = Message(role="user", content=[TextContent(text="the live turn")])
    rows = [
        {
            "id": kept.id,
            "ts": time.time(),
            "type": "message",
            "payload": json.loads(kept.model_dump_json()),
        }
    ]
    stored: list[dict[str, str]] = []
    # Genuine first, injections AFTER it — the real interleaving, and the one
    # oldest-first eviction gets backwards.
    stored.append({"id": "genuine-1", "text": CONSTRAINT})
    rows.append(
        {
            "id": "genuine-1",
            "ts": time.time(),
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [], "id": "genuine-1"},
        }
    )
    for index in range(6):
        turn_id = f"inj-{index}"
        stored.append({"id": turn_id, "text": STATE_TEXT})
        rows.append(
            {
                "id": turn_id,
                "ts": time.time(),
                "type": "message",
                "payload": {
                    "kind": "custom",
                    "custom_type": "session_state",
                    "details": {"text": STATE_TEXT},
                },
            }
        )
    rows.append(
        {
            "id": "compaction-1",
            "ts": time.time(),
            "type": "compaction",
            "payload": {
                "summary": "a summary",
                "first_kept_entry_id": kept.id,
                "tokens_before": 100,
                "preserved_user_turns": stored,
            },
        }
    )
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

    replayed = Transcript(directory).build_llm_history()
    block = [
        m
        for m in replayed
        if isinstance(m, Message) and (m.provider_payload or {}).get(PRESERVED_USER_TURN_KEY)
    ]
    texts = [m.text for m in block]

    assert not any(STATE_TEXT in text for text in texts), "stored injections survived the heal"
    assert any(CONSTRAINT in text for text in texts), "the heal dropped a genuine constraint"
    # The shed is reported, and as a harness drop rather than as the operator's.
    notice = next((t for t in texts if t.startswith("[")), None)
    assert notice is not None and "harness-injected" in notice
    assert "you wrote" not in notice


def test_a_legacy_record_is_healed_on_resume(tmp_path):
    """The fleet-wide half: a session already poisoned on disk must recover by
    RESUMING, not by hand-editing a transcript.

    A pre-fix record carries an unbounded block and no ``preserved_turns_cap``,
    so replay bounds it under the shipped default. Written as a raw transcript
    line because that is exactly what the poisoned sessions on disk contain.
    """
    import json
    import time

    directory = tmp_path / "legacy"
    directory.mkdir()
    kept = Message(role="user", content=[TextContent(text="the live turn")])
    # 400 turns of ~1k tokens each: ~400k, comfortably past the 100k default,
    # which is the scale the real session reached (362,780).
    bloated = [{"id": f"old{i}", "text": "word " * 800} for i in range(400)]
    lines = [
        json.dumps(
            {
                "id": kept.id,
                "ts": time.time(),
                "type": "message",
                "payload": json.loads(kept.model_dump_json()),
            }
        ),
        json.dumps(
            {
                "id": "compaction-1",
                "ts": time.time(),
                "type": "compaction",
                "payload": {
                    "summary": "a summary",
                    "first_kept_entry_id": kept.id,
                    "tokens_before": 900_000,
                    "preserved_user_turns": bloated,
                },
            }
        ),
    ]
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n")

    from local_operator.compaction.cutpoint import (
        DEFAULT_PRESERVED_TURN_CAP,
        PRESERVED_USER_TURN_KEY,
        _encode_len,
    )

    replayed = Transcript(directory).build_llm_history()
    preserved = [
        m
        for m in replayed
        if isinstance(m, Message) and (m.provider_payload or {}).get(PRESERVED_USER_TURN_KEY)
    ]

    assert len(preserved) < len(bloated), "a poisoned session stayed poisoned across a resume"
    total = sum(_encode_len(m.text) for m in preserved)
    assert total <= DEFAULT_PRESERVED_TURN_CAP
    # The newest constraints are what survived, and the drop is announced.
    assert any(f"old{len(bloated) - 1}" == m.id for m in preserved)
    assert any(m.id.startswith(PRESERVED_TURN_ELISION_ID_PREFIX) for m in preserved)
    # The kept window is untouched by the heal.
    assert "the live turn" in _texts(replayed)
