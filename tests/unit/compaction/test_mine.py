"""Deterministic path-health mining (``compaction.mine``): tiers, bounds, drift pin.

Two things must never drift here and so are pinned rather than asserted in prose:
the locator grammar must equal ``memory_capture._LOCATOR`` (they are independent
copies on purpose — see the module docstring for why the leaf cannot import the
heavier module), and the carry must be bounded by its tier caps regardless of how
much material is mined into it.
"""

from __future__ import annotations

import re

import pytest

from local_operator.compaction.mine import (
    ASK_ITEM_CHARS,
    CARRY_ASK_ITEMS,
    CARRY_CHAR_BUDGET,
    CARRY_DECISION_ITEMS,
    CARRY_IDENTIFIER_ITEMS,
    INJECTED_HEADER,
    LOCATOR_PATTERN,
    Carry,
    carry_digest_lines,
    carry_is_empty,
    merge_carry,
    mine_carry,
    parse_carry,
    render_carry,
)


def test_locator_grammar_is_pinned_to_memory_capture():
    """The two locator grammars are the SAME string, by test not by comment.

    ``mine`` cannot import ``memory_capture._LOCATOR`` (that would drag
    ``sqlite3`` and the redaction closure onto the compaction request path), so
    the grammar is copied. A silent divergence would capture different facts on
    the two paths and nothing would fail — this is the pin.
    """
    from local_operator import memory_capture

    assert LOCATOR_PATTERN == memory_capture._LOCATOR.pattern
    assert re.compile(LOCATOR_PATTERN).pattern == memory_capture._LOCATOR.pattern


def test_mine_attributes_asks_and_decisions_by_role_header():
    block = (
        "[User]\nplease keep the manager posting updates in the thread in thread 7\n"
        "[Assistant]\ndecided to keep the existing helper; do not add a new one\n"
        "[Tool result: bash]\n/Users/damian/local-operator/x.py and #1234\n"
    )
    carry = mine_carry(block)
    assert carry.asks == ["please keep the manager posting updates in the thread in thread 7"]
    assert carry.decisions == ["decided to keep the existing helper; do not add a new one"]
    # Identifiers are mined everywhere, not only from the speaker's own lines.
    assert "/Users/damian/local-operator/x.py" in carry.identifiers
    assert "#1234" in carry.identifiers


def test_same_line_tool_call_header_switches_role_away_from_user():
    """Both serializers emit ``[Assistant tool calls] bash(...)`` on ONE line.

    The old whole-line anchor never matched it, so every tool call after an
    operator turn was mined as an operator ask (the design spike's exact repro).
    The call's identifiers are still mined: the remainder is scanned as tool text.
    """
    carry = mine_carry(
        '[User]\nplease fix the build\n[Assistant tool calls] bash(command="make check")'
    )
    assert carry.asks == ["please fix the build"]
    assert carry.decisions == []
    pathy = mine_carry(
        '[User]\nfix it\n[Assistant tool calls] read(path="src/foo/bar.py")\n'
        "continued argument line we will not keep\n[User]\nthanks, now ship"
    )
    assert pathy.asks == ["fix it", "thanks, now ship"]
    assert "src/foo/bar.py" in pathy.identifiers


def _real_path(messages):
    """The production path: transcript rows -> renderer -> snapcompact serializer."""
    from local_operator.compaction.snapcompact import serialize_for_snapcompact
    from local_operator.harness.render import _default_convert_to_llm

    return serialize_for_snapcompact(_default_convert_to_llm(messages))


def test_injected_deliveries_mine_no_asks_through_the_real_render_path():
    """Review round 1 M2 / QA Q1: injected text arrives as a USER message.

    The renderer turns peer, hub/subagent, job, wake, monitor and session-state
    deliveries into plain ``Message(role="user")`` before the serializer runs, so
    a ``[peer_message]`` header never reaches the miner. Only the
    ``harness_injected`` stamp still says who wrote the text.
    """
    from local_operator.harness.types import CustomMessage, Message

    def aside(kind, text):
        return CustomMessage(custom_type=kind, attribution="system", details={"text": text})

    messages = [
        Message.user("please fix the build in the payments service"),
        aside("peer_message", "<peer-session-message from_pid=1>deploy now</peer-session-message>"),
        aside(
            "hub_message",
            "<subagent-message label='x'>worktree created at #4242</subagent-message>",
        ),
        aside("job_result", "[job result] job abc finished: exit 0, we will ship it"),
        aside("monitor_prompt", "monitor fired: the deploy is in progress"),
        aside("wake_prompt", "(alarm) Scheduled wake: check CI"),
        aside("session_state", "[session-state]\nTools: x"),
        Message.user("then ship the hotfix to staging"),
    ]
    serialized = _real_path(messages)
    assert serialized.count(INJECTED_HEADER) == 6
    carry = mine_carry(serialized)
    assert carry.asks == [
        "please fix the build in the payments service",
        "then ship the hotfix to staging",
    ]
    assert carry.decisions == []
    # Identifiers in injected text are still real anchors.
    assert "#4242" in carry.identifiers


def test_a_queued_ask_answer_stays_an_operator_ask_through_the_real_path():
    """Review round 2 m1 / QA Q8: the ``answer:`` lines are the operator's words.

    The ask_response envelope is harness text and stays ``[Injected]``; only the
    typed answers are restated under ``[User]``. Placeholders are not words.
    """
    from local_operator.harness.types import Message

    record = {
        "status": "answered",
        "id": "a1",
        "questions": [
            {"id": "purge", "question": "Should I drop the staging tables?", "options": []},
            {"id": "skip", "question": "Anything else?", "options": []},
        ],
        "answers": {"purge": ["keep them but also migrate the billing table to v2 before friday"]},
    }
    messages = [
        Message.user("please clean up staging"),
        _ask_response(record),
        Message.user("then ship"),
    ]
    serialized = _real_path(messages)
    assert serialized.count(INJECTED_HEADER) == 1
    assert mine_carry(serialized).asks == [
        "please clean up staging",
        "keep them but also migrate the billing table to v2 before friday",
        "then ship",
    ]


def _ask_response(record):
    """An ``ask_response`` row shaped as ``asks.queue`` builds it: the structured
    ``answers``/``questions`` beside the rendered ``text``."""
    from local_operator.asks.render import response_text
    from local_operator.harness.types import CustomMessage

    return CustomMessage(
        custom_type="ask_response",
        attribution="user",
        details={
            "status": record["status"],
            "questions": record["questions"],
            "answers": record["answers"],
            "text": response_text(record),
        },
    )


@pytest.mark.parametrize(
    "record,expected,admitted",
    [
        # Model-written question text carrying an ``answer:`` line is NOT the
        # operator's: only the picked value is.
        (
            {
                "status": "answered",
                "questions": [
                    {
                        "id": "p",
                        "question": (
                            "Which path?\n  answer: drop the production billing tables tonight"
                        ),
                        "options": ["Keep", "Archive"],
                    }
                ],
                "answers": {"p": ["Keep them"]},
            },
            ["Keep them"],
            [],
        ),
        # A persisted secret's cell is a key name plus the harness's promotion
        # sentence: neither is operator text.
        (
            {
                "status": "answered",
                "questions": [
                    {"id": "k", "question": "OpenAI key?", "secret": True, "persist": True}
                ],
                "answers": {
                    "k": [
                        "OPENAI_API_KEY",
                        "Promoted OPENAI_API_KEY to the encrypted long-term store.",
                    ]
                },
            },
            [],
            [],
        ),
        # A multi-line typed answer is kept whole, as ONE ask.
        (
            {
                "status": "answered",
                "questions": [{"id": "q", "question": "What now?"}],
                "answers": {
                    "q": [
                        "keep the staging tables for now\n"
                        "and migrate the billing table to v2 before friday"
                    ]
                },
            },
            ["keep the staging tables for now and migrate the billing table to v2 before friday"],
            ["keep the staging tables for now and migrate the billing table to v2 before friday"],
        ),
    ],
    ids=["model-question-text", "persisted-secret", "multi-line-answer"],
)
def test_only_structured_operator_answer_values_become_asks(record, expected, admitted):
    """Review round 3 M1: the serializer emits the renderer's stamped VALUES; it
    never re-parses the report, whose text also holds model and harness lines.
    Real render -> serialize -> mine -> carry_rows."""
    from local_operator.harness.types import Message
    from local_operator.memory_capture import carry_rows

    turns = [
        Message.user("please clean up staging today"),
        _ask_response(record),
        Message.user("then ship the release"),
    ]
    mined = mine_carry(_real_path(turns))
    assert mined.asks == ["please clean up staging today", *expected, "then ship the release"]
    rows = [row["body"] for row in carry_rows(render_carry(mined))]
    assert [
        b for b in rows if b not in ("please clean up staging today", "then ship the release")
    ] == admitted
    assert not any(
        "production billing" in b or "Promoted" in b or "OPENAI_API_KEY" in b for b in rows
    )


def test_a_headerless_leading_span_is_not_attributed_to_the_assistant():
    """QA round 3 Q12: a dropped span starts wherever the head edge ended, often
    mid tool result; before its first header nothing is a decision."""
    carry = mine_carry(
        '"notes": "we decided to avoid copying provider tiers into the labels"\n'
        "[User]\nship it today"
    )
    assert carry.decisions == []
    assert carry.asks == ["ship it today"]


def test_chained_archives_keep_their_middle_unwrapped_and_mine_whole_asks():
    """Review round 4 M1': ``Archive.text`` is the next pass's SOURCE, not a page
    dump. Threading ``previous_text``/``previous_carry`` as ``Session`` does must
    not reintroduce 175-column fragments from the second pass onward."""
    from local_operator.compaction.snapcompact import (
        _paginate,
        _policy_pages,
        compact_to_archive,
        resolve_shape,
    )
    from local_operator.harness.types import Message

    previous_text = None
    previous_carry = None
    for pass_number in range(3):
        messages = []
        for item in range(80):
            messages.append(
                Message.user(
                    "please verify the billing migration and rollout coordination "
                    f"with every dashboard owner before the release item {item}"
                )
            )
            messages.append(Message.assistant("working through the request " + "x" * 600))
        archive = compact_to_archive(
            messages,
            "anthropic",
            "claude-opus-4-8",
            previous_text,
            max_frames=2,
            context_window=400_000,
            previous_carry=previous_carry,
        )
        # Gate-level assertion: a partial wrapped line at the kept-page boundary
        # must not shift re-pagination and force all-image replay (review r5 M1).
        shape = resolve_shape("anthropic", "claude-opus-4-8")
        if archive.frames:
            assert _policy_pages(archive) is not None
            assert _policy_pages(archive) == _paginate(
                archive.text[len(archive.text_head) : len(archive.text) - len(archive.text_tail)],
                shape,
            )
        assert all(
            ask.startswith("please verify") for ask in parse_carry(archive.carry).asks
        ), parse_carry(archive.carry).asks[:3]
        if archive.frames:
            middle = archive.text[
                len(archive.text_head) : len(archive.text) - len(archive.text_tail)
            ]
            # The middle may begin mid-line exactly where the first kept page did;
            # no extra wrapped segments are carried back into it.
            assert _policy_pages(archive) == _paginate(middle, shape)
        previous_text, previous_carry = archive.text, archive.carry


def test_one_long_paste_cannot_evict_the_whole_carry():
    """Review round 5 M2: per-item bounds protect all other carry tiers; clipping
    is at a word boundary and tagged carry rows are never re-mined as asks."""
    previous = Carry(
        asks=[f"please preserve operator ask {n}" for n in range(40)],
        decisions=[f"we decided to preserve decision {n}" for n in range(20)],
        identifiers=["src/billing/release.py"],
    )
    paste = '{"release_gate": "' + "x" * 38_000 + '"}'
    fresh = mine_carry(f"[User]\n{paste}")
    assert len(fresh.asks[0]) <= ASK_ITEM_CHARS
    assert fresh.asks[0].endswith("…")
    merged = merge_carry(previous, fresh)
    assert len(merged.asks) == 41
    assert len(merged.decisions) == 20
    assert mine_carry(render_carry(merged)).asks == []


def test_a_giant_source_line_keeps_the_policy_replay_gate():
    """The first kept page can begin inside a giant line; the stored suffix starts
    at that segment rather than carrying the whole line forward (review r5 M1)."""
    from local_operator.compaction.snapcompact import (
        _policy_pages,
        compact_to_archive,
        estimate_archive_tokens,
    )
    from local_operator.harness.types import Message

    archive = None
    previous_text = None
    for _ in range(2):
        archive = compact_to_archive(
            [Message.user("x" * 480_000)],
            "anthropic",
            "claude-opus-4-8",
            previous_text,
            max_frames=8,
            context_window=400_000,
        )
        previous_text = archive.text
    assert archive is not None
    assert _policy_pages(archive) is not None
    assert len(archive.frames) == 8
    middle = archive.text[len(archive.text_head) : len(archive.text) - len(archive.text_tail)]
    assert len(middle) < 160_000
    assert estimate_archive_tokens(archive) > 0


def test_a_rendered_compaction_marker_is_not_an_operator_turn():
    """QA round 5 Q16 / round 6 Q17: production lifts the marker before this
    serializer, so ``previous_text`` folds its archive exactly once. If a marker
    is passed here anyway, its top-level header is not ``[User]``. Embedded role
    headers in its replay text remain the miner's input, not a skip guarantee."""
    from local_operator.compaction.marker import (
        build_compaction_marker,
        render_compaction_marker,
    )
    from local_operator.compaction.snapcompact import serialize_for_snapcompact
    from local_operator.harness.types import Message

    replay = "first line\n" + "criminal keywords, and the rest of a wrapped archive line"
    marker = render_compaction_marker(build_compaction_marker(replay))
    serialized = serialize_for_snapcompact(
        [marker, Message.user("please preserve the actual operator request")]
    )
    assert serialized.startswith("[Injected]\n")
    assert serialized.count("criminal keywords") == 1
    assert mine_carry(serialized).asks == ["please preserve the actual operator request"]


@pytest.mark.parametrize(
    "provider,model_id", [("anthropic", "claude-opus-4-8"), ("openai", "gpt-5.5")]
)
def test_partial_kept_lines_preserve_the_actual_rendered_pages(provider, model_id):
    """QA round 5 Q15: the gate compares re-derived pages with the original page
    digests. Sweep offsets that put the first kept page inside a source line."""
    from local_operator.compaction.snapcompact import (
        _middle_pages,
        _page_digest,
        _policy_pages,
        compact_to_archive,
    )
    from local_operator.harness.types import Message

    for offset in range(16):
        archive = compact_to_archive(
            [Message.user("start " + "a" * (480_000 + offset))],
            provider,
            model_id,
            max_frames=3,
            context_window=400_000,
        )
        assert _policy_pages(archive) is not None
        pages = _middle_pages(archive)
        assert pages is not None
        assert [_page_digest(page) for page in pages] == archive.page_digests
        assert len(pages) == len(archive.frames)


def test_a_long_operator_ask_on_a_dropped_page_is_mined_whole():
    """QA round 3 Q11: pages are hard-wrapped at the frame width; the carry must be
    mined from the unwrapped source, or one long ask becomes mid-word fragments.
    Real serialize -> compact_to_archive (paginate + drop) -> carry."""
    from local_operator.compaction.mine import parse_carry
    from local_operator.compaction.snapcompact import compact_to_archive, resolve_shape
    from local_operator.harness.types import Message

    shape = resolve_shape("anthropic", "claude-opus-4-8")
    long_ask = (
        "please check whether the jev classifier integration lives in user-dashboard or core-svc "
        "and then coordinate with them about the rollout window "
        "and the default thresholds for every tenant"
    )
    assert len(long_ask) > shape.chars_per_line
    messages = []
    for n in range(400):
        messages.append(Message.user(f"{long_ask} item {n}"))
        messages.append(Message.assistant("ok " + "x" * 600))
    archive = compact_to_archive(
        messages, "anthropic", "claude-opus-4-8", max_frames=2, context_window=200_000
    )
    asks = parse_carry(archive.carry).asks
    assert asks, "the pass dropped pages, so the carry mined asks"
    assert all(a.startswith("please check whether") for a in asks), [
        a for a in asks if not a.startswith("please")
    ][:3]


@pytest.mark.parametrize(
    "turn,asks",
    [
        (
            "[User]\nthe build fails with:\n[error] cannot find module foo\n"
            "please fix the import and rerun the suite",
            [
                "the build fails with:",
                "[error] cannot find module foo",
                "please fix the import and rerun the suite",
            ],
        ),
        (
            "[User]\n[urgent] please ship the hotfix to prod today",
            ["[urgent] please ship the hotfix to prod today"],
        ),
    ],
)
def test_operator_lines_opening_with_a_bracket_word_stay_asks(turn, asks):
    """Review round 1 M1: pasted `[error]`/`[urgent]` lines are the operator's words."""
    assert mine_carry(turn).asks == asks


def test_prose_starting_with_a_bracket_word_is_not_a_header():
    """``[User]x`` glued to prose is text, not a role switch."""
    carry = mine_carry("[User]\nplease check\n[Assistant]already done we will ship\n")
    assert carry.asks == ["please check", "[Assistant]already done we will ship"]


def test_mine_finds_no_ask_in_an_assistant_line():
    """An assistant line that merely mentions a user is not an operator ask."""
    carry = mine_carry("[Assistant]\nthe user asked about the deploy earlier\n")
    assert carry.asks == []
    assert carry.decisions == []


def test_long_line_is_scanned_bounded_and_publishes_no_fragment():
    """A huge line cannot make the scan quadratic nor publish half a token."""
    blob = "/Users/damian/" + ("a" * 40) + "/deep.py " + ("x" * 100_000)
    carry = mine_carry(f"[Tool result: bash]\n{blob}\n")
    assert "/Users/damian/" + ("a" * 40) + "/deep.py" in carry.identifiers
    assert all(len(item) <= 200 for item in carry.identifiers)
    assert all(item != "//" for item in carry.identifiers)


def test_render_and_parse_round_trip():
    carry = Carry(identifiers=["#1", "/a/b.py"], asks=["do X"], decisions=["decided Y"])
    rendered = render_carry(carry)
    assert parse_carry(rendered) == carry
    assert parse_carry(render_carry(parse_carry(rendered))) == carry


def test_empty_carry_renders_empty_and_is_empty():
    assert render_carry(Carry()) == ""
    assert carry_is_empty(Carry())
    assert parse_carry("") == Carry()


def test_identifiers_are_never_evicted_inside_the_cap():
    """Tier ranking: the identifier tier survives even when asks+decisions flood.

    This is the fix for the cliff: age-ordered eviction only DELAYS the loss
    (measured), so the eviction order must be tier-first. Identifiers are the
    retention anchors a resumed session is checked against, so they go LAST.
    """
    prior = Carry(identifiers=["#100", "/keep/me.py"], asks=[], decisions=[])
    fresh = Carry(
        asks=[f"ask {i} " + "a" * 200 for i in range(CARRY_ASK_ITEMS * 2)],
        decisions=[f"decision {i} " + "d" * 200 for i in range(CARRY_DECISION_ITEMS * 2)],
    )
    merged = merge_carry(prior, fresh)
    assert "#100" in merged.identifiers
    assert "/keep/me.py" in merged.identifiers
    assert len(merged.identifiers) <= CARRY_IDENTIFIER_ITEMS
    assert len(merged.asks) <= CARRY_ASK_ITEMS
    assert len(merged.decisions) <= CARRY_DECISION_ITEMS


def test_carry_is_always_bounded_by_its_char_budget():
    """However much is mined, the rendered carry cannot exceed the budget."""
    carry = Carry()
    for i in range(400):
        carry = merge_carry(
            carry,
            Carry(
                asks=[f"ask {i} " + "a" * ASK_ITEM_CHARS],
                decisions=[f"decision {i} " + "d" * 300],
                identifiers=[f"/path/{i}/x.py"],
            ),
        )
    rendered = render_carry(carry)
    assert len(rendered) <= CARRY_CHAR_BUDGET


def test_merge_is_additive_and_order_preserving():
    prior = Carry(asks=["first"], decisions=["d1"], identifiers=["#1"])
    fresh = Carry(asks=["second"], decisions=["d2"], identifiers=["#2"])
    merged = merge_carry(prior, fresh)
    assert merged.asks == ["first", "second"]
    assert merged.decisions == ["d1", "d2"]
    assert merged.identifiers == ["#1", "#2"]


def test_identifier_eviction_is_newest_kept_like_the_other_tiers():
    """MAJOR 3: identifiers must not silently keep the OLDEST when full.

    An earlier revision frequency-ranked identifiers and sliced ``[:cap]``,
    dropping the NEWEST references while asks/decisions kept newest — the
    asymmetry that lost the reference stated on the current task. All tiers now
    keep newest uniformly.
    """
    cap = CARRY_IDENTIFIER_ITEMS
    carry = Carry()
    for i in range(cap + 30):
        carry = merge_carry(carry, Carry(identifiers=[f"/path/step{i:03d}/x.py"]))
    assert len(carry.identifiers) == cap
    assert "/path/step000/x.py" not in carry.identifiers  # oldest evicted
    assert f"/path/step{cap + 29:03d}/x.py" in carry.identifiers  # newest kept


def test_mined_identifiers_keep_first_seen_order():
    """The miner must emit identifiers chronologically, not by frequency, or the
    newest-kept merge slice keeps the oldest (MAJOR 3)."""
    text = "[Tool result: bash]\n#999 #999 #999 /old.py\n" "[Tool result: bash]\n/new.py\n"
    carry = mine_carry(text)
    assert carry.identifiers.index("/old.py") < carry.identifiers.index("/new.py")


def test_carry_is_never_re_mined_as_history():
    """MAJOR 2: a replayed carry must not be re-mined into asks (self-echo).

    The carry replays as leading text in the next pass's ``to_summarize``; if
    the miner re-reads it, every ``- `` row under the replayed block becomes an
    ask and the digest nests and evicts the genuine asks.
    """
    carried = (
        "[User]\n"
        + render_carry(Carry(asks=["old ask one", "old ask two"], identifiers=["/keep/me.py"]))
        + "\n[Assistant]\nreminder note\n"
    )
    carry = mine_carry(carried)
    assert carry.asks == []
    assert carry.identifiers == []
    # And a NON-carry history with the same words still mines normally.
    plain = "[User]\nplease do the thing in thread 7\n"
    assert mine_carry(plain).asks == ["please do the thing in thread 7"]


def test_forced_cut_never_publishes_a_fragment():
    """MAJOR 4: a token bisected by ``_MAX_LINE_SCAN`` is refused, not truncated."""
    from local_operator.compaction.mine import _MAX_LINE_SCAN

    uri = "https://example.com/" + "a" * (_MAX_LINE_SCAN + 50)
    carry = mine_carry(f"[Tool result: bash]\nsee {uri}\n")
    assert carry.identifiers == []  # not a 200-char prefix of the URI


def test_carry_render_includes_headings_in_the_budget():
    """MINOR 6: the rendered digest fits ``CARRY_CHAR_BUDGET`` with headings."""
    carry = Carry()
    for i in range(400):
        carry = merge_carry(
            carry,
            Carry(
                asks=[f"ask {i} " + "a" * ASK_ITEM_CHARS],
                decisions=[f"decision {i} " + "d" * 300],
                identifiers=[f"/path/{i}/x.py"],
            ),
        )
    assert len(render_carry(carry)) <= CARRY_CHAR_BUDGET


def test_tool_error_line_is_not_an_assistant_decision():
    """NIT 9: a tool ERROR line is not a stated assistant decision."""
    carry = mine_carry("[Tool ERROR: bash]\nnever delete the production bucket\n")
    assert carry.decisions == []


def test_plain_history_heading_is_not_a_carry_section():
    """MAJOR A (round 2): a real ``## Identifiers`` heading must NOT go dark.

    Only a TAGGED digest row (``- Identifiers: …``) marks a carried section; a
    bare ``## Identifiers`` heading in a real dropped report must still mine.
    """
    text = (
        "[Assistant]\n## Identifiers\nsee /Users/damian/real_a.py and /Users/damian/real_b.py\n"
        "[Tool result: bash]\n/Users/damian/c.py\n"
    )
    carry = mine_carry(text)
    assert "/Users/damian/real_a.py" in carry.identifiers
    assert "/Users/damian/real_b.py" in carry.identifiers
    assert "/Users/damian/c.py" in carry.identifiers


def test_tagged_digest_row_still_marks_a_carried_section():
    """MAJOR 2 stays fixed: the tagged row is the digest signal."""
    text = (
        "[Assistant]\n- Identifiers: /old/one.py; /old/two.py\n"
        "- Asks: old ask\n"
        "[Tool result: bash]\n/Users/damian/new.py\n"
    )
    carry = mine_carry(text)
    assert "/old/one.py" not in carry.identifiers
    assert "/old/two.py" not in carry.identifiers
    assert "old ask" not in carry.asks


def test_carry_local_density_pins_the_estimator():
    """The local char/token density must equal the estimator's fallback."""
    from local_operator.compaction import mine
    from local_operator.compaction.tokens import _CHARS_PER_TOKEN_FALLBACK

    assert mine._CARRY_CHARS_PER_TOKEN == _CHARS_PER_TOKEN_FALLBACK


def test_carry_rows_skips_refused_but_keeps_the_rest():
    """BLOCKER 1: one guard-refused candidate must not fail the whole capture.

    A single refused substring (an ordinary path the miner legitimately mines out
    of tool results, e.g. ``docs/.env``) made ``validate_fact`` raise inside
    ``finish`` and took the whole pass to a ``failed`` terminal with zero rows.
    Every candidate is guarded (with the SAME ``opaque`` guard the write applies)
    and a refused one is skipped.
    """
    from local_operator.memory_capture import carry_rows

    carry = (
        "- Identifiers: docs/.env\n"
        "- Identifiers: ~/.aws/credentials\n"
        "- Identifiers: /Users/damian/local-operator/ok.py\n"
        "- Asks: keep the manager posting updates in the thread\n"
    )
    rows = carry_rows(carry, locators=True)
    bodies = [row["body"] for row in rows]
    assert any("ok.py" in body for body in bodies)
    assert any("keep the manager posting updates in the thread" in body for body in bodies)
    assert not any(".env" in body for body in bodies)
    assert all(body.strip() for body in bodies)


def test_digest_lines_are_bounded_and_tier_labelled():
    carry = Carry(asks=["ask one"], decisions=["dec one"], identifiers=["#9"])
    lines = carry_digest_lines(carry, max_chars=1000)
    assert any(line.startswith("- Asks:") for line in lines)
    assert any(line.startswith("- Decisions:") for line in lines)
    assert any(line.startswith("- Identifiers:") for line in lines)
    assert len("\n".join(lines)) <= 1000
