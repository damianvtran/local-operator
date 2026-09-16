"""The two phone folds must agree about the same row.

WHY THIS FILE EXISTS
--------------------
``mobile/projection.py`` used to contain two independent folds over
``AgentMessage`` — one for lazy-loaded history pages, one for the attach seed
— each carrying a comment asserting the other could not disagree with it.
They disagreed twelve ways (``docs/design/history-fold-convergence.md`` §3).
The worst was a hub steer rendering as a clean ``parent_message`` card on a
scrolled-up page and leaking the raw ``<parent-message>`` XML envelope as a
``notice`` on attach: the phone contradicted ITSELF one scroll gesture apart.

Nothing caught that, because ``test_projection.py`` pins each fold
SEPARATELY and never runs two folds over one input. A contract asserted by
comment is a contract nothing checks. These tests run the folds over the same
history and compare, so the next divergence fails here rather than shipping.

WHAT THIS FILE DOES **NOT** DO
------------------------------
It does not compare the phone against the TUI. Every assertion here is
phone-fold vs phone-fold, or pins one fold's output against a literal. The
cross-SURFACE contract is enforced structurally instead: the row decisions
both hosts share live in ``harness/rows.py`` and each host calls them, so
there is no second implementation left to disagree with — which is why the
convergence work moved those decisions out of the hosts rather than adding a
test to watch two copies stay in step.

That structural guarantee stops at the module boundary. A host can still
misuse a shared helper (feed it differently-normalized text, ignore a field
it emits), and nothing in this file would catch it. Closing THAT gap needs a
test mounting the real ``OperatorApp`` through ``_project_settled_rows`` over
this corpus and comparing rendered row kinds and text against the phone's,
with the legitimate mounting differences (the TUI's dedicated ``WakeBlock``
against the phone's tagged notice) named in an allowance table.

An earlier version of this docstring claimed that comparison and that table
already existed. Neither did — no test here ever built a TUI row, and no
``TUI_ALLOWANCES`` was ever defined. The claim is recorded as absent rather
than quietly deleted because a docstring promising a guarantee the file does
not provide is worse than no docstring: it tells the next reader a whole
class of defect is already covered.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.cutpoint import (
    PRESERVED_USER_TURN_KEY,
    RENDERED_INJECTION_KEY,
)
from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
from local_operator.harness.comms import (
    PARENT_MESSAGE_CLOSE_TAG,
    PARENT_MESSAGE_TAG,
    TO_CHILD_INSTRUCTIONS,
)
from local_operator.harness.loop import (
    LENGTH_ENDED_CALL_RESULT_TEXT,
    TRUNCATED_RESULT_TEXT,
)
from local_operator.harness.message_types import HUB_MESSAGE_TYPE
from local_operator.harness.rows import (
    assistant_row_text,
    assistant_stop_notice,
    harness_chrome_prompts,
    is_harness_chrome,
    user_row_text,
    wake_receipt_headline,
)
from local_operator.harness.types import (
    OUTPUT_LIMIT_ARGUMENTS,
    OUTPUT_LIMIT_KEY,
    OUTPUT_LIMIT_TURN,
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.incidents import format_model_switch_message
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.session.shell_record import shell_record_messages
from local_operator.session.transcript import ENTRY_MESSAGE
from local_operator.session.transcript import TranscriptEntry as JournalEntry
from local_operator.session.transcript import replay_entries


def _page_rows(history: Sequence[AgentMessage]) -> list[TranscriptEntry]:
    """The lazy-loaded history page's rows."""
    return fold_messages_to_entries(list(history))


def _attach_rows(history: Sequence[AgentMessage]) -> list[TranscriptEntry]:
    """The attach seed's rows."""
    fold = ProjectionFold(SessionProjection(session_id="s", pid=1))
    fold.fold_history(list(history))
    return list(fold.projection.transcript)


def _shape(rows: list[TranscriptEntry]) -> list[tuple[Any, ...]]:
    """The comparable shape of a row list: everything a reader would notice."""
    return [
        (
            row.kind,
            row.text,
            row.tool_name,
            row.tool_state,
            row.details.get("severity", ""),
            bool(row.details.get("user_run")),
        )
        for row in rows
    ]


def _envelope(body: str, kind: str = "steer") -> str:
    """Exactly what ``SubagentComms._format_to_child`` emits."""
    return (
        f"{PARENT_MESSAGE_TAG}\n{TO_CHILD_INSTRUCTIONS[kind]}\n\n{body}\n{PARENT_MESSAGE_CLOSE_TAG}"
    )


def _hub_steer(body: str) -> CustomMessage:
    """Exactly what ``SubagentComms`` journals for a parent steer."""
    return CustomMessage(
        custom_type=HUB_MESSAGE_TYPE,
        attribution="user",
        details={
            "direction": "to_child",
            "body": body,
            "expects_reply": False,
            "steer": True,
            "text": _envelope(body),
        },
    )


def _switch_notice() -> str:
    """The failover notice ``journal_model_switch`` renders, verbatim.

    Built by the real producer rather than hand-written: the display rule is
    about the STAMP, so a test that keyed on remembered wording would pass
    while a reworded notice leaked.
    """
    return format_model_switch_message(
        "zai/glm-5.3",
        "anthropic/claude-opus-5",
        reason="anthropic quota exhausted (0% remaining)",
        transient=True,
    )


def _injected_notice(text: str | None = None) -> Message:
    """Exactly what ``_injected_user_message`` mints: a stamped user row."""
    return Message(
        role="user",
        content=[TextContent(text=_switch_notice() if text is None else text)],
        provider_payload={RENDERED_INJECTION_KEY: True},
    )


def _carried_notice() -> Message:
    """The legacy shape: a notice a compaction block carried forward.

    Written before the stamp existed, so it is re-seated with
    ``compaction_preserved`` and no stamp at all — QA measured eight of these on
    the operator's own session, painted behind the user gutter twice each.
    """
    return Message(
        role="user",
        content=[TextContent(text=_switch_notice())],
        provider_payload={PRESERVED_USER_TURN_KEY: True},
    )


def _assistant(text: str = "", calls=(), stop=None, payload=None) -> Message:
    message = Message(
        role="assistant",
        content=[TextContent(text=text)] if text else [],
        stop_reason=stop,
        provider_payload=payload or {},
    )
    message.tool_calls = list(calls)
    return message


#: Every conversation shape the convergence review enumerated, keyed by the
#: divergence it closes. Built from the REAL producers wherever one exists
#: (``shell_record_messages``, the comms envelope constants) rather than from
#: hand-written approximations, because a hand-written shape tests the wrong
#: path — an envelope missing its exact instruction preamble does not extract.
CORPUS: dict[str, Sequence[AgentMessage]] = {
    "D1 hub steer": [Message.user("start"), _hub_steer("focus on the parser")],
    "D1 persisted envelope": [Message.user(_envelope("focus on the parser"))],
    "D2 refusal": [
        Message.user("do it"),
        _assistant(
            "I started to answer but", stop="refusal", payload={"refusal": "content policy"}
        ),
    ],
    "D3 failed turn": [Message.user("do it"), _assistant(stop="error")],
    "D4 interrupted turn": [Message.user("do it"), _assistant(stop="aborted")],
    "D5 gate timeout": [
        CustomMessage(
            custom_type=GATE_TIMEOUT_CUSTOM_TYPE,
            details={"tool": "bash", "description": "rm -rf /x", "waited_s": 7200},
        )
    ],
    "D6 wake": [
        CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            details={"text": "(alarm) build finished", "wake_id": "w1", "occurrence": 1},
        )
    ],
    "D7 compaction refused": [
        CustomMessage(
            custom_type=COMPACTION_REFUSED_TYPE,
            details={"detail": "compaction skipped: history too short"},
        )
    ],
    "D9 chrome prompts": [Message.user(p) for p in harness_chrome_prompts()],
    "D10 unanswered call": [
        _assistant("reading", calls=[ToolCall(id="t9", name="read", arguments={"path": "/a"})])
    ],
    "D11 bang mode": shell_record_messages(
        "ls -la",
        ToolResult(tool_call_id="sh1", content=[TextContent(text="total 0")], is_error=False),
    ),
    "D12 harness injection": [Message.user("why did the model change?"), _injected_notice()],
    "D13 carried notice": [Message.user("why did the model change?"), _carried_notice()],
    "settled conversation": [
        Message.user("edit it"),
        _assistant("editing", calls=[ToolCall(id="e1", name="edit", arguments={"path": "/x"})]),
        Message(
            role="tool",
            content=[TextContent(text="done")],
            tool_call_id="e1",
            tool_name="edit",
            provider_payload={"duration_s": 9.75, "details": {"added": 3, "removed": 1}},
        ),
    ],
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_the_two_phone_folds_agree_about_every_row(name: str) -> None:
    """The attach seed and a history page must render one history identically.

    This is the property D1 violated: the phone contradicted itself across
    one scroll gesture. It holds now because there is only one fold — this
    test is what keeps it that way if someone reintroduces a second.
    """
    history = CORPUS[name]

    assert _shape(_page_rows(history)) == _shape(_attach_rows(history))


def test_the_attach_seed_never_leaks_the_model_facing_envelope() -> None:
    """D1's headline: the raw XML wrapper must never reach a user surface.

    ``extract_parent_message`` exists to hide it, and the attach seed used to
    render it verbatim inside a notice row.
    """
    history = [_hub_steer("focus on the parser"), Message.user(_envelope("and retries"))]

    for rows in (_page_rows(history), _attach_rows(history)):
        rendered = " ".join(row.text for row in rows)
        assert PARENT_MESSAGE_TAG not in rendered
        assert PARENT_MESSAGE_CLOSE_TAG not in rendered
        assert [row.kind for row in rows] == ["parent_message", "parent_message"]
        assert [row.text for row in rows] == ["focus on the parser", "and retries"]


@pytest.mark.parametrize(
    ("stop_reason", "payload", "expected"),
    [
        ("refusal", {"refusal": "content policy"}, "content policy"),
        ("refusal", {}, "model refused the request (no details recorded)"),
        ("error", {}, "turn failed"),
        ("aborted", {}, "interrupted"),
    ],
)
def test_a_turn_that_did_not_finish_says_so_on_the_phone(
    stop_reason: str, payload: dict[str, Any], expected: str
) -> None:
    """D2-D4: the phone had NO ``stop_reason`` branch, so a refused turn read
    as a complete (oddly short) answer and a failed turn as the agent
    ignoring the user."""
    history = [Message.user("do it"), _assistant(stop=stop_reason, payload=payload)]

    rows = _page_rows(history)

    assert rows[-1].kind == "notice"
    assert rows[-1].text == expected
    assert rows[-1].details["severity"] == "error"


def test_a_refusal_shows_even_when_the_model_streamed_prose_first() -> None:
    """A provider safety stop often cuts a PARTIAL answer. The prose alone
    reads as a complete reply, so the notice fires alongside it rather than
    instead of it."""
    history = [
        _assistant("Here is the first half", stop="refusal", payload={"refusal": "content policy"})
    ]

    rows = _page_rows(history)

    assert [row.kind for row in rows] == ["assistant", "notice"]
    assert rows[0].text == "Here is the first half"


def test_an_ordinary_turn_gets_no_notice() -> None:
    """Negative control for the ``stop_reason`` branch: a normal answer and a
    normal tool call must not acquire a notice row."""
    history = [
        Message.user("edit it"),
        _assistant("editing", calls=[ToolCall(id="e1", name="edit", arguments={"path": "/x"})]),
    ]

    assert [row.kind for row in _page_rows(history)] == ["user", "assistant", "tool"]


def test_an_unattended_gate_timeout_reaches_the_phone() -> None:
    """D5: the most expensive event in the detached feature — up to a day of
    held residency — rendered nowhere on the phone."""
    history = [
        CustomMessage(
            custom_type=GATE_TIMEOUT_CUSTOM_TYPE,
            details={"tool": "bash", "description": "rm -rf /x", "waited_s": 7200},
        )
    ]

    row = _page_rows(history)[0]

    assert row.kind == "notice"
    assert row.text == (
        "waited 2h for approval with nobody attached, then denied it — bash · rm -rf /x"
    )
    assert row.details["severity"] == "warning"


def test_an_unanswered_ask_says_it_moved_on_rather_than_denied() -> None:
    """An ``ask`` is a QUESTION, and an unanswered question was not denied —
    the severity vocabulary must not be borrowed from the approval gate."""
    history = [
        CustomMessage(
            custom_type=GATE_TIMEOUT_CUSTOM_TYPE,
            details={"tool": "ask", "description": "which env?", "waited_s": 45, "kind": "ask"},
        )
    ]

    assert "moved on" in _page_rows(history)[0].text


@pytest.mark.parametrize(
    ("detail", "severity"),
    [
        ("compaction skipped: history too short", "warning"),
        ("compaction failed: provider error", "error"),
    ],
)
def test_a_refused_compaction_keeps_its_severity_on_the_phone(detail: str, severity: str) -> None:
    """D7: the phone dropped this row entirely, and flattening the two cases
    into one ink would lose the difference between "not worth it" and "I
    could not" — which is what the reader decides their next step on."""
    history = [CustomMessage(custom_type=COMPACTION_REFUSED_TYPE, details={"detail": detail})]

    row = _page_rows(history)[0]

    assert row.kind == "notice"
    assert row.details["severity"] == severity


def test_an_unanswered_tool_call_is_not_asserted_successful() -> None:
    """D10: a call whose result never arrived rendered ✓ on the phone while
    the TUI showed it interrupted. A default that means "success" is how this
    bug class keeps recurring."""
    history = [
        _assistant("reading", calls=[ToolCall(id="t9", name="read", arguments={"path": "/a"})])
    ]

    assert _page_rows(history)[-1].tool_state == "interrupted"


def test_the_bare_transcript_entry_default_is_not_success() -> None:
    """The root cause of D10, pinned at the type: a row nobody set a state on
    has no observed outcome, and must not claim one."""
    assert TranscriptEntry(id="x", kind="tool").tool_state == "interrupted"


def test_a_paired_call_still_settles_done() -> None:
    """Negative control for D10: the interrupted default must not leak into a
    call that DID return."""
    history = [
        _assistant("editing", calls=[ToolCall(id="e1", name="edit", arguments={"path": "/x"})]),
        Message(
            role="tool", content=[TextContent(text="done")], tool_call_id="e1", tool_name="edit"
        ),
    ]

    assert _page_rows(history)[-1].tool_state == "done"


def test_a_skill_invocation_shows_the_typed_line_not_the_payload() -> None:
    """D8: a ``$skill`` persists as its EXPANDED payload, and painting it
    verbatim put the whole SKILL.md body in the user's bubble — and titled
    the session after it."""
    from local_operator.skills.discovery import Skill
    from local_operator.skills.invoke import SkillInvocation, render_invocation

    # The REAL Skill type and the REAL renderer: a stub here would test a
    # payload shape the product never produces, and it is the product's own
    # ``invocation`` attribute that ``typed_line_of`` reads back.
    skill = Skill(
        name="research",
        description="research things",
        file_path=Path("/skills/research/SKILL.md"),
        base_dir=Path("/skills/research"),
        source="user",
    )
    typed = "$research the widget market"
    payload = render_invocation(
        SkillInvocation(skill=skill, request="the widget market", token="$research", typed=typed),
        "# Research\n\n" + "a very long skill body\n" * 50,
    )
    assert len(payload) > 500

    rows = _page_rows([Message.user(payload)])

    assert rows[0].kind == "user"
    assert rows[0].text == typed


def test_ordinary_prose_mentioning_a_skill_is_left_alone() -> None:
    """Negative control for D8: only a rendered payload collapses."""
    rows = _page_rows([Message.user("what does the $research skill do?")])

    assert rows[0].text == "what does the $research skill do?"


def test_a_harness_injected_row_is_never_painted_as_the_users_words() -> None:
    """A row the harness minted from a ``CustomMessage`` is not the user's words.

    The transient failover notice is the reported case: a compaction pass baked
    it into the rebuilt context as a plain user row (the root-cause fix is in
    ``Session._render_for_compaction``), and it is still on disk in every
    session an older build wrote — so the DISPLAY decision has to hold for
    rows already in a transcript, not only for new ones. Both folds, because
    the phone's own bare ``role == "user"`` test is exactly how the two
    surfaces drift apart.
    """
    notice = _switch_notice()
    history = [Message.user("why did the model change?"), _injected_notice(notice)]

    for rows in (_page_rows(history), _attach_rows(history)):
        assert [row.kind for row in rows] == ["user"]
        assert rows[0].text == "why did the model change?"
        assert notice not in " ".join(row.text for row in rows)

    # …and the same decision covers the LEGACY shape: a notice a compaction
    # block carried forward from before the stamp existed (QA Q1, measured on
    # the operator's own session). Both folds, again.
    carried = [Message.user("why did the model change?"), _carried_notice()]
    for rows in (_page_rows(carried), _attach_rows(carried)):
        assert [row.kind for row in rows] == ["user"]
        assert notice not in " ".join(row.text for row in rows)

    # The LIMIT of the rule, pinned here because it is a trade and not a free
    # win: the same wording with no stamp and no carried marker — a pasted notice,
    # a realistic prompt — is ALSO hidden on both folds, because the audit phase
    # serves stored rows whose only surviving evidence is the text (QA round 2 Q1
    # found four such rows painted on the operator's session). The row is not lost
    # anywhere else; only the renderer drops it, exactly as the chrome prompts
    # above already do.
    quoted = [Message(role="user", content=[TextContent(text=notice)])]
    assert _page_rows(quoted) == []
    assert _attach_rows(quoted) == []


def test_a_stored_notice_row_is_hidden_in_the_audit_phase_too() -> None:
    """QA round 2 Q1: the heal must not open the mirror.

    Shedding the carried copies removed their ids from the hoisted suppression
    set, so the plain stored rows an older build wrote — no stamp, no marker,
    served verbatim by the audit phase — came back into view. The decision is
    therefore text-based for any ``role="user"`` row, in whichever phase serves
    it, and this pins the audit arm of that: a stored notice replayed through
    ``replay_entries(..., mode="audit")`` paints nothing on either fold while the
    row itself stays in the journal.
    """
    notice = _switch_notice()

    def journal_row(entry_id: str, text: str) -> JournalEntry:
        """A stored row as the JOURNAL holds it: no ``provider_payload`` at all."""
        return JournalEntry(
            id=entry_id,
            ts=1.0,
            type=ENTRY_MESSAGE,
            payload={
                "kind": "message",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        )

    entries = [
        journal_row("stored-notice", notice),
        journal_row("mine", "why did the model change?"),
    ]

    replayed = replay_entries(entries, None, mode="audit")
    assert "stored-notice" in [getattr(row, "id", "") for row in replayed]

    for rows in (_page_rows(replayed), _attach_rows(replayed)):
        assert [row.kind for row in rows] == ["user"]
        assert rows[0].text == "why did the model change?"


def test_no_harness_prompt_is_painted_as_the_users_words() -> None:
    """D9: the phone suppressed ONE of the three continuation prompts and
    rendered the other two as the user's own words — a partially copied
    list, which is the drift signature itself. All three now come from one
    shared list."""
    prompts = harness_chrome_prompts()
    assert len(prompts) == 3

    assert _page_rows([Message.user(p) for p in prompts]) == []


def test_every_connectivity_instruction_shape_is_chrome() -> None:
    """Round-1 M1: the shared list only held the PROSE continuation prompt.

    ``_continuation_instruction`` composes the persisted instruction per cut, so
    the incident's own shape — partial prose with a call still being dictated —
    is prose + a space + the tool-call half, and a prose-less cut gets the
    tool-call half alone. Neither was an exact member of
    ``harness_chrome_prompts()``, so a resumed session painted harness words as
    the operator's own on both surfaces. Built from the producer, not typed out,
    so the shapes cannot drift from the strings the loop actually persists.
    """
    from local_operator.harness.loop import _continuation_instruction

    call = ToolCall(name="write", raw_arguments='{"path": "/tmp/notes", "content": "hel')
    shapes = {
        "prose only": _continuation_instruction(resumable_text=True, interrupted=[]),
        "tool-call only": _continuation_instruction(resumable_text=False, interrupted=[call]),
        "composed": _continuation_instruction(resumable_text=True, interrupted=[call]),
    }
    multi = [
        ToolCall(name="write", raw_arguments="{"),
        ToolCall(name="shell", raw_arguments="{"),
    ]
    shapes["composed, two tools"] = _continuation_instruction(
        resumable_text=True, interrupted=multi
    )

    for label, text in shapes.items():
        assert is_harness_chrome(text), f"{label} must not be painted as the user's words"
        assert _page_rows([Message.user(text)]) == [], label


@pytest.mark.parametrize(
    "resembling",
    [
        # An operator QUOTING the instruction — asking about a log line is a
        # realistic thing to do — must keep their own row, so the recogniser
        # matches whole shapes and not a distinctive prefix.
        "why does the transcript say: " + harness_chrome_prompts()[2],
        "[system] A tool call (write) was aborted by the network interruption "
        "before it finished, so it never ran. If you still need that action, "
        "issue the call again from scratch. ok?",
    ],
)
def test_an_operator_message_resembling_the_instruction_is_not_swallowed(
    resembling: str,
) -> None:
    """Negative control for the recogniser above.

    The pre-fix rule was exact membership of harness-minted constants, and the
    extension must not turn "the operator typed something similar" into
    "the harness said it": a swallowed operator turn is invisible and
    unrecoverable, which is worse than the leak it fixes.
    """
    assert not is_harness_chrome(resembling)

    rows = _page_rows([Message.user(resembling)])

    assert [row.kind for row in rows] == ["user"]
    assert rows[0].text == resembling


def test_a_human_quoting_the_envelope_keeps_their_own_words() -> None:
    """Negative control for D1/D9 suppression: asking about the wrapper is a
    realistic thing to do, and must not be rewritten as a parent steer."""
    quoted = f"{PARENT_MESSAGE_TAG}\nwhy does my log show this?\n{PARENT_MESSAGE_CLOSE_TAG}"

    rows = _page_rows([Message.user(quoted)])

    assert rows[0].kind == "user"
    assert rows[0].text == quoted


def test_a_bang_mode_command_opens_expanded() -> None:
    """D11: the user typed this command and is waiting to read its output, so
    the card opens rather than asking for a tap to reveal what they asked
    for. Built from the real ``shell_record_messages`` shape."""
    history = shell_record_messages(
        "ls -la",
        ToolResult(tool_call_id="sh1", content=[TextContent(text="total 0")], is_error=False),
    )

    rows = _page_rows(history)

    assert rows[0].text == "! ls -la"
    assert rows[-1].details["user_run"] is True


def test_a_model_issued_call_does_not_open_expanded() -> None:
    """Negative control for D11: only the user's own command self-opens."""
    history = [
        Message.user("list the files"),
        _assistant(calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})]),
    ]

    assert not _page_rows(history)[-1].details.get("user_run")


def test_a_wake_receipt_strips_the_model_facing_envelope() -> None:
    """D6/design D3: the phone rendered ``wake.py``'s payload VERBATIM.

    The envelope — ``(alarm) Scheduled wake w-9 (1, every 6h) — cancel with
    wake({op:"cancel",id:"w-9"})`` — is markup addressed to the model. The
    TUI stripped it inside ``WakeBlock._summary``, so the strip was
    unreachable from the phone and the raw JSON-ish cancel instruction landed
    on a human surface: the same defect class as D1's leaked
    ``<parent-message>`` rows.
    """
    history = [
        CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            details={
                "text": (
                    "(alarm) Scheduled wake w-9 (1, every 6h) — cancel with "
                    'wake({op:"cancel",id:"w-9"})\n\nCheck the deploy pipeline'
                ),
                "wake_id": "w-9",
            },
        )
    ]

    for rows in (_page_rows(history), _attach_rows(history)):
        assert [row.kind for row in rows] == ["notice"]
        row = rows[0]
        assert row.details["notice_kind"] == "wake"
        # The cancel how-to and the (alarm) prefix are for the model.
        assert "cancel with wake(" not in row.text
        assert "(alarm)" not in row.text
        # What the user needs: which wake fired, and what it delivered.
        assert row.text == "w-9 (1, every 6h) — Check the deploy pipeline"


def test_a_length_stop_is_announced_on_both_surfaces() -> None:
    """Agent review round 1 (B1): nothing folded ``stop_reason == "length"`` into
    a notice, so a reply cut by the generation bound replayed as a complete one.

    Both variants are asserted because they need opposite treatment: the turn
    WITH prose is the one whose text lies (it reads as a finished, oddly short
    answer), and the turn with NOTHING still needs a line because there is no
    text to explain the silence. The tier is ``warning`` on both, matching the
    live loop's own truncation notices, so one event is never described in two
    voices by the two surfaces that render it.
    """
    cut = assistant_stop_notice(
        text="1, 2, 3, 4", has_tool_calls=False, stop_reason="length", provider_payload=None
    )
    assert cut == ("answer cut off at the output limit", "warning")

    empty = assistant_stop_notice(
        text="   ", has_tool_calls=False, stop_reason="length", provider_payload=None
    )
    assert empty == ("no answer: the model spent its whole output budget", "warning")

    # A truncated TOOL CALL produced something, but not an ANSWER, so it takes
    # its own arm rather than the content one: the call card directly above
    # already says what happened to the call, and repeating "answer cut off"
    # there was a second, false row for one event (design round 1, D3).
    #
    # The line is ARM-NEUTRAL when the caller cannot tell which arm the limit
    # was in, which is what the call above does: it passes no arm at all (design
    # round 1, D1). The two arms' lines are asserted in
    # ``test_the_length_notice_reads_the_arm_off_the_turns_own_results``.
    with_call = assistant_stop_notice(
        text="", has_tool_calls=True, stop_reason="length", provider_payload=None
    )
    assert with_call == ("turn cut off at the output limit — nothing ran", "warning")

    # Prose and a cut call together is the content arm: there IS an answer, and
    # the live loop agrees -- it tests ``has_text`` before ``tool_calls`` too
    # (design round 2, D7). It used to check the call first, so this same turn
    # was "mid tool call" live and "answer cut off" here. The call half still
    # reaches the reader, on its own row: the placeholder result appended for it
    # says it was cut and nothing ran.
    both = assistant_stop_notice(
        text="here is the file", has_tool_calls=True, stop_reason="length", provider_payload=None
    )
    assert both == ("answer cut off at the output limit", "warning")

    # An ordinary stop still needs nothing, which is what keeps the notice
    # meaningful rather than decorative.
    assert (
        assistant_stop_notice(
            text="done", has_tool_calls=False, stop_reason="stop", provider_payload=None
        )
        is None
    )

    # And the phone's fold actually renders it, on the same history the TUI
    # would replay: the defect was invisible on BOTH surfaces, so the helper
    # being right is not on its own the claim.
    history = [Message.user("count to a million"), _assistant("1, 2, 3", stop="length")]
    page = [row.kind for row in _page_rows(history)]
    assert page == ["user", "assistant", "notice"]
    notice_row = _page_rows(history)[-1]
    assert "output limit" in notice_row.text


def _limit_turn(arm: str | None) -> list[AgentMessage]:
    """One length-stopped turn as the harness persists it, arm included.

    The call the model was still dictating plus the SYNTHETIC result the loop
    pairs it with (``_synthetic_result``'s shape). Built from the real
    constants and the real marker, so this pins the CONTRACT — marker to
    receipt to notice line — rather than a remembered wording.
    """
    call = ToolCall(id="c_limit", name="write", arguments={"path": "a.txt"})
    model_text = (
        TRUNCATED_RESULT_TEXT if arm == OUTPUT_LIMIT_ARGUMENTS else LENGTH_ENDED_CALL_RESULT_TEXT
    )
    payload = None if arm is None else {"details": {OUTPUT_LIMIT_KEY: arm, "__synthetic": True}}
    result = Message(
        role="tool",
        content=[TextContent(text=model_text)],
        tool_call_id="c_limit",
        tool_name="write",
        is_error=True,
        provider_payload=payload,
    )
    return [Message.user("go"), _assistant("", calls=[call], stop="length"), result]


#: The two arms' operator lines, spelled out because they are USER-VISIBLE COPY:
#: a reworded notice is the change this test exists to catch, and comparing
#: against the module's own constant would follow the reword instead of pinning
#: what the operator reads (review round 2, MINOR-1).
_CUT_RECEIPT = "tool call cut off at the output limit (nothing ran)"
_TURN_RECEIPT = "turn cut off at the output limit before this call ran"
_CUT_NOTICE = "turn cut off at the output limit mid tool call — nothing ran"
_TURN_NOTICE = "turn cut off at the output limit — nothing ran"


def test_the_limit_receipt_is_the_rows_line_and_not_the_expansions() -> None:
    """The display half of review F2, pinned on the package that changed.

    Nothing outside ``tests/unit/harness/test_loop.py`` mentioned the receipt
    strings, so reverting ``receipt or result_text`` on ANY single row surface
    stayed green (review round 2, MINOR-1 == QA Q-R2-2). This is the mobile
    half of that pin, and it asserts the row the operator reads rather than the
    helper's return value.

    It pins the DECISION'S shape too: the receipt is the row's error line and
    nothing else. Writing it to the expansion as well is what made one tap show
    one sentence twice in two styles, and the notice a third time (design round
    1, D5). A call that never ran has no output to expand; it has arguments.
    """
    for arm, receipt in (
        (OUTPUT_LIMIT_ARGUMENTS, _CUT_RECEIPT),
        (OUTPUT_LIMIT_TURN, _TURN_RECEIPT),
    ):
        rows = _page_rows(_limit_turn(arm))
        row = next(r for r in rows if r.kind == "tool")
        assert row.error == receipt
        assert row.details.get("output", "") == ""
        assert "args" in row.details
        # Neither the model-facing prose nor a size claim reaches the operator.
        for model_text in (TRUNCATED_RESULT_TEXT, LENGTH_ENDED_CALL_RESULT_TEXT):
            assert model_text not in str(row.details)
            assert model_text not in row.error
        assert "oversize" not in str(rows)


def test_the_length_notice_reads_the_arm_off_the_turns_own_results() -> None:
    """One limit, two arms, and the notice may only name the one that happened.

    On the arm where every call's arguments arrived COMPLETE, the turn-level
    notice read "tool call cut off at the output limit (nothing ran)" two rows
    under a card that said "turn cut off at the output limit before this call
    ran": one event, two opposite explanations of it, while the model was being
    re-asked for a smaller call it had no reason to shrink (design round 1, D1;
    QA Q-R2-1; review round 2, MINOR-2).

    The arm is read off the turn's OWN results — the marker the loop stamps on
    the synthetic result — so the notice and the row cannot disagree. A
    transcript the fold cannot read an arm from (one written before the marker
    existed) takes the line that is true either way, never the dramatic one.
    """
    cut_rows = _page_rows(_limit_turn(OUTPUT_LIMIT_ARGUMENTS))
    turn_rows = _page_rows(_limit_turn(OUTPUT_LIMIT_TURN))

    assert [r.text for r in cut_rows if r.kind == "notice"] == [_CUT_NOTICE]
    assert [r.text for r in turn_rows if r.kind == "notice"] == [_TURN_NOTICE]

    # The notice states the TURN and the row states the CALL, so the sentence
    # is painted once: no notice line is a receipt, verbatim (design round 1,
    # D2 measured the two rows byte-identical before this).
    assert {_CUT_NOTICE, _TURN_NOTICE}.isdisjoint({_CUT_RECEIPT, _TURN_RECEIPT})

    # Unmarked (legacy) result: the arm is unknown, so the notice makes no arm
    # claim at all — and the row keeps its own text, because the receipt is
    # keyed on the marker and not on the shape of an error row.
    legacy_rows = _page_rows(_limit_turn(None))
    assert [r.text for r in legacy_rows if r.kind == "notice"] == [_TURN_NOTICE]
    legacy_row = next(r for r in legacy_rows if r.kind == "tool")
    assert legacy_row.error != _TURN_RECEIPT


def test_the_shared_helpers_normalize_so_the_hosts_cannot_diverge() -> None:
    """Review round 1: the two hosts fed the shared helpers differently.

    The TUI strips a message's text at the top of its replay loop; the phone
    fold passed ``message.text`` verbatim. A whitespace-only assistant turn
    with ``stop_reason="error"`` therefore said "turn failed" on the TUI and
    NOTHING on the phone — D3 reopening inside the module built to close it.
    The strip belongs to the helper, so neither host's normalization can
    decide the answer.
    """
    padded = "  \n\t "

    assert assistant_stop_notice(
        text=padded, has_tool_calls=False, stop_reason="error", provider_payload=None
    ) == assistant_stop_notice(
        text=padded.strip(), has_tool_calls=False, stop_reason="error", provider_payload=None
    )

    # Chrome and skill payloads take the same treatment, for the same reason.
    chrome = harness_chrome_prompts()[0]
    assert is_harness_chrome(f"  {chrome}\n")
    assert user_row_text("  hello  ") == "hello"


def test_a_whitespace_only_turn_produces_no_assistant_row_on_either_surface() -> None:
    """QA round 2 (Q3): the phone emitted an extra EMPTY assistant row.

    ``if message.text:`` is truthy for ``"   "``, so a whitespace-only turn
    painted a blank row on the phone that the TUI — which tests its stripped
    text — never produces. The notice beside it rendered correctly on both
    surfaces, so the visible symptom was only ~8px of blank space; the real
    defect is that the row SEQUENCES stopped matching, which is the whole
    claim this delta makes.

    Asserted as sequence equality rather than by looking at a frame: the
    property is "these two lists are the same", and a screenshot can only
    show that some blank space went away.
    """
    padded = "   \n\t "

    # The helper both hosts read: whitespace is nothing, prose is itself.
    assert assistant_row_text(padded) == ""
    assert assistant_row_text("  hello  ") == "hello"

    history = [Message.user("do it"), _assistant(padded, stop="error")]

    # Both phone folds, and the kinds in order.
    page = [row.kind for row in _page_rows(history)]
    attach = [row.kind for row in _attach_rows(history)]
    assert page == attach

    # The notice still renders — this fix removes the blank row, NOT the
    # row the reviewer's MAJOR was about.
    assert page == ["user", "notice"]
    assert "assistant" not in page

    # And the ordinary case is untouched: real prose still gets its row. No
    # notice here — `assistant_stop_notice` guards its error arm on there
    # being NOTHING produced, which is the same emptiness question this test
    # is about, answered by the same strip.
    real = [Message.user("do it"), _assistant("here is the answer", stop="error")]
    assert [row.kind for row in _page_rows(real)] == ["user", "assistant"]


def test_the_wake_headline_strips_every_model_facing_prefix() -> None:
    """Review round 2 (MINOR-1): a single strip leaves a doubled prefix.

    ``wake_receipt_headline`` removed one ``(alarm) `` and stopped, so
    ``(alarm) (alarm) x`` leaked the marker onto a human surface. No producer
    emits a doubled prefix today; this pins the shape so the function that
    exists to keep model-facing markup off the screen cannot itself pass some
    through.
    """
    assert wake_receipt_headline("(alarm) build finished") == "build finished"
    assert wake_receipt_headline("(alarm) (alarm) build finished") == "build finished"
    assert wake_receipt_headline("(alarm) " * 5 + "build finished") == "build finished"
    # A message that merely MENTIONS the marker keeps its own words.
    assert wake_receipt_headline("see (alarm) in the logs") == "see (alarm) in the logs"


def test_both_surfaces_strip_the_reference_block() -> None:
    """A `@path` expansion is model-facing payload, not the user's words.

    An `@` reference is expanded ONCE at submit and the file rides the message
    as an `<operator-references>` block. The model needs it; a transcript row
    must not show it, or a one-line question about a file paints as the whole
    file.

    Asserted through `user_row_text` because that is the ONE function both
    surfaces paint through (`tui/session_presentation.py:1093` and
    `mobile/projection.py:865`). Stripping in a host is how the phone once got
    a rule the TUI had and the other did not, which is the divergence this whole
    file exists to prevent — so the test lives here rather than in a new
    `test_rows.py` beside it.

    THE FIXTURE IS THE RESOLVER'S REAL OUTPUT, preamble included. It used to
    build `<file path="auth.py">`, a shape nothing emits — the strip is
    shape-blind, so the test passed either way and the fixture was a fiction
    that would mislead the next reader into thinking `<file>` was the contract.
    """
    from local_operator.references import REFERENCE_BLOCK_CLOSE, REFERENCE_BLOCK_OPEN

    typed = "what does @auth.py do?"
    sent = (
        f"{typed}\n\n{REFERENCE_BLOCK_OPEN}\n\n"
        "The operator's message references these paths. "
        "Content is included below.\n\n"
        '<reference path="auth.py" typed="@auth.py" bytes="31" lines="2">\n'
        "def login():\n    return SECRET\n\n</reference>\n\n"
        f"{REFERENCE_BLOCK_CLOSE}"
    )

    row = user_row_text(sent)

    assert row == typed
    assert "SECRET" not in row
    assert REFERENCE_BLOCK_OPEN not in row
    # The token itself SURVIVES: it is what the operator typed, and the row is
    # a record of that rather than of what the model was handed.
    assert "@auth.py" in row


def test_an_unexpanded_message_is_untouched_by_the_reference_strip() -> None:
    """The overwhelmingly common case must not be rewritten.

    Every message that cites no file goes through this same function, so a
    strip that trimmed or normalised ordinary prose would change every row on
    both surfaces to fix a case that is not present.
    """
    assert user_row_text("just a question about @ signs") == "just a question about @ signs"
    assert user_row_text("plain prose") == "plain prose"


def test_a_message_that_QUOTES_the_block_marker_keeps_all_its_words() -> None:
    """Quoting the tag is ordinary prose, and must not truncate the row.

    The marker is part of the product's visible vocabulary, so an operator
    asking about this feature, quoting a log line, or pasting a prompt will type
    it. An unanchored `text.find(REFERENCE_BLOCK_OPEN)` treated every one of
    those as the start of a block and dropped the rest of the sentence —
    silently, with no notice, on BOTH surfaces. R4 says the transcript shows
    what the operator typed; it showed strictly less.

    The sibling rule this is modelled on is the same one
    `test_a_wake_receipt_headline…` asserts for `(alarm)`: a message that merely
    MENTIONS the marker keeps its own words.
    """
    from local_operator.references import REFERENCE_BLOCK_CLOSE, REFERENCE_BLOCK_OPEN

    quoted = f"why does my message contain {REFERENCE_BLOCK_OPEN} in it?"
    assert user_row_text(quoted) == quoted

    mid = f"explain {REFERENCE_BLOCK_OPEN} and then tell me about the resolver"
    assert user_row_text(mid) == mid

    # A closer in the MIDDLE of a sentence: the text does not end with the
    # marker, so `reference_block_stripped` returns at its `endswith` check and
    # never reaches the no-opener branch. This is the quoted-tag case, not the
    # half-a-block one.
    closer_only = f"what does {REFERENCE_BLOCK_CLOSE} mean?"
    assert user_row_text(closer_only) == closer_only

    # A TRAILING closer with no opener is what actually reaches the
    # `opened == -1` guard (`harness/rows.py`): the text ends with the marker,
    # so the `endswith` check passes and the `rfind` for an opener returns -1.
    # That is half a quoted tag, not a block — guessing a span for it would be
    # the same defect in the other direction. The assertion above cannot
    # observe this branch, which left the guard untested.
    trailing_closer = f"paste went wrong, here is a stray closer {REFERENCE_BLOCK_CLOSE}"
    assert user_row_text(trailing_closer) == trailing_closer


def test_a_real_block_is_still_stripped_when_the_prose_also_quotes_the_marker() -> None:
    """The case that makes the anchor a rule rather than a special case.

    A message can legitimately do both: ask about the tag AND carry a real
    expansion. Only the appended block may go, and the operator's own sentence —
    marker and all — must survive intact. Taking the LAST opener rather than the
    first is what buys this.
    """
    from local_operator.references import REFERENCE_BLOCK_CLOSE, REFERENCE_BLOCK_OPEN

    typed = f"what is {REFERENCE_BLOCK_OPEN} for? see @auth.py"
    sent = (
        f"{typed}\n\n{REFERENCE_BLOCK_OPEN}\n\n"
        "The operator's message references these paths. "
        "Content is included below.\n\n"
        '<reference path="auth.py" typed="@auth.py" bytes="31" lines="2">\n'
        "def login():\n    return SECRET\n\n</reference>\n\n"
        f"{REFERENCE_BLOCK_CLOSE}"
    )

    row = user_row_text(sent)

    assert row == typed
    assert "SECRET" not in row
