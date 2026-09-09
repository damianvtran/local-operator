"""The two phone folds and the TUI fold must agree about the same row.

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

The TUI-parity half deliberately compares ROW KINDS and TEXT rather than
widgets: the two surfaces legitimately differ in how they mount a row (the
TUI has a dedicated ``WakeBlock`` where the phone has a tagged notice), and a
divergence has to be NAMED in ``TUI_ALLOWANCES`` to be legal. An entry
disappearing from that list is a fix; an entry appearing without a reason is
the failure mode this file exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
from local_operator.harness.comms import (
    HUB_MESSAGE_TYPE,
    PARENT_MESSAGE_CLOSE_TAG,
    PARENT_MESSAGE_TAG,
    TO_CHILD_INSTRUCTIONS,
)
from local_operator.harness.rows import harness_chrome_prompts
from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.session.shell_record import shell_record_messages


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


def test_no_harness_prompt_is_painted_as_the_users_words() -> None:
    """D9: the phone suppressed ONE of the three continuation prompts and
    rendered the other two as the user's own words — a partially copied
    list, which is the drift signature itself. All three now come from one
    shared list."""
    prompts = harness_chrome_prompts()
    assert len(prompts) == 3

    assert _page_rows([Message.user(p) for p in prompts]) == []


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
