"""Every auto-approval path refuses or asks a person for a person-only gate (review R1-2).

``harness.approval.never_auto_approved`` names gates no auto mode may answer:
an account profile save and a community post. Each row below is one place a
gate can be answered WITHOUT a card, driven through its real function:

* SDK ``ApprovalPolicy.auto()`` (``sdk._auto_approve_gate``) — the gap R1-2 found;
* ``--yolo`` and headless ``lop exec`` (``session_factory._make_request_approval``):
  person-only goes to the tty prompt, and with no tty that raises the typed refusal;
* TUI approve-all and desktop/mobile full-auto, which share
  ``bridge_egress.session_grant_covers`` (``tui/app.py`` ``request_tool_approval``,
  ``session/runtime/serving.py`` ``approval_gate``), at both grant scopes;
* the card-side discharge predicate ``bridge_egress.requires_person``;
* a declared unattended inventory (``Session._tool_approval_gate``), which
  must forward a person-only title to the base gate even if it were declared.

Subagents and team members inherit their parent's base gate
(``harness/subagent.py``: ``parent_session._request_approval``), so they reach
one of the rows above rather than a gate of their own.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import sdk, session_factory
from local_operator.harness.approval import (
    ACCOUNT_PROFILE_GATE,
    COMMUNITY_POST_GATE,
    ApprovalUnavailableError,
    ask_approval,
)
from local_operator.harness.bridge_egress import (
    ApprovalGrant,
    requires_person,
    session_grant_covers,
)
from local_operator.session.session import Session

PERSON_ONLY = [ACCOUNT_PROFILE_GATE, COMMUNITY_POST_GATE]


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", PERSON_ONLY)
async def test_sdk_auto_refuses_person_only_gates(gate: str) -> None:
    with pytest.raises(ApprovalUnavailableError):
        await sdk._auto_approve_gate(gate, "card")
    assert await sdk._auto_approve_gate("bash", "run: ls") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", PERSON_ONLY)
async def test_yolo_and_headless_exec_never_answer_person_only_gates(
    gate: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No tty (a headless `lop exec`, a daemon): the y/N prompt cannot be asked.
    monkeypatch.setattr(sys, "stdin", None)
    yolo = session_factory._make_request_approval(True)
    with pytest.raises(ApprovalUnavailableError):
        await yolo(gate, "card")
    assert await yolo("bash", "run: ls") is True
    with pytest.raises(ApprovalUnavailableError):
        await session_factory._make_request_approval(False)(gate, "card")


@pytest.mark.parametrize("gate", PERSON_ONLY)
@pytest.mark.parametrize("scope", ["session", "global"])
def test_approve_all_and_full_auto_grants_do_not_cover_person_only_gates(
    gate: str, scope: Any
) -> None:
    grant = ApprovalGrant(auto=True, scope=scope)
    assert session_grant_covers(grant, title=gate) is False
    assert session_grant_covers(grant, title="bash") is True
    assert requires_person(gate, bridge_latched=False) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", PERSON_ONLY)
async def test_declared_unattended_inventory_forwards_person_only_gates(gate: str) -> None:
    asked: list[str] = []

    async def base(tool_name: str, description: str) -> bool:
        asked.append(tool_name)
        return False

    # Declared on purpose, to pin the belt-and-braces check: a title with a
    # space cannot be declared today, but the rule must not depend on that.
    host = SimpleNamespace(
        _yolo=False,
        _request_approval=base,
        _declared_tools=frozenset({gate, "bash"}),
        _declared_tools_unattended=True,
    )
    gate_fn = Session._tool_approval_gate(host)  # type: ignore[arg-type]
    assert gate_fn is not None
    assert await ask_approval(gate_fn, "bash", "run: ls", None) is True
    assert await ask_approval(gate_fn, gate, "card", None) is False
    assert asked == [gate]


@pytest.mark.asyncio
@pytest.mark.parametrize("yolo", [False, True])
async def test_terminal_prompt_shows_the_whole_community_post_before_asking(
    yolo: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R2-1: the plain-terminal y/N never approves post text it did not display.

    The generic one-line prompt flattens the card and cuts it at 500 characters
    with no marker, so a ``y`` there would publish text the person never saw.
    Driven with the real card builder and a 3,689-character post whose markers
    sit far past that cut; ``--yolo`` reaches the same prompt (R1-4).
    """
    from local_operator.ansi import sanitize_prompt_line
    from local_operator.community.agent_ops import post_card

    body = "x" * 300 + " MARK-300 " + ("filler " * 480) + "\x1b[2J MARK-END"
    lines = [body[i : i + 600] for i in range(0, len(body), 600)]
    text = "\n".join(lines)
    assert len(text) == 3689  # the round-2 repro's size
    card = post_card(community="Tran Robotics", channel="#help", text=text, tainted=True)
    # Negative control: the generic one-line rendering loses the tail.
    assert "MARK-300" not in sanitize_prompt_line(card)

    class _Tty:
        def isatty(self) -> bool:
            return True

    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        # Everything printed before the question is what the person saw.
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(session_factory, "_fullscreen_app_owns_terminal", lambda: False)
    monkeypatch.setattr("builtins.input", fake_input)
    gate = session_factory._make_request_approval(yolo)
    assert await gate(COMMUNITY_POST_GATE, card) is True
    shown = capsys.readouterr().out
    assert "MARK-300" in shown and "MARK-END" in shown
    assert "\x1b" not in shown and "\x1b" not in prompts[0]
    # Every post line reached the screen, uncut, before the question.
    for line in lines:
        assert sanitize_prompt_line(f"┊ {line}", limit=len(line) + 2) in shown
    assert "shown above in full" in prompts[0]
    assert len(prompts[0]) < 200
