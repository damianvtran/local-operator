"""The ``ask`` tool: what it advertises, what it validates, what it reports.

The tool replaced PROSE. Without it a model that needs a decision writes its
options into the reply — observed verbatim as "(A) Drop email … (B) Escalate it
properly … (C) You have context I don't" — which nobody can click, nobody can
key-select, and the agent then has to re-parse from free text.

So the contracts pinned here are the ones that keep it from regressing into that
surface or into something worse than it: the tool must not EXIST where no human
can answer it, a refusal to answer must not read as a failure, and the free-text
answer must survive the round trip, because "an answer that was not on the list"
is what the prose version was reaching for with its third option.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import AgentTool, AskQuestion, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


def _questions(*, multi: bool = False, recommended: int | None = None) -> list[dict[str, Any]]:
    return [
        {
            "id": "stale",
            "question": "What should happen to the stale rows?",
            "options": [
                {"label": "Drop them", "description": "nothing reads the column"},
                {"label": "Backfill from the audit log", "description": "slower, keeps history"},
            ],
            "multi": multi,
            "recommended": recommended,
        }
    ]


def _context(hook: Any | None, *, has_ui: bool = True) -> ToolContext:
    return ToolContext(cwd=".", session_id="s", has_ui=has_ui, ask_user=hook)


def _tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    tool = _tools(context)["ask"]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


async def _answer_with(result: dict[str, list[str]] | None):
    """A host hook that records the questions it was given and answers ``result``."""
    seen: list[list[AskQuestion]] = []

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        seen.append(questions)
        return result

    return hook, seen


# --- availability -----------------------------------------------------------


def test_ask_exists_when_a_host_can_actually_answer_it() -> None:
    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        return None

    assert "ask" in _tools(_context(hook))


def test_ask_is_absent_without_a_ui_to_draw_the_question_on() -> None:
    """A server, exec mode or a scheduler run has nobody at a keyboard. An
    advertised tool that can only fail to be answered is worse than none: the
    model spends a call finding out what the tool list could have told it."""

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        return None

    assert "ask" not in _tools(_context(hook, has_ui=False))


def test_ask_is_absent_without_an_ask_hook_even_when_a_ui_is_claimed() -> None:
    """This is the case that keeps SUBAGENTS out. A child inherits ``has_ui``
    from its parent and is built with no ask handler, so the hook's absence — not
    the flag — is what stops a delegated agent from mounting a question on the
    parent's screen and blocking on a human who was never shown it."""
    assert "ask" not in _tools(_context(None))


# --- schema validation ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_question_with_one_option_is_refused() -> None:
    """A single-option question is an announcement. Rendering it as a picker
    asks the user to ratify a decision that has already been made."""
    hook, seen = await _answer_with({"stale": ["Drop them"]})
    context = _context(hook)
    args = _questions()
    args[0]["options"] = [{"label": "Drop them"}]

    result = await _call(context, {"questions": args})

    assert result.is_error is True
    assert "invalid arguments" in result.text
    assert seen == []  # nothing was put on screen


@pytest.mark.asyncio
async def test_a_recommendation_outside_the_options_is_refused() -> None:
    """Not clamped: a clamp would preselect and visibly endorse a DIFFERENT
    option than the model meant to, which is worse than an error the model can
    correct."""
    hook, seen = await _answer_with({"stale": ["Drop them"]})
    result = await _call(_context(hook), {"questions": _questions(recommended=5)})

    assert result.is_error is True
    assert "recommended" in result.text
    assert seen == []


@pytest.mark.asyncio
async def test_an_empty_question_list_is_refused() -> None:
    hook, seen = await _answer_with(None)
    result = await _call(_context(hook), {"questions": []})

    assert result.is_error is True
    assert "invalid arguments" in result.text
    assert seen == []


@pytest.mark.asyncio
async def test_a_valid_recommendation_reaches_the_host_at_the_top() -> None:
    """The recommendation has to survive parsing, and it arrives HOISTED: the
    model authored it at index 1 and the host is handed it at index 0, because
    position is the only channel some surfaces have for it."""
    hook, seen = await _answer_with({"stale": ["Drop them"]})
    result = await _call(_context(hook), {"questions": _questions(recommended=1)})

    assert result.is_error is False
    assert seen[0][0].recommended == 0
    assert [option.label for option in seen[0][0].options] == [
        "Backfill from the audit log",
        "Drop them",
    ]


# --- results ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_single_choice_comes_back_as_the_label_under_its_question() -> None:
    """The report echoes the QUESTION as well as its id: the ask may be several
    turns back by the time the model reads it, and ``stale: Drop them`` alone
    does not say what was agreed to."""
    hook, _seen = await _answer_with({"stale": ["Drop them"]})
    result = await _call(_context(hook), {"questions": _questions()})

    assert result.is_error is False
    assert "stale — What should happen to the stale rows?" in result.text
    assert "answer: Drop them" in result.text
    assert result.details == {"answers": {"stale": ["Drop them"]}}


@pytest.mark.asyncio
async def test_a_multi_select_answer_keeps_every_label() -> None:
    hook, _seen = await _answer_with(
        {"stale": ["Drop them", "Backfill from the audit log"]},
    )
    result = await _call(_context(hook), {"questions": _questions(multi=True)})

    assert result.is_error is False
    assert "answer: Drop them; Backfill from the audit log" in result.text


@pytest.mark.asyncio
async def test_free_text_comes_back_verbatim_although_no_option_carried_it() -> None:
    """The whole reason the picker offers an "Other" row. The prose surface this
    replaces needed one constantly — its third option was literally "You have
    context I don't" — and an answer reported as an option INDEX could not carry
    it at all."""
    typed = "neither — archive them to S3 first"
    hook, _seen = await _answer_with({"stale": [typed]})
    result = await _call(_context(hook), {"questions": _questions()})

    assert result.is_error is False
    assert f"answer: {typed}" in result.text
    assert result.details == {"answers": {"stale": [typed]}}


@pytest.mark.asyncio
async def test_answering_nothing_is_a_result_and_not_an_error() -> None:
    """Refusing to answer is a decision. Reported as an error the model either
    retries the same question or stops; reported as this text it takes its own
    recommendation and says what it assumed."""
    hook, _seen = await _answer_with(None)
    result = await _call(_context(hook), {"questions": _questions()})

    assert result.is_error is False
    assert result.text == builtin.ASK_UNANSWERED_TEXT
    assert "closed the question without answering" in result.text


@pytest.mark.asyncio
async def test_a_mapping_with_nothing_in_it_reads_as_answering_nothing() -> None:
    """A confirmed-but-empty multi-select and an escape are the same outcome to
    a model: nothing was chosen. Two different results would offer a
    distinction it cannot act on differently."""
    hook, _seen = await _answer_with({"stale": [], "other": ["   "]})
    result = await _call(_context(hook), {"questions": _questions()})

    assert result.is_error is False
    assert result.text == builtin.ASK_UNANSWERED_TEXT


@pytest.mark.asyncio
async def test_a_question_the_user_skipped_is_reported_as_not_answered() -> None:
    """Escaping out of question three does not discard the answer to question
    one, so a partial mapping has to render honestly."""
    questions = _questions() + [
        {
            "id": "timing",
            "question": "When should this ship?",
            "options": [{"label": "Now"}, {"label": "After the backfill"}],
        }
    ]
    hook, _seen = await _answer_with({"stale": ["Drop them"]})
    result = await _call(_context(hook), {"questions": questions})

    assert result.is_error is False
    assert "answer: Drop them" in result.text
    assert "timing — When should this ship?" in result.text
    assert "answer: (not answered)" in result.text


@pytest.mark.asyncio
async def test_a_missing_host_hook_is_an_error_not_a_user_refusal() -> None:
    """Unreachable through the advertised tool, and it must not be reported as
    "the user declined": that would have the model act as though a person had
    seen the question on a session where nothing was ever drawn."""
    hook, _seen = await _answer_with(None)
    # Built WITH a hook (the builder refuses to create it without one) and then
    # executed against a context that has none — the only way this session's
    # tool list and its executor can disagree.
    tool = builtin.build_ask_tool(_context(hook))
    assert tool is not None
    result = await tool.execute("call-1", {"questions": _questions()}, None, None, _context(None))

    assert result.is_error is True
    assert "cannot" in result.text


# --- secret questions -------------------------------------------------------


def _secret_question(qid: str = "GITHUB_TOKEN") -> dict[str, Any]:
    return {
        "id": qid,
        "question": "Paste the deploy token.",
        "options": [],
        "secret": True,
    }


@pytest.mark.asyncio
async def test_a_secret_question_with_options_is_refused() -> None:
    """A secret question is a masked paste, not a picker. Options would put a
    choice list on a surface that must never echo the value."""
    hook, seen = await _answer_with({"GITHUB_TOKEN": ["sk-secret"]})
    args = _secret_question()
    args["options"] = [{"label": "Use the env one"}, {"label": "Skip"}]
    result = await _call(_context(hook), {"questions": [args]})
    assert result.is_error is True
    assert seen == []


@pytest.mark.asyncio
async def test_a_secret_question_stores_the_value_and_reports_only_the_key() -> None:
    """The pasted bytes must not ride the tool result: that text is persisted
    to the transcript and replayed to the provider."""
    from local_operator.variables import VariableStore

    secret = "ghp_this_is_a_real_looking_token"
    hook, seen = await _answer_with({"GITHUB_TOKEN": [secret]})
    store = VariableStore(cwd="/tmp", env={})
    context = ToolContext(cwd=".", session_id="s", has_ui=True, ask_user=hook, variables=store)
    result = await _call(context, {"questions": [_secret_question()]})

    assert result.is_error is False
    assert secret not in result.text
    assert "answer: GITHUB_TOKEN" in result.text
    assert result.details == {"answers": {"GITHUB_TOKEN": ["GITHUB_TOKEN"]}}
    assert store.credential_names() == ["GITHUB_TOKEN"]
    assert store.credential_env()["GITHUB_TOKEN"] == secret
    assert seen[0][0].secret is True


@pytest.mark.asyncio
async def test_a_stored_secret_question_announces_the_key_to_the_session() -> None:
    """The ask result names the key once; the session journal is what makes
    it findable on every LATER turn. Without the announce hook call, the
    operator's pasted credential was stored and then forgotten by the model
    two turns later (the failure behind session 835fbcafdc27)."""
    from local_operator.variables import VariableStore

    secret = "ghp_announce_me_never_show"
    hook, _seen = await _answer_with({"GITHUB_TOKEN": [secret]})
    store = VariableStore(cwd="/tmp", env={})
    announced: list[tuple[str, dict[str, Any]]] = []

    def journal(key: str, **kwargs: Any) -> None:
        announced.append((key, kwargs))

    context = ToolContext(
        cwd=".",
        session_id="s",
        has_ui=True,
        ask_user=hook,
        variables=store,
        journal_credential=journal,
    )
    result = await _call(context, {"questions": [_secret_question()]})

    assert result.is_error is False
    assert announced == [("GITHUB_TOKEN", {"replaced": False})]
    # The announce payload must carry the key, never the value.
    assert secret not in str(announced)


@pytest.mark.asyncio
async def test_an_announce_hook_failure_does_not_fail_the_ask() -> None:
    """The credential is stored before the announcement runs; a host whose
    journal hook raises must not lose the answer it already holds."""
    from local_operator.variables import VariableStore

    secret = "ghp_hook_blows_up"
    hook, _seen = await _answer_with({"GITHUB_TOKEN": [secret]})
    store = VariableStore(cwd="/tmp", env={})

    def broken_journal(key: str, **kwargs: Any) -> None:
        raise RuntimeError("journal unavailable")

    context = ToolContext(
        cwd=".",
        session_id="s",
        has_ui=True,
        ask_user=hook,
        variables=store,
        journal_credential=broken_journal,
    )
    result = await _call(context, {"questions": [_secret_question()]})
    assert result.is_error is False
    assert "answer: GITHUB_TOKEN" in result.text
    assert store.credential_names() == ["GITHUB_TOKEN"]


@pytest.mark.asyncio
async def test_a_declined_secret_question_is_the_same_as_answering_nothing() -> None:
    """Escaping a secret question is a refusal, not a stored blank. The
    existing unanswered path already tells the model not to ask again."""
    hook, _seen = await _answer_with(None)
    from local_operator.variables import VariableStore

    store = VariableStore(cwd="/tmp", env={})
    context = ToolContext(cwd=".", session_id="s", has_ui=True, ask_user=hook, variables=store)
    result = await _call(context, {"questions": [_secret_question()]})
    assert result.is_error is False
    assert "nothing was chosen" in result.text
    assert store.credential_names() == []
