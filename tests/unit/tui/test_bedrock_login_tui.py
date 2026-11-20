"""The TUI's rendering of Amazon Bedrock's structured AWS login (#4270 round 1).

Regressions for the round-1 findings on the TUI host, not the flow itself
(``tests/unit/providers/bedrock/test_login_flow.py`` owns that):

* Q1 — every typed field's hint rendered as the raw id
  ``wire.login.bedrock.field_hint`` because the host translated it with the
  tui.app-only ``_t``;
* D2/D3/D5 — the login reused the agent's ``ask`` card unchanged: a free-text
  row whose answer cancelled the login, the "the agent needs your decision"
  header, and a RECOMMENDED badge on the default profile/Region;
* D4/Q7 — "I'll run it myself" painted as a red "login failed".
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.providers.oauth.callback_server import LoginDeferredError
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.providers.bedrock.awsfixtures import isolate_aws
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_login_key_prompt import (
    _boot,
    _controller,
    _LoginSession,
    _notices,
    _run_login,
)


async def _awaited(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _bedrock() -> Any:
    from local_operator.providers.registry import PROVIDER_REGISTRY

    return next(p for p in PROVIDER_REGISTRY if p.id == "bedrock")


@pytest.mark.asyncio
async def test_field_prompts_render_their_hint_not_a_raw_key_id(tmp_path: Path) -> None:
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=_controller(tmp_path))
    seen: list[dict[str, Any]] = []

    async def fake_prompt(provider_label: str, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "typed"

    app._request_login_key = fake_prompt  # type: ignore[method-assign]
    callbacks = app._aws_login_callbacks(_bedrock(), lambda *a: None, print, print)
    setup = callbacks.on_setup_input
    assert setup is not None
    for label, secret in (("Access key ID", False), ("Secret access key", True)):
        assert await _awaited(setup(label, "", secret)) == "typed"

    hints = [kwargs["instructions"] for kwargs in seen]
    assert hints == ["Enter submits; Esc cancels."] * 2
    assert not any("wire." in hint for hint in hints)


@pytest.mark.asyncio
async def test_secret_setup_preserves_caller_supplied_aws_hint(tmp_path: Path) -> None:
    from local_operator.tui.widgets.key_prompt import KeyPromptBlock

    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=_controller(tmp_path))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        expected = "Enter submits; Esc cancels."
        task = asyncio.create_task(
            app._request_login_key(
                "Amazon Bedrock",
                field_label="Secret access key",
                secret=True,
                instructions=expected,
                required_field=False,
            )
        )
        try:
            await pilot.pause()
            block = app.query_one(KeyPromptBlock)
            assert block.instructions == expected
            block.action_cancel()
            assert await task is None
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_choices_use_a_login_card_without_free_text_or_badge(tmp_path: Path) -> None:
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=_controller(tmp_path))
    calls: list[tuple[list[AskQuestion], dict[str, Any]]] = []

    async def fake_choice(questions: list[AskQuestion], **kwargs: Any):
        calls.append((questions, kwargs))
        return {"bedrock_login": ["eu-west-1"]}

    app.request_user_choice = fake_choice  # type: ignore[method-assign]
    callbacks = app._aws_login_callbacks(_bedrock(), lambda *a: None, print, print)
    options = [
        ("us-east-1", "us-east-1", "Default"),
        ("eu-west-1", "eu-west-1", ""),
        ("__other__", "Other…", "Type any AWS Region id"),
    ]
    choose = callbacks.on_choice
    assert choose is not None
    assert await _awaited(choose("Which AWS Region?", options, "us-east-1")) == "eu-west-1"

    (question,), kwargs = calls[0]
    card = kwargs["card_options"]
    assert card["allow_free_text"] is False
    assert card["title"] == "Connect Amazon Bedrock"
    assert card["exit_hint"] == ("esc", "cancel")
    assert card["initial_rows"] == [0]
    assert question.recommended is None


def test_initial_rows_place_the_cursor_without_a_recommended_badge() -> None:
    question = AskQuestion(
        id="q",
        question="Which AWS profile?",
        options=[AskOption(label=name) for name in ("admin", "default", "dev")],
    )
    card = AskPickerScreen([question], allow_free_text=False, initial_rows=[1])
    assert card._states[0].selected == 1
    assert card._questions[0].recommended_rows == ()
    # Out of range is ignored rather than parking the cursor on nothing.
    assert AskPickerScreen([question], initial_rows=[9])._states[0].selected == 0


@pytest.mark.asyncio
async def test_bedrock_login_mounts_the_connect_card_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real `/login bedrock` path: the method chooser's card."""
    isolate_aws(tmp_path, monkeypatch)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(110, 40)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "bedrock")
        for _ in range(60):
            await pilot.pause()
            if app.query(AskPickerScreen):
                break
        card = app.query_one(AskPickerScreen)
        assert card._title == "Connect Amazon Bedrock"
        assert card._allow_free_text is False
        labels = [option.label for option in card._questions[0].options]
        assert labels == [
            "AWS profile",
            "AWS SSO",
            "Access key",
            "Bedrock API key",
            "Default credentials",
        ]
        assert card._questions[0].recommended_rows == ()
        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if any("cancelled" in note for note in _notices(app)):
                break
    assert controller.auth_store.list_credentials(provider=None) == []


@pytest.mark.asyncio
async def test_run_it_yourself_is_a_note_with_the_command_set_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.bedrock import login as blogin

    command = "aws sso login --profile corp-sso"

    async def deferred(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise LoginDeferredError(
            "Nothing was stored. Run this in a terminal, then connect Amazon Bedrock again:",
            command=command,
        )

    monkeypatch.setattr(blogin, "login_bedrock", deferred)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(110, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "bedrock")
        for _ in range(40):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if any(command in note for note in _notices(app)):
                break
        blocks = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and command in block._text
        ]
        assert len(blocks) == 1, _notices(app)
        assert blocks[0].notice_kind == "note"
        assert blocks[0]._accent == command
        # Own row, so the fold never splits (and un-accents) the command.
        assert blocks[0]._text.endswith(f"\n  {command}")
        assert not any("failed" in note for note in _notices(app)), _notices(app)
    assert controller.auth_store.list_credentials(provider=None) == []
