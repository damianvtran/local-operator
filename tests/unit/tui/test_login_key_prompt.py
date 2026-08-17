"""`/login <paste-a-key provider>`: the prompt the TUI never had.

The reported failure was `/login alibaba-token-plan` ending in
`login failed: QwenCloud Token Plan login requires an interactive key prompt`.
Eleven providers offer an interactive login and nine of them read a key from the
user; the TUI attached no `on_manual_code_input` at all, so for those nine the
flow could only ever fail. It was not a QwenCloud bug and it was not reachable
from any provider-specific test.

So the tests here are split by what they can each see:

* The REGISTRY-driven test at the bottom runs the real `ProviderController`
  over a temp `AuthStore` and drives `/login` end to end, because the defect
  lived in the seam between the registry's requirement and the app's callbacks
  and neither side alone could show it.
* The widget tests drive real keys against `KeyPromptBlock`, because the
  masking and the never-echo property are keystroke-level guarantees a test
  calling the actions directly would not check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.key_prompt import MASK_CHAR, KeyPromptBlock
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class _PromptHost(App[None]):
    """A host whose only job is to own the block under test."""

    def compose(self) -> ComposeResult:
        return iter(())

    async def open_prompt(self, label: str = "Alibaba Cloud") -> KeyPromptBlock:
        block = KeyPromptBlock(label)
        await self.mount(block)
        return block


def _rendered(block: KeyPromptBlock) -> str:
    """Everything the block draws, flattened — what a user could read off it."""
    from tests.unit.tui.test_app_pilot import _renderable_plain

    content = block._content
    assert content is not None
    return _renderable_plain(content)


@pytest.mark.asyncio
async def test_typed_key_is_masked_and_never_rendered() -> None:
    """The value is a secret: it is masked on screen and absent from the render.

    `/login` gets run on shared screens and in recorded demos, and the key stays
    in the scrollback afterwards. The mask still has to answer "did my paste
    arrive, and did it arrive whole", which is what the per-character length is
    for.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt()
        await pilot.pause()
        for char in "sk-secret1":
            await pilot.press(char)
        await pilot.pause()

        text = _rendered(block)
        assert "sk-secret1" not in text, text
        assert MASK_CHAR * 10 in text, text
        assert block.typed_length == 10


@pytest.mark.asyncio
async def test_backspace_edits_and_enter_submits_the_real_value() -> None:
    """The block reports what was typed, not what it displayed."""
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt()
        await pilot.pause()
        for char in "sk-abcX":
            await pilot.press(char)
        await pilot.press("backspace")
        await pilot.pause()
        assert block.typed_length == 6

        await pilot.press("enter")
        await pilot.pause()
        assert block.wait().result() == "sk-abc"
        assert block.answered


@pytest.mark.asyncio
async def test_pasted_key_arrives_whole() -> None:
    """A bracketed paste is ONE event, not a stream of keys.

    Pasting is the primary gesture here, so without a `Paste` handler the key
    would be silently dropped and the user would be looking at an empty prompt
    after a paste that appeared to work. The trailing newline a dashboard copy
    carries must not submit on its own, or the paste and the confirmation become
    the same gesture.
    """
    from textual.events import Paste

    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt()
        await pilot.pause()
        block.post_message(Paste("sk-pasted-secret\n"))
        await pilot.pause()

        assert block.typed_length == len("sk-pasted-secret")
        assert not block.answered, "a trailing newline must not submit the paste"
        assert "sk-pasted-secret" not in _rendered(block)

        await pilot.press("enter")
        await pilot.pause()
        assert block.wait().result() == "sk-pasted-secret"


@pytest.mark.asyncio
async def test_escape_and_empty_enter_both_cancel() -> None:
    """Two ways out, one outcome: `None`, never an empty-string credential.

    An empty key stored as a credential shadows a working environment key in the
    stream-time cascade, so every later request fails to authenticate with
    nothing on screen to explain it.
    """
    for keys in (["escape"], ["enter"], ["space", "enter"]):
        app = _PromptHost()
        async with app.run_test() as pilot:
            block = await app.open_prompt()
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
            await pilot.pause()
            assert block.answered, keys
            assert block.wait().result() is None, keys


@pytest.mark.asyncio
async def test_resolve_is_idempotent() -> None:
    """Several paths end one prompt (enter, escape, abort, unmount, clear) and a
    second `set_result` on a settled future raises. Losing that race must not
    take the app down."""
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt()
        await pilot.pause()
        block.resolve("sk-first")
        block.resolve("sk-second")
        block.resolve(None)
        assert block.wait().result() == "sk-first"


@pytest.mark.asyncio
async def test_settled_block_stops_holding_the_key() -> None:
    """A settled prompt sits in the transcript for the rest of the session; it
    must not still be holding the user's key in memory while it does."""
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt()
        await pilot.pause()
        for char in "sk-xyz":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        assert block.typed_length == 0
        assert "sk-xyz" not in _rendered(block)


# --- the seam: registry requirement -> app callbacks -> stored credential ----


class _LoginSession(FakeSession):
    pass


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _controller(tmp_path: Path):
    """The REAL ProviderController over a throwaway store.

    A fake controller cannot show this defect: the requirement is declared by
    the shipped registry and honoured (or not) by the app's callbacks, so a
    stub standing in for either end is a test of the stub.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    return ProviderController(AuthStore(tmp_path / "auth.db"))


async def _run_login(pilot, app: OperatorApp, provider: str) -> None:
    editor = app.query_one(Editor)
    editor.text = f"/login {provider}"
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if app.query(KeyPromptBlock):
            return


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_login_to_a_paste_key_provider_stores_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported bug, end to end, through the real registry and store.

    Before this change the flow reached `login failed: … requires an
    interactive code prompt` with nothing to type into.
    """
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")

        prompts = list(app.query(KeyPromptBlock))
        assert prompts, f"no key prompt was offered; notices={_notices(app)}"
        prompt = prompts[0]
        assert prompt.has_focus, "the prompt must take focus or the keys reach the composer"

        for char in "sk-real-key":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if any("Stored API key" in note for note in _notices(app)):
                break

    stored = [c for c in controller.auth_store.list_credentials(provider=None)]
    assert [c.provider for c in stored] == ["alibaba"], stored
    assert stored[0].data["key"] == "sk-real-key"
    assert stored[0].data["source"] == "login"


@pytest.mark.asyncio
async def test_cancelling_a_login_stores_nothing_and_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escape is an outcome, not a failure.

    Reported as a red `login failed: … cancelled` it tells the user their own
    Escape broke something, which is both false and the kind of message that
    sends someone hunting a problem that does not exist.
    """
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")
        assert list(app.query(KeyPromptBlock)), _notices(app)

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if any("cancelled" in note for note in _notices(app)):
                break

        notices = _notices(app)
        assert any("cancelled" in note for note in notices), notices
        assert not any("failed" in note for note in notices), notices

    assert controller.auth_store.list_credentials(provider=None) == []


@pytest.mark.asyncio
async def test_loopback_provider_is_offered_no_prompt(tmp_path: Path) -> None:
    """The other half of the contract, and the reason the blanket omission
    existed: for a loopback provider the callback server is already listening,
    and a prompt racing it leaves the terminal blocked on a line nobody types.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        loopback = controller.provider("openai")
        assert loopback is not None and not loopback.accepts_paste_prompt
        assert app._login_callbacks(loopback).on_manual_code_input is None

        # Non-zero control: the same method DOES attach one for a paste provider,
        # so the None above cannot be a callbacks builder that attaches nothing.
        paste = controller.provider("alibaba")
        assert paste is not None
        assert app._login_callbacks(paste).on_manual_code_input is not None


@pytest.mark.asyncio
async def test_every_paste_provider_gets_a_prompt_from_the_tui(tmp_path: Path) -> None:
    """Enumerated over the registry rather than checked for the one provider in
    the bug report: eight others were in the identical state, and a test naming
    providers individually is what let them ship.
    """
    from local_operator.providers.registry import PROVIDER_REGISTRY

    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        wanted = [p for p in PROVIDER_REGISTRY if p.login is not None and p.paste_prompt_required]
        assert len(wanted) >= 8, [p.id for p in wanted]
        missing = [p.id for p in wanted if app._login_callbacks(p).on_manual_code_input is None]
        assert missing == [], missing


@pytest.mark.asyncio
async def test_teardown_settles_a_live_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt still on screen at teardown holds a future the login worker is
    parked on, and `dispose` awaits teardown — the same hang the approval card
    and the ask picker are settled to prevent."""
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")
        prompt: Any = app._key_prompt
        assert prompt is not None

    assert prompt.answered, "unmount left the login parked on an unresolved future"
    assert prompt.wait().result() is None
