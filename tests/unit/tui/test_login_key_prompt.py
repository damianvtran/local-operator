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
async def test_tab_cannot_move_focus_off_a_live_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design round 1 D1: one Tab used to send the API key to the MODEL.

    Textual's default Tab moves focus to the next focusable widget, which is the
    composer. The prompt went on looking live (its unfocused and focused grounds
    differ by ~1.04:1, which nobody perceives), so the user kept typing, pressed
    Enter, and the key was submitted as a chat message: into the transcript in
    plain text and into the provider's logs. Verified end to end before the
    guard existed.

    The last assertion is the one that matters. Asserting focus alone would pass
    on a widget that keeps focus and drops the characters.
    """
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    controller = _controller(tmp_path)
    session = _LoginSession()
    app = OperatorApp(lambda: _factory(session), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")
        prompt = app._key_prompt
        assert prompt is not None and prompt.has_focus

        for key in ("tab", "shift+tab", "tab"):
            await pilot.press(key)
        await pilot.pause()
        assert prompt.has_focus, "Tab moved focus off a prompt the app is parked on"

        for char in "sk-live9f3c":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if prompt.answered:
                break

        assert app.query_one(Editor).text == "", "the key was typed into the composer"
        assert session.prompts == [], "the API key was sent to the model"
        assert prompt.wait().result() == "sk-live9f3c"


@pytest.mark.asyncio
async def test_declining_anthropics_paste_is_not_reported_as_a_cancelled_login() -> None:
    """Design round 1 D2: the receipt claimed a cancel while the login ran on.

    Anthropic's paste races the loopback callback, and `_await_code` RE-PARKS on
    a declined paste by design — the browser flow is still listening. Reporting
    "login cancelled" told the user something false, and their next `/login` was
    then refused with "a login is already in progress".
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        fallback = KeyPromptBlock("Anthropic (Claude Pro/Max)", secret=False, sole_path=False)
        await app.mount(fallback)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        receipt = _rendered(fallback)
        assert "cancelled" not in receipt, receipt
        assert "still waiting for the browser" in receipt, receipt

        # Non-zero control: where the paste IS the only path, the same gesture
        # still reports a cancelled login, so the assertion above is about
        # `sole_path` and not about the widget never saying "cancelled".
        sole = await app.open_prompt("Alibaba Cloud")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert "cancelled" in _rendered(sole)


@pytest.mark.asyncio
async def test_receipts_stay_distinguishable_when_the_row_is_truncated() -> None:
    """Design round 1 D5: success and cancel rendered byte-identical at 10-27
    columns, so the narrowest terminals could not tell a stored credential from
    a cancelled login. The glyph carries the outcome, as it does on the sibling
    approval card."""
    for width in (100, 27, 24, 20, 14, 10):
        app = _PromptHost()
        async with app.run_test(size=(width, 20)) as pilot:
            # All FOUR settled states, not just the pair originally filed:
            # design round 2 (D7) found the two informational receipts
            # colliding at 14-24 columns by the same mechanism, and they are
            # opposites — the login is still running versus the login is over.
            rendered: list[str] = []
            for value, sole_path, superseded in (
                ("sk-abc123", True, False),  # key received
                (None, True, False),  # login cancelled
                (None, False, False),  # paste skipped, browser still listening
                (None, True, True),  # superseded: the login finished elsewhere
            ):
                block = KeyPromptBlock("Alibaba Cloud (Qwen)", sole_path=sole_path)
                await app.mount(block)
                await pilot.pause()
                block.resolve(value, superseded=superseded)
                await pilot.pause()
                rendered.append(_rendered(block))

            assert len(set(rendered)) == 4, (width, rendered)


@pytest.mark.asyncio
async def test_the_prompt_names_the_provider_once() -> None:
    """Design round 1 D3: `Paste your xAI (Grok API key) API key`.

    Registry names carry a parenthetical that disambiguates rows in a LIST; in
    this sentence it names the credential twice. The user has already chosen the
    provider by typing it.
    """
    app = _PromptHost()
    async with app.run_test(size=(100, 20)) as pilot:
        block = await app.open_prompt("xAI (Grok API key)")
        await pilot.pause()
        assert "Paste your xAI API key" in _rendered(block)

        # A name that is nothing but a parenthetical keeps it rather than
        # collapsing to an empty provider name.
        odd = await app.open_prompt("(unnamed)")
        await pilot.pause()
        assert "(unnamed)" in _rendered(odd)


@pytest.mark.asyncio
async def test_hint_row_sheds_whole_choices_and_keeps_submit_longest() -> None:
    """Design round 1 D6: the row cut mid-word (`esc ca…`), teaching a key that
    does not exist. It sheds whole choices instead, and `enter submit` is the
    last to go: a prompt that only tells you how to give up is one you cannot
    get past."""
    seen = []
    for width in (100, 24, 16, 12):
        app = _PromptHost()
        async with app.run_test(size=(width, 20)) as pilot:
            block = await app.open_prompt("Alibaba Cloud")
            await pilot.pause()
            lines = _rendered(block).split("\n")
            hint = [line for line in lines if "submit" in line or "cancel" in line]
            seen.append((width, hint))
            for line in hint:
                assert "ca…" not in line and "canc…" not in line, (width, line)

    assert any("cancel" in "".join(h) for _w, h in seen), seen
    assert any(h and "cancel" not in "".join(h) for _w, h in seen), seen


@pytest.mark.asyncio
async def test_a_long_paste_reports_its_length_at_narrow_widths() -> None:
    """Design round 1 D4: past the bar's useful length the COUNT is the message.

    Written bar-first it was truncated away, so at 40 columns a 33-character and
    a 300-character paste rendered as the same row and "did the whole key
    arrive" stopped being answerable exactly where the bar had stopped answering
    it.
    """
    from textual.events import Paste

    for width in (100, 40, 24, 16):
        app = _PromptHost()
        async with app.run_test(size=(width, 20)) as pilot:
            short = KeyPromptBlock("Alibaba Cloud")
            await app.mount(short)
            await pilot.pause()
            short.post_message(Paste("k" * 33))

            long = KeyPromptBlock("Alibaba Cloud")
            await app.mount(long)
            await pilot.pause()
            long.post_message(Paste("k" * 300))
            await pilot.pause()

            assert "33 chars" in _rendered(short), width
            assert "300 chars" in _rendered(long), width
            assert _rendered(short) != _rendered(long), width


@pytest.mark.asyncio
async def test_clearing_the_transcript_frees_the_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 F1: Ctrl+L during a login used to wedge `/login` for the session.

    The widget goes with the transcript, but the login worker parked on its
    future is NOT a turn, so neither the approval settle nor anything else in
    the clear hook reached it. The future stayed pending and `_login_lock`
    stayed held, so every later `/login` answered "a login is already in
    progress" and offered no prompt — unrecoverable without restarting.

    The second login at the end is the part that matters: settling the future
    alone would satisfy a weaker assertion while leaving the lock held.
    """
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")
        block = app._key_prompt
        assert block is not None

        await pilot.press("ctrl+l")
        for _ in range(30):
            await pilot.pause()
            await asyncio.sleep(0.01)

        assert block.answered, "the login is parked on a future nothing will resolve"
        assert block.wait().result() is None
        assert app._login_lock is None or not app._login_lock.locked()

        await _run_login(pilot, app, "xai")
        assert len(list(app.query(KeyPromptBlock))) == 1, "a later /login must still work"


@pytest.mark.asyncio
async def test_an_oauth_code_paste_is_echoed_and_named_a_code() -> None:
    """Round 1 F3: the prompt must not call an OAuth code an "API key".

    Anthropic's fallback pastes a single-use `code#state`, not a key. Calling it
    a key sends the user hunting a credential Anthropic never issued, and
    masking it hides a long opaque string they need to read back to check the
    paste landed whole. It is spent on redemption, so echoing costs nothing.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = KeyPromptBlock("Anthropic (Claude Pro/Max)", secret=False)
        await app.mount(block)
        await pilot.pause()
        text = _rendered(block)
        assert "authorization code" in text, text
        assert "API key" not in text, text

        for char in "abc123":
            await pilot.press(char)
        await pilot.pause()
        # Echoed, not masked — the opposite of the API-key case.
        assert "abc123" in _rendered(block)
        assert MASK_CHAR not in _rendered(block)


@pytest.mark.asyncio
async def test_secret_and_code_prompts_are_wired_from_the_registry(tmp_path: Path) -> None:
    """The `secret` choice is the registry's, not a per-call guess.

    Non-zero control on both sides: a paste-a-key provider masks, Anthropic's
    fallback echoes, so neither answer can be a constant.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(_LoginSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        alibaba = controller.provider("alibaba")
        anthropic = controller.provider("anthropic")
        assert alibaba is not None and anthropic is not None
        assert alibaba.paste_prompt_required and not anthropic.paste_prompt_required
        # Both are offered a prompt; they differ only in what it reads.
        assert app._login_callbacks(alibaba).on_manual_code_input is not None
        assert app._login_callbacks(anthropic).on_manual_code_input is not None


@pytest.mark.asyncio
async def test_a_superseded_prompt_does_not_claim_a_cancel() -> None:
    """Round 1 F4: a SUCCESSFUL Anthropic login painted "login cancelled".

    The paste races the loopback callback; when the browser redirect wins,
    `_await_code` cancels every waiter in a `finally`, so the prompt task is
    cancelled on the success path. Resolving that as a plain `None` printed
    "login cancelled" directly above the success notice.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = await app.open_prompt("Anthropic (Claude Pro/Max)")
        await pilot.pause()
        block.resolve(None, superseded=True)
        await pilot.pause()

        receipt = _rendered(block)
        assert "cancelled" not in receipt, receipt
        assert "no longer needed" in receipt, receipt
        # Still a cancel to the flow: no value was pasted.
        assert block.wait().result() is None

        # And the ordinary cancel still says cancelled, or the assertion above
        # would pass on a block that never says it either way.
        other = await app.open_prompt("Alibaba Cloud")
        await pilot.pause()
        other.resolve(None)
        await pilot.pause()
        assert "cancelled" in _rendered(other)


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


@pytest.mark.asyncio
async def test_a_rejected_paste_does_not_claim_a_successful_receipt() -> None:
    """Design round 1 D1(b): a paste the flow could not use settled as
    `✓ … code received (107 chars)`.

    The prompt settles the instant the value is handed over, but whether that
    value parses is decided one layer out in `_parse_pasted_callback`. So a
    mis-copied URL painted a success glyph, the word "received", and a
    character count that read as corroboration — directly above the notice
    saying the authorization had failed. The login is still live at that point
    (the flow re-prompts, the loopback callback is still racing), so this is
    not a cancel either.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = KeyPromptBlock("Z.AI", secret=False, sole_path=False)
        await app.mount(block)
        await pilot.pause()
        block.resolve("http://localhost:54548/callback?error=access_denied")
        await pilot.pause()

        # Before the flow rules on it, the receipt is the ordinary success one.
        assert "received" in _rendered(block)

        block.mark_unusable()
        await pilot.pause()
        receipt = _rendered(block)
        assert "received" not in receipt, receipt
        assert "cancelled" not in receipt, receipt
        assert "not usable" in receipt, receipt
        assert "still waiting for the browser" in receipt, receipt
        # The length is gone: it read as evidence the paste was fine.
        assert "chars" not in receipt, receipt

        # Control: an accepted paste is untouched by the same instrument, so
        # the assertions above distinguish the two outcomes rather than
        # describing every receipt.
        good = KeyPromptBlock("Z.AI", secret=False, sole_path=False)
        await app.mount(good)
        await pilot.pause()
        good.resolve("code#state")
        await pilot.pause()
        assert "received" in _rendered(good)


@pytest.mark.asyncio
async def test_mark_unusable_cannot_rewrite_a_cancel_or_a_live_prompt() -> None:
    """`mark_unusable` corrects a SUCCESS claim and nothing else.

    A declined paste and a still-open prompt both reach the flow's rejection
    path in principle (a host may call the hook at any time), and neither has a
    success claim to correct — turning either into "paste not usable" would
    invent a paste the user never made.

    Asserts the FLAG and not only the rendered row (agent review round 4, Z9):
    `_build`'s precedence tests `_superseded` and "nothing submitted" before it
    reaches the `_unusable` branch, so the renderer produces the right output
    even with the guard deleted. A rendering-only assertion therefore passes
    for a reason other than the one it names, and this test's whole subject is
    the guard.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        declined = KeyPromptBlock("Z.AI", secret=False, sole_path=False)
        await app.mount(declined)
        await pilot.pause()
        declined.resolve(None)
        await pilot.pause()
        before = _rendered(declined)
        declined.mark_unusable()
        await pilot.pause()
        assert declined._unusable is False, "a cancel was rewritten as an unusable paste"
        assert _rendered(declined) == before

        live = KeyPromptBlock("Z.AI", secret=False, sole_path=False)
        await app.mount(live)
        await pilot.pause()
        before_live = _rendered(live)
        live.mark_unusable()
        await pilot.pause()
        assert live._unusable is False, "a live prompt was rewritten as an unusable paste"
        assert _rendered(live) == before_live
        assert not live.answered


@pytest.mark.asyncio
async def test_a_settled_prompt_does_not_retain_the_pasted_value() -> None:
    """Agent review round 4, Z7: the block held the submitted value verbatim.

    `resolve` drops `_typed` precisely so a settled block in the transcript is
    not still holding the user's key, but the value was then kept under
    `_submitted` for the receipt's character count — and the app retains the
    last prompt to correct its receipt, so that reference outlived even
    `/clear`. The receipt only ever needed the LENGTH.
    """
    app = _PromptHost()
    async with app.run_test() as pilot:
        block = KeyPromptBlock("Alibaba Cloud")
        await app.mount(block)
        await pilot.pause()
        block.resolve("sk-SUPERSECRET-abcdef")
        await pilot.pause()

        held = [
            value
            for value in vars(block).values()
            if isinstance(value, str) and "SUPERSECRET" in value
        ]
        assert held == [], f"the settled block still holds the pasted value: {held}"

        # The receipt still reports the length, or the assertion above could be
        # satisfied by a block that simply stopped describing the paste.
        assert "21 chars" in _rendered(block)
