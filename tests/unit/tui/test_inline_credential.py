"""Inline ``/credential``: capture a secret out of the composer, never leak it.

The gesture: type ``/credential`` ANYWHERE in the composer, paste the secret,
watch it become ``[Credential #N, <len> chars]``, keep typing a description, and
send. The value goes to the session credential store under a generated name; the
model learns the NAME, the LENGTH and the operator's description, and never the
bytes.

The bulk of this file is the SAFETY property, because it is the one no other
marker type has. Every existing marker exists to be re-expanded into the prompt
at submit; a credential marker must be skipped at every seam where that happens.
There are six such seams, each pinned by a test below under "the six expansion
seams" — a regression at any one of them puts a plaintext secret somewhere it
can never be recalled from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult

from local_operator.tui.app import (
    CREDENTIAL_ARMED_NOTICE,
    CREDENTIAL_HELD_NOTICE,
    CREDENTIAL_PLACEHOLDER,
    CREDENTIAL_TYPING_NOTICE,
    OperatorApp,
)
from local_operator.tui.widgets.editor import (
    ASIDE_PLACEHOLDER,
    ATTACHMENT_MARKER,
    CREDENTIAL_KEY_PREFIX,
    CREDENTIAL_MASK_CHAR,
    Attachment,
    CredentialUnredacted,
    Editor,
    EditorSubmitted,
    Marked,
    PastedCredential,
    PastedText,
    _credential_label,
    _paste_label,
    credential_payloads,
    describe_unstored,
    expand_pastes,
    generate_credential_key,
    resolve_markers,
    strip_paste_citations,
    substitute_credentials,
)
from local_operator.variables import (
    CredentialStoreResult,
    VariableStore,
    normalize_credential_key,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _boot

#: A realistic 64-character token. Length matters: it is what the marker label
#: reports, and it is far below every size threshold the ordinary collapse uses
#: (COLLAPSE_ROWS=20, MIN_PASTE_ROWS=4) — which is exactly why the credential
#: mode gate has to be asked BEFORE the size question.
SECRET = "ghp_" + "x" * 56 + "TAIL"


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Editor()


async def _armed_capture(pilot, editor: Editor, prefix: str, secret: str = SECRET) -> None:
    """Type ``prefix``, arm with ``/credential``, and paste ``secret``."""
    for char in f"{prefix}/credential ":
        await pilot.press(char)
    await pilot.pause()
    editor.app.post_message(events.Paste(secret))
    await pilot.pause()
    await pilot.pause()


def _painted(app: App[None]) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


# -- the gesture --------------------------------------------------------------
@pytest.mark.asyncio
async def test_paste_while_armed_is_replaced_by_a_marker_naming_its_length() -> None:
    """The reported ask: the secret never appears in the composer at all."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        assert editor.text == "deploy with [Credential #1, 64 chars] "
        assert SECRET not in editor.text
        assert SECRET not in _painted(app)


@pytest.mark.asyncio
async def test_the_caret_lands_after_the_marker_so_typing_continues_inline() -> None:
    """ "Keep typing a description in the same message" is the whole gesture."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        for char in " for staging":
            await pilot.press(char)
        await pilot.pause()
        assert editor.text == "deploy with [Credential #1, 64 chars]  for staging"


@pytest.mark.asyncio
async def test_arming_works_mid_line_not_only_at_the_start() -> None:
    """``/credential`` ANYWHERE in the line, per the operator's request."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "here is the key we use in CI, ")
        assert editor.text.startswith("here is the key we use in CI, [Credential #1,")
        assert SECRET not in editor.text


@pytest.mark.asyncio
async def test_the_mid_line_command_picker_opens_and_highlights_credential() -> None:
    """The picker's mid-line predicate, pinned against the REAL app.

    Scope note: the picker already supported an inline slash token, so this
    asserts rather than adds. It is pinned because the whole gesture is
    discoverable only through this list.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "check this /cred":
            await pilot.press(char)
        await pilot.pause()
        assert editor._picker.is_open()
        assert editor._picker_query() == "cred"
        assert editor._picker.highlighted_name() == "cred"


@pytest.mark.asyncio
async def test_a_short_secret_is_captured_though_it_is_far_below_the_collapse_size() -> None:
    """The mode gate must precede the size question.

    A one-line secret is nowhere near ``MIN_PASTE_ROWS``; if the size branch
    were asked first the value would be inserted verbatim and never redacted.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "", secret="short-key")
        assert editor.text == "[Credential #1, 9 chars] "


@pytest.mark.asyncio
async def test_an_unarmed_paste_of_the_same_secret_is_left_alone() -> None:
    """The gate is the ARMING TOKEN, not the payload: no paste is special."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "just some text ":
            await pilot.press(char)
        app.post_message(events.Paste("short-key"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text == "just some text short-key"
        assert not editor._attachments


@pytest.mark.parametrize("start_armed", [False, True], ids=["cold", "armed"])
@pytest.mark.asyncio
async def test_a_mention_of_the_command_in_text_never_typed_through_the_arm_does_not_arm(
    start_armed: bool,
) -> None:
    """A draft that ARRIVES holding the word never passed through the gesture.

    Recalled history, a restored draft and a pasted paragraph all land through
    ``load_text``, and none of them is the operator typing the arming gesture.
    Arming is still :data:`CREDENTIAL_ARM` at the caret and nothing else — what
    changed in design round 1 (D2) is how long an arm SURVIVES once taken, not
    how one is taken.

    PARAMETERISED OVER BOTH STARTING STATES because the single cold case could
    not fail. ``_sync_credential_arm`` has two branches and the cold editor only
    ever reaches the first ("should this arm?"); every way this property has
    actually broken lives in the second ("where did the armed token go?"), where
    a latched arm re-anchored onto a token in text that merely ARRIVED. The test
    asserted the right property against the one path that could not violate it
    (review round 2, R6; QA round 2, Q4).
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        if start_armed:
            for char in "/credential ":
                await pilot.press(char)
            await pilot.pause()
            assert editor.credential_armed(), "precondition: the gesture armed"
        editor.load_text("fix the /credential command please ")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert not editor.credential_armed()
        app.post_message(events.Paste("an ordinary paste"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text.endswith("an ordinary paste")
        assert not any(
            isinstance(value, PastedCredential) for value in editor._attachments.values()
        )


@pytest.mark.asyncio
async def test_an_arm_does_not_migrate_to_another_token_when_the_buffer_is_replaced() -> None:
    """R6b: the latch tracks THE token that armed, not the first one in the text.

    The re-anchor used to be a whole-buffer ``search``, so replacing the buffer
    while armed handed the arm to whatever ``/credential`` the new text happened
    to contain — and the operator's next ordinary paste was swallowed as a
    secret they can neither expand nor recover (review round 2, R6b).
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed()
        editor.load_text("unrelated draft that says /cred somewhere in it")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert not editor.credential_armed(), "the arm did not survive onto another token"
        app.post_message(events.Paste("ordinary pasted text"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text.endswith("ordinary pasted text"), "the paste is visible, not stashed"
        assert not editor._attachments


@pytest.mark.asyncio
async def test_an_arm_is_not_inherited_across_history_recall() -> None:
    """R6c: recalling a prompt that MENTIONS the command must not stay armed.

    The route an operator reaches without doing anything unusual: arm, press Up
    to check what they sent last time, paste. The recalled text arrives through
    ``load_text`` and is not a gesture, so the arm ends with it.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor._history.append("how do I use /cred to store a token")
        for char in "/credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed()
        await pilot.press("up")
        await pilot.pause()
        assert editor.text == "how do I use /cred to store a token"
        assert not editor.credential_armed(), "the recalled prompt did not inherit the arm"
        app.post_message(events.Paste("ordinary pasted text"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text.endswith("ordinary pasted text")
        assert not any(
            isinstance(value, PastedCredential) for value in editor._attachments.values()
        )


@pytest.mark.asyncio
async def test_a_restored_draft_does_not_inherit_a_live_arm() -> None:
    """R6c/Q4: parking a draft over an armed composer must not carry the arm.

    ``_load_editor_draft`` restores a parked draft straight into the composer
    with no empty-buffer guard, so this is reachable in the product rather than
    only through the widget — and a draft that merely mentions the command
    would otherwise swallow the operator's next paste.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed()
        # Exactly what `_load_editor_draft` does with a parked draft.
        editor.load_text("ask about /credential rotation before the release")
        await pilot.pause()
        assert not editor.credential_armed()
        app.post_message(events.Paste("a 200-line deploy log"))
        await pilot.pause()
        await pilot.pause()
        assert "a 200-line deploy log" in editor.text
        assert not any(
            isinstance(value, PastedCredential) for value in editor._attachments.values()
        )


@pytest.mark.asyncio
async def test_the_arm_stays_on_its_own_token_when_an_earlier_one_is_typed_in() -> None:
    """R6b by TYPING — the route ``load_text`` disarming does not cover.

    The operator arms mid-line, then goes back to the front of the draft and
    types a sentence that also mentions the command. Under a whole-buffer
    ``search`` the arm jumped to that EARLIER token, and the capture then spliced
    the marker into the wrong half of the sentence — rewriting the operator's
    words where they were not pasting, while the token they actually armed sat
    untouched. This is the case that makes the re-anchor an identity question
    rather than a text-arrival question (review round 2, R6b).
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "deploy /credential ":
            await pilot.press(char)
        await pilot.pause()
        armed_at = editor._credential_arm
        assert armed_at is not None
        editor.move_cursor((0, 0))
        for char in "ask about /cred later ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.text == "ask about /cred later deploy /credential "
        assert editor._credential_arm is not None
        # The span carries the token's trailing whitespace too (see
        # `_arm_credential`), which is what the capture splices out.
        armed_text = editor.text[slice(*editor._credential_arm)]
        assert armed_text.strip() == "/credential", f"arm stayed on its token: {armed_text!r}"
        assert editor._credential_arm[0] == editor.text.index("deploy ") + len("deploy ")

        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text
        # The marker replaced the ARMED token; the earlier mention is untouched.
        assert editor.text.startswith("ask about /cred later deploy [Credential #1,")


@pytest.mark.asyncio
async def test_the_armed_token_is_still_tracked_while_text_before_it_is_edited() -> None:
    """The case the whole-buffer search was written for still works.

    Anchoring to the armed occurrence must not break the reason the re-anchor
    exists: the token SLIDES as the operator edits text before it, and a stale
    span would splice the wrong range out of the buffer at capture time.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "deploy /credential ":
            await pilot.press(char)
        await pilot.pause()
        armed_at = editor._credential_arm
        assert armed_at is not None
        editor.move_cursor((0, 0))
        for char in "please ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed(), "typing before the token keeps the arm"
        assert editor._credential_arm is not None
        assert editor._credential_arm[0] == armed_at[0] + len("please "), "it tracked the slide"
        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text, "still captures at the tracked span"
        assert editor.text.startswith("please deploy [Credential #1,")


#: Single edits that move the armed token FAR in one step, each a keystroke an
#: operator reaches without doing anything unusual. The parameter is what the
#: token slides by, and the values straddle the ``_ARM_DRIFT = 64`` boundary
#: that used to sit in ``_relocate_armed_token``: 63 captured and 64 leaked.
_ONE_BIG_EDIT = (
    ("shift+home", ("shift+home", "backspace")),
    ("delete_line", ("ctrl+shift+k",)),
    ("kill_to_start", ("ctrl+u",)),
    ("cut_block", ("shift+home", "ctrl+x")),
)


@pytest.mark.parametrize("name,keys", _ONE_BIG_EDIT, ids=[case[0] for case in _ONE_BIG_EDIT])
@pytest.mark.parametrize("pad", (63, 64, 70, 120))
@pytest.mark.asyncio
async def test_one_large_edit_before_the_token_cannot_disarm_it(
    pad: int, name: str, keys: tuple[str, ...]
) -> None:
    """R7 / QA Q5: a SINGLE edit may move the token ANY distance and stay armed.

    ``_relocate_armed_token`` used to bound the move by ``_ARM_DRIFT = 64`` and
    return ``None`` past it, which ``_sync_credential_arm`` reads as "the
    operator deleted the gesture" — so it disarmed while the token was STILL
    PLAINLY IN THE BUFFER, and silently, because ``reason="gone"`` has no
    notice branch. The next paste landed as ordinary text: in the composer, in
    scrollback, and in the prompt sent to the model in plaintext.

    Measured at exactly that boundary: 63 characters above the token captured,
    64 leaked. That is why this is parameterised over the distance rather than
    asserting one case — a window of any size has the same cliff one character
    further along, which is why the fix removes the distance test rather than
    widening it.

    The delta's own drift test could not catch this: it types seven characters
    ONE KEYSTROKE AT A TIME, so the token never moves more than one character
    per sync and no single edit ever crosses the window.
    """
    app = Host()
    # Wide enough that the padding line is ONE visual row for every `pad` here:
    # `home` moves to the start of the WRAPPED row, so a narrower composer would
    # select only part of the line and the token would move less than `pad`.
    async with app.run_test(size=(200, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        # Context ABOVE the token, then the gesture — all typed, because the
        # arm is only reachable by keystrokes (`load_text` disarms).
        for char in "x" * pad:
            await pilot.press(char)
        await pilot.press("shift+enter")
        for char in "deploy /credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed(), "precondition: the gesture armed"

        # ONE edit, removing the whole line above and moving the token by pad+1.
        editor.move_cursor((0, pad))
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
        await pilot.pause()
        assert editor.credential_armed(), f"{name} moved the token {pad} chars; it is still there"

        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text, "the paste was CAPTURED, not left in plaintext"
        assert "[Credential #1, 64 chars]" in editor.text
        assert SECRET not in _painted(app), "and nothing painted the secret"


@pytest.mark.asyncio
async def test_typing_over_a_large_selection_before_the_token_does_not_disarm_it() -> None:
    """The same cliff reached by REPLACING a block rather than deleting it.

    One keystroke over a wide selection is a single edit that both removes and
    inserts, so the token's net move is the block's length. It is the route
    that looks least like an edit to the token — the operator is retyping a
    sentence somewhere else entirely — and under the drift window it disarmed
    just as silently.
    """
    app = Host()
    # Wide enough that the 150-character line is ONE visual row: `home` moves to
    # the start of the WRAPPED row, so a narrower composer would select only
    # part of it and measure a shorter move than the test claims.
    async with app.run_test(size=(200, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "z" * 150:
            await pilot.press(char)
        await pilot.press("shift+enter")
        for char in "deploy /credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed()

        # Select the whole 150-character line above and type ONE character over
        # it: a single edit that moves the token by 149.
        editor.move_cursor((0, 150))
        await pilot.pause()
        await pilot.press("shift+home")
        await pilot.press("q")
        await pilot.pause()
        assert editor.text.startswith("q\n"), "precondition: the block really was replaced"
        assert editor.credential_armed(), "replacing a block before the token keeps the arm"

        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text, "the secret was captured, not left in plaintext"
        assert "[Credential #1, 64 chars]" in editor.text
        assert SECRET not in _painted(app)


@pytest.mark.asyncio
async def test_a_large_edit_then_submit_never_discloses_the_secret() -> None:
    """R7/Q5 as the DISCLOSURE property, on the real app and through submit.

    ``credential_armed() is True`` is the mechanism; this is the thing that
    actually matters and the thing that regressed — the secret must not reach
    the model's prompt, the transcript, or the painted screen. Asserted end to
    end because every one of those is a separate seam from the latch.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "x" * 120:
            await pilot.press(char)
        await pilot.press("shift+enter")
        for char in "deploy /credential ":
            await pilot.press(char)
        await pilot.pause()
        assert editor.credential_armed()

        # ONE edit, moving the token by 121. `delete_line` rather than
        # `shift+home`: it takes the whole LOGICAL line, so the move does not
        # depend on where the composer soft-wraps it at this width.
        editor.move_cursor((0, 120))
        await pilot.pause()
        await pilot.press("ctrl+shift+k")
        await pilot.pause()
        assert editor.text.startswith("deploy /credential"), "precondition: the line above is gone"

        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        for char in " the staging key":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()

        sent = " ".join(str(prompt) for prompt in session.prompts)
        assert SECRET not in sent, "the secret must NEVER reach the model"
        assert "the staging key" in sent, "precondition: the message really was sent"
        keys = session.variables.credential_names()
        assert len(keys) == 1, "it went to the store instead"
        assert session.variables.credential_env()[keys[0]] == SECRET
        assert keys[0] in sent, "the model learns the NAME"
        assert SECRET not in _painted(app), "and it is not on screen"


# -- the edge cases the gesture has to decide ---------------------------------
@pytest.mark.asyncio
async def test_an_empty_paste_while_armed_creates_no_zero_length_credential() -> None:
    """A blank value is refused by the store anyway; advertising it would lie."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential ":
            await pilot.press(char)
        app.post_message(events.Paste("   \n  "))
        await pilot.pause()
        await pilot.pause()
        assert not editor._attachments
        assert "Credential #" not in editor.text


@pytest.mark.asyncio
async def test_pasting_twice_while_armed_captures_two_distinct_credentials() -> None:
    """Two pastes are two secrets: the first marker keeps its own payload."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "", secret="first-secret")
        for char in "/credential ":
            await pilot.press(char)
        app.post_message(events.Paste("second-secret"))
        await pilot.pause()
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert [payload.value for payload in payloads] == ["first-secret", "second-secret"]
        assert len({payload.key for payload in payloads}) == 2


@pytest.mark.asyncio
async def test_backspacing_the_marker_forgets_the_stashed_secret() -> None:
    """No orphaned secret survives in memory after the marker is deleted."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "")
        assert any(isinstance(value, PastedCredential) for value in editor._attachments.values())
        # Backspace over the trailing space, then over the marker itself, which
        # deletes atomically as one token.
        await pilot.press("backspace")
        await pilot.press("backspace")
        await pilot.pause()
        assert "Credential #" not in editor.text
        assert not editor._attachments


@pytest.mark.asyncio
async def test_ctrl_r_refuses_to_expand_a_credential_back_into_the_buffer() -> None:
    """The escape hatch for a paste must not become a way to reveal a secret."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "")
        # SkipAction is the refusal: the key falls through the binding chain
        # rather than silently doing nothing, exactly as it does on an image.
        with pytest.raises(SkipAction):
            editor.action_expand_paste()
        await pilot.pause()
        assert SECRET not in editor.text
        assert "[Credential #1, 64 chars]" in editor.text


# -- the random name ----------------------------------------------------------
def test_a_generated_key_round_trips_through_normalize_credential_key() -> None:
    """A name the store would renormalise would advertise the wrong key."""
    for _ in range(200):
        key = generate_credential_key()
        assert key.startswith(CREDENTIAL_KEY_PREFIX)
        assert normalize_credential_key(key) == key


def test_a_generated_key_avoids_names_already_taken() -> None:
    """A collision would SILENTLY REPLACE a live credential."""
    taken = {generate_credential_key() for _ in range(50)}
    for _ in range(50):
        assert generate_credential_key(taken) not in taken


def test_generated_keys_are_not_all_the_same() -> None:
    assert len({generate_credential_key() for _ in range(100)}) > 90


# -- the six expansion seams --------------------------------------------------
# Each existing marker type is designed to be re-expanded into the prompt at
# submit. A credential must be skipped at EVERY seam where that happens, and
# these six are all of them. They are listed individually rather than in a loop
# so a failure names the seam that leaked.
def _mixed() -> tuple[str, dict[int, Marked]]:
    """A draft citing a credential, a collapsed paste and an image at once."""
    from local_operator.harness.types import ImageContent

    text = (
        "look at [Paste #1, 9 lines] and [Credential #2, 64 chars] "
        "and [Image #3, 10x20] together"
    )
    attachments: dict[int, Marked] = {
        1: PastedText("PAYLOAD-TEXT", "[Paste #1, 9 lines]"),
        2: PastedCredential(SECRET, "LOP_SECRET_ABCD2345", "[Credential #2, 64 chars]"),
        3: Attachment(
            ImageContent(data="AAAA", mime_type="image/png"),
            "[Image #3, 10x20]",
        ),
    }
    return text, attachments


def test_seam_1_expand_pastes_does_not_splice_a_credential_value() -> None:
    """``expand_pastes`` is THE prompt-side expansion: /goal, /team, $skill…"""
    text, attachments = _mixed()
    expanded = expand_pastes(text, attachments)
    assert SECRET not in expanded
    assert "PAYLOAD-TEXT" in expanded, "the ordinary paste must still expand"
    assert "[Credential #2, 64 chars]" in expanded


def test_seam_2_resolve_markers_does_not_resolve_a_credential_to_an_image() -> None:
    text, attachments = _mixed()
    images = resolve_markers(text, attachments)
    assert len(images) == 1, "only the real image resolves"


def test_seam_3_strip_paste_citations_removes_the_credential_from_history() -> None:
    """History cannot carry the payload, so it must not carry the citation."""
    text, attachments = _mixed()
    stripped = strip_paste_citations(text, attachments)
    assert SECRET not in stripped
    assert "Credential #2" not in stripped
    assert "Paste #1" not in stripped
    assert "[Image #3, 10x20]" in stripped, "images are deliberately left alone"


def test_seam_4_substitute_credentials_emits_the_name_and_never_the_value() -> None:
    """The one rewrite a credential marker gets: name + length, no bytes."""
    text, attachments = _mixed()
    substituted = substitute_credentials(text, attachments)
    assert SECRET not in substituted
    assert "LOP_SECRET_ABCD2345" in substituted
    assert "64 chars" in substituted


def test_seam_5_the_draft_store_never_writes_a_secret_to_disk(tmp_path: Path) -> None:
    """The sidebar spills unsubmitted drafts to a temp JSON file."""
    from local_operator.tui.session_drafts import SessionDraftStore
    from local_operator.tui.session_interaction import SessionDraft

    text, attachments = _mixed()
    store = SessionDraftStore()
    payload = store._encode_draft(SessionDraft(text=text, attachments=dict(attachments)))
    encoded = json.dumps(payload)
    assert SECRET not in encoded
    assert "LOP_SECRET_ABCD2345" in encoded, "the key is kept so the marker still paints"
    assert "PAYLOAD-TEXT" in encoded, "an ordinary paste is still spilled"
    # And it survives the round trip as a valued-less credential.
    restored = store._decode_attachments(payload["attachments"])
    assert isinstance(restored[2], PastedCredential)
    assert restored[2].value == ""
    assert restored[2].key == "LOP_SECRET_ABCD2345"


def test_seam_6_the_marker_grammar_keeps_its_groups_named() -> None:
    """A positional group here silently makes ``index`` mean ``'Credential'``."""
    match = ATTACHMENT_MARKER.search("x [Credential #7, 64 chars] y")
    assert match is not None
    assert match.group("kind") == "Credential"
    assert match.group("index") == "7"


# -- the submit path, end to end ----------------------------------------------
@pytest.mark.asyncio
async def test_submitting_stores_the_secret_and_sends_only_its_name() -> None:
    """The whole property, on the real app: value to the store, name to the model."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        for char in " use this for staging":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        keys = session.variables.credential_names()
        assert len(keys) == 1
        assert keys[0].startswith(CREDENTIAL_KEY_PREFIX)
        assert session.variables.credential_env()[keys[0]] == SECRET

        sent = " ".join(str(prompt) for prompt in session.prompts)
        assert SECRET not in sent
        assert keys[0] in sent, "the model must learn the NAME"
        assert "64 chars" in sent, "…and the length"
        assert "use this for staging" in sent, "…and the operator's description"
        assert SECRET not in _painted(app)


@pytest.mark.asyncio
async def test_the_arming_token_is_consumed_so_a_leading_gesture_is_not_dispatched() -> None:
    """A leading ``/credential`` + paste must SEND, not run the slash command.

    Without consuming the token the submitted line would start with
    ``/credential`` and ``_dispatchable_slash`` would route it to the masked
    prompt with the marker as its KEY.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "")
        for char in " the staging deploy key":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert session.variables.credential_names(), "the secret was stored"
        sent = " ".join(str(prompt) for prompt in session.prompts)
        assert "the staging deploy key" in sent, "the message was SENT, not dispatched"
        assert SECRET not in sent


@pytest.mark.asyncio
async def test_the_secret_never_reaches_the_prompt_history() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        for char in " for staging":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert editor._history
        assert not any(SECRET in entry for entry in editor._history)
        assert not any("Credential #" in entry for entry in editor._history)


@pytest.mark.asyncio
async def test_a_credential_inside_a_goal_argument_is_not_expanded() -> None:
    """``prompt_arg`` is flag-derived; a credential must not ride into /goal."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "/goal ship it with /credential ":
            await pilot.press(char)
        await pilot.pause()
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        goal = getattr(session, "goal", "") or ""
        assert SECRET not in goal
        assert SECRET not in _painted(app)


@pytest.mark.asyncio
async def test_the_secret_never_reaches_the_session_journal() -> None:
    """The journal records the KEY; a resume must not replay the value."""
    session = FakeSession()
    recorded: list[tuple[str, str]] = []

    def _journal(key: str, *, action: str = "stored", replaced: bool = False) -> None:
        recorded.append((key, action))

    session.journal_credential_change = _journal  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
    assert recorded, "the capture must be announced to the model"
    assert all(key.startswith(CREDENTIAL_KEY_PREFIX) for key, _ in recorded)
    assert not any(SECRET in key for key, _ in recorded)


@pytest.mark.asyncio
async def test_arming_from_inside_another_commands_argument_slot_still_captures() -> None:
    """No existing command combines an argument list with a modal paste state.

    The scout flagged this as genuinely unknown behaviour, so it is pinned
    rather than reasoned about: with the caret sitting in ``/model``'s argument
    slot, an inline ``/credential`` still arms and the paste is still captured.
    The gate is a caret-anchored text predicate, so it does not care which list
    the picker happens to be showing — which is the property that makes the
    combination safe.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _armed_capture(pilot, editor, "/model gpt ")
        assert editor.text == "/model gpt [Credential #1, 64 chars] "
        assert SECRET not in editor.text


@pytest.mark.asyncio
async def test_a_credential_captured_while_the_aside_is_open_does_not_leak_into_it() -> None:
    """The ``/btw`` exit returns early, so the capture must precede it."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "/btw ":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        if not app._aside_is_open():
            pytest.skip("the aside did not open in this configuration")
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "what is ")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        assert SECRET not in _painted(app)
        asked = " ".join(str(question) for question in getattr(session, "aside_questions", []))
        assert SECRET not in asked


@pytest.mark.asyncio
async def test_the_secret_never_reaches_a_shell_command() -> None:
    """Bang-mode runs the submitted line as a command — so it must not hold bytes.

    The capture runs ahead of the shell branch, so what reaches
    ``_run_shell_command`` (and therefore the operator's shell history) is the
    substituted name, never the value.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    ran: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_shell_command = lambda command: ran.append(command)  # type: ignore[assignment]
        editor = app._editor()
        editor.focus()
        editor.set_shell_mode(True)
        await pilot.pause()
        await _armed_capture(pilot, editor, "echo ")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
    assert not any(SECRET in command for command in ran), ran


# -- degraded paths -----------------------------------------------------------
def test_the_store_key_is_usable_by_a_real_variable_store() -> None:
    """The generated name must actually work in the store it is destined for."""
    store = VariableStore()
    key = generate_credential_key()
    result = store.store_credential(key, SECRET, "command")
    assert result.ok
    assert result.credential is not None
    assert result.credential.key == key
    assert store.credential_env()[key] == SECRET
    assert store.credential_names() == [key]


class _ViewerSession(FakeSession):
    """A follower: no local variable store, exactly as a real viewer has none.

    A SUBCLASS rather than a patched attribute on the shared double, so the
    viewer shape cannot leak into any other test in the suite.
    """

    @property
    def variables(self):  # type: ignore[override]
        return None


@pytest.mark.asyncio
async def test_a_session_without_a_store_says_so_instead_of_silently_dropping() -> None:
    """A viewer must never advertise a key ``bash`` cannot read."""
    session = _ViewerSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        painted = _painted(app)
        assert SECRET not in painted
        sent = " ".join(str(prompt) for prompt in session.prompts)
        assert SECRET not in sent
        assert "NOT stored" in sent, "the model must not be promised a key nothing holds"


# -- the degraded store paths (review round 1 R1/R2, QA round 1 Q1-Q3) --------
class _RefusingStore(VariableStore):
    """A store that refuses named keys, so a PARTIAL store is reachable in test.

    The shape this doubles is not hypothetical: a draft that spilled to the
    sidebar's temp JSON comes back with ``value=""`` by design and the real
    ``store_credential`` refuses a blank. This double reaches the same branch
    without depending on the spill machinery, so the test pins the CITATION
    rule rather than one route to it.
    """

    def __init__(self, *args, refuse: set[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refuse = refuse or set()

    def store_credential(self, raw_key, value, source="command"):  # type: ignore[override]
        if raw_key in self.refuse:
            return CredentialStoreResult(ok=False, reason="empty-value")
        return super().store_credential(raw_key, value, source)


@pytest.mark.asyncio
async def test_a_partial_store_never_advertises_the_key_that_did_not_land() -> None:
    """THE regression test: some stored, some refused, each citation its own truth.

    Two independent review streams found this one defect (review round 1 R1,
    QA round 1 Q1): the guard was ``if not stored`` — the ALL-failed case — so a
    single success rewrote EVERY citation to the confident "available to bash as
    $KEY" form, including credentials the store had refused. The model was told
    a secret was usable that ``credential_env()`` does not contain, and would
    then debug an auth failure with no cause in the world it can see.

    Asserted per citation, not in aggregate, because an aggregate assertion is
    exactly the mistake the code made.
    """
    session = FakeSession()
    # Seeded through the double's own lazy slot rather than the read-only
    # property, so the refusing store IS the session's store everywhere.
    session._variables = _RefusingStore(cwd="/tmp", env={})
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "client id ", secret="A" * 45)
        # The FIRST capture is the one that will be refused, so the test also
        # pins that the two citations are told apart by identity rather than by
        # position — the failure mode a "first one wins" fix would still have.
        first_key = next(
            payload.key
            for payload in editor._attachments.values()
            if isinstance(payload, PastedCredential)
        )
        session.variables.refuse = {first_key}
        for char in " and secret /credential ":
            await pilot.press(char)
        app.post_message(events.Paste("B" * 44))
        await pilot.pause()
        await pilot.pause()
        for char in " for the oauth app":
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        stored = session.variables.credential_names()
        assert len(stored) == 1, "exactly one landed"
        second_key = stored[0]
        assert second_key != first_key

        sent = " ".join(str(prompt) for prompt in session.prompts)
        # The one that LANDED is advertised, with the env var the agent can use.
        assert f"available to bash and eval as ${second_key}" in sent
        # The one that did NOT is named as unstored, and its key is never
        # offered to the agent in the confident form.
        assert f"available to bash and eval as ${first_key}" not in sent
        assert "NOT stored" in sent
        # And the phantom key is not promised anywhere in the outgoing prompt.
        assert first_key not in sent
        assert "A" * 45 not in sent
        assert "B" * 44 not in sent
        # The operator's own words survive between the two citations.
        assert "for the oauth app" in sent


@pytest.mark.asyncio
async def test_a_store_refusal_is_reported_to_the_operator_not_only_the_model() -> None:
    """Q2: the composer clears and the marker vanishes — silence is not an option.

    The viewer branch already emitted a notice; the refusal branch emitted
    nothing, so an operator whose store said no was left believing they had
    handed over a credential.
    """
    session = FakeSession()
    # Seeded through the double's own lazy slot rather than the read-only
    # property, so the refusing store IS the session's store everywhere.
    session._variables = _RefusingStore(cwd="/tmp", env={})
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        key = next(
            payload.key
            for payload in editor._attachments.values()
            if isinstance(payload, PastedCredential)
        )
        session.variables.refuse = {key}
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        painted = _painted(app)
        assert "could not be stored" in painted, "the OPERATOR is told, not just the model"
        assert key in painted, "…and which credential it was"
        assert SECRET not in painted


def test_the_refusal_phrase_names_the_cause_that_actually_applied() -> None:
    """Q3: "no session store available" is false when a store refused the write."""
    assert "no session store available" in describe_unstored(None)
    # A present store that refused must not blame an absent one.
    refused = describe_unstored("empty-value")
    assert "no session store available" not in refused
    assert "NOT stored" in refused, "the outcome leads, whatever the cause"


# -- the armed state is legible and cannot be disarmed by accident (D1/D2) ----
@pytest.mark.asyncio
async def test_the_arm_survives_a_newline_and_a_typed_word() -> None:
    """D2: the two ordinary edits that used to disarm SILENTLY into a plaintext paste.

    The failure is asymmetric — a false negative puts the secret on screen and
    in scrollback where nothing can recall it, while a false positive costs one
    backspace — so the arm survives ordinary typing and only a flag ends it.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        assert editor.credential_armed()
        await pilot.press("shift+enter")
        assert editor.credential_armed(), "a newline must not disarm"
        for char in "the prod key ":
            await pilot.press(char)
        assert editor.credential_armed(), "typing a description must not disarm"
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text, "the secret must NOT be in the buffer"
        assert "[Credential #1, 64 chars]" in editor.text


@pytest.mark.asyncio
async def test_a_flag_argument_disarms_so_the_command_verbs_stay_reachable() -> None:
    """The one continuation that is unambiguously the COMMAND, not a description.

    Credential verbs are flag-shaped precisely so they cannot collide with a
    key (keys normalize to ``[A-Z0-9_]`` and cannot begin with ``-``), so a
    leading ``-`` is a safe partition.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential ":
            await pilot.press(char)
        assert editor.credential_armed()
        await pilot.press("-")
        assert not editor.credential_armed(), "an argument is the command, not an arm"


@pytest.mark.asyncio
async def test_the_armed_token_is_painted_even_mid_line() -> None:
    """D2: mid-line arming is the headline of the gesture and had NO ink at all.

    ``_compute_slash_runs`` is a leading-command rule, so a mid-line
    ``/credential`` returned ``None`` and the composer was byte-identical armed
    and disarmed.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        runs = editor._slash_runs()
        assert runs is not None, "a mid-line arm must paint"
        line, spans = runs
        assert line == 0
        assert [component for _, _, component in spans] == ["text-area--credential-armed"]
        start, end, _ = spans[0]
        assert editor.text[start:end].startswith("/credential")


@pytest.mark.asyncio
async def test_arming_suppresses_the_destructive_argument_rows() -> None:
    """D1: arm + Enter + Enter used to complete and RUN ``--forget-all``.

    The row was preselected with a ghost under the caret, no confirmation and
    no undo, and Tab on the same row consumed the arming token so the next
    paste landed in plaintext. Both keys act on a highlighted row, so removing
    the rows while armed closes both.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        # Seed a live credential so a wipe would be visible.
        session.variables.store_credential("LOP_SECRET_LIVEKEY1", "kept", "command")
        for char in "/credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor.credential_armed()
        assert editor.picker.highlighted_name() is None, "no row is preselected while armed"
        await pilot.press("enter")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        assert session.variables.credential_names() == [
            "LOP_SECRET_LIVEKEY1"
        ], "the live credential survives the reflex double-Enter"


@pytest.mark.asyncio
async def test_tab_while_armed_cannot_consume_the_token_into_a_plaintext_paste() -> None:
    """D1(b): Tab accepted the ghost, which disarmed, and the next paste leaked."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        await pilot.press("tab")
        for _ in range(4):
            await pilot.pause()
        assert "--forget-all" not in editor.text, "no ghost to accept"
        assert editor.credential_armed(), "Tab must not silently disarm"
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert SECRET not in editor.text
        assert SECRET not in _painted(app)


# -- the small ones (R3, R5, D3, D6) ------------------------------------------
def test_a_generated_key_avoids_names_the_session_store_already_holds() -> None:
    """R3: the composer's own map resets each submit, so it cannot be the whole set."""
    taken = {f"{CREDENTIAL_KEY_PREFIX}{'A' * 8}"}
    for _ in range(50):
        assert generate_credential_key(taken) not in taken


def test_a_multi_line_secret_reports_characters_not_lines() -> None:
    """R5: a line count is the weakest integrity check exactly where it matters.

    A PEM block reads ``27 lines`` whether or not its tail arrived; the
    character count moves for every lost byte, and the length is the operator's
    only channel for noticing a wrong or truncated capture.
    """
    pem = "-----BEGIN KEY-----\n" + "\n".join("x" * 40 for _ in range(6)) + "\n-----END KEY-----"
    assert "lines" not in _credential_label(pem)
    assert _credential_label(pem) == f"{len(pem)} chars"
    # The paste vocabulary is deliberately unchanged for ordinary pastes.
    assert _paste_label(pem).endswith("lines")


@pytest.mark.asyncio
async def test_a_blank_paste_while_armed_says_so_and_stays_armed() -> None:
    """D3: the operator who fumbled the copy saw neither "captured" nor "failed"."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "/credential ":
            await pilot.press(char)
        app.post_message(events.Paste("   \n  "))
        for _ in range(6):
            await pilot.pause()
        assert not editor._attachments
        assert editor.credential_armed(), "the arm survives, and the notice says so"
        assert "still armed" in _painted(app)


@pytest.mark.asyncio
async def test_ctrl_r_on_a_credential_says_why_instead_of_doing_nothing() -> None:
    """D6: a bare SkipAction left a byte-identical frame, read as a missed key."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _armed_capture(pilot, editor, "deploy with ")
        editor.move_cursor(editor._offset_to_location(editor.text.index("[Credential") + 2))
        # The ACTION, as its sibling test drives it: `pilot.press("ctrl+r")`
        # does not reach the binding under this host, and the refusal (a
        # SkipAction that falls through the binding chain) is the behaviour
        # being asserted around.
        with pytest.raises(SkipAction):
            editor.action_expand_paste()
        for _ in range(8):
            await pilot.pause()
        painted = _painted(app)
        assert SECRET not in editor.text, "the refusal still holds"
        assert SECRET not in painted
        assert "can't be expanded" in painted


@pytest.mark.asyncio
async def test_arming_does_not_relabel_a_composer_another_mode_owns() -> None:
    """The placeholder is a SHARED channel; the armed copy only borrows it.

    Bang-mode, the aside and the read-only subagent page each own the
    placeholder in their own mode, so restoring the RESTING copy on disarm
    would silently relabel a surface this feature has no business touching.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        editor.placeholder = ASIDE_PLACEHOLDER
        for char in "/credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor.credential_armed()
        assert editor.placeholder == ASIDE_PLACEHOLDER, "the aside keeps its own voice"
        await pilot.press("-")
        for _ in range(4):
            await pilot.pause()
        assert not editor.credential_armed()
        assert editor.placeholder == ASIDE_PLACEHOLDER, "…and keeps it on the way out"

        editor.load_text("")
        editor.placeholder = editor.resting_placeholder
        await pilot.pause()
        for char in "/credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor.placeholder == CREDENTIAL_PLACEHOLDER, "a resting composer does say it"


@pytest.mark.asyncio
async def test_the_armed_placeholder_is_state_only_and_never_paints() -> None:
    """D8: the placeholder cannot render, so nothing may count it as a channel.

    A placeholder needs an EMPTY buffer and arming needs the token IN it, so the
    two conditions are mutually exclusive. The old assertion checked the
    ATTRIBUTE — which is set correctly — and so passed while the channel was
    invisible. This pins what is PAINTED, in both arming forms, so the comment
    claiming two live channels cannot silently become false.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for form in ("/credential ", "deploy with /credential "):
            editor.load_text("")
            editor.placeholder = editor.resting_placeholder
            await pilot.pause()
            for char in form:
                await pilot.press(char)
            for _ in range(4):
                await pilot.pause()
            assert editor.credential_armed(), f"{form!r} arms"
            assert editor.placeholder == CREDENTIAL_PLACEHOLDER, "the state is set"
            assert CREDENTIAL_PLACEHOLDER not in _painted(app), "but it does not paint"
            assert editor.text, "because an armed buffer is never empty"


@pytest.mark.asyncio
async def test_a_flag_disarm_clears_the_pickers_armed_notice() -> None:
    """D7: the row promising capture must not outlive the capture.

    ``CREDENTIAL_ARMED_NOTICE`` was written only when the argument list OPENED,
    so a disarm while it was already open left it on screen. The flag disarm is
    the reachable case — deleting the token closes the list, a capture replaces
    the buffer — and it is the disarm the operator is least likely to expect.
    The stale row then sat NEARER the caret than the transcript's warning and
    contradicted it, promising "never shown" directly above a plaintext paste.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor.credential_armed()
        # Typed through the space, so the capture is open and the row carries
        # the masking notice rather than the bare armed one.
        assert CREDENTIAL_TYPING_NOTICE in _painted(app), "precondition: the row is up"

        await pilot.press("-")
        for _ in range(8):
            await pilot.pause()
        assert not editor.credential_armed()
        painted = _painted(app)
        assert CREDENTIAL_ARMED_NOTICE not in painted, "the promise went with the state"
        # The typed-capture promise must go with it for the same reason: a `-`
        # means the operator is addressing the COMMAND, so nothing may still
        # claim their keystrokes are being masked.
        assert CREDENTIAL_TYPING_NOTICE not in painted, "nor the masking promise"

        # And it STAYS cleared: the operator pastes into the disarmed composer,
        # which is the moment the stale row would have lied about.
        app.post_message(events.Paste("an ordinary paste"))
        for _ in range(4):
            await pilot.pause()
        assert CREDENTIAL_ARMED_NOTICE not in _painted(app)


# -- D9: a leftover token is a live destroy-everything control ----------------
async def _capture_leaving_a_leftover_token(pilot, app, editor) -> None:
    """Post-capture state that leaves a SECOND, unarmed ``/credential`` behind.

    The draft mentions the command in prose before the operator makes the real
    gesture — ``fix the /credential command`` is an ordinary sentence, and it
    ARMS, because it passes through ``/credential`` at end-of-line while being
    typed (that is the same keystroke shape as ``deploy with /credential the
    prod key``, which must stay armed; see ``CREDENTIAL_TOKEN``). The latch
    therefore sits on the EARLIER token, the marker splices there, and the
    token the operator typed at the caret is left in the buffer, unarmed and
    with an empty argument slot.

    THE ``escape`` IS THE PROSE AUTHOR'S KEYSTROKE, and it is what the typed
    capture makes necessary. The space after that first mention now opens a
    masked span, so ``command`` is taken as a secret and rendered as bullets —
    loudly and immediately, which is the point. Esc unredacts it back to the
    word the operator typed and ends the capture, leaving exactly the prose
    draft this helper is about. That is the intended cost of the mode: a false
    positive is visible on the frame and one keystroke to undo, where the false
    negative it replaces put a plaintext secret in scrollback for good.
    """
    for char in "fix the /credential command":
        await pilot.press(char)
    for _ in range(4):
        await pilot.pause()
    # Back out of the capture the prose opened; the masked word returns as text.
    await pilot.press("escape")
    for _ in range(4):
        await pilot.pause()
    assert "command" in editor.text, "Esc gave the prose back"
    for char in " /credential ":
        await pilot.press("space" if char == " " else char)
    for _ in range(4):
        await pilot.pause()
    app.post_message(events.Paste(SECRET))
    for _ in range(4):
        await pilot.pause()
    assert "/credential" in editor.text, "precondition: a leftover token remains"
    assert not editor.credential_armed(), "precondition: the capture ended the arm"


@pytest.mark.parametrize(
    ("label", "keys"),
    [
        ("end", ["end", "enter", "enter"]),
        ("right-walk", ["right"] * 30 + ["enter", "enter"]),
    ],
)
@pytest.mark.asyncio
async def test_the_post_capture_leftover_token_cannot_wipe_the_store(
    label: str, keys: list[str]
) -> None:
    """D9: pure keystrokes out of the feature's own output wiped everything.

    The leftover token is not a cosmetic blemish, it is a live, PRESELECTED,
    unconfirmed ``--forget-all``. Its argument slot is empty, so the picker
    offered every row with the destructive one highlighted, and ``end`` then
    Enter twice — or an equivalent walk to end-of-line — completed and RAN it
    against a live store, with no confirmation and no undo.

    D1's guard does not cover this: it keys on ARMED, and the capture itself
    disarms, so by this frame the arm is already gone. The condition that
    tracks the hazard is that the composer is HOLDING a secret the operator can
    see, which is what a citation means.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        # A live credential from BEFORE this draft: the one a wipe destroys
        # that the operator never consented to lose.
        session.variables.store_credential("LOP_SECRET_LIVEKEY1", "kept", "command")
        await _capture_leaving_a_leftover_token(pilot, app, editor)

        for key in keys:
            await pilot.press(key)
            await pilot.pause()
        for _ in range(8):
            await pilot.pause()

        assert (
            "LOP_SECRET_LIVEKEY1" in session.variables.credential_names()
        ), f"the pre-existing credential survives {label},enter,enter"


@pytest.mark.asyncio
async def test_the_chip_and_the_store_never_contradict_each_other() -> None:
    """D9's load-bearing half: the wipe took the credential the chip cites.

    The marker is a RECEIPT. Destroying what it points at while it is still on
    screen put two rows of one frame in contradiction — a chip promising a
    credential beside "No credentials stored for this session" — which is the
    same defect class as the stale armed row (D7). So the property is not only
    "the store survives" but "what is chipped is what is held".
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await _capture_leaving_a_leftover_token(pilot, app, editor)
        cited = [payload.key for payload in credential_payloads(editor.text, editor._attachments)]
        assert len(cited) == 1, "precondition: the draft cites exactly one capture"

        await pilot.press("end")
        await pilot.press("enter")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        # The draft submitted, so its citation was honoured: the captured
        # credential is IN the store rather than having been wiped alongside it.
        assert (
            cited[0] in session.variables.credential_names()
        ), "the credential the chip cited reached the store"
        assert SECRET not in _painted(app), "and the value itself never painted"


@pytest.mark.asyncio
async def test_a_held_credential_suppresses_the_destructive_rows_and_says_why() -> None:
    """The mechanism behind D9, and the notice that keeps the list legible.

    A list that simply emptied would read as the gesture having been dropped,
    which is why the armed state has a notice; the held state needs its own,
    because the armed one says "paste the secret" and by this point the
    operator already has.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        session.variables.store_credential("LOP_SECRET_LIVEKEY1", "kept", "command")
        await _capture_leaving_a_leftover_token(pilot, app, editor)
        assert editor.credential_cited(), "the buffer still cites the capture"

        await pilot.press("end")
        for _ in range(6):
            await pilot.pause()
        assert editor.picker.highlighted_name() is None, "no row is preselected"
        assert CREDENTIAL_HELD_NOTICE in _painted(app), "the row says why it is empty"
        assert CREDENTIAL_ARMED_NOTICE not in _painted(
            app
        ), "and does not promise a capture that already happened"


@pytest.mark.asyncio
async def test_the_forget_verbs_stay_reachable_by_typing_while_a_chip_is_held() -> None:
    """The guard withholds ROWS, never the command.

    Suppression that made the destructive verb unreachable would be a
    regression dressed as a fix. The operator who means it still types it, and
    typing the flag is the explicit act the preselected row was not.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        session.variables.store_credential("LOP_SECRET_LIVEKEY1", "kept", "command")
        await _capture_leaving_a_leftover_token(pilot, app, editor)

        # Clear the draft, then ask for the destructive verb deliberately.
        editor.text = ""
        editor.move_cursor(editor._end_of_buffer())
        for _ in range(4):
            await pilot.pause()
        for char in "/credential --forget-all":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()

        assert (
            session.variables.credential_names() == []
        ), "an explicitly typed --forget-all still wipes"


@pytest.mark.asyncio
async def test_the_guard_lifts_once_the_chip_is_gone() -> None:
    """The suppression is scoped to the citation, not sticky for the session.

    A capture the operator backspaced away is one they visibly withdrew, and
    the map can still hold its payload until the edit funnel releases it — so
    the guard asks about the CITATION, which tracks what is on screen.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        session.variables.store_credential("LOP_SECRET_LIVEKEY1", "kept", "command")
        await _capture_leaving_a_leftover_token(pilot, app, editor)
        assert editor.credential_cited()

        # The operator clears the draft: nothing is chipped any more.
        editor.text = ""
        editor.move_cursor(editor._end_of_buffer())
        for _ in range(4):
            await pilot.pause()
        assert not editor.credential_cited(), "no chip, no reason to withhold the rows"
        assert app._credential_choices(armed=False, cited=False), "the rows come back"


@pytest.mark.asyncio
async def test_a_second_capture_still_arms_while_the_first_chip_is_held() -> None:
    """The two guards compose: holding a chip must not block a further capture.

    Suppressing the destructive ROWS while a credential is cited must not touch
    the gesture itself — handing over a key and then a secret is an ordinary
    thing to do, and it is the case ``_capture_credential`` documents. The
    ARMED notice takes precedence over the HELD one here because the operator
    is mid-gesture, so what the NEXT paste does is the more urgent thing to say.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "deploy with /credential ":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        app.post_message(events.Paste(SECRET))
        for _ in range(4):
            await pilot.pause()
        assert editor.credential_cited() and not editor.credential_armed()

        for char in "and /credential ":
            await pilot.press(char)
        for _ in range(6):
            await pilot.pause()
        assert editor.credential_armed(), "the second gesture still arms"
        painted = _painted(app)
        # The separating space also OPENS the typed capture, so the row shows
        # the more specific of the two mid-gesture notices: the operator's next
        # keystroke is being masked, which is what they need told. The HELD
        # notice is what must not win here, and that is the property this line
        # has always guarded.
        assert CREDENTIAL_TYPING_NOTICE in painted, "mid-gesture, the capture notice wins"
        assert CREDENTIAL_HELD_NOTICE not in painted, "and the two do not both show"

        app.post_message(events.Paste("second-secret-value"))
        for _ in range(4):
            await pilot.pause()
        assert editor.text.count("[Credential #") == 2, "both captures are chipped"
        assert "second-secret-value" not in _painted(app)


# ---------------------------------------------------------------------------
# THE TYPED capture.
#
# Everything above this line is about the PASTED gesture. `/credential` shipped
# in v0.53.0 with a clipboard path and no typed path at all, so the obvious
# human gesture — typing `/credential 12345` — fell through to the legacy
# `/credential <KEY>` command with the SECRET as its key argument, and landed in
# the transcript in plaintext with no chip and no warning. These tests pin the
# typed path and, more importantly, the several ways it can silently stop
# masking. Each one is a reproduced failure, not a hypothetical.
# ---------------------------------------------------------------------------

#: A value shaped like a REAL credential: punctuation, mixed case, digits. The
#: alphanumeric secrets used above cannot catch the punctuation leak below,
#: which is exactly how that bug survived its first implementation.
TYPED_SECRET = "zQ7-TYPED.canary_4417/x+y"


async def _type_secret(pilot, editor: Editor, secret: str, prefix: str = "") -> None:
    """Type ``prefix``, the token, the opening SPACE, then ``secret``.

    No paste anywhere — this is the gesture the operator actually made. The
    picker is dismissed the way a user does it, so the hand-typed route is
    measured rather than the completion route.
    """
    for char in f"{prefix}/credential":
        await pilot.press("space" if char == " " else char)
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("space")
    await pilot.pause()
    for char in secret:
        await pilot.press("space" if char == " " else char)
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_typed_secret_never_enters_the_document() -> None:
    """THE HEADLINE. Typing the secret masks it and mints the same chip a paste does.

    The reproduction of the reported defect: `/credential 12345` typed by hand
    produced no chip and put the secret in the submitted text.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, TYPED_SECRET)
        assert TYPED_SECRET not in editor.text, "the buffer must never hold it"
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * len(TYPED_SECRET)
        assert editor.credential_typing(), "the capture is open"

        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == f"[Credential #1, {len(TYPED_SECRET)} chars] "
        payloads = credential_payloads(editor.text, editor._attachments)
        assert len(payloads) == 1
        assert payloads[0].value == TYPED_SECRET, "the stored value is exactly what was typed"


@pytest.mark.asyncio
async def test_every_printable_character_is_masked_not_only_alphanumerics() -> None:
    """The punctuation leak, pinned.

    Textual spells punctuation keys as WORDS (`minus`, `full_stop`), so a
    handler gated on ``len(event.key) == 1`` masks letters and digits and lets
    every punctuation character fall through into the document. Measured, the
    canary rendered as ``•••-TYPED-LEAK-CANARY-4417``: the first hyphen ended
    the masking and the rest of the secret was typed in plaintext and submitted.
    Real credentials are mostly punctuation, so that gate leaked nearly every
    actual secret while passing against an alphanumeric test value.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        # Deliberately not LEADING with `-`: that is the documented flag
        # escape (`--forget-all`). Every other punctuation character, and a
        # hyphen in a non-leading position, must be masked.
        punctuation = "a-_.+/=:@!~"
        await _type_secret(pilot, editor, punctuation)
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * len(punctuation)
        for char in punctuation:
            assert char not in editor.text[len("/credential ") :], f"{char!r} reached the buffer"


@pytest.mark.asyncio
async def test_the_separating_space_is_not_counted_in_the_length() -> None:
    """``K`` counts the secret, never the delimiter that opened the span.

    The operator can never see the value again, so a length that disagreed with
    it by one is a receipt they cannot check.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "12345")
        await pilot.press("enter")
        await pilot.pause()
        assert "[Credential #1, 5 chars]" in editor.text, "5, not 6"
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == "12345"
        assert not payloads[0].value.startswith(" ")


@pytest.mark.asyncio
async def test_a_space_inside_the_span_is_part_of_the_secret() -> None:
    """Passphrases contain spaces; Enter is the terminator, not the space.

    Terminating on the space would truncate a passphrase and type its remainder
    in PLAINTEXT — this feature's own defect, reappearing inside its fix.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        phrase = "correct horse battery staple"
        await _type_secret(pilot, editor, phrase)
        assert phrase not in editor.text
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == phrase, "every space was kept"
        assert f"{len(phrase)} chars" in editor.text


@pytest.mark.asyncio
async def test_enter_ends_the_secret_and_leaves_the_operator_in_the_composer() -> None:
    """Enter mints the chip; it does NOT submit.

    The gesture is specified as "hand over a secret and then describe it", so an
    Enter that also sent the message would make the description impossible to
    write.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        submitted: list[str] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, EditorSubmitted):
                submitted.append(message.text)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        await _type_secret(pilot, editor, "12345")
        await pilot.press("enter")
        await pilot.pause()
        assert not submitted, "the first Enter must not send the message"
        for char in "the staging key":
            await pilot.press("space" if char == " " else char)
        await pilot.pause()
        assert editor.text == "[Credential #1, 5 chars] the staging key"


@pytest.mark.asyncio
async def test_escape_unredacts_the_typed_characters() -> None:
    """Esc gives the operator their text back rather than discarding it.

    Retyping a secret from memory is precisely what an operator cannot do, so
    the recoverable reading is the only safe one. It also ENDS the arm: the
    operator has visibly backed out, so the next paste must not still be
    swallowed.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "12345")
        assert "12345" not in editor.text
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "/credential 12345", "the characters came back"
        assert not editor.credential_typing()
        assert not editor.credential_armed(), "Esc ends the gesture too"


@pytest.mark.asyncio
async def test_a_typed_capture_never_routes_to_the_legacy_command() -> None:
    """The reported defect, at the seam where it did its damage.

    The typed line used to reach `_dispatchable_slash` as
    ``/credential <secret>``, which the legacy command reads as a KEY NAME — so
    the app prompted "Paste the value for 12345" with the secret rendered as the
    key, in the transcript, in plaintext.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        submitted: list[str] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, EditorSubmitted):
                submitted.append(message.text)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        await _type_secret(pilot, editor, TYPED_SECRET)
        await pilot.press("enter")  # mints the chip
        await pilot.pause()
        await pilot.press("enter")  # sends the message
        for _ in range(8):
            await pilot.pause()
        assert submitted, "the second Enter sends"
        assert TYPED_SECRET not in " ".join(submitted)
        assert "[Credential #1," in " ".join(submitted), "the chip goes instead"
        assert TYPED_SECRET not in _painted(app), "and nothing paints it"


@pytest.mark.asyncio
async def test_bare_credential_with_a_key_still_dispatches_as_the_command() -> None:
    """The shipped command is untouched when no capture is open.

    ``/credential MY_API_KEY`` arriving as TEXT (a recalled prompt, a restored
    draft) has passed through no arming gesture, so it must still reach the
    legacy handler exactly as it always has.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        submitted: list[str] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, EditorSubmitted):
                submitted.append(message.text)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        editor.load_text("/credential MY_API_KEY")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert not editor.credential_armed(), "arriving text never arms"
        assert not editor.credential_typing()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert submitted == ["/credential MY_API_KEY"], "dispatched unchanged"


@pytest.mark.asyncio
async def test_a_leading_flag_escapes_the_mask_so_the_verbs_stay_typable() -> None:
    """``--forget-all`` must survive the opening space.

    The credential verbs are flag-shaped precisely so they cannot collide with a
    key (keys normalize to ``[A-Z0-9_]``). Without the escape the opening space
    masked them and Enter minted a chip instead of forgetting anything, which
    regresses the shipped command.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential":
            await pilot.press(char)
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("space")
        for char in "--forget-all":
            await pilot.press(char)
        await pilot.pause()
        assert editor.text == "/credential --forget-all", "typed verbatim"
        assert not editor.credential_typing(), "a flag is the command, not a secret"
        assert not editor.credential_armed()


@pytest.mark.asyncio
async def test_a_hyphen_inside_a_secret_is_masked_like_any_character() -> None:
    """The flag escape is scoped to the FIRST character only.

    ``-`` is common inside real keys (base64url), so once any character has been
    typed the operator is entering a value and the flag reading is gone.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "abc-def")
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * len("abc-def")
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == "abc-def"


@pytest.mark.asyncio
async def test_the_caret_leaving_the_span_ends_the_capture_without_disclosing() -> None:
    """A caret move is not a request for the plaintext back.

    The mask is positional, so a capture left open after the caret moved would
    append to the value in one place while inserting its cell in another. Asked
    on ``watch_selection`` because `move_cursor`, a mouse click and an
    app-set selection all move the caret WITHOUT a caret key being pressed.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "12345")
        editor.move_cursor((0, 0))
        await pilot.pause()
        assert not editor.credential_typing(), "the capture ended"
        assert "12345" not in editor.text, "and did not write the secret out"


@pytest.mark.asyncio
async def test_the_caret_returning_to_the_token_re_opens_the_capture() -> None:
    """The span is POSITIONAL: open exactly while the caret is in it.

    Without the reopen, leaving and coming back left an armed token whose next
    typed character landed in PLAINTEXT — the capture only ever opened on the
    edit that inserted the space, and that space had already been typed.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential":
            await pilot.press(char)
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        editor.move_cursor((0, 0))
        await pilot.pause()
        assert not editor.credential_typing()
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert editor.credential_typing(), "back in the span, masking again"
        for char in "9999":
            await pilot.press(char)
        await pilot.pause()
        assert "9999" not in editor.text


@pytest.mark.asyncio
async def test_an_empty_secret_mints_nothing() -> None:
    """``/credential `` + Enter advertises no key.

    A zero-length credential would name a key to the model that can never hold
    anything, and `store_credential` refuses a blank value regardless.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential":
            await pilot.press(char)
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert "[Credential #" not in editor.text
        assert not credential_payloads(editor.text, editor._attachments)


@pytest.mark.asyncio
async def test_backspacing_the_delimiting_space_backs_out_of_the_gesture() -> None:
    """Erasing the span and then the space un-terminates the token.

    The two counters cannot disagree about the length: a backspace inside the
    span retracts the held character AND its cell together.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "12345")
        for _ in range(5):
            await pilot.press("backspace")
        await pilot.pause()
        assert editor.text == "/credential ", "the cells went with the characters"
        assert editor.credential_typing(), "still open, just empty"
        await pilot.press("backspace")
        await pilot.pause()
        assert editor.text == "/credential"
        assert not editor.credential_typing(), "the gesture is withdrawn"


@pytest.mark.asyncio
async def test_a_typed_capture_is_dropped_when_the_buffer_is_replaced() -> None:
    """History recall, a restored draft and `/clear` must not reveal it.

    Abandoned rather than cancelled: restoring the plaintext into a buffer the
    operator is navigating AWAY from is the one thing that must never happen.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, TYPED_SECRET)
        editor.clear_content()
        await pilot.pause()
        assert not editor.credential_typing()
        assert TYPED_SECRET not in editor.text
        assert TYPED_SECRET not in repr(getattr(editor, "_history", []))


@pytest.mark.asyncio
async def test_accepting_the_picker_row_with_enter_arms_and_masks() -> None:
    """Route 2: Enter accepts the completion; it must not submit the command.

    TWO DIFFERENT ENTERS in one gesture, separated by STATE rather than timing:
    the picker's accept can only happen while no capture is open, and the mint
    only once one is. Before this, the accepting Enter ran the command, cleared
    the buffer, and the operator's next keystrokes were typed into an ordinary
    unmasked composer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        submitted: list[str] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, EditorSubmitted):
                submitted.append(message.text)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        for char in "/credential":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor._picker.is_open(), "precondition: the row is offered"
        await pilot.press("enter")
        await pilot.pause()
        assert not submitted, "accepting a completion must not send"
        assert editor.text == "/credential ", "the token and its space are inserted"
        assert editor.credential_typing(), "and that space armed the mask"
        for char in "12345":
            await pilot.press(char)
        await pilot.pause()
        assert "12345" not in editor.text
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "[Credential #1, 5 chars] "
        assert not submitted, "the mint Enter does not send either"


@pytest.mark.asyncio
async def test_accepting_the_picker_row_with_tab_arms_and_masks() -> None:
    """Route 2, Tab. Both keys insert the same completion, so both must arm."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        for char in "/credential":
            await pilot.press(char)
        for _ in range(4):
            await pilot.pause()
        assert editor._picker.is_open()
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "/credential "
        assert editor.credential_typing(), "Tab arms identically to Enter"
        for char in "12345":
            await pilot.press(char)
        await pilot.pause()
        assert "12345" not in editor.text
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "[Credential #1, 5 chars] "


@pytest.mark.asyncio
async def test_pasting_while_armed_still_chips_instantly_in_place() -> None:
    """The v0.53.0 gesture must not regress.

    The opening space now starts a capture immediately, so a plain
    ``/credential `` + paste arrives with an OPEN but EMPTY span. That still
    mints in place rather than being appended and left masked until a further
    Enter.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/credential":
            await pilot.press(char)
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        app.post_message(events.Paste(SECRET))
        await pilot.pause()
        await pilot.pause()
        assert editor.text == f"[Credential #1, {len(SECRET)} chars] ", "chipped in place"
        assert SECRET not in editor.text


@pytest.mark.asyncio
async def test_pasting_into_a_partly_typed_span_appends_to_the_same_secret() -> None:
    """One value, one chip: typing a prefix and pasting the rest is ordinary.

    Two chips would split one credential into two the model cannot recombine.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "abc")
        app.post_message(events.Paste("def"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * 6
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert len(payloads) == 1, "one credential, not two"
        assert payloads[0].value == "abcdef"


# -- the exits, and editing inside the span -----------------------------------
#
# Every test below covers a state the 95 tests above passed straight through.
# The empty-span Esc is the sharpest instance: `test_escape_unredacts_the_typed_
# characters` types "12345" FIRST, so the zero-character case — the one where
# the cancel had nothing to replace and re-armed itself — was never reached
# (review round 1, R1). A guard that cannot go red is worse than no guard.


@pytest.mark.asyncio
async def test_escape_on_an_empty_span_actually_ends_the_capture() -> None:
    """Esc BEFORE any character is typed must exit, not silently re-arm.

    The regression: `_cancel_credential_typing`'s own `replace()` re-entered the
    edit funnel, which re-derived the arm from a buffer still ending in
    `/credential ` with the caret still at its end, and re-opened the capture the
    cancel had just closed. Esc was inert on the exact key the on-screen notice
    advertises — measured, three presses changed nothing.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "")
        assert editor.credential_typing(), "precondition: the span is open and empty"

        await pilot.press("escape")
        await pilot.pause()
        assert not editor.credential_typing(), "Esc ends the capture"
        assert not editor.credential_armed(), "and the arm with it"
        assert editor.text == "/credential ", "the token is left as inert text"


@pytest.mark.asyncio
async def test_prose_after_an_empty_span_escape_is_not_captured() -> None:
    """The consequence the inert Esc had, pinned at the seam it did damage.

    An operator who armed the mode, changed their mind and pressed Esc had their
    next sentence swallowed: `is broken please fix` became a stored credential
    and the message never reached the model.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "")
        await pilot.press("escape")
        await pilot.pause()
        for char in "is broken please fix":
            await pilot.press("space" if char == " " else char)
        await pilot.pause()
        assert editor.text == "/credential is broken please fix", "typed as ordinary prose"
        assert CREDENTIAL_MASK_CHAR not in editor.text, "nothing was masked"
        assert not credential_payloads(editor.text, editor._attachments), "and nothing minted"


@pytest.mark.asyncio
async def test_escape_with_characters_typed_announces_the_plaintext() -> None:
    """The unredact is the one exit that ENDS with the secret in the buffer.

    It restores by design (retyping a secret from memory is what an operator
    cannot do), but the frame after it looks completely ordinary — the amber
    marker reverts, the masking notice goes — while the next Enter re-commits the
    original leak. So the exit has to say what it did (design round 1, D1).
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        announced: list[int] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, CredentialUnredacted):
                announced.append(message.length)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        await _type_secret(pilot, editor, "12345")
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "/credential 12345", "the characters came back"
        assert announced == [5], "and the operator is told they are now plaintext"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry",
    [
        # The shape UX walked: the operator explains themselves, then re-arms.
        "actually /credential",
        # Straight back to the gesture with nothing in between.
        "/credential",
        # Words first, token last — the commonest mid-prose arm.
        "ok now /credential",
    ],
)
async def test_retrying_the_gesture_after_an_escape_masks_again(retry: str) -> None:
    """U6 — the cancel's OWN flow: retry on the same line and it must mask.

    Making Esc real (R1/U2) created the state this fails in, so it is the
    remediation's own regression. Typing a SECOND token walks through spellings
    (`/crede`, `/creden`, …) that are not tokens, so the latched arm found no
    match at its anchor and the nearest-match tie-break MIGRATED it back onto
    the first token earlier in the line. The migration is one-way — the anchor
    moved with it — so the arm never came home, the caret-at-span-end gate never
    fired, and the secret typed next was painted in the CLEAR and then parsed as
    a credential NAME, which is the half the model does learn (UX round 2, U6).
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "")
        await pilot.press("escape")
        await pilot.pause()
        assert not editor.credential_typing(), "precondition: the cancel landed"

        for char in retry:
            await pilot.press("space" if char == " " else char)
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert editor.credential_typing(), f"the retry re-opens the capture after {retry!r}"

        for char in SECRET:
            await pilot.press(char)
        await pilot.pause()
        assert SECRET not in editor.text, "the retyped secret is masked, not painted"
        assert editor.text.count(CREDENTIAL_MASK_CHAR) == len(SECRET), "one cell per character"
        assert editor._credential_typed == SECRET, "and the held value is the one typed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "word",
    [
        # Longer words that merely START WITH the token must stay prose. The
        # typed-through rule is PREFIX-OF, not starts-with, precisely so these
        # cannot inherit an arm.
        "/credentials",
        "/credentialx",
        # Shortened BELOW the `/cred` floor is the operator withdrawing the
        # gesture, which is the visible way out and must keep working.
        "/cre",
        "/credx",
        # An unrelated command, and ordinary prose.
        "/help",
        "notes about the deploy",
    ],
)
async def test_the_retry_rule_does_not_arm_anything_but_the_token(word: str) -> None:
    """U6's fix must widen the masked set ONLY onto the operator's own gesture.

    The counterpart to the test above, and the one that stops it being satisfied
    by masking everything: a rule that re-armed on any word at the anchor would
    swallow prose as a secret, which is the unrecoverable direction. So each
    word here is typed into exactly the post-cancel state U6 lives in and must
    leave the following text as PLAIN TEXT.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "")
        await pilot.press("escape")
        await pilot.pause()

        for char in word:
            await pilot.press("space" if char == " " else char)
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert not editor.credential_typing(), f"{word!r} must not arm a capture"

        for char in "PLAINTEXT_STAYS_VISIBLE":
            await pilot.press(char)
        await pilot.pause()
        assert "PLAINTEXT_STAYS_VISIBLE" in editor.text, "it stays readable"
        assert CREDENTIAL_MASK_CHAR not in editor.text, "and nothing was masked"


@pytest.mark.asyncio
async def test_an_empty_escape_announces_nothing() -> None:
    """No characters restored, no warning owed — the notice must not cry wolf."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        announced: list[int] = []
        original = editor.post_message

        def _spy(message):  # type: ignore[no-untyped-def]
            if isinstance(message, CredentialUnredacted):
                announced.append(message.length)
            return original(message)

        editor.post_message = _spy  # type: ignore[method-assign]
        await _type_secret(pilot, editor, "")
        await pilot.press("escape")
        await pilot.pause()
        assert announced == [], "nothing came back, so nothing is announced"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys, typed, expected",
    [
        # Arrow back two, type two: the obvious way to fix a typo mid-token.
        (("left", "left"), "xy", "ABCDEFxyGH"),
        # One arrow back, one character.
        (("left",), "z", "ABCDEFGzH"),
        # Home, then a character: the whole span jumped, not one cell.
        (("home",), "0", None),
    ],
)
async def test_typing_inside_the_span_lands_where_the_operator_sees_it(
    keys: tuple[str, ...], typed: str, expected: str | None
) -> None:
    """The held value is a POSITIONAL MIRROR of the mask cells, not an append.

    The regression (UX round 1, U1): the bullet was inserted AT THE CARET while
    the value appended to the END, and the capture deliberately stays open
    anywhere inside the span — so the interaction invited the edit and then
    mis-applied it. Measured, typed `ABCDEFGH` + `←←` + `xy` stored
    `ABCDEFGHxy` for an intended `ABCDEFxyGH`: the chip reported the RIGHT
    length with the WRONG order, so the documented integrity check passed on a
    value that can never be displayed again to catch it.

    ``expected`` is ``None`` where the caret leaves the span entirely: that is a
    departure, and the capture closes rather than mis-applying the edit.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "ABCDEFGH")
        for key in keys:
            await pilot.press(key)
        await pilot.pause()
        if expected is None:
            assert not editor.credential_typing(), "leaving the span ends the capture"
            return
        for char in typed:
            await pilot.press(char)
        await pilot.pause()
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * len(expected)
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == expected, "the stored value is the one typed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys, expected",
    [
        # Backspace mid-span drops the character BEFORE the caret, not the last.
        (("left", "left", "backspace"), "ABCDEGH"),
        # Forward delete drops the one UNDER it.
        (("left", "left", "delete"), "ABCDEFH"),
        # Backspace at the end is the ordinary case, and still works.
        (("backspace",), "ABCDEFG"),
        # A selection dragged over two cells and deleted takes exactly those two.
        (("shift+left", "shift+left", "backspace"), "ABCDEF"),
    ],
)
async def test_deleting_inside_the_span_removes_the_character_under_the_caret(
    keys: tuple[str, ...], expected: str
) -> None:
    """Deletion is positional too, by the same mirror and not a second rule.

    Backspace used to drop the LAST held character wherever the caret was, so a
    backspace between F and G removed H. `delete` and select-and-delete had no
    handler at all: the mask cell went and the held value did not, leaving the
    buffer and the value disagreeing about the secret's LENGTH.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "ABCDEFGH")
        for key in keys:
            await pilot.press(key)
        await pilot.pause()
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * len(expected)
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == expected, "the stored value is the one left on screen"


@pytest.mark.asyncio
async def test_selecting_inside_the_span_and_typing_replaces_those_characters() -> None:
    """Select-and-replace is one edit, and the mirror applies it as one.

    `insert()` ignores the selection, so a printable key over a selection left
    the selected mask cells in the buffer and appended a third — the buffer and
    the held value then disagreed on LENGTH, not merely order.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await _type_secret(pilot, editor, "ABCDEFGH")
        await pilot.press("shift+left")
        await pilot.press("shift+left")
        await pilot.press("Z")
        await pilot.pause()
        assert editor.text == "/credential " + CREDENTIAL_MASK_CHAR * 7
        await pilot.press("enter")
        await pilot.pause()
        payloads = credential_payloads(editor.text, editor._attachments)
        assert payloads[0].value == "ABCDEFZ", "GH was replaced by Z, in place"


@pytest.mark.asyncio
async def test_the_typing_notice_keeps_its_way_out_at_every_width() -> None:
    """The exit is kept by a LADDER, not by a comment asserting it must be.

    `CREDENTIAL_TYPING_NOTICE`'s own docstring said "Esc cancels" was "exactly
    the half that must not be the part that crops" — and measured on the real
    app it was the FIRST thing to crop, lost at 68 columns while `chip` went at
    56 (design round 1, D2; UX round 1, U3; review round 1, R3). Prose asserting
    an invariant is not a mechanism enforcing it, so this pins the mechanism at
    the widths the sweep named.
    """
    session = FakeSession()
    for width in (45, 56, 60, 66, 68, 69, 80, 120):
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(width, 24)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await _type_secret(pilot, editor, "hunter2-typed-test")
            for _ in range(3):
                await pilot.pause()
            painted = _painted(app).replace("\xa0", " ")
            assert "Esc cancels" in painted, f"the way out crops at {width} columns"
            assert (
                "Enter chips" in painted or "Enter turns it into a chip" in painted
            ), f"and the key that ENDS the entry is gone at {width} columns"
