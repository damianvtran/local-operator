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

from local_operator.tui.app import CREDENTIAL_PLACEHOLDER, OperatorApp
from local_operator.tui.widgets.editor import (
    ASIDE_PLACEHOLDER,
    ATTACHMENT_MARKER,
    CREDENTIAL_KEY_PREFIX,
    Attachment,
    Editor,
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


@pytest.mark.asyncio
async def test_a_mention_of_the_command_in_text_never_typed_through_the_arm_does_not_arm() -> None:
    """A draft that ARRIVES holding the word never passed through the gesture.

    Recalled history, a restored draft and a pasted paragraph all land through
    ``load_text``, and none of them is the operator typing the arming gesture.
    Arming is still :data:`CREDENTIAL_ARM` at the caret and nothing else — what
    changed in design round 1 (D2) is how long an arm SURVIVES once taken, not
    how one is taken.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
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
