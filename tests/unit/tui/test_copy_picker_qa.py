"""Independent QA probes of the `/copy` picker: parity, payload, and refusals.

Written against the RUNNING app rather than against the helpers, because the
questions this file asks are ones only the assembled system answers: whether a
fence grammar edge reaches the clipboard intact, whether the truncation caveat
fires for the node the user actually chose, and whether a chord that was
removed is really gone from the key map rather than merely from the help text.

It sits beside `test_copy_command.py` (payload and refusals) and
`test_copy_picker.py` (layout and navigation) and deliberately does not repeat
them. What is here is the ground those two leave uncovered:

* **Reference parity on CONTENT.** The port's own tests assert our behaviour;
  these assert the reference's semantics for the same input — user messages
  never listed, most-recent-first ordering, the 50-message cap exercised past
  its edge, and the `All N` rows appearing only above one block of a kind.
* **The grammar's hostile inputs.** Adjacent fences, an info string carrying
  spaces, `~~~`, an unclosed fence, a `>` inside a fence, CRLF, tabs. Each is a
  case where a plausible parser produces a *plausible* wrong answer rather than
  an exception, so only the payload distinguishes them.
* **The truncation contract on CHILDREN.** The caveat is a claim about what is
  on the clipboard. A complete fence inside a cut-off answer is complete, so
  copying it must stay silent; the message node above it must not.

Payload assertions read the DRIVER, not ``app._clipboard``: Textual assigns
that attribute before it checks for a driver, so it cannot tell "the escape
sequence went out" from "the app remembered a string".
"""

from __future__ import annotations

import base64
import re

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.copy_targets import MAX_MESSAGES, build_copy_targets
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.copy_picker import CopyPickerScreen
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# -- harness ------------------------------------------------------------------
# Shaped after `test_copy_command.py`'s helpers rather than importing them: that
# module's `_copy_latest` folds "open the picker" and "take the default row"
# into one call, and every probe here needs to steer between those two steps.

_OSC52_RE = re.compile(r"\x1b\]52;c;([^\x07\x1b]*)(?:\x07|\x1b\\)")


async def _boot(pilot, app: OperatorApp) -> None:
    """Settle until the session exists, so `_turn_is_live` reads the real flag."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _stream(pilot, app: OperatorApp, text: str, *, finish: bool = True) -> None:
    """Paint one agent message through the real event path."""
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(text))
    await pilot.pause()
    if finish:
        app.post_message(AssistantMessageEnd(text))
        await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter."""
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _picker(app: OperatorApp) -> CopyPickerScreen | None:
    screen = app.screen
    return screen if isinstance(screen, CopyPickerScreen) else None


def _tap_driver(app: OperatorApp) -> list[str]:
    """Record every raw driver write, so OSC 52 can be counted and decoded."""
    sink: list[str] = []
    driver = app._driver
    assert driver is not None, "no driver: the pilot would prove nothing about the write"
    original = driver.write

    def write(data: str) -> None:
        sink.append(data)
        original(data)

    driver.write = write  # type: ignore[method-assign]
    return sink


def _osc52_payloads(sink: list[str]) -> list[str]:
    """Every OSC 52 payload written, decoded back to the source text."""
    return [
        base64.b64decode(match.group(1)).decode("utf-8")
        for chunk in sink
        for match in _OSC52_RE.finditer(chunk)
    ]


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _rows(screen: CopyPickerScreen) -> list[str]:
    """Every row's `id`, in draw order — the tree's shape as a flat list."""
    return [node.target.id for node in screen.visible_rows]


def _labels(screen: CopyPickerScreen) -> list[str]:
    return [node.target.label for node in screen.visible_rows]


def _answer(text: str, truncated: bool = False) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    if truncated:
        block.mark_truncated()
    return block


async def _copy_row(pilot, app: OperatorApp, steps: int) -> str | None:
    """Open the picker, move down `steps` rows, press Enter; return the row id."""
    await _submit(pilot, app, "/copy")
    picker = _picker(app)
    if picker is None:
        return None
    for _ in range(steps):
        await pilot.press("down")
        await pilot.pause()
    target = picker.selected_target()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()
    return target.id if target else None


# -- reference parity on content ---------------------------------------------


@pytest.mark.asyncio
async def test_a_user_message_is_never_a_row_even_between_two_answers() -> None:
    """The reference skips every non-assistant message outright
    (`if (msg.role !== "assistant") continue`), and a user turn sitting between
    two answers is where a walk that merely takes "the last N blocks" would
    leak one in. Asserted on the ROWS, not on the clipboard: a leaked user row
    the user never selects is still a leak."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "first answer")
        await _submit(pilot, app, "a question only the user asked")
        await _stream(pilot, app, "second answer")
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        labels = _labels(picker)
        ids = _rows(picker)
        await pilot.press("escape")
        await pilot.pause()

    assert labels == ["second answer", "first answer"], labels
    assert ids == ["msg:1", "msg:2"], ids
    assert not any("question" in label for label in labels), labels


def test_the_cap_counts_from_the_most_recent_end_not_the_oldest() -> None:
    """Sixty answers, fifty rows — and the fifty must be the fifty NEWEST. A
    walk that capped from the front would list the same COUNT while listing the
    wrong messages, which no length assertion can catch."""
    targets = build_copy_targets([_answer(f"answer {index}") for index in range(60)])

    assert len(targets) == MAX_MESSAGES
    assert targets[0].label == "answer 59"
    assert targets[-1].label == "answer 10"
    assert [t.label for t in targets] == [f"answer {i}" for i in range(59, 9, -1)]


def test_the_all_rows_appear_only_above_more_than_one_block_of_that_kind() -> None:
    """`All N` is a convenience over several blocks; above one it would be a
    second row copying byte-for-byte what the row above copies. The two kinds
    are counted INDEPENDENTLY, so a message with two fences and one quote gets
    `All 2 blocks` and no `All N quotes`."""
    one_code_two_quotes = build_copy_targets([_answer("```py\na\n```\n\n> q1\n\ntext\n\n> q2\n")])[
        0
    ]
    two_code_one_quote = build_copy_targets([_answer("```py\na\n```\n\n```js\nb\n```\n\n> q1\n")])[
        0
    ]

    assert [c.label for c in one_code_two_quotes.children] == [
        "Block 1",
        "Quote 1",
        "Quote 2",
        "All 2 quotes",
    ]
    assert [c.label for c in two_code_one_quote.children] == [
        "Block 1",
        "Block 2",
        "Quote 1",
        "All 2 blocks",
    ]


def test_the_all_blocks_row_joins_bodies_with_one_blank_line() -> None:
    """The documented separator is `"\\n\\n"` — the reference's
    `.join("\\n\\n")`. A single newline would weld the last line of one block
    onto the first of the next when the payload is pasted into a file."""
    target = build_copy_targets([_answer("```py\na\nb\n```\n\n```js\nc\n```\n")])[0]
    combined = next(c for c in target.children if c.label == "All 2 blocks")

    assert combined.content == "a\nb\n\nc"


# -- the grammar's hostile inputs ---------------------------------------------


@pytest.mark.parametrize(
    "name, message, expected",
    [
        # Two fences with no prose between them are two blocks, not one block
        # whose body swallowed the middle markers.
        ("adjacent", "```py\na\n```\n```js\nb\n```\n", [("code", "a"), ("code", "b")]),
        # The info string is everything after the marker, trimmed — a language
        # plus attributes stays one string rather than being split on space.
        ("info string", "```python title=x.py\nbody\n```\n", [("code", "body")]),
        # `~~~` is a fence here, which the reference does not recognise at all.
        ("tilde", "~~~\nbody\n~~~\n", [("code", "body")]),
        # No closer: ordinary text, so a streaming or cut-off answer's half
        # block never becomes a copyable "block" with a body it never had.
        ("unclosed", "before\n```py\nhalf written\n", []),
        # Fences mask their bodies: this `>` is code, not a quote.
        ("quote in fence", "```\n> not a quote\n```\n", [("code", "> not a quote")]),
        # A quote run ends where the fence begins; both survive, in order.
        ("quote then fence", "> q\n```py\nc\n```\n", [("quote", "q"), ("code", "c")]),
        # `>` plus at most ONE space comes off, so a bare `>foo` de-prefixes.
        ("no space after gt", ">tight\n", [("quote", "tight")]),
        # Only one optional space: further indentation is the quote's own.
        ("quote keeps indent", ">     deep\n", [("quote", "    deep")]),
        # A message that is nothing but a fence still yields its body.
        ("only a fence", "```py\nbody\n```", [("code", "body")]),
        # Tabs are payload, not layout: they must reach the clipboard as tabs.
        ("tabs", "```py\n\tif x:\n\t\treturn\n```\n", [("code", "\tif x:\n\t\treturn")]),
    ],
)
def test_the_block_grammar_holds_on_its_edges(
    name: str, message: str, expected: list[tuple[str, str]]
) -> None:
    """Each of these produces a PLAUSIBLE wrong answer under a plausible wrong
    parser — a swallowed separator, a language split on its first space, a half
    block presented as whole — so the body is what distinguishes them, not an
    exception. Asserted through `build_copy_targets` rather than
    `extract_blocks` so the child `content` (what is actually copied) is what
    is checked, not an intermediate."""
    target = build_copy_targets([_answer(message)])[0]
    got = [
        ("code" if child.id.split(":")[2] == "code" else "quote", child.content)
        for child in target.children
        if not child.id.endswith(("all", "all-quotes"))
    ]

    assert got == expected, f"{name}: {got}"


def test_twenty_code_blocks_all_become_rows_and_the_all_row_counts_them() -> None:
    """A long answer is where an off-by-one in the child walk hides: the labels
    are 1-based while the ids are 0-based, and only a message with many blocks
    makes the two disagree visibly."""
    message = "\n".join(f"```py\nblock {index}\n```\n" for index in range(20))
    target = build_copy_targets([_answer(message)])[0]
    blocks = [c for c in target.children if ":code:" in c.id]
    combined = next(c for c in target.children if c.id.endswith(":all"))

    assert len(blocks) == 20
    assert blocks[0].id == "msg:1:code:0" and blocks[0].label == "Block 1"
    assert blocks[-1].id == "msg:1:code:19" and blocks[-1].label == "Block 20"
    assert combined.label == "All 20 blocks"
    assert combined.content == "\n\n".join(f"block {index}" for index in range(20))


def test_the_info_string_reaches_the_language_whole_rather_than_first_word() -> None:
    """The reference takes the info string as ONE trimmed string
    (`open[1].trim()`), and so must we: a fence tagged
    ```` ```python title=x.py ```` carries an attribute, not a second field.

    Pinned because splitting it on the first space is the natural-looking
    "fix" for the preview lexer — Pygments cannot resolve `python title=x.py`
    — and it would silently change the LANGUAGE our tree reports away from the
    reference's. The lexer's own tolerance is checked below rather than
    papered over here: an unresolvable name renders plain, it does not raise.
    """
    plain = build_copy_targets([_answer("```python\nx = 1\n```\n")])[0]
    attributed = build_copy_targets([_answer("```python title=x.py\nx = 1\n```\n")])[0]
    bare = build_copy_targets([_answer("```\nx = 1\n```\n")])[0]

    assert plain.children[0].language == "python"
    assert attributed.children[0].language == "python title=x.py"
    assert attributed.children[0].hint.startswith("python title=x.py · ")
    # An empty info string is no language at all, so the preview stays plain.
    assert bare.children[0].language is None


@pytest.mark.asyncio
async def test_an_unresolvable_info_string_still_renders_a_preview() -> None:
    """Because the info string is carried whole, the preview lexer is
    regularly handed something Pygments cannot resolve. That must degrade to a
    plain preview, not raise inside a repaint — a modal that throws while
    painting takes the app down with the user's answer still uncopied."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "Here:\n\n```python title=x.py extra\nx = 1\n```\n")
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        await pilot.press("down")
        await pilot.pause()
        drawn = picker.render_lines_for_test()
        await pilot.press("escape")
        await pilot.pause()

    assert any("x = 1" in line for line in drawn), drawn


def test_a_crlf_answer_does_not_carry_a_stray_return_into_the_payload() -> None:
    """A pasted-in Windows transcript reaches the block as `\\r\\n`. The split
    is on `\\n`, so a `\\r` rides on the end of every line — including the
    fence CLOSER, which is why the block is found at all.

    Marked `xfail(strict=True)` rather than deleted or relaxed, following the
    precedent at `test_transcript_selection.py:4966`: the assertion IS the
    acceptance check for the defect, so it must start passing the moment the
    defect is fixed and fail loudly if someone "fixes" it by changing what is
    expected instead. A trailing `\\r` is invisible in a diff and breaks a
    shell script pasted out of the picker."""
    target = build_copy_targets([_answer("Intro\r\n```py\r\nx = 1\r\n```\r\n")])[0]
    block = target.children[0]

    assert block.content == "x = 1", repr(block.content)


# -- payload correctness, off the driver --------------------------------------


@pytest.mark.asyncio
async def test_a_quote_child_reaches_the_clipboard_de_prefixed() -> None:
    """The `>` markers are the transcript's syntax, not the user's text. This
    goes through OSC 52 rather than `app._clipboard` because that attribute is
    assigned before the driver check and cannot prove a write went out."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "Quoting:\n\n> first line\n> second line\n")
        sink = _tap_driver(app)
        chosen = await _copy_row(pilot, app, 1)
        payloads = _osc52_payloads(sink)

    assert chosen == "msg:1:quote:0"
    assert payloads == ["first line\nsecond line"], payloads
    assert ">" not in payloads[0]


@pytest.mark.asyncio
async def test_the_message_node_copies_markdown_source_not_the_painted_frame() -> None:
    """The frame carries box-drawing, a left bar (`\\u258c`) and the wrap of
    whatever width the terminal happened to be. The clipboard must carry the
    SOURCE: fences intact, `**bold**` as characters."""
    answer = "A **bold** claim.\n\n```python\ndef f():\n    return 1\n```\n"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, answer)
        sink = _tap_driver(app)
        chosen = await _copy_row(pilot, app, 0)
        payloads = _osc52_payloads(sink)

    assert chosen == "msg:1"
    assert len(payloads) == 1, payloads
    copied = payloads[0]
    assert "**bold**" in copied, "the source's emphasis markers, not a style"
    assert "```python" in copied, "the fence lines are source"
    assert "\u258c" not in copied and "─" not in copied, "no frame decoration"


@pytest.mark.asyncio
async def test_exactly_one_escape_goes_out_per_copy_and_decodes_byte_identical() -> None:
    """One gesture, one write. A second escape would double-write the clipboard
    on terminals that treat OSC 52 as an append, and a payload that does not
    decode byte-identical means the base64 round trip lost something."""
    body = "unicode → ok\n\ttabbed\ntrailing spaces   "
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, f"Here:\n\n```txt\n{body}\n```\n")
        sink = _tap_driver(app)
        chosen = await _copy_row(pilot, app, 1)
        payloads = _osc52_payloads(sink)

    assert chosen == "msg:1:code:0"
    assert len(payloads) == 1, f"expected one escape, got {len(payloads)}"
    assert payloads[0] == body, repr(payloads[0])


# -- the truncation contract --------------------------------------------------


@pytest.mark.asyncio
async def test_copying_a_complete_block_inside_a_cut_off_answer_stays_silent() -> None:
    """The caveat is a claim about what is on the CLIPBOARD. A closed fence
    inside an aborted answer is itself complete — the abort ended the message,
    not that block — so raising the caveat there would be a false warning, and
    a false warning on the common case is how a true one stops being read.

    The same run asserts the message node above it DOES warn, so the test
    cannot pass by the notice never firing at all."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta("Partial:\n\n```py\nx = 1\n```\n\nand then it st"))
        await pilot.pause()
        app.post_message(AssistantMessageEnd(""))
        await pilot.pause()
        await pilot.pause()

        picker_rows = None
        before_child = len(_notices(app))
        child_id = await _copy_row(pilot, app, 1)
        after_child = _notices(app)[before_child:]

        before_message = len(_notices(app))
        message_id = await _copy_row(pilot, app, 0)
        after_message = _notices(app)[before_message:]

        picker_rows = message_id

    assert child_id == "msg:1:code:0", child_id
    assert after_child == [], f"a complete block must not raise the caveat: {after_child}"
    assert picker_rows == "msg:1"
    assert any("cut off" in notice for notice in after_message), after_message


@pytest.mark.asyncio
async def test_a_cut_off_answer_is_listed_and_marked_in_its_hint() -> None:
    """Listed, because it is the message the user was reading and most likely
    means; marked, because it is not the whole answer. `truncated` LEADS the
    hint — the hint is right-aligned and the label is what gives way, so a
    trailing marker is the first thing a narrow terminal cuts."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta("half a sen"))
        await pilot.pause()
        app.post_message(AssistantMessageEnd(""))
        await pilot.pause()
        await pilot.pause()
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        target = picker.selected_target()
        drawn = picker.render_lines_for_test()
        await pilot.press("escape")
        await pilot.pause()

    assert target is not None and target.truncated
    assert target.hint.startswith("truncated · "), target.hint
    assert any("truncated" in line for line in drawn), drawn


# -- the preview's overflow claim ---------------------------------------------


@pytest.mark.asyncio
async def test_the_overflow_marker_counts_the_lines_that_actually_exist() -> None:
    """The marker is the preview's one quantitative claim, and the only signal
    that the pane is not the whole answer. It has to scale with the answer: a
    600-line message and a 1000-line message reporting the identical number
    tells the user their answer is a fixed size, which is worse than no number
    at all because it looks precise.

    The wrap budget is the right optimisation — folding thousands of rows to
    show fifteen is real work avoided — but the design that introduced it says
    the marker "counts SOURCE lines, so the number stays honest without the
    work" (`copy_picker.py`, `PREVIEW_WRAP_BUDGET`). It counts wrapped rows
    instead, so the honesty the budget was allowed on the strength of is the
    part that went missing.

    Two sizes in one test, because a single size cannot distinguish "off by a
    constant" from "saturated": the failure is that the two numbers are EQUAL.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "\n".join(f"line {index}" for index in range(600)))
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        drawn = picker.render_lines_for_test()
        shown = sum(1 for line in drawn if line.strip().startswith("line "))
        marker = next(line for line in drawn if "more lines" in line)
        claimed = int(marker.strip().split()[1])
        await pilot.press("escape")
        await pilot.pause()

    assert (
        claimed == 600 - shown
    ), f"preview claims {claimed} more lines; {600 - shown} are actually unshown"


# -- navigation and refusals --------------------------------------------------


@pytest.mark.asyncio
async def test_enter_on_a_group_node_copies_the_whole_message() -> None:
    """Drilling in must never be the only way to get the answer out: the group
    row itself is copyable, and what it copies is the WHOLE message including
    the blocks its children expose separately."""
    answer = "Intro line.\n\n```py\nx = 1\n```\n\n> a quote\n"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, answer)
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        group = picker.selected_target()
        assert group is not None
        assert group.children, "the fixture must be a GROUP node"
        sink = _tap_driver(app)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        payloads = _osc52_payloads(sink)

    assert len(payloads) == 1, payloads
    assert payloads[0].strip() == answer.strip()


@pytest.mark.asyncio
async def test_esc_writes_no_escape_sequence_and_leaves_the_draft_alone() -> None:
    """A cancelled picker is not an event. Asserted on the DRIVER — the
    clipboard attribute cannot tell "nothing was written" from "the app
    remembered nothing" — and on the composer, because a modal that eats or
    edits the draft behind it is the failure Esc is supposed to avoid."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "an answer")
        await _submit(pilot, app, "/copy")
        assert _picker(app) is not None
        sink = _tap_driver(app)
        before = len(_notices(app))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        payloads = _osc52_payloads(sink)
        new_notices = _notices(app)[before:]
        toast = app.query_one(Toast).message
        still_open = _picker(app)

    assert still_open is None
    assert payloads == [], payloads
    assert new_notices == [], new_notices
    assert toast == ""


@pytest.mark.asyncio
async def test_ctrl_o_does_nothing_now_that_the_chord_is_gone() -> None:
    """The chord was withdrawn with the picker. "Gone" has to mean gone from
    the KEY MAP, not merely from the help text: a stale binding would still
    copy, or still crash on a handler that no longer exists. The draft is
    checked too, because an unbound control key that reaches the editor as text
    is its own regression."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "an answer worth copying")
        editor = app.query_one(Editor)
        editor.text = "a draft the user is still writing"
        editor.focus()
        await pilot.pause()
        sink = _tap_driver(app)
        await pilot.press("ctrl+o")
        await pilot.pause()
        await pilot.pause()
        payloads = _osc52_payloads(sink)
        draft = editor.text
        toast = app.query_one(Toast).message
        screen_now = app.screen

    assert payloads == [], f"ctrl+o still wrote to the clipboard: {payloads}"
    assert draft == "a draft the user is still writing", repr(draft)
    assert toast == ""
    assert not isinstance(screen_now, CopyPickerScreen), "ctrl+o must not open the picker"


@pytest.mark.asyncio
async def test_the_two_empty_cases_keep_their_distinct_wording() -> None:
    """Two states the reference's single string conflates: nothing has ever
    been said, versus the first answer is still arriving. Asserted in ONE test
    so the pair cannot silently converge on the same wording — two separate
    tests would both still pass if someone flattened them into one message."""
    fresh = OperatorApp(lambda: _factory(FakeSession()))
    async with fresh.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, fresh)
        before = len(_notices(fresh))
        await _submit(pilot, fresh, "/copy")
        empty_conversation = _notices(fresh)[before:]
        assert _picker(fresh) is None, "an empty tree must not open an overlay"

    live = OperatorApp(lambda: _factory(FakeSession()))
    async with live.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, live)
        live._session.streaming = True  # type: ignore[union-attr]
        live.post_message(AssistantMessageStart())
        await pilot.pause()
        live.post_message(AssistantDelta("still writ"))
        await pilot.pause()
        before = len(_notices(live))
        await _submit(pilot, live, "/copy")
        mid_first_answer = _notices(live)[before:]
        assert _picker(live) is None

    assert empty_conversation == ["nothing to copy — no agent message in this conversation"]
    assert mid_first_answer == ["nothing to copy yet — the first answer is still coming"]
    assert empty_conversation != mid_first_answer, "the two states must stay distinguishable"
