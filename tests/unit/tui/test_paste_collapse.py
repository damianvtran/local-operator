"""Collapsing a LARGE text paste to a marker.

The image marker's mechanism (``test_paste_images.py``) with a text payload.
The defect it answers is legibility, not performance: nothing here is slow —
``max-height: 8`` decouples render cost from document size — but a user who
types "why does this build fail? here is the log:" and pastes 500 lines can no
longer see their own question, and the field is capped at eight rows inside a
padded shell, so scrolling back is the only way to check what they are about to
send.

Every paste is posted to the APP, not to the widget. ``App.on_event`` forwards a
non-forwarded ``Paste`` to the focused widget, so posting straight to the widget
delivers it twice and every "inserted once" assertion silently doubles;
``test_paste_images.py`` pins that with a control test.

The submit assertions read the captured ``EditorSubmitted`` (the ``:551-554``
pattern there) rather than the widget, because ``_submit`` clears the buffer
synchronously right after posting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.widgets.text_area import Selection

from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import CompactionEnded
from local_operator.tui.widgets.editor import (
    ATTACHMENT_MARKER,
    COLLAPSE_ROWS,
    MIN_PASTE_ROWS,
    Attachment,
    Editor,
    EditorSubmitted,
    PastedText,
    _looks_path_shaped,
    expand_pastes,
    strip_paste_citations,
)
from tests.unit.tui._scratch import scratch_dir
from tests.unit.tui.conftest import TCSS_PATH
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text

#: The `$skill` ordering tests need a real discovered vocabulary, and
#: ``test_skill_invocation`` already builds one as the ONLY root (via the
#: documented env var, with a cwd chdir so a dev machine's own project root
#: cannot leak in). Imported rather than rebuilt: a second fixture defining
#: `$research` differently is how two files come to disagree about what the
#: skill body says. Activated with ``usefixtures`` rather than taken as a
#: parameter, because a parameter of the same name shadows this import (F811)
#: and none of these tests reads the root it returns.
from tests.unit.tui.test_skill_invocation import skill_root  # noqa: F401

#: Wide enough that a pasted line never soft-wraps, so "lines pasted" and "rows
#: consumed" are the same number and each test says what it means. The
#: threshold is measured in ROWS, so a narrow terminal would collapse a paste
#: with fewer lines — which is correct behaviour and its own test below.
WIDE = (100, 30)


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Editor()


class ChipHost(Host):
    """A composer under the REAL sheet, for the paint assertions.

    Same reasoning as ``test_marker_chip.py``: the chip's colours are resolved
    by the stylesheet from ``$lo-*``, so a host with a convenient inline style
    would assert this file's opinion rather than the shipped one.
    """

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.pause()


def _lines(count: int, body: str = "ERROR line {}") -> str:
    return "\n".join(body.format(index) for index in range(count))


def _grounds(editor: Editor, y: int, start: int, end: int) -> set[str | None]:
    """The distinct backgrounds under cells ``[start, end)`` of rendered row ``y``."""
    out: list[str | None] = []
    for segment in editor.render_line(y):
        style = segment.style
        bg = style.bgcolor.get_truecolor().hex.lower() if style and style.bgcolor else None
        out.extend(bg for _ in segment.text)
    return set(out[start:end])


# -- the threshold ------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_paste_that_fits_the_composer_is_left_alone() -> None:
    """The load-bearing negative. A 9-line function pasted to be EDITED must
    arrive as text: collapsing it would make the expand action routine, which is
    the signal that the default is wrong. The threshold is 2.5x the field's
    eight rows for exactly this reason.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        snippet = _lines(9, "    value = {}")
        await _paste(app, pilot, snippet)

        assert editor.text == snippet
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_a_paste_that_overruns_the_composer_collapses() -> None:
    """The reported case: the question is still on screen, the log is a receipt."""
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("why does this build fail? ")
        await _paste(app, pilot, _lines(240))

        assert editor.text == "why does this build fail? [Paste #1, 240 lines] "
        (pasted,) = editor.attachments().values()
        assert isinstance(pasted, PastedText)
        assert pasted.text == _lines(240)


@pytest.mark.asyncio
async def test_the_threshold_measures_the_PROSPECTIVE_buffer() -> None:
    """Six lines pasted into a draft that already holds many overflows the field
    just as surely, and the user loses their question just the same. Measuring
    the fragment alone would answer a question nobody asked.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Already past the budget on its own, so six more rows must tip it.
        editor.insert(_lines(COLLAPSE_ROWS, "draft line {}") + "\n")
        await pilot.pause()
        await _paste(app, pilot, _lines(6, "pasted {}"))

        assert "[Paste #1, 6 lines]" in editor.text
        (pasted,) = editor.attachments().values()
        assert isinstance(pasted, PastedText)
        assert pasted.text == _lines(6, "pasted {}")


@pytest.mark.asyncio
async def test_a_tiny_paste_is_never_chipped_however_tall_the_draft() -> None:
    """Review round 1, MAJOR 1. The prospective rule alone is not sufficient.

    Once a draft is already past the budget, EVERY later paste overruns it — so
    a two-character paste into a 20-row draft was chipped, growing the composer
    by seventeen characters to hide two characters of visible text behind a
    marker naming them. The design's justification ("the user loses their
    question just the same") does not hold for a fragment that occupies one row:
    nothing was lost that the chip does not now hide.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert(_lines(COLLAPSE_ROWS * 2, "draft line {}"))
        await pilot.pause()
        assert editor.wrapped_document.height > COLLAPSE_ROWS, "the draft must already overrun"

        await _paste(app, pilot, "hi")

        assert editor.text.endswith("hi")
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_the_floor_does_not_rescue_a_paste_that_swamps_the_draft() -> None:
    """The other side of MAJOR 1's fix: the floor must not become an escape
    hatch for pastes the feature exists to catch. A paste at the floor still
    collapses when the prospective total overruns, which is what keeps
    ``test_the_threshold_measures_the_PROSPECTIVE_buffer`` meaningful.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert(_lines(COLLAPSE_ROWS, "draft line {}") + "\n")
        await pilot.pause()

        await _paste(app, pilot, _lines(MIN_PASTE_ROWS, "pasted {}"))

        assert f"[Paste #1, {MIN_PASTE_ROWS} lines]" in editor.text


@pytest.mark.asyncio
async def test_a_single_long_line_is_labelled_in_characters() -> None:
    """A line count of 1 says nothing; the width is what the user can check.
    It still collapses, because it still wraps past the field."""
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        payload = "x" * 4000
        await _paste(app, pilot, payload)

        assert editor.text == f"[Paste #1, {len(payload)} chars] "


@pytest.mark.asyncio
async def test_the_threshold_follows_the_TERMINAL_width() -> None:
    """Rows, not characters. The same paste that fits a wide terminal overruns a
    narrow one, which is precisely why a stored character threshold cannot be
    right at both ends.
    """
    payload = _lines(12, "a fairly long line of log output number {}")
    app = Host()
    async with app.run_test(size=(200, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text == payload, "12 unwrapped rows fit the budget at 200 columns"

    app = Host()
    async with app.run_test(size=(30, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert "[Paste #1" in editor.text, "the same lines wrap past the budget at 30"


# -- what must never collapse -------------------------------------------------
@pytest.mark.asyncio
async def test_a_path_shaped_paste_never_collapses(tmp_path) -> None:
    """Design D5, the dangerous shape. ``_attach_pasted_images`` returns ``None``
    for "not an image" AND for "refused", so a 40-file drag where one file is a
    PDF lands in the text branch under the all-or-nothing rule. Collapsing it
    would hide the very paths the user needs in order to see what went wrong.
    """
    paths = "\n".join(str(tmp_path / f"file{index}.pdf") for index in range(40))
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, paths)

        assert editor.text == paths
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_the_path_guard_survives_the_size_that_makes_it_matter(tmp_path) -> None:
    """Caught by the test above, and worth its own assertion because the bug was
    SIZE-DEPENDENT and therefore invisible to a small fixture.

    ``_pasted_paths`` refuses anything over 4096 characters — correctly, since
    it must not shlex-parse a pasted essay on the keystroke that pasted it. But
    a paste only reaches the collapse guard when it is large, and a 40-file drag
    of real macOS paths is 5-6 KB, so gating on that helper alone collapsed
    exactly the refused drag D5 exists to protect. The guard has to hold at the
    sizes this feature sees.
    """
    long_paths = "\n".join(f"/Users/someone/Pictures/{'deep/' * 16}shot{i}.png" for i in range(40))
    assert len(long_paths) > 4096, "the fixture must exceed the shlex bound to be the regression"

    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, long_paths)

        assert editor.text == long_paths
        assert editor.attachments() == {}


@pytest.mark.parametrize(
    "drag",
    [
        pytest.param("/Users/me/Pictures/shot{}.png", id="absolute"),
        pytest.param("~/Pictures/shot{}.png", id="tilde"),
        pytest.param("./relative/shot{}.png", id="relative"),
        pytest.param("/Users/me/Screenshot\\ {}.png", id="escaped-space"),
        pytest.param("'/Users/me/Screen shot {}.png'", id="quoted-space"),
    ],
)
@pytest.mark.asyncio
async def test_a_file_drag_never_collapses_in_any_quoting(drag: str) -> None:
    """D5 across the shapes a terminal actually delivers. macOS screenshots are
    named ``Screenshot 2026-08-11 at 4.48.41 PM.png``, so escaped and quoted
    spaces are the common case, not an edge one — and ``shlex`` is the grammar
    terminals quote for.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        paste = "\n".join(drag.format(index) for index in range(40))
        await _paste(app, pilot, paste)

        assert editor.text == paste
        assert editor.attachments() == {}


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("/usr/local/lib/pkg/mod{}.py:{}: error: bad thing", id="mypy"),
        pytest.param('  File "/app/src/mod{}.py", line {}, in fn', id="stack-trace"),
        pytest.param("/var/log/build.log:{}: warning: unused variable {}", id="compiler"),
    ],
)
@pytest.mark.asyncio
async def test_path_PREFIXED_tool_output_still_collapses(line: str) -> None:
    """Review round 1, MAJOR 2. The guard was widened into a broad one.

    Requiring only that each line OPEN with a separator is true of most build
    and type-check output, so a 500-line mypy log and a 1.17 MB ``find /``
    listing both refused to collapse — and "paste 500 log lines" is the
    canonical case this feature exists for. A line counts as a path only if the
    WHOLE segment parses as one; ``error:`` does not.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "\n".join(line.format(i, i) for i in range(500)))

        assert "[Paste #1, 500 lines]" in editor.text


@pytest.mark.asyncio
async def test_a_SHORT_path_prefixed_log_collapses_on_the_whole_path_rule() -> None:
    """The whole-path rule on its own, with the size ceiling deliberately out of
    reach.

    The 500-line cases above are also caught by ``_MAX_DRAG_LINES``, so they do
    not isolate the grammar — a mutation that reverts to "opens with a
    separator" leaves them green. Ninety lines is well under the ceiling, so
    only "``error:`` is not a path" can keep this collapsing. Found by mutation
    testing the fix for MAJOR 2.
    """
    log = "\n".join(f"/usr/local/lib/pkg/mod{i}.py:{i}: error: bad thing" for i in range(90))
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, log)

        assert "[Paste #1, 90 lines]" in editor.text


@pytest.mark.parametrize("count", [40, 255, 256, 257, 300, 20000])
@pytest.mark.asyncio
async def test_a_refused_drag_stays_readable_at_EVERY_size(count: int) -> None:
    """Review round 2, MAJOR. The size bound must not decide the verdict.

    An earlier revision returned "not a drag" past ``_MAX_DRAG_LINES``, which
    flipped the guard from protective to permissive at the boundary: 256 files
    stayed readable and 257 collapsed. One ``Cmd+A`` in a folder of 300 refused
    files then hid the very paths the user needed in order to see which one was
    rejected — verbatim the D5 failure the guard exists to prevent.

    The boundary is parametrised either side because that is where the defect
    lived; 20,000 is here because the design's position is that a refused drag
    stays readable *whatever* its size.
    """
    drag = "\n".join(f"/Users/me/Pictures/shot{index}.png" for index in range(count))
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, drag)

        assert editor.text == drag
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_a_huge_listing_of_bare_paths_does_NOT_collapse() -> None:
    """The accepted cost of D5 holding at every size, asserted so it is a
    DECISION rather than an accident, and so that flipping it back requires
    arguing with this docstring.

    A ``find`` dump, ``ls -1`` or ``git ls-files`` is byte-identical line by
    line to a drag of the same files, and nothing outside the text separates
    them either: ``_attach_pasted_images`` returns ``None`` for "not images" and
    "refused" alike, so both arrive by the identical route. An earlier revision
    used SIZE as the discriminator, which is what re-opened D5 at 257 files
    (round 2, MAJOR) — a ``Cmd+A`` in a folder of 300 refused files hid the
    paths the user needed to see which one was rejected.

    D5 breaks the tie because the failure modes are asymmetric. Collapsing a
    refused drag introduces a NEW harm: the diagnostic information vanishes at
    the moment it is needed. Declining to collapse a listing merely leaves the
    status quo ante — the composer as it is today, nothing hidden, ``ctrl+o``
    not even required because nothing collapsed. A new harm outweighs a benefit
    not extended, whichever paste turns out to be more common.

    Path-PREFIXED tool output is unaffected — see the tests above — because
    ``error:`` is not a path, so a diagnostic line is not a path list.
    """
    listing = "\n".join(f"/usr/share/doc/pkg{index}/README" for index in range(20000))
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, listing)

        assert editor.text == listing
        assert editor.attachments() == {}


def test_the_drag_bound_caps_cost_without_deciding_the_verdict() -> None:
    """The cost half of ``_MAX_DRAG_LINES``, which the MAJOR fix had to keep.

    The bound exists so the shlex parse cannot run on the keystroke that pasted
    a multi-megabyte payload. Asserted as a time bound on the predicate itself
    rather than through the widget, because a paste that stays raw is then
    rendered by Textual and that cost belongs to the buffer, not here.
    """
    import time

    payload = "\n".join("/a/b/c/some/path/file.png" for _ in range(200_000))
    start = time.perf_counter()
    assert _looks_path_shaped(payload) is True
    # Two orders of magnitude of headroom over the measured ~7 ms, so this
    # pins "bounded" without becoming a flaky benchmark on a loaded CI box.
    assert time.perf_counter() - start < 1.0


# -- the payload reaches the model verbatim -----------------------------------
@pytest.mark.asyncio
async def test_the_payload_is_delivered_verbatim_at_submit() -> None:
    """The whole contract: the composer shows a chip, the model gets the text."""
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    payload = _lines(240)
    app = Capturing()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("why does this fail? ")
        await _paste(app, pilot, payload)
        await pilot.press("enter")
        await pilot.pause()

        (message,) = submitted
        # The editor posts the CHIP text; expansion is the app's job (slice B).
        assert message.text == "why does this fail? [Paste #1, 240 lines] "
        assert expand_pastes(message.text, message.attachments) == (
            f"why does this fail? {payload} "
        )
        # A collapsed paste is not an image and must not be sent as one.
        assert message.images == []


@pytest.mark.asyncio
async def test_a_CRLF_payload_is_delivered_as_LF() -> None:
    """A REGRESSION GUARD, not a nicety.

    Textual already normalises CRLF for text that goes through the buffer
    (verified against 8.2.8: pasting ``a\\r\\nb`` yields the lines ``a``, ``b``).
    A collapsed payload is held in the map and spliced in at submit, BYPASSING
    the document — so without normalising at capture, raw ``\\r\\n`` would reach
    the model for the first time BECAUSE the paste was large. Collapsed and
    uncollapsed pastes must deliver identical bytes.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(240).replace("\n", "\r\n"))

        (pasted,) = editor.attachments().values()
        assert isinstance(pasted, PastedText)
        assert "\r" not in pasted.text
        assert pasted.text == _lines(240)
        assert editor.text == "[Paste #1, 240 lines] "


@pytest.mark.asyncio
async def test_regex_metacharacters_in_the_payload_survive_expansion() -> None:
    """``re.sub`` with a string template interprets ``\\1`` and ``\\g<…>`` found
    in the PASTED CONTENT — a corruption with no error and no frame to notice it
    on (omp hit this at editor.ts:2070-2077). Splicing by span cannot express
    the bug; this pins that it stays spliced.
    """
    payload = _lines(240, r"s/(a)/\1\g<0>$0\\/ and \g<index> line {}")
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)

        assert expand_pastes(editor.text, editor.attachments()) == payload + " "


# -- the marker is an object, not text ----------------------------------------
@pytest.mark.asyncio
async def test_backspace_takes_the_whole_marker_and_its_payload() -> None:
    """Atomic, exactly like an image marker: a half-eaten ``[Paste #1, 240 lin``
    is neither text the user meant nor a reference anything can resolve."""
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(240))
        assert editor.text == "[Paste #1, 240 lines] "

        # Caret onto the marker's closing bracket, then one backspace.
        editor.move_cursor((0, len("[Paste #1, 240 lines]")))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == " "
        assert editor.attachments() == {}, "the payload goes with its marker"


@pytest.mark.asyncio
async def test_the_number_space_is_shared_with_images(tmp_path) -> None:
    """One map, one number space. A paste after an image is #2, never a second
    #1 — a draft can never hold the same number twice.
    """
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGB", (10, 20), (30, 30, 40)).save(path)

    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(path))
        await _paste(app, pilot, _lines(240))

        assert editor.text == "[Image #1, 10x20] [Paste #2, 240 lines] "
        attachments = editor.attachments()
        assert isinstance(attachments[1], Attachment)
        assert isinstance(attachments[2], PastedText)


def test_adjacent_mixed_markers_cannot_be_matched_ACROSS() -> None:
    """Round 20 / D12, re-armed by a second payload type.

    The tail excludes ``[`` as well as ``]``, so a bracket inside one can only
    mean the tail ran past its own marker into the next. Without that, two
    adjacent markers merge into a single match whose start is not where the live
    marker begins — the chip vanishes while the attachment stays attached and
    sent, and the live marker drops out of the atomic set.
    """
    text = "[Image #1, 10x20][Paste #2, 12 lines]"
    matches = list(ATTACHMENT_MARKER.finditer(text))
    assert [match.group(0) for match in matches] == [
        "[Image #1, 10x20]",
        "[Paste #2, 12 lines]",
    ]
    assert [match.group("kind") for match in matches] == ["Image", "Paste"]
    assert [match.group("index") for match in matches] == ["1", "2"]


def test_the_index_group_is_NAMED_not_positional() -> None:
    """``match.group(1)`` meant "the number" at every site in editor.py and one
    in app.py. The ``Image|Paste`` alternation makes group 1 the KIND — a
    rewrite with no type error and no failing test until an index comes back as
    ``'Image'``. This pins the trap rather than the fix.
    """
    match = ATTACHMENT_MARKER.search("[Paste #7, 240 lines]")
    assert match is not None
    assert match.group(1) == "Paste", "group 1 is the kind — do not read it as the number"
    assert int(match.group("index")) == 7


@pytest.mark.asyncio
async def test_release_is_scoped_to_the_marker_the_edit_TOUCHED() -> None:
    """Rounds 24-26, the mechanism where four earlier rules were each wrong.

    Damage a ``[Paste #N`` prefix well away from another marker and the OTHER
    attachment must survive: "does the buffer still parse" released a marker the
    user was mid-way through repairing, and "is the number mentioned anywhere"
    let an unrelated fragment keep a properly deleted payload alive.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(240, "first {}"))
        editor.insert(" " * 30)
        await _paste(app, pilot, _lines(120, "second {}"))
        await pilot.pause()
        assert set(editor.attachments()) == {1, 2}

        # Damage the FIRST marker's `[Paste #` prefix: delete its opening
        # bracket, thirty columns from the second marker.
        editor.delete((0, 0), (0, 1), maintain_selection_offset=False)
        await pilot.pause()

        assert 2 in editor.attachments(), "an edit elsewhere must not adjudicate #2"
        second = editor.attachments()[2]
        assert isinstance(second, PastedText)
        assert second.text == _lines(120, "second {}")


# -- the chip -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_paste_marker_is_chipped_like_an_image_marker() -> None:
    """One style for both payload shapes: the rationale in the tcss is about
    attachment-ness, not image-ness, so a second colour would be a second way of
    saying one thing. The class is renamed to match what it marks.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(240))
        await pilot.pause()

        marker = "[Paste #1, 240 lines]"
        assert editor.text == marker + " "
        assert _grounds(editor, 0, 0, len(marker)) == {theme_mod.semantic_color("tint-attach")}


@pytest.mark.asyncio
async def test_a_hand_typed_lookalike_is_NOT_chipped() -> None:
    """What is chipped is what is sent. A marker the user typed resolves to
    nothing, so it must paint as the prose it is (design round 16, D1).
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("[Paste #1, 240 lines] typed by hand")
        await pilot.pause()

        assert _grounds(editor, 0, 0, 21) != {theme_mod.semantic_color("tint-attach")}


def test_a_hand_typed_lookalike_is_NOT_expanded() -> None:
    """Prose that merely looks like a citation. The user's sentence about a chip
    is theirs, and rewriting it on the way to the model would be the silent
    class of failure this feature exists to avoid.
    """
    prose = "the [Paste #1, 240 lines] chip confused me"
    attachments = {1: PastedText("the real payload", "[Paste #1, 240 lines]")}
    # With no map entry it cannot expand at all...
    assert expand_pastes(prose, {}) == prose
    # ...and even WITH one, only the app's own citation is spliced — which here
    # is the first (and only) occurrence, so the sentence is the citation. The
    # property under test is that expansion goes through `cite`, so a second
    # copy below is prose.
    doubled = f"{prose} and again [Paste #1, 240 lines]"
    expanded = expand_pastes(doubled, attachments)
    assert expanded.count("the real payload") == 1
    assert expanded.endswith("and again [Paste #1, 240 lines]")


# -- recall -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_recalled_prompt_comes_back_WITHOUT_its_paste_citation() -> None:
    """The accepted trade, asserted as behaviour rather than left implicit.

    ``_navigate_history`` clears the map and numbering restarts at #1 each
    submit, so the payload cannot follow the text. Left in place, the recalled
    ``[Paste #1, 240 lines]`` is a chip-shaped string the user submits believing
    it carries 240 lines, and the model receives twenty-five characters —
    silent, and round 17 judged this class the worst failure this feature has.

    So the citation is stripped where the map is still live (the record seam),
    and the user SEES the prompt without its paste BEFORE pressing Enter.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("why does this fail? ")
        await _paste(app, pilot, _lines(240))
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()

        # Visible in the buffer, before Enter — not a rewrite on the way out.
        assert editor.text == "why does this fail?"
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_a_recalled_IMAGE_marker_is_left_alone(tmp_path) -> None:
    """The asymmetry is deliberate. An orphaned image marker fails LOUDLY — no
    chip, no picture, and the prompt never implied one — so round 17's settled
    behaviour stands. Changing it here would be a second rule drifting from a
    decided one.
    """
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGB", (10, 20), (30, 30, 40)).save(path)

    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(path))
        editor.insert("what is this")
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()

        assert editor.text == "[Image #1, 10x20] what is this"


@pytest.mark.asyncio
async def test_recall_does_not_edit_the_users_own_prose() -> None:
    """A history entry may legitimately contain a hand-typed lookalike from the
    original draft. Stripping goes through the same first-citation predicate the
    chip uses, so text the app did not write is never touched.
    """
    prose = "the [Paste #1, 240 lines] chip confused me"
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert(prose)
        await pilot.press("enter")
        await pilot.pause()

        assert editor.prompt_history() == [prose]
        await pilot.press("up")
        await pilot.pause()
        assert editor.text == prose


def test_stripping_removes_only_the_apps_own_citation() -> None:
    """The unit form of the rule above, over both payload shapes at once."""
    pasted = PastedText("payload", "[Paste #2, 9 lines]")
    image = Attachment(image=None, marker="[Image #1, 10x20]")  # type: ignore[arg-type]
    text = "look [Image #1, 10x20] and [Paste #2, 9 lines] plus typed [Paste #5, 3 lines]"

    stripped = strip_paste_citations(text, {1: image, 2: pasted})

    assert stripped == "look [Image #1, 10x20] and plus typed [Paste #5, 3 lines]"


# -- the expand action --------------------------------------------------------
@pytest.mark.asyncio
async def test_ctrl_o_expands_the_paste_at_the_caret() -> None:
    """The no-setting escape hatch (design §1), so it has to work.

    "I wanted THIS paste raw" is a per-gesture want, not a per-user one, which
    is why this is a key and not a toggle.
    """
    payload = _lines(40)
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text == "[Paste #1, 40 lines] "

        editor.move_cursor((0, 3))  # caret standing inside the chip
        await pilot.press("ctrl+o")
        await pilot.pause()

        assert editor.text == payload + " "
        assert editor.attachments() == {}, "the entry goes, or submit would splice it twice"
        # The caret lands at the END of the restored text, so the next keystroke
        # does not type before it.
        assert editor.selection.end == (39, len("ERROR line 39"))


@pytest.mark.asyncio
async def test_ctrl_o_on_ordinary_prose_does_nothing() -> None:
    """A key that silently rewrites a buffer it was not aimed at is worse than
    one that no-ops; the action raises ``SkipAction`` so the press keeps walking
    the chain."""
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("just a sentence")
        editor.move_cursor((0, 4))
        await pilot.press("ctrl+o")
        await pilot.pause()

        assert editor.text == "just a sentence"


@pytest.mark.asyncio
async def test_ctrl_o_leaves_a_live_selection_alone() -> None:
    """A real selection is the user's own range; the expand action must not
    widen or replace it, exactly as ``_delete_marker`` refuses to."""
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(40))
        editor.selection = Selection((0, 0), (0, 5))
        await pilot.press("ctrl+o")
        await pilot.pause()

        assert editor.text == "[Paste #1, 40 lines] "
        assert set(editor.attachments()) == {1}


# -- Slice B: the payload reaches the MODEL -----------------------------------
#
# Everything above this line asserts the COMPOSER: what collapses, what paints,
# what the map holds. These drive the whole app, because the seam Slice B closes
# is app-side — the editor posts the CHIP text and `on_editor_submitted` is what
# splices the payload back in, so a widget-level assertion cannot see the defect
# at all. `session.prompts` is the proof: it is what the model received.
async def _booted(app: OperatorApp, pilot) -> None:
    """Wait for the session to EXIST before submitting anything.

    `_submit_prompt` refuses outright while `_session is None`, appending
    "session is still starting…" and returning, so a prompt sent one tick early
    is never delivered and no wait afterwards recovers it. Bounded by pumps
    rather than by the clock for the reason ``test_conversation_naming._ready``
    records: adoption takes an unknown NUMBER of pumps, not an interval.
    """
    for _ in range(200):
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError("the session was never adopted")


async def _sent(pilot, session: FakeSession) -> str:
    """The one prompt the model received, once the turn worker has run."""
    for _ in range(200):
        await pilot.pause()
        if session.prompts:
            return session.prompts[0]
    raise AssertionError("no prompt reached the session")


async def _submit(app: OperatorApp, pilot, text: str) -> None:
    """Type AROUND the chip already in the buffer, then send.

    `load_text` would be the shorter route and is the wrong one: it replaces the
    buffer wholesale, and the point of these tests is the draft the paste itself
    produced. The caret goes to the end for the reason
    ``test_skill_invocation._submit`` records — every picker parse is
    caret-anchored, so a caret left at offset 0 completes the row instead of
    submitting it.
    """
    editor = app.query_one(Editor)
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_the_model_receives_the_PAYLOAD_not_the_chip() -> None:
    """THE defect Slice B closes, and the reason the chip is safe to show.

    The editor posts `[Paste #1, 40 lines]` and holds the log in its map, so
    before the splice existed the model was sent that bare string — the feature
    would have silently replaced the user's evidence with a label describing it.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text == "[Paste #1, 40 lines] ", "the fixture stopped collapsing"
        await _submit(app, pilot, editor.text + "why does this build fail?")

        sent = await _sent(pilot, session)
        assert payload in sent, "the model got the chip, not the log"
        assert "[Paste #1" not in sent, "the marker survived into the prompt"
        assert sent.endswith("why does this build fail?")


@pytest.mark.asyncio
async def test_the_transcript_shows_the_FULL_text() -> None:
    """Design §2.5: the composer collapses, the transcript does not.

    Asserted because it is what makes `text == sent` again, and that equality is
    what lets the steer queue, the echo registry and the compaction hold need no
    new plumbing. A chip in the row would mean a persistence answer this feature
    deliberately does not have.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, editor.text + "why?")
        await _sent(pilot, session)

        painted = _transcript_text(app)
        assert "ERROR line 39" in painted, "the transcript row collapsed the paste"
        assert "[Paste #1" not in painted


# -- Slice B: `$skill` ordering (design §2.6) ---------------------------------
@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_a_draft_STARTING_with_a_chip_does_not_invoke_a_skill() -> None:
    """The user's first token was a PASTE, not a skill name.

    The ordering rule stated as a defence: parse the TYPED text, expand after.
    Reversed, a pasted document whose first line happens to read `$research …`
    would fire a skill nobody asked for — and the payload here is exactly that
    document, so this fails loudly against a parse-after-expand implementation.
    """
    payload = "$research the vendor's own writeup\n" + _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text.startswith("[Paste #1"), "the fixture stopped collapsing"
        await _submit(app, pilot, editor.text + "is this right?")

        sent = await _sent(pilot, session)
        assert payload in sent, "the payload did not reach the model"
        assert "Read primary sources." not in sent, "a pasted document fired a skill"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_a_skill_invocation_expands_the_REQUEST_but_not_the_BODY() -> None:
    """`$research <chip>` sends the body AND the payload, and splices only once.

    The body-scoping half is the mirror of the image defect
    ``on_compaction_ended`` records: a `[Paste #N]` inside a SKILL.md would be
    spliced too if expansion ran over the rendered payload instead of over
    `invocation.request`. Counted, not merely tested for presence — a second
    splice is the shape that bug takes.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, "$research " + editor.text + "why does this fail?")

        sent = await _sent(pilot, session)
        assert "Read primary sources." in sent, "the skill body was not sent"
        assert payload in sent, "the request's paste was not expanded"
        assert sent.count("ERROR line 39") == 1, "the payload was spliced twice"
        # The typed line rides the opening tag so replay repaints the row the
        # live session showed. It must carry the CHIP, not 40 lines of log.
        assert 'invocation="$research [Paste #1, 40 lines] why does this fail?"' in sent


# -- Slice B: the slash-command tail ------------------------------------------
@pytest.mark.asyncio
async def test_team_request_carries_the_pasted_payload() -> None:
    """`/team <name> <request>` sends the payload the request cites.

    The same bug shape as the already-fixed image one: `_submit_command_prompt`
    submits only the request TAIL, so without its own `expand_pastes` call the
    manager is told `[Paste #1, 40 lines]` and nothing else.
    """
    payload = _lines(40)
    session = FakeSession()
    registry = TeamRegistry(scratch_dir())
    registry.create_team(
        TeamEditFields(name="ops", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, "/team ops fix this: " + editor.text)

        sent = await _sent(pilot, session)
        assert [team.name for team in session.attached_teams] == ["ops"]
        assert payload in sent, "the manager was sent the chip, not the log"
        assert "[Paste #1" not in sent


# -- Slice B: naming (design §2.5) --------------------------------------------
@pytest.mark.asyncio
async def test_the_conversation_is_named_from_the_TYPED_line() -> None:
    """A thread titled after a pasted stack trace is the knock-on §2.5 names.

    `_submit_prompt` now receives the EXPANDED text, so without the `typed=`
    split the naming errand would see 40 lines of log where the user wrote one
    question. The provider call is what proves it: the prompt handed to the
    titler is the chip line.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, editor.text + "why does this build fail?")
        await _sent(pilot, session)
        for _ in range(200):
            await pilot.pause()
            if session.completions:
                break

        assert session.completions, "the naming errand never ran"
        _system, named = session.completions[0]
        assert "why does this build fail?" in named
        assert "ERROR line 39" not in named, "the thread was titled after the log"


# -- Slice B: the round-trip surfaces -----------------------------------------
#
# These key on the marker NUMBER and carry `.marker`, so they SHOULD work
# unchanged — which is exactly why they are exercised rather than reasoned
# about: review round 19 found the compaction hold reading the widget instead of
# the message, and every one of these is the same shape of round trip.
@pytest.mark.asyncio
async def test_the_aside_stash_returns_the_chip_AND_its_payload() -> None:
    """The aside borrows the composer and hands the draft back.

    `adopt_attachments` re-keys on the markers now in the buffer, so the payload
    has to survive a stash that only ever carried the chip text. Proved by
    SENDING afterwards: a chip whose payload was lost still looks right.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        editor.insert("why does this fail?")
        draft = editor.text

        app._open_aside()
        await pilot.pause()
        app._close_aside()
        await pilot.pause()

        assert editor.text == draft, "the draft did not come back"
        await _submit(app, pilot, editor.text)
        sent = await _sent(pilot, session)
        assert payload in sent, "the stash lost the payload behind the chip"


@pytest.mark.asyncio
async def test_a_paste_held_through_a_compaction_is_sent_in_full() -> None:
    """The hold sends the payload; the hand-back returns the line.

    Both halves, because they read different fields for different reasons — and
    `_typed_held_for_compaction` is the one Slice B repointed at the chip so
    ``on_compaction_ended``'s marker walk keeps seeing what the user wrote.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        app._compacting = True
        await _submit(app, pilot, editor.text + "why does this fail?")
        await pilot.pause()

        assert payload in app._prompt_held_for_compaction, "the hold kept the chip"
        assert app._typed_held_for_compaction == "[Paste #1, 40 lines] why does this fail?"

        # And the pass ending sends what was held, in full.
        app._compacting = False
        app.on_compaction_ended(CompactionEnded(reason="manual", success=True))
        sent = await _sent(pilot, session)
        assert payload in sent


@pytest.mark.asyncio
async def test_a_queued_STEER_carries_the_payload_and_recalls_LOSSLESSLY() -> None:
    """Mid-turn the submit queues instead of prompting, and Esc takes it back.

    Two halves. The queued Message must hold the PAYLOAD, because it is what
    will be sent at the next boundary. The recall then hands back what the ROW
    holds — which for a paste is the expanded text, not the chip, and that is
    the settled §2.5 decision rather than an oversight: the transcript shows the
    full text, so ``text == sent``, and it is exactly that equality which lets
    the steer queue, the echo registry and the compaction hold carry a collapsed
    paste with no new plumbing. Contrast ``$skill``, whose row legitimately
    differs from its payload and which therefore recalls the typed line
    (``TestHandBackPathsReturnTheTypedLine`` in ``test_skill_invocation``).

    So the property that matters here is LOSSLESSNESS, and it is asserted by
    resending: the composer holds the log as ordinary text, the attachment map
    is legitimately empty (there is no marker left to key on), and the model
    still receives every byte. The wart is legibility — the user gets 40 raw
    lines back where they had a chip — and it is the same shape as the design's
    open question 1 about ``Up``-arrow recall, recorded there as accepted.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        session.streaming = True
        await _submit(app, pilot, editor.text + "why does this fail?")
        for _ in range(200):
            await pilot.pause()
            if app._held_steer_blocks:
                break

        assert app._held_steer_blocks, "the submit did not queue"
        queued = app._held_steer_blocks[0][0]
        assert payload in queued.text, "the queue holds the chip, not the log"

        app._recall_queued_steers()
        await pilot.pause()
        assert payload in editor.text, "the recall lost the payload"
        assert editor.text.endswith("why does this fail?")
        assert editor.attachments() == {}, "no marker survives, so nothing keys on one"

        # The whole point of a recall is the resend, so prove THAT still
        # delivers: the text is its own payload now, and a second expansion
        # would have nothing to splice.
        session.streaming = False
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        sent = await _sent(pilot, session)
        assert payload in sent
        assert sent.endswith("why does this fail?")


# -- Slice B, round 2: every prompt-bearing exit from the composer -------------
#
# The first round spliced where it looked — the ordinary submit and the two
# commands that call `_submit_prompt` — and missed the aside and the four other
# `consumes_prompt` commands. The structural test below is the one that matters:
# it pins the SET against the registry, so a seventh such command cannot be
# added later without either inheriting the splice or failing here.
def test_every_consumes_prompt_command_is_accounted_for() -> None:
    """The registry's ``consumes_prompt`` set, pinned against this file's cover.

    ``consumes_prompt`` means "this argument is free text destined for a model"
    (``autocomplete.py``), which is exactly the condition under which a chip has
    to become its payload. So the flag IS the checklist, and a per-command list
    of names is what let round 1 ship with four of six unspliced.

    A new command that sets the flag lands in ``SLASH_COMMANDS`` but not in
    ``_SPLICED`` and fails here, naming itself — which is the prompt to route it
    through an expansion and add it below, not to edit this constant blindly.
    """
    from local_operator.tui.app import SLASH_COMMANDS

    flagged = {command.name for command in SLASH_COMMANDS if command.consumes_prompt}
    assert flagged == _SPLICED, (
        "a command's consumes_prompt flag changed: its argument reaches a model, "
        "so a collapsed paste in it must be expanded before the handler reads it"
    )


#: The commands whose argument is expanded, and WHERE. Two shapes, because the
#: two groups cannot share one seam:
#:
#: * ``goal``/``loop``/``fork``/``btw`` expand at DISPATCH
#:   (``_run_slash_command``'s ``prompt_arg``), because their handlers consume
#:   the argument directly — `/goal` stores it as the standing objective.
#: * ``team``/``agent`` expand DOWNSTREAM in ``_submit_command_prompt``, after
#:   ``resolve_markers`` has walked the request: that walk orders images by
#:   where the citation sits, so a payload spliced in ahead of one would send
#:   the pictures in the wrong order.
_SPLICED = {"fork", "goal", "loop", "btw", "team", "agent"}


@pytest.mark.asyncio
async def test_the_ASIDE_sends_the_payload_not_the_chip() -> None:
    """BLOCKER-1, and the worst of the family because it is UNRECOVERABLE.

    The aside branch returns before the submit path's splice, so a log pasted
    into an open `/btw` card reached `complete_aside` as the 21-character chip.
    Unlike a dropped image there is nothing to recover it from: the composer
    clears on submit and `_record_history` strips the citation
    (``editor.py:6243``), so there is no chip in history, no map entry and no
    Up-arrow.

    Read off `complete_aside`'s turns rather than the card, because the question
    the MODEL was given is the whole claim.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        app._open_aside()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text == "[Paste #1, 40 lines] ", "the fixture stopped collapsing"
        await _submit(app, pilot, editor.text + "what is this?")
        for _ in range(200):
            await pilot.pause()
            if session.asides:
                break

        assert session.asides, "the aside never reached the session"
        asked = "\n".join(
            block.text
            for turn in session.asides[0]
            for block in (turn.content or [])
            if getattr(block, "text", None)
        )
        assert payload in asked, "the aside was asked about the chip, not the log"
        assert "[Paste #1" not in asked


@pytest.mark.asyncio
async def test_goal_stores_the_payload_not_the_chip() -> None:
    """MAJOR-1. The goal rides the system prompt's volatile tail on EVERY later
    turn and every compaction, so a chip stored here is a dead reference the
    session carries for the rest of its life."""
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, f"/goal {editor.text.strip()} do the thing")
        await pilot.pause()

        assert payload in session.goal, "the standing objective is a dead chip"
        assert "[Paste #1" not in session.goal


@pytest.mark.asyncio
async def test_loop_pursues_the_payload_not_the_chip() -> None:
    """`/loop <goal text>` runs goal mode against its argument directly \u2014 a
    second consumer of the same dispatch splice, and one that never touches
    `set_goal`, so it could not inherit `/goal`'s fix by accident."""
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        # Captured at the launcher rather than run: goal mode is an unbounded,
        # judge-gated loop, and the claim here is only what it was handed.
        launched: list[str] = []
        app._start_goal_loop = (  # type: ignore[method-assign]
            lambda goal, notice: launched.append(goal)
        )
        await _submit(app, pilot, f"/loop {editor.text.strip()} until it passes")
        await pilot.pause()

        assert launched, "goal mode never launched"
        assert payload in launched[0], "the loop is pursuing a dead chip"


@pytest.mark.asyncio
@pytest.mark.usefixtures("skill_root")
async def test_a_skill_with_a_paste_shows_the_SAME_row_live_and_on_replay() -> None:
    """MAJOR-2. The property that broke is the EQUALITY, so that is what is pinned.

    `render_invocation` records the typed line in the payload's ``invocation=``
    attribute and replay repaints from it, so expanding the live row made the
    two disagree for the paste case alone. The attribute is deliberately left
    holding the chip \u2014 it travels inside the payload the model receives, so
    expanding it would send the paste twice on every turn of that message's life
    \u2014 and the live row is put back on the typed line to match.

    Asserted as live == replay rather than against either value alone: a future
    change that moves both together is fine, one that moves only one is the bug.
    """
    payload = _lines(40)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=WIDE) as pilot:
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        await _submit(app, pilot, "$research " + editor.text + "why does this fail?")
        sent = await _sent(pilot, session)
        live = _transcript_text(app)

    # The model still receives the whole log; only the ROW is the typed line.
    assert payload in sent, "the request's paste stopped being expanded"
    assert "[Paste #1, 40 lines]" in live, "the live row lost the paste receipt"
    assert "ERROR line 39" not in live, "the live row shows the payload again"

    replay_session = FakeSession()
    replay_session._history = [SimpleNamespace(role="user", text=sent, content=[])]
    replay_app = OperatorApp(lambda: _factory(replay_session))
    async with replay_app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        replay_app._project_settled_rows(list(replay_session._history))
        await pilot.pause()
        replayed = _transcript_text(replay_app)

    assert "$research [Paste #1, 40 lines] why does this fail?" in replayed
    assert _user_row(live) == _user_row(replayed), "live and replay disagree"


def _user_row(painted: str) -> str:
    """The `$research …` line out of a rendered transcript.

    Compared line-wise rather than whole-transcript, because the two apps paint
    different surroundings (a live session has notices and a status band a
    replayed one does not) and the claim is only about the USER row.
    """
    return next(line.strip() for line in painted.split("\n") if "$research" in line)
